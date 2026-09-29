import math

import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.LevyHooks import build_gateway_hook
from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

# Additive stand-in for log(0). Finite (fp16-safe) yet exp()es to exactly 0.
LOG_ZERO = -1e4

# Floor for the L2 norm when normalising features for the cosine channel.
NORM_EPS = 1e-6

# Slack on "warped observation mask == 1" when deciding full coverage of a cell.
COVERAGE_EPS = 1e-3


def poisson_inverse_moment(lam, max_terms=96, asymptotic_threshold=30.0):
    """
    phi(Lambda) = E[ 1{Z >= 1} / Z ] for Z ~ Poisson(Lambda).

    Parameters
    ----------
    lam : torch.Tensor
        Non-negative intensities, any shape.

    Returns
    -------
    torch.Tensor
        phi(lam), same shape/dtype as ``lam``.
    """
    lam = lam.clamp(min=0.0)
    lam_series = lam.clamp(max=asymptotic_threshold)

    if lam_series.numel() == 0:
        n_terms = 1
    else:
        lam_max = float(lam_series.detach().max())
        n_terms = int(
            min(max_terms, math.ceil(lam_max + 6.0 * math.sqrt(lam_max) + 8.0))
        )

    pmf = torch.exp(-lam_series)  # p_0
    phi = torch.zeros_like(lam_series)
    for k in range(1, n_terms + 1):
        pmf = pmf * lam_series / k
        phi = phi + pmf / k

    if bool((lam > asymptotic_threshold).any()):
        inv = 1.0 / lam.clamp(min=asymptotic_threshold)
        phi_asym = inv * (1.0 + inv * (1.0 + inv * (2.0 + 6.0 * inv)))
        phi = torch.where(lam > asymptotic_threshold, phi_asym, phi)

    return phi


def _sample_multinomial_fuse(features, p, counts, eps=1e-12):
    """
    Draw the stochastic fused feature for one batch element.

    Given ``counts[s] = z`` atoms at cell ``s``, the atom-to-agent assignment is
    Multinomial(z, p(s)); we simulate it with the standard chain of conditional
    binomials over the (small) agent axis, so the whole BEV grid is sampled in
    ``N - 1`` vectorised draws.

    Parameters
    ----------
    features : torch.Tensor
        Aligned agent features, (N, C, H, W).
    p : torch.Tensor
        Multinomial probabilities over agents, (N, 1, H, W); sums to 1 (or to 0 at
        cells no agent observes).
    counts : torch.Tensor
        Poisson draws Z_s, (1, H, W).

    Returns
    -------
    torch.Tensor
        (C, H, W); cells with z = 0 are left at 0 and must be back-filled with the
        mean path by the caller.
    """
    n_agents = features.shape[0]
    remaining_count = counts
    remaining_prob = torch.ones_like(counts)
    acc = torch.zeros_like(features[0])

    for j in range(n_agents - 1):
        prob_j = (p[j] / remaining_prob.clamp(min=eps)).clamp(0.0, 1.0)
        n_j = torch.binomial(remaining_count, prob_j)
        acc = acc + n_j * features[j]
        remaining_count = remaining_count - n_j
        remaining_prob = remaining_prob - p[j]
    acc = acc + remaining_count * features[n_agents - 1]

    return acc / counts.clamp(min=1.0)


def levy_weighted_fuse(
    x,
    score,
    record_len,
    affine_matrix,
    align_corners,
    tau=0.5,
    gamma=1.0,
    disagreement_reduction="mean",
    disagreement_mode="raw",
    aux_disagreement_modes=(),
    stochastic=False,
    agent_logit_fn=None,
    decode_fn=None,
    agent_deviation=False,
):
    """
    Levy replacement for :func:`opencood.models.fuse_modules.pyramid_fuse.weighted_fuse`.

    Identical warping and agent bookkeeping; the agent-wise ``softmax`` is replaced by
    the normalised Poisson intensity, and the evidence / disagreement that the softmax
    would have thrown away are returned alongside the fused feature.

    Parameters
    ----------
    x : torch.Tensor
        Input features, (sum(n_cav), C, H, W).
    score : torch.Tensor
        HEAL foreground score, (sum(n_cav), 1, H, W). Strictly positive where the
        agent observes; exactly 0 marks "not observed" (warp padding or camera crop),
        which is how HEAL itself flags invalid cells.
    record_len : list
        cav num in each sample, shape (B).
    affine_matrix : torch.Tensor
        normalized affine matrix from 'normalize_pairwise_tfm', (B, L, L, 2, 3).
    tau : float
        Base Poisson rate. Scales the evidence; cancels out of the fusion weights.
    gamma : float or torch.Tensor
        Logit temperature. ``gamma == 1`` reproduces HEAL's weights exactly.
    disagreement_reduction : str
        'sum', 'mean' over the channel axis. 'sum' makes tr Sigma grow
        with the level's channel count, which the per-level omega_l weights are there
        to rebalance; 'mean' removes that dependence up front. Ignored under
        ``disagreement_mode='cosine'``, which is already channel-count free.
    disagreement_mode : str
        'raw' computes tr Sigma on the features as they are. Because tr Sigma is
        quadratically homogeneous in feature magnitude, that channel reads out
        feature *energy*: it peaks near the ego where returns are dense and real
        objects sit, and decays to ~0 in the far field regardless of how much the
        agents actually conflict.

        'cosine' L2-normalises every agent's feature vector first, so the statistic
        becomes 1 - ||sum_j p_j F_hat_j||^2 -- a scale-free angular dispersion in
        [0, 1] that measures conflict of *direction* rather than magnitude. The
        fused feature itself is always built from the un-normalised features; only
        the diagnostic changes.
    stochastic : bool
        Return one draw of the stochastic operator instead of its mean. Non
        differentiable; evaluation only.
    agent_logit_fn : callable, optional
        ``agent_logit_fn(b, feature_in_ego, s_map, valid, base_weights)`` returning a
        per-agent additive log-intensity offset (N, 1, H, W). Added to
        log(tau) + gamma * S before normalisation, so the weights stay convex. None
        (default) leaves the operator exactly as without it.
    decode_fn : callable, optional
        Linear part of this level's decoder, ``decode_fn(F) -> G`` mapping
        (N, C, H, W) to (N, C', sH, sW). When given, raw tr Sigma is also taken
        over the per-agent decoded features G_j with the nearest-upsampled weights,
        which is exact for a non-overlapping decoder (see
        :meth:`LevyPyramidFusion._make_level_decoder`). Always channel-summed.

    Returns
    -------
    out : torch.Tensor
        Fused features, (B, C, H, W).
    log_evidence : torch.Tensor
        log Lambda(s), (B, 1, H, W). Log space so that large gamma cannot overflow.
    disagreement : torch.Tensor
        tr Sigma(s) >= 0, (B, 1, H, W).
    n_valid : torch.Tensor
        N_valid(s), (B, 1, H, W) int64 -- how many agents *fully* observe each
        cell. ``n_valid >= 2`` is the overlap mask M_overlap: the only region
        where tr Sigma carries signal, since a single contributing agent gives
        p_j = 1 and hence tr Sigma = 0 identically.

        Stricter than the fusion's own ``score != 0``: bilinear warping blends
        zero padding into cells on a footprint (or camera crop) edge, which leaves
        them nonzero but attenuated and so inflates raw tr Sigma there. A cell only
        counts for an agent when all of its bilinear samples were observed, i.e.
        the warped observation mask is 1. Fusion weights are unaffected.
    agent_dev : list or None
        Per batch element, the per-agent term of the cosine tr Sigma *before* it is
        summed over agents: {'deviation', 'cosine', 'weight'}, each (N, 1, H, W).
        ``tr Sigma = sum_j p_j * deviation_j``, so this is the same quantity
        decomposed -- the cell map says the agents disagree, these say which one is
        out of line. None unless ``agent_deviation``.
    aux : dict
        {mode: tr Sigma (B, 1, H, W)} for each of ``aux_disagreement_modes``.
    decoded_disagreement : torch.Tensor or None
        Channel-summed raw tr Sigma of the decoded features, (B, 1, sH, sW); None
        unless ``decode_fn`` is given.
    """
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)

    out = []
    log_evidence_list = []
    disagreement_list = []
    n_valid_list = []
    aux_lists = {}
    decoded_list = []
    agent_dev_list = [] if agent_deviation else None

    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0  # ego
        feature_in_ego = warp_affine_simple(
            split_x[b], t_matrix[i, :, :, :], (H, W), align_corners=align_corners
        )
        scores_in_ego = warp_affine_simple(
            split_score[b], t_matrix[i, :, :, :], (H, W), align_corners=align_corners
        )
        # HEAL masks these to -inf before the softmax; here they simply carry zero
        # intensity, so they contribute neither weight nor evidence.
        valid = scores_in_ego != 0

        s_map = scores_in_ego

        # lambda_j = tau * exp(gamma * S_j), held in log space.
        log_lambda = math.log(tau) + gamma * s_map
        log_lambda = log_lambda.masked_fill(~valid, LOG_ZERO)

        # Lambda = sum_j lambda_j   (spatial evidence)
        log_evidence = torch.logsumexp(log_lambda, dim=0, keepdim=True)

        # p_j = lambda_j / Lambda. The explicit mask keeps cells that *no* agent
        # observes at p = 0 (HEAL's nan -> 0 branch) instead of a spurious 1/N.
        weights = torch.exp(log_lambda - log_evidence) * valid.to(log_lambda.dtype)

        if agent_logit_fn is not None:
            # Learned per-agent offset on top of the base intensity; invalid cells
            # stay at LOG_ZERO, so they still get neither weight nor evidence.
            delta = agent_logit_fn(b, feature_in_ego, s_map, valid, weights)
            log_lambda = (log_lambda + delta).masked_fill(~valid, LOG_ZERO)
            log_evidence = torch.logsumexp(log_lambda, dim=0, keepdim=True)
            weights = torch.exp(log_lambda - log_evidence) * valid.to(log_lambda.dtype)

        # Mean path: E[stochastic fused feature] (Theorem 1).
        fused = torch.sum(feature_in_ego * weights, dim=0)

        # tr Sigma = sum_j p_j ||F_j||^2 - ||F_mean||^2  (weighted variance).
        # Under 'cosine' the same weighted variance is taken over unit-norm
        # features, which cancels the ||F||^2 scaling. Cells no agent observes
        # keep weights = 0 and so still come out at exactly 0.
        def _disagreement(mode):
            if mode == "cosine":
                feat_d = feature_in_ego / feature_in_ego.norm(
                    dim=1, keepdim=True
                ).clamp(min=NORM_EPS)
                mean_d = torch.sum(feat_d * weights, dim=0)
            else:
                feat_d = feature_in_ego
                mean_d = fused
            sq_norm = torch.sum(feat_d**2, dim=1, keepdim=True)
            second_moment = torch.sum(weights * sq_norm, dim=0)
            d = second_moment - torch.sum(mean_d**2, dim=0, keepdim=True)
            d = d.clamp(min=0.0)
            if disagreement_reduction == "mean" and mode != "cosine":
                d = d / C
            return d

        disagreement = _disagreement(disagreement_mode)
        for _m in aux_disagreement_modes:
            aux_lists.setdefault(_m, []).append(_disagreement(_m))

        if agent_deviation:
            # The individual terms of the cosine tr Sigma: how far each agent's
            # feature direction sits from the consensus at that cell.
            unit = feature_in_ego / feature_in_ego.norm(dim=1, keepdim=True).clamp(
                min=NORM_EPS
            )
            consensus = torch.sum(unit * weights, dim=0, keepdim=True)
            dev = torch.sum((unit - consensus) ** 2, dim=1, keepdim=True)
            cos = torch.sum(unit * consensus, dim=1, keepdim=True) / consensus.norm(
                dim=1, keepdim=True
            ).clamp(min=NORM_EPS)
            agent_dev_list.append({
                "deviation": dev.detach(),
                "cosine": cos.detach(),
                "weight": weights.detach(),
                "valid": valid.detach(),
            })

        if decode_fn is not None:
            # Same weighted variance, taken after the decoder. The weights are
            # constant over each decoder block, so nearest upsampling is exact.
            decoded = decode_fn(feature_in_ego)
            weights_up = F.interpolate(weights, size=decoded.shape[-2:], mode="nearest")
            decoded_mean = torch.sum(decoded * weights_up, dim=0)
            decoded_second = torch.sum(
                weights_up * torch.sum(decoded**2, dim=1, keepdim=True), dim=0
            )
            decoded_list.append(
                (
                    decoded_second - torch.sum(decoded_mean**2, dim=0, keepdim=True)
                ).clamp(min=0.0)
            )

        if stochastic:
            with torch.no_grad():
                counts = torch.poisson(torch.exp(log_evidence.clamp(max=60.0))[0])
                sampled = _sample_multinomial_fuse(feature_in_ego, weights, counts)
                # Z = 0 falls back to the mean path, which keeps the operator
                # unbiased across all regimes.
                fused = torch.where(counts >= 1, sampled, fused)

        out.append(fused)
        log_evidence_list.append(log_evidence[0])
        disagreement_list.append(disagreement)

        # Warp the binary observation mask (camera crop included) the same way
        # as the score; anything below 1 mixed in padding or cropped cells.
        coverage = warp_affine_simple(
            (split_score[b] != 0).to(scores_in_ego.dtype),
            t_matrix[i, :, :, :],
            (H, W),
            align_corners=align_corners,
        )
        covered = valid & (coverage >= 1.0 - COVERAGE_EPS)
        n_valid_list.append(covered.sum(dim=0))

    return (
        torch.stack(out),
        torch.stack(log_evidence_list),
        torch.stack(disagreement_list),
        torch.stack(n_valid_list),
        {m: torch.stack(v) for m, v in aux_lists.items()},
        torch.stack(decoded_list) if decoded_list else None,
        agent_dev_list,
    )


def overlap_masked_stats(tensor, overlap_mask, quantiles=(0.5, 0.9, 0.99)):
    """
    Reduce a BEV map over the overlap mask only.

    Every grid-level metric on evidence / disagreement / sigma must go through
    this. Reducing over the full grid instead is not a mild dilution: outside
    M_overlap at most one agent contributes, so tr Sigma is exactly 0 by
    construction, and the unmasked mean is driven by how much of the BEV the
    collaborator covers rather than by any property of the uncertainty.

    Parameters
    ----------
    tensor : torch.Tensor
        Map to reduce, (B, 1, H, W) / (1, H, W) / (H, W).
    overlap_mask : torch.Tensor
        Binary M_overlap of the *same* spatial size -- pass the mask belonging to
        the level being reduced, since levels differ in resolution.
    quantiles : sequence of float
        Quantiles to report over the selected cells.

    Returns
    -------
    dict
        'n_cells' selected, 'coverage' as a fraction of the grid, 'mean', 'std',
        'max', and 'q50' / 'q90' / ... . Stats are float('nan') when the mask
        selects nothing (e.g. a single-agent frame).
    """
    if tensor.shape[-2:] != overlap_mask.shape[-2:]:
        raise ValueError(
            f"overlap mask {tuple(overlap_mask.shape[-2:])} does not match map "
            f"{tuple(tensor.shape[-2:])}; use the mask of the same pyramid level"
        )

    mask = (
        overlap_mask.expand_as(tensor)
        if overlap_mask.shape != tensor.shape
        else overlap_mask
    )
    selected = tensor[mask.bool()].float()

    stats = {
        "n_cells": int(selected.numel()),
        "coverage": float(selected.numel()) / float(tensor.numel()),
    }
    if selected.numel() == 0:
        stats.update({"mean": float("nan"), "std": float("nan"), "max": float("nan")})
        stats.update({f"q{int(q * 100)}": float("nan") for q in quantiles})
        return stats

    qs = torch.quantile(selected, torch.tensor(list(quantiles), device=selected.device))
    stats.update(
        {
            "mean": selected.mean().item(),
            "std": selected.std().item() if selected.numel() > 1 else 0.0,
            "max": selected.max().item(),
        }
    )
    stats.update({f"q{int(q * 100)}": qs[k].item() for k, q in enumerate(quantiles)})
    return stats


class LevyPyramidFusion(PyramidFusion):
    """
    HEAL's PyramidFusion with the agent-wise softmax replaced by Levy attention.

    Same parameters, same ``state_dict``, same ``forward_single`` / ``forward_collab``
    signatures and return values as :class:`PyramidFusion`; ``forward_collab``
    additionally publishes a BEV spatial trust map on
    ``self.uncertainty_output`` (and returns it when ``return_uncertainty=True``).

    With the default config (``temp_gamma=1.0``) the fused
    features are numerically identical to ``PyramidFusion``'s, so an existing
    collaboration base can be swapped over without retraining and backward alignment
    of new agents is untouched.
    """

    def __init__(self, model_cfg, input_channels=64):
        super().__init__(model_cfg, input_channels)

        levy_cfg = model_cfg.get("levy", {}) or {}

        self.tau = float(levy_cfg.get("rate_tau", 0.5))
        assert self.tau > 0, "levy.rate_tau must be positive"

        gamma = float(levy_cfg.get("temp_gamma", 1.0))
        if levy_cfg.get("learnable_gamma", False):
            # Optional single scalar; note this adds a key to the state_dict and so
            # breaks strict loading of a plain PyramidFusion checkpoint.
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma)))
        else:
            self.log_gamma = None
            self.gamma_value = gamma

        self.disagreement_reduction = levy_cfg.get("disagreement_reduction", "sum")
        assert self.disagreement_reduction in ("sum", "mean")

        self.disagreement_mode = levy_cfg.get("disagreement_mode", "raw")
        assert self.disagreement_mode in ("raw", "cosine"), (
            f"unknown levy.disagreement_mode {self.disagreement_mode}"
        )

        # Extra channels computed in the same forward. Publishing them together
        # is what makes a raw-vs-cosine comparison strictly paired: two separate
        # runs are not bit-identical (scores near score_threshold cross it), so
        # they do not even produce the same set of boxes.
        self.aux_disagreement_modes = tuple(
            levy_cfg.get("aux_disagreement_modes", ()) or ()
        )
        for _m in self.aux_disagreement_modes:
            assert _m in ("raw", "cosine"), f"unknown aux disagreement mode {_m}"
        if self.disagreement_mode == "cosine" and self.disagreement_reduction == "mean":
            print(
                "Levy: disagreement_reduction='mean' is ignored under "
                "disagreement_mode='cosine' (already channel-count free)."
            )

        # Run the ResNeXt trunk and the single-agent heads without gradients, e.g.
        # to train only what sits on top of a frozen collaboration base. Off by
        # default, which matches PyramidFusion.
        self.trunk_no_grad = bool(levy_cfg.get("trunk_no_grad", False))

        # Diagnostic: publish the per-agent terms of the cosine tr Sigma on
        # ``self.agent_deviation_output`` (see ``levy_weighted_fuse``).
        self.publish_agent_deviation = bool(levy_cfg.get("agent_deviation", False))
        self.agent_deviation_output = None

        self.compute_uncertainty = bool(levy_cfg.get("compute_uncertainty", True))
        self.detach_uncertainty = bool(levy_cfg.get("detach_uncertainty", True))

        scale_weights = levy_cfg.get("scale_weights", None)
        if scale_weights is None:
            scale_weights = [1.0] * self.num_levels
        assert len(scale_weights) == self.num_levels, (
            f"levy.scale_weights must have {self.num_levels} entries"
        )
        total = float(sum(scale_weights))
        self.scale_weights = [float(w) / total for w in scale_weights]

        # How the per-level maps become the level-0 trust map; see _build_trust_map.
        self.combined_mode = levy_cfg.get("combined_mode", "average")
        assert self.combined_mode in ("average", "decoded"), (
            f"unknown levy.combined_mode {self.combined_mode}"
        )
        if self.combined_mode == "decoded":
            self._check_decodable()
            if self.disagreement_mode == "cosine":
                print(
                    "Levy: combined_mode='decoded' is raw-only; the combined "
                    "disagreement/sigma ignore disagreement_mode='cosine'."
                )

        calibration = levy_cfg.get("calibration", {}) or {}
        # Plain floats, not buffers: the state_dict must stay identical to
        # PyramidFusion's so checkpoints remain interchangeable.
        self.calib_a = float(calibration.get("a", 1.0))
        self.calib_b = float(calibration.get("b", 0.0))

        # Runtime "a vehicle is probably here" mask: the fitted logistic over the
        # Levy signals, evaluated on the level-0 grid during the forward pass. The
        # coefficients come from ``levy_cells_study.py --export`` and are plain
        # config numbers, so the state_dict stays interchangeable with PyramidFusion.
        mask_cfg = levy_cfg.get("vehicle_mask", {}) or {}
        self.vehicle_mask_cfg = mask_cfg if mask_cfg.get("enabled", False) else None
        if self.vehicle_mask_cfg is not None:
            need = ("features", "mean", "scale", "coef", "intercept")
            missing = [k for k in need if k not in mask_cfg]
            if missing:
                raise ValueError(f"levy.vehicle_mask is missing {missing}")
            n = len(mask_cfg["features"])
            if not all(len(mask_cfg[k]) == n for k in ("mean", "scale", "coef")):
                raise ValueError("levy.vehicle_mask: features/mean/scale/coef differ in length")
            if not self.compute_uncertainty:
                raise ValueError("levy.vehicle_mask needs levy.compute_uncertainty: true")
            self.mask_threshold = float(mask_cfg.get("threshold", 0.5))
            self.mask_min_agents = int(mask_cfg.get("min_agents", 2))
            # Evidence has a different scale per regime (one contributor gives
            # Lambda = tau*exp(S) of that agent alone), so a rule fitted for one
            # regime must be confined to it.
            self.mask_max_agents = int(mask_cfg.get("max_agents", 0)) or None
        self.vehicle_mask_output = None

        self.uncertainty_output = None

        # Upstream hook: each agent attenuates its own features where its own
        # evidence is low, before they are warped and fused (pre-transmission).
        # None unless `levy.gateway.enable` is set -- see opencood/models/LevyHooks/.
        # Parameter-free, so the state_dict is unchanged either way.
        self.gateway = build_gateway_hook(
            levy_cfg.get("gateway"), num_levels=self.num_levels)

        print(
            f"Levy pyramid fusion: tau={self.tau}, gamma={gamma}, "
            f"disagreement={self.disagreement_mode}"
            f"/{self.disagreement_reduction}, scale_weights={self.scale_weights}, "
            f"combined_mode={self.combined_mode}"
        )

    @property
    def gamma(self):
        """Logit temperature; a tensor when ``learnable_gamma`` is on."""
        if self.log_gamma is not None:
            return torch.exp(self.log_gamma)
        return self.gamma_value

    def set_calibration(self, a, b):
        """
        Set the validation-fitted affine map sigma -> physical error scale
        (blueprint Sec. 5.3). Fit (a, b) once on a val split, e.g. by grid search
        against CRPS, then persist them in the yaml under ``levy.calibration``.
        """
        self.calib_a = float(a)
        self.calib_b = float(b)

    def forward_collab(
        self,
        spatial_features,
        record_len,
        affine_matrix,
        agent_modality_list=None,
        cam_crop_info=None,
        return_uncertainty=False,
        stochastic=False,
    ):
        """
        spatial_features : torch.tensor
            [sum(record_len), C, H, W]

        record_len : list
            cav num in each sample

        affine_matrix : torch.tensor
            [B, L, L, 2, 3]

        agent_modality_list : list
            len = sum(record_len), modality of each cav

        cam_crop_info : dict
            {'m2': {'crop_ratio_W_m2': 0.5, 'crop_ratio_H_m2': 0.5}}

        return_uncertainty : bool
            Also return the spatial trust map dict. It is stored on
            ``self.uncertainty_output`` either way.

        stochastic : bool
            Emit one draw of the stochastic operator instead of its mean.
            Evaluation only -- the draw is not differentiable.

        Returns
        -------
        fused_feature : torch.tensor
            [B, C, H, W]
        occ_map_list : list[torch.tensor]
            per-level foreground logits, as in PyramidFusion
        uncertainty : dict, only if ``return_uncertainty``
            'evidence', 'disagreement', 'sigma', 'sigma_calibrated', each
            [B, 1, H, W] at level-0 resolution, plus 'per_scale'. 'evidence' is
            None under ``combined_mode='decoded'``.
        """
        if stochastic and self.training:
            raise RuntimeError(
                "LevyPyramidFusion: the stochastic path is not differentiable; "
                "training must use the deterministic mean path (Theorem 1)."
            )

        crop_mask_flag = False
        if cam_crop_info is not None and len(cam_crop_info) > 0:
            crop_mask_flag = True
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {}
            for cam_modality in cam_modality_set:
                mask_list = [1 if x == cam_modality else 0 for x in agent_modality_list]
                mask_tensor = torch.tensor(mask_list, dtype=torch.bool)
                cam_agent_mask_dict[cam_modality] = mask_tensor

                # e.g. {m2: [0,0,0,1], m4: [0,1,0,0]}

        agent_logit_fns = [None] * self.num_levels
        decode_fns = [
            self._make_level_decoder(i)
            if self.compute_uncertainty and self.combined_mode == "decoded"
            else None
            for i in range(self.num_levels)
        ]

        with torch.set_grad_enabled(torch.is_grad_enabled()):
            feature_list = self.get_multiscale_feature(spatial_features)
            occ_map_list = [
                getattr(self, f"single_head_{i}")(feature_list[i])  # [N, 1, H, W]
                for i in range(self.num_levels)
            ]
        fused_feature_list = []
        per_scale = []
        level_dev = []
        if self.gateway is not None:
            self.gateway.begin_forward()

        for i in range(self.num_levels):
            occ_map = occ_map_list[i]
            score = torch.sigmoid(occ_map) + 1e-4

            # Upstream hook: gate the agents' own features before fusion. The
            # occupancy map (and so the fusion weights and the evidence) is left
            # as computed from the ungated features; only what each agent
            # contributes changes. Exempt levels/agents pass through untouched.
            if self.gateway is not None:
                feature_list[i] = self.gateway.apply(
                    i, feature_list[i], occ_map, record_len,
                    agent_modality_list=agent_modality_list,
                    cam_crop_info=cam_crop_info)

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = (
                        H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"]
                        - 4
                    )  # There may be unstable response values at the edges.
                    crop_W = (
                        W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"]
                        - 4
                    )  # There may be unstable response values at the edges.

                    start_h = int(H // 2 - crop_H // 2)
                    end_h = int(H // 2 + crop_H // 2)
                    start_w = int(W // 2 - crop_W // 2)
                    end_w = int(W // 2 + crop_W // 2)

                    cam_crop_mask[
                        cam_agent_mask_dict[cam_modality],
                        :,
                        start_h:end_h,
                        start_w:end_w,
                    ] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = (
                        1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]
                    )

                score = score * cam_crop_mask

            fused, log_evidence, disagreement, n_valid, aux, decoded, agent_dev = (
                levy_weighted_fuse(
                    feature_list[i],
                    score,
                    record_len,
                    affine_matrix,
                    self.align_corners,
                    tau=self.tau,
                    gamma=self.gamma,
                    disagreement_reduction=self.disagreement_reduction,
                    disagreement_mode=self.disagreement_mode,
                    aux_disagreement_modes=self.aux_disagreement_modes,
                    stochastic=stochastic,
                    agent_logit_fn=agent_logit_fns[i],
                    decode_fn=decode_fns[i],
                    agent_deviation=self.publish_agent_deviation,
                )
            )
            if agent_dev is not None:
                level_dev.append(agent_dev)
            fused_feature_list.append(fused)
            per_scale.append((log_evidence, disagreement, n_valid, aux, decoded))

        self.agent_deviation_output = level_dev if level_dev else None

        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        self.uncertainty_output = (
            self._build_trust_map(per_scale) if self.compute_uncertainty else None
        )
        # A mask that uses the detector's score map has to wait for the detection
        # head, which runs after fusion; the model then calls
        # ``complete_vehicle_mask``. A Levy-only mask is finished here.
        self.vehicle_mask_output = (
            self._vehicle_mask(self.uncertainty_output)
            if self.vehicle_mask_cfg is not None and self.uncertainty_output is not None
            else None
        )
        if self.vehicle_mask_output is not None:
            self.uncertainty_output.update(self.vehicle_mask_output)

        if return_uncertainty:
            return fused_feature, occ_map_list, self.uncertainty_output
        return fused_feature, occ_map_list

    @property
    def mask_needs_detector(self):
        """True when the configured mask uses the detection head's score map."""
        return (self.vehicle_mask_cfg is not None
                and "det_score" in self.vehicle_mask_cfg["features"])

    @torch.no_grad()
    def complete_vehicle_mask(self, det_score):
        """
        Finish a mask that needs the detector's score map.

        ``det_score`` is (B, 1, H, W) in [0, 1] -- the detection head's class
        probability, nearest-resampled to the level-0 grid if it differs. Called by
        the model right after its head; a Levy-only mask never needs this.
        """
        if self.vehicle_mask_cfg is None or self.uncertainty_output is None:
            return None
        self.vehicle_mask_output = self._vehicle_mask(self.uncertainty_output, det_score)
        self.uncertainty_output.update(self.vehicle_mask_output)
        return self.vehicle_mask_output

    @staticmethod
    def _mask_feature(uncertainty, key, size, det_score=None):
        """One model input as a level-0 map, transformed as it was when fitted."""
        if key == "det_score":
            if det_score is None:
                raise ValueError("levy.vehicle_mask uses 'det_score'; the model must call "
                                 "complete_vehicle_mask() after its detection head")
            value = det_score.float()
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="nearest")
            return value
        level, kind = key.split("_", 1)
        maps = uncertainty["per_scale"]
        source = uncertainty if level == "fused" else maps[int(level[1:])]
        if kind not in source:
            raise KeyError(f"levy.vehicle_mask needs '{key}', which this run does not publish")
        value = source[kind]
        if value.shape[-2:] != size:
            # Same nearest upsampling the level mean uses.
            value = F.interpolate(value.float(), size=size, mode="nearest")
        value = value.float()
        if kind == "disagreement":
            value = torch.log(value + 1e-6)
        elif kind == "evidence":
            value = torch.log(value.clamp(min=1e-30))
        return value

    @torch.no_grad()
    def _vehicle_mask(self, uncertainty, det_score=None):
        """
        P(a vehicle covers this cell) per level-0 cell, and its threshold mask.

        The logistic fitted offline on labelled cells, applied here to the maps
        this forward pass produced -- no ground truth and no second pass. Cells
        seen by fewer than ``min_agents`` agents are left at probability 0, since
        disagreement is identically zero there and the model never saw such cells.
        """
        cfg = self.vehicle_mask_cfg
        if self.mask_needs_detector and det_score is None:
            return None  # deferred until complete_vehicle_mask()
        size = uncertainty["n_valid"].shape[-2:]
        device = uncertainty["n_valid"].device
        logit = torch.full((uncertainty["n_valid"].shape[0], 1) + tuple(size),
                           float(cfg["intercept"]), device=device)
        for key, mean, scale, coef in zip(
            cfg["features"], cfg["mean"], cfg["scale"], cfg["coef"]
        ):
            x = self._mask_feature(uncertainty, key, size, det_score)
            logit = logit + float(coef) * (x - float(mean)) / float(scale)
        prob = torch.sigmoid(logit)
        usable = uncertainty["n_valid"] >= self.mask_min_agents
        if self.mask_max_agents is not None:
            usable = usable & (uncertainty["n_valid"] <= self.mask_max_agents)
        prob = prob * usable.to(prob.dtype)
        return {
            "vehicle_prob": prob,
            "vehicle_mask": prob >= self.mask_threshold,
            "vehicle_mask_usable": usable,
        }

    def _make_level_decoder(self, level):
        """
        Linear, pre-ReLU part of one level's decoder: G = BN(ConvT(F)).

        ConvT has kernel == stride and no bias, so each coarse cell only writes its
        own s x s block and a convex combination of agents passes through it
        unchanged; BN with running statistics is per-channel affine, which also
        commutes with a combination whose weights sum to 1. So sum_j p_j G_j is
        exactly the fused feature HEAL's decoder holds just before its ReLU.

        BN is always applied in inference form, so this diagnostic never updates
        the running statistics (and is only exact in eval, like BN itself).
        """
        convt, bn = self.deblocks[level][0], self.deblocks[level][1]

        def decode(feat):
            with torch.set_grad_enabled(
                torch.is_grad_enabled() and not self.detach_uncertainty
            ):
                return F.batch_norm(
                    convt(feat),
                    bn.running_mean,
                    bn.running_var,
                    bn.weight,
                    bn.bias,
                    training=False,
                    eps=bn.eps,
                )

        return decode

    def _check_decodable(self):
        """Refuse ``combined_mode='decoded'`` where the exactness argument fails."""
        if len(self.deblocks) != self.num_levels:
            raise ValueError(
                "levy.combined_mode='decoded' needs exactly one deblock per level "
                f"(got {len(self.deblocks)} for {self.num_levels} levels)"
            )
        for level, block in enumerate(self.deblocks):
            convt = block[0]
            ok = (
                isinstance(convt, nn.ConvTranspose2d)
                and convt.kernel_size == convt.stride
                and convt.padding == (0, 0)
                and convt.output_padding == (0, 0)
                and convt.dilation == (1, 1)
                and convt.bias is None
                and isinstance(block[1], nn.BatchNorm2d)
                and block[1].track_running_stats
            )
            if not ok:
                raise ValueError(
                    f"levy.combined_mode='decoded': deblock {level} is not a "
                    "bias-free ConvTranspose2d with kernel == stride followed by "
                    "BatchNorm2d, so the decoded moments would not be exact"
                )

    def _build_trust_map(self, per_scale):
        """
        per level, sigma_l = sqrt(tr Sigma_l * phi(Lambda_l)); the levels are then
        combined on the level-0 BEV grid according to ``combined_mode``:

        'average'
            nearest-upsample every level and take the omega-weighted average of
            evidence, disagreement and sigma, and of each aux channel
            (``disagreement_<mode>`` / ``sigma_<mode>``). ``log_evidence`` is the
            log of the averaged evidence. Nearest upsampling copies a coarse cell
            onto the s x s block its kernel == stride deblock writes to.
        'decoded'
            the levels are concatenated by HEAL's decoder, and the weighted
            variance of a concatenation is the sum of its blocks' variances; the
            levels' atoms are independent, so their stochastic variances add too:

                disagreement = sum_l tr Sigma_l(G)
                sigma        = sqrt( sum_l tr Sigma_l(G) * phi(Lambda_l) )

            with G the pre-ReLU decoded features (raw mode). ``scale_weights`` do
            not apply, ``disagreement_reduction='mean'`` divides by the total
            decoded channel count, and ``evidence`` is None: Lambda is defined per
            level only (see ``per_scale``).

        Each level also publishes ``n_valid`` and the binary overlap mask
        M_overlap = 1{N_valid >= 2}. Every grid-level metric must be reduced over
        M_overlap only: outside it at most one agent contributes, so tr Sigma is
        algebraically zero rather than merely small, and an unmasked mean reports
        the collaborator's footprint area instead of its disagreement.

        The fused ``n_valid`` is the per-cell minimum over every contributing level
        (nonzero omega under 'average', all levels under 'decoded'),
        nearest-upsampled, and the fused ``overlap_mask`` is derived from it --
        i.e. the AND of those levels' overlap masks.
        """
        base_size = per_scale[0][0].shape[-2:]
        decoded_mode = self.combined_mode == "decoded"

        sigma_acc = None
        evidence_acc = None
        disagreement_acc = None
        decoded_acc = None
        decoded_var_acc = None
        decoded_channels = 0
        aux_acc = {}
        scale_maps = []
        base_n_valid = None

        for i, (log_evidence, disagreement, n_valid, aux, decoded) in enumerate(
            per_scale
        ):
            if self.detach_uncertainty:
                log_evidence = log_evidence.detach()
                disagreement = disagreement.detach()

            evidence = torch.exp(log_evidence.clamp(max=60.0))
            phi = poisson_inverse_moment(evidence)
            sigma = torch.sqrt(F.relu(disagreement * phi))
            overlap_mask = n_valid >= 2

            scale_maps.append(
                {
                    "evidence": evidence,
                    "log_evidence": log_evidence,
                    "disagreement": disagreement,
                    "sigma": sigma,
                    "n_valid": n_valid,
                    "overlap_mask": overlap_mask,
                }
            )
            aux_maps = {}
            for _m, _d in aux.items():
                _d = _d.detach() if self.detach_uncertainty else _d
                aux_maps[f"disagreement_{_m}"] = _d
                aux_maps[f"sigma_{_m}"] = torch.sqrt(F.relu(_d * phi))
            scale_maps[-1].update(aux_maps)

            if decoded is not None:
                decoded = decoded.detach() if self.detach_uncertainty else decoded
                if decoded.shape[-2:] != base_size:
                    raise ValueError(
                        f"level {i} decodes to {tuple(decoded.shape[-2:])}, "
                        f"expected the level-0 grid {tuple(base_size)}"
                    )
                # Channel-summed, on the level-0 grid (not this level's own).
                scale_maps[-1]["disagreement_decoded"] = decoded

            if i > 0:
                evidence = F.interpolate(evidence, size=base_size, mode="nearest")
                disagreement = F.interpolate(
                    disagreement, size=base_size, mode="nearest"
                )
                sigma = F.interpolate(sigma, size=base_size, mode="nearest")
                phi = F.interpolate(phi, size=base_size, mode="nearest")
                n_valid = F.interpolate(
                    n_valid.float(), size=base_size, mode="nearest"
                ).to(n_valid.dtype)
                aux_maps = {
                    k: F.interpolate(v, size=base_size, mode="nearest")
                    for k, v in aux_maps.items()
                }

            w = self.scale_weights[i]
            # A fused cell is only as observed as its least-observed contributing
            # level; otherwise a level with a single agent there adds tr Sigma = 0
            # and silently dilutes the combination inside the mask.
            if decoded_mode or w > 0:
                base_n_valid = (
                    n_valid
                    if base_n_valid is None
                    else torch.minimum(base_n_valid, n_valid)
                )

            if decoded_mode:
                decoded_acc = decoded if decoded_acc is None else decoded_acc + decoded
                decoded_var = decoded * phi
                decoded_var_acc = (
                    decoded_var
                    if decoded_var_acc is None
                    else decoded_var_acc + decoded_var
                )
                decoded_channels += self.deblocks[i][0].out_channels
                continue

            sigma_acc = w * sigma if sigma_acc is None else sigma_acc + w * sigma
            evidence_acc = (
                w * evidence if evidence_acc is None else evidence_acc + w * evidence
            )
            disagreement_acc = (
                w * disagreement
                if disagreement_acc is None
                else disagreement_acc + w * disagreement
            )
            for k, v in aux_maps.items():
                aux_acc[k] = w * v if k not in aux_acc else aux_acc[k] + w * v

        if decoded_mode:
            norm = decoded_channels if self.disagreement_reduction == "mean" else 1
            disagreement_acc = decoded_acc / norm
            sigma_acc = torch.sqrt(F.relu(decoded_var_acc / norm))
            evidence_acc = None

        return {
            **aux_acc,
            "evidence": evidence_acc,
            "log_evidence": (
                None if evidence_acc is None else torch.log(evidence_acc.clamp(min=1e-30))
            ),
            "disagreement": disagreement_acc,
            "sigma": sigma_acc,
            "sigma_calibrated": self.calib_a * sigma_acc + self.calib_b,
            "n_valid": base_n_valid,
            "overlap_mask": base_n_valid >= 2,
            "per_scale": scale_maps,
        }
