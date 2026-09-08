"""
Levy Spatial Pyramid Fusion for HEAL.

Drop-in replacement for :class:`opencood.models.fuse_modules.pyramid_fuse.PyramidFusion`
that re-casts the agent-wise ``softmax`` fusion as stochastic integration against an
inhomogeneous Poisson random measure over the BEV grid ("Levy Attention").

Reference: ``levy-heal-fusion-blueprint.md`` at the repository root.

Why
---
HEAL fuses aligned agent features with agent-wise weights
``W_j(s) = softmax_j(S_j(s))``.  The row normalisation divides out the *absolute*
scale of the foreground evidence: two agents that both see nothing (S = -10) and two
agents that both firmly see a vehicle (S = +10) produce the identical weight pair
(0.5, 0.5) and therefore the identical fused feature.  Coverage and sensor conflict
are invisible downstream.

What changes
------------
Per BEV cell ``s`` and pyramid level ``l`` we keep an *unnormalised* intensity

    lambda_j(s) = tau * exp(gamma * S_j(s))
    Lambda(s)   = sum_j lambda_j(s)                       (spatial evidence)
    p_j(s)      = lambda_j(s) / Lambda(s)                 (fusion weights)

For ``gamma == 1`` the weights ``p_j`` are algebraically identical to HEAL's
``W_j`` (``tau`` cancels), so the fused features -- and hence a frozen HEAL
collaboration base -- are bit-for-bit reproduced.  ``LevyPyramidFusion`` also adds no
parameters, so it loads a ``PyramidFusion`` ``state_dict`` unchanged (unless
``learnable_gamma`` is enabled).

What we get for free, in the same forward pass:

    evidence      Lambda(s)                                    coverage / visibility
    disagreement  tr Sigma(s) = sum_j p_j ||F_j||^2 - ||F||^2   sensor conflict
    sigma_hat(s)  sqrt(tr Sigma(s) * phi(Lambda(s)))            calibrated scale
    coverage      #{j : agent j observes s}                     exact agent count

``coverage`` is bookkeeping rather than part of the Levy construction, but it is
what makes the other three readable: the ego is warped by the identity, so it
contributes intensity at every cell of its own grid and ``evidence > 0`` is
therefore true everywhere -- useless as a mask.  Conflict statistics have to be
conditioned on ``coverage >= 2``, since tr Sigma is identically 0 below that.

with ``phi(Lambda) = E[1{Z>=1}/Z]``, ``Z ~ Poisson(Lambda)`` (Theorem 2, exact
variance identity).  For ``Lambda >~ 2`` this factorises as
``sigma_hat ~= sqrt(disagreement / evidence)``.

The returned fused feature is always the deterministic mean path
``F(s) = sum_j p_j(s) F_j(s)``, which by Theorem 1 is the exact expectation of the
stochastic operator -- exact gradients, zero train/inference gap.  The stochastic
operator itself is available via ``stochastic=True`` for Monte-Carlo studies.

Usage
-----
Swap the import in any of the HEAL model wrappers::

    from opencood.models.fuse_modules.levy_pyramid_fuse import LevyPyramidFusion
    self.pyramid_backbone = LevyPyramidFusion(args['fusion_backbone'])

    fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(...)
    trust_map = self.pyramid_backbone.uncertainty_output['sigma']   # [B, 1, H, W]

or ask for it explicitly::

    fused_feature, occ_outputs, unc = self.pyramid_backbone.forward_collab(
        ..., return_uncertainty=True)

YAML (all keys optional, defaults shown)::

    fusion_backbone:
      resnext: true
      layer_nums: [3, 5, 8]
      ...
      levy:
        rate_tau: 0.5             # base Poisson rate
        temp_gamma: 1.0           # logit temperature; 1.0 == HEAL softmax weights
        learnable_gamma: false    # adds one scalar parameter (breaks state_dict parity)
        score_mode: heal          # 'heal' (sigmoid score, HEAL-equivalent) | 'logit'
        disagreement_reduction: sum   # 'sum' (blueprint) | 'mean' over channels;
                                  # 'mean' makes levels with different channel
                                  # counts (64/128/256) directly comparable
        scale_weights: null       # per-level omega_l, defaults to uniform
        compute_uncertainty: true
        detach_uncertainty: true  # keep the trust map out of the training graph
        calibration: {a: 1.0, b: 0.0}   # sigma -> physical error scale

Author: added on top of HEAL's PyramidFusion by Yifan Lu <yifan_lu@sjtu.edu.cn>.
License: TDG-Attribution-NonCommercial-NoDistrib
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple

# Additive stand-in for log(0). Finite (fp16-safe) yet exp()es to exactly 0.
LOG_ZERO = -1e4


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

    # One host sync to size the series: the k-th term peaks at k ~ L with spread
    # sqrt(L), so L + 6*sqrt(L) + 8 terms is far past machine precision.
    if lam_series.numel() == 0:
        n_terms = 1
    else:
        lam_max = float(lam_series.detach().max())
        n_terms = int(min(max_terms,
                          math.ceil(lam_max + 6.0 * math.sqrt(lam_max) + 8.0)))

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
    Draw the stochastic fused feature of blueprint Sec. 3 for one batch element.

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


def levy_weighted_fuse(x, score, record_len, affine_matrix, align_corners,
                       tau=0.5, gamma=1.0, logits=None,
                       disagreement_reduction='sum', stochastic=False):
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
    logits : torch.Tensor, optional
        Raw foreground logits, (sum(n_cav), 1, H, W). When given they drive the
        intensity instead of ``score`` (``score_mode='logit'``); ``score`` still
        supplies the validity mask.
    disagreement_reduction : str
        'sum' (blueprint) or 'mean' over the channel axis. 'sum' makes tr Sigma grow
        with the level's channel count, which the per-level omega_l weights are there
        to rebalance; 'mean' removes that dependence up front.
    stochastic : bool
        Return one draw of the stochastic operator instead of its mean. Non
        differentiable; evaluation only.

    Returns
    -------
    out : torch.Tensor
        Fused features, (B, C, H, W).
    log_evidence : torch.Tensor
        log Lambda(s), (B, 1, H, W). Log space so that large gamma cannot overflow.
    disagreement : torch.Tensor
        tr Sigma(s) >= 0, (B, 1, H, W).
    coverage : torch.Tensor
        Number of agents contributing to each cell, (B, 1, H, W). This is the
        exact agent count, not a proxy: tr Sigma is identically 0 wherever
        coverage < 2, so any statistic about sensor conflict has to be
        conditioned on this rather than on the evidence, which the ego alone
        already makes positive everywhere in its own grid.
    """
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)
    split_logits = regroup(logits, record_len) if logits is not None else None

    out = []
    log_evidence_list = []
    disagreement_list = []
    coverage_list = []

    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = affine_matrix[b][:N, :N, :, :]
        i = 0  # ego
        feature_in_ego = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W), align_corners=align_corners)
        scores_in_ego = warp_affine_simple(split_score[b],
                                           t_matrix[i, :, :, :],
                                           (H, W), align_corners=align_corners)
        # HEAL masks these to -inf before the softmax; here they simply carry zero
        # intensity, so they contribute neither weight nor evidence.
        valid = (scores_in_ego != 0)

        if split_logits is None:
            s_map = scores_in_ego
        else:
            s_map = warp_affine_simple(split_logits[b],
                                       t_matrix[i, :, :, :],
                                       (H, W), align_corners=align_corners)

        # lambda_j = tau * exp(gamma * S_j), held in log space.
        log_lambda = math.log(tau) + gamma * s_map
        log_lambda = log_lambda.masked_fill(~valid, LOG_ZERO)

        # Lambda = sum_j lambda_j   (spatial evidence)
        log_evidence = torch.logsumexp(log_lambda, dim=0, keepdim=True)

        # p_j = lambda_j / Lambda. The explicit mask keeps cells that *no* agent
        # observes at p = 0 (HEAL's nan -> 0 branch) instead of a spurious 1/N.
        weights = torch.exp(log_lambda - log_evidence) * valid.to(log_lambda.dtype)

        # Mean path: E[stochastic fused feature] (Theorem 1).
        fused = torch.sum(feature_in_ego * weights, dim=0)

        # tr Sigma = sum_j p_j ||F_j||^2 - ||F_mean||^2  (weighted variance).
        sq_norm = torch.sum(feature_in_ego ** 2, dim=1, keepdim=True)
        second_moment = torch.sum(weights * sq_norm, dim=0)
        disagreement = (second_moment - torch.sum(fused ** 2, dim=0, keepdim=True))
        disagreement = disagreement.clamp(min=0.0)
        if disagreement_reduction == 'mean':
            disagreement = disagreement / C

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
        coverage_list.append(valid.to(log_lambda.dtype).sum(dim=0))

    return (torch.stack(out),
            torch.stack(log_evidence_list),
            torch.stack(disagreement_list),
            torch.stack(coverage_list))


class LevyPyramidFusion(PyramidFusion):
    """
    HEAL's PyramidFusion with the agent-wise softmax replaced by Levy attention.

    Same parameters, same ``state_dict``, same ``forward_single`` / ``forward_collab``
    signatures and return values as :class:`PyramidFusion`; ``forward_collab``
    additionally publishes a BEV spatial trust map on
    ``self.uncertainty_output`` (and returns it when ``return_uncertainty=True``).

    With the default config (``temp_gamma=1.0``, ``score_mode='heal'``) the fused
    features are numerically identical to ``PyramidFusion``'s, so an existing
    collaboration base can be swapped over without retraining and backward alignment
    of new agents is untouched.
    """

    def __init__(self, model_cfg, input_channels=64):
        super().__init__(model_cfg, input_channels)

        levy_cfg = model_cfg.get('levy', {}) or {}

        self.tau = float(levy_cfg.get('rate_tau', 0.5))
        assert self.tau > 0, "levy.rate_tau must be positive"

        gamma = float(levy_cfg.get('temp_gamma', 1.0))
        if levy_cfg.get('learnable_gamma', False):
            # Optional single scalar; note this adds a key to the state_dict and so
            # breaks strict loading of a plain PyramidFusion checkpoint.
            self.log_gamma = nn.Parameter(torch.tensor(math.log(gamma)))
        else:
            self.log_gamma = None
            self.gamma_value = gamma

        self.score_mode = levy_cfg.get('score_mode', 'heal')
        assert self.score_mode in ('heal', 'logit'), \
            f"unknown levy.score_mode {self.score_mode}"

        self.disagreement_reduction = levy_cfg.get('disagreement_reduction', 'sum')
        assert self.disagreement_reduction in ('sum', 'mean')

        self.compute_uncertainty = bool(levy_cfg.get('compute_uncertainty', True))
        self.detach_uncertainty = bool(levy_cfg.get('detach_uncertainty', True))

        scale_weights = levy_cfg.get('scale_weights', None)
        if scale_weights is None:
            scale_weights = [1.0] * self.num_levels
        assert len(scale_weights) == self.num_levels, \
            f"levy.scale_weights must have {self.num_levels} entries"
        total = float(sum(scale_weights))
        self.scale_weights = [float(w) / total for w in scale_weights]

        calibration = levy_cfg.get('calibration', {}) or {}
        # Plain floats, not buffers: the state_dict must stay identical to
        # PyramidFusion's so checkpoints remain interchangeable.
        self.calib_a = float(calibration.get('a', 1.0))
        self.calib_b = float(calibration.get('b', 0.0))

        self.uncertainty_output = None

        print(f'Levy pyramid fusion: tau={self.tau}, gamma={gamma}, '
              f'score_mode={self.score_mode}, scale_weights={self.scale_weights}')

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

    def forward_collab(self, spatial_features, record_len, affine_matrix,
                       agent_modality_list=None, cam_crop_info=None,
                       return_uncertainty=False, stochastic=False):
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
            'evidence', 'disagreement', 'sigma', 'sigma_calibrated', 'coverage',
            each [B, 1, H, W] at level-0 resolution, plus 'per_scale'.
        """
        if stochastic and self.training:
            raise RuntimeError(
                "LevyPyramidFusion: the stochastic path is not differentiable; "
                "training must use the deterministic mean path (Theorem 1).")

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

        feature_list = self.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        occ_map_list = []
        per_scale = []

        for i in range(self.num_levels):
            occ_map = eval(f"self.single_head_{i}")(feature_list[i])  # [N, 1, H, W]
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map, device=occ_map.device)
                _, _, H, W = cam_crop_mask.shape
                for cam_modality in cam_modality_set:
                    crop_H = H / cam_crop_info[cam_modality][f"crop_ratio_H_{cam_modality}"] - 4 # There may be unstable response values at the edges.
                    crop_W = W / cam_crop_info[cam_modality][f"crop_ratio_W_{cam_modality}"] - 4 # There may be unstable response values at the edges.

                    start_h = int(H//2-crop_H//2)
                    end_h = int(H//2+crop_H//2)
                    start_w = int(W//2-crop_W//2)
                    end_w = int(W//2+crop_W//2)

                    cam_crop_mask[cam_agent_mask_dict[cam_modality],:,start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[cam_modality]] = 1 - cam_crop_mask[cam_agent_mask_dict[cam_modality]]

                score = score * cam_crop_mask

            fused, log_evidence, disagreement, coverage = levy_weighted_fuse(
                feature_list[i], score, record_len, affine_matrix,
                self.align_corners,
                tau=self.tau,
                gamma=self.gamma,
                logits=occ_map if self.score_mode == 'logit' else None,
                disagreement_reduction=self.disagreement_reduction,
                stochastic=stochastic,
            )
            fused_feature_list.append(fused)
            per_scale.append((log_evidence, disagreement, coverage))

        fused_feature = self.decode_multiscale_feature(fused_feature_list)

        self.uncertainty_output = self._build_trust_map(per_scale) \
            if self.compute_uncertainty else None

        if return_uncertainty:
            return fused_feature, occ_map_list, self.uncertainty_output
        return fused_feature, occ_map_list

    def _build_trust_map(self, per_scale):
        """
        Blueprint Sec. 4-5: per level, sigma_l = sqrt(tr Sigma_l * phi(Lambda_l));
        then bilinearly upsample every level to the level-0 BEV grid and take the
        omega-weighted average.
        """
        base_size = per_scale[0][0].shape[-2:]

        sigma_acc = None
        evidence_acc = None
        disagreement_acc = None
        scale_maps = []

        for i, (log_evidence, disagreement, coverage) in enumerate(per_scale):
            if self.detach_uncertainty:
                log_evidence = log_evidence.detach()
                disagreement = disagreement.detach()
                coverage = coverage.detach()

            evidence = torch.exp(log_evidence.clamp(max=60.0))
            phi = poisson_inverse_moment(evidence)
            sigma = torch.sqrt(F.relu(disagreement * phi))

            scale_maps.append({
                'evidence': evidence,
                'log_evidence': log_evidence,
                'disagreement': disagreement,
                'sigma': sigma,
                'coverage': coverage,
            })

            if i > 0:
                evidence = F.interpolate(evidence, size=base_size,
                                         mode='bilinear', align_corners=False)
                disagreement = F.interpolate(disagreement, size=base_size,
                                             mode='bilinear', align_corners=False)
                sigma = F.interpolate(sigma, size=base_size,
                                      mode='bilinear', align_corners=False)

            w = self.scale_weights[i]
            sigma_acc = w * sigma if sigma_acc is None else sigma_acc + w * sigma
            evidence_acc = w * evidence if evidence_acc is None else evidence_acc + w * evidence
            disagreement_acc = w * disagreement if disagreement_acc is None \
                else disagreement_acc + w * disagreement

        return {
            'evidence': evidence_acc,
            'disagreement': disagreement_acc,
            'sigma': sigma_acc,
            'sigma_calibrated': self.calib_a * sigma_acc + self.calib_b,
            # Level 0 is the reference grid, so its agent count is exact. The
            # deeper levels are not omega-averaged in: interpolating counts
            # would produce fractional agents and blur the >= 2 boundary that
            # tells "sensor conflict" apart from "only one agent here".
            'coverage': scale_maps[0]['coverage'],
            'per_scale': scale_maps,
        }
