import math

import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

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
    """
    _, C, H, W = x.shape
    B, L = affine_matrix.shape[:2]
    split_x = regroup(x, record_len)
    split_score = regroup(score, record_len)
    split_logits = regroup(logits, record_len) if logits is not None else None

    out = []
    log_evidence_list = []
    disagreement_list = []

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

    return (torch.stack(out),
            torch.stack(log_evidence_list),
            torch.stack(disagreement_list))


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
            'evidence', 'disagreement', 'sigma', 'sigma_calibrated', each
            [B, 1, H, W] at level-0 resolution, plus 'per_scale'.
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

            fused, log_evidence, disagreement = levy_weighted_fuse(
                feature_list[i], score, record_len, affine_matrix,
                self.align_corners,
                tau=self.tau,
                gamma=self.gamma,
                logits=occ_map if self.score_mode == 'logit' else None,
                disagreement_reduction=self.disagreement_reduction,
                stochastic=stochastic,
            )
            fused_feature_list.append(fused)
            per_scale.append((log_evidence, disagreement))

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

        for i, (log_evidence, disagreement) in enumerate(per_scale):
            if self.detach_uncertainty:
                log_evidence = log_evidence.detach()
                disagreement = disagreement.detach()

            evidence = torch.exp(log_evidence.clamp(max=60.0))
            phi = poisson_inverse_moment(evidence)
            sigma = torch.sqrt(F.relu(disagreement * phi))

            scale_maps.append({
                'evidence': evidence,
                'log_evidence': log_evidence,
                'disagreement': disagreement,
                'sigma': sigma,
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
            'per_scale': scale_maps,
        }