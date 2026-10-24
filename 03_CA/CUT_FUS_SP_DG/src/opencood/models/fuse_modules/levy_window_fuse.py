"""
Windowed cross-agent Levy attention for HEAL's pyramid fusion.

PyramidFusion fuses each BEV cell from the <= N agents' features at that same
cell, weighted by a softmax over each agent's own foreground score. Here every
ego cell s instead *queries* all agents' features in a k x k window around s:

    q(s)        = W_q F_ego(s)                         (query: the ego's feature)
    k_j(s+o)    = W_k F_j(s+o) + e_role(j)              (key: agent j at offset o)
    logit(j, o) = sqrt(d) cos(q, k) + b_o + beta * S_j(s+o)  (S = sigmoid(occ) + 1e-4)
    kappa       = exp(logit),  Lambda = tau * sum kappa (evidence)
    p           = kappa / sum kappa
    fused(s)    = sum_{j,o} p F_j(s+o)                  (the mean path, Thm 1)
    trSigma(s)  = sum p ||F||^2 - ||fused||^2            (disagreement, Thm 2)

The kernel is the Levy Attention kernel sqrt(d) * cos(q, k) (bounded, so
Lambda is comparable across cells). Keys whose agent did not observe or did not
send the cell (score 0 after the camera crop / comm mask, or outside the map)
are excluded.

The paper's mollifier is dropped: keys sit on grid cells, and as its bandwidth
goes to 0 the operator is exactly this discrete kernel attention, for which
Theorems 1-2 hold as stated.

Config (model.args.fusion_backbone):
    core_method: levy_window
    window:
      sizes: [3, 5, 5]      # odd window per pyramid level
      d_qk: 32              # query/key width
      rate_tau: 0.5
      beta_init: 1.0        # weight of the foreground prior S_j
      publish_signals: true # keep per-level log Lambda / trSigma / n_keys
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.fuse_modules.fusion_in_one import regroup
from opencood.models.fuse_modules.levy_pyramid_fuse import poisson_inverse_moment
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

NEG_INF = -1e9


class WindowLevel(nn.Module):
    """Query/key projections and window biases for one pyramid level."""

    def __init__(self, channels, d_qk, window, beta_init):
        super().__init__()
        assert window % 2 == 1, "window sizes must be odd"
        self.window = window
        self.d_qk = d_qk
        self.q_proj = nn.Conv2d(channels, d_qk, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, d_qk, kernel_size=1)
        # 0 = ego, 1 = collaborator; added to the keys. Zero-init.
        self.role = nn.Parameter(torch.zeros(2, d_qk))
        self.offset_bias = nn.Parameter(torch.zeros(window * window))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))


class ResidualGate(nn.Module):
    """
    alpha(s) in (0, 1) blending HEAL's per-cell fusion (alpha -> 0) with the
    window fusion (alpha -> 1) at one pyramid level.

    Inputs per ego cell: a 3x3 context of the ego's own feature (whether the ego
    observed the area or sits in a LiDAR shadow), the ego's foreground score, the
    best collaborator score, the agent count, and the window's own evidence
    log(1 + Lambda) and cosine disagreement (both detached).
    """

    N_SCALARS = 5

    def __init__(self, channels, init_alpha=0.5, hidden=16, ctx=8):
        super().__init__()
        self.ctx = nn.Sequential(nn.Conv2d(channels, ctx, 3, padding=1), nn.ReLU())
        self.mlp = nn.Sequential(nn.Conv2d(ctx + self.N_SCALARS, hidden, 1), nn.ReLU(),
                                 nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.mlp[2].weight)
        a = min(max(float(init_alpha), 1e-4), 1 - 1e-4)
        nn.init.constant_(self.mlp[2].bias, math.log(a / (1 - a)))

    def forward(self, ego_feat, scalars):
        """ego_feat [1, C, H, W], scalars [1, 5, H, W] -> alpha [H, W]."""
        return torch.sigmoid(self.mlp(torch.cat([self.ctx(ego_feat), scalars], 1)))[0, 0]


class LevyWindowFusion(PyramidFusion):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__(model_cfg, input_channels)
        cfg = model_cfg.get("window", {}) or {}
        assert cfg.get("kernel", "cosine") == "cosine", "only the cosine kernel is supported"
        sizes = cfg.get("sizes", [3, 5, 5])
        assert len(sizes) == self.num_levels
        self.d_qk = int(cfg.get("d_qk", 32))
        self.tau = float(cfg.get("rate_tau", 0.5))
        self.publish_signals = bool(cfg.get("publish_signals", True))
        # Evaluation-only: replace the learned beta of every level by this value
        # (scalar or one per level) without touching the checkpoint.
        bo = cfg.get("beta_override", None)
        if bo is not None and not isinstance(bo, (list, tuple)):
            bo = [bo] * len(sizes)
        self.beta_override = None if bo is None else [float(b) for b in bo]
        self.levels = nn.ModuleList(
            WindowLevel(self.model_cfg["num_filters"][i], self.d_qk, int(sizes[i]),
                        float(cfg.get("beta_init", 1.0)))
            for i in range(self.num_levels)
        )
        # Option B: fused = (1 - alpha) * HEAL + alpha * window, alpha from a gate.
        rcfg = cfg.get("heal_residual", {}) or {}
        self.residual = bool(rcfg.get("enable", False))
        self.gates = (
            nn.ModuleList(ResidualGate(self.model_cfg["num_filters"][i],
                                       float(rcfg.get("init_alpha", 0.5)))
                          for i in range(self.num_levels))
            if self.residual else None
        )
        # per level: dict of [B, 1, H_l, W_l] maps, detached
        self.signals = None
        # Same layout as LevyPyramidFusion.uncertainty_output, so the Levy
        # recorders (tools/levy/levy_collect.py) read this model unchanged.
        self.uncertainty_output = None
        self.compute_uncertainty = self.publish_signals
        print(f"Levy window fusion: windows={list(sizes)}, "
              f"d_qk={self.d_qk}, tau={self.tau}"
              + (f", beta_override={self.beta_override}" if self.beta_override else "")
              + (f", HEAL residual gate (init alpha {rcfg.get('init_alpha', 0.5)})"
                 if self.residual else ""))

    def _logit(self, q, k):
        """q [1, d, H, W], k [N, d, H, W] (both unit-norm) -> [N, H, W]."""
        return math.sqrt(self.d_qk) * (q * k).sum(1)

    def window_fuse(self, level, x, score, record_len, affine_matrix):
        """
        x [sum N, C, H, W], score [sum N, 1, H, W] (0 = not observed / not sent).
        Returns fused [B, C, H, W] and, if publishing, a signal dict.
        """
        lv = self.levels[level]
        beta = lv.beta if self.beta_override is None else self.beta_override[level]
        r = lv.window // 2
        _, C, H, W = x.shape
        B = affine_matrix.shape[0]
        split_x = regroup(x, record_len)
        split_s = regroup(score, record_len)

        out, log_ev, tr_sig, tr_cos, n_keys, n_agents, alphas = [], [], [], [], [], [], []
        need_stats = self.publish_signals or self.residual
        for b in range(B):
            N = int(record_len[b])
            t = affine_matrix[b][:N, :N][0]  # into the ego (agent 0) frame
            feat = warp_affine_simple(split_x[b], t, (H, W), align_corners=self.align_corners)
            s = warp_affine_simple(split_s[b], t, (H, W), align_corners=self.align_corners)
            valid = s > 0

            q = lv.q_proj(feat[0:1])
            k = lv.k_proj(feat)
            role = torch.ones(N, dtype=torch.long, device=x.device)
            role[0] = 0
            k = k + lv.role[role][:, :, None, None]
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)

            pad = (r, r, r, r)
            kp = F.pad(k, pad)
            fp = F.pad(feat, pad)
            sp = F.pad(s, pad)
            vp = F.pad(valid.to(s.dtype), pad)

            # logits for every (agent, offset): [N, K, H, W], K = window^2
            logits, masks = [], []
            o = 0
            for dy in range(lv.window):
                for dx in range(lv.window):
                    k_o = kp[:, :, dy:dy + H, dx:dx + W]
                    s_o = sp[:, 0, dy:dy + H, dx:dx + W]
                    logits.append(self._logit(q, k_o) + lv.offset_bias[o] + beta * s_o)
                    masks.append(vp[:, 0, dy:dy + H, dx:dx + W] > 0)
                    o += 1
            logits = torch.stack(logits, 1)
            masks = torch.stack(masks, 1)
            logits = logits.masked_fill(~masks, NEG_INF)

            # normalise jointly over agents and offsets
            flat = logits.permute(2, 3, 0, 1).reshape(H, W, -1)
            p = torch.softmax(flat, dim=-1)
            any_valid = masks.flatten(0, 1).any(0)  # [H, W]
            p = p * any_valid[..., None].to(p.dtype)
            p = p.reshape(H, W, N, -1).permute(2, 3, 0, 1)  # [N, K, H, W]

            fused = torch.zeros(C, H, W, device=x.device, dtype=x.dtype)
            sq = torch.zeros(H, W, device=x.device, dtype=x.dtype) if need_stats else None
            unit = torch.zeros(C, H, W, device=x.device, dtype=x.dtype) if need_stats else None
            o = 0
            for dy in range(lv.window):
                for dx in range(lv.window):
                    f_o = fp[:, :, dy:dy + H, dx:dx + W]
                    w_o = p[:, o]
                    fused = fused + (w_o[:, None] * f_o).sum(0)
                    if sq is not None:
                        with torch.no_grad():
                            sq += (w_o * f_o.pow(2).sum(1)).sum(0)
                            unit += (w_o[:, None] * F.normalize(f_o, dim=1)).sum(0)
                    o += 1
            window_fused = fused

            if self.residual:
                # HEAL: per-cell softmax over the agents' own scores (weighted_fuse).
                s_heal = s.masked_fill(~valid, NEG_INF)
                w_heal = torch.softmax(s_heal, dim=0) * valid.any(0, keepdim=True).to(s.dtype)
                heal_fused = (w_heal * feat).sum(0)
                with torch.no_grad():
                    lse_g = torch.logsumexp(flat, dim=-1)
                    lam = torch.exp((math.log(self.tau) + lse_g).clamp(max=30.0)) * any_valid
                    cos_g = (any_valid.to(unit.dtype) - unit.pow(2).sum(0)).clamp_min(0)
                    s_col = s[1:, 0].max(0).values if N > 1 else torch.zeros_like(s[0, 0])
                    scalars = torch.stack([
                        s[0, 0], s_col, valid[:, 0].sum(0).to(s.dtype) / 5.0,
                        torch.log1p(lam) / 8.0, cos_g])[None]
                alpha = self.gates[level](feat[0:1], scalars)
                fused = heal_fused + alpha[None] * (window_fused - heal_fused)
                alphas.append(alpha.detach())
            out.append(fused)

            if self.publish_signals:
                with torch.no_grad():
                    lse = torch.logsumexp(flat.detach(), dim=-1)
                    log_ev.append((math.log(self.tau) + lse).masked_fill(~any_valid, NEG_INF))
                    tr_sig.append((sq - window_fused.detach().pow(2).sum(0)).clamp_min(0))
                    # magnitude-free variant: 1 - ||sum p F_hat||^2 on unit features
                    cos = (any_valid.to(unit.dtype) - unit.pow(2).sum(0)).clamp_min(0)
                    tr_cos.append(cos)
                    n_keys.append(masks.flatten(0, 1).sum(0))
                    n_agents.append(masks.any(1).sum(0))

        fused = torch.stack(out)
        signals = None
        if self.publish_signals:
            signals = {
                "log_evidence": torch.stack(log_ev)[:, None],
                "disagreement": torch.stack(tr_sig)[:, None],
                "n_keys": torch.stack(n_keys)[:, None],
                "n_agents": torch.stack(n_agents)[:, None],
                "disagreement_cosine": torch.stack(tr_cos)[:, None],
            }
            if alphas:
                signals["alpha"] = torch.stack(alphas)[:, None]
        return fused, signals

    def forward_collab(self, spatial_features, record_len, affine_matrix,
                       agent_modality_list=None, cam_crop_info=None):
        crop_mask_flag = cam_crop_info is not None and len(cam_crop_info) > 0
        if crop_mask_flag:
            cam_modality_set = set(cam_crop_info.keys())
            cam_agent_mask_dict = {
                m: torch.tensor([x == m for x in agent_modality_list], dtype=torch.bool)
                for m in cam_modality_set
            }

        feature_list = self.get_multiscale_feature(spatial_features)
        occ_map_list = [
            getattr(self, f"single_head_{i}")(feature_list[i])
            for i in range(self.num_levels)
        ]
        if self.comm_mask is not None:
            self.comm_mask.begin_forward(occ_map_list, feature_list)

        fused_feature_list = []
        signals = []
        for i in range(self.num_levels):
            occ_map = occ_map_list[i]
            score = torch.sigmoid(occ_map) + 1e-4

            if crop_mask_flag and not self.training:
                cam_crop_mask = torch.ones_like(occ_map)
                _, _, H, W = cam_crop_mask.shape
                for m in cam_modality_set:
                    crop_H = H / cam_crop_info[m][f"crop_ratio_H_{m}"] - 4
                    crop_W = W / cam_crop_info[m][f"crop_ratio_W_{m}"] - 4
                    start_h, end_h = int(H // 2 - crop_H // 2), int(H // 2 + crop_H // 2)
                    start_w, end_w = int(W // 2 - crop_W // 2), int(W // 2 + crop_W // 2)
                    cam_crop_mask[cam_agent_mask_dict[m], :, start_h:end_h, start_w:end_w] = 0
                    cam_crop_mask[cam_agent_mask_dict[m]] = 1 - cam_crop_mask[cam_agent_mask_dict[m]]
                score = score * cam_crop_mask

            if self.comm_mask is not None:
                feature_list[i], score = self.comm_mask.apply(
                    i, feature_list[i], score, occ_map, record_len)

            fused, sig = self.window_fuse(i, feature_list[i], score, record_len, affine_matrix)
            fused_feature_list.append(fused)
            signals.append(sig)

        self.signals = signals if self.publish_signals else None
        self.uncertainty_output = (
            self._uncertainty(signals) if self.publish_signals else None)
        fused_feature = self.decode_multiscale_feature(fused_feature_list)
        return fused_feature, occ_map_list

    @staticmethod
    def _uncertainty(signals):
        """
        LevyPyramidFusion-style trust map from the per-level signals.

        per level: evidence Lambda, log_evidence, disagreement (raw tr Sigma,
        channel sum), disagreement_cosine, sigma = sqrt(tr Sigma * phi(Lambda))
        (Theorem 2), n_keys, n_valid = agents with >= 1 valid key in the window,
        and overlap_mask = n_valid >= 2. The combined (level-0) entries are the
        level-0 maps.
        """
        per_scale = []
        for sig in signals:
            log_ev = sig["log_evidence"]
            valid = sig["n_keys"] > 0
            evidence = torch.exp(log_ev.clamp(max=60.0)) * valid
            phi = poisson_inverse_moment(evidence)
            per_scale.append({
                "evidence": evidence,
                "log_evidence": log_ev.masked_fill(~valid, float("nan")),
                "disagreement": sig["disagreement"],
                "disagreement_cosine": sig["disagreement_cosine"],
                "sigma": torch.sqrt(sig["disagreement"] * phi),
                "n_keys": sig["n_keys"],
                "n_valid": sig["n_agents"],
                "overlap_mask": sig["n_agents"] >= 2,
            })
            if "alpha" in sig:
                per_scale[-1]["alpha"] = sig["alpha"]
        fused = {k: per_scale[0][k] for k in
                 ("evidence", "log_evidence", "disagreement", "disagreement_cosine",
                  "sigma", "n_valid", "overlap_mask")}
        fused["per_scale"] = per_scale
        return fused

