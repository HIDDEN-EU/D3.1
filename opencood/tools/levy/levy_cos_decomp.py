"""
Run ``inference_heter_in_order.py`` while recording the parts of the cosine
disagreement, per pyramid level, alongside the usual Levy statistics.

For two agents with weights p, 1-p and unit-norm features u_ego, u_col,

    disagreement_cosine = 1 - ||p u_ego + (1-p) u_col||^2
                        = 2 p (1-p) * (1 - cos(u_ego, u_col))
                        = cd_balance * cd_gap

Extra per-level maps (zero outside cells both agents observe):
    cd_balance   1 - sum_j p_j^2          weight balance (0.5 when p = 0.5)
    cd_gap       1 - cos(F_ego, F_col)    angular conflict (two agents only)
    cd_norm_ego  ||F_ego||,  cd_norm_col ||F_col||
    cd_p_col     weight on the collaborator

The fusion itself is untouched: the wrapper recomputes the weights exactly as
``levy_weighted_fuse`` does and then calls the original function.

Example
    python opencood/tools/levy_cos_decomp.py --use_cav [2] --save_levy \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy/ --note _cosdecomp
"""

import math

import torch

from opencood.models.fuse_modules import levy_pyramid_fuse as lpf
from opencood.tools import inference_heter_in_order
from opencood.tools.levy import levy_collect

EXTRA_KEYS = ("cd_balance", "cd_gap", "cd_norm_ego", "cd_norm_col", "cd_p_col")
_stash = []
_original_fuse = lpf.levy_weighted_fuse
_original_build = lpf.LevyPyramidFusion._build_trust_map


@torch.no_grad()
def _decompose(x, score, record_len, affine_matrix, align_corners, tau, gamma, logits):
    _, C, H, W = x.shape
    split_x = lpf.regroup(x, record_len)
    split_score = lpf.regroup(score, record_len)
    split_logits = lpf.regroup(logits, record_len) if logits is not None else None
    maps = {k: [] for k in EXTRA_KEYS}
    for b in range(affine_matrix.shape[0]):
        N = int(record_len[b])
        t = affine_matrix[b][:N, :N][0]
        feat = lpf.warp_affine_simple(split_x[b], t, (H, W), align_corners=align_corners)
        sc = lpf.warp_affine_simple(split_score[b], t, (H, W), align_corners=align_corners)
        valid = sc != 0
        s_map = (
            sc
            if split_logits is None
            else lpf.warp_affine_simple(split_logits[b], t, (H, W), align_corners=align_corners)
        )
        log_lambda = (math.log(tau) + gamma * s_map).masked_fill(~valid, lpf.LOG_ZERO)
        log_ev = torch.logsumexp(log_lambda, dim=0, keepdim=True)
        p = torch.exp(log_lambda - log_ev) * valid.to(log_lambda.dtype)  # (N,1,H,W)

        zero = torch.zeros((1, H, W), device=x.device, dtype=x.dtype)
        if N != 2:
            for k in EXTRA_KEYS:
                maps[k].append(zero)
            continue
        both = (valid[0] & valid[1]).to(x.dtype)  # (1,H,W)
        norm = feat.norm(dim=1, keepdim=True)  # (N,1,H,W)
        u = feat / norm.clamp(min=lpf.NORM_EPS)
        cos = (u[0] * u[1]).sum(dim=0, keepdim=True)
        maps["cd_balance"].append((1 - (p**2).sum(dim=0)) * both)
        maps["cd_gap"].append((1 - cos) * both)
        maps["cd_norm_ego"].append(norm[0] * both)
        maps["cd_norm_col"].append(norm[1] * both)
        maps["cd_p_col"].append(p[1] * both)
    return {k: torch.stack(v) for k, v in maps.items()}


def _fuse(x, score, record_len, affine_matrix, align_corners, tau=0.5, gamma=1.0, logits=None, **kwargs):
    _stash.append(
        _decompose(x, score, record_len, affine_matrix, align_corners, tau, gamma, logits)
    )
    return _original_fuse(
        x, score, record_len, affine_matrix, align_corners, tau=tau, gamma=gamma, logits=logits, **kwargs
    )


def _build(self, per_scale):
    out = _original_build(self, per_scale)
    for level, extra in enumerate(_stash):
        out["per_scale"][level].update(extra)
    _stash.clear()
    return out


lpf.levy_weighted_fuse = _fuse
lpf.LevyPyramidFusion._build_trust_map = _build
levy_collect.LEVEL_KEYS = tuple(levy_collect.LEVEL_KEYS) + EXTRA_KEYS

if __name__ == "__main__":
    inference_heter_in_order.main()
