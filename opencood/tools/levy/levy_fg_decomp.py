"""
Run ``inference_heter_in_order.py --save_levy`` while also recording each agent's
OWN foreground probability at every pyramid level.

The Levy trust map compares agents with each other (tr Sigma over the agent axis
of the feature maps). This wrapper records the inputs needed to move the same
evidence / disagreement idea onto the foreground estimators themselves:

    L{l}_fg_ego   S_ego = sigmoid(occ_l) of the ego, warped into the ego frame
    L{l}_fg_col   S_col of the collaborator (post camera-crop mask), warped too
    L{l}_fg_nval  how many agents observe the cell (1 or 2)

From those, an analysis can build
  * within-agent evidence:      lambda_j = tau * exp(gamma * S_j) per level (monotone in S_j)
  * within-agent disagreement:  spread of one agent's own S_j across levels
  * cross-agent on probabilities: p(1-p)(S_ego - S_col)^2, the scalar analogue of
    the feature-space tr Sigma

Unlike the cross-agent maps, the within-agent ones are defined wherever that agent
observes, so they also cover boxes outside the overlap region.

The fusion is untouched: the wrapper reads the same warped score maps that
``levy_weighted_fuse`` uses and then calls the original function.

Example
    python opencood/tools/levy_fg_decomp.py --use_cav [2] --save_levy \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy/ --note _fgdecomp
"""

import torch

from opencood.models.fuse_modules import levy_pyramid_fuse as lpf
from opencood.tools import inference_heter_in_order, levy_collect

EXTRA_KEYS = ("fg_ego", "fg_col", "fg_nval")
_stash = []
_original_fuse = lpf.levy_weighted_fuse
_original_build = lpf.LevyPyramidFusion._build_trust_map


@torch.no_grad()
def _foreground(x, score, record_len, affine_matrix, align_corners):
    _, _, H, W = x.shape
    split_score = lpf.regroup(score, record_len)
    maps = {k: [] for k in EXTRA_KEYS}
    for b in range(affine_matrix.shape[0]):
        N = int(record_len[b])
        t = affine_matrix[b][:N, :N][0]
        sc = lpf.warp_affine_simple(
            split_score[b], t, (H, W), align_corners=align_corners
        )  # (N, 1, H, W); exactly 0 = not observed
        valid = (sc != 0).to(sc.dtype)
        maps["fg_ego"].append(sc[0])
        maps["fg_col"].append(sc[1] if N > 1 else torch.zeros_like(sc[0]))
        maps["fg_nval"].append(valid.sum(dim=0))
    return {k: torch.stack(v) for k, v in maps.items()}


def _fuse(x, score, record_len, affine_matrix, align_corners, **kwargs):
    _stash.append(_foreground(x, score, record_len, affine_matrix, align_corners))
    return _original_fuse(x, score, record_len, affine_matrix, align_corners, **kwargs)


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
