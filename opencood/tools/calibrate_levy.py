# -*- coding: utf-8 -*-
# Author: added on top of HEAL's inference tooling.
# License: TDG-Attribution-NonCommercial-NoDistrib
"""
Calibrate the Levy spatial trust map to physical error scales.

Blueprint Sec. 5.3: the raw ``sigma_hat`` map that ``LevyPyramidFusion`` emits is
a *feature-space* deviation. This tool fits the global affine map

    s(x, y) = a * sigma_hat(x, y) + b

that turns it into a physical translation-error scale (metres), by a coarse grid
search over ``(a, b)`` on the validation split, minimising the Continuous Ranked
Probability Score (CRPS) of a zero-mean Gaussian ``N(0, s)`` against the observed
centre residuals of matched predictions.

The fitted ``(a, b)`` are printed, written to ``<model_dir>/levy_calibration.yaml``
as a standalone snippet, and -- with ``--write_yaml`` -- spliced into
``levy.calibration`` of the model config in place.

Usage
-----
    python opencood/tools/calibrate_levy.py \
        --model_dir opencood/logs/HEAL_m1_based/stage1/m1_base_levy \
        --range 102.4,102.4 --write_yaml

    # quick look on a subset
    python opencood/tools/calibrate_levy.py --model_dir <dir> --max_frames 300
"""

import argparse
import importlib
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils, train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.common_utils import update_dict


# --------------------------------------------------------------------------- #
# CRPS for a zero-mean Gaussian predictive distribution
# --------------------------------------------------------------------------- #
_SQRT2 = np.sqrt(2.0)
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_cdf(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / _SQRT2))


def _norm_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def crps_gaussian_zero_mean(sigma, y):
    """
    Mean CRPS of ``N(0, sigma^2)`` evaluated at observations ``y``.

    Closed form (Gneiting & Raftery 2007): for ``omega = y / sigma``
    ``CRPS = sigma * [ omega (2 Phi(omega) - 1) + 2 phi(omega) - 1/sqrt(pi) ]``.
    """
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    y = np.asarray(y, dtype=np.float64)
    omega = y / sigma
    crps = sigma * (omega * (2.0 * _norm_cdf(omega) - 1.0)
                    + 2.0 * _norm_pdf(omega) - _INV_SQRT_PI)
    return float(np.mean(crps))


# --------------------------------------------------------------------------- #
# prediction <-> ground-truth matching
# --------------------------------------------------------------------------- #
def match_by_center(pred_boxes, gt_boxes, max_dist):
    """
    Greedy nearest-centre matching in the BEV plane.

    Parameters
    ----------
    pred_boxes : np.ndarray
        ``(N, 8, 3)``.
    gt_boxes : np.ndarray
        ``(M, 8, 3)``.
    max_dist : float
        Gate in metres; a pred/gt pair further apart is never matched.

    Returns
    -------
    list[tuple[int, int, float]]
        ``(pred_idx, gt_idx, centre_distance)`` for every accepted match.
    """
    if pred_boxes is None or gt_boxes is None \
            or len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    pred_c = pred_boxes[:, :, :2].mean(axis=1)  # (N, 2)
    gt_c = gt_boxes[:, :, :2].mean(axis=1)      # (M, 2)
    dist = np.linalg.norm(pred_c[:, None, :] - gt_c[None, :, :], axis=2)  # (N, M)

    order = np.argsort(dist, axis=None)
    used_pred, used_gt, pairs = set(), set(), []
    for flat in order:
        pi, gi = np.unravel_index(flat, dist.shape)
        d = dist[pi, gi]
        if d > max_dist:
            break
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pairs.append((int(pi), int(gi), float(d)))
    return pairs


# --------------------------------------------------------------------------- #
# yaml splicing (no ruamel available -> targeted text edit)
# --------------------------------------------------------------------------- #
def write_calibration_to_yaml(yaml_path, a, b):
    """
    Splice ``a`` / ``b`` into the ``levy.calibration`` block of ``yaml_path``.

    Replaces an existing ``calibration:`` sub-block (nested or inline) under the
    first ``levy:`` mapping, or inserts one if none is present. Keeps every other
    line -- including anchors and comments -- untouched. A ``.bak`` copy is made
    first.
    """
    with open(yaml_path) as f:
        lines = f.read().split('\n')

    levy_idx = None
    for i, line in enumerate(lines):
        if line.strip() == 'levy:' and line.rstrip().endswith('levy:'):
            levy_idx = i
            break
    if levy_idx is None:
        raise RuntimeError(f"no 'levy:' mapping found in {yaml_path}")

    levy_indent = len(lines[levy_idx]) - len(lines[levy_idx].lstrip())
    child_indent = levy_indent + 2

    # Walk the levy block, looking for an existing 'calibration:' key.
    j = levy_idx + 1
    cal_idx = None
    while j < len(lines):
        raw = lines[j]
        if raw.strip() == '' or raw.strip().startswith('#'):
            j += 1
            continue
        indent = len(raw) - len(raw.lstrip())
        if indent <= levy_indent:
            break  # left the levy block
        if indent == child_indent and raw.strip().split(':')[0] == 'calibration':
            cal_idx = j
            break
        j += 1

    new_block = [
        ' ' * child_indent + 'calibration:',
        ' ' * (child_indent + 2) + f'a: {a:.6g}',
        ' ' * (child_indent + 2) + f'b: {b:.6g}',
    ]

    if cal_idx is None:
        lines[levy_idx + 1:levy_idx + 1] = new_block
    else:
        # Drop the old calibration line and any deeper-indented children.
        k = cal_idx + 1
        while k < len(lines):
            raw = lines[k]
            if raw.strip() == '':
                k += 1
                continue
            if len(raw) - len(raw.lstrip()) <= child_indent:
                break
            k += 1
        lines[cal_idx:k] = new_block

    shutil.copyfile(yaml_path, yaml_path + '.bak')
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(lines))


# --------------------------------------------------------------------------- #
def _parse_grid(spec, default):
    """``"min,max,steps"`` -> ``np.linspace``; empty -> ``default``."""
    if not spec:
        return default
    lo, hi, n = spec.split(',')
    return np.linspace(float(lo), float(hi), int(n))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_dir', required=True,
                   help='checkpoint dir (holds config.yaml + net_epoch_*.pth)')
    p.add_argument('--range', type=str, default="102.4,102.4",
                   help='BEV half-extent, matches inference.py --range')
    p.add_argument('--split', choices=['validate', 'test'], default='validate',
                   help="which split to fit on (blueprint: the val split)")
    p.add_argument('--match_dist', type=float, default=2.0,
                   help='centre-distance gate (m) for pred<->gt matching')
    p.add_argument('--reduce', choices=['mean', 'max', 'center'], default='mean',
                   help='how to read sigma under a box (see levy_sigma_under_boxes)')
    p.add_argument('--channel', default='sigma',
                   help="trust-map channel to calibrate ('sigma' = raw sigma_hat)")
    p.add_argument('--score_thresh', type=float, default=0.0,
                   help='ignore predictions below this confidence')
    p.add_argument('--max_frames', type=int, default=None,
                   help='cap the number of frames (quick runs)')
    p.add_argument('--a_grid', type=str, default="",
                   help='"min,max,steps" for a (default 0,30,151)')
    p.add_argument('--b_grid', type=str, default="",
                   help='"min,max,steps" for b (default 0,3,121)')
    p.add_argument('--write_yaml', action='store_true',
                   help='splice the fit into levy.calibration of the config '
                        '(a .bak copy is kept)')
    p.add_argument('--yaml_path', type=str, default=None,
                   help='config to edit with --write_yaml '
                        '(default <model_dir>/config.yaml)')
    return p


def load_hypes_with_range(opt):
    """Reproduce inference.py's range override so the grid matches training-time."""
    hypes = yaml_utils.load_yaml(None, opt)
    if 'heter' in hypes:
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        new_cav_range = [x_min, y_min,
                         hypes['postprocess']['anchor_args']['cav_lidar_range'][2],
                         x_max, y_max,
                         hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        })
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
        hypes = parser_func(hypes)
    return hypes


def main():
    opt = build_parser().parse_args()

    hypes = load_hypes_with_range(opt)
    if opt.split == 'test':
        hypes['validate_dir'] = hypes['test_dir']
    pc_range = hypes['postprocess']['gt_range']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch.")
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    print('Dataset Building')
    dataset = build_dataset(hypes, visualize=False, train=False)
    if opt.max_frames is not None:
        dataset = Subset(dataset, range(min(opt.max_frames, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=(dataset.dataset if isinstance(dataset, Subset)
                                    else dataset).collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    sigmas, res_x, res_y = [], [], []
    n_frames = n_pred = n_matched = n_no_levy = 0

    for i, batch_data in enumerate(loader):
        if batch_data is None:
            continue
        n_frames += 1
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            infer = inference_utils.inference_intermediate_fusion(
                batch_data, model, dataset.dataset if isinstance(dataset, Subset)
                else dataset)

        levy_unc = infer.get('levy_uncertainty', None)
        if levy_unc is None:
            n_no_levy += 1
            continue
        pred_box = infer['pred_box_tensor']
        gt_box = infer['gt_box_tensor']
        pred_score = infer['pred_score']
        if pred_box is None or gt_box is None or pred_box.shape[0] == 0:
            continue

        if opt.score_thresh > 0 and pred_score is not None:
            keep = pred_score >= opt.score_thresh
            pred_box = pred_box[keep]
            if pred_box.shape[0] == 0:
                continue

        n_pred += int(pred_box.shape[0])
        pred_np = pred_box.detach().cpu().numpy()
        gt_np = gt_box.detach().cpu().numpy()

        box_sigma = inference_utils.levy_sigma_under_boxes(
            pred_box, levy_unc, pc_range,
            channel=opt.channel, reduce=opt.reduce)

        pred_c = pred_np[:, :, :2].mean(axis=1)
        gt_c = gt_np[:, :, :2].mean(axis=1)
        for pi, gi, _d in match_by_center(pred_np, gt_np, opt.match_dist):
            sigmas.append(box_sigma[pi])
            res_x.append(pred_c[pi, 0] - gt_c[gi, 0])
            res_y.append(pred_c[pi, 1] - gt_c[gi, 1])
            n_matched += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1} frames, {n_matched} matched pairs so far")
        torch.cuda.empty_cache()

    if n_no_levy and n_no_levy == n_frames:
        raise RuntimeError(
            "the model emitted no 'levy_uncertainty' on any frame -- the fusion "
            "backbone is not LevyPyramidFusion, or levy.compute_uncertainty is off.")
    if n_matched < 20:
        raise RuntimeError(f"only {n_matched} matched pairs; cannot calibrate. "
                           "Loosen --match_dist or drop --score_thresh.")

    sigma_arr = np.asarray(sigmas, dtype=np.float64)
    # Pool the two signed axis residuals: each is modelled as N(0, s_i).
    sigma_stack = np.concatenate([sigma_arr, sigma_arr])
    residuals = np.concatenate([np.asarray(res_x), np.asarray(res_y)])
    abs_dist = np.sqrt(np.asarray(res_x) ** 2 + np.asarray(res_y) ** 2)

    a_grid = _parse_grid(opt.a_grid, np.linspace(0.0, 30.0, 151))
    b_grid = _parse_grid(opt.b_grid, np.linspace(0.0, 3.0, 121))

    best = {'crps': np.inf, 'a': 1.0, 'b': 0.0}
    for a in a_grid:
        s_col = a * sigma_stack
        for b in b_grid:
            crps = crps_gaussian_zero_mean(s_col + b, residuals)
            if crps < best['crps']:
                best = {'crps': crps, 'a': float(a), 'b': float(b)}

    # Reference points for context.
    identity_crps = crps_gaussian_zero_mean(sigma_stack, residuals)
    const_scale = float(np.sqrt(np.mean(residuals ** 2)))  # RMS per-axis error
    const_crps = crps_gaussian_zero_mean(
        np.full_like(residuals, const_scale), residuals)
    pear = float(np.corrcoef(sigma_arr, abs_dist)[0, 1])
    try:
        from scipy.stats import spearmanr
        spear = float(spearmanr(sigma_arr, abs_dist).correlation)
    except Exception:
        spear = float('nan')

    print("\n================ Levy calibration ================")
    print(f"frames processed        : {n_frames}")
    print(f"predictions kept        : {n_pred}")
    print(f"matched pairs (< {opt.match_dist:g} m): {n_matched}")
    print(f"sigma_hat  mean/p50/p90 : {sigma_arr.mean():.4f} / "
          f"{np.percentile(sigma_arr, 50):.4f} / {np.percentile(sigma_arr, 90):.4f}")
    print(f"|Δcentre|  mean/p50/p90 : {abs_dist.mean():.4f} / "
          f"{np.percentile(abs_dist, 50):.4f} / {np.percentile(abs_dist, 90):.4f} m")
    print(f"corr(sigma, |Δ|)       : Pearson {pear:.3f}, Spearman {spear:.3f}")
    print("-------------------------------------------------")
    print(f"CRPS  identity (a=1,b=0): {identity_crps:.4f} m")
    print(f"CRPS  const  s={const_scale:.3f} m : {const_crps:.4f} m")
    print(f"CRPS  best affine       : {best['crps']:.4f} m")
    print(f">>> levy.calibration:  a = {best['a']:.6g}   b = {best['b']:.6g}")
    print("=================================================\n")

    snippet_path = os.path.join(opt.model_dir, 'levy_calibration.yaml')
    with open(snippet_path, 'w') as f:
        f.write("# Fitted by opencood/tools/calibrate_levy.py\n"
                "# Paste under model.args.fusion_backbone.levy in your config.\n"
                "calibration:\n"
                f"  a: {best['a']:.6g}\n"
                f"  b: {best['b']:.6g}\n"
                f"# n_pairs={n_matched}  CRPS={best['crps']:.4f}m  "
                f"(identity {identity_crps:.4f}m)\n")
    print(f"snippet written to {snippet_path}")

    if opt.write_yaml:
        yaml_path = opt.yaml_path or os.path.join(opt.model_dir, 'config.yaml')
        write_calibration_to_yaml(yaml_path, best['a'], best['b'])
        print(f"spliced levy.calibration into {yaml_path} "
              f"(backup at {yaml_path}.bak)")


if __name__ == '__main__':
    main()
