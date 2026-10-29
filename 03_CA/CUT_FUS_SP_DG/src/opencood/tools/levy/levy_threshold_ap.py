"""
Average precision when a Levy signal is used as a fixed cut-off at inference.

A deployable rule keeps a predicted box only when its signal at the box centre
passes a threshold -- unlike a sparsification curve, which drops the weakest
fraction of a set of predictions you have to have in hand first.

The thresholds are quantiles of the signal on the FIT run (validation), so the
rule is chosen without looking at the evaluation run, and both runs are scored.

    python opencood/tools/levy/levy_threshold_ap.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy \\
        --fit_prefix levy_intermediate_204.8_102.4_val_epoch1_use_cav4 \\
        --prefix levy_intermediate_propfull_204.8_102.4_epoch1_use_cav4 \\
        --column L0_log_evidence
"""

import argparse
import os

import numpy as np
import pandas as pd

from opencood.utils import common_utils
from opencood.utils.eval_utils import calculate_ap

IOUS = (0.3, 0.5, 0.7)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", required=True, help="run the rule is evaluated on")
    p.add_argument("--fit_prefix", default=None, help="run the thresholds come from")
    p.add_argument("--column", default="L0_log_evidence")
    p.add_argument("--drop", nargs="*", type=float,
                   default=[0.5, 1, 2, 5, 10, 15, 20, 30],
                   help="percent of fit-run boxes the threshold would drop")
    return p.parse_args()


def load(model_dir, prefix):
    base = os.path.join(model_dir, prefix)
    boxes = pd.read_csv(f"{base}_boxes.csv")
    geometry = np.load(f"{base}_geometry.npz")
    # One IoU matrix per frame, computed once and reused for every threshold.
    cache = {}
    for frame, rows in boxes.groupby("frame"):
        gt = geometry[f"gt_{frame}"]
        pred = geometry[f"pred_{frame}"]
        if len(pred) == 0 or len(gt) == 0:
            cache[frame] = (np.zeros((len(pred), len(gt))), len(gt))
            continue
        gp = common_utils.convert_format(gt)
        iou = np.stack([common_utils.compute_iou(p, gp)
                        for p in common_utils.convert_format(pred)])
        cache[frame] = (iou, len(gt))
    return boxes, cache


def greedy(iou, order, thresh):
    """TP flags in the given (score-descending) order, one GT box per detection."""
    flags = np.zeros(iou.shape[0], dtype=int)
    free = np.ones(iou.shape[1], dtype=bool)
    for k in order:
        if not free.any():
            break
        row = np.where(free, iou[k], -1.0)
        j = int(np.argmax(row))
        if row[j] >= thresh:
            flags[k] = 1
            free[j] = False
    return flags


def average_precision(boxes, cache, keep_col=None, threshold=None):
    stat = {t: {"tp": [], "fp": [], "gt": 0, "score": []} for t in IOUS}
    kept = total = 0
    for frame, rows in boxes.groupby("frame"):
        iou, n_gt = cache[frame]
        keep = (np.ones(len(rows), bool) if keep_col is None
                else rows[keep_col].to_numpy() >= threshold)
        total += len(rows)
        kept += int(keep.sum())
        idx = rows["box_id"].to_numpy()[keep]
        scores = rows["score"].to_numpy()[keep]
        sub = iou[idx] if len(idx) and iou.size else np.zeros((len(idx), n_gt))
        order = np.argsort(-scores)
        for t in IOUS:
            stat[t]["gt"] += n_gt
            stat[t]["score"].extend(scores.tolist())
            flags = greedy(sub, order, t) if len(idx) and n_gt else np.zeros(len(idx), int)
            stat[t]["tp"].extend(flags.tolist())
            stat[t]["fp"].extend((1 - flags).tolist())
    return {t: calculate_ap(stat, t)[0] for t in IOUS}, kept, total


def report(name, boxes, cache, column, thresholds, base):
    print(f"\n{name}")
    print(f"{'threshold':>10} {'kept':>7} {'TP kept':>9} {'FP kept':>9}   " +
          "  ".join(f"{'AP' + str(int(t * 100)):>17}" for t in IOUS))
    for thr in thresholds:
        ap, kept, total = average_precision(boxes, cache, column, thr)
        keep = boxes[column].to_numpy() >= thr
        tp_kept = boxes.loc[keep, "tp70"].sum() / max(boxes["tp70"].sum(), 1)
        fp_kept = ((~boxes["tp70"].astype(bool)) & keep).sum() / max((~boxes["tp70"].astype(bool)).sum(), 1)
        print(f"{thr:>10.4f} {100 * kept / total:>6.1f}% {100 * tp_kept:>8.1f}% {100 * fp_kept:>8.1f}%   " +
              "  ".join(f"{ap[t]:>8.4f} ({ap[t] - base[t]:+.4f})" for t in IOUS))


def main():
    opt = parse_args()
    boxes, cache = load(opt.model_dir, opt.prefix)
    base, _, total = average_precision(boxes, cache)
    print(f"evaluation run: {opt.prefix}\n{total} boxes, no cut-off:  " +
          "  ".join(f"AP{int(t * 100)} {base[t]:.4f}" for t in IOUS))

    source = boxes
    if opt.fit_prefix:
        source = pd.read_csv(os.path.join(opt.model_dir, f"{opt.fit_prefix}_boxes.csv"))
        print(f"thresholds from {opt.fit_prefix} ({len(source)} boxes), "
              f"at the percentiles of {opt.column} that drop each share of ITS boxes")
    thresholds = [float(np.percentile(source[opt.column], d)) for d in opt.drop]
    report(f"cut-off on {opt.column} (TP/FP kept are at IoU 0.7)",
           boxes, cache, opt.column, thresholds, base)

    if opt.fit_prefix:
        fit_boxes, fit_cache = load(opt.model_dir, opt.fit_prefix)
        fit_base, _, _ = average_precision(fit_boxes, fit_cache)
        print(f"\nfit run for reference, no cut-off:  " +
              "  ".join(f"AP{int(t * 100)} {fit_base[t]:.4f}" for t in IOUS))
        report(f"same thresholds on the fit run", fit_boxes, fit_cache, opt.column,
               thresholds, fit_base)


if __name__ == "__main__":
    main()
