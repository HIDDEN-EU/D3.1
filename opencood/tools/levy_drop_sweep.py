# -*- coding: utf-8 -*-
"""
Replay a Levy Sparsification Filter *drop* sweep offline.

Reads the per-box arrays that ``inference.py --levy_diag`` writes
(``levy_diag_boxes.npz``; schema in :mod:`opencood.tools.levy_metrics`) and, for
a list of thresholds on one channel, removes every box on the flagged side of
the cut and recomputes AP / false-positive rate / how many true and false
positives went with it. No GPU, no re-running inference -- dropping a box is
exact offline.

Measured verdict on this checkpoint: the drop filter does not pay at any
threshold, on any channel, at any noise level. Re-scoring dominates it on every
axis (see ``levy_fp_benchmark.py``), because discarding trades recall for
precision permanently and AP charges more for the lost recall than it pays for
the gained precision. Kept as the tool that establishes that.

Only 'drop' mode is modelled: 'attenuate' rescales scores and cannot be
reconstructed without redoing the score-ordered matching.

Usage
-----
    python opencood/tools/levy_drop_sweep.py \
        --npz opencood/logs/.../levy_diag_boxes.npz \
        --channel cos_shift.sigma --pct 1,5

``--pct 1,5`` sweeps thresholds at the 1st and 5th percentile *of the tail being
cut* -- the low tail for an inverted channel, the high tail otherwise, chosen by
``--direction auto``. Explicit values go in ``--thresh``; both lists are merged.
"""
import argparse
import numpy as np

from opencood.tools.levy_metrics import BoxDump, ap_from_labels, auroc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True,
                   help='levy_diag_boxes.npz from inference.py --levy_diag')
    p.add_argument('--channel', default='sigma',
                   help="which per-box array to threshold. Any key in the npz: "
                        "'sigma', 'evidence', 'disagreement', or a variant "
                        "channel such as 'var__cos_shift.sigma' / "
                        "'var__cos.disagreement' (the 'var__' prefix is optional).")
    p.add_argument('--direction', choices=['low', 'high', 'auto'], default='auto',
                   help="flag values below ('low') or above ('high') the "
                        "threshold. 'auto' (default) reads the direction off the "
                        "data: AUROC < 0.5 means the channel is inverted and the "
                        "false positives sit in its low tail, so 'low' is chosen; "
                        "otherwise 'high'. Thresholding a channel in the wrong "
                        "direction removes true positives by construction.")
    p.add_argument('--thresh', default='',
                   help='comma-separated explicit thresholds to sweep')
    p.add_argument('--pct', default='1,5',
                   help='comma-separated percentiles of the channel, taken on '
                        'the tail being cut, to add as thresholds (default 1,5)')
    p.add_argument('--ref_iou', type=float, default=0.5,
                   help='IoU whose TP/FP labels drive the FP-rate / recall '
                        'columns (default 0.5)')
    opt = p.parse_args()

    dump = BoxDump(opt.npz)
    key = opt.channel
    try:
        sigma = dump.channel(key)
    except KeyError as e:
        raise SystemExit(e.args[0])
    score = dump.score
    gt_total = dump.gt_total
    tp_by_iou = dump.tp_by_iou
    ref = dump.labels(opt.ref_iou)
    n = sigma.shape[0]
    n_fp_ref = int((~ref).sum())
    n_tp_ref = int(ref.sum())

    # Which tail holds the false positives, measured rather than assumed.
    auroc_fp = auroc(sigma, ~ref)
    direction = opt.direction
    if direction == 'auto':
        direction = 'low' if auroc_fp < 0.5 else 'high'
    print(f"channel {key}: AUROC(->FP) {auroc_fp:.4f} "
          f"({'inverted, FPs in the low tail' if auroc_fp < 0.5 else 'FPs in the high tail'})"
          f"  -> dropping the {direction} side")

    # sigma-like channels are structurally 0 for single-coverage cells; that
    # zero-spike is a discrete lump at the bottom of the distribution, so a
    # percentile landing inside it is meaningless -- fall back to values > 0.
    zero = sigma == 0.0
    pos = sigma[~zero]

    thr = []
    if opt.thresh.strip():
        thr += [float(x) for x in opt.thresh.split(',')]
    pcts = [float(x) for x in opt.pct.split(',')] if opt.pct.strip() else []
    for q in pcts:
        # a percentile is only meaningful on the tail being cut
        qq = q if direction == 'low' else 100.0 - q
        t = float(np.percentile(sigma, qq))
        if t == 0.0:
            t = float(np.percentile(pos, qq))
        thr.append(t)
    thr = sorted(set(round(t, 5) for t in thr))
    if zero.any() and direction == 'low':
        thr = [1e-6] + thr  # a row that flags exactly the zero-spike

    print(f"loaded {n} boxes  ({n_tp_ref} TP / {n_fp_ref} FP @ IoU {opt.ref_iou:g}, "
          f"gt_total={gt_total})")
    print(f"{key}==0 spike: {int(zero.sum())} boxes ({zero.mean():.1%}), "
          f"FP rate {float((~ref[zero]).mean()):.1%} vs {float((~ref[~zero]).mean()):.1%} "
          f"among sigma>0")
    print(f"all values percentiles: "
          + "  ".join(f"p{q}={np.percentile(sigma, q):.4f}"
                      for q in (1, 5, 10, 25, 50, 75, 90, 99)))
    print(f"values>0   percentiles: "
          + "  ".join(f"p{q}={np.percentile(pos, q):.4f}"
                      for q in (1, 5, 10, 25, 50, 75, 90, 99)))
    base = {iou: ap_from_labels(tp_by_iou[iou], score, gt_total) for iou in (0.3, 0.5, 0.7)}
    base_fpr = n_fp_ref / n
    print(f"\nbaseline (no drop):  AP {base[0.3]:.4f}/{base[0.5]:.4f}/{base[0.7]:.4f}"
          f"  FPrate@{opt.ref_iou:g} {base_fpr:.4f}\n")

    hdr = (f"{'thresh':>8} {'%drop':>7} {'FPrec':>7} {'TPloss':>7} "
           f"{'FPrate':>7} {'dFPrate':>8} | {'AP.30':>7} {'AP.50':>7} {'AP.70':>7} "
           f"| {'dAP.50':>8} {'dAP.70':>8}")
    print(hdr)
    print('-' * len(hdr))
    for t in thr:
        flag = sigma < t if direction == 'low' else sigma > t
        keep = ~flag
        n_drop = int(flag.sum())
        fp_removed = int((flag & ~ref).sum())
        tp_removed = int((flag & ref).sum())
        kept_ref = ref[keep]
        fpr = float((~kept_ref).sum() / max(keep.sum(), 1))
        aps = {iou: ap_from_labels(tp_by_iou[iou][keep], score[keep], gt_total)
               for iou in (0.3, 0.5, 0.7)}
        print(f"{t:8.4f} {n_drop / n:7.1%} "
              f"{fp_removed / max(n_fp_ref, 1):7.1%} "
              f"{tp_removed / max(n_tp_ref, 1):7.1%} "
              f"{fpr:7.4f} {fpr - base_fpr:+8.4f} | "
              f"{aps[0.3]:7.4f} {aps[0.5]:7.4f} {aps[0.7]:7.4f} | "
              f"{aps[0.5] - base[0.5]:+8.4f} {aps[0.7] - base[0.7]:+8.4f}")


if __name__ == '__main__':
    main()
