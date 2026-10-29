"""
Box-level study of the Levy signals (evidence Lambda, disagreement tr Sigma) of the
windowed Levy fusion against the read-outs of HEAL's per-cell softmax fusion.

Input: the ``levy_<info>_boxes.csv`` / ``_gt.csv`` tables that ``inference.py
--save_levy`` writes (one row per predicted / ground-truth box, signals sampled
at the box centre on each pyramid level).

Every signal is oriented so that larger = less trusted, as in the paper
(evidence enters as -log Lambda). Reported per model:

* regime: Lambda / n_keys / phi(Lambda) ranges per level;
* FP ranking: AUROC of the signal for "box is a false positive at IoU t", over
  all boxes and within detection-score deciles (the part of the signal the score
  does not already carry), and split by how many agents cover the box;
* localisation error: Spearman between the signal and 1 - IoU over true
  positives at IoU 0.5, overall and within score deciles;
* misses: AUROC of the signal at ground-truth centres for "this object was missed".

    python opencood/tools/levy/levy_window_signals.py \
        --run window=<boxes.csv> --run heal=<boxes.csv> [--out report.txt]
"""

import argparse
import math

import numpy as np
import pandas as pd


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auroc(signal, label):
    signal, label = np.asarray(signal, float), np.asarray(label, bool)
    keep = np.isfinite(signal)
    signal, label = signal[keep], label[keep]
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(signal)
    return (r[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[keep]), rankdata(b[keep])
    return float(np.corrcoef(ra, rb)[0, 1])


def within_deciles(df, fn, n=10):
    """Mean of fn(sub-frame) over detection-score deciles, weighted by size."""
    q = pd.qcut(df["score"].rank(method="first"), n, labels=False)
    vals, weights = [], []
    for d in range(n):
        sub = df[q == d]
        v = fn(sub)
        if np.isfinite(v):
            vals.append(v)
            weights.append(len(sub))
    return float(np.average(vals, weights=weights)) if vals else float("nan")


def phi(lam):
    """E[1{Z>=1}/Z], Z ~ Poisson(lam) (paper eq. 14), by series."""
    lam = float(lam)
    if lam <= 0:
        return 0.0
    if lam > 60:
        return 1.0 / lam + 1.0 / lam ** 2
    total, term = 0.0, math.exp(-lam)
    for z in range(1, 400):
        term *= lam / z
        total += term / z
    return total


def signals(df):
    out = {"score (control)": -df["score"]}
    for lv in range(3):
        if f"L{lv}_log_evidence" not in df:
            continue
        out[f"L{lv} -log Lambda"] = -df[f"L{lv}_log_evidence"]
        out[f"L{lv} trSigma raw"] = df[f"L{lv}_disagreement"]
        out[f"L{lv} trSigma cos"] = df[f"L{lv}_disagreement_cosine"]
        out[f"L{lv} sigma"] = df[f"L{lv}_sigma"]
    out["dist (control)"] = df["dist"]
    return out


def report(name, boxes, gt, lines):
    p = lambda s="": lines.append(s)
    p(f"=== {name}: {len(boxes)} boxes in {boxes['frame'].nunique()} frames; "
      f"FP rate @0.7 {1 - boxes['tp70'].mean():.3f}, @0.5 {1 - boxes['tp50'].mean():.3f}")

    p("-- regime (box centres)")
    for lv in range(3):
        col = f"L{lv}_evidence"
        if col not in boxes:
            continue
        lam = boxes[col].replace(0, np.nan).dropna()
        keys = boxes.get(f"L{lv}_n_keys")
        p(f"  L{lv}: Lambda p5/p50/p95 {lam.quantile(.05):8.2f} {lam.median():8.2f} "
          f"{lam.quantile(.95):8.2f} | phi(p50) {phi(lam.median()):.3f} vs 1/Lambda "
          f"{1 / lam.median():.3f}"
          + (f" | keys p50 {keys.median():.0f}" if keys is not None else "")
          + f" | agents p50 {boxes[f'L{lv}_n_valid'].median():.0f}")

    sig = signals(boxes)
    for t in (0.7, 0.5):
        fp = boxes[f"tp{int(t * 100)}"] == 0
        p(f"-- FP@{t} AUROC (larger signal = FP; 0.5 = no information)")
        p(f"  {'signal':22s} {'all':>6s} {'|score':>7s} {'1 agent':>8s} {'>=2 ag.':>8s}")
        one = boxes["L0_n_valid"] <= 1
        for k, v in sig.items():
            a_all = auroc(v, fp)
            a_dec = within_deciles(boxes.assign(_s=v, _y=fp), lambda s: auroc(s["_s"], s["_y"])) \
                if not k.startswith("score") else float("nan")
            a1 = auroc(v[one], fp[one])
            a2 = auroc(v[~one], fp[~one])
            p(f"  {k:22s} {a_all:6.3f} {a_dec:7.3f} {a1:8.3f} {a2:8.3f}")

    tp = boxes[boxes["tp50"] == 1]
    err = 1 - tp["best_iou"]
    p(f"-- localisation error of TPs@0.5 (n={len(tp)}): Spearman(signal, 1-IoU)")
    p(f"  {'signal':22s} {'all':>6s} {'|score':>7s}")
    for k, v in signals(tp).items():
        r_all = spearman(v, err)
        r_dec = within_deciles(tp.assign(_s=v, _e=err), lambda s: spearman(s["_s"], s["_e"])) \
            if not k.startswith("score") else float("nan")
        p(f"  {k:22s} {r_all:6.3f} {r_dec:7.3f}")

    if gt is not None and len(gt):
        miss = gt["hit70"] == 0
        p(f"-- missed objects (GT centres, n={len(gt)}, missed@0.7 {miss.mean():.3f}): AUROC")
        for lv in range(3):
            if f"L{lv}_log_evidence" not in gt:
                continue
            p(f"  L{lv} -log Lambda {auroc(-gt[f'L{lv}_log_evidence'], miss):.3f}   "
              f"trSigma raw {auroc(gt[f'L{lv}_disagreement'], miss):.3f}   "
              f"cos {auroc(gt[f'L{lv}_disagreement_cosine'], miss):.3f}")
        p(f"  dist (control) {auroc(gt['dist'], miss):.3f}")
    p()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="name=path/to/levy_<info>_boxes.csv (the _gt.csv next to it is used too)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        boxes = pd.read_csv(path)
        gt_path = path.replace("_boxes.csv", "_gt.csv")
        try:
            gt = pd.read_csv(gt_path)
        except FileNotFoundError:
            gt = None
        report(name, boxes, gt, lines)
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
