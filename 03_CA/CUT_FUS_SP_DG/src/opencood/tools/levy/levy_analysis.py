"""
Error-ranking analysis of Levy trust signals, from the CSVs written by
``inference_heter_in_order.py --save_levy``.

Box level (one row per predicted box):
    AUSE and Spearman rho of every signal against a per-box error, restricted to
    boxes whose centre lies in the overlap region (``--mask_col >= --min_valid``),
    overall and per ``use_cav`` / per mask value. Each signal is also scored
    *within* the quantile cells of ``--cond_cols`` (score and distance deciles by
    default, separately and jointly), which is the check that separates a signal
    carrying its own information from one restating what score or range already
    says -- the latter beats random unconditionally and collapses to chance here.
Frame level (one row per frame):
    triage -- flag the ``--flag_frac`` least-trusted frames by a frame signal and
    report their excess error over all frames.

Example
    python opencood/tools/levy_analysis.py \\
        --boxes  <model_dir>/levy_intermediate_..._use_cav4_boxes.csv \\
        --frames <model_dir>/levy_intermediate_..._use_cav4_frames.csv
"""

import argparse
import csv
import json
import math
import re

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Box columns ranked by default: every per-level and combined signal present,
# plus two uncertainty-free baselines derived below.
SIGNAL_PATTERN = re.compile(
    r"^(L\d+|fused)_(evidence|log_evidence|disagreement|sigma|disagreement_cosine"
    r"|sigma_cosine|disagreement_decoded)$"
)
BASELINES = ("neg_score", "dist")
# Higher evidence means *more* trust, so it is ranked by its negative.
TRUST_SIGNALS = ("evidence", "log_evidence")

FRAME_ERRORS = {
    "fp_rate50": lambda r: r["fp50"] / r["n_pred"] if r["n_pred"] > 0 else math.nan,
    "fp_rate70": lambda r: r["fp70"] / r["n_pred"] if r["n_pred"] > 0 else math.nan,
    "miss_rate50": lambda r: 1 - r["tp50"] / r["n_gt"] if r["n_gt"] > 0 else math.nan,
    "miss_rate70": lambda r: 1 - r["tp70"] / r["n_gt"] if r["n_gt"] > 0 else math.nan,
}


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--boxes", required=True, help="levy_*_boxes.csv")
    p.add_argument("--frames", default=None, help="levy_*_frames.csv (optional)")
    p.add_argument(
        "--error",
        default="iou",
        choices=["iou", "fp50", "fp70"],
        help="per-box error: 1 - best IoU with any GT box, or the FP indicator",
    )
    p.add_argument(
        "--mask_col",
        default="fused_n_valid",
        help="box column holding the agent count used to keep overlap boxes",
    )
    p.add_argument("--min_valid", type=int, default=2)
    p.add_argument(
        "--signals",
        nargs="*",
        default=None,
        help="box columns to rank (default: all Levy signals + baselines)",
    )
    p.add_argument("--steps", type=int, default=100, help="sparsification steps")
    p.add_argument(
        "--cond_cols",
        nargs="*",
        default=["score", "dist"],
        help="condition the ranking on each of these box columns and on their "
        "joint quantile grid; pass nothing to skip the conditional columns",
    )
    p.add_argument(
        "--cond_bins",
        type=int,
        default=10,
        help="quantile bins per conditioning column (10 = deciles)",
    )
    p.add_argument(
        "--cond_min",
        type=int,
        default=20,
        help="skip joint-grid cells holding fewer boxes than this",
    )
    p.add_argument(
        "--frame_signals",
        nargs="*",
        default=None,
        help="frame columns used as trust scores (default: every *_disagreement"
        "*_mean / *_sigma*_mean column)",
    )
    p.add_argument("--flag_frac", type=float, default=0.1)
    p.add_argument("--out", default=None, help="also dump all results as JSON")
    return p.parse_args()


def load_csv(path):
    """Rows as dicts, numeric strings converted to float (empty -> nan)."""
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value) if value != "" else math.nan
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def rankdata(x):
    """Average ranks (ties share their mean rank), like scipy.stats.rankdata."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    start = 0
    for end in range(1, len(x) + 1):
        if end == len(x) or sorted_x[end] != sorted_x[start]:
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return math.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(values, labels):
    """P(signal ranks a positive above a negative), ties counted as half."""
    positive = labels > 0
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(values)  # 0-based average ranks
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def quantile_bins(values, n_bins):
    """Quantile-bin index of each value; non-finite values get -1."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full(len(values), -1, dtype=np.int64)
    # Ties collapse edges, so a heavily tied column yields fewer than n_bins.
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.searchsorted(edges, values, side="right").astype(np.int64)
    bins[~np.isfinite(values)] = -1
    return bins


def joint_bins(rows, cols, n_bins):
    """
    Cell index of the joint quantile grid over ``cols`` (e.g. the 10x10 score x
    distance deciles); -1 wherever any of the columns is missing.
    """
    codes = np.zeros(len(rows), dtype=np.int64)
    for col in cols:
        values = np.array([r.get(col, math.nan) for r in rows], dtype=float)
        bins = quantile_bins(values, n_bins)
        codes = np.where(
            (codes < 0) | (bins < 0), -1, codes * (n_bins + 1) + np.maximum(bins, 0)
        )
    return codes


def conditional_association(values, error, strata, min_size, binary):
    """
    Within-stratum association between signal and error, averaged over strata
    and weighted by stratum size.

    This is what separates a signal carrying its own information from one that
    merely restates the conditioning variables: a signal whose unconditional
    ranking comes from being a proxy for score or distance collapses to chance
    here, while a weaker but complementary signal holds its value.

    Strata smaller than ``min_size``, or with no variation in the signal or the
    error, are skipped -- an all-TP score decile says nothing about ranking.
    """
    rho_sum = auc_sum = weight = 0.0
    n_strata = 0
    for code in np.unique(strata):
        if code < 0:
            continue
        keep = strata == code
        if keep.sum() < min_size:
            continue
        v, e = values[keep], error[keep]
        if v.std() == 0 or e.std() == 0:
            continue
        w = float(keep.sum())
        rho_sum += spearman(v, e) * w
        if binary:
            auc_sum += auroc(v, e) * w
        weight += w
        n_strata += 1
    if weight == 0:
        return {"rho": math.nan, "auroc": math.nan, "n": 0, "n_strata": 0}
    return {
        "rho": rho_sum / weight,
        "auroc": auc_sum / weight if binary else math.nan,
        "n": int(weight),
        "n_strata": n_strata,
    }


def sparsification_curve(error, uncertainty, steps):
    """
    Mean error of the boxes kept after removing the most uncertain fraction f,
    for f = 0, 1/steps, ..., (steps-1)/steps.

    Ties in ``uncertainty`` are broken by averaging over their members: removal
    within a tied block is spread evenly, so a constant signal behaves exactly
    like random removal instead of inheriting the input order.
    """
    n = len(error)
    order = np.argsort(-uncertainty, kind="mergesort")
    err_sorted = error[order]
    unc_sorted = uncertainty[order]
    # replace each tied block by its mean error
    start = 0
    for end in range(1, n + 1):
        if end == n or unc_sorted[end] != unc_sorted[start]:
            err_sorted[start:end] = err_sorted[start:end].mean()
            start = end
    suffix = np.cumsum(err_sorted[::-1])[::-1]
    curve = []
    for k in range(steps):
        removed = int(math.floor(k / steps * n))
        curve.append(suffix[removed] / (n - removed))
    return np.array(curve)


def ause(error, uncertainty, steps):
    fractions = np.arange(steps) / steps
    curve = sparsification_curve(error.copy(), uncertainty, steps)
    oracle = sparsification_curve(error.copy(), error, steps)
    return {
        "ause": float(_trapezoid(curve - oracle, fractions)),
        "ause_random": float(_trapezoid(error.mean() - oracle, fractions)),
        "curve": curve.tolist(),
    }


def box_error(rows, kind):
    if kind == "iou":
        return np.array([1.0 - r["best_iou"] for r in rows])
    tag = kind[2:]
    return np.array([1.0 - r[f"tp{tag}"] for r in rows])


def signal_values(rows, name):
    if name == "neg_score":
        return np.array([-r["score"] for r in rows])
    values = np.array([r.get(name, math.nan) for r in rows], dtype=float)
    match = SIGNAL_PATTERN.match(name)
    if match and match.group(2) in TRUST_SIGNALS:
        values = -values
    return values


def rank_boxes(rows, signals, kind, steps, conditions=(), n_bins=10, min_size=20):
    """
    AUSE / Spearman of each signal against the per-box error, plus -- for every
    tuple of columns in ``conditions`` -- the same association taken within the
    cells of those columns' joint quantile grid.
    """
    error = box_error(rows, kind)
    binary = kind != "iou"
    strata = {cols: joint_bins(rows, cols, n_bins) for cols in conditions}
    results = {}
    for name in signals:
        values = signal_values(rows, name)
        keep = np.isfinite(values) & np.isfinite(error)
        if keep.sum() < 3:
            continue
        a = ause(error[keep], values[keep], steps)
        results[name] = {
            "n": int(keep.sum()),
            "spearman": spearman(values[keep], error[keep]),
            "ause": a["ause"],
            "ause_random": a["ause_random"],
            "auroc": auroc(values[keep], error[keep]) if binary else math.nan,
            "conditional": {
                "+".join(cols): conditional_association(
                    values[keep], error[keep], codes[keep], min_size, binary
                )
                for cols, codes in strata.items()
            },
        }
    return results


def triage_frames(rows, signals, flag_frac):
    results = {}
    for name in signals:
        for err_name, err_fn in FRAME_ERRORS.items():
            pairs = [(r.get(name, math.nan), err_fn(r)) for r in rows]
            pairs = [(s, e) for s, e in pairs if math.isfinite(s) and math.isfinite(e)]
            if len(pairs) < 10:
                continue
            score = np.array([p[0] for p in pairs])
            err = np.array([p[1] for p in pairs])
            n_flag = max(1, int(round(flag_frac * len(pairs))))
            flagged = err[np.argsort(-score, kind="mergesort")[:n_flag]]
            mean_all = err.mean()
            results.setdefault(name, {})[err_name] = {
                "n_frames": len(pairs),
                "n_flagged": n_flag,
                "mean_error_all": float(mean_all),
                "mean_error_flagged": float(flagged.mean()),
                "excess_pct": float(100 * (flagged.mean() / mean_all - 1))
                if mean_all > 0
                else math.nan,
                "spearman": spearman(score, err),
            }
    return results


def print_box_table(title, results, binary=False):
    print(f"\n{title}")
    if not results:
        print("  (not enough boxes)")
        return
    first = next(iter(results.values()))
    conditions = list(first["conditional"])
    stat = "AUC" if binary else "rho"
    header = f"  {'signal':<34}{'n':>7}{'rho':>8}{'AUSE':>9}{'AUSE_rnd':>10}"
    header += "".join(f"{stat + '|' + c:>16}" for c in conditions)
    print(header)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["ause"]):
        line = (
            f"  {name:<34}{r['n']:>7}{r['spearman']:>8.3f}"
            f"{r['ause']:>9.4f}{r['ause_random']:>10.4f}"
        )
        for cond in conditions:
            value = r["conditional"][cond]["auroc" if binary else "rho"]
            line += f"{value:>16.3f}" if math.isfinite(value) else f"{'--':>16}"
        print(line)


def main():
    opt = parser()
    boxes = load_csv(opt.boxes)
    if boxes and opt.mask_col not in boxes[0]:
        raise KeyError(f"{opt.mask_col} not in {opt.boxes}")

    signals = opt.signals
    if signals is None:
        signals = [c for c in boxes[0] if SIGNAL_PATTERN.match(c)] if boxes else []
        signals += list(BASELINES)

    missing = [c for c in opt.cond_cols if boxes and c not in boxes[0]]
    if missing:
        raise KeyError(f"conditioning columns {missing} not in {opt.boxes}")
    # Each column on its own, then -- with more than one -- their joint grid: a
    # signal can beat score alone by carrying distance and still add nothing
    # over the two together.
    conditions = [(c,) for c in opt.cond_cols]
    if len(opt.cond_cols) > 1:
        conditions.append(tuple(opt.cond_cols))
    binary = opt.error != "iou"

    def rank(group):
        return rank_boxes(
            group,
            signals,
            opt.error,
            opt.steps,
            conditions=tuple(conditions),
            n_bins=opt.cond_bins,
            min_size=opt.cond_min,
        )

    overlap = [r for r in boxes if r[opt.mask_col] >= opt.min_valid]
    print(
        f"{len(overlap)}/{len(boxes)} boxes with {opt.mask_col} >= {opt.min_valid}; "
        f"error = {'1 - best IoU' if opt.error == 'iou' else opt.error.upper()}"
    )
    print("AUSE: lower is better; AUSE_rnd is random removal (0 = oracle).")
    print("rho: Spearman(signal, error); evidence is negated (more evidence = trust).")
    if conditions:
        stat = "AUC" if binary else "rho"
        print(
            f"{stat}|cols: the same association within the cells of those columns' "
            f"{opt.cond_bins}-bin quantile grid, weighted by cell size "
            f"({'0.5' if binary else '0'} = no information the cell does not "
            "already hold)."
        )

    output = {"boxes": {}, "frames": {}}
    output["boxes"]["all"] = rank(overlap)
    print_box_table("All overlap boxes", output["boxes"]["all"], binary)

    for col, label in ((opt.mask_col, opt.mask_col), ("use_cav", "use_cav")):
        for value in sorted({r[col] for r in overlap}):
            group = [r for r in overlap if r[col] == value]
            key = f"{label}={int(value)}"
            output["boxes"][key] = rank(group)
            print_box_table(
                f"{key}  ({len(group)} boxes)", output["boxes"][key], binary
            )

    if opt.frames:
        frames = load_csv(opt.frames)
        frame_signals = opt.frame_signals
        if frame_signals is None:
            frame_signals = [
                c
                for c in (frames[0] if frames else {})
                if re.match(r"^(L\d+|fused)_(disagreement|sigma)\w*_mean$", c)
            ]
        for value in [None] + sorted({r["use_cav"] for r in frames}):
            group = [r for r in frames if value is None or r["use_cav"] == value]
            key = "all" if value is None else f"use_cav={int(value)}"
            res = triage_frames(group, frame_signals, opt.flag_frac)
            output["frames"][key] = res
            print(
                f"\nFrame triage, {key}: worst {opt.flag_frac:.0%} by signal "
                "(excess error vs all frames)"
            )
            if not res:
                print("  (not enough frames with overlap)")
                continue
            print(f"  {'signal':<34}{'error':<13}{'n':>6}{'excess%':>9}{'rho':>8}")
            for name, per_err in res.items():
                for err_name, r in per_err.items():
                    print(
                        f"  {name:<34}{err_name:<13}{r['n_frames']:>6}"
                        f"{r['excess_pct']:>9.1f}{r['spearman']:>8.3f}"
                    )

    if opt.out:
        with open(opt.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {opt.out}")


if __name__ == "__main__":
    main()
