"""
Precision gate: drop the least-trusted k% of boxes by a signal and report what
comes out with them, from the CSVs written by ``--save_levy``.

The question is whether a signal removes false positives *faster than score
does*. Dropping the lowest-score k% is exactly raising the detection threshold,
so `score` -- not random -- is the bar: a signal is only worth having if, at the
same number of boxes removed, it takes out more FPs and fewer TPs.

``enrichment`` is the FP rate among the removed boxes over the FP rate of the
pool. Removing boxes raises overall precision iff enrichment > 1; below that the
gate is discarding TPs faster than FPs and is worse than not gating at all.

Example
    python opencood/tools/levy/levy_gate.py \\
        --boxes <model_dir>/levy_..._boxes.csv \\
        --frames <model_dir>/levy_..._frames.csv
"""

import argparse

import numpy as np
import pandas as pd

DEFAULT_KS = (1, 2, 5, 10, 20)
DEFAULT_SIGNALS = ("L0_evidence", "L1_evidence", "L2_evidence", "mean_evidence")


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--boxes", required=True)
    p.add_argument("--frames", default=None, help="for the ground-truth count")
    p.add_argument("--signals", nargs="*", default=list(DEFAULT_SIGNALS))
    p.add_argument("--ks", nargs="*", type=float, default=list(DEFAULT_KS))
    p.add_argument("--iou", type=int, default=70, choices=[30, 50, 70])
    p.add_argument("--mask_col", default="fused_n_valid")
    p.add_argument("--min_valid", type=int, default=2)
    p.add_argument(
        "--stratify_col",
        default=None,
        help="drop k%% within each stratum of this column (e.g. fused_n_valid) "
        "instead of globally; evidence scales with the agent count, so a global "
        "gate otherwise ranks by observer count",
    )
    p.add_argument(
        "--all_boxes",
        action="store_true",
        help="gate every box instead of only the overlap region",
    )
    return p.parse_args()


def add_derived(b):
    """Signals that are combinations of CSV columns rather than columns."""
    levels = [c for c in b.columns if c.endswith("_evidence") and c.startswith("L")]
    if levels:
        # Every level's Lambda is in the same units (sum_j tau*exp(gamma*S_j)),
        # so a plain mean is meaningful without rescaling.
        b["mean_evidence"] = b[levels].mean(axis=1)
    # Lambda is a sum over contributing agents, so it scales with the observer
    # count. Dividing by it gives the mean per-agent intensity, which under
    # the sigmoid score lands in [tau, tau*e] for every stratum -- the same
    # support regardless of how many agents saw the cell, so one global
    # threshold is meaningful without stratifying. Within a stratum this is a
    # monotone transform of Lambda, so it changes nothing there; it only
    # changes how the removal budget is shared *between* strata.
    for col in [c for c in b.columns if c.endswith("_evidence")]:
        n_col = col.replace("_evidence", "_n_valid")
        if n_col in b.columns:
            b[f"{col}_pa"] = b[col] / b[n_col].clip(lower=1)
    b["random"] = np.random.default_rng(0).random(len(b))
    return b


def lowest_k(b, signal, k, stratify_col=None):
    """
    Positional index of the lowest-``signal`` k% of ``b``.

    With ``stratify_col`` the k% is taken *within each stratum* instead of
    globally. That matters for evidence: Lambda = sum_j tau*exp(gamma*S_j) is a
    sum over contributing agents, so its scale is set by how many agents see the
    cell. A global gate on raw Lambda therefore ranks by observer count first and
    confidence second, and mostly just deletes the thinnest-observed stratum --
    which looks like it works only because that stratum is FP-rich for reasons
    the signal is not measuring.
    """
    if stratify_col is None:
        return np.argsort(b[signal].values, kind="stable")[: round(k / 100 * len(b))]
    pos = np.arange(len(b))
    out = []
    for idx in b.groupby(b[stratify_col].values, sort=True).indices.values():
        local = idx[np.argsort(b[signal].values[idx], kind="stable")]
        out.append(local[: round(k / 100 * len(idx))])
    return np.concatenate(out) if out else pos[:0]


def gate(b, signal, k, tp_col, n_gt, stratify_col=None):
    """Drop the lowest-``signal`` k% of ``b``; report what left with them."""
    rm_idx = lowest_k(b, signal, k, stratify_col)
    n_rm = len(rm_idx)
    if n_rm == 0:
        return None
    mask = np.zeros(len(b), dtype=bool)
    mask[rm_idx] = True
    removed, kept = b[mask], b[~mask]

    tp_all, fp_all = b[tp_col].sum(), (1 - b[tp_col]).sum()
    fp_rm = int((1 - removed[tp_col]).sum())
    tp_rm = int(removed[tp_col].sum())
    base_rate = fp_all / len(b)
    return {
        "k%": k,
        "n_rm": n_rm,
        "FP_rm": fp_rm,
        "TP_rm": tp_rm,
        "%FP_gone": 100 * fp_rm / max(fp_all, 1),
        "%TP_lost": 100 * tp_rm / max(tp_all, 1),
        "enr": (fp_rm / n_rm) / base_rate if base_rate > 0 else np.nan,
        "prec": kept[tp_col].mean(),
        "d_prec": kept[tp_col].mean() - b[tp_col].mean(),
        "rec": kept[tp_col].sum() / n_gt if n_gt else np.nan,
        "d_rec": (kept[tp_col].sum() - tp_all) / n_gt if n_gt else np.nan,
    }


def main():
    opt = parser()
    b = add_derived(pd.read_csv(opt.boxes))
    if not opt.all_boxes:
        b = b[b[opt.mask_col] >= opt.min_valid].reset_index(drop=True)
    n_gt = int(pd.read_csv(opt.frames).n_gt.sum()) if opt.frames else 0

    tp_col = f"tp{opt.iou}"
    signals = [s for s in opt.signals if s in b.columns]
    missing = [s for s in opt.signals if s not in b.columns]
    # score and random frame the result: score is the bar, random is the floor.
    signals = (
        ["score"] + signals + (["fused_evidence"] if "fused_evidence" in b else [])
    )
    signals += ["random"]

    scope = "all boxes" if opt.all_boxes else f"{opt.mask_col} >= {opt.min_valid}"
    if opt.stratify_col:
        counts = dict(sorted(b[opt.stratify_col].value_counts().items()))
        scope += f"; k%% taken within each {opt.stratify_col} stratum {counts}"
    print(f"{opt.boxes}\n{len(b)} boxes ({scope}), IoU {opt.iou / 100:.2f}")
    print(
        f"base: precision {b[tp_col].mean():.4f}  FP rate {1 - b[tp_col].mean():.4f}  "
        f"TP {int(b[tp_col].sum())}  FP {int((1 - b[tp_col]).sum())}"
        + (f"  n_gt {n_gt}  recall {b[tp_col].sum() / n_gt:.4f}" if n_gt else "")
    )
    if missing:
        print(f"missing from CSV, skipped: {missing}")
    print(
        "enr > 1 = removed set is FP-enriched (precision rises); enr < 1 = gate hurts"
    )

    rows = []
    for signal in signals:
        for k in opt.ks:
            r = gate(b, signal, k, tp_col, n_gt, opt.stratify_col)
            if r:
                rows.append({"signal": signal, **r})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for signal in signals:
        sub = table[table.signal == signal]
        if sub.empty:
            continue
        print(f"\n{signal}")
        print(
            sub.drop(columns="signal").to_string(
                index=False,
                formatters={
                    "%FP_gone": "{:6.1f}".format,
                    "%TP_lost": "{:6.1f}".format,
                    "enr": "{:5.2f}".format,
                    "prec": "{:.4f}".format,
                    "d_prec": "{:+.4f}".format,
                    "rec": "{:.4f}".format,
                    "d_rec": "{:+.4f}".format,
                },
            )
        )


if __name__ == "__main__":
    main()
