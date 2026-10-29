"""
Two deployable per-box rules built from Levy signals, scored by average precision.

* cut-off   -- keep a box only when every named signal passes its threshold.
* re-score  -- keep every box and replace its confidence with P(true positive)
               from a logistic over the detection score and the signals.

Both act on one box at a time, so either can run inside inference. Thresholds and
weights come from the fit run (validation) and are applied unchanged to the
evaluation run (test).

    python opencood/tools/levy/levy_signal_rules.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy \\
        --fit_prefix levy_intermediate_204.8_102.4_val_epoch1_use_cav4 \\
        --prefix levy_intermediate_propfull_204.8_102.4_epoch1_use_cav4
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opencood.tools.levy.levy_threshold_ap import IOUS, average_precision, load

EVIDENCE = "fused_log_evidence"
DISAGREE = "fused_disagreement"
SETS = {
    "evidence (level mean)": [EVIDENCE],
    "raw disagreement (level mean)": [DISAGREE],
    "both (level mean)": [EVIDENCE, DISAGREE],
}
LABEL_IOU = 70


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--fit_prefix", required=True)
    p.add_argument("--drop", nargs="*", type=float, default=[1, 5, 10, 20])
    return p.parse_args()


def features(df, cols):
    out = []
    for c in cols:
        v = df[c].to_numpy(float)
        out.append(np.log(v + 1e-6) if c.endswith("_disagreement") else v)
    return np.stack(out, 1)


def cutoff_rows(name, cols, boxes, cache, fit, base, drops):
    rows = []
    for d in drops:
        thresholds = [float(np.percentile(fit[c], d)) for c in cols]
        keep = np.ones(len(boxes), bool)
        for c, t in zip(cols, thresholds):
            keep &= boxes[c].to_numpy() >= t
        ap, kept, total = average_precision_mask(boxes, cache, keep)
        tp = boxes[f"tp{LABEL_IOU}"].astype(bool).to_numpy()
        rows.append({
            "rule": name, "drop%": d, "kept": 100 * kept / total,
            "true kept": 100 * tp[keep].sum() / max(tp.sum(), 1),
            "false kept": 100 * (~tp)[keep].sum() / max((~tp).sum(), 1),
            **{f"AP{int(t * 100)}": ap[t] - base[t] for t in IOUS},
        })
    return rows


def average_precision_mask(boxes, cache, keep):
    """average_precision with an explicit per-row keep mask."""
    tmp = boxes.assign(_keep=keep.astype(float))
    return average_precision(tmp, cache, "_keep", 0.5)


def main():
    opt = parse_args()
    boxes, cache = load(opt.model_dir, opt.prefix)
    fit = pd.read_csv(os.path.join(opt.model_dir, f"{opt.fit_prefix}_boxes.csv"))
    base, _, total = average_precision(boxes, cache)
    print(f"evaluation run: {opt.prefix}")
    print(f"{total} boxes, unchanged:  " + "  ".join(f"AP{int(t * 100)} {base[t]:.4f}" for t in IOUS))
    print(f"thresholds / weights fitted on {opt.fit_prefix} ({len(fit)} boxes)")

    print("\n== cut-off: drop boxes whose signal is below the threshold")
    print(f"{'rule':<32}{'drop%':>6}{'kept':>7}{'true kept':>10}{'false kept':>11}   " +
          "  ".join(f"{'AP' + str(int(t * 100)):>10}" for t in IOUS))
    for name, cols in SETS.items():
        for r in cutoff_rows(name, cols, boxes, cache, fit, base, opt.drop):
            print(f"{r['rule']:<32}{r['drop%']:>6.0f}{r['kept']:>6.1f}%{r['true kept']:>9.1f}%"
                  f"{r['false kept']:>10.1f}%   " +
                  "  ".join(f"{r['AP' + str(int(t * 100))]:>+10.4f}" for t in IOUS))

    print("\n== re-score: keep every box, replace its confidence with P(true positive)")
    print(f"{'rule':<32}{'inputs':>8}   " + "  ".join(f"{'AP' + str(int(t * 100)):>10}" for t in IOUS))
    y_fit = fit[f"tp{LABEL_IOU}"].to_numpy()
    for name, cols in [("detection score only (control)", [])] + list(SETS.items()):
        X_fit = np.concatenate([fit[["score"]].to_numpy(), features(fit, cols)], 1) if cols \
            else fit[["score"]].to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X_fit, y_fit)
        X = np.concatenate([boxes[["score"]].to_numpy(), features(boxes, cols)], 1) if cols \
            else boxes[["score"]].to_numpy()
        rescored = boxes.assign(score=model.predict_proba(X)[:, 1])
        ap, _, _ = average_precision(rescored, cache)
        print(f"{name:<32}{1 + len(cols):>8}   " +
              "  ".join(f"{ap[t] - base[t]:>+10.4f}" for t in IOUS))


if __name__ == "__main__":
    main()
