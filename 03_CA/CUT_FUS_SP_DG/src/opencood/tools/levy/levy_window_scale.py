"""
The Levy uncertainty scale used as in the paper (sec. 5.5): a validation-calibrated
Gaussian on the box-centre error, scored by CRPS, interval coverage and a
split-conformal wrapper, plus the self-monitoring check (fig. 3).

Query = one true positive (IoU >= 0.5 with its best ground-truth box); its scalar
residuals are the centre errors along x and along y (two queries per box). A scale
signal u gives the predictive law N(0, s^2) with s = a*u + b (or a linear mix of
several signals) fitted once on VALIDATION by minimising mean CRPS, then applied
unchanged to TEST.

    python opencood/tools/levy/levy_window_scale.py \
        --model window <val_prefix> <test_prefix> --model heal <val_prefix> <test_prefix> \
        [--monitor name=<prefix> ...] [--out report.txt]

Prefixes are the paths up to ``_boxes.csv`` written by inference.py --save_levy.
"""

import argparse
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from opencood.utils import common_utils

LEVELS = (0, 1, 2)


def centre_errors(prefix):
    """TP@0.5 boxes with their centre residual (dx, dy) to the best-IoU GT box."""
    boxes = pd.read_csv(prefix + "_boxes.csv")
    geo = np.load(prefix + "_geometry.npz")
    rows = []
    for f, sub in boxes[boxes["tp50"] == 1].groupby("frame"):
        pred, gt = geo[f"pred_{f}"], geo[f"gt_{f}"]
        if not len(gt):
            continue
        gp = list(common_utils.convert_format(gt))
        gc = gt.mean(axis=1)[:, :2]
        for _, r in sub.iterrows():
            k = int(r["box_id"])
            iou = common_utils.compute_iou(common_utils.convert_format(pred[k:k + 1])[0], gp)
            j = int(np.argmax(iou))
            d = pred[k].mean(axis=0)[:2] - gc[j]
            row = r.to_dict()
            row.update(dx=float(d[0]), dy=float(d[1]))
            rows.append(row)
    df = pd.DataFrame(rows)
    for lv in LEVELS:
        if f"L{lv}_evidence" in df:
            df[f"inv_lam{lv}"] = 1.0 / df[f"L{lv}_evidence"].clip(lower=1e-6)
            phi = df[f"L{lv}_sigma"] ** 2 / df[f"L{lv}_disagreement"].clip(lower=1e-12)
            df[f"sigma_cos{lv}"] = np.sqrt(df[f"L{lv}_disagreement_cosine"] * phi.fillna(0))
    df["one_minus_score"] = 1 - df["score"]
    return df


def stack(df, cols):
    """Two queries per box (x and y residual), with the scale features repeated."""
    r = np.concatenate([df["dx"].values, df["dy"].values])
    x = np.concatenate([df[cols].values, df[cols].values]) if cols else np.zeros((2 * len(df), 0))
    return r, x


def crps(r, s):
    z = r / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def fit(r, x):
    """s = b + x @ a, a >= 0, minimising mean CRPS (features standardised)."""
    mu, sd = x.mean(0), x.std(0) + 1e-12
    xs = (x - mu) / sd

    def scale(p):
        return np.maximum(p[0] + xs @ np.abs(p[1:]), 1e-3)

    p0 = np.r_[np.sqrt(np.mean(r ** 2)), np.zeros(x.shape[1])]
    res = minimize(lambda p: crps(r, scale(p)).mean(), p0, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-7})
    return lambda xn: np.maximum(res.x[0] + ((xn - mu) / sd) @ np.abs(res.x[1:]), 1e-3)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(name, val, test, lines):
    p = lambda s="": lines.append(s)
    sigs = {"constant": []}
    sigs["dist"] = ["dist"]
    sigs["1-score"] = ["one_minus_score"]
    sigs["dist + 1-score"] = ["dist", "one_minus_score"]
    for lv in LEVELS:
        if f"L{lv}_sigma" in val:
            sigs[f"sigma L{lv}"] = [f"L{lv}_sigma"]
    for lv in (1, 2):
        if f"L{lv}_sigma" in val:
            sigs[f"1/Lambda L{lv}"] = [f"inv_lam{lv}"]
            sigs[f"trSigma cos L{lv}"] = [f"L{lv}_disagreement_cosine"]
            sigs[f"sigma_cos L{lv}"] = [f"sigma_cos{lv}"]
    if "L2_sigma" in val:
        sigs["dist+score+sigma L1,L2"] = ["dist", "one_minus_score", "L1_sigma", "L2_sigma"]
        sigs["dist+score+sigma_cos L1,L2"] = ["dist", "one_minus_score", "sigma_cos1", "sigma_cos2"]

    rv, _ = stack(val, [])
    rt, _ = stack(test, [])
    p(f"=== {name}: val {len(val)} TP boxes ({len(rv)} residuals), test {len(test)} "
      f"({len(rt)}); test |residual| median {np.median(np.abs(rt)):.3f} m, "
      f"p90 {np.quantile(np.abs(rt), .9):.3f} m")
    p(f"  {'scale':28s} {'CRPS':>7s} {'Gauss 80/90/95 cov':>20s} {'w90':>6s} "
      f"{'conf90 cov':>10s} {'w90':>6s} {'rho|r|':>7s}")
    for k, cols in sigs.items():
        _, xv = stack(val, cols)
        _, xt = stack(test, cols)
        xv = np.nan_to_num(xv, nan=np.nanmedian(xv) if xv.size else 0)
        xt = np.nan_to_num(xt, nan=np.nanmedian(xv) if xv.size else 0)
        sfun = fit(rv, xv)
        sv, st = sfun(xv), sfun(xt)
        c = crps(rt, st).mean()
        cov = [np.mean(np.abs(rt) <= norm.ppf(0.5 + q / 2) * st) for q in (.8, .9, .95)]
        w = np.median(2 * norm.ppf(.95) * st)
        q90 = np.quantile(np.abs(rv) / sv, min(1, math.ceil((len(rv) + 1) * .9) / len(rv)))
        ccov = np.mean(np.abs(rt) <= q90 * st)
        cw = np.median(2 * q90 * st)
        rho = spearman(st, np.abs(rt)) if cols else float("nan")
        p(f"  {k:28s} {c:7.4f} {cov[0]:6.3f} {cov[1]:6.3f} {cov[2]:6.3f} {w:6.3f} "
          f"{ccov:10.3f} {cw:6.3f} {rho:7.3f}")
    p()


def monitor(specs, lines):
    """Uncalibrated self-monitoring: mean scale vs true centre error per condition."""
    p = lambda s="": lines.append(s)
    p("=== self-monitoring (TP@0.5 boxes; ratios to the first condition of each model)")
    p(f"  {'condition':22s} {'n':>5s} {'|e| mean':>9s} {'ratio':>6s} "
      + " ".join(f"{c:>13s}" for c in ("sigma L1", "sigma L2", "1/Lambda L2", "trScos L2")))
    base = {}
    for spec in specs:
        name, prefix = spec.split("=", 1)
        model = name.split(":")[0]
        df = centre_errors(prefix)
        e = np.hypot(df["dx"], df["dy"]).mean()
        vals = [df["L1_sigma"].mean(), df["L2_sigma"].mean(), df["inv_lam2"].mean(),
                df["L2_disagreement_cosine"].mean()]
        if model not in base:
            base[model] = (e, vals)
        b_e, b_v = base[model]
        p(f"  {name:22s} {len(df):5d} {e:9.3f} {e / b_e:6.2f} "
          + " ".join(f"{v:7.4g} ({v / bv:4.2f})" for v, bv in zip(vals, b_v)))
    p()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", default=[],
                    metavar=("NAME", "VAL_PREFIX", "TEST_PREFIX"))
    ap.add_argument("--monitor", nargs="*", default=[], help="model:condition=prefix ...")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = []
    for name, vp, tp in args.model:
        evaluate(name, centre_errors(vp), centre_errors(tp), lines)
        print("\n".join(lines), flush=True)
    if args.monitor:
        monitor(args.monitor, lines)
        print("\n".join(lines[-12:]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
