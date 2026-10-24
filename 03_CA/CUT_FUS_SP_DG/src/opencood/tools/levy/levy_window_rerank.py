"""
Held-out re-ranking test for the Levy signals.

For each model's box table (``inference.py --save_levy``), fit a logistic model
P(box is a TP at IoU t) on one group of scenarios and use it as the new box score
on the held-out scenarios (GroupKFold by scenario, so no scene is scored by a
model that saw it). Boxes are then re-matched to ground truth greedily in the
NEW score order (a different order can hand a GT box to a different prediction)
and AP is recomputed exactly as eval_utils does.

Feature sets, all on top of logit(score):
    S        score only (monotone -> reproduces the original AP: sanity check)
    S+D      + distance                         (the control to beat)
    S+D+E    + evidence  -log Lambda, 3 levels
    S+D+V    + disagreement: cos trSigma, log sigma, log1p raw trSigma, levels 1-2
    S+D+L12  + evidence and disagreement, levels 1-2
    S+D+all  + evidence and disagreement, all 3 levels

Paired frame bootstrap gives a 95% interval for each AP difference vs S+D.

    python opencood/tools/levy/levy_window_rerank.py \
        --run window=<prefix> --run heal=<prefix> [--folds 4] [--boot 1000]

where <prefix> is the path up to ``_boxes.csv`` (``_frames.csv`` and
``_geometry.npz`` must sit next to it).
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opencood.utils import common_utils
from opencood.utils.eval_utils import voc_ap

IOUS = (0.3, 0.5, 0.7)


def features(df, levels):
    cols = {}
    for lv in levels:
        cols[f"E{lv}"] = -df[f"L{lv}_log_evidence"]
    ev = list(cols)
    dis = {}
    for lv in levels:
        dis[f"C{lv}"] = df[f"L{lv}_disagreement_cosine"]
        dis[f"G{lv}"] = np.log(df[f"L{lv}_sigma"] + 1e-6)
        dis[f"R{lv}"] = np.log1p(df[f"L{lv}_disagreement"])
    return pd.DataFrame(cols), pd.DataFrame(dis)


def design(df):
    s = df["score"].clip(1e-4, 1 - 1e-4)
    base = pd.DataFrame({"logit": np.log(s / (1 - s))})
    dist = pd.DataFrame({"dist": df["dist"]})
    e_all, v_all = features(df, (0, 1, 2))
    e_12, v_12 = features(df, (1, 2))
    sets = {
        "S": base,
        "S+D": pd.concat([base, dist], axis=1),
        "S+D+E": pd.concat([base, dist, e_all], axis=1),
        "S+D+V": pd.concat([base, dist, v_12], axis=1),
        "S+D+L12": pd.concat([base, dist, e_12, v_12], axis=1),
        "S+D+all": pd.concat([base, dist, e_all, v_all], axis=1),
    }
    for k, x in sets.items():
        sets[k] = x.replace([np.inf, -np.inf], np.nan).fillna(x.median())
    return sets


def frame_ious(geometry, frames):
    """{frame: (iou [n_pred, n_gt], n_gt)} from the stored corner geometry."""
    out = {}
    for f in frames:
        pred = geometry[f"pred_{f}"]
        gt = geometry[f"gt_{f}"]
        if len(pred) and len(gt):
            dp = list(common_utils.convert_format(pred))
            gp = list(common_utils.convert_format(gt))
            iou = np.stack([common_utils.compute_iou(d, gp) for d in dp])
        else:
            iou = np.zeros((len(pred), len(gt)))
        out[f] = (iou, len(gt))
    return out


def match(iou, scores, thresh):
    """Greedy score-descending matching, as eval_utils.caluclate_tp_fp."""
    tp = np.zeros(len(scores), dtype=int)
    if iou.shape[1] == 0:
        return tp
    free = np.ones(iou.shape[1], dtype=bool)
    for k in np.argsort(-scores):
        if not free.any():
            break
        row = np.where(free, iou[k], -1.0)
        j = int(np.argmax(row))
        if row[j] >= thresh:
            tp[k] = 1
            free[j] = False
    return tp


def ap_from(per_frame, frames):
    """per_frame[f] = (scores, tp, n_gt); AP over the listed frames."""
    scores = np.concatenate([per_frame[f][0] for f in frames])
    tp = np.concatenate([per_frame[f][1] for f in frames])
    n_gt = sum(per_frame[f][2] for f in frames)
    order = np.argsort(-scores, kind="mergesort")
    tp = tp[order]
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = (ctp / n_gt).tolist()
    prec = (ctp / np.maximum(ctp + cfp, 1)).tolist()
    ap, _, _ = voc_ap(rec, prec)
    return ap


def run(name, prefix, n_folds, n_boot, seed, lines):
    p = lambda s="": lines.append(s)
    boxes = pd.read_csv(prefix + "_boxes.csv")
    frames_df = pd.read_csv(prefix + "_frames.csv")
    geometry = np.load(prefix + "_geometry.npz")
    frames = sorted(frames_df["frame"].unique())
    scen = frames_df.set_index("frame")["scenario"]
    boxes["scenario"] = boxes["frame"].map(scen)

    # every predicted box must have a row, or re-scoring would mix scales
    for f in frames:
        n_geo = len(geometry[f"pred_{f}"])
        n_row = int((boxes["frame"] == f).sum())
        assert n_geo == n_row, f"frame {f}: {n_geo} boxes in geometry, {n_row} rows"
    boxes = boxes.sort_values(["frame", "box_id"]).reset_index(drop=True)

    ious = frame_ious(geometry, frames)
    sets = design(boxes)
    groups = boxes["scenario"].values
    gkf = GroupKFold(n_splits=n_folds)

    p(f"=== {name}: {len(boxes)} boxes, {len(frames)} frames, "
      f"{boxes['scenario'].nunique()} scenarios, {n_folds} scenario folds")
    results = {}
    for t in IOUS:
        y = boxes[f"tp{int(t * 100)}"].values
        per_set = {}
        for key, x in sets.items():
            pred = np.zeros(len(boxes))
            for tr, te in gkf.split(x, y, groups):
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(C=1.0, max_iter=5000))
                clf.fit(x.iloc[tr], y[tr])
                pred[te] = clf.predict_proba(x.iloc[te])[:, 1]
            per_frame = {}
            for f, sub in boxes.groupby("frame"):
                iou, n_gt = ious[f]
                sc = pred[sub.index.values]
                per_frame[f] = (sc, match(iou, sc, t), n_gt)
            for f in frames:
                if f not in per_frame:
                    per_frame[f] = (np.zeros(0), np.zeros(0, dtype=int), ious[f][1])
            per_set[key] = per_frame
        # original score as a reference
        orig = {}
        for f in frames:
            sub = boxes[boxes["frame"] == f]
            sc = sub["score"].values
            orig[f] = (sc, match(ious[f][0], sc, t), ious[f][1])
        per_set["original"] = orig
        results[t] = per_set

    rng = np.random.default_rng(seed)
    boots = [rng.choice(frames, size=len(frames), replace=True) for _ in range(n_boot)]
    p(f"  {'scores':10s} " + " ".join(f"{'AP' + str(int(t * 100)):>8s}" for t in IOUS)
      + "   " + " ".join(f"{'d' + str(int(t * 100)) + ' vs S+D [95% CI]':>28s}" for t in IOUS))
    for key in ["original", "S", "S+D", "S+D+E", "S+D+V", "S+D+L12", "S+D+all"]:
        aps, deltas = [], []
        for t in IOUS:
            aps.append(ap_from(results[t][key], frames))
            if key in ("original", "S", "S+D"):
                deltas.append("")
                continue
            d = [ap_from(results[t][key], b) - ap_from(results[t]["S+D"], b) for b in boots]
            point = aps[-1] - ap_from(results[t]["S+D"], frames)
            lo, hi = np.percentile(d, [2.5, 97.5])
            deltas.append(f"{point:+.4f} [{lo:+.4f},{hi:+.4f}]")
        p(f"  {key:10s} " + " ".join(f"{a:8.4f}" for a in aps) + "   "
          + " ".join(f"{d:>28s}" for d in deltas))
    p()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=prefix")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = []
    for spec in args.run:
        name, prefix = spec.split("=", 1)
        run(name, prefix, args.folds, args.boot, args.seed, lines)
        print("\n".join(lines[-12:]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
