"""
Cell-level study: do the Levy signals mark the cells where vehicles are?

Rows are bird's-eye-view cells sampled by ``levy_cells.py`` (all vehicle cells per
frame plus as many background cells), not detections, so the question is asked of
the fused feature map itself rather than of the detector's output.

Two labels, run separately:

* ``vehicle``  -- a ground-truth vehicle covers the cell (vs background).
* ``phantom``  -- among cells covered by a predicted box, the box is a false
  positive (vs a true positive).

Scoring, splits and metrics are the ones in ``levy_tpfp_study.py``; with
``--fit_prefix`` the models are fitted once on those cells (validation) and
applied unchanged to ``--prefix`` (test).

    python opencood/tools/levy/levy_cells_study.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy \\
        --prefix levy_intermediate_204.8_102.4_epoch1_use_cav4 \\
        --fit_prefix levy_intermediate_204.8_102.4_val_epoch1_use_cav4
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opencood.tools.levy.levy_tpfp_study import (
    BLOCK_FRAMES,
    COMPOSITIONS,
    FIT_GROUP,
    FUSED,
    GT_COMPARISONS,
    PER_LEVEL,
    auroc_table,
    build_scores,
    heldout_logistic,
    paired_tests,
    transformed,
    write_with_legend,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", default="levy_intermediate_204.8_102.4_epoch1_use_cav4")
    p.add_argument("--fit_prefix", default=None)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--max_rows", type=int, default=300000,
                   help="subsample whole frames down to about this many cells; "
                   "the confidence intervals are otherwise hours of work")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--export", default=None,
                   help="write the deployable mask parameters (fitted on the fitting "
                   "rows) to this JSON, for levy.vehicle_mask in the model config")
    p.add_argument("--export_only", action="store_true",
                   help="only write --export, skipping every table")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def subsample(df, max_rows, rng):
    """Keep whole frames, so 50-frame blocks and scenarios stay intact."""
    if not max_rows or len(df) <= max_rows:
        return df
    keys = df["scenario"] + "/" + df["frame"].astype(str)
    uniq = keys.unique()
    take = rng.permutation(len(uniq))[: max(1, int(len(uniq) * max_rows / len(df)))]
    keep = df[keys.isin(set(uniq[take]))].reset_index(drop=True)
    print(f"subsampled {len(df)} -> {len(keep)} cells ({len(take)} of {len(uniq)} frames)")
    return keep


def load_cells(model_dir, prefix):
    d = np.load(os.path.join(model_dir, f"{prefix}_cells.npz"), allow_pickle=True)
    df = pd.DataFrame({k: d[k] for k in d.files})
    df["composition"] = np.where(df["modalities"].isin(COMPOSITIONS), df["modalities"], "other")
    df["block"] = (df["scenario"] + "/" + (df["frame"] // BLOCK_FRAMES).astype(str))
    return df


def cell_weights(df):
    """How many grid cells each sampled cell stands for, per frame."""
    veh = df["n_vehicle_cells"].to_numpy(float)
    usable = df["n_usable_cells"].to_numpy(float)
    y = df["is_vehicle"].to_numpy()
    key = df["scenario"] + "/" + df["frame"].astype(str)
    counts = pd.DataFrame({"key": key, "y": y}).groupby(["key", "y"]).size()
    n_pos = counts.reindex(list(zip(key, np.ones(len(df), int)))).to_numpy()
    n_neg = counts.reindex(list(zip(key, np.zeros(len(df), int)))).to_numpy()
    w = np.where(y == 1, veh / np.maximum(n_pos, 1),
                 np.maximum(usable - veh, 0) / np.maximum(n_neg, 1))
    return w


def live_operating_points(score, y, w, recalls=(0.95, 0.9, 0.8, 0.6, 0.4, 0.2)):
    """Recall / precision / cells flagged per frame at the true class balance."""
    order = np.argsort(-score)
    ys, ws = y[order], w[order]
    tp = np.cumsum(ws * ys)
    fp = np.cumsum(ws * (1 - ys))
    total_pos = tp[-1]
    rows = []
    for target in recalls:
        i = int(np.searchsorted(tp / total_pos, target))
        i = min(i, len(tp) - 1)
        rows.append({
            "target_recall": target,
            "recall": tp[i] / total_pos,
            "precision": tp[i] / max(tp[i] + fp[i], 1e-9),
            "threshold_score": float(score[order][i]),
        })
    return rows


def export_mask(opt, src, label):
    """Fit the deployable mask on the fitting rows and write its parameters."""
    if not opt.export:
        return
    full = src
    # Every variant below runs with min_agents = 2, so it is fitted on that
    # regime; evidence has a different scale where a single agent contributes,
    # and the separate l0_evidence_single variant covers that case.
    if "fused_n_valid" in src.columns:
        src = src[src["fused_n_valid"] >= 2].reset_index(drop=True)
    y_src = label(src)
    w_src = cell_weights(src)
    variants = {
        "levy_only": list(PER_LEVEL),
        # Level 0 is the only one at the native 0.8 m grid; the coarser levels
        # enter the others as 1.6 m / 3.2 m blocks, which blurs boundaries.
        "l0_evidence": ["L0_log_evidence"],
        "l0_only": [f"L0_{k}" for k in ("log_evidence", "disagreement", "disagreement_cosine")],
        # The level mean, (L0 + up2 L1 + up4 L2) / 3, on the level-0 grid.
        "fused_evidence": ["fused_log_evidence"],
    }
    # Fitted on cells with a single contributor, where disagreement is zero but
    # evidence is not. Its thresholds are recall points within that regime.
    single = full[full["fused_n_valid"] == 1] if "fused_n_valid" in full.columns else full[:0]
    if "det_score" in src.columns:
        variants["levy_plus_detector"] = list(PER_LEVEL) + ["det_score"]
    export = {"fitted_on": opt.fit_prefix or opt.prefix,
              "n_fit_rows": int(len(src)),
              "vehicle_cell_share": float((w_src * y_src).sum() / w_src.sum()),
              "note": "raw disagreement enters as log(value + 1e-6), evidence as log "
                      "evidence, det_score as the head's probability"}
    for vname, feats in variants.items():
        X = transformed(src, [f for f in feats if f != "det_score"]).to_numpy()
        if "det_score" in feats:
            X = np.concatenate([X, src[["det_score"]].to_numpy(float)], 1)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X, y_src)
        scaler, logit = model[0], model[-1]
        prob = model.predict_proba(X)[:, 1]
        # Thresholds are recall points over the fitting subset's vehicle cells.
        # Recall needs no prevalence weighting, and the per-frame totals stored
        # with each cell count the whole grid rather than this subset, so a
        # weighted version would not mean what it says.
        table = live_operating_points(prob, y_src, np.ones(len(y_src)))
        export[vname] = {
            "enabled": True,
            "features": feats,
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
            "coef": [float(v) for v in logit.coef_[0]],
            "intercept": float(logit.intercept_[0]),
            "min_agents": 2,
            "threshold": float(next(r["threshold_score"] for r in table
                                    if r["target_recall"] == 0.8)),
            "threshold_table": [
                {k: float(v) for k, v in r.items()} for r in table],
        }
    if len(single) > 1000:
        y_s = label(single)
        X_s = transformed(single, ["L0_log_evidence"]).to_numpy()
        m_s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m_s.fit(X_s, y_s)
        prob_s = m_s.predict_proba(X_s)[:, 1]
        order = np.argsort(-prob_s)
        rec = np.cumsum(y_s[order]) / max(y_s.sum(), 1)
        table_s = []
        for target in (0.95, 0.9, 0.8, 0.6, 0.4, 0.2):
            i = min(int(np.searchsorted(rec, target)), len(rec) - 1)
            table_s.append({"target_recall": target, "recall": float(rec[i]),
                            "threshold_score": float(prob_s[order][i]),
                            "flagged_share": float((i + 1) / len(prob_s))})
        export["l0_evidence_single"] = {
            "enabled": True, "features": ["L0_log_evidence"],
            "mean": [float(m_s[0].mean_[0])], "scale": [float(m_s[0].scale_[0])],
            "coef": [float(m_s[-1].coef_[0][0])], "intercept": float(m_s[-1].intercept_[0]),
            "min_agents": 1, "max_agents": 1,
            "threshold": float(next(r["threshold_score"] for r in table_s
                                    if r["target_recall"] == 0.8)),
            "threshold_table": table_s,
            "n_fit_rows": int(len(single)),
            "note": "fitted on single-agent cells only; thresholds are recall points in that regime",
        }
    with open(opt.export, "w") as f:
        json.dump(export, f, indent=1)
    print(f"mask parameters written to {opt.export} "
          f"({len(src)} fitting cells, {100 * export['vehicle_cell_share']:.2f}% vehicle)")


def main():
    opt = parse_args()
    rng = np.random.default_rng(opt.seed)
    transfer = opt.fit_prefix is not None
    out = opt.out_dir or os.path.join(
        opt.model_dir, "analysis_cells_test" if transfer else "analysis_cells")
    os.makedirs(out, exist_ok=True)

    ev = subsample(load_cells(opt.model_dir, opt.prefix), opt.max_rows, rng)
    fit = (subsample(load_cells(opt.model_dir, opt.fit_prefix), opt.max_rows, rng)
           if transfer else None)
    protocol = (f"fitted once on {opt.fit_prefix} and applied unchanged to these rows"
                if transfer else "leave-one-scenario-out held-out predictions")

    for study, select, label, what in (
        ("vehicle", lambda d: np.ones(len(d), bool), lambda d: d["is_vehicle"].to_numpy(int),
         "vehicle cell"),
        ("phantom", lambda d: (d["pred_tp"] >= 0).to_numpy(),
         lambda d: (1 - d["pred_tp"]).to_numpy(int), "cell under a false detection"),
    ):
        df = ev[select(ev)].reset_index(drop=True)
        y = label(df)
        groups = df["scenario"].to_numpy()
        print(f"\n[{study}] {len(df)} cells, {y.mean():.3f} positive "
              f"({df['scenario'].nunique()} scenarios)")
        if len(np.unique(y)) < 2:
            continue

        extra = {}
        if "det_score" in ev.columns:
            extra["detector_score_map"] = (
                "baseline: the detector's own classification score at the cell")

        if transfer:
            df_fit = fit[select(fit)].reset_index(drop=True)
            y_fit = label(df_fit)
            both = pd.concat([df_fit, df], ignore_index=True)
            y_both = np.concatenate([y_fit, y])
            g_both = np.concatenate([np.full(len(df_fit), FIT_GROUP, dtype=object),
                                     groups.astype(object)])
            is_eval = g_both != FIT_GROUP
            scores, notes, weights, weight_names = build_scores(both, y_both, g_both, False)
            if extra:
                det = both["det_score"].to_numpy(float)[:, None]
                scores["detector_score_map"] = det[:, 0]
                notes["detector_score_map"] = extra["detector_score_map"]
                lv = transformed(both, PER_LEVEL).to_numpy()
                scores["logistic_detector_only"], _ = heldout_logistic(det, y_both, g_both)
                notes["logistic_detector_only"] = "control: the detector's score map alone (fitted)"
                scores["logistic_levy_plus_detector"], weights["logistic_levy_plus_detector"] = (
                    heldout_logistic(np.concatenate([lv, det], 1), y_both, g_both))
                notes["logistic_levy_plus_detector"] = "9 Levy signals + the detector's score map"
                weight_names["logistic_levy_plus_detector"] = PER_LEVEL + ["det_score"]
            scores = {k: v[is_eval] for k, v in scores.items()}
            shared = set(df_fit["scenario"]) & set(df["scenario"])
            print(f"[{study}] fitted on {len(df_fit)} cells ({y_fit.mean():.3f} positive); "
                  f"scenarios shared with evaluation rows: {len(shared)} (should be 0)")
        else:
            scores, notes, weights, weight_names = build_scores(df, y, groups, False)
            if extra:
                det = df["det_score"].to_numpy(float)[:, None]
                scores["detector_score_map"] = det[:, 0]
                notes["detector_score_map"] = extra["detector_score_map"]
                lv = transformed(df, PER_LEVEL).to_numpy()
                scores["logistic_detector_only"], _ = heldout_logistic(det, y, groups)
                notes["logistic_detector_only"] = "control: the detector's score map alone (fitted)"
                scores["logistic_levy_plus_detector"], weights["logistic_levy_plus_detector"] = (
                    heldout_logistic(np.concatenate([lv, det], 1), y, groups))
                notes["logistic_levy_plus_detector"] = "9 Levy signals + the detector's score map"
                weight_names["logistic_levy_plus_detector"] = PER_LEVEL + ["det_score"]

        if opt.export_only:
            if study == "vehicle":
                export_mask(opt, fit[select(fit)].reset_index(drop=True) if transfer else df,
                            label)
            continue

        table = auroc_table(df, y, scores, notes, opt.bootstrap, rng, False)
        write_with_legend(table, os.path.join(out, f"auroc_cells_{study}.csv"), [
            f"rows = sampled bird's-eye-view cells covered by >= 2 agents; positive = {what}",
            "auroc[...] = area under the ROC curve; 0.5 = chance; < 0.5 = the signal points the other way",
            f"single signals are raw values (no fitting); {protocol}",
            "auroc[within_distance_deciles] is the key column: background cells sit farther from the ego "
            "vehicle on average, so the pooled value partly restates range",
            "auroc_ci_low/high = 95% interval over 50-frame blocks",
        ])

        comparisons = list(GT_COMPARISONS)
        if "detector_score_map" in scores:
            comparisons += [("logistic_B_levy", "detector_score_map"),
                            ("logistic_levy_plus_detector", "detector_score_map"),
                            ("logistic_levy_plus_detector", "logistic_B_levy")]
        paired = paired_tests(df, y, scores, groups, comparisons, opt.bootstrap, rng, False)
        write_with_legend(paired, os.path.join(out, f"paired_cells_{study}.csv"), [
            f"paired comparison of model_a vs model_b on the same cells; positive = {what}",
            "*_difference_ci = 95% range of a - b over resamples where both models see the same frames",
            "misses_captured_flagging_X = share of positive cells inside the least-trusted X of cells",
        ])

        wrows = []
        for nm, w in weights.items():
            if w is None:
                continue
            for feat, val in zip(weight_names[nm], w):
                wrows.append({"model": nm, "feature": feat, "standardised_weight": val})
        write_with_legend(pd.DataFrame(wrows), os.path.join(out, f"logistic_weights_cells_{study}.csv"),
                          [f"standardised weights, averaged over folds; positive = pushes towards {what}"])

        # distance-matched check: equal numbers of positive / negative cells per 10 m ring
        rings = (df["dist"] // 10).astype(int).to_numpy()
        keep = np.zeros(len(df), bool)
        for r in np.unique(rings):
            pos = np.flatnonzero((rings == r) & (y == 1))
            neg = np.flatnonzero((rings == r) & (y == 0))
            n = min(len(pos), len(neg))
            if n == 0:
                continue
            keep[rng.choice(pos, n, replace=False)] = True
            keep[rng.choice(neg, n, replace=False)] = True
        rows = []
        for nm, s in scores.items():
            rows.append({"score": nm, "auroc_distance_matched": roc_auc_score(y[keep], s[keep]),
                         "auroc_all": roc_auc_score(y, s), "n": int(keep.sum())})
        write_with_legend(pd.DataFrame(rows), os.path.join(out, f"distance_matched_cells_{study}.csv"), [
            f"positive = {what}; cells resampled so that each 10 m ring holds equally many "
            "positive and negative cells, which removes the range difference entirely",
        ])

        # What a live mask would do on the real grid, where vehicle cells are rare.
        # Only meaningful for the vehicle study: the per-frame totals count vehicle
        # cells, not cells under a predicted box.
        if study == "vehicle" and "n_usable_cells" in df.columns:
            w = cell_weights(df)
            prevalence = (w * y).sum() / w.sum()
            per_frame = df.groupby(df["scenario"] + "/" + df["frame"].astype(str))[
                "n_usable_cells"].first().median()
            rows = []
            for nm in ("logistic_D_per_agent_count", "logistic_B_levy",
                       "logistic_levy_plus_detector", "detector_score_map",
                       "rank_average_all_levels", "distance"):
                if nm not in scores:
                    continue
                for r in live_operating_points(scores[nm], y, w):
                    r["score"] = nm
                    r["cells_flagged_per_frame"] = (
                        (r["recall"] * prevalence / max(r["precision"], 1e-9)) * per_frame)
                    rows.append(r)
            live = pd.DataFrame(rows)[
                ["score", "target_recall", "recall", "precision", "cells_flagged_per_frame",
                 "threshold_score"]]
            write_with_legend(live, os.path.join(out, f"live_operating_points_{study}.csv"), [
                f"what a live mask would do on the full grid: vehicle cells are only "
                f"{100 * prevalence:.2f}% of usable cells, and the sampled rows are re-weighted "
                "back to that share",
                f"cells_flagged_per_frame uses the median frame ({per_frame:,.0f} usable cells)",
                "precision = share of flagged cells that really hold a vehicle",
            ])
            print(f"[{study}] true share of vehicle cells: {100 * prevalence:.2f}%")

        if study == "vehicle":
            export_mask(opt, df_fit if transfer else df, label)

        np.savez_compressed(os.path.join(out, f"heldout_scores_cells_{study}.npz"),
                            label=y, scenario=groups, dist=df["dist"].to_numpy(),
                            fused_n_valid=df["fused_n_valid"].to_numpy(), **scores)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
