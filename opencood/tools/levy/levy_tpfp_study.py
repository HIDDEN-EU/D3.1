"""
Do the Levy trust signals mark real objects?

Two separate studies on the CSVs written by
``inference_heter_in_order.py --save_levy`` (use_cav 2, 3 and 4 pooled):

* predicted boxes -- does a signal at the box centre separate true positives
  from false positives (area under the ROC curve), and does removing the least
  trusted boxes first lower the false-positive rate of the rest (area under the
  sparsification error curve)?
* ground-truth vehicles -- does a signal at the vehicle centre separate the
  vehicles the detector found from the ones it missed?

Every signal and combination is scored under one shared protocol: the same rows,
the same labels, the same leave-one-scenario-out folds, and directions / weights
learned on the training folds only.

With ``--fit_prefix`` the models are instead fitted once on those CSVs (e.g. the
validation split) and evaluated unchanged on ``--prefix`` (e.g. the test split).

Example
    python opencood/tools/levy/levy_tpfp_study.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy

    # fit on validation, evaluate on test
    python opencood/tools/levy/levy_tpfp_study.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy \\
        --prefix levy_intermediate_204.8_102.4_epoch1 \\
        --fit_prefix levy_intermediate_204.8_102.4_val_epoch1
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from opencood.tools.levy.levy_analysis import (
    auroc,
    ause,
    conditional_association,
    quantile_bins,
    sparsification_curve,
)

LEVELS = ("L0", "L1", "L2")
KINDS = ("log_evidence", "disagreement", "disagreement_cosine")
KIND_NAMES = {
    "log_evidence": "evidence (log)",
    "disagreement": "raw disagreement (trace of Sigma)",
    "disagreement_cosine": "cosine disagreement",
}
LEVEL_NAMES = {"L0": "level 0", "L1": "level 1", "L2": "level 2", "fused": "level mean"}
PER_LEVEL = [f"{lv}_{k}" for lv in LEVELS for k in KINDS]
FUSED = [f"fused_{k}" for k in KINDS]
COMPOSITIONS = ("m1m4", "m1m4m3", "m1m4m3m4")
BLOCK_FRAMES = 50
DISAGREEMENT_EPS = 1e-6
# Group label of the fitting rows in fit-once mode (see ``folds``).
FIT_GROUP = "__fit__"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", default="levy_intermediate_204.8_102.4_val_epoch1",
                   help="CSVs the results are reported on")
    p.add_argument("--fit_prefix", default=None,
                   help="fit every model once on these CSVs and evaluate on --prefix; "
                   "default: leave-one-scenario-out within --prefix")
    p.add_argument("--use_cav", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--iou", nargs="*", type=int, default=[70, 50])
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def signal_label(col):
    level, kind = col.split("_", 1)
    return f"{KIND_NAMES[kind]}, {LEVEL_NAMES[level]}"


# --------------------------------------------------------------------------- data


def load(model_dir, prefix, use_cavs):
    """Pooled, de-duplicated box and ground-truth rows with scenario/composition."""
    boxes, gts, frames = [], [], []
    for n in use_cavs:
        base = os.path.join(model_dir, f"{prefix}_use_cav{n}")
        fr = pd.read_csv(f"{base}_frames.csv", usecols=lambda c: c in (
            "frame", "use_cav", "scenario", "timestamp", "modalities", "n_agents"))
        fr["timestamp"] = fr["timestamp"].astype(str)
        frames.append(fr)
        boxes.append(pd.read_csv(f"{base}_boxes.csv"))
        gts.append(pd.read_csv(f"{base}_gt.csv"))
    frames = pd.concat(frames, ignore_index=True)

    # The same frame with the same agents appears in several settings (e.g. a
    # two-car scenario is identical under use_cav 2, 3 and 4); keep it once.
    frames = frames.sort_values(["use_cav", "frame"])
    before = len(frames)
    frames = frames.drop_duplicates(["scenario", "timestamp", "modalities"], keep="first")
    print(f"frames: {before} pooled, {len(frames)} after removing repeated compositions")

    frames["composition"] = np.where(
        frames["modalities"].isin(COMPOSITIONS), frames["modalities"], "other"
    )
    frames["block"] = (
        frames["scenario"] + "/" + (frames["frame"] // BLOCK_FRAMES).astype(str)
        + "/" + frames["use_cav"].astype(str)
    )
    keep = frames[["use_cav", "frame", "scenario", "composition", "block"]]
    boxes = pd.concat(boxes, ignore_index=True).merge(keep, on=["use_cav", "frame"])
    gts = pd.concat(gts, ignore_index=True).merge(keep, on=["use_cav", "frame"])
    return boxes, gts


def transformed(df, cols):
    """Model inputs: log evidence as is, log(raw disagreement), cosine as is."""
    out = {}
    for c in cols:
        v = df[c].to_numpy(dtype=float)
        if c.endswith("_disagreement"):
            v = np.log(v + DISAGREEMENT_EPS)
        out[c] = v
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------- scoring


def folds(groups):
    """Leave-one-scenario-out splits, or the single fit/evaluate split when the
    fitting rows are labelled ``FIT_GROUP``."""
    if (groups == FIT_GROUP).any():
        fit = groups == FIT_GROUP
        yield fit, ~fit
        return
    for g in np.unique(groups):
        test = groups == g
        yield ~test, test


def percentile(values):
    """Rank in [0, 1], ties averaged."""
    ranks = pd.Series(values).rank(method="average").to_numpy() - 1
    return ranks / max(len(values) - 1, 1)


def heldout_logistic(X, y, groups, strata=None):
    """Leave-one-scenario-out predicted probability of the positive class."""
    pred = np.full(len(y), np.nan)
    weights = []
    for train, test in folds(groups):
        parts = [(train, test)] if strata is None else [
            (train & (strata == s), test & (strata == s)) for s in np.unique(strata)
        ]
        for tr, te in parts:
            if te.sum() == 0:
                continue
            if tr.sum() < 20 or len(np.unique(y[tr])) < 2:
                pred[te] = y[tr].mean() if tr.sum() else 0.5
                continue
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(X[tr], y[tr])
            pred[te] = model.predict_proba(X[te])[:, 1]
            if strata is None:
                weights.append(model[-1].coef_[0])
    return pred, (np.mean(weights, axis=0) if weights else None)


def heldout_rank_average(X, y, groups):
    """Mean percentile of the columns, each oriented by its training-fold AUROC."""
    pred = np.full(len(y), np.nan)
    for train, test in folds(groups):
        acc = np.zeros(test.sum())
        for j in range(X.shape[1]):
            sign = 1.0 if roc_auc_score(y[train], X[train, j]) >= 0.5 else -1.0
            acc += percentile(sign * X[test, j])
        pred[test] = acc / X.shape[1]
    return pred


def heldout_best_single(X, y, groups):
    """Per fold: the column with the strongest training AUROC, oriented."""
    pred = np.full(len(y), np.nan)
    chosen = []
    for train, test in folds(groups):
        aucs = np.array([roc_auc_score(y[train], X[train, j]) for j in range(X.shape[1])])
        j = int(np.argmax(np.abs(aucs - 0.5)))
        chosen.append(j)
        pred[test] = percentile((1.0 if aucs[j] >= 0.5 else -1.0) * X[test, j])
    return pred, chosen


def bootstrap_ci(score, y, blocks, reps, rng):
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by_block = [np.flatnonzero(inv == b) for b in range(len(ub))]
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(ub), len(ub))
        idx = np.concatenate([idx_by_block[b] for b in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(vals, [2.5, 97.5])


def fp_removed_at_tp_budget(score, is_fp, budget=0.05):
    """Share of false positives removed when the least trusted boxes are dropped
    until ``budget`` of the true positives are gone."""
    order = np.argsort(-score, kind="mergesort")
    fp = is_fp[order]
    tp_cum = np.cumsum(1 - fp)
    n_tp = tp_cum[-1]
    cut = np.searchsorted(tp_cum, budget * n_tp, side="right")
    return float(fp[:cut].sum() / max(fp.sum(), 1))


def capture_curve(score, positive, fractions):
    order = np.argsort(-score, kind="mergesort")
    pos = positive[order]
    n = len(pos)
    return [float(pos[: int(round(f * n))].sum() / max(pos.sum(), 1)) for f in fractions]


# ------------------------------------------------------------------------- study


def build_scores(df, y, groups, with_score):
    """All scores, oriented so that higher = more likely the positive label."""
    scores, notes, weights = {}, {}, {}
    for c in PER_LEVEL + FUSED:
        scores[c] = df[c].to_numpy(dtype=float)
        notes[c] = f"single signal: {signal_label(c)} (raw value, no fitting)"
    scores["n_observing_agents"] = -df["fused_n_valid"].to_numpy(dtype=float)
    notes["n_observing_agents"] = "baseline: fewer observing agents = less trusted"
    scores["distance"] = df["dist"].to_numpy(dtype=float)
    notes["distance"] = "baseline: farther from the ego vehicle = less trusted"
    if with_score:
        scores["detection_score"] = -df["score"].to_numpy(dtype=float)
        notes["detection_score"] = "baseline: lower detection confidence = less trusted"

    Xl = transformed(df, PER_LEVEL).to_numpy()
    Xf = transformed(df, FUSED).to_numpy()
    scores["rank_average_all_levels"] = heldout_rank_average(Xl, y, groups)
    notes["rank_average_all_levels"] = "rank average of the 9 per-level signals"
    scores["rank_average_level_mean"] = heldout_rank_average(Xf, y, groups)
    notes["rank_average_level_mean"] = "rank average of the 3 level-mean signals"
    scores["best_single_signal"], chosen = heldout_best_single(
        np.concatenate([Xl, Xf], 1), y, groups)
    names = PER_LEVEL + FUSED
    notes["best_single_signal"] = "per fold, the single signal with the best training AUROC; chosen: " + ", ".join(
        sorted({names[j] for j in chosen}))

    for kind in KINDS:
        cols = [f"{lv}_{kind}" for lv in LEVELS]
        name = f"logistic_A_{kind}"
        scores[name], weights[name] = heldout_logistic(
            transformed(df, cols).to_numpy(), y, groups)
        notes[name] = f"logistic regression on {KIND_NAMES[kind]} at levels 0, 1, 2"
    scores["logistic_B_levy"], weights["logistic_B_levy"] = heldout_logistic(Xl, y, groups)
    notes["logistic_B_levy"] = "logistic regression on all 9 per-level Levy signals"

    nv = df["fused_n_valid"].clip(upper=4).to_numpy()
    onehot = np.stack([(nv == k).astype(float) for k in (3, 4)], 1)
    extra = [Xl]
    extra_names = list(PER_LEVEL)
    if with_score:
        s = df["score"].to_numpy(dtype=float)[:, None]
        scores["logistic_C_levy_plus_score"], weights["logistic_C_levy_plus_score"] = \
            heldout_logistic(np.concatenate([Xl, s], 1), y, groups)
        notes["logistic_C_levy_plus_score"] = "logistic regression on the 9 Levy signals + detection score"
        extra.append(s)
        extra_names.append("score")
        ctrl = np.concatenate([s, df[["dist"]].to_numpy()], 1)
        scores["logistic_control_score_distance"], weights["logistic_control_score_distance"] = \
            heldout_logistic(ctrl, y, groups)
        notes["logistic_control_score_distance"] = "control without Levy: detection score + distance"
    else:
        scores["logistic_control_distance"], weights["logistic_control_distance"] = \
            heldout_logistic(df[["dist"]].to_numpy(), y, groups)
        notes["logistic_control_distance"] = "control without Levy: distance only"
    Xd = np.concatenate(extra + [onehot], 1)
    tag = "C" if with_score else "B"
    scores["logistic_D_plus_agent_count"], weights["logistic_D_plus_agent_count"] = \
        heldout_logistic(Xd, y, groups)
    notes["logistic_D_plus_agent_count"] = (
        f"logistic {tag} + number of observing agents (3 or 4 vs 2, as indicators)")
    scores["logistic_D_per_agent_count"], _ = heldout_logistic(
        np.concatenate(extra, 1), y, groups, strata=nv)
    notes["logistic_D_per_agent_count"] = f"logistic {tag}, fitted separately for 2, 3 and 4 observing agents"
    weight_names = {
        **{f"logistic_A_{k}": [f"{lv}_{k}" for lv in LEVELS] for k in KINDS},
        "logistic_B_levy": PER_LEVEL,
        "logistic_C_levy_plus_score": PER_LEVEL + ["score"],
        "logistic_control_score_distance": ["score", "dist"],
        "logistic_control_distance": ["dist"],
        "logistic_D_plus_agent_count": extra_names + ["3_agents", "4_agents"],
    }
    return scores, notes, weights, weight_names


def scenario_columns(s, y, groups, overall):
    """
    Within-scenario percentile AUROC (each scenario rescaled to 0-1, then pooled)
    and the spread of the AUROC over the scenarios taken one at a time.
    """
    pct = np.empty(len(s))
    per = []
    for g in np.unique(groups):
        m = groups == g
        pct[m] = percentile(s[m])
        if len(np.unique(y[m])) == 2:
            per.append(roc_auc_score(y[m], s[m]))
    per = np.array(per)
    same_side = int(((per - 0.5) * (overall - 0.5) > 0).sum())
    return {
        "auroc_within_scenario_percentile": roc_auc_score(y, pct),
        "auroc_per_scenario_min": per.min(),
        "auroc_per_scenario_median": float(np.median(per)),
        "auroc_per_scenario_max": per.max(),
        "scenarios_same_direction_as_overall": f"{same_side}/{len(per)}",
    }


def paired_tests(df, y, scores, groups, comparisons, reps, rng, with_score):
    """
    Model A vs model B on the same resampled frames: the 95% range of the
    difference, and per held-out scenario which model wins.
    """
    blocks = df["block"].to_numpy()
    ub, inv = np.unique(blocks, return_inverse=True)
    idx_by_block = [np.flatnonzero(inv == b) for b in range(len(ub))]
    draws = [np.concatenate([idx_by_block[b] for b in rng.integers(0, len(ub), len(ub))])
             for _ in range(reps)]

    if with_score:
        metrics = {
            "auroc": lambda s, t: roc_auc_score(t, s),
            "fp_removed_at_5pct_tp_loss": lambda s, t: fp_removed_at_tp_budget(s, t, 0.05),
            "fp_removed_at_10pct_tp_loss": lambda s, t: fp_removed_at_tp_budget(s, t, 0.10),
        }
    else:
        metrics = {
            "auroc": lambda s, t: roc_auc_score(t, s),
            "misses_captured_flagging_20pct": lambda s, t: capture_curve(s, t, [0.2])[0],
            "misses_captured_flagging_30pct": lambda s, t: capture_curve(s, t, [0.3])[0],
        }

    rows = []
    for a, b in comparisons:
        sa, sb = scores[a], scores[b]
        row = {"model_a": a, "model_b": b}
        for mname, fn in metrics.items():
            diffs = np.array([fn(sa[i], y[i]) - fn(sb[i], y[i]) for i in draws])
            row[f"{mname}_a"] = fn(sa, y)
            row[f"{mname}_b"] = fn(sb, y)
            row[f"{mname}_difference"] = row[f"{mname}_a"] - row[f"{mname}_b"]
            row[f"{mname}_difference_ci_low"], row[f"{mname}_difference_ci_high"] = \
                np.percentile(diffs, [2.5, 97.5])
            row[f"{mname}_share_of_resamples_a_better"] = float((diffs > 0).mean())
        wins, per_diff = 0, []
        for g in np.unique(groups):
            m = groups == g
            if len(np.unique(y[m])) < 2:
                continue
            d = roc_auc_score(y[m], sa[m]) - roc_auc_score(y[m], sb[m])
            per_diff.append(d)
            wins += int(d > 0)
        row["auroc_scenarios_a_wins"] = f"{wins}/{len(per_diff)}"
        row["auroc_per_scenario_difference_min"] = min(per_diff)
        row["auroc_per_scenario_difference_median"] = float(np.median(per_diff))
        row["auroc_per_scenario_difference_max"] = max(per_diff)
        rows.append(row)
    return pd.DataFrame(rows)


BOX_COMPARISONS = (
    ("logistic_B_levy", "detection_score"),
    ("logistic_C_levy_plus_score", "detection_score"),
    ("logistic_C_levy_plus_score", "logistic_control_score_distance"),
    ("logistic_B_levy", "logistic_control_score_distance"),
    ("logistic_D_per_agent_count", "logistic_C_levy_plus_score"),
    ("logistic_D_plus_agent_count", "logistic_C_levy_plus_score"),
)
GT_COMPARISONS = (
    ("logistic_B_levy", "distance"),
    ("logistic_B_levy", "logistic_control_distance"),
    ("logistic_D_plus_agent_count", "logistic_B_levy"),
    ("logistic_D_per_agent_count", "logistic_B_levy"),
)


def auroc_table(df, y, scores, notes, reps, rng, with_score):
    rows = []
    groups = df["scenario"].to_numpy()
    strata = {"all": np.ones(len(df), bool)}
    for comp in COMPOSITIONS + ("other",):
        strata[f"composition={comp}"] = (df["composition"] == comp).to_numpy()
    for k in (2, 3, 4):
        strata[f"observing_agents={k}"] = (df["fused_n_valid"].clip(upper=4) == k).to_numpy()
    blocks = df["block"].to_numpy()
    cond = {"within_distance_deciles": quantile_bins(df["dist"].to_numpy(dtype=float), 10)}
    if with_score:
        cond["within_score_deciles"] = quantile_bins(df["score"].to_numpy(dtype=float), 10)
    for name, s in scores.items():
        row = {"score": name, "description": notes[name]}
        for sname, m in strata.items():
            if m.sum() < 50 or len(np.unique(y[m])) < 2:
                row[f"auroc[{sname}]"] = np.nan
            else:
                row[f"auroc[{sname}]"] = roc_auc_score(y[m], s[m])
            row[f"n[{sname}]"] = int(m.sum())
        lo, hi = bootstrap_ci(s, y, blocks, reps, rng)
        row["auroc_ci_low"], row["auroc_ci_high"] = lo, hi
        row.update(scenario_columns(s, y, groups, row["auroc[all]"]))
        for cname, codes in cond.items():
            row[f"auroc[{cname}]"] = conditional_association(s, y, codes, 50, True)["auroc"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_with_legend(df, path, legend):
    with open(path, "w") as f:
        for line in legend:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False, float_format="%.4f")


# -------------------------------------------------------------------------- main


def sanity_checks(rng):
    y = rng.integers(0, 2, 5000)
    v = rng.normal(size=5000) + y * 0.5
    assert abs(auroc(v, y) - roc_auc_score(y, v)) < 1e-9, "auroc mismatch"
    oracle = sparsification_curve(y.astype(float), y.astype(float), 100)
    assert oracle.min() == 0.0, "oracle sparsification never reaches 0"
    print("sanity: auroc matches sklearn, oracle sparsification reaches 0")


def main():
    opt = parse_args()
    rng = np.random.default_rng(opt.seed)
    transfer = opt.fit_prefix is not None
    out = opt.out_dir or os.path.join(
        opt.model_dir, "analysis_tpfp_test" if transfer else "analysis_tpfp")
    os.makedirs(out, exist_ok=True)
    sanity_checks(rng)

    boxes, gts = load(opt.model_dir, opt.prefix, opt.use_cav)
    fit_boxes, fit_gts = (
        load(opt.model_dir, opt.fit_prefix, opt.use_cav) if transfer else (None, None))
    needed = PER_LEVEL + FUSED
    protocol = (
        f"combinations fitted once on {opt.fit_prefix} and applied unchanged to these rows"
        if transfer else
        "combinations are leave-one-scenario-out held-out predictions")

    def overlap_rows(df_all):
        finite = np.isfinite(df_all[needed].to_numpy(dtype=float)).all(1)
        return df_all[(df_all["fused_n_valid"] >= 2).to_numpy() & finite].reset_index(drop=True)

    summary = {}
    for study, df_all, fit_all, label_col, with_score in (
        ("boxes", boxes, fit_boxes, "tp", True),
        ("gt", gts, fit_gts, "hit", False),
    ):
        counts = df_all["fused_n_valid"].clip(upper=4).value_counts().sort_index()
        print(f"\n[{study}] rows {len(df_all)}, by number of observing agents: {counts.to_dict()}")
        finite = np.isfinite(df_all[needed].to_numpy(dtype=float)).all(1)
        overlap = (df_all["fused_n_valid"] >= 2).to_numpy() & finite
        single = (df_all["fused_n_valid"] == 1).to_numpy() & finite
        df = df_all[overlap].reset_index(drop=True)
        df1 = df_all[single].reset_index(drop=True)
        print(f"[{study}] overlap rows (>=2 observing agents): {len(df)}; "
              f"single-agent rows: {len(df1)}; compositions: {df.composition.value_counts().to_dict()}")

        for iou in opt.iou:
            y = (1 - df[f"{label_col}{iou}"]).to_numpy().astype(int)
            groups = df["scenario"].to_numpy()
            what = "false positive" if with_score else "missed vehicle"
            print(f"[{study}] IoU 0.{iou // 10}: {what} rate {y.mean():.3f}")

            if transfer:
                # Fit rows first, evaluation rows after; every model sees one
                # split (fit -> evaluate) through ``folds``.
                df_fit = overlap_rows(fit_all)
                y_fit = (1 - df_fit[f"{label_col}{iou}"]).to_numpy().astype(int)
                both = pd.concat([df_fit, df], ignore_index=True)
                y_both = np.concatenate([y_fit, y])
                g_both = np.concatenate([np.full(len(df_fit), FIT_GROUP, dtype=object),
                                         groups.astype(object)])
                is_eval = g_both != FIT_GROUP
                scores, notes, weights, weight_names = build_scores(both, y_both, g_both, with_score)
                scores = {k: v[is_eval] for k, v in scores.items()}
                print(f"[{study}] fitted on {len(df_fit)} rows ({y_fit.mean():.3f} positive)")
                # Leakage control: the fitting and evaluation scenarios must be
                # disjoint (a shuffled-label refit is not a valid check here -
                # a random mix of these co-directional signals still ranks well).
                shared = set(df_fit["scenario"]) & set(df["scenario"])
                print(f"[{study}] scenarios shared by fit and evaluation rows: {len(shared)} (should be 0)")
            else:
                scores, notes, weights, weight_names = build_scores(df, y, groups, with_score)
                y_shuf = rng.permutation(y)
                p_shuf, _ = heldout_logistic(transformed(df, PER_LEVEL).to_numpy(), y_shuf, groups)
                check = roc_auc_score(y_shuf, p_shuf)
                # leakage control: a model fitted on shuffled labels must give chance
                print(f"[{study}] shuffled-label logistic AUROC: {check:.3f} (should be ~0.5)")

            table = auroc_table(df, y, scores, notes, opt.bootstrap, rng, with_score)
            legend = [
                f"study: {study}; positive label = {what} at IoU 0.{iou // 10}; rows = "
                f"{'predicted boxes' if with_score else 'ground-truth vehicles'} whose centre cell is observed by >= 2 agents",
                "auroc[...] = area under the ROC curve; 0.5 = chance; > 0.5 = higher score marks the positive label; "
                "< 0.5 = the signal points the other way",
                f"single signals are raw values (no fitting); {protocol}",
                "auroc_ci_low/high = 95% interval from a bootstrap over 50-frame blocks",
                "auroc[within_*_deciles] = area under the ROC curve computed inside each decile, size-weighted",
                "auroc_within_scenario_percentile = each score ranked 0-1 within its scenario, then pooled "
                "(ignores differences between scenes)",
                "auroc_per_scenario_min/median/max = the area under the ROC curve in each of the 9 scenarios separately; "
                "scenarios_same_direction_as_overall = scenarios on the same side of 0.5 as the pooled value",
                "n[...] = rows in that subset",
            ]
            write_with_legend(table, os.path.join(out, f"auroc_{study}_iou{iou}.csv"), legend)

            paired = paired_tests(df, y, scores, groups,
                                  BOX_COMPARISONS if with_score else GT_COMPARISONS,
                                  opt.bootstrap, rng, with_score)
            write_with_legend(paired, os.path.join(out, f"paired_{study}_iou{iou}.csv"), [
                f"paired comparison of model_a vs model_b on the same rows; positive label = {what} at IoU 0.{iou // 10}",
                "*_difference = a - b on all rows; *_difference_ci_low/high = 95% range of a - b over "
                f"{opt.bootstrap} resamples of 50-frame blocks, both models scored on the same resample",
                "*_share_of_resamples_a_better = fraction of resamples where a beats b",
                "auroc_scenarios_a_wins = held-out scenarios where a has the higher area under the ROC curve; "
                "per_scenario_difference = a - b in each scenario",
            ])
            np.savez_compressed(
                os.path.join(out, f"heldout_scores_{study}_iou{iou}.npz"),
                label=y, scenario=groups, frame=df["frame"].to_numpy(),
                use_cav=df["use_cav"].to_numpy(), **scores)

            wrows = []
            for name, w in weights.items():
                if w is None:
                    continue
                for feat, val in zip(weight_names[name], w):
                    wrows.append({"model": name, "feature": feat, "standardised_weight": val})
            write_with_legend(pd.DataFrame(wrows), os.path.join(out, f"logistic_weights_{study}_iou{iou}.csv"), [
                "weights of the standardised features, averaged over the folds (one fold in fit-once mode); positive = pushes towards "
                f"{what}; raw disagreement enters as log(value + 1e-6), evidence as log evidence",
            ])

            # single-agent region: evidence only (disagreement is exactly zero)
            y1 = (1 - df1[f"{label_col}{iou}"]).to_numpy().astype(int)
            single_rows = []
            if len(np.unique(y1)) == 2:
                for c in [f"{lv}_log_evidence" for lv in LEVELS + ("fused",)] + (
                        ["score"] if with_score else []) + ["dist"]:
                    v = df1[c].to_numpy(dtype=float) * (-1 if c == "score" else 1)
                    single_rows.append({"score": c, "auroc": roc_auc_score(y1, v), "n": len(y1)})
            write_with_legend(pd.DataFrame(single_rows),
                              os.path.join(out, f"auroc_{study}_single_agent_iou{iou}.csv"), [
                f"rows observed by exactly one agent; only evidence is defined there. positive = {what}"])

            if with_score:
                ause_rows, curves = [], {}
                chosen = ["detection_score", "best_single_signal", "rank_average_all_levels",
                          "rank_average_level_mean", "logistic_B_levy", "logistic_C_levy_plus_score",
                          "logistic_D_plus_agent_count", "logistic_D_per_agent_count",
                          "logistic_control_score_distance"]
                fractions = np.arange(opt.steps) / opt.steps
                for name in chosen:
                    s = scores[name]
                    a = ause(y.astype(float), s, opt.steps)
                    curves[name] = a["curve"]
                    ause_rows.append({
                        "score": name, "description": notes[name],
                        "ause": a["ause"], "ause_random": a["ause_random"],
                        "fp_rate_after_removing_10pct": a["curve"][10],
                        "fp_rate_after_removing_30pct": a["curve"][30],
                        "fp_removed_at_5pct_tp_loss": fp_removed_at_tp_budget(s, y),
                        "fp_removed_at_10pct_tp_loss": fp_removed_at_tp_budget(s, y, 0.10),
                    })
                    for comp in COMPOSITIONS:
                        m = (df["composition"] == comp).to_numpy()
                        if m.sum() > 100:
                            ause_rows[-1][f"ause[{comp}]"] = ause(y[m].astype(float), s[m], opt.steps)["ause"]
                            ause_rows[-1][f"ause_random[{comp}]"] = ause(
                                y[m].astype(float), s[m], opt.steps)["ause_random"]
                curves["oracle"] = sparsification_curve(y.astype(float), y.astype(float), opt.steps).tolist()
                curves["random"] = [float(y.mean())] * opt.steps
                write_with_legend(pd.DataFrame(ause_rows), os.path.join(out, f"ause_boxes_iou{iou}.csv"), [
                    "sparsification: remove boxes from least to most trusted and track the false-positive rate of the rest",
                    "ause = area between that curve and the oracle curve (removes the real false positives first); lower is better",
                    "ause_random = the same area for random removal; a useful score is well below it",
                    "fp_removed_at_X_tp_loss = share of false positives removed when X of the true positives have been removed",
                ])
                with open(os.path.join(out, f"sparsification_boxes_iou{iou}.json"), "w") as f:
                    json.dump({"fractions": fractions.tolist(), "curves": curves}, f)
            else:
                fr = [0.05, 0.1, 0.2, 0.3, 0.5]
                cap = []
                for name in ["distance", "best_single_signal", "rank_average_all_levels",
                             "logistic_B_levy", "logistic_D_plus_agent_count", "logistic_control_distance"]:
                    row = {"score": name, "description": notes[name]}
                    for f_, v in zip(fr, capture_curve(scores[name], y, fr)):
                        row[f"misses_captured_flagging_{int(f_ * 100)}pct"] = v
                    cap.append(row)
                row = {"score": "random", "description": "flag vehicles at random"}
                row.update({f"misses_captured_flagging_{int(f_ * 100)}pct": f_ for f_ in fr})
                cap.append(row)
                row = {"score": "oracle", "description": "flag the missed vehicles first"}
                row.update({f"misses_captured_flagging_{int(f_ * 100)}pct": min(1.0, f_ / y.mean()) for f_ in fr})
                cap.append(row)
                write_with_legend(pd.DataFrame(cap), os.path.join(out, f"miss_capture_gt_iou{iou}.csv"), [
                    "flag the given share of ground-truth vehicles, least trusted first, and report the share of all "
                    "missed vehicles that were flagged",
                ])
            summary[f"{study}_iou{iou}"] = {
                "n_rows": int(len(y)), "positive_rate": float(y.mean()),
                "auroc": dict(zip(table["score"], table["auroc[all]"])),
            }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
