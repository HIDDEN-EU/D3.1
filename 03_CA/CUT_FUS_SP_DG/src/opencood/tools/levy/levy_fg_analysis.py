"""Evidence / disagreement built on the foreground estimators themselves.

within-agent : each agent's own S across the 3 pyramid levels (no other agent)
cross-agent  : p(1-p)(S_ego - S_col)^2, the scalar analogue of the feature tr Sigma
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/home/giorgos/projects/PHCP/opencood/tools")
from levy_noise_analysis import auroc, average_precision, rho, scene_folds  # noqa: E402

D = "opencood/logs/HEAL_m1_based/final_infer_levy/"
fg = pd.read_csv(D + "levy_intermediate_fgdecomp_204.8_102.4_epoch1_use_cav2_boxes.csv")
frames = pd.read_csv(D + "levy_intermediate_fgdecomp_204.8_102.4_epoch1_use_cav2_frames.csv")
b = fg
b = b.merge(frames[["frame", "scenario"]], on="frame", how="left")
b["fp70"] = 1 - b.tp70
b["fp50"] = 1 - b.tp50
pd.set_option("display.width", 220)

S = {a: np.c_[[b[f"L{lv}_fg_{a}"].values for lv in (0, 1, 2)]].T for a in ("ego", "col")}
for a in ("ego", "col"):
    b[f"{a}_fg_mean"] = S[a].mean(1)
    b[f"{a}_fg_min"] = S[a].min(1)
    b[f"{a}_fg_max"] = S[a].max(1)
    b[f"{a}_fg_std"] = S[a].std(1)  # within-agent disagreement across levels
    b[f"{a}_fg_range"] = S[a].max(1) - S[a].min(1)
    b[f"{a}_fg_d01"] = np.abs(S[a][:, 0] - S[a][:, 1])
    b[f"{a}_fg_d12"] = np.abs(S[a][:, 1] - S[a][:, 2])
    # coefficient of variation: scale-free spread, the closest analogue of the
    # cosine channel's normalisation
    b[f"{a}_fg_cv"] = b[f"{a}_fg_std"] / np.maximum(b[f"{a}_fg_mean"], 1e-6)
# cross-agent, on probabilities instead of features
for lv in (0, 1, 2):
    se, sc = b[f"L{lv}_fg_ego"], b[f"L{lv}_fg_col"]
    p = np.exp(sc) / (np.exp(se) + np.exp(sc))
    both = ((se > 0) & (sc > 0)).astype(float)
    b[f"L{lv}_probdis"] = both * p * (1 - p) * (se - sc) ** 2
    b[f"L{lv}_absdiff"] = both * (se - sc).abs()
for lv in (0, 1, 2):
    se, sc = b[f"L{lv}_fg_ego"], b[f"L{lv}_fg_col"]
    b[f"L{lv}_p_col"] = np.exp(sc) / (np.exp(se) + np.exp(sc)) * ((se > 0) & (sc > 0))
b["probdis_mean"] = b[[f"L{lv}_probdis" for lv in (0, 1, 2)]].mean(1)

WITHIN = [
    "ego_fg_mean", "ego_fg_min", "ego_fg_std", "ego_fg_range", "ego_fg_cv",
    "ego_fg_d01", "ego_fg_d12", "col_fg_mean", "col_fg_std", "col_fg_cv",
]
CROSS_PROB = ["L0_probdis", "L1_probdis", "L2_probdis", "probdis_mean", "L1_absdiff"]
FEATURE = [
    "L0_disagreement", "L1_disagreement", "L1_disagreement_cosine",
    "L2_disagreement_cosine", "L0_disagreement_decoded", "L1_log_evidence", "L1_p_col",
]


def table(d, cols, label):
    d = d.copy()
    d["sbin"] = pd.qcut(d.score, 10, labels=False, duplicates="drop")
    rows = []
    for c in cols:
        if c not in d:
            continue
        per = {70: [], 50: []}
        for _, g in d.groupby("sbin"):
            for t in (70, 50):
                a = auroc(g[f"fp{t}"], g[c])
                if not np.isnan(a):
                    per[t].append((a, len(g)))
        rows.append({
            "signal": c,
            "rho_score": rho(d[c], d.score),
            "AUROC_fp70": auroc(d.fp70, d[c]),
            "fp70|score": np.average([a for a, _ in per[70]], weights=[n for _, n in per[70]]),
            "fp50|score": np.average([a for a, _ in per[50]], weights=[n for _, n in per[50]]),
        })
    print(f"\n{label}  (n={len(d)}, FP70 {d.fp70.mean():.3f})")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:7.3f}"))


print("=" * 100)
print("A) AUROC for FP (>0.5 means HIGH value -> FP), alone and within score deciles")
print("   'neg_score' = the detector's own score, negated, for reference")
b["neg_score"] = -b.score
allb = b
ov = b[b.fused_n_valid >= 2]
for d, lab in ((allb, "ALL boxes"), (ov, "OVERLAP boxes only")):
    table(d, ["neg_score"] + WITHIN, f"{lab} -- within-agent (own foreground, across levels)")
    table(d, CROSS_PROB, f"{lab} -- cross-agent on probabilities")
    table(d, FEATURE, f"{lab} -- cross-agent on features (for reference)")

print("\n" + "=" * 100)
print("B) Held-out re-ranking (2 scene folds): AP70 of 1-P(FP) from a logistic on")
print("   [score] vs [score + extra]. AP70 of the raw detector score is the baseline.")
n_gt_total = int(frames.n_gt.sum())
folds = scene_folds(b.rename(columns={"scenario": "scenario"}))
b = b.assign(fold=folds)
SETS = {
    "score only": [],
    "+ ego own fg (mean,min,std)": ["ego_fg_mean", "ego_fg_min", "ego_fg_std"],
    "+ ego own fg spread only": ["ego_fg_std", "ego_fg_range", "ego_fg_cv"],
    "+ both agents own fg": ["ego_fg_mean", "ego_fg_min", "ego_fg_std", "col_fg_mean", "col_fg_std"],
    "+ cross-agent probdis": ["L0_probdis", "L1_probdis", "L2_probdis"],
    "+ feature cosine dis": ["L1_disagreement_cosine", "L2_disagreement_cosine"],
    "+ feature raw dis": ["L0_disagreement", "L1_disagreement"],
    "+ camera weight p_col": ["L1_p_col", "L0_p_col"],
    "+ dist": ["dist"],
}
base_ap = average_precision(b.score.values, b.tp70.values, n_gt_total)
print(f"  raw detector score: AP70 {base_ap:.4f}   (n_gt {n_gt_total})")
print(f"  {'features':<32}{'AP70':>8}{'d_AP70':>9}{'AUROC':>8}")
for name, extra in SETS.items():
    cols = ["score"] + extra
    pred = np.zeros(len(b))
    for k in (0, 1):
        tr, te = b[b.fold != k], b[b.fold == k]
        X = tr[cols].values
        mu, sd = X.mean(0), X.std(0) + 1e-9
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit((X - mu) / sd, tr.tp70.values)
        pred[(b.fold == k).values] = model.predict_proba((te[cols].values - mu) / sd)[:, 1]
    ap = average_precision(pred, b.tp70.values, n_gt_total)
    print(f"  {name:<32}{ap:>8.4f}{ap - base_ap:>+9.4f}{auroc(b.tp70, pred):>8.3f}")

print("\n" + "=" * 100)
print("C) Where the within-agent spread is large, what happens to precision?")
q = pd.qcut(b.ego_fg_std, 5, labels=False)
print(b.groupby(q).agg(fp70=("fp70", "mean"), fp50=("fp50", "mean"), score=("score", "mean"),
                       ego_fg_mean=("ego_fg_mean", "mean"), dist=("dist", "mean"),
                       n=("fp70", "size")).round(3).to_string())
print("\n   ego_fg_mean quintiles:")
q2 = pd.qcut(b.ego_fg_mean, 5, labels=False)
print(b.groupby(q2).agg(fp70=("fp70", "mean"), score=("score", "mean"), std=("ego_fg_std", "mean"),
                        n=("fp70", "size")).round(3).to_string())
print("\n   correlation of the detector score with each own-foreground summary:")
for c in ["ego_fg_mean", "ego_fg_min", "ego_fg_max", "ego_fg_std", "ego_fg_cv", "col_fg_mean"]:
    print(f"     rho(score, {c:<12}) = {rho(b.score, b[c]):+.3f}")
