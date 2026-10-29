"""All evidence-gate tables, redone with L2_evidence (coarsest level, 3.2 m cells).

Gate = drop the bottom k% by the signal, inside a band of score deciles.
enrichment = FP rate among the removed boxes / FP rate of the band (>1 = better
than removing band members at random). A drop raises OVERALL precision only if
the removed set's FP rate exceeds the overall FP rate, i.e. enrichment must clear
(overall FP rate)/(band FP rate) -- printed as 'need' in each header.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = "opencood/logs/HEAL_m1_based/final_infer_levy/"
b = pd.read_csv(D + "levy_intermediate_fgdecomp_204.8_102.4_epoch1_use_cav2_boxes.csv")
frames = pd.read_csv(D + "levy_intermediate_fgdecomp_204.8_102.4_epoch1_use_cav2_frames.csv")
N_GT = int(frames.n_gt.sum())
import sys
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "L2"
KS = tuple(float(x) / 100 for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "5", "10", "20"]))
b["evidence"] = b[f"{LEVEL}_evidence"]
b["dist_desc"] = -b.dist
b["random"] = np.random.default_rng(0).random(len(b))
b["sdec"] = pd.qcut(b.score, 10, labels=False, duplicates="drop")
GATES = ["evidence", "score", "dist_desc", "random"]
BASE = {t: (b[f"tp{t}"].mean(), b[f"tp{t}"].sum() / N_GT, 1 - b[f"tp{t}"].mean()) for t in (50, 70)}
pd.set_option("display.width", 240)
F = lambda v: f"{v:8.4f}"  # noqa: E731

CELL = {"L0": "0.8 m", "L1": "1.6 m", "L2": "3.2 m"}[LEVEL]
print(f"GATE SIGNAL = {LEVEL}_evidence (Lambda at {CELL} cells), k% = {[100 * k for k in KS]}")
print(f"n boxes {len(b)}, n_gt {N_GT}")
print(f"base: FP50 {BASE[50][2]:.3f} FP70 {BASE[70][2]:.3f} | prec50 {BASE[50][0]:.3f} "
      f"rec50 {BASE[50][1]:.3f} | prec70 {BASE[70][0]:.3f} rec70 {BASE[70][1]:.3f}")

COLS = ["gate", "k%", "n_rm", "rm_dist", "enr_band70", "enr_all70", "TP:FP70", "prec70", "rec70",
        "d_prec70", "enr_band50", "prec50", "rec50", "d_prec50"]


def run(mask, label, ks, stratify=False):
    band = b[mask] if mask is not None else b
    need70 = BASE[70][2] / (1 - band.tp70.mean())
    need50 = BASE[50][2] / (1 - band.tp50.mean())
    print(f"\n{label}: n={len(band)}, score {band.score.min():.3f}-{band.score.max():.3f}, "
          f"FP70 {1 - band.tp70.mean():.3f} ({int((1 - band.tp70).sum())} FPs), "
          f"holds {band.tp70.sum() / b.tp70.sum():.1%} of TP70 | need enrichment "
          f">{need70:.2f}x @0.7, >{need50:.2f}x @0.5 to raise precision"
          + ("  [within distance deciles]" if stratify else ""))
    rows = []
    for k in ks:
        for g in GATES:
            if stratify:
                rm = []
                for _, grp in band.groupby(pd.qcut(band.dist, 10, labels=False, duplicates="drop")):
                    order = np.argsort(grp[g].values, kind="stable")
                    rm += list(grp.index[order[: int(round(k * len(grp)))]])
            else:
                order = np.argsort(band[g].values, kind="stable")
                rm = list(band.index[order[: int(round(k * len(band)))]])
            removed, kept = b.loc[rm], b.drop(index=rm)
            row = {"gate": g, "k%": 100 * k, "n_rm": len(rm), "rm_dist": removed.dist.mean()}
            for t in (50, 70):
                fp_rm = 1 - removed[f"tp{t}"].mean() if len(rm) else np.nan
                tp_rm = int(removed[f"tp{t}"].sum())
                row[f"enr_band{t}"] = fp_rm / (1 - band[f"tp{t}"].mean())
                row[f"enr_all{t}"] = fp_rm / BASE[t][2]
                row[f"TP:FP{t}"] = tp_rm / max(len(rm) - tp_rm, 1)
                row[f"prec{t}"] = kept[f"tp{t}"].mean()
                row[f"rec{t}"] = kept[f"tp{t}"].sum() / N_GT
                row[f"d_prec{t}"] = kept[f"tp{t}"].mean() - BASE[t][0]
            rows.append(row)
    print(pd.DataFrame(rows)[COLS].to_string(index=False, float_format=F))


print("\n" + "=" * 118 + "\n1) GLOBAL gate (all boxes ranked together)")
run(None, "  ALL boxes", KS)
run(None, "  ALL boxes", KS, stratify=True)

print("\n" + "=" * 118 + "\n2) LOW score bands")
for bd in (1, 2, 3):
    run(b.sdec < bd, f"  lowest {bd} decile(s)", KS)
run(b.sdec < 3, "  lowest 3 deciles", KS, stratify=True)

print("\n" + "=" * 118 + "\n3) MIDDLE and HIGH score bands")
run(b.sdec.between(4, 6), "  deciles 4-6", KS)
run(b.sdec.between(4, 6), "  deciles 4-6", KS, stratify=True)
run(b.sdec >= 7, "  deciles 7-9", KS)
run(b.sdec >= 7, "  deciles 7-9", KS, stratify=True)
run(b.sdec == 9, "  decile 9", KS)

print("\n" + "=" * 118 + "\n4) Per-decile AUROC for FP (per level, low -> FP) and breakeven")
rows = []
for d, g in b.groupby("sdec"):
    fp70 = 1 - g.tp70.mean()
    rows.append({
        "decile": d, "score": f"{g.score.min():.2f}-{g.score.max():.2f}", "n_FP70": int((1 - g.tp70).sum()),
        "FP70": fp70, "AUROC70_L0": roc_auc_score(1 - g.tp70, -g.L0_evidence),
        "AUROC70_L1": roc_auc_score(1 - g.tp70, -g.L1_evidence),
        "AUROC70_L2": roc_auc_score(1 - g.tp70, -g.L2_evidence),
        "AUROC70_score": roc_auc_score(1 - g.tp70, -g.score),
        "need_enrich": BASE[70][2] / fp70,
    })
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

print("\n5) Equal-removal: same number of boxes dropped, L2 evidence (in band) vs global score")
print("\n6) enrichment@0.7 by level and k, per band")
for lab, mask in (("all", b.index == b.index), ("d0-2", b.sdec <= 2), ("d4-6", b.sdec.between(4, 6)), ("d7-9", b.sdec >= 7)):
    band = b[mask]
    need = BASE[70][2] / (1 - band.tp70.mean())
    for lv in ("L0", "L1", "L2"):
        line = f"  {lab:<5}{lv}  need {need:5.2f}x |"
        for k in KS:
            rm = band.iloc[np.argsort(band[f"{lv}_evidence"].values, kind="stable")[: int(round(k * len(band)))]]
            line += f"  k={100 * k:4.0f}%: {(1 - rm.tp70.mean()) / (1 - band.tp70.mean()):5.2f}x"
        print(line)

for lab, mask, k in (("deciles 0-2", b.sdec <= 2, 0.5), ("deciles 4-6", b.sdec.between(4, 6), 0.1),
                     ("deciles 7-9", b.sdec >= 7, 0.1)):
    band = b[mask]
    n = int(round(k * len(band)))
    rm_ev = list(band.index[np.argsort(band.evidence.values, kind="stable")[:n]])
    rm_sc = list(b.index[np.argsort(b.score.values, kind="stable")[:n]])
    for name, rm in ((f"{LEVEL} evidence in {lab}", rm_ev), ("global score", rm_sc)):
        kept = b.drop(index=rm)
        print(f"  n={n:6d}  {name:<28} rm_TP70 {b.loc[rm].tp70.mean():.3f}  "
              f"prec70 {kept.tp70.mean():.4f} ({kept.tp70.mean() - BASE[70][0]:+.4f})  "
              f"rec70 {kept.tp70.sum() / N_GT:.4f} ({kept.tp70.sum() / N_GT - BASE[70][1]:+.4f})")
