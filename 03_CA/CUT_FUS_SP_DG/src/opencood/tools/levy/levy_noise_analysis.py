"""
Does the Levy trust map detect pose-noise misalignment, and does acting on it help?

Reads the CSVs written by ``inference_heter_in_order.py --save_levy`` for a clean
run, a ``--pose_noise`` sweep and an ego-only (``--use_cav [1]``) run of the same
checkpoint, all on the full test split so frame indices line up.

1. AP vs pose noise, for the collaborative and ego-only runs.
2. Frame level -- misalignment detection. Per frame signal (overlap-masked grid
   means): how often it rises from the clean to the noisy version of the *same*
   frame, AUROC separating clean from noisy frames, and Spearman with the
   realized misalignment inside a noise level.
3. Box level -- AUROC of each signal for FP@0.7 on overlap boxes, alone and
   within score deciles, per noise level; Spearman with the per-box pose
   displacement within score deciles.
4. Gating -- drop the collaborator (fall back to ego-only boxes) on frames whose
   signal crosses a threshold set on *clean* frames of the other scene fold, and
   compare AP with random / oracle gating that drops the same number of frames.

Example
    python opencood/tools/levy_noise_analysis.py \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy
"""

import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

FRAME_SIGNAL = re.compile(
    r"^(L\d+|fused)_(log_evidence|disagreement|sigma|disagreement_cosine"
    r"|sigma_cosine|disagreement_decoded)_mean$"
)
BOX_SIGNALS = (
    "L0_disagreement",
    "L1_disagreement",
    "L2_disagreement",
    "L0_disagreement_cosine",
    "L1_disagreement_cosine",
    "L2_disagreement_cosine",
    "L0_disagreement_decoded",
    "L1_log_evidence",
    "fused_disagreement",
    "dist",
)
# Pre-specified from the clean analysis; direction is +1 when a HIGH value means
# "distrust the collaborator".
GATES = {
    "L1_disagreement_cosine_mean": +1,
    "L2_disagreement_cosine_mean": +1,
    "L0_disagreement_cosine_mean": +1,
    "L0_disagreement_mean": +1,
    "L1_disagreement_mean": +1,
    "L2_disagreement_mean": +1,
    "fused_disagreement_mean": +1,
    "L1_log_evidence_mean": -1,
    "score_mean": -1,
    # non-Levy crowding control: L1 cosine tracks n_pred at rho ~0.56 on clean frames
    "n_pred": +1,
}
GATE_QUANTILES = (0.1, 0.25, 0.5)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", default="levy_intermediate_204.8_102.4_epoch1")
    p.add_argument("--levels", default="0,0.2,0.4,0.6")
    p.add_argument("--n_random", type=int, default=50)
    p.add_argument("--out", default=None, help="also dump all results as JSON")
    return p.parse_args()


def load(model_dir, prefix, use_cav, noise):
    tag = f"{prefix}_use_cav{use_cav}"
    if noise:
        tag += f"_noise{noise}_{noise}_0_0"
    frames = pd.read_csv(os.path.join(model_dir, f"{tag}_frames.csv"))
    boxes = pd.read_csv(os.path.join(model_dir, f"{tag}_boxes.csv"))
    per_frame = boxes.groupby("frame").agg(score_mean=("score", "mean"))
    frames = frames.merge(per_frame, on="frame", how="left")
    if "pose_disp" in boxes:
        overlap = boxes[boxes.fused_n_valid >= 2]
        disp = overlap.groupby("frame").pose_disp.mean().rename("misalign")
        frames = frames.merge(disp, on="frame", how="left")
        frames["misalign"] = frames.misalign.fillna(frames.pose_err_xy)
    else:
        frames["misalign"] = 0.0
    return frames.set_index("frame", drop=False), boxes


def average_precision(scores, tp, n_gt):
    """VOC-2010 AP, as in ``eval_utils.calculate_ap``."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(tp)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = np.concatenate([[0.0], ctp / n_gt, [1.0]])
    prec = np.concatenate([[0.0], ctp / np.maximum(ctp + cfp, 1), [0.0]])
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.where(rec[1:] != rec[:-1])[0] + 1
    return float(np.sum((rec[idx] - rec[idx - 1]) * prec[idx]))


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    if len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], s[ok]))


def rho(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).correlation)


class GatedAP:
    """AP after replacing the collaborative boxes of chosen frames by ego-only ones."""

    def __init__(self, collab_boxes, ego_boxes, n_gt):
        self.total = int(n_gt.sum())
        self.collab = {c: collab_boxes[c].values for c in ("frame", "score", "tp50", "tp70")}
        self.ego = {c: ego_boxes[c].values for c in ("frame", "score", "tp50", "tp70")}

    def __call__(self, drop):
        drop = np.asarray(list(drop))
        keep_c = ~np.isin(self.collab["frame"], drop)
        keep_e = np.isin(self.ego["frame"], drop)
        score = np.concatenate([self.collab["score"][keep_c], self.ego["score"][keep_e]])
        return {
            f"AP{t}": average_precision(
                score,
                np.concatenate(
                    [self.collab[f"tp{t}"][keep_c], self.ego[f"tp{t}"][keep_e]]
                ),
                self.total,
            )
            for t in (50, 70)
        }


def scene_folds(frames):
    scenes = sorted(frames.scenario.unique())
    fold = {s: i % 2 for i, s in enumerate(scenes)}
    return frames.scenario.map(fold).values


def main():
    opt = parser()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*constant.*")
    levels = [float(x) for x in opt.levels.split(",")]
    runs = {s: load(opt.model_dir, opt.prefix, 2, s) for s in levels}
    ego_frames, ego_boxes = load(opt.model_dir, opt.prefix, 1, 0)
    clean_frames = runs[0.0][0]
    n_gt = clean_frames.n_gt
    for s, (fr, _) in runs.items():
        assert (fr.n_gt.values == n_gt.values).all(), f"frames misaligned at {s}"
    assert (ego_frames.n_gt.values == n_gt.values).all(), "ego-only frames misaligned"
    out = {}

    # ---------------------------------------------------------------- 1. AP
    print("1) AP vs pose noise (std = pos m = rot deg)")
    print(f"  {'setting':<18}{'AP50':>8}{'AP70':>8}{'misalign m':>12}")
    total = int(n_gt.sum())
    out["ap"] = {}
    for s, (fr, bx) in runs.items():
        ap = {
            t: average_precision(bx.score, bx[f"tp{t}"], total) for t in (50, 70)
        }
        out["ap"][s] = ap
        print(
            f"  {'m1+m4 sigma=' + str(s):<18}{ap[50]:>8.4f}{ap[70]:>8.4f}"
            f"{fr.misalign.mean():>12.3f}"
        )
    ap_ego = {t: average_precision(ego_boxes.score, ego_boxes[f"tp{t}"], total) for t in (50, 70)}
    out["ap"]["ego_only"] = ap_ego
    print(f"  {'ego only (m1)':<18}{ap_ego[50]:>8.4f}{ap_ego[70]:>8.4f}")

    # --------------------------------------------------- 2. frame detection
    signals = [c for c in clean_frames.columns if FRAME_SIGNAL.match(c)]
    signals += ["score_mean", "n_pred"]
    print("\n2) Frame-level misalignment detection (overlap-masked grid means)")
    print("   up%   = share of frames whose signal rises vs the same clean frame")
    print("   AUROC = clean vs noisy frames, unpaired (>0.5: higher under noise)")
    print("   rho   = Spearman with realized misalignment within the noise level;")
    print("   rhoD  = same, on the paired change (removes scene content)")
    out["frames"] = {}
    for s in levels[1:]:
        fr = runs[s][0]
        rows = []
        for sig in signals:
            clean = clean_frames[sig].values
            noisy = fr[sig].values
            delta = noisy - clean
            ok = ~(np.isnan(clean) | np.isnan(noisy))
            y = np.r_[np.zeros(ok.sum()), np.ones(ok.sum())]
            rows.append(
                {
                    "signal": sig,
                    "up%": 100 * float(np.mean(delta[ok] > 0)),
                    "rel_change%": 100
                    * float(np.nanmedian(delta) / (abs(np.nanmedian(clean)) + 1e-12)),
                    "AUROC": auroc(y, np.r_[clean[ok], noisy[ok]]),
                    "rho": rho(noisy, fr.misalign),
                    "rhoD": rho(delta, fr.misalign),
                }
            )
        table = pd.DataFrame(rows).sort_values("AUROC", key=lambda a: -abs(a - 0.5))
        out["frames"][s] = table.to_dict("records")
        print(f"\n  sigma={s}  (mean misalignment {fr.misalign.mean():.2f} m)")
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # -------------------------------------------------------- 3. box level
    print("\n3) Box level, overlap boxes: AUROC for FP@0.7 (signal oriented so that")
    print("   >0.5 means HIGH -> FP), alone and averaged within score deciles;")
    print("   rho_disp = Spearman with per-box pose displacement within score deciles")
    out["boxes"] = {}
    for s in levels:
        bx = runs[s][1]
        ov = bx[bx.fused_n_valid >= 2].copy()
        ov["fp70"] = 1 - ov.tp70
        ov["neg_score"] = -ov.score
        ov["sbin"] = pd.qcut(ov.score, 10, labels=False, duplicates="drop")
        rows = []
        cols = ["neg_score", *BOX_SIGNALS] + (["pose_disp"] if s else [])
        for sig in cols:
            within, weights, disp = [], [], []
            for _, g in ov.groupby("sbin"):
                a = auroc(g.fp70, g[sig])
                if not np.isnan(a):
                    within.append(a)
                    weights.append(len(g))
                if s and sig != "pose_disp":
                    disp.append(rho(g[sig], g.pose_disp))
            rows.append(
                {
                    "signal": sig,
                    "AUROC": auroc(ov.fp70, ov[sig]),
                    "AUROC|score": float(np.average(within, weights=weights))
                    if within
                    else np.nan,
                    "rho_disp": float(np.nanmean(disp)) if disp else np.nan,
                }
            )
        table = pd.DataFrame(rows)
        out["boxes"][s] = table.to_dict("records")
        print(
            f"\n  sigma={s}  n={len(ov)}  FP70 rate={ov.fp70.mean():.3f}"
        )
        print(table.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # ------------------------------------------------------------ 4. gating
    print("\n4) Gating: fall back to ego-only on frames past a threshold taken from")
    print("   CLEAN frames of the other scene fold (quantile q). Random / oracle")
    print("   (largest realized misalignment) drop the same number of frames.")
    rng = np.random.default_rng(0)
    mixed_sigma = rng.choice(levels, size=len(n_gt))
    settings = {f"sigma={s}": np.full(len(n_gt), s) for s in levels}
    settings["mixed"] = mixed_sigma
    folds = scene_folds(clean_frames)
    out["gating"] = {}
    for name, sigma_per_frame in settings.items():
        frame_ids = clean_frames.frame.values
        fr = pd.concat(
            [runs[s][0].loc[frame_ids[sigma_per_frame == s]] for s in levels]
        ).loc[frame_ids]
        bx = pd.concat(
            [
                runs[s][1][runs[s][1].frame.isin(frame_ids[sigma_per_frame == s])]
                for s in levels
            ]
        )
        gate = GatedAP(bx, ego_boxes, n_gt)
        base = gate([])
        print(
            f"\n  {name}: collaborate always AP50 {base['AP50']:.4f} AP70 "
            f"{base['AP70']:.4f} | ego only AP50 {ap_ego[50]:.4f} AP70 {ap_ego[70]:.4f}"
        )
        print(
            f"  {'gate':<30}{'q':>5}{'drop%':>7}{'AP70':>8}{'rnd':>8}{'oracle':>8}"
            f"{'AP50':>8}{'rnd':>8}{'oracle':>8}"
        )
        rows = []
        for sig, sign in GATES.items():
            for q in GATE_QUANTILES:
                drop = np.zeros(len(fr), dtype=bool)
                for k in (0, 1):
                    ref = sign * clean_frames[sig].values[folds != k]
                    thr = np.nanquantile(ref, 1 - q)
                    val = sign * fr[sig].values
                    drop |= (folds == k) & (np.nan_to_num(val, nan=-np.inf) > thr)
                n_drop = int(drop.sum())
                res = gate(frame_ids[drop])
                rnd = [gate(rng.choice(frame_ids, n_drop, replace=False)) for _ in range(opt.n_random)]
                oracle = gate(frame_ids[np.argsort(-fr.misalign.values)[:n_drop]])
                row = {
                    "gate": sig,
                    "q": q,
                    "drop%": 100 * n_drop / len(fr),
                    "AP70": res["AP70"],
                    "AP70_rnd": float(np.mean([r["AP70"] for r in rnd])),
                    "AP70_rnd_sd": float(np.std([r["AP70"] for r in rnd])),
                    "AP70_oracle": oracle["AP70"],
                    "AP50": res["AP50"],
                    "AP50_rnd": float(np.mean([r["AP50"] for r in rnd])),
                    "AP50_oracle": oracle["AP50"],
                }
                rows.append(row)
                print(
                    f"  {sig:<30}{q:>5.2f}{row['drop%']:>7.1f}{row['AP70']:>8.4f}"
                    f"{row['AP70_rnd']:>8.4f}{row['AP70_oracle']:>8.4f}"
                    f"{row['AP50']:>8.4f}{row['AP50_rnd']:>8.4f}{row['AP50_oracle']:>8.4f}"
                )
        out["gating"][name] = {"always": base, "rows": rows}

    if opt.out:
        with open(opt.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nResults written to {opt.out}")


if __name__ == "__main__":
    main()
