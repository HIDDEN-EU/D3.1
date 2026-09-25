"""
Re-score predicted boxes with a logistic model on the Levy signals.

The models (the first two are those of ``levy_tpfp_study.py``):

* B (``logistic_B_levy``): the 9 per-level signals (log evidence, log raw
  disagreement, cosine disagreement at levels 0-2) sampled at the BEV cell
  under the box centre; boxes whose centre is seen by >= 2 agents;
* C (``logistic_C_levy_plus_score``): the same 9 signals + the detection score;
  boxes seen by >= 2 agents;
* C_L0 / C_L01: model C restricted to L0 evidence + L0 raw disagreement, or to
  all three signals at levels 0 and 1 (offline, level 2 did not help);
* S (``logistic_S_single_agent``): log evidence at levels 0-2 + the detection
  score; boxes seen by exactly one agent, where disagreement is identically zero;
* M (``logistic_M_mean_evidence``): the level-mean log evidence (levels 0-2
  averaged on the level-0 grid) + the detection score; boxes seen by one agent.

Each is fitted on the validation split to predict a true positive at IoU 0.7,
and at inference the box's confidence becomes that probability. Several models
can be combined (e.g. C and S); a box that no model covers keeps the detector's
score.

Fit once and export the parameters (no GPU needed)::

    python opencood/tools/levy/levy_rescore.py --model C \\
        --model_dir opencood/logs/HEAL_m1_based/final_infer_levy

then pass ``--levy_rescore <model_dir>/box_rescore_params_C.json
<model_dir>/box_rescore_params_S.json`` (model B: ``box_rescore_params.json``)
to ``inference_heter_in_order.py``.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from opencood.tools.levy.levy_collect import _sample
from opencood.utils import common_utils

FEATURES = [f"L{lv}_{k}" for lv in range(3)
            for k in ("log_evidence", "disagreement", "disagreement_cosine")]
EVIDENCE = [f"L{lv}_log_evidence" for lv in range(3)]
# name -> (study name, features, (min, max) observing agents; max None = no limit)
MODELS = {
    "B": ("logistic_B_levy", FEATURES, (2, None)),
    "C": ("logistic_C_levy_plus_score", FEATURES + ["score"], (2, None)),
    "C_L0": ("logistic_C_L0", ["L0_log_evidence", "L0_disagreement", "score"], (2, None)),
    "C_L01": ("logistic_C_L01", FEATURES[:6] + ["score"], (2, None)),
    "S": ("logistic_S_single_agent", EVIDENCE + ["score"], (1, 1)),
    "M": ("logistic_M_mean_evidence", ["fused_log_evidence", "score"], (1, 1)),
}
DISAGREEMENT_EPS = 1e-6
MIN_AGENTS = 2


class _Model:
    def __init__(self, params_path):
        with open(params_path) as f:
            params = json.load(f)
        self.name = params.get("model", os.path.basename(params_path))
        self.features = params["features"]
        self.mean = np.array(params["mean"])
        self.scale = np.array(params["scale"])
        self.coef = np.array(params["coef"])
        self.intercept = float(params["intercept"])
        self.min_agents = int(params.get("min_agents", MIN_AGENTS))
        max_agents = params.get("max_agents")
        self.max_agents = np.inf if max_agents is None else int(max_agents)
        self.n_rescored = 0


class BoxRescorer:
    """Replaces ``pred_score`` with the fitted models' true-positive probability."""

    def __init__(self, params_paths, pc_range):
        if isinstance(params_paths, str):
            params_paths = [params_paths]
        self.models = [_Model(p) for p in params_paths]
        self.pc_range = pc_range
        self.n_kept = 0
        self.n_frames = 0
        self.seconds = 0.0

    @property
    def n_rescored(self):
        return sum(m.n_rescored for m in self.models)

    def features_at(self, uncertainty, xy, det_score, features):
        """(N, F) inputs of ``features`` at ego-frame box centres."""
        per_scale = uncertainty["per_scale"]
        cols = []
        for name in features:
            if name == "score":
                cols.append(det_score.astype(float))
                continue
            level, key = name.split("_", 1)
            bev_map = uncertainty[key] if level == "fused" else per_scale[int(level[1:])][key]
            v = _sample(bev_map, xy, self.pc_range).astype(float)
            if key == "disagreement":
                v = np.log(v + DISAGREEMENT_EPS)
            cols.append(v)
        return np.stack(cols, 1)

    @torch.no_grad()
    def __call__(self, uncertainty, infer_result):
        """New score tensor for ``infer_result``'s boxes (unchanged if none)."""
        pred_box = infer_result["pred_box_tensor"]
        pred_score = infer_result["pred_score"]
        if pred_box is None or uncertainty is None or pred_box.shape[0] == 0:
            return pred_score
        if pred_score.is_cuda:
            torch.cuda.synchronize()  # charge only the re-scoring, not pending GPU work
        start = time.perf_counter()
        xy = common_utils.torch_tensor_to_numpy(pred_box).mean(axis=1)[:, :2]
        det_score = common_utils.torch_tensor_to_numpy(pred_score)
        n_valid = _sample(uncertainty["n_valid"], xy, self.pc_range)
        score = pred_score.clone()
        done = np.zeros(len(xy), dtype=bool)
        for m in self.models:
            X = self.features_at(uncertainty, xy, det_score, m.features)
            usable = ((n_valid >= m.min_agents) & (n_valid <= m.max_agents)
                      & np.isfinite(X).all(1) & ~done)
            if not usable.any():
                continue
            logit = ((X[usable] - m.mean) / m.scale) @ m.coef + m.intercept
            score[torch.from_numpy(usable).to(score.device)] = torch.as_tensor(
                1.0 / (1.0 + np.exp(-logit)), dtype=score.dtype, device=score.device)
            m.n_rescored += int(usable.sum())
            done |= usable
        self.n_kept += int((~done).sum())
        if score.is_cuda:
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - start
        self.n_frames += 1
        return score

    def summary(self):
        total = max(self.n_rescored + self.n_kept, 1)
        parts = [f"{m.name} {m.n_rescored}" for m in self.models]
        return (f"Levy re-scoring: {self.n_rescored} boxes re-scored ({', '.join(parts)}), "
                f"{self.n_kept} kept the detector score "
                f"({100 * self.n_rescored / total:.1f}% re-scored); "
                f"{1000 * self.seconds / max(self.n_frames, 1):.2f} ms/frame")


def fit(model_dir, prefix, use_cavs, iou, model_name="B"):
    """Fit one of ``MODELS`` on the study's box rows and return its parameters."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from opencood.tools.levy.levy_tpfp_study import FUSED, PER_LEVEL, load, transformed

    assert PER_LEVEL == FEATURES
    study_name, features, (lo, hi) = MODELS[model_name]
    boxes, _ = load(model_dir, prefix, use_cavs)
    nv = boxes["fused_n_valid"].to_numpy()
    in_band = (nv >= lo) & (nv <= (np.inf if hi is None else hi))
    if model_name in ("S", "M"):
        # only evidence is defined with one agent
        finite = np.isfinite(boxes[features].to_numpy(dtype=float)).all(1)
    else:
        # exactly the study's overlap rows
        finite = np.isfinite(boxes[PER_LEVEL + FUSED].to_numpy(dtype=float)).all(1)
    df = boxes[in_band & finite]
    X = transformed(df, features).to_numpy()  # the detection score enters as is
    y = df[f"tp{iou}"].to_numpy().astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X, y)
    scaler, logistic = model
    return {
        "model": study_name,
        "fitted_on": f"{prefix}_use_cav{use_cavs}",
        "target": f"true positive at IoU 0.{iou // 10}",
        "n_fit_rows": int(len(y)),
        "true_positive_rate": float(y.mean()),
        "note": "raw disagreement enters as log(value + 1e-6), evidence as log evidence, "
                "score as the detector's probability; applies to boxes whose centre cell "
                "has min_agents..max_agents observers (max null = no limit)",
        "min_agents": lo,
        "max_agents": hi,
        "features": features,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": logistic.coef_[0].tolist(),
        "intercept": float(logistic.intercept_[0]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prefix", default="levy_intermediate_204.8_102.4_val_epoch1")
    p.add_argument("--use_cav", nargs="*", type=int, default=[2, 3, 4])
    p.add_argument("--iou", type=int, default=70)
    p.add_argument("--model", choices=sorted(MODELS), default="B",
                   help="B: 9 Levy signals; C: 9 Levy signals + detection score; C_L0 / "
                   "C_L01: C on L0 evidence + L0 disagreement / on levels 0-1 (all >= 2 "
                   "agents); S: per-level evidence + detection score, "
                   "M: level-mean evidence + detection score (both 1 agent)")
    p.add_argument("--out", default=None, help="default: <model_dir>/box_rescore_params.json "
                   "for model B, box_rescore_params_<model>.json otherwise")
    opt = p.parse_args()
    params = fit(opt.model_dir, opt.prefix, opt.use_cav, opt.iou, opt.model)
    name = "box_rescore_params.json" if opt.model == "B" else f"box_rescore_params_{opt.model}.json"
    out = opt.out or os.path.join(opt.model_dir, name)
    with open(out, "w") as f:
        json.dump(params, f, indent=1)
    std_w = np.array(params["coef"])
    print(f"fitted on {params['n_fit_rows']} boxes (true positive rate "
          f"{params['true_positive_rate']:.3f}); standardised weights (towards true positive):")
    for name, w in zip(params["features"], std_w):
        print(f"  {name:24s} {w:+.4f}")
    print(f"intercept {params['intercept']:+.4f}\nwritten to {out}")


if __name__ == "__main__":
    main()
