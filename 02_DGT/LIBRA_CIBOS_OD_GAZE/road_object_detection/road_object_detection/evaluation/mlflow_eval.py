"""
MLflow logging for occlusion-aware evaluation (recall + mAP views).

Kept separate from `occlusion_eval.py` so that module stays MLflow-free (the metric
itself is MLflow-agnostic; this is integration glue). Shared by the post-training hook
(`training/post_eval.py`, recall view on the val slice) and the `evaluate-model` CLI
(both views on the test slice) so they never drift in what they log.

Scalar naming (all under the `occ/` prefix, filterable in the MLflow UI):
  recall view: occ/recall_<group>_<scope>, occ/found_*, occ/missed_*, occ/total_*,
               occ/gap_visible_minus_occluded_<scope>
  mAP view:    occ/map50_<group>_<scope>, occ/map5095_<group>_<scope>,
               occ/gap_map50_visible_minus_occluded_<scope>
  PR curves:   pr/<class> (stepped, step = recall × 1000), ap50/<class>, ap5095/<class>
where <group> ∈ {occluded, visible, overall, truncated} and <scope> ∈ {all, vru}.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from road_object_detection.evaluation import occlusion_eval as oe

# Occlusion groups emitted as scalars (the full table still ships as the JSON artifact).
_GROUPS = ["occluded", "visible", "overall", "truncated"]
_SCOPES = (("all_classes", "all"), ("VRU", "vru"))


def recall_scalars(doc: dict, headline_conf: float) -> dict:
    """Flatten the recall view's headline-conf cells into {metric_name: value}."""
    res = doc["results"].get(oe._conf_key(headline_conf), {})
    out: dict = {}
    for g in _GROUPS:
        for scope, tag in _SCOPES:
            c = res[g][scope]
            out[f"occ/recall_{g}_{tag}"] = c["recall"]
            out[f"occ/found_{g}_{tag}"] = c["detected"]
            out[f"occ/missed_{g}_{tag}"] = c["missed"]
            out[f"occ/total_{g}_{tag}"] = c["total_gt"]
    for scope, tag in _SCOPES:
        gap = res["visible"][scope]["recall"] - res["occluded"][scope]["recall"]
        out[f"occ/gap_visible_minus_occluded_{tag}"] = round(gap, 6)
    return out


def map_scalars(doc: dict) -> dict:
    """Flatten the mAP view's cells into {metric_name: value}."""
    res = doc["results"]
    out: dict = {}
    for g in _GROUPS:
        for scope, tag in _SCOPES:
            c = res[g][scope]
            out[f"occ/map50_{g}_{tag}"] = c["mAP50"]
            out[f"occ/map5095_{g}_{tag}"] = c["mAP50_95"]
    for scope, tag in _SCOPES:
        gap = res["visible"][scope]["mAP50"] - res["occluded"][scope]["mAP50"]
        out[f"occ/gap_map50_visible_minus_occluded_{tag}"] = round(gap, 6)
    return out


def log_recall(doc: dict, json_path: Path, headline_conf: float, *,
               artifact_path: str = "occlusion_eval", extra_tags: dict | None = None) -> None:
    """Log recall scalars + the recall JSON into the CURRENT active MLflow run.

    Only the JSON is logged (renders in MLflow); the webview galleries are HTML/JS that
    MLflow's viewer can't render, so they stay on disk in the output dir. The caller owns
    the run context (resume-by-id for the hook, a fresh run for the CLI).
    """
    if extra_tags:
        mlflow.set_tags(extra_tags)
    mlflow.log_metrics(recall_scalars(doc, headline_conf))
    mlflow.log_artifact(str(json_path), artifact_path=artifact_path)


def log_map(doc: dict, json_path: Path, *, artifact_path: str = "occlusion_eval") -> None:
    """Log mAP scalars + the mAP JSON into the CURRENT active MLflow run."""
    mlflow.log_metrics(map_scalars(doc))
    mlflow.log_artifact(str(json_path), artifact_path=artifact_path)


def log_pr_curves(pr_curves: dict) -> int:
    """Log per-class PR curves (overlayable in Model Metrics) into the CURRENT run.

    `pr_curves` is the mAP pass's `_pr_curves` (from `occlusion_eval.evaluate`): a mapping
    class_name -> {"recall": [...], "precision": [...], "ap50": float, "ap5095": float}.
    NO extra inference — the points are derived from detections the mAP pass already built.

    Precision is logged as a stepped metric `pr/<class>` keyed by `step = int(recall * 1000)`
    — MLflow's only natively-overlayable curve form, so the same class compares across runs
    (the x-axis shows as "step", = recall × 1000). Per-class `ap50/<class>` / `ap5095/<class>`
    scalars ship alongside. These use the repo's overall-stratum IoU-0.5 matching, consistent
    with the `occ/*` mAP. Returns the number of classes logged.
    """
    for name, c in pr_curves.items():
        key = str(name).replace(" ", "_")
        for r, p in zip(c["recall"], c["precision"]):
            mlflow.log_metric(f"pr/{key}", float(p), step=int(round(r * 1000)))
        mlflow.log_metric(f"ap50/{key}", float(c["ap50"]))
        mlflow.log_metric(f"ap5095/{key}", float(c["ap5095"]))
    return len(pr_curves)
