"""
Post-training evaluation hook.

After a training run finishes, this runs the accepted count-based occlusion-recall view on the best checkpoint and logs the
found/missed/recall numbers — for occluded objects, both VRU and non-VRU — plus the
found/missed webview galleries into the SAME MLflow run as the training.

It reuses the validated `evaluate_recall_counts` metric unchanged; this module is only the
training-loop + MLflow glue (kept out of `occlusion_eval.py`, which stays MLflow-free).

The eval runs on the training-time *val* slice by default (the clean per-run monitoring
signal), leaving the held-out seed-42 test slice pristine for final sign-off via the
`eval-occlusion-recall` CLI. Failures here are non-fatal: a completed training run is never
lost because its post-eval raised.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from road_object_detection.config import Experiment
from road_object_detection.evaluation import mlflow_eval
from road_object_detection.evaluation import occlusion_eval as oe
from road_object_detection.training.utils import resolve_dataset_yaml


def run_and_log_occlusion_recall(
    experiment: Experiment, run_dir: Path, run_id: str, device,
) -> None:
    """Run the occlusion-recall eval on best.pt and log it into MLflow run `run_id`.

    No-op (with a printed reason) when disabled, when best.pt is missing, or when the
    dataset has no occlusion.csv sidecar. Never raises into the caller's happy path beyond
    what train.py's try/except already guards.
    """
    ec = experiment.evaluation
    if not ec.occlusion_recall:
        print("[post-eval] evaluation.occlusion_recall=false — skipping")
        return

    data = resolve_dataset_yaml(experiment.dataset.data_yaml.resolve())
    class_names = data["names"]
    root = Path(data["path"])
    label_dir = root / "labels" / Path(data["val"]).name
    occ_csv = label_dir / "occlusion.csv"
    if not occ_csv.exists():
        print(f"[post-eval] no occlusion.csv at {occ_csv} — skipping occlusion-recall eval "
              f"(non-tiled dataset?)")
        return

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"[post-eval] best.pt not found at {weights} — skipping")
        return

    seed = experiment.dataset.split_seed
    ratio = experiment.dataset.val_test_ratio
    if ec.eval_on == "test":
        image_paths = oe.derive_test_slice(data, seed, ratio)
    else:
        image_paths = oe.derive_val_slice(data, seed, ratio)
    if ec.max_images:
        image_paths = image_paths[: ec.max_images]
    if not image_paths:
        print("[post-eval] no eval images resolved — skipping")
        return

    print(f"[post-eval] occlusion-recall on {len(image_paths)} '{ec.eval_on}'-slice images "
          f"(conf={ec.conf}) -> resuming MLflow run {run_id}")

    from ultralytics import YOLO
    model = YOLO(str(weights))
    occ_flags = oe.load_occlusion_flags(occ_csv)

    out = oe.evaluate_recall_counts(
        model, image_paths, occ_flags, label_dir, class_names,
        conf_thresholds=ec.conf, nms_iou=ec.nms_iou, imgsz=experiment.model.imgsz,
        device=device, weights_label=str(weights), gallery_seed=seed,
        gallery_n=ec.gallery_n,
    )

    doc = {
        "metadata": {
            "weights": str(weights),
            "dataset": str(experiment.dataset.data_yaml),
            "split": ec.eval_on,
            "eval_images_n": out["_eval_images_n"],
            "seed": seed,
            "conf_thresholds": ec.conf,
            "protocol_commit": out["_protocol_commit"],
            "miss_rate": out["_miss_rate"],
            "stratum_gt_counts": out["_stratum_gt_counts"],
            "caveats": out["_caveats"],
        },
        "results": out["results"],
    }

    out_dir = run_dir / "occlusion_recall"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{weights.stem}_{ec.eval_on}_recall.json"
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    # Galleries are written to the local run dir for inspection but NOT logged to
    # MLflow: they are webview HTML/JS, which MLflow's artifact viewer can't render.
    oe.write_galleries(out["_galleries"], out_dir)

    headline_conf = ec.conf[0] if ec.conf else 0.25
    with mlflow.start_run(run_id=run_id):
        mlflow_eval.log_recall(doc, json_path, headline_conf,
                               artifact_path="occlusion_recall", extra_tags={
            "occ_eval_slice": ec.eval_on,
            "occ_eval_conf": str(headline_conf),
            "occ_eval_images": str(out["_eval_images_n"]),
        })

    r = doc["results"][oe._conf_key(headline_conf)]
    occ = r["occluded"]
    print(f"[post-eval] logged to MLflow. conf={headline_conf} occluded recall: "
          f"all {occ['all_classes']['recall'] * 100:.1f}% "
          f"({occ['all_classes']['detected']}/{occ['all_classes']['total_gt']}), "
          f"VRU {occ['VRU']['recall'] * 100:.1f}% "
          f"({occ['VRU']['detected']}/{occ['VRU']['total_gt']})")
