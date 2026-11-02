"""
CLI: evaluate-model — occlusion-aware detection evaluation (recall + mAP).

One command, both views over the occlusion groups (visible / occluded / truncated, plus
the VRU rollup), on the held-out seed-42 test slice by default:

  - recall view  — found vs missed GT counts + recall, at conf 0.25 (headline) + 0.001.
  - mAP view     — precision-aware occlusion-stratified mAP@0.5 / mAP@0.5:0.95.

Runs inference (model.predict) only — no training. Results are logged to MLflow as
a single run named '<model-run>_<slice>' (e.g. '<exp>_test'): if MLFLOW_TRACKING_URI is not
set, the run goes to a local ./mlruns store. The full
per-group tables ship as JSON artifacts; the found/missed webview galleries are written to
the output dir (HTML, which MLflow can't render).

NOTE: currently two inference passes (one per view) to keep the two contract-validated
metric functions byte-identical; a single-pass refactor is a planned follow-up.

Usage
-----
evaluate-model \
    --weights   runs/train/<exp>/weights/best.pt \
    --data-yaml data/bdd100k_r_v030_mz16/data.yaml

# verify an existing artifact's schema (recall or mAP, auto-detected) and exit:
evaluate-model --verify-schema output/occlusion_eval/best_val_recall.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from road_object_detection.evaluation import occlusion_eval as oe

WEBVIEW_SRC = Path(__file__).resolve().parent / "webview"


def _default_run_base(weights: Path) -> str:
    """MLflow run base from the weights path: runs/train/<exp>/weights/best.pt -> '<exp>'."""
    p = weights.resolve()
    return p.parent.parent.name if p.parent.name == "weights" else p.stem


def _verify_schema(path: Path) -> None:
    """Auto-detect recall vs mAP artifact and verify its schema."""
    doc = json.loads(path.read_text())
    sample = next(iter(next(iter(doc.get("results", {}).values()), {}).values()), {})
    is_map = "mAP50" in sample
    problems = oe.verify_schema(doc) if is_map else oe.verify_recall_schema(doc)
    kind = "mAP" if is_map else "recall"
    if problems:
        print(f"[eval] SCHEMA FAIL ({kind}, {len(problems)}):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"[eval] schema OK ({kind}): {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Occlusion-aware model evaluation (recall + mAP).")
    ap.add_argument("--verify-schema", type=Path, metavar="JSON",
                    help="verify an existing artifact's schema (recall or mAP) and exit")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--data-yaml", type=Path,
                    help="generated dataset data.yaml (post-merge, nc=11)")
    ap.add_argument("--image-list", type=Path,
                    help="optional newline-separated image paths; default derives the test slice")
    ap.add_argument("--split", default="val", help="split label used in output filenames")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-test-ratio", type=float, default=0.3)
    ap.add_argument("--conf", type=float, nargs="+", default=oe.RECALL_CONF_THRESHOLDS,
                    help="recall-view confidence thresholds (first is the headline). Default: 0.25 0.001")
    ap.add_argument("--map-conf", type=float, default=0.001,
                    help="mAP-view inference conf (val-style, near-zero). Default: 0.001")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="inference chunk size (bounds GPU mem)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--gallery-n", type=int, default=oe.GALLERY_N, help="max images per gallery")
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap eval to the first N slice images (smoke testing only)")
    ap.add_argument("--output", type=Path, default=Path("output/occlusion_eval"))
    ap.add_argument("--run-name", default=None,
                    help="override the MLflow run name (a '_<slice>' suffix is ensured)")
    ap.add_argument("--experiment", default="road-object-detection",
                    help="MLflow experiment to log under (default: road-object-detection)")
    args = ap.parse_args()

    if args.verify_schema is not None:
        _verify_schema(args.verify_schema)
        return

    if not args.weights or not args.data_yaml:
        ap.error("--weights and --data-yaml are required (unless --verify-schema)")
    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}")

    # MLflow is mandatory — refuse to run if we can't record the result.
    # find_dotenv(usecwd=True) searches from the working directory, so .env is found
    # regardless of where the installed console script lives (a bare load_dotenv()
    # searches from the script's install location and misses the repo .env).
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

    data = yaml.safe_load(args.data_yaml.read_text())
    if "class_merges" in data:
        sys.exit(f"{args.data_yaml} is a recipe (has class_merges); pass the generated "
                 f"post-merge data.yaml at <recipe.path>/data.yaml instead.")
    class_names = data["names"]
    if len(class_names) != 11:
        print(f"[eval] WARNING: expected 11 classes (conservative merge), got {len(class_names)}")

    root = Path(data["path"])
    label_dir = root / "labels" / Path(data["val"]).name
    occ_csv = label_dir / "occlusion.csv"
    if not occ_csv.exists():
        sys.exit(f"occlusion.csv not found at {occ_csv}")

    if args.image_list:
        image_paths = [Path(p) for p in args.image_list.read_text().split() if p.strip()]
        slice_label = args.split
    else:
        image_paths = oe.derive_test_slice(data, args.seed, args.val_test_ratio)
        slice_label = "test"
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        sys.exit("no eval images resolved")
    print(f"[eval] weights={args.weights}  images={len(image_paths)}  slice={slice_label}  "
          f"recall-conf={args.conf}  map-conf={args.map_conf}  nms_iou={args.nms_iou}")

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    occ_flags = oe.load_occlusion_flags(occ_csv)

    # --- recall view (pass 1) ---
    recall_out = oe.evaluate_recall_counts(
        model, image_paths, occ_flags, label_dir, class_names,
        conf_thresholds=args.conf, nms_iou=args.nms_iou, imgsz=args.imgsz,
        device=args.device, weights_label=str(args.weights), batch_size=args.batch,
        gallery_seed=args.seed, gallery_n=args.gallery_n,
    )
    # --- mAP view (pass 2) ---
    map_out = oe.evaluate(
        model, image_paths, occ_flags, label_dir, class_names,
        conf=args.map_conf, nms_iou=args.nms_iou, imgsz=args.imgsz,
        device=args.device, batch_size=args.batch,
    )

    recall_doc = {
        "metadata": {
            "weights": str(args.weights), "dataset": str(args.data_yaml),
            "split": args.split, "eval_images_n": recall_out["_eval_images_n"],
            "seed": args.seed, "conf_thresholds": args.conf,
            "protocol_commit": recall_out["_protocol_commit"],
            "miss_rate": recall_out["_miss_rate"],
            "stratum_gt_counts": recall_out["_stratum_gt_counts"],
            "caveats": recall_out["_caveats"],
        },
        "results": recall_out["results"],
    }
    map_doc = {
        "metadata": {
            "weights": str(args.weights), "dataset": str(args.data_yaml),
            "split": args.split, "eval_images_n": map_out["_eval_images_n"],
            "seed": args.seed, "iou_thresholds": oe.IOU_THRESHOLDS,
            "protocol_commit": map_out["_protocol_commit"], "miss_rate": map_out["_miss_rate"],
            "conf": args.map_conf, "nms_iou": args.nms_iou,
            "stratum_gt_counts": map_out["_stratum_gt_counts"],
        },
        "results": map_out["results"],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    recall_path = args.output / f"{args.weights.stem}_{args.split}_recall.json"
    map_path = args.output / f"{args.weights.stem}_{args.split}_strata.json"
    recall_path.write_text(json.dumps(recall_doc, indent=2, sort_keys=True) + "\n")
    map_path.write_text(json.dumps(map_doc, indent=2, sort_keys=True) + "\n")
    oe.write_galleries(recall_out["_galleries"], args.output, WEBVIEW_SRC)

    # --- headline lines ---
    headline_conf = args.conf[0] if args.conf else 0.25
    ck = oe._conf_key(headline_conf)
    print(f"\n[eval] RECALL (conf={headline_conf}):")
    for g in oe.STRATA:
        for scope in ("all_classes", "VRU"):
            c = recall_doc["results"][ck][g][scope]
            lc = "  [low_confidence]" if c["low_confidence"] else ""
            print(f"[eval]   {g:9s} {scope:11s}: {c['total_gt']:>6d} total — found "
                  f"{c['detected']:>6d} ({c['recall'] * 100:5.1f}%), missed {c['missed']:>6d}{lc}")
    print(f"[eval] mAP@0.5 (all): overall {map_doc['results']['overall']['all_classes']['mAP50']:.3f}  "
          f"visible {map_doc['results']['visible']['all_classes']['mAP50']:.3f}  "
          f"occluded {map_doc['results']['occluded']['all_classes']['mAP50']:.3f}")

    # --- log everything to MLflow (single run) ---
    import mlflow

    from road_object_detection.evaluation import mlflow_eval

    run_name = args.run_name or _default_run_base(args.weights)
    if not run_name.endswith(f"_{slice_label}"):
        run_name += f"_{slice_label}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow_eval.log_recall(recall_doc, recall_path, headline_conf,
                               artifact_path="occlusion_eval", extra_tags={
            "occ_eval_slice": slice_label,
            "occ_eval_conf": str(headline_conf),
            "occ_eval_images": str(recall_out["_eval_images_n"]),
            "occ_eval_weights": str(args.weights),
        })
        mlflow_eval.log_map(map_doc, map_path, artifact_path="occlusion_eval")
        # Per-class PR curves, derived from the mAP pass's detections (no extra inference).
        # Overlayable across runs in Model Metrics. Non-fatal: never discard the logged eval.
        try:
            n = mlflow_eval.log_pr_curves(map_out["_pr_curves"])
            print(f"[eval] logged per-class PR curves for {n} classes (pr/*, ap50/*, ap5095/*)")
        except Exception as e:  # noqa: BLE001 — PR curves are a bonus, not the eval
            print(f"[eval] PR-curve logging failed (non-fatal): {e}")
    print(f"\n[eval] logged MLflow run '{run_name}' (experiment '{args.experiment}')")
    print(f"[eval] -> {recall_path}\n[eval] -> {map_path}")


if __name__ == "__main__":
    main()
