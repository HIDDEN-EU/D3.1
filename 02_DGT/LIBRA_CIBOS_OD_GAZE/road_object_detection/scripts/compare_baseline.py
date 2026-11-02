"""
Side-by-side comparison: COCO-pretrained predictions (left) vs BDD100K ground truth (right).

Reads the image filenames already produced in output/baseline/, re-runs the model
on the original images to get clean predictions, fetches the matching ground truth
from the BDD100K aggregate JSON, then saves a combined image to output/comparison/.

One important thing to be aware of:
  The baseline model uses COCO class names. BDD100K uses different names for the
  same objects. Approximate mapping:
    COCO "person"    ↔  BDD100K "pedestrian"
    COCO "stop sign" ↔  BDD100K "traffic sign"
  Everything else (car, truck, bus, bicycle, motorcycle, traffic light) matches.
  This means some ground truth boxes will have labels that look slightly different
  from the predictions — that is expected, not a bug.

Usage:
    python scripts/compare_baseline.py

    Optional overrides:
    --baseline   output/baseline          (folder with already-generated baseline images)
    --images     .../bdd100k/images/100k/val
    --gt         .../bdd100k/labels/det_20/det_val.json
    --output     output/comparison
    --conf       0.3
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Colours for predictions (left panel)
PRED_COLOURS = {
    "person": (0, 255, 0),
    "car": (255, 100, 0),
    "truck": (255, 0, 100),
    "bus": (255, 0, 200),
    "motorcycle": (0, 0, 255),
    "bicycle": (0, 150, 255),
    "traffic light": (0, 255, 255),
    "stop sign": (200, 0, 255),
}
DEFAULT_PRED = (180, 180, 180)

# Colours for ground truth (right panel) — use a distinct palette so the two panels
# are clearly different at a glance.
GT_COLOURS = {
    "pedestrian": (0, 255, 0),
    "rider": (0, 200, 100),
    "car": (255, 100, 0),
    "truck": (255, 0, 100),
    "bus": (255, 0, 200),
    "motorcycle": (0, 0, 255),
    "bicycle": (0, 150, 255),
    "traffic light": (0, 255, 255),
    "traffic sign": (200, 0, 255),
}
DEFAULT_GT = (180, 180, 180)


def draw_predictions(frame: np.ndarray, results, model) -> np.ndarray:
    img = frame.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = round(float(box.conf[0]), 2)
        colour = PRED_COLOURS.get(label, DEFAULT_PRED)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            img,
            f"{label} {conf}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
        )
    return img


def draw_ground_truth(frame: np.ndarray, gt_labels: list) -> np.ndarray:
    img = frame.copy()
    for label in gt_labels:
        category = label.get("category", "")
        box = label.get("box2d")
        if box is None:
            continue
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        colour = GT_COLOURS.get(category, DEFAULT_GT)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            img,
            category,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
        )
    return img


def add_panel_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a title bar at the top of a panel image."""
    bar = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("output/baseline"),
        help="Folder containing the baseline output images",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/bdd100k/images/100k/val"),
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=Path("data/bdd100k/labels/det_20/det_val.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("output/comparison"))
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # --- Collect image names from the baseline folder ---
    image_names = [p.name for p in sorted(args.baseline.glob("*.jpg"))]
    if not image_names:
        print(f"No images found in {args.baseline}")
        return
    print(f"Found {len(image_names)} images in baseline folder.")

    # --- Build ground truth lookup: filename -> list of label dicts ---
    print(f"Loading ground truth from {args.gt} ...")
    with open(args.gt) as f:
        gt_data = json.load(f)
    gt_lookup = {entry["name"]: entry.get("labels") or [] for entry in gt_data}

    # --- Load model ---
    print("Loading yolov8s.pt ...")
    model = YOLO("yolov8s.pt")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Running on: {'GPU' if device == 0 else 'CPU'}")

    # --- Generate comparison images ---
    for i, name in enumerate(image_names, 1):
        img_path = args.images / name
        if not img_path.exists():
            print(f"  [{i}] WARNING: original image not found: {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        gt_labels = gt_lookup.get(name, [])

        # Run prediction on the original (clean) image
        results = model.predict(frame, conf=args.conf, verbose=False, device=device)[0]

        left = draw_predictions(frame, results, model)
        right = draw_ground_truth(frame, gt_labels)

        left = add_panel_label(left, f"PREDICTED  (yolov8s COCO)  conf>{args.conf}")
        right = add_panel_label(
            right, f"GROUND TRUTH  (BDD100K)  {len(gt_labels)} objects"
        )

        # Divider line between panels
        divider = np.zeros((left.shape[0], 4, 3), dtype=np.uint8)
        divider[:] = (80, 80, 80)

        combined = np.hstack([left, divider, right])
        out_path = args.output / name
        cv2.imwrite(str(out_path), combined)

        n_pred = len(results.boxes)
        print(f"  [{i}/{len(image_names)}] {name}  pred={n_pred}  gt={len(gt_labels)}")

    print(f"\nDone. Comparison images saved to {args.output}/")
    print("Note: COCO uses 'person' where BDD100K says 'pedestrian', and")
    print("      'stop sign' where BDD100K says 'traffic sign'. That is expected.")


if __name__ == "__main__":
    main()
