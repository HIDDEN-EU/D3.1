"""
Tile dataset visual checker.

Samples 12 tiles from a dataset produced by tile-dataset, runs yolo11s.pt,
and saves annotated images to output/verify_tiles/ showing:

  - Ground-truth boxes  (green)   — from the re-projected YOLO label files
  - Model detections    (blue)    — from yolo11s.pt (COCO-pretrained)

Use this to visually confirm that label re-projection is correct and that
the model sees sensible content in the tiles.

Usage
-----
verify-tiles --data-yaml data/bdd100k_tiled_4/data.yaml
verify-tiles --data-yaml data/bdd100k_tiled_9/data.yaml --split train --n 20
verify-tiles --data-yaml data/bdd100k_tiled_4/data.yaml --weights runs/train/exp/weights/best.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

GT_COLOUR  = (0,   200, 0)    # green  — ground truth
DET_COLOUR = (200, 0,   0)    # blue   — model detection
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def _draw_gt(img, label_path: Path, class_names: dict[int, str]):
    """Draw ground-truth boxes (green) from a YOLO label file."""
    h, w = img.shape[:2]
    if not label_path.exists():
        return img
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = (float(v) for v in parts[1:])
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        label = class_names.get(cls, str(cls))
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOUR, 2)
        cv2.putText(img, f"GT:{label}", (x1, max(y1 - 4, 10)),
                    FONT, 0.45, GT_COLOUR, 1)
    return img


def _draw_detections(img, results, model):
    """Draw model detections (blue) with COCO class name + confidence."""
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf  = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), DET_COLOUR, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, min(y2 + 14, img.shape[0] - 2)),
                    FONT, 0.45, DET_COLOUR, 1)
    return img


def cmd_verify(args: argparse.Namespace) -> None:
    if not args.data_yaml.exists():
        print(f"ERROR: {args.data_yaml} not found"); sys.exit(1)
    if not args.weights.exists():
        print(f"ERROR: weights not found at {args.weights}"); sys.exit(1)

    cfg         = yaml.safe_load(args.data_yaml.read_text())
    root        = Path(cfg["path"])
    class_names = {int(k): v for k, v in cfg["names"].items()}
    img_dir     = root / f"images/{args.split}"
    lbl_dir     = root / f"labels/{args.split}"

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found"); sys.exit(1)

    all_images = sorted(img_dir.glob("*.jpg"))
    if not all_images:
        print(f"ERROR: no .jpg tiles found in {img_dir}"); sys.exit(1)

    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n, len(all_images)))

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\nverify-tiles")
    print(f"  dataset : {root}")
    print(f"  split   : {args.split}  ({len(all_images):,} tiles total)")
    print(f"  sample  : {len(sample)}")
    print(f"  weights : {args.weights}")
    print(f"  output  : {args.output}\n")
    print(f"  Legend  — green: ground truth   blue: model detections\n")

    model = YOLO(str(args.weights))

    for i, img_path in enumerate(sample, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i}/{len(sample)}] SKIP (unreadable): {img_path.name}")
            continue

        results = model(img, conf=args.conf, verbose=False)[0]

        _draw_gt(img, lbl_dir / f"{img_path.stem}.txt", class_names)
        _draw_detections(img, results, model)

        # Tile name in top-left corner
        cv2.putText(img, img_path.stem, (6, 16), FONT, 0.4, (255, 255, 255), 1)

        out_path = args.output / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"  [{i}/{len(sample)}] {img_path.name}  "
              f"gt={sum(1 for _ in (lbl_dir / f'{img_path.stem}.txt').open() if _.strip()) if (lbl_dir / f'{img_path.stem}.txt').exists() else 0}  "
              f"det={len(results.boxes)}  →  {out_path.name}")

    print(f"\nDone. Open {args.output}/ to review the images.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="verify-tiles",
        description=(
            "Visual quality check for a tiled dataset. "
            "Saves annotated tile images (GT=green, detections=blue) to --output."
        ),
    )
    parser.add_argument(
        "--data-yaml", type=Path, required=True,
        help="data.yaml inside the tiled dataset directory",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("yolo11s.pt"),
        help="YOLO weights (default: ./yolo11s.pt)",
    )
    parser.add_argument(
        "--split", choices=["train", "val"], default="val",
        help="Split to sample from (default: val)",
    )
    parser.add_argument(
        "--n", type=int, default=12,
        help="Number of tile images to produce (default: 12)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/verify_tiles"),
        help="Output directory (default: output/verify_tiles/)",
    )

    args = parser.parse_args()
    cmd_verify(args)


if __name__ == "__main__":
    main()
