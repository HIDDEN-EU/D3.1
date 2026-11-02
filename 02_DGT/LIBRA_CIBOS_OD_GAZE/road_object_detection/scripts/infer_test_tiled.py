"""
Tiled inference viewer for the unlabeled BDD100K test images.

Picks N random images from the BDD100K test set (`bdd100k_test/`, 20k unlabeled
frames), splits each into a 2x2 grid of tiles, runs the fine-tuned model on each
tile independently, maps detections back to full-frame coordinates, merges
overlapping detections across tile seams with class-wise NMS, and writes a
self-contained HTML viewer to output/test_infer/.

Why tiled inference (the 2x2 split is the whole point):
the model was fine-tuned on ~640x360 tiles, so feeding it a full 1280x720 frame
downsamples away the small/distant objects that tiling was meant to recover.
Splitting the frame and inferring per tile mirrors production tiled inference.

This is a qualitative viewer only — the test set has no labels, so there are no
metrics here. For mAP / occlusion-stratified metrics use the labeled held-out
val slice instead (`evaluate-model`).

Usage
-----
infer-test-tiled
infer-test-tiled --num 15 --seed 42 --conf 0.25
infer-test-tiled --weights runs/train/tiled-v030-mz16-5ep/weights/best.pt \
                 --test-dir data/bdd100k_test

Open the printed output/test_infer/index.html directly in a browser.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from road_object_detection import inference

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/train/tiled-v030-mz16-5ep/weights/best.pt"
DEFAULT_TEST_DIR = Path("data/bdd100k_test")
DEFAULT_OUTPUT = REPO_ROOT / "output/test_infer"
WEBVIEW_SRC = Path(__file__).resolve().parent / "webview"


# Geometry is the shared definition in road_object_detection.inference — the same one
# edge service (../edge_inference) runs on the Jetson. These names stay for the other viewers.
_tile_regions = inference.tile_regions
_grid_regions = inference.grid_regions


def _grid_shape(regions):
    """Infer (rows, cols) of a grid from its region list (distinct x0/y0 edges)."""
    cols = len({r[0] for r in regions})
    rows = len({r[1] for r in regions})
    return rows, cols


def _resolve_tiles(select, n_regions):
    """Validate user-supplied 0-based tile indices against the grid size.

    Returns a sorted set of indices to annotate, or None meaning 'all tiles'.
    Exits with a clear message on any out-of-range index."""
    if not select:
        return None
    bad = sorted({t for t in select if t < 0 or t >= n_regions})
    if bad:
        sys.exit(
            f"--tiles {bad} out of range: grid has {n_regions} tiles "
            f"(valid indices 0..{n_regions - 1})"
        )
    return set(select)


def _print_grid(regions, active):
    """Print the row-major tile index map, marking which tiles are annotated."""
    rows, cols = _grid_shape(regions)
    print(f"[tiles] {rows}x{cols} grid = {len(regions)} tiles (row-major, 0-based):")
    for r in range(rows):
        cells = [
            (f"[{r * cols + c:>2}]" if active is None or (r * cols + c) in active
             else f" {r * cols + c:>2} ")
            for c in range(cols)
        ]
        print("        " + "".join(cells))
    print(f"[tiles] annotating: {'all' if active is None else sorted(active)}")


def _infer_tiled(model, img, conf: float, imgsz: int, merge_iou: float, regions=None,
                 select=None):
    """Run the model on each tile, lift boxes to full-frame coords, NMS-merge across tiles.

    `regions` (list of (x0,y0,x1,y1)) overrides the default fixed 2×2 split — pass
    `_grid_regions(w, h)` for resolution-aware tiling.
    `select` (set/iterable of region indices) restricts inference to those tiles;
    None (default) runs every tile. Region geometry and the `tile` index in each
    detection are unchanged across selections, so indices stay stable."""
    h, w = img.shape[:2]
    if regions is None:
        regions = _tile_regions(w, h)
    active = set(range(len(regions))) if select is None else set(select)
    xyxy, scores, classes, tiles = [], [], [], []
    for ti, (x0, y0, x1, y1) in enumerate(regions):
        if ti not in active:
            continue
        crop = img[y0:y1, x0:x1]
        res = model.predict(crop, conf=conf, iou=inference.TILE_IOU, imgsz=imgsz,
                            verbose=False)[0]
        for b in res.boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            xyxy.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0])
            scores.append(float(b.conf[0]))
            classes.append(int(b.cls[0]))
            tiles.append(ti)

    dets = []
    if xyxy:
        keep = inference.merge_tiles(
            np.asarray(xyxy, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            np.asarray(classes, dtype=np.int64),
            merge_iou,
        )
        for i in keep.tolist():
            dets.append(
                {
                    "cls": classes[i],
                    "name": model.names[classes[i]],
                    "conf": round(scores[i], 4),
                    "xyxy": [round(v, 1) for v in xyxy[i]],
                    "tile": tiles[i],
                }
            )
    return regions, dets


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tiled (2x2) inference viewer on unlabeled BDD100K test images."
    )
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    ap.add_argument("--num", type=int, default=15, help="number of random images")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (reproducible)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size per tile")
    ap.add_argument("--merge-iou", type=float, default=inference.MERGE_IOU,
                    help="cross-tile NMS IoU")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}")
    if not args.test_dir.is_dir():
        sys.exit(f"test dir not found: {args.test_dir}")

    images = sorted(args.test_dir.glob("*.jpg"))
    if not images:
        sys.exit(f"no .jpg images under {args.test_dir}")
    sample = random.Random(args.seed).sample(images, min(args.num, len(images)))

    print(f"[infer] loading {args.weights}")
    model = YOLO(str(args.weights))
    print(f"[infer] {len(model.names)} classes: {model.names}")
    print(f"[infer] {len(sample)} images, conf>={args.conf}, 2x2 tiles, merge-IoU {args.merge_iou}")

    img_out = args.output / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "weights": str(args.weights),
        "conf": args.conf,
        "merge_iou": args.merge_iou,
        "images": [],
    }
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[infer] skip unreadable {p.name}")
            continue
        h, w = img.shape[:2]
        regions, dets = _infer_tiled(model, img, args.conf, args.imgsz, args.merge_iou)
        shutil.copy(p, img_out / p.name)
        manifest["images"].append(
            {
                "file": f"images/{p.name}",
                "width": w,
                "height": h,
                "tiles": [list(r) for r in regions],
                "detections": dets,
            }
        )
        print(f"[infer] {p.name}: {len(dets)} detections")

    # manifest.js (not .json) so index.html works by double-click — file:// fetch is
    # blocked in Chrome, but a <script>-loaded global is not.
    (args.output / "manifest.js").write_text(
        "window.MANIFEST = " + json.dumps(manifest, indent=2) + ";\n"
    )
    for f in ("index.html", "style.css", "app.js"):
        shutil.copy(WEBVIEW_SRC / f, args.output / f)

    print(f"\n[infer] done — {len(manifest['images'])} images -> {args.output}")
    print(f"[infer] open in a browser: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
