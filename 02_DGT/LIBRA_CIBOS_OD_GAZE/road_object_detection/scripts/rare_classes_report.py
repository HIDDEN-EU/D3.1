"""
Visual report for rare BDD100K classes: other person, other vehicle, trailer, train.

Scans train + val ground-truth JSONs, collects up to --samples images per class,
draws bounding boxes (target class highlighted, rest dimmed), and saves a grid image
per class plus a printed summary table.

Usage:
    python scripts/rare_classes_report.py

    Optional overrides:
    --gt-train   .../bdd100k/labels/det_20/det_train.json
    --gt-val     .../bdd100k/labels/det_20/det_val.json
    --img-train  .../bdd100k/images/100k/train
    --img-val    .../bdd100k/images/100k/val
    --output     output/rare_classes
    --samples    12          (max images per class; use 0 for all)
    --cols       4           (grid columns)
    --cell-w     480         (width each cell is resized to)
    --cell-h     270         (height each cell is resized to)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TARGET_CLASSES = ["other person", "other vehicle", "trailer", "train"]

# Highlight colour per target class (BGR)
HIGHLIGHT = {
    "other person": (0, 255, 128),
    "other vehicle": (0, 180, 255),
    "trailer": (255, 80, 0),
    "train": (200, 0, 255),
}
DIM_BOX = (90, 90, 90)        # colour for non-target boxes
DIM_LABEL = (140, 140, 140)   # text colour for non-target labels


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_boxes(frame: np.ndarray, labels: list, focus_class: str) -> np.ndarray:
    """Draw all GT boxes; focus_class gets the highlight colour, others are dimmed."""
    img = frame.copy()
    colour = HIGHLIGHT[focus_class]

    # Draw non-focus boxes first so focus boxes appear on top
    for lbl in labels:
        cat = lbl.get("category", "")
        box = lbl.get("box2d")
        if box is None or cat == focus_class:
            continue
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), DIM_BOX, 1)
        cv2.putText(img, cat, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM_LABEL, 1)

    for lbl in labels:
        cat = lbl.get("category", "")
        box = lbl.get("box2d")
        if box is None or cat != focus_class:
            continue
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img, cat, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    return img


def make_cell(img: np.ndarray, title: str, colour: tuple,
              cell_w: int, cell_h: int) -> np.ndarray:
    """Resize image to cell dimensions and add a title bar."""
    resized = cv2.resize(img, (cell_w, cell_h))
    bar = np.zeros((22, cell_w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1)
    return np.vstack([bar, resized])


def make_grid(cells: list, cols: int) -> np.ndarray:
    """Arrange cells into a grid, padding the last row if needed."""
    if not cells:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h, w = cells[0].shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    # Pad to full rows
    remainder = len(cells) % cols
    if remainder:
        cells = cells + [blank] * (cols - remainder)
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt(json_path: Path) -> dict:
    """Return {filename: {labels: [...], attributes: {...}}}."""
    with open(json_path) as f:
        data = json.load(f)
    return {
        entry["name"]: {
            "labels": entry.get("labels") or [],
            "attributes": entry.get("attributes") or {},
        }
        for entry in data
    }


def collect_examples(gt: dict, img_dir: Path, target: str, max_samples: int) -> list:
    """
    Return a list of (img_path, labels, attributes) for images containing target.
    Shuffled then capped at max_samples (0 = all).
    """
    found = []
    for name, info in gt.items():
        if any(lbl.get("category") == target for lbl in info["labels"]):
            img_path = img_dir / name
            if img_path.exists():
                found.append((img_path, info["labels"], info["attributes"]))
    random.shuffle(found)
    if max_samples > 0:
        found = found[:max_samples]
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-train", type=Path,
        default=Path("data/bdd100k/labels/det_20/det_train.json"))
    parser.add_argument("--gt-val", type=Path,
        default=Path("data/bdd100k/labels/det_20/det_val.json"))
    parser.add_argument("--img-train", type=Path,
        default=Path("data/bdd100k/images/100k/train"))
    parser.add_argument("--img-val", type=Path,
        default=Path("data/bdd100k/images/100k/val"))
    parser.add_argument("--output", type=Path, default=Path("output/rare_classes"))
    parser.add_argument("--samples", type=int, default=12,
        help="Max images per class (0 = all)")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--cell-w", type=int, default=480)
    parser.add_argument("--cell-h", type=int, default=270)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading ground-truth JSONs …")
    gt_train = load_gt(args.gt_train)
    gt_val   = load_gt(args.gt_val)

    # --- Summary table ---
    print(f"\n{'Class':<18} {'Train imgs':>10} {'Val imgs':>10} {'Total instances (train)':>24} {'Total instances (val)':>22}")
    print("-" * 90)
    class_stats = {}
    for cls in TARGET_CLASSES:
        t_imgs = sum(
            1 for info in gt_train.values()
            if any(l.get("category") == cls for l in info["labels"])
        )
        v_imgs = sum(
            1 for info in gt_val.values()
            if any(l.get("category") == cls for l in info["labels"])
        )
        t_inst = sum(
            sum(1 for l in info["labels"] if l.get("category") == cls)
            for info in gt_train.values()
        )
        v_inst = sum(
            sum(1 for l in info["labels"] if l.get("category") == cls)
            for info in gt_val.values()
        )
        class_stats[cls] = dict(t_imgs=t_imgs, v_imgs=v_imgs, t_inst=t_inst, v_inst=v_inst)
        print(f"  {cls:<16} {t_imgs:>10} {v_imgs:>10} {t_inst:>24} {v_inst:>22}")
    print()

    # --- Generate one grid image per class ---
    for cls in TARGET_CLASSES:
        colour = HIGHLIGHT[cls]
        print(f"[{cls}] collecting examples …")

        examples_train = collect_examples(gt_train, args.img_train, cls, 0)
        examples_val   = collect_examples(gt_val,   args.img_val,   cls, 0)
        all_examples = examples_train + examples_val

        if not all_examples:
            print(f"  No images found — skipping.")
            continue

        n_total = len(all_examples)
        # Respect --samples cap (0 = all)
        if args.samples > 0:
            sampled = all_examples[:args.samples]
        else:
            sampled = all_examples

        print(f"  Found {n_total} images total → showing {len(sampled)}")

        cells = []
        for img_path, labels, attrs in sampled:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            annotated = draw_boxes(frame, labels, cls)
            n_inst = sum(1 for l in labels if l.get("category") == cls)
            scene  = attrs.get("scene", "")
            tod    = attrs.get("timeofday", "")
            title  = f"{img_path.name}  [{n_inst}x]  {scene} {tod}"
            cell = make_cell(annotated, title, colour, args.cell_w, args.cell_h)
            cells.append(cell)

        if not cells:
            print("  Could not read any images — skipping.")
            continue

        grid = make_grid(cells, args.cols)

        # Header banner
        banner_h = 40
        banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)
        stats = class_stats[cls]
        banner_text = (
            f"CLASS: {cls.upper()}   "
            f"train: {stats['t_imgs']} imgs / {stats['t_inst']} instances   "
            f"val: {stats['v_imgs']} imgs / {stats['v_inst']} instances   "
            f"showing {len(cells)} of {n_total}"
        )
        cv2.putText(banner, banner_text, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
        final = np.vstack([banner, grid])

        out_path = args.output / f"{cls.replace(' ', '_')}_grid.jpg"
        cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  Saved → {out_path}")

    print(f"\nDone. Grid images written to {args.output}/")
    print("Tip: open with any image viewer or VS Code image preview.")


if __name__ == "__main__":
    main()
