"""
BDD100K dataset utilities — convert, validate, find-missing, make-list.

All BDD100K-related operations are in this single script.

Subcommands
-----------
convert       Convert aggregate JSON labels to YOLO .txt format.
validate      Validate the dataset on disk (read-only).
find-missing  Search det_20 individual JSONs for stems reported as missing.
make-list     Write a train.txt with only images that have a label file.

Usage
-----
python scripts/bdd100k.py convert \\
    --aggregate data/bdd100k/labels/det_20/det_train.json \\
    --output    data/bdd100k/labels/100k/train

python scripts/bdd100k.py validate \\
    --root data/bdd100k [--full] [--sample N] [--log FILE]

python scripts/bdd100k.py find-missing \\
    --missing missing_train.txt \\
    --det20-dir data/bdd100k/labels/det_20/train

python scripts/bdd100k.py make-list \\
    --root   data/bdd100k \\
    --output data/bdd100k/train.txt
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants — single source of truth for class definitions.
# Must match configs/data.yaml exactly.
# ---------------------------------------------------------------------------

CLASS_MAP: dict[str, int] = {
    "pedestrian":    0,
    "rider":         1,
    "car":           2,
    "truck":         3,
    "bus":           4,
    "motorcycle":    5,
    "bicycle":       6,
    "traffic light": 7,
    "traffic sign":  8,
    "other person":  9,
    "other vehicle": 10,
    "trailer":       11,
    "train":         12,
}

CLASS_NAMES: dict[int, str] = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)
VALID_CLASS_IDS = set(range(NUM_CLASSES))

# All BDD100K images are 1280×720 — confirmed from actual data.
IMG_W, IMG_H = 1280, 720


# ---------------------------------------------------------------------------
# Shared print helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'─' * 60}\n  {title}\n{'─' * 60}")

def ok(msg: str)   -> None: print(f"  [OK]   {msg}")
def warn(msg: str) -> None: print(f"  [WARN] {msg}")
def fail(msg: str) -> None: print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------

def _convert_box(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    return (x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H, (x2 - x1) / IMG_W, (y2 - y1) / IMG_H


def _convert_entry(
    entry: dict, output_dir: Path
) -> tuple[int, str, list[tuple[int, bool, bool]]]:
    """
    Write one YOLO .txt label file and return (n_objects, stem, occlusion_rows).
    occlusion_rows: list of (box_index, occluded, truncated) for each written box.
    """
    stem = Path(entry["name"]).stem
    txt_path = output_dir / (stem + ".txt")
    labels = entry.get("labels") or []
    lines: list[str] = []
    occ_rows: list[tuple[int, bool, bool]] = []
    for label in labels:
        cls_id = CLASS_MAP.get(label.get("category", ""))
        if cls_id is None:
            print(f"  WARNING: unknown category '{label.get('category')}' — skipped")
            continue
        box = label.get("box2d")
        if box is None:
            continue
        x_c, y_c, w, h = _convert_box(box["x1"], box["y1"], box["x2"], box["y2"])
        if w <= 0 or h <= 0:
            continue
        box_idx = len(lines)
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        attrs = label.get("attributes") or {}
        occ_rows.append((box_idx, bool(attrs.get("occluded", False)), bool(attrs.get("truncated", False))))
    txt_path.write_text("\n".join(lines))
    return len(lines), stem, occ_rows


def cmd_convert(args: argparse.Namespace) -> None:
    if not args.aggregate.exists():
        print(f"ERROR: {args.aggregate} not found"); sys.exit(1)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.aggregate} ...")
    data = json.loads(args.aggregate.read_text())

    print(f"Converting {len(data)} entries to {args.output} ...")
    total_objects = 0
    occ_csv_path = args.output / "occlusion.csv"
    with occ_csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stem", "box_index", "occluded", "truncated"])
        for i, entry in enumerate(data, 1):
            n, stem, occ_rows = _convert_entry(entry, args.output)
            total_objects += n
            for box_idx, occluded, truncated in occ_rows:
                writer.writerow([stem, box_idx, occluded, truncated])
            if i % 10000 == 0:
                print(f"  {i}/{len(data)} ...")

    print(f"\n  Label files written : {len(data)}")
    print(f"  Total objects       : {total_objects}")
    print(f"  Occlusion sidecar   : {occ_csv_path}")
    print(f"\nDone.")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _check_structure(root: Path) -> list[str]:
    required = [
        root / "images" / "100k" / "train",
        root / "images" / "100k" / "val",
        root / "labels" / "100k" / "train",
        root / "labels" / "100k" / "val",
        root / "labels" / "det_20" / "det_train.json",
        root / "labels" / "det_20" / "det_val.json",
    ]
    errors = [f"Missing: {p}" for p in required if not p.exists()]
    return errors


def _check_split(root: Path, split: str) -> tuple[list[Path], list[Path], list[str]]:
    images = sorted((root / "images" / "100k" / split).glob("*.jpg"))
    labels = sorted((root / "labels" / "100k" / split).glob("*.txt"))
    ok(f"{split}: {len(images)} images,  {len(labels)} label files on disk")
    errors = []
    if len(labels) != len(images):
        msg = (
            f"{split}: {len(images)} images but {len(labels)} label files "
            f"({abs(len(images) - len(labels))} mismatch)"
        )
        errors.append(msg)
        fail(msg)
    else:
        ok(f"{split}: counts match  ✓")
    return images, labels, errors


def _check_pairing(images: list[Path], labels: list[Path], split: str) -> tuple[list[str], set[str]]:
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    missing = image_stems - label_stems
    orphans = label_stems - image_stems
    errors = []
    if missing:
        msg = f"{split}: {len(missing)} images have no label  (sample: {sorted(missing)[:5]})"
        errors.append(msg); fail(msg)
    if orphans:
        msg = f"{split}: {len(orphans)} labels have no image  (sample: {sorted(orphans)[:5]})"
        errors.append(msg); fail(msg)
    if not missing and not orphans:
        ok(f"{split}: all image↔label pairs match  ✓")
    return errors, missing


def _check_label_format(label_files: list[Path], split: str, sample_n: int | None) -> tuple[list[str], Counter]:
    if sample_n is not None and sample_n < len(label_files):
        check_files = random.sample(label_files, sample_n)
        scope = f"sample of {sample_n}"
    else:
        check_files = label_files
        scope = f"all {len(label_files)}"

    errors: list[str] = []
    class_counts: Counter = Counter()
    bad = 0

    for lf in check_files:
        text = lf.read_text()
        if not text.strip():
            continue
        for lineno, line in enumerate(text.strip().splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{lf.name}:{lineno} — expected 5 values, got {len(parts)}"); bad += 1; break
            try:
                cls_id = int(parts[0]); vals = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append(f"{lf.name}:{lineno} — non-numeric: {line!r}"); bad += 1; break
            if cls_id not in VALID_CLASS_IDS:
                errors.append(f"{lf.name}:{lineno} — class_id {cls_id} out of range [0,{NUM_CLASSES-1}]"); bad += 1; break
            if not all(0.0 < v < 1.0 for v in vals):
                errors.append(f"{lf.name}:{lineno} — coords out of (0,1): {vals}"); bad += 1; break
            class_counts[cls_id] += 1

    if bad == 0:
        ok(f"{split}: label format valid ({scope} checked)  ✓")
    else:
        for e in errors[:10]: fail(e)
        if len(errors) > 10: fail(f"... and {len(errors) - 10} more")

    return errors, class_counts


def cmd_validate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    sample_n = None if args.full else args.sample
    all_errors: list[str] = []
    total_counts: Counter = Counter()

    section("1 — Directory structure")
    errs = _check_structure(args.root)
    for e in errs: fail(e)
    if not errs: ok("All required paths present  ✓")
    all_errors.extend(errs)
    if errs:
        fail("Aborting — fix missing paths first."); sys.exit(1)

    for split in ("train", "val"):
        section(f"2/3/4 — {split} split")
        images, labels, count_errs = _check_split(args.root, split)
        all_errors.extend(count_errs)
        pair_errs, missing = _check_pairing(images, labels, split)
        all_errors.extend(pair_errs)
        if missing and args.log:
            log_path = args.log.with_stem(f"{args.log.stem}_{split}")
            log_path.write_text("\n".join(sorted(missing)))
            warn(f"{split}: missing stems written to {log_path}")
        fmt_errs, counts = _check_label_format(labels, split, sample_n)
        all_errors.extend(fmt_errs)
        total_counts.update(counts)

    section("5 — Class distribution (sampled labels)")
    if total_counts:
        total = sum(total_counts.values())
        print(f"  {'Class':<20} {'Count':>8}  {'%':>6}")
        print(f"  {'─'*20}  {'─'*8}  {'─'*6}")
        for i in range(NUM_CLASSES):
            n = total_counts.get(i, 0)
            print(f"  {CLASS_NAMES[i]:<20} {n:>8}  {100*n/total:>5.1f}%")
        print(f"\n  Total objects in sample: {total:,}")
    else:
        warn("No class counts — format checks may have all failed.")

    section("Summary")
    if all_errors:
        fail(f"{len(all_errors)} error(s) — dataset is NOT ready for training."); sys.exit(1)
    else:
        ok("All checks passed — dataset is ready for training.")


# ---------------------------------------------------------------------------
# find-missing
# ---------------------------------------------------------------------------

def cmd_find_missing(args: argparse.Namespace) -> None:
    if not args.missing.exists():
        print(f"ERROR: {args.missing} not found"); sys.exit(1)
    if not args.det20_dir.exists():
        print(f"ERROR: {args.det20_dir} not found"); sys.exit(1)

    stems = [s.strip() for s in args.missing.read_text().splitlines() if s.strip()]
    print(f"Searching {len(stems)} missing stems in {args.det20_dir}\n")

    found_with_labels, found_empty, not_found = [], [], []
    for stem in stems:
        p = args.det20_dir / f"{stem}.json"
        if not p.exists():
            not_found.append(stem); continue
        labels = json.loads(p.read_text()).get("labels") or []
        if labels:
            found_with_labels.append((stem, len(labels)))
        else:
            found_empty.append(stem)

    print(f"Found in det_20 WITH annotations : {len(found_with_labels)}")
    for stem, n in found_with_labels:
        print(f"  {stem}  ({n} objects)")

    print(f"\nFound in det_20 but EMPTY : {len(found_empty)}")
    for stem in found_empty:
        print(f"  {stem}")

    print(f"\nNot found in det_20 at all : {len(not_found)}")
    for stem in not_found:
        print(f"  {stem}")

    print(f"\nSummary: {len(stems)} missing — "
          f"{len(found_with_labels)} have annotations, "
          f"{len(found_empty)} empty, "
          f"{len(not_found)} not in det_20.")


# ---------------------------------------------------------------------------
# make-list
# ---------------------------------------------------------------------------

def cmd_make_list(args: argparse.Namespace) -> None:
    images_dir = args.root / "images" / "100k" / "train"
    labels_dir = args.root / "labels" / "100k" / "train"

    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found"); sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: {labels_dir} not found"); sys.exit(1)

    all_images = sorted(images_dir.glob("*.jpg"))
    included = [p for p in all_images if (labels_dir / f"{p.stem}.txt").exists()]
    excluded = len(all_images) - len(included)

    args.output.write_text("\n".join(str(p) for p in included))

    print(f"Total images : {len(all_images)}")
    print(f"Included     : {len(included)}")
    print(f"Excluded     : {excluded}  (no label file)")
    print(f"Written to   : {args.output}")
    print(f"\nUpdate configs/data.yaml:  train: train.txt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="bdd100k.py")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # convert
    p_conv = sub.add_parser("convert", help="Convert aggregate JSON to YOLO labels")
    p_conv.add_argument("--aggregate", type=Path, required=True)
    p_conv.add_argument("--output",    type=Path, required=True)

    # validate
    p_val = sub.add_parser("validate", help="Validate dataset on disk (read-only)")
    p_val.add_argument("--root",   type=Path, required=True)
    p_val.add_argument("--sample", type=int,  default=2000)
    p_val.add_argument("--full",   action="store_true")
    p_val.add_argument("--log",    type=Path, default=None)
    p_val.add_argument("--seed",   type=int,  default=42)

    # find-missing
    p_fm = sub.add_parser("find-missing", help="Search det_20 individual JSONs for missing stems")
    p_fm.add_argument("--missing",   type=Path, required=True)
    p_fm.add_argument("--det20-dir", type=Path, required=True)

    # make-list
    p_ml = sub.add_parser("make-list", help="Write train.txt with only images that have labels")
    p_ml.add_argument("--root",   type=Path, required=True)
    p_ml.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    {"convert": cmd_convert, "validate": cmd_validate,
     "find-missing": cmd_find_missing, "make-list": cmd_make_list}[args.cmd](args)


if __name__ == "__main__":
    main()
