"""
Prepare FER+ manifest.

FER+ ships without images — only crowdsourced label CSVs and the original
FER2013 pixel data (fer2013_original/fer2013.csv).  This script runs in two
phases:

  1. Image extraction (skipped if images already exist, unless --force):
     Reads pixel strings from fer2013_original/fer2013.csv and the split
     assignments from fer2013new.csv, then writes 48×48 grayscale PNGs to
       {root}/FER2013Train/
       {root}/FER2013Valid/
       {root}/FER2013Test/

  2. Manifest generation:
     Reads the per-split label CSVs (no header), resolves image paths, applies
     majority-vote label selection (excluding unknown / NF / contempt), and
     writes ferplus_manifest.csv.

Label strategy
--------------
FER+ provides 10 votes per image across 8 classes:
  neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
The majority-vote label is the argmax over the first 7 emotion columns
(contempt excluded).  Rows are dropped when:
  - the image filename is blank (invalid entry)
  - unknown or NF has the most votes (unclear face)
  - contempt has the most votes (no canonical mapping)

Canonical emotion ordering (shared across all datasets):
  0=angry  1=disgust  2=fear  3=happy  4=neutral  5=sad  6=surprise

Usage
-----
python scripts/prepare_ferplus.py --root data/FERPlus
python scripts/prepare_ferplus.py --root data/FERPlus --force
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}
MANIFEST = "ferplus_manifest.csv"

# FER+ vote column order (columns 2-11 in fer2013new.csv / label.csv after filename+bbox)
VOTE_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear",
                "contempt", "unknown", "NF"]

# Map FER+ emotion names → canonical names (contempt has no canonical mapping)
FERPLUS_TO_CANONICAL = {
    "neutral":   "neutral",
    "happiness": "happy",
    "surprise":  "surprise",
    "sadness":   "sad",
    "anger":     "angry",
    "disgust":   "disgust",
    "fear":      "fear",
}

# The 7 scoreable emotions (contempt excluded from argmax)
SCOREABLE = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

# Per-split subdirectories and their source_split labels
SPLIT_DIRS = {
    "FER2013Train": "train",
    "FER2013Valid": "val",
    "FER2013Test":  "test",
}

# FER2013 pixel CSV → split folder mapping (used during extraction)
FERPLUS_SPLIT_MAP = {
    "Training":    "FER2013Train",
    "PublicTest":  "FER2013Valid",
    "PrivateTest": "FER2013Test",
}


# ── Phase 1: image extraction ─────────────────────────────────────────────────


def _images_exist(root: Path) -> bool:
    """True if at least one PNG exists in the expected output dirs."""
    for subdir in SPLIT_DIRS:
        pngs = list((root / subdir).glob("*.png"))
        if pngs:
            return True
    return False


def extract_images(root: Path) -> None:
    """
    Convert fer2013_original/fer2013.csv pixel strings to PNGs.

    Mirrors the logic of the upstream generate_training_data.py.
    """
    fer_csv     = root / "fer2013_original" / "fer2013.csv"
    ferplus_csv = root / "fer2013new.csv"

    if not fer_csv.exists():
        sys.exit(f"FER2013 pixel CSV not found: {fer_csv}")
    if not ferplus_csv.exists():
        sys.exit(f"FER+ label CSV not found: {ferplus_csv}")

    # Create output directories
    for subdir in SPLIT_DIRS:
        (root / subdir).mkdir(exist_ok=True)

    # Read FER+ entries (skip header)
    ferplus_entries: list[list[str]] = []
    with open(ferplus_csv) as f:
        for row in islice(csv.reader(f), 1, None):
            ferplus_entries.append(row)

    print(f"  Extracting images from {fer_csv} ...")
    count = 0
    with open(fer_csv) as f:
        for idx, row in enumerate(islice(csv.reader(f), 1, None)):
            ferplus_row = ferplus_entries[idx]
            file_name = ferplus_row[1].strip()
            if not file_name:
                continue
            original_split = row[2].strip()  # "Training" / "PublicTest" / "PrivateTest"
            subdir = FERPLUS_SPLIT_MAP.get(original_split)
            if subdir is None:
                continue
            # Decode pixel string → 48×48 grayscale PNG
            pixels = np.asarray(row[1].split(), dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(pixels)
            img.save(root / subdir / file_name, compress_level=0)
            count += 1

    print(f"  Wrote {count:,} images")


# ── Phase 2: manifest generation ──────────────────────────────────────────────


def process_split(root: Path, subdir: str, source_split: str) -> tuple[list[dict], dict[str, int]]:
    """Parse one split's label CSV and return (rows, skipped_counts)."""
    label_csv = root / "data" / subdir / "label.csv"
    if not label_csv.exists():
        print(f"  [skip] label CSV not found: {label_csv}", file=sys.stderr)
        return [], {}

    rows: list[dict] = []
    skipped: dict[str, int] = {"blank": 0, "unknown_nf": 0, "contempt": 0, "missing_img": 0}

    with open(label_csv) as f:
        for line in csv.reader(f):
            if not line:
                continue
            filename = line[0].strip()
            if not filename:
                skipped["blank"] += 1
                continue

            # Votes start at index 2 (after filename and bbox); 10 values
            votes = [int(v) for v in line[2:12]]
            vote_dict = dict(zip(VOTE_COLUMNS, votes))

            # Check if unknown or NF dominates
            max_vote = max(votes)
            if max_vote == 0:
                skipped["unknown_nf"] += 1
                continue
            if vote_dict["unknown"] == max_vote or vote_dict["NF"] == max_vote:
                skipped["unknown_nf"] += 1
                continue

            # Majority vote over the 7 scoreable emotions
            scoreable_votes = [vote_dict[e] for e in SCOREABLE]
            best_scoreable = max(scoreable_votes)

            # If contempt beats all scoreable emotions, drop the row
            if vote_dict["contempt"] > best_scoreable:
                skipped["contempt"] += 1
                continue

            # Argmax over scoreable emotions
            winner = SCOREABLE[scoreable_votes.index(best_scoreable)]
            canonical = FERPLUS_TO_CANONICAL[winner]

            img_path = root / subdir / filename
            if not img_path.exists():
                skipped["missing_img"] += 1
                continue

            rows.append({
                "path":         str(img_path.resolve()),
                "label":        EMOTION_TO_LABEL[canonical],
                "emotion":      canonical,
                "source_split": source_split,
            })

    return rows, skipped


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FER+ manifest CSV")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Re-extract images and regenerate manifest"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    manifest_path = root / MANIFEST

    # ── Phase 1: extract images ────────────────────────────────────────────────
    if _images_exist(root) and not args.force:
        print("Images already present — skipping extraction  (use --force to redo)")
    else:
        extract_images(root)

    # ── Phase 2: build manifest ────────────────────────────────────────────────
    if manifest_path.exists() and not args.force:
        print(f"Manifest already exists: {manifest_path}  (use --force to regenerate)")
        return

    print(f"\nBuilding manifest ...")
    all_rows: list[dict] = []
    total_skipped: dict[str, int] = {}

    for subdir, source_split in SPLIT_DIRS.items():
        rows, skipped = process_split(root, subdir, source_split)
        print(f"  {source_split} ({subdir}): {len(rows):,} rows kept", end="")
        if any(skipped.values()):
            details = ", ".join(f"{k}={v}" for k, v in skipped.items() if v)
            print(f"  (dropped: {details})", end="")
        print()
        all_rows.extend(rows)
        for k, v in skipped.items():
            total_skipped[k] = total_skipped.get(k, 0) + v

    if not all_rows:
        sys.exit("No images resolved — run with --force to re-extract images first")

    df = pd.DataFrame(all_rows)
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}  ({len(df):,} rows)")

    total_dropped = sum(total_skipped.values())
    if total_dropped:
        print(f"Total dropped: {total_dropped}  " +
              "  ".join(f"{k}={v}" for k, v in total_skipped.items() if v))

    print(df.groupby(["source_split", "emotion"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
