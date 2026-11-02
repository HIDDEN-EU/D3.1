"""
Prepare RAF-DB manifest.

RAF-DB ships as DATASET/train/{1-7}/*.jpg and DATASET/test/{1-7}/*.jpg with
label CSVs at {root}/train_labels.csv and {root}/test_labels.csv.
This script resolves each image to its absolute path, maps the numeric class
to an emotion name, and writes a manifest CSV.

No files are moved or copied.

Label mapping (RAF-DB basic expressions, standard):
  1 = surprise  2 = fear  3 = disgust  4 = happy
  5 = sad       6 = angry 7 = neutral

Usage
-----
python scripts/prepare_rafdb.py --root data/raf-db
python scripts/prepare_rafdb.py --root data/raf-db --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

LABEL_MAP = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral",
}
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}
MANIFEST = "rafdb_manifest.csv"

# RAF-DB numeric class dirs are stored inside DATASET/{split}/{cls}/
DATASET_DIR = "DATASET"


def process_split(root: Path, source_split: str) -> list[dict]:
    label_csv = root / f"{source_split}_labels.csv"
    if not label_csv.exists():
        sys.exit(f"Label CSV not found: {label_csv}")

    df_labels = pd.read_csv(label_csv)  # columns: image, label

    rows: list[dict] = []
    missing: list[str] = []

    for _, row in df_labels.iterrows():
        filename: str = row["image"]           # e.g. train_00006_aligned.jpg
        numeric_label: int = int(row["label"]) # 1-7

        emotion = LABEL_MAP[numeric_label]
        label   = EMOTION_TO_LABEL[emotion]

        # Images live under DATASET/{source_split}/{numeric_label}/{filename}
        img_path = root / DATASET_DIR / source_split / str(numeric_label) / filename
        if not img_path.exists():
            missing.append(str(img_path))
            continue

        rows.append(
            {
                "path": str(img_path.resolve()),
                "label": label,
                "emotion": emotion,
                "source_split": source_split,
            }
        )

    if missing:
        print(
            f"  [warn] {len(missing)} images not found in {source_split} split",
            file=sys.stderr,
        )
        for p in missing[:5]:
            print(f"    {p}", file=sys.stderr)
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more", file=sys.stderr)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAF-DB manifest CSV")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if manifest exists"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    manifest_path = root / MANIFEST
    if manifest_path.exists() and not args.force:
        print(f"Manifest already exists: {manifest_path}  (use --force to regenerate)")
        return

    print(f"Scanning {root} ...")
    rows: list[dict] = []
    for source_split in ("train", "test"):
        split_rows = process_split(root, source_split)
        print(f"  {source_split}: {len(split_rows):,} images")
        rows.extend(split_rows)

    if not rows:
        sys.exit("No images resolved — check --root path and DATASET/ structure")

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}  ({len(df):,} rows)")
    print(df.groupby(["source_split", "emotion"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
