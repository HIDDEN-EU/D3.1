"""
Prepare FER2013 manifest.

FER2013 is already organised as {root}/train/{emotion}/*.jpg and
{root}/test/{emotion}/*.jpg, so no files are moved.  This script scans the
directory tree and writes a manifest CSV that the data loader reads at runtime.

Usage
-----
python scripts/prepare_fer2013.py --root data/fer2013
python scripts/prepare_fer2013.py --root data/fer2013 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}
SENTINEL = ".prepared"
MANIFEST = "fer2013_manifest.csv"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def scan_split(root: Path, source_split: str) -> list[dict]:
    split_dir = root / source_split
    if not split_dir.exists():
        print(f"  [skip] {split_dir} not found", file=sys.stderr)
        return []
    rows: list[dict] = []
    for emotion in EMOTIONS:
        emotion_dir = split_dir / emotion
        if not emotion_dir.exists():
            print(f"  [warn] missing emotion dir: {emotion_dir}", file=sys.stderr)
            continue
        for img_path in sorted(emotion_dir.iterdir()):
            if img_path.suffix.lower() in VALID_SUFFIXES:
                rows.append(
                    {
                        "path": str(img_path.resolve()),
                        "label": EMOTION_TO_LABEL[emotion],
                        "emotion": emotion,
                        "source_split": source_split,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FER2013 manifest CSV")
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
        rows.extend(scan_split(root, source_split))

    if not rows:
        sys.exit("No images found — check --root path")

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}  ({len(df):,} rows)")
    print(df.groupby(["source_split", "emotion"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
