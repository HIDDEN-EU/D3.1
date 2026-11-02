"""
Prepare KDEF manifest.

KDEF ships with an existing ``kdef_manifest_splits.csv`` written for the
original Kaggle environment (paths starting with ``/kaggle/input/...``).
This script resolves each image to its actual local path and writes a
standard manifest CSV that the data loader reads at runtime.

No files are moved or copied.

Usage
-----
python scripts/prepare_kdef.py --root data/kdef-dataset
python scripts/prepare_kdef.py --root data/kdef-dataset --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}
SOURCE_MANIFEST = "kdef_manifest_splits.csv"
MANIFEST = "kdef_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KDEF manifest CSV")
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

    source_path = root / SOURCE_MANIFEST
    if not source_path.exists():
        sys.exit(f"Source manifest not found: {source_path}")

    print(f"Reading {source_path} ...")
    df = pd.read_csv(source_path)

    rows: list[dict] = []
    missing: list[str] = []

    for _, row in df.iterrows():
        emotion: str = row["expression"]
        source_split: str = row["split"]  # train / val / test
        filename: str = row["filename"]

        # Resolve local path: {root}/{split}/{emotion}/{filename}
        img_path = root / source_split / emotion / filename
        if not img_path.exists():
            missing.append(str(img_path))
            continue

        rows.append(
            {
                "path": str(img_path.resolve()),
                "label": EMOTION_TO_LABEL[emotion],
                "emotion": emotion,
                "source_split": source_split,
                "subject": row["subject"],
                "session": row["session"],
                "gender": row["gender"],
                "angle": row["angle"],
            }
        )

    if missing:
        print(
            f"  [warn] {len(missing)} images not found on disk",
            file=sys.stderr,
        )
        for p in missing[:5]:
            print(f"    {p}", file=sys.stderr)
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more", file=sys.stderr)

    if not rows:
        sys.exit("No images resolved — check --root path")

    out = pd.DataFrame(rows)
    out.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}  ({len(out):,} rows)")
    print(out.groupby(["source_split", "emotion"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
