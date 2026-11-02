"""
Prepare AffectNet manifest.

AffectNet (HuggingFace subset) is organised as {root}/{emotion}/*.jpg.
This script scans the directory tree, maps class names to the 7-class label
scheme (dropping 'contempt' which has no counterpart in FER2013 / KDEF), and
writes affectnet_manifest.csv.

Usage
-----
python scripts/prepare_affectnet.py --root data/affectnet
python scripts/prepare_affectnet.py --root data/affectnet --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# AffectNet folder names → our canonical 7-class emotion names
# 'contempt' is dropped (no equivalent in FER2013 / KDEF)
FOLDER_TO_EMOTION: dict[str, str] = {
    "anger":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "surprise": "surprise",
}
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}
MANIFEST = "affectnet_manifest.csv"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AffectNet manifest CSV")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Regenerate even if manifest exists")
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
    skipped_contempt = 0

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == "contempt":
            skipped_contempt = sum(
                1 for f in folder.iterdir() if f.suffix.lower() in VALID_SUFFIXES
            )
            print(f"  [skip] contempt ({skipped_contempt} images) — not in 7-class scheme")
            continue
        emotion = FOLDER_TO_EMOTION.get(folder.name)
        if emotion is None:
            print(f"  [warn] unknown folder: {folder.name}", file=sys.stderr)
            continue
        label = EMOTION_TO_LABEL[emotion]
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in VALID_SUFFIXES:
                rows.append({
                    "path": str(img_path.resolve()),
                    "label": label,
                    "emotion": emotion,
                    "source_split": "train",  # no official split — ratio split used at load time
                })

    if not rows:
        sys.exit("No images found — check --root path")

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path}  ({len(df):,} rows)")
    print(df.groupby("emotion")["path"].count().to_string())


if __name__ == "__main__":
    main()
