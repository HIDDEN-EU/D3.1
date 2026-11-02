"""
Generate a CSV manifest for a dataset variant.

A manifest is a flat CSV inventory of every image in a variant, with one row
per image.  It lets training code work from a stable file list instead of
scanning the filesystem, and makes train/val/test splits a config-level filter.

Usage
-----
# Original / merged variant (source_image and tile_index will be empty):
generate-manifest \\
    --data-yaml configs/data.yaml \\
    --output    data/bdd100k

# Tiled variant (back-references each tile to its 1280x720 source image):
generate-manifest \\
    --data-yaml        data/bdd100k_tiled_4/data.yaml \\
    --source-data-yaml configs/data.yaml \\
    --variant          data_tiled_4x
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

FIELDNAMES = ["path", "label_path", "source_split", "variant", "source_image", "tile_index"]

# Matches the _t{row}{col} suffix tile-dataset appends to each tile stem.
_TILE_RE = re.compile(r"^(.+)_t(\d)(\d)$")


def _detect_n_cols(images_dir: Path) -> int:
    """Infer tile grid width by checking whether any tile has col index >= 2."""
    for p in images_dir.glob("*.jpg"):
        m = _TILE_RE.match(p.stem)
        if m and int(m.group(3)) >= 2:
            return 3
    return 2


def _build_original_rows(cfg: dict, variant: str) -> list[dict]:
    root = Path(cfg["path"])
    rows: list[dict] = []
    for split in ("train", "val"):
        images_dir = root / cfg[split]
        labels_dir = root / cfg[split].replace("images/", "labels/", 1)
        for img_path in sorted(images_dir.glob("*.jpg")):
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            rows.append({
                "path":         str(img_path.resolve()),
                "label_path":   str(lbl_path.resolve()),
                "source_split": split,
                "variant":      variant,
                "source_image": "",
                "tile_index":   "",
            })
    return rows


def _build_tiled_rows(tiled_cfg: dict, src_cfg: dict, variant: str) -> list[dict]:
    tiled_root = Path(tiled_cfg["path"])
    src_root   = Path(src_cfg["path"])
    rows: list[dict] = []

    for split in ("train", "val"):
        images_dir     = tiled_root / "images" / split
        labels_dir     = tiled_root / "labels" / split
        src_images_dir = src_root / src_cfg[split]

        if not images_dir.exists():
            continue

        n_cols = _detect_n_cols(images_dir)

        for img_path in sorted(images_dir.glob("*.jpg")):
            m = _TILE_RE.match(img_path.stem)
            if not m:
                continue
            src_stem = m.group(1)
            row, col = int(m.group(2)), int(m.group(3))
            tile_index = row * n_cols + col

            lbl_path = labels_dir / f"{img_path.stem}.txt"
            src_img  = (src_images_dir / f"{src_stem}.jpg").resolve()

            rows.append({
                "path":         str(img_path.resolve()),
                "label_path":   str(lbl_path.resolve()),
                "source_split": split,
                "variant":      variant,
                "source_image": str(src_img),
                "tile_index":   tile_index,
            })

    return rows


def generate(
    data_yaml: Path,
    source_data_yaml: Path | None,
    variant: str,
    output_dir: Path,
) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text())

    if source_data_yaml is not None:
        src_cfg = yaml.safe_load(source_data_yaml.read_text())
        rows = _build_tiled_rows(cfg, src_cfg, variant)
    else:
        rows = _build_original_rows(cfg, variant)

    if not rows:
        print("WARNING: no rows written — check that image and label directories exist.")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "manifest.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows):,} rows written → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate-manifest",
        description="Generate a CSV manifest for a dataset variant.",
    )
    parser.add_argument(
        "--data-yaml", type=Path, required=True,
        help="data.yaml for the variant to manifest",
    )
    parser.add_argument(
        "--source-data-yaml", type=Path, default=None,
        help="For tiled variants: data.yaml of the original source dataset",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant key written into the manifest (default: data-yaml filename stem)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Directory to write manifest.csv into (default: same dir as --data-yaml)",
    )

    args = parser.parse_args()

    if not args.data_yaml.exists():
        print(f"ERROR: {args.data_yaml} not found"); sys.exit(1)
    if args.source_data_yaml is not None and not args.source_data_yaml.exists():
        print(f"ERROR: {args.source_data_yaml} not found"); sys.exit(1)

    variant    = args.variant or args.data_yaml.stem
    output_dir = args.output  or args.data_yaml.parent

    generate(args.data_yaml, args.source_data_yaml, variant, output_dir)


if __name__ == "__main__":
    main()
