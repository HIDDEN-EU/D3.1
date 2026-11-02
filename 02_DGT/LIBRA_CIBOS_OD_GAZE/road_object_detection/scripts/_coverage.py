"""
Shared tiling-survival ("coverage") computation.

Given a source dataset (via its manifest rows: image path + label path) and
the exact tiling parameters `tile-dataset` would use for `--mode r`, compute
what fraction of each class's source boxes survive into >= 1 output tile.

This is a generalisation of `output/measure_allclass_coverage.py` (which was
hardcoded to the ov12 dataset's names/paths/15 classes). The keep-test here
must stay byte-identical to `_project_labels`'s clip -> visibility -> min-size
-> remap logic, and tile geometry must come from `_compute_random_tiles`, so
`stats.json`'s `coverage` numbers agree with the standalone probe (tolerance <= 0.5pp).

Only mode 'r' is supported — coverage-as-a-question ("did this box survive
tiling") is meaningful for random-crop tiling where placement is stochastic;
modes 4/9 are deterministic full covers and every box is either fully outside
the crop or handled the same way every time, so this module (and `tile-dataset`
itself, see caller) only computes coverage for mode 'r'.
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

import cv2


def _covered(
    box: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    tiles: list[tuple[int, int, int, int]],
    min_vis: float,
    min_size: int,
) -> bool:
    """
    True if `box` (normalised xc, yc, w, h) survives into >= 1 tile under the
    exact keep-test `_project_labels` applies: clip to tile, visibility >=
    min_vis, clipped size >= min_size (when min_size > 0).
    """
    xc, yc, bw, bh = box
    xc_px, yc_px, bw_px, bh_px = xc * img_w, yc * img_h, bw * img_w, bh * img_h
    x1, x2 = xc_px - bw_px / 2, xc_px + bw_px / 2
    y1, y2 = yc_px - bh_px / 2, yc_px + bh_px / 2
    orig_area = bw_px * bh_px

    for tx, ty, tw, th in tiles:
        cx1, cx2 = max(x1, tx), min(x2, tx + tw)
        cy1, cy2 = max(y1, ty), min(y2, ty + th)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if orig_area > 0 and (cx2 - cx1) * (cy2 - cy1) / orig_area < min_vis:
            continue
        if min_size > 0 and ((cx2 - cx1) < min_size or (cy2 - cy1) < min_size):
            continue
        return True
    return False


def compute_coverage(
    manifest_rows: list[dict],
    split: str,
    names: dict[int, str],
    num_tiles: int,
    max_iou: float,
    min_visibility: float,
    min_size: int,
    seed: int | None,
    remap: dict[int, int | None] | None = None,
    guarantee_classes: set[int] | None = None,
) -> dict[str, dict]:
    """
    Compute per-class tiling survival for one split of a mode-r tiling run.

    `manifest_rows` — rows from a `generate-manifest`-style CSV (dicts with
    `path`, `label_path`, `source_split`) filtered to the source dataset (NOT
    a tiled variant's manifest — this replays tiling from scratch on the
    original images).
    `names` — {class_id: class_name} in SOURCE id space (before `remap`).
    `remap` — same `{source_id: target_id | None}` mapping `tile-dataset`
    computes from `class_merges`/`class_drops`; a box whose class remaps to
    `None` (dropped) is excluded from both numerator and denominator, since a
    dropped class was never a training target — coverage isn't a defined
    question for it. Pass `None` for datasets with no merges.
    `guarantee_classes` — same POST-remap id set passed to `tile-dataset`'s
    `--guarantee-classes`. When given AND `split == "train"`, the guaranteed
    crops `tile-dataset` appends are replayed here too, so the reported
    coverage matches the dataset actually written (guarantee is train-only, so
    val coverage is unaffected). `None` = plain random tiling.

    Returns `{class_name: {n_source, n_survived, pct_survived}}` — keyed by
    the SOURCE class name (pre-remap), so a merged-away class like
    `other_person` still gets its own row (its "survival" is into whichever
    tile ends up holding its merged-target label).
    """
    # Deferred import — avoids a circular import (tile_dataset imports this
    # module to reuse compute_coverage; this module reuses tile_dataset's
    # tile-geometry function).
    from scripts.tile_dataset import _compute_guarantee_tiles, _compute_random_tiles

    apply_guarantee = bool(guarantee_classes) and split == "train"

    totals: dict[int, int] = {cid: 0 for cid in names}
    survived: dict[int, int] = {cid: 0 for cid in names}

    for row in manifest_rows:
        if row.get("source_split") != split:
            continue
        label_path_s = row.get("label_path")
        if not label_path_s:
            continue
        label_path = Path(label_path_s)
        if not label_path.exists():
            continue
        text = label_path.read_text().strip()
        if not text:
            continue

        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if remap is not None and remap.get(cls) is None:
                continue
            boxes.append(
                (cls, (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            )
        if not boxes:
            continue

        img_path_s = row["path"]
        img = cv2.imread(img_path_s)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        h = int(hashlib.md5(img_path_s.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(h ^ (seed if seed is not None else 0))
        tiles = _compute_random_tiles(img_w, img_h, num_tiles, max_iou, rng)

        if apply_guarantee:
            parsed = [(cls, b[0], b[1], b[2], b[3]) for cls, b in boxes]
            tiles = tiles + _compute_guarantee_tiles(
                parsed, tiles, img_w, img_h, guarantee_classes,
                min_visibility, min_size, remap, Path(img_path_s).stem,
                quiet=True,
            )

        for cls, box in boxes:
            totals[cls] = totals.get(cls, 0) + 1
            if _covered(box, img_w, img_h, tiles, min_visibility, min_size):
                survived[cls] = survived.get(cls, 0) + 1

    result: dict[str, dict] = {}
    for cid, name in names.items():
        n_source = totals.get(cid, 0)
        n_ok = survived.get(cid, 0)
        pct = round(100.0 * n_ok / n_source, 2) if n_source > 0 else 0.0
        result[name] = {
            "n_source": n_source,
            "n_survived": n_ok,
            "pct_survived": pct,
        }
    return result


def load_manifest_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open() as fh:
        return list(csv.DictReader(fh))


def scan_source_rows(images_dir: Path, labels_dir: Path, split: str) -> list[dict]:
    """
    Build manifest-shaped rows `{path, label_path, source_split}` by scanning
    the filesystem directly — used when `tile-dataset` is invoked without
    `--manifest` (mirrors `_process_split`'s no-manifest filesystem-scan path,
    so coverage numbers are computed over the same image set that got tiled).
    """
    rows: list[dict] = []
    if not images_dir.exists():
        return rows
    existing_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
    for img_path in sorted(images_dir.glob("*.jpg")):
        lbl_s = str(labels_dir / f"{img_path.stem}.txt") if img_path.stem in existing_stems else ""
        rows.append({"path": str(img_path), "label_path": lbl_s, "source_split": split})
    return rows
