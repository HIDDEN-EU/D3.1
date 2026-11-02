"""
Tile-based dataset generator for BDD100K-style YOLO datasets.

Splits each source image into tiled crops — no padding, no black fill.
Tile dimensions are derived from the image size, so any resolution works.

  --mode 4  →  4 tiles  (2×2 grid, 640×360 for 1280×720 input)
               --overlap N extends each tile N pixels into its neighbour's region
               (default 0; N=64 gives tiles of 704×424 with a 128 px shared strip)

  --mode 9  →  9 tiles  (3×3 sliding window, 50 % overlap, 640×360 tiles for 1280×720)
               Positions: x ∈ {0, 320, 640},  y ∈ {0, 180, 360} — covers the full
               image exactly with no black fill and no out-of-bounds crops.

  --mode r  →  N random crops per image (default 4, set with --num-tiles)
               Tile size: uniform in [img_w/4, img_w/2] × [img_h/4, img_h/2]
               Tile position: uniform random over the full image (always in-bounds).
               Overlap is capped: a candidate tile is rejected if its IoU with any
               already-placed tile exceeds --max-iou (default 0.2); up to 200 retries,
               then placed regardless to avoid infinite loops.
               --seed N  global seed; per-image seed = MD5(path) XOR seed.

YOLO bounding boxes are re-projected into each tile.  Boxes whose visible
area after clipping is below --min-visibility (default 0.3) are dropped.
Empty tile label files are still written (YOLO expects them).

Usage
-----
tile-dataset \\
    --data-yaml configs/data.yaml \\
    --output    data/bdd100k_tiled_4 \\
    --mode      4

tile-dataset \\
    --data-yaml configs/data.yaml \\
    --output    data/bdd100k_tiled_9 \\
    --mode      9  --splits train  --workers 8

tile-dataset \\
    --data-yaml configs/data.yaml \\
    --output    data/bdd100k_tiled_4_ov64 \\
    --mode      4  --overlap 64
"""

from __future__ import annotations

import argparse
import sys
import csv
import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

from scripts import dataset_stats
from scripts._coverage import _covered, compute_coverage, load_manifest_rows, scan_source_rows


# ---------------------------------------------------------------------------
# Tile geometry  (computed at runtime from actual image dimensions)
# ---------------------------------------------------------------------------

def _compute_tiles(
    img_w: int,
    img_h: int,
    mode: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """
    Return (x, y, tile_w, tile_h) for every tile.

    Mode 4 — 2×2 grid:
      tile_w = img_w//2 + overlap   (clamped to img_w)
      tile_h = img_h//2 + overlap   (clamped to img_h)
      Tiles are anchored to the four corners so the full image is always covered.

    Mode 9 — 3×3 sliding window, 50 % overlap:
      tile_w = img_w//2,  stride_x = tile_w//2
      tile_h = img_h//2,  stride_y = tile_h//2
      For 1280×720: tile=640×360, stride=320×180 → x∈{0,320,640}, y∈{0,180,360}
    """
    if mode == 4:
        tw = min(img_w // 2 + overlap, img_w)
        th = min(img_h // 2 + overlap, img_h)
        return [
            (0,          0,          tw, th),
            (img_w - tw, 0,          tw, th),
            (0,          img_h - th, tw, th),
            (img_w - tw, img_h - th, tw, th),
        ]

    # mode 9 — 50 % overlap, 3×3
    tw = img_w // 2
    th = img_h // 2
    sx = tw // 2
    sy = th // 2
    return [
        (c * sx, r * sy, tw, th)
        for r in range(3)
        for c in range(3)
    ]


# ---------------------------------------------------------------------------
# Random-crop tile geometry
# ---------------------------------------------------------------------------

_MAX_PLACEMENT_TRIES = 200


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw);  iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter)


def _compute_random_tiles(
    img_w: int,
    img_h: int,
    n_tiles: int,
    max_iou: float,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    """
    Return (x, y, tile_w, tile_h) for n_tiles non-overlapping random crops.

    Tile size is uniform in [img_w/4, img_w/2] x [img_h/4, img_h/2].
    Position is uniform over [0, img_w-tw] x [0, img_h-th] (always in-bounds).
    A candidate is rejected if IoU with any already-placed tile exceeds max_iou.
    After _MAX_PLACEMENT_TRIES retries the last candidate is accepted anyway.
    """
    placed: list[tuple[int, int, int, int]] = []
    for _ in range(n_tiles):
        candidate = None
        for _ in range(_MAX_PLACEMENT_TRIES):
            tw = int(rng.uniform(img_w / 4, img_w / 2))
            th = int(rng.uniform(img_h / 4, img_h / 2))
            tx = int(rng.uniform(0, img_w - tw))
            ty = int(rng.uniform(0, img_h - th))
            candidate = (tx, ty, tw, th)
            if all(_iou(candidate, p) <= max_iou for p in placed):
                break
        placed.append(candidate)
    return placed


# ---------------------------------------------------------------------------
# Guaranteed-crop placement (mode r, train split only)
# ---------------------------------------------------------------------------

def _parse_yolo_label(label_path_s: str) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into `[(cls, xc, yc, bw, bh), ...]` (source
    normalised coords). Malformed lines are skipped."""
    text = Path(label_path_s).read_text().strip()
    if not text:
        return []
    parsed: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        parsed.append((int(parts[0]), float(parts[1]), float(parts[2]),
                       float(parts[3]), float(parts[4])))
    return parsed


def _compute_guarantee_tiles(
    parsed_boxes: list[tuple[int, float, float, float, float]],
    existing_tiles: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    guarantee_classes: set[int],
    min_vis: float,
    min_size: int,
    remap: dict[int, int | None] | None,
    stem: str,
    quiet: bool = False,
) -> list[tuple[int, int, int, int]]:
    """
    Return extra crops that guarantee coverage of in-scope guarantee-set boxes.

    For each source box whose post-remap class is in `guarantee_classes` and
    which is not already covered at >= `min_vis` by any existing tile, append one
    crop centred on the box (sized to the largest normal tile, img/2 per side,
    expanded to the full image dimension for an oversized box). Additive only:
    never removes or mutates an existing tile, never touches the RNG, returns
    only the newly-placed tiles.

    In-scope = both source pixel dims >= `min_size` (a smaller box cannot
    survive `_project_labels`' min_size clip even in a perfectly centred crop,
    so a guaranteed crop for it would inject noise rather than signal). Boxes
    already covered (by a random tile OR an earlier guarantee tile placed in
    this call) are skipped — one crop can rescue several nearby boxes. Boxes are
    processed in source-file order so this dedup is deterministic.
    """
    guaranteed: list[tuple[int, int, int, int]] = []
    seen_tiles = list(existing_tiles)
    for cls, xc, yc, bw, bh in parsed_boxes:
        post_cls = remap[cls] if (remap and cls in remap) else cls
        if post_cls is None or post_cls not in guarantee_classes:
            continue
        bw_px, bh_px = bw * img_w, bh * img_h
        if min_size > 0 and (bw_px < min_size or bh_px < min_size):
            continue  # sub-min_size: unrecoverable, out of scope by design
        if _covered((xc, yc, bw, bh), img_w, img_h, seen_tiles, min_vis, min_size):
            continue
        tw, th = img_w // 2, img_h // 2
        if bw_px > tw or bh_px > th:
            if not quiet:
                print(f"[WARN] guarantee: oversized box in {stem}, using best-effort crop")
            if bw_px > tw:
                tw = img_w
            if bh_px > th:
                th = img_h
        xc_px, yc_px = xc * img_w, yc * img_h
        tx = max(0, min(int(round(xc_px - tw / 2)), img_w - tw))
        ty = max(0, min(int(round(yc_px - th / 2)), img_h - th))
        tile = (tx, ty, tw, th)
        guaranteed.append(tile)
        seen_tiles.append(tile)
    return guaranteed


# ---------------------------------------------------------------------------
# Occlusion sidecar — source CSV loader
# ---------------------------------------------------------------------------

def _load_source_occlusion(csv_path: Path) -> dict[str, dict[int, tuple[bool, bool]]]:
    """
    Load a source `occlusion.csv` (written by `bdd100k convert`) into
    `{source_stem -> {box_index -> (occluded, truncated)}}`.

    Returns an empty dict if the file is missing — callers treat that as
    "occlusion propagation disabled for this split".
    """
    if not csv_path.exists():
        return {}
    by_stem: dict[str, dict[int, tuple[bool, bool]]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            stem      = row["stem"]
            box_index = int(row["box_index"])
            occluded  = row["occluded"].strip().lower()  == "true"
            truncated = row["truncated"].strip().lower() == "true"
            by_stem.setdefault(stem, {})[box_index] = (occluded, truncated)
    return by_stem


# ---------------------------------------------------------------------------
# Label re-projection
# ---------------------------------------------------------------------------

def _project_labels(
    label_path: str,
    tiles: list[tuple[int, int, int, int]],
    src_w: int,
    src_h: int,
    min_vis: float,
    min_size: int = 0,
    remap: dict[int, int | None] | None = None,
    src_occlusion: dict[int, tuple[bool, bool]] | None = None,
    _timing: dict | None = None,
) -> dict[int, list[tuple[str, tuple[bool, bool] | None]]]:
    """
    Re-project YOLO labels (normalised to source image) into each tile.

    Returns `{tile_index -> [(yolo_line, (occluded, truncated) | None), ...]}`.
    The second element is `None` when no `src_occlusion` mapping was provided
    for the source box (or when the source box wasn't in the CSV).
    """
    result: dict[int, list[tuple[str, tuple[bool, bool] | None]]] = {
        i: [] for i in range(len(tiles))
    }

    t0 = time.perf_counter()
    text = Path(label_path).read_text().strip()
    if _timing is not None:
        _timing["t_label_read"] = _timing.get("t_label_read", 0.0) + (time.perf_counter() - t0)
    if not text:
        return result

    parsed: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = (float(v) for v in parts[1:])
        parsed.append((cls, xc, yc, bw, bh))

    t1 = time.perf_counter()
    for tile_idx, (tx, ty, tw, th) in enumerate(tiles):
        lines: list[tuple[str, tuple[bool, bool] | None]] = []
        for src_idx, (cls, xc, yc, bw, bh) in enumerate(parsed):
            # To pixel coords in source image
            xc_px = xc * src_w
            yc_px = yc * src_h
            bw_px = bw * src_w
            bh_px = bh * src_h

            x1 = xc_px - bw_px / 2
            x2 = xc_px + bw_px / 2
            y1 = yc_px - bh_px / 2
            y2 = yc_px + bh_px / 2

            # Clip to tile
            cx1 = max(x1, tx)
            cx2 = min(x2, tx + tw)
            cy1 = max(y1, ty)
            cy2 = min(y2, ty + th)

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            # Visibility filter
            orig_area = bw_px * bh_px
            if orig_area > 0 and (cx2 - cx1) * (cy2 - cy1) / orig_area < min_vis:
                continue

            # Absolute-size filter — drop boxes too small for the detector
            if min_size > 0 and ((cx2 - cx1) < min_size or (cy2 - cy1) < min_size):
                continue

            # Normalise to tile
            new_xc = ((cx1 + cx2) / 2 - tx) / tw
            new_yc = ((cy1 + cy2) / 2 - ty) / th
            new_w  = (cx2 - cx1) / tw
            new_h  = (cy2 - cy1) / th

            out_cls = remap[cls] if remap and cls in remap else cls
            if out_cls is None:        # dropped class — discard the box (becomes background)
                continue
            yolo_line = f"{out_cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
            occ_flags = src_occlusion.get(src_idx) if src_occlusion is not None else None
            lines.append((yolo_line, occ_flags))

        result[tile_idx] = lines

    if _timing is not None:
        _timing["t_project"] = _timing.get("t_project", 0.0) + (time.perf_counter() - t1)

    return result


# ---------------------------------------------------------------------------
# Per-image worker (module-level so it's picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_image(args: tuple) -> tuple[int, int, dict, list]:
    """
    Tile one image, write crops + label files. Returns
    `(n_imgs, n_labels, timings, occ_rows)` where `occ_rows` is a list of
    `(tile_stem, tile_box_index, occluded, truncated)` tuples for any tile
    boxes whose source box had occlusion metadata.
    """
    img_path_s, label_path_s, out_img_s, out_lbl_s, mode, overlap, cols, min_vis, \
        min_size, num_tiles, max_iou, seed, remap, src_occlusion, split, \
        guarantee_classes, max_background_ratio = args

    timings: dict[str, float] = {
        "t_imread": 0.0, "t_label_read": 0.0, "t_project": 0.0,
        "t_imwrite": 0.0, "t_label_write": 0.0,
    }
    occ_rows: list[tuple[str, int, bool, bool]] = []

    t0 = time.perf_counter()
    img = cv2.imread(img_path_s)
    timings["t_imread"] = time.perf_counter() - t0

    if img is None:
        return 0, 0, timings, occ_rows

    src_h, src_w = img.shape[:2]
    if mode == 'r':
        h = int(hashlib.md5(img_path_s.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(h ^ (seed if seed is not None else 0))
        tiles = _compute_random_tiles(src_w, src_h, num_tiles, max_iou, rng)
        # Guaranteed crops: additive, train split only, never touch the RNG.
        if guarantee_classes and split == "train" and label_path_s is not None:
            tiles = tiles + _compute_guarantee_tiles(
                _parse_yolo_label(label_path_s), tiles, src_w, src_h,
                guarantee_classes, min_vis, min_size, remap,
                Path(img_path_s).stem,
            )
    else:
        tiles = _compute_tiles(src_w, src_h, mode, overlap)
    label_map = (
        _project_labels(
            label_path_s, tiles, src_w, src_h, min_vis, min_size, remap,
            src_occlusion=src_occlusion, _timing=timings,
        )
        if label_path_s is not None
        else {i: [] for i in range(len(tiles))}
    )

    # Background cap.
    # Trim redundant empty tiles on source images that already produced
    # foreground. Purely subtractive and RNG-free: selection order is
    # `tile_idx`, so output stays deterministic for a given --seed. Images with
    # no non-empty tile (n_nonempty == 0) are BDD100K's intentional hard
    # negatives and are left entirely untouched, by design.
    skip_tiles: set[int] = set()
    if max_background_ratio is not None:
        n_nonempty = sum(1 for i in range(len(tiles)) if label_map[i])
        if n_nonempty > 0:
            budget = round(max_background_ratio * n_nonempty)
            for i in range(len(tiles)):
                if label_map[i]:
                    continue          # never drop a labelled tile
                if budget > 0:
                    budget -= 1       # keep this empty tile against the budget
                else:
                    skip_tiles.add(i)

    stem = Path(img_path_s).stem
    n_imgs = n_labels = 0

    for tile_idx, (tx, ty, tw, th) in enumerate(tiles):
        if tile_idx in skip_tiles:
            continue
        tile_name = (
            f"{stem}_r{tile_idx:02d}"
            if mode == 'r'
            else f"{stem}_t{tile_idx // cols}{tile_idx % cols}"
        )

        t1 = time.perf_counter()
        cv2.imwrite(
            f"{out_img_s}/{tile_name}.jpg",
            img[ty : ty + th, tx : tx + tw],
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        timings["t_imwrite"] += time.perf_counter() - t1
        n_imgs += 1

        entries    = label_map[tile_idx]
        yolo_lines = [e[0] for e in entries]

        t2 = time.perf_counter()
        Path(f"{out_lbl_s}/{tile_name}.txt").write_text("\n".join(yolo_lines))
        timings["t_label_write"] += time.perf_counter() - t2
        n_labels += 1

        # Tile-box index is the line index in the just-written .txt file.
        for tile_box_idx, (_, occ) in enumerate(entries):
            if occ is not None:
                occ_rows.append((tile_name, tile_box_idx, occ[0], occ[1]))

    return n_imgs, n_labels, timings, occ_rows


# ---------------------------------------------------------------------------
# Split processor
# ---------------------------------------------------------------------------

def _process_split(
    images_dir: Path,
    labels_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    mode: int | str,
    overlap: int,
    min_vis: float,
    split: str,
    manifest: Path | None,
    max_images: int | None = None,
    workers: int = 1,
    min_size: int = 0,
    num_tiles: int | None = None,
    max_iou: float | None = None,
    seed: int | None = None,
    remap: dict[int, int | None] | None = None,
    guarantee_classes: set[int] | None = None,
    max_background_ratio: float | None = None,
) -> dict:
    """Returns a stats dict for README generation."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Occlusion propagation: locate the source CSV and load it if present.
    #
    # - Manifest mode: the recipe's `path:` is the output dir, so `labels_dir`
    #   does NOT point at the source labels. Derive the source labels dir from
    #   the first manifest row that has a `label_path`.
    # - Non-manifest mode: `labels_dir` (passed in by cmd_tile) is the source.
    src_occ_csv = labels_dir / "occlusion.csv"
    if manifest is not None:
        with manifest.open() as fh:
            for row in csv.DictReader(fh):
                if row["source_split"] == split and row["label_path"]:
                    src_occ_csv = Path(row["label_path"]).parent / "occlusion.csv"
                    break

    src_occlusion_by_stem = _load_source_occlusion(src_occ_csv)
    if src_occlusion_by_stem:
        total_rows = sum(len(v) for v in src_occlusion_by_stem.values())
        print(f"  occlusion: loaded {total_rows:,} rows from {src_occ_csv}")
    else:
        print(f"  [WARN] occlusion sidecar not found at {src_occ_csv} — skipping propagation")

    cols      = 3 if mode == 9 else 2  # unused for mode 'r'
    out_img_s = str(out_img_dir)
    out_lbl_s = str(out_lbl_dir)
    tasks     = []
    background = 0

    def _build_task(img_path_s: str, lbl_s: str | None) -> tuple:
        src_stem = Path(img_path_s).stem
        src_occ  = src_occlusion_by_stem.get(src_stem) if src_occlusion_by_stem else None
        return (img_path_s, lbl_s, out_img_s, out_lbl_s, mode, overlap, cols, min_vis,
                min_size, num_tiles, max_iou, seed, remap, src_occ, split,
                guarantee_classes, max_background_ratio)

    if manifest is not None:
        # Fast path — read image/label paths from manifest; no filesystem scan.
        with manifest.open() as fh:
            for row in csv.DictReader(fh):
                if row["source_split"] != split:
                    continue
                lbl_s = row["label_path"] if row["label_path"] else None
                if lbl_s is None:
                    background += 1
                tasks.append(_build_task(row["path"], lbl_s))
        print(f"  {len(tasks)} images from manifest  ({background} background)")
    else:
        # Slow path — scan filesystem (two full directory listings on disk).
        image_paths = sorted(images_dir.glob("*.jpg"))
        if not image_paths:
            print(f"  [WARN] No .jpg images found in {images_dir}")
            return {
                "source_images": 0, "background": 0, "output_tiles": 0,
                "occlusion_propagated": False, "occlusion_rows": 0,
            }

        existing_stems = {p.stem for p in labels_dir.glob("*.txt")}
        for img_path in image_paths:
            lbl_s = str(labels_dir / f"{img_path.stem}.txt") if img_path.stem in existing_stems else None
            if lbl_s is None:
                background += 1
            tasks.append(_build_task(str(img_path), lbl_s))
        print(f"  {len(tasks)} images to tile  ({background} background — no label file, included as hard negatives)")

    if max_images is not None:
        tasks = tasks[:max_images]

    n_tasks = len(tasks)
    total_imgs = total_labels = done = 0
    stage_totals: dict[str, float] = {
        "t_imread": 0.0, "t_label_read": 0.0, "t_project": 0.0,
        "t_imwrite": 0.0, "t_label_write": 0.0,
    }
    all_occ_rows: list[tuple[str, int, bool, bool]] = []
    wall_start = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_image, task): task for task in tasks}
            for fut in as_completed(futures):
                ni, nl, timings, occ_rows = fut.result()
                total_imgs   += ni
                total_labels += nl
                all_occ_rows.extend(occ_rows)
                for k in stage_totals:
                    stage_totals[k] += timings[k]
                done += 1
                if done % 1000 == 0 or done == n_tasks:
                    print(f"    {done}/{n_tasks} images processed …")
    else:
        for task in tasks:
            ni, nl, timings, occ_rows = _process_image(task)
            total_imgs   += ni
            total_labels += nl
            all_occ_rows.extend(occ_rows)
            for k in stage_totals:
                stage_totals[k] += timings[k]
            done += 1
            if done % 1000 == 0 or done == n_tasks:
                print(f"    {done}/{n_tasks} images processed …")

    wall_total = time.perf_counter() - wall_start
    stage_sum  = sum(stage_totals.values())
    throughput = done / wall_total if wall_total > 0 else 0.0

    labels = {
        "t_imread":      "imread      ",
        "t_label_read":  "label_read  ",
        "t_project":     "project     ",
        "t_imwrite":     "imwrite     ",
        "t_label_write": "label_write ",
    }
    print(f"\n── Timing summary ({done} images) ──────────────────────────────")
    for k, label in labels.items():
        t = stage_totals[k]
        pct = 100 * t / stage_sum if stage_sum > 0 else 0.0
        avg_ms = 1000 * t / done if done > 0 else 0.0
        print(f"  {label}  total={t:6.1f} s   avg={avg_ms:6.1f} ms/img   ({pct:4.1f} %)")
    overhead = wall_total - stage_sum
    print(f"  overhead     total={overhead:6.1f} s   (scheduling / idle)")
    print(f"  TOTAL (wall) total={wall_total:6.1f} s   throughput: {throughput:.1f} img/s")
    print(f"──────────────────────────────────────────────────────────────\n")

    n_tiles = num_tiles if mode == 'r' else mode
    print(f"  Done — {total_imgs} tile images, {total_labels} label files "
          f"({n_tiles} tiles per source image)")

    # Write the tiled occlusion sidecar — only when source data was available.
    occlusion_propagated = bool(src_occlusion_by_stem)
    if occlusion_propagated:
        out_occ_csv = out_lbl_dir / "occlusion.csv"
        all_occ_rows.sort()  # deterministic order: (tile_stem, tile_box_index)
        with out_occ_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["stem", "box_index", "occluded", "truncated"])
            for stem, idx, occ, trc in all_occ_rows:
                writer.writerow([stem, idx, occ, trc])
        print(f"  occlusion: wrote {len(all_occ_rows):,} rows → {out_occ_csv}")

    return {
        "source_images":        len(tasks),
        "background":           background,
        "output_tiles":         total_imgs,
        "occlusion_propagated": occlusion_propagated,
        "occlusion_rows":       len(all_occ_rows) if occlusion_propagated else 0,
    }


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def _write_readme(
    args: argparse.Namespace,
    src_cfg: dict,
    src_root: Path,
    split_stats: dict[str, dict],
    out_nc: int,
    out_names: dict[int, str],
) -> None:
    tile_desc = {
        4:   "2×2 grid — 4 tiles of 640×360 per source image",
        9:   "3×3 sliding window — 9 tiles of 640×360, 50 % overlap (stride 320×180)",
        'r': (
            f"{args.num_tiles} random crops per image — uniform random position, "
            f"size uniform in [img/4, img/2], max IoU {args.max_iou}"
        ),
    }[args.mode]

    overlap_line = (
        f"- **Overlap:** {args.overlap} px per edge "
        f"(tile size with overlap: {640 + args.overlap}×{360 + args.overlap})\n"
        if args.mode == 4 and args.overlap > 0
        else "- **Overlap:** none (mode 4 default)\n"
        if args.mode == 4
        else f"- **Overlap:** capped at IoU ≤ {args.max_iou} (rejection sampling)\n"
        if args.mode == 'r'
        else "- **Overlap:** 50 % in both dimensions (stride 320×180)\n"
    )

    reproduce_cmd = (
        f"tile-dataset \\\n"
        f"    --data-yaml {args.data_yaml} \\\n"
        f"    --output    {args.output} \\\n"
        f"    --mode      {args.mode}"
        + (f" \\\n    --overlap   {args.overlap}" if args.mode == 4 and args.overlap else "")
        + (f" \\\n    --num-tiles {args.num_tiles}" if args.mode == 'r' and args.num_tiles != 4 else "")
        + (f" \\\n    --max-iou   {args.max_iou}" if args.mode == 'r' and args.max_iou != 0.2 else "")
        + (f" \\\n    --seed      {args.seed}" if args.seed is not None else "")
        + (f" \\\n    --splits    {' '.join(args.splits)}" if args.splits != ["train", "val"] else "")
        + (f" \\\n    --min-visibility {args.min_visibility}" if args.min_visibility != 0.3 else "")
        + (f" \\\n    --min-size       {args.min_size}" if args.min_size > 0 else "")
        + (f" \\\n    --guarantee-classes {args.guarantee_classes}" if args.guarantee_classes else "")
        + (f" \\\n    --max-background-ratio {args.max_background_ratio}"
           if args.max_background_ratio is not None else "")
    )

    guarantee_line = (
        f"- **Guarantee classes:** {args.guarantee_classes} "
        f"(train split — one additive crop per uncovered in-scope box)\n"
        if args.guarantee_classes
        else ""
    )

    background_cap_line = (
        f"- **Max background ratio:** `{args.max_background_ratio}` "
        f"(empty tiles kept per foreground source image, as a ratio of that "
        f"image's non-empty tiles; images with no non-empty tile are exempt)\n"
        if args.max_background_ratio is not None
        else ""
    )

    stats_rows = ""
    for split, s in split_stats.items():
        occ_cell = (
            f"{s['occlusion_rows']:>10,}" if s.get("occlusion_propagated")
            else f"{'—':>10}"
        )
        stats_rows += (
            f"| {split:5} | {s['source_images']:>13,} | {s['background']:>10,} "
            f"| {s['output_tiles']:>12,} | {occ_cell} |\n"
        )

    any_occlusion = any(s.get("occlusion_propagated") for s in split_stats.values())
    occlusion_line = (
        "- **Occlusion sidecar:** written to `labels/<split>/occlusion.csv` "
        "(per-box `occluded` / `truncated` flags carried from source)\n"
        if any_occlusion
        else "- **Occlusion sidecar:** not generated (source `occlusion.csv` was not found)\n"
    )

    readme = (
        f"# Tiled dataset\n\n"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"## Parameters\n\n"
        f"- **Source:** `{src_root}`\n"
        f"- **Source config:** `{args.data_yaml}`\n"
        f"- **Mode:** {args.mode} — {tile_desc}\n"
        f"{overlap_line}"
        f"- **Min visibility:** {args.min_visibility}\n"
        + (f"- **Min box size:** {args.min_size} px (drop clipped boxes smaller than this on either side)\n" if args.min_size > 0 else "")
        + f"- **Classes:** {out_nc} ({', '.join(out_names.values())})\n"
        + guarantee_line
        + background_cap_line
        + occlusion_line
        + "\n"
        f"## Output stats\n\n"
        f"| Split | Source images | Background | Output tiles | Occlusion rows |\n"
        f"|-------|--------------|------------|--------------|----------------|\n"
        f"{stats_rows}\n"
        f"## Reproduce\n\n"
        f"```bash\n{reproduce_cmd}\n```\n\n"
        f"## Use for training\n\n"
        f"Point your experiment JSON at `{args.output}/data.yaml`.\n"
    )

    readme_path = args.output / "README.md"
    readme_path.write_text(readme)
    print(f"README.md written  → {readme_path}")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def cmd_tile(args: argparse.Namespace) -> None:
    if args.mode in ('4', '9'):
        args.mode = int(args.mode)

    if not args.data_yaml.exists():
        print(f"ERROR: {args.data_yaml} not found"); sys.exit(1)

    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            print(
                f"ERROR: output directory already exists and is not empty:\n"
                f"  {args.output}\n"
                f"Use --overwrite to write over it, or choose a different --output path."
            )
            sys.exit(1)
        print(f"  [overwrite] existing dataset at {args.output} will be overwritten in place")

    cfg  = yaml.safe_load(args.data_yaml.read_text())
    root = Path(cfg["path"])

    # Compute class remap if the source yaml declares class_merges and/or class_drops.
    # A merged class's boxes are relabeled to the target; a dropped class's boxes are
    # discarded entirely (remap → None, skipped at write time). Both kinds of source class
    # are removed from the output schema and the survivors renumbered to a contiguous range.
    remap: dict[int, int | None] | None = None
    out_nc    = cfg["nc"]
    out_names = cfg["names"]
    if cfg.get("class_merges") or cfg.get("class_drops"):
        raw_merges = {int(k): int(v) for k, v in (cfg.get("class_merges") or {}).items()}
        drops      = {int(c) for c in (cfg.get("class_drops") or [])}
        clash      = drops & set(raw_merges.values())
        if clash:
            print(f"ERROR: class_drops {sorted(clash)} are also merge targets — cannot "
                  f"drop a class that other classes' boxes are merged into"); sys.exit(1)
        kept       = [i for i in range(cfg["nc"]) if i not in raw_merges and i not in drops]
        renumber   = {old: new for new, old in enumerate(kept)}
        remap      = {i: (None if i in drops
                          else renumber[raw_merges[i]] if i in raw_merges
                          else renumber[i])
                      for i in range(cfg["nc"])}
        out_names  = {renumber[i]: cfg["names"][i] for i in kept}
        out_nc     = len(kept)

    # Guarantee-set class ids are given in POST-remap (output) id space.
    guarantee_classes: set[int] | None = (
        {int(c) for c in args.guarantee_classes.split(",")}
        if args.guarantee_classes else None
    )

    tile_desc = {
        4:   "2×2 grid, 640×360 per tile (1280×720 source)",
        9:   "3×3 sliding window, 50 % overlap, 640×360 per tile (1280×720 source)",
        'r': f"{args.num_tiles} random crops per image (uniform, max IoU {args.max_iou})",
    }[args.mode]
    overlap_note = f"  overlap : {args.overlap} px per edge\n" if args.mode == 4 else ""

    print(f"\ntile-dataset  mode={args.mode}  — {tile_desc}")
    print(f"  source  : {root}")
    print(f"  output  : {args.output}")
    print(f"  splits  : {args.splits}")
    print(f"{overlap_note}  min-vis : {args.min_visibility}")
    if remap is not None:
        parts = []
        if raw_merges:
            parts.append("merges " + ", ".join(
                f"{cfg['names'][s]}→{cfg['names'][t]}" for s, t in raw_merges.items()))
        if drops:
            parts.append("drops " + ", ".join(f"{cfg['names'][d]}" for d in sorted(drops)))
        print(f"  remap   : {'; '.join(parts)}  (nc {cfg['nc']} → {out_nc})")
    if guarantee_classes is not None:
        names_str = ", ".join(
            f"{cid}={out_names.get(cid, '?')}" for cid in sorted(guarantee_classes))
        print(f"  guarantee: {names_str}  (train split — one crop per uncovered in-scope box)")
    if args.max_background_ratio is not None:
        print(f"  bg cap   : {args.max_background_ratio} "
              f"(empty tiles per foreground image; hard negatives exempt)")
    print()

    split_stats: dict[str, dict] = {}
    for split in args.splits:
        images_dir = root / cfg[split]
        labels_dir = root / "labels" / "100k" / split
        out_img    = args.output / "images" / split
        out_lbl    = args.output / "labels" / split

        print(f"── {split} ──")
        split_stats[split] = _process_split(
            images_dir, labels_dir, out_img, out_lbl,
            args.mode, args.overlap, args.min_visibility,
            split=split, manifest=args.manifest,
            max_images=args.max_images,
            workers=args.workers,
            min_size=args.min_size,
            num_tiles=args.num_tiles if args.mode == 'r' else None,
            max_iou=args.max_iou if args.mode == 'r' else None,
            seed=args.seed,
            remap=remap,
            guarantee_classes=guarantee_classes,
            max_background_ratio=args.max_background_ratio,
        )

    out_yaml = args.output / "data.yaml"
    out_cfg  = {
        "path":  str(args.output),
        "train": "images/train",
        "val":   "images/val",
        "nc":    out_nc,
        "names": out_names,
    }
    out_yaml.write_text(yaml.dump(out_cfg, allow_unicode=True, sort_keys=False))
    print(f"\ndata.yaml written → {out_yaml}")

    _write_readme(args, cfg, root, split_stats, out_nc, out_names)

    _write_stats_json(args, cfg, root, remap, out_cfg)

    print("Done.")


# ---------------------------------------------------------------------------
# stats.json — counts/occlusion (via dataset_stats) + coverage (source replay)
# ---------------------------------------------------------------------------

def _write_stats_json(
    args: argparse.Namespace,
    src_cfg: dict,
    src_root: Path,
    remap: dict[int, int | None] | None,
    out_cfg: dict,
) -> None:
    """
    Auto-emit `<output>/stats.json`: counts + occlusion computed on the
    OUTPUT dataset via `dataset_stats.build_report` (guarantees exact-match
    counts with a standalone `dataset-stats` run — contract criterion 3), plus
    per-class `coverage` computed on the SOURCE via the shared coverage
    function (contract criterion 5), injected into each split's dict.

    Coverage is computed unconditionally for `--mode r` runs (the only mode
    the contract's coverage question applies to — see `scripts/_coverage.py`
    module docstring). Modes 4/9 get `coverage: None` per split; they still
    get the rest of the schema.
    """
    report = dataset_stats.build_report(
        out_cfg, args.output, args.splits, args.workers,
    )

    if args.mode == 'r':
        src_names = {int(k): v for k, v in src_cfg["names"].items()}
        guarantee_classes: set[int] | None = (
            {int(c) for c in args.guarantee_classes.split(",")}
            if args.guarantee_classes else None
        )
        for split in args.splits:
            if split not in report["splits"] or not report["splits"][split]:
                continue
            if args.manifest is not None:
                rows = [
                    r for r in load_manifest_rows(args.manifest)
                    if r.get("source_split") == split
                ]
            else:
                images_dir = src_root / src_cfg[split]
                labels_dir = src_root / "labels" / "100k" / split
                rows = scan_source_rows(images_dir, labels_dir, split)
            if args.max_images is not None:
                rows = rows[: args.max_images]

            coverage = compute_coverage(
                manifest_rows=rows,
                split=split,
                names=src_names,
                num_tiles=args.num_tiles,
                max_iou=args.max_iou,
                min_visibility=args.min_visibility,
                min_size=args.min_size,
                seed=args.seed,
                remap=remap,
                guarantee_classes=guarantee_classes,
            )
            report["splits"][split]["coverage"] = coverage

    out_path = args.output / "stats.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"stats.json written → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tile-dataset",
        description=(
            "Generate a tiled dataset from BDD100K-style YOLO data.  "
            "No padding — tiles are derived from the natural image dimensions."
        ),
    )
    parser.add_argument(
        "--data-yaml", type=Path, required=True,
        help="Source data.yaml (e.g. configs/data.yaml)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Destination directory for the tiled dataset",
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["4", "9", "r"],
        help="4 = 2×2 grid  |  9 = 3×3 sliding window with 50 %% overlap  |  r = random crops",
    )
    parser.add_argument(
        "--overlap", type=int, default=0,
        help=(
            "Mode-4 only: pixels each tile extends into its neighbour's region "
            "(default 0).  E.g. --overlap 64 gives 704×424 tiles with a "
            "128 px shared strip at each seam."
        ),
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val"],
        help="Splits to process (default: train val)",
    )
    parser.add_argument(
        "--min-visibility", type=float, default=0.3,
        help="Min fraction of box area that must remain in a tile (default 0.3)",
    )
    parser.add_argument(
        "--min-size", type=int, default=0,
        help=(
            "Drop clipped boxes whose width or height is below this many pixels "
            "(default 0 = disabled). Use 16 to filter sub-YOLO-threshold boxes."
        ),
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help=(
            "Path to a manifest.csv generated by generate-manifest. "
            "When provided, image and label paths are read from the manifest "
            "instead of scanning the filesystem (much faster on slow drives)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Write over an existing output directory",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Cap the number of source images processed per split (useful for profiling)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--num-tiles", type=int, default=4,
        help="Mode r only: number of random crops per image (default: 4)",
    )
    parser.add_argument(
        "--max-iou", type=float, default=0.2,
        help="Mode r only: maximum IoU allowed between any two placed tiles (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Global random seed; per-image seed = MD5(path) XOR seed (default: no seed)",
    )
    parser.add_argument(
        "--guarantee-classes", type=str, default=None,
        help=(
            "Mode r only: comma-separated POST-remap class ids (e.g. '1,5,6'). "
            "For each uncovered in-scope box of these classes in the TRAIN split, "
            "append one additive crop centred on it so the box is guaranteed to "
            "reach >= 1 tile. Val tiling is never affected."
        ),
    )

    parser.add_argument(
        "--max-background-ratio", type=float, default=None,
        help=(
            "Mode r only: cap empty (zero-box) tiles kept per FOREGROUND source "
            "image at round(F * n_nonempty) of that image's non-empty tiles. "
            "Purely subtractive and RNG-free (tiles walked in index order), so "
            "output stays reproducible for a given --seed. Source images that "
            "produce no non-empty tile are hard negatives and are left untouched. "
            "Default: keep every empty tile."
        ),
    )

    args = parser.parse_args()

    if args.overlap != 0 and args.mode != "4":
        parser.error("--overlap is only valid with --mode 4")
    if args.mode != "r":
        if args.num_tiles != 4:
            parser.error("--num-tiles is only valid with --mode r")
        if args.max_iou != 0.2:
            parser.error("--max-iou is only valid with --mode r")
        if args.guarantee_classes is not None:
            parser.error("--guarantee-classes is only valid with --mode r")
        if args.max_background_ratio is not None:
            parser.error("--max-background-ratio is only valid with --mode r")
    if args.guarantee_classes is not None:
        try:
            {int(c) for c in args.guarantee_classes.split(",")}
        except ValueError:
            parser.error("--guarantee-classes must be comma-separated integers, e.g. '1,5,6'")
    if args.max_background_ratio is not None and args.max_background_ratio < 0:
        parser.error("--max-background-ratio must be >= 0")
    if args.manifest is not None and not args.manifest.exists():
        parser.error(f"manifest not found: {args.manifest}")

    cmd_tile(args)


if __name__ == "__main__":
    main()
