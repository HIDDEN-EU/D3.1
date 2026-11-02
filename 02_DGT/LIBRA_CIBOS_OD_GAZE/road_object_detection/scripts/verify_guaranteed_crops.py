"""
Verifier for the rare-class guaranteed crops of `tile-dataset --guarantee-classes`.

Checks AC-1 … AC-7 against the ACTUAL written tiled dataset (not a pure replay):
tile geometry is replayed from the source so we know which crop each source box
should land in, then we confirm the box is genuinely present in that crop's
`.txt` file. A generator that silently drops a box therefore FAILS the check —
the replay says "should be here", the real output disagrees.

  AC-1/2/3  every in-scope (source dims >= min_size) rider / motorcycle /
            bicycle source-train box reaches >= 1 tiled-train crop at >= min_vis.
            sub-min_size boxes are reported as skipped (expected), not failures.
  AC-4      500 sampled val label `.txt` files are byte-identical between the
            guarantee-on and guarantee-off datasets (train-only invariant).
  AC-5      no non-guarantee-set class loses tiled-train instances vs baseline
            (read from each dataset's stats.json).
  AC-6      guaranteed-tile count (train crops with index >= num_tiles) <= 2300.
  AC-7      every guaranteed crop actually contains a guarantee-class box.

Usage
-----
python scripts/verify_guaranteed_crops.py \
    --source            data/bdd100k \
    --recipe            configs/recipes/data_tiled_v030_mz16_no_ov.yaml \
    --tiled-gc          data/bdd100k_r_v030_mz16_no_ov_gc \
    --tiled-baseline    data/bdd100k_r_v030_mz16_no_ov \
    --guarantee-classes 1,5,6 \
    --min-visibility    0.30 \
    --sample-seed       42 \
    --output            output/verify_guaranteed_crops/summary.txt

Replay params (--num-tiles / --max-iou / --min-size / --seed) MUST match the
values used to generate --tiled-gc; defaults match the contract's protocol
(6 / 0.18 / 16 / 42).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

from scripts.tile_dataset import _compute_guarantee_tiles, _compute_random_tiles


# ---------------------------------------------------------------------------
# Recipe remap (mirrors tile_dataset.cmd_tile)
# ---------------------------------------------------------------------------

def load_recipe_remap(recipe_path: Path) -> tuple[dict[int, int | None], dict[int, str]]:
    """Return (remap, out_names) for a recipe with class_merges / class_drops."""
    cfg = yaml.safe_load(recipe_path.read_text())
    raw_merges = {int(k): int(v) for k, v in (cfg.get("class_merges") or {}).items()}
    drops = {int(c) for c in (cfg.get("class_drops") or [])}
    kept = [i for i in range(cfg["nc"]) if i not in raw_merges and i not in drops]
    renumber = {old: new for new, old in enumerate(kept)}
    remap = {
        i: (None if i in drops else renumber[raw_merges[i]] if i in raw_merges else renumber[i])
        for i in range(cfg["nc"])
    }
    out_names = {renumber[i]: cfg["names"][i] for i in kept}
    return remap, out_names


# ---------------------------------------------------------------------------
# Geometry — projection + IoU (mirror _project_labels exactly)
# ---------------------------------------------------------------------------

def project_box(
    box: tuple[float, float, float, float],
    tile: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    min_vis: float,
    min_size: int,
) -> tuple[float, float, float, float] | None:
    """Project a source-normalised box into a tile. Returns tile-normalised
    (xc, yc, w, h) if it survives the keep-test, else None — byte-identical
    logic to `tile_dataset._project_labels`."""
    xc, yc, bw, bh = box
    xc_px, yc_px, bw_px, bh_px = xc * img_w, yc * img_h, bw * img_w, bh * img_h
    x1, x2 = xc_px - bw_px / 2, xc_px + bw_px / 2
    y1, y2 = yc_px - bh_px / 2, yc_px + bh_px / 2
    tx, ty, tw, th = tile
    cx1, cx2 = max(x1, tx), min(x2, tx + tw)
    cy1, cy2 = max(y1, ty), min(y2, ty + th)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    orig_area = bw_px * bh_px
    if orig_area > 0 and (cx2 - cx1) * (cy2 - cy1) / orig_area < min_vis:
        return None
    if min_size > 0 and ((cx2 - cx1) < min_size or (cy2 - cy1) < min_size):
        return None
    return (
        ((cx1 + cx2) / 2 - tx) / tw,
        ((cy1 + cy2) / 2 - ty) / th,
        (cx2 - cx1) / tw,
        (cy2 - cy1) / th,
    )


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def parse_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    text = path.read_text().strip()
    if not text:
        return []
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        out.append((int(parts[0]), float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4])))
    return out


_IOU_MATCH = 0.90  # projected box vs written box — same math, so ~1.0 in practice


# ---------------------------------------------------------------------------
# AC-1/2/3/7 — coverage of in-scope guarantee boxes + guaranteed-crop content
# ---------------------------------------------------------------------------

def check_coverage(
    manifest: Path,
    remap: dict[int, int | None],
    out_names: dict[int, str],
    guarantee: set[int],
    tiled_gc: Path,
    num_tiles: int,
    max_iou: float,
    min_vis: float,
    min_size: int,
    seed: int,
) -> dict:
    gc_lbl_dir = tiled_gc / "labels" / "train"

    covered = defaultdict(int)      # post-cls -> in-scope boxes covered in >=1 real tile
    in_scope = defaultdict(int)     # post-cls -> in-scope box total
    sub_min = defaultdict(int)      # post-cls -> sub-min_size box total (expected skip)
    read_errors = 0
    guar_tiles_seen = 0             # guarantee tiles (idx>=num_tiles) present in output
    guar_tiles_empty = 0           # ... that contain NO guarantee-class box (AC-7 violation)

    # Iterate the manifest — the generator seeds its per-image RNG from the
    # exact `path` string here, so the replay geometry only matches the written
    # tiles if we hash the same string (NOT a reconstructed path).
    with manifest.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["source_split"] == "train"]
    total = len(rows)
    for n, row in enumerate(rows, 1):
        if n % 10000 == 0:
            print(f"    coverage scan {n:,}/{total:,} …")
        img_path_s = row["path"]
        lbl_s = row["label_path"]
        if not lbl_s:
            continue
        boxes = parse_label(Path(lbl_s))
        # post-remap class per box; keep only guarantee-set members
        g_boxes = []
        for cls, xc, yc, bw, bh in boxes:
            post = remap.get(cls, cls)
            if post in guarantee:
                g_boxes.append((post, (xc, yc, bw, bh)))
        if not g_boxes:
            continue

        stem = Path(img_path_s).stem
        img = cv2.imread(img_path_s)
        if img is None:
            read_errors += 1
            continue
        img_h, img_w = img.shape[:2]

        # Replay the EXACT tiles the generator wrote (random + guarantee).
        h = int(hashlib.md5(img_path_s.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(h ^ seed)
        tiles = _compute_random_tiles(img_w, img_h, num_tiles, max_iou, rng)
        parsed = [(cls, xc, yc, bw, bh) for cls, xc, yc, bw, bh in boxes]
        tiles = tiles + _compute_guarantee_tiles(
            parsed, tiles, img_w, img_h, guarantee, min_vis, min_size, remap, stem, quiet=True,
        )

        # Read the real tiled labels for this source image, once.
        actual: dict[int, list] = {}
        for idx in range(len(tiles)):
            actual[idx] = parse_label(gc_lbl_dir / f"{stem}_r{idx:02d}.txt")

        # AC-7: every guarantee tile (idx>=num_tiles) must hold a guarantee-class box.
        for idx in range(num_tiles, len(tiles)):
            guar_tiles_seen += 1
            if not any(ln[0] in guarantee for ln in actual.get(idx, [])):
                guar_tiles_empty += 1

        # AC-1/2/3: classify each guarantee box and confirm it is present in >=1 real tile.
        for post, (xc, yc, bw, bh) in g_boxes:
            bw_px, bh_px = bw * img_w, bh * img_h
            if min_size > 0 and (bw_px < min_size or bh_px < min_size):
                sub_min[post] += 1
                continue
            in_scope[post] += 1
            found = False
            for idx, tile in enumerate(tiles):
                proj = project_box((xc, yc, bw, bh), tile, img_w, img_h, min_vis, min_size)
                if proj is None:
                    continue
                for ln in actual.get(idx, []):
                    if ln[0] == post and box_iou(proj, (ln[1], ln[2], ln[3], ln[4])) >= _IOU_MATCH:
                        found = True
                        break
                if found:
                    break
            if found:
                covered[post] += 1

    return {
        "covered": dict(covered),
        "in_scope": dict(in_scope),
        "sub_min": dict(sub_min),
        "read_errors": read_errors,
        "guar_tiles_seen": guar_tiles_seen,
        "guar_tiles_empty": guar_tiles_empty,
    }


# ---------------------------------------------------------------------------
# AC-4 — val byte-identity (sampled)
# ---------------------------------------------------------------------------

def check_val_identity(tiled_gc: Path, tiled_baseline: Path, sample_seed: int, n: int = 500) -> dict:
    gc_val = tiled_gc / "labels" / "val"
    base_val = tiled_baseline / "labels" / "val"
    gc_files = sorted(p.name for p in gc_val.glob("*.txt"))
    rng = random.Random(sample_seed)
    sample = rng.sample(gc_files, min(n, len(gc_files)))
    mismatches, missing = 0, 0
    for name in sample:
        gcp, bp = gc_val / name, base_val / name
        if not bp.exists():
            missing += 1
            continue
        if hashlib.md5(gcp.read_bytes()).digest() != hashlib.md5(bp.read_bytes()).digest():
            mismatches += 1
    return {"sampled": len(sample), "mismatches": mismatches, "missing": missing}


# ---------------------------------------------------------------------------
# AC-5 — non-guarantee-set counts non-decreasing  (from stats.json)
# ---------------------------------------------------------------------------

def check_counts(tiled_gc: Path, tiled_baseline: Path, guarantee_names: set[str]) -> dict:
    import json
    gc = json.loads((tiled_gc / "stats.json").read_text())["splits"]["train"]["classes"]
    base = json.loads((tiled_baseline / "stats.json").read_text())["splits"]["train"]["classes"]
    rows, drops = [], []
    for name, v in base.items():
        if name in guarantee_names:
            continue
        b = v["count"]
        g = gc.get(name, {}).get("count", 0)
        rows.append((name, b, g, g - b))
        if g < b:
            drops.append((name, b, g))
    return {"rows": rows, "drops": drops}


# ---------------------------------------------------------------------------
# AC-6 — guaranteed-tile count
# ---------------------------------------------------------------------------

def count_guarantee_tiles(tiled_gc: Path, num_tiles: int) -> int:
    n = 0
    for p in (tiled_gc / "labels" / "train").glob("*_r*.txt"):
        try:
            idx = int(p.stem.rsplit("_r", 1)[1])
        except (ValueError, IndexError):
            continue
        if idx >= num_tiles:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(prog="verify-guaranteed-crops")
    ap.add_argument("--source", type=Path, required=True,
                    help="Source dataset root (used to locate manifest.csv if --manifest omitted)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Source manifest.csv (default: <source>/manifest.csv). Its `path` "
                         "column MUST match what tile-dataset was run with, or the RNG replay "
                         "will not line up with the written tiles.")
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--tiled-gc", type=Path, required=True)
    ap.add_argument("--tiled-baseline", type=Path, required=True)
    ap.add_argument("--guarantee-classes", type=str, required=True, help="e.g. 1,5,6 (post-remap ids)")
    ap.add_argument("--min-visibility", type=float, default=0.30)
    ap.add_argument("--min-size", type=int, default=16)
    ap.add_argument("--num-tiles", type=int, default=6)
    ap.add_argument("--max-iou", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--max-guarantee-tiles", type=int, default=2300, help="AC-6 upper bound")
    ap.add_argument("--output", type=Path, default=Path("output/verify_guaranteed_crops/summary.txt"))
    args = ap.parse_args()

    manifest = args.manifest or (args.source / "manifest.csv")
    if not manifest.exists():
        raise SystemExit(f"ERROR: manifest not found: {manifest}")
    guarantee = {int(c) for c in args.guarantee_classes.split(",")}
    remap, out_names = load_recipe_remap(args.recipe)
    guarantee_names = {out_names[c] for c in guarantee}
    print(f"guarantee classes: {sorted(guarantee)} → {sorted(guarantee_names)}")
    print(f"replay params: num_tiles={args.num_tiles} max_iou={args.max_iou} "
          f"min_vis={args.min_visibility} min_size={args.min_size} seed={args.seed}\n")

    print("── AC-1/2/3/7: coverage + guaranteed-crop content ──")
    cov = check_coverage(
        manifest, remap, out_names, guarantee, args.tiled_gc,
        args.num_tiles, args.max_iou, args.min_visibility, args.min_size, args.seed,
    )
    print("── AC-4: val byte-identity ──")
    val = check_val_identity(args.tiled_gc, args.tiled_baseline, args.sample_seed)
    print("── AC-5: non-guarantee counts ──")
    counts = check_counts(args.tiled_gc, args.tiled_baseline, guarantee_names)
    print("── AC-6: guaranteed-tile count ──")
    n_guar = count_guarantee_tiles(args.tiled_gc, args.num_tiles)

    # ---- assemble report ----
    lines: list[str] = []
    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("verify-guaranteed-crops — Phase 1 (rare_class_guaranteed_crops.md)")
    emit("=" * 70)
    emit(f"source          : {args.source}")
    emit(f"tiled (gc)      : {args.tiled_gc}")
    emit(f"tiled (baseline): {args.tiled_baseline}")
    emit(f"guarantee       : {sorted(guarantee)} {sorted(guarantee_names)}")
    emit(f"replay          : num_tiles={args.num_tiles} max_iou={args.max_iou} "
         f"min_vis={args.min_visibility} min_size={args.min_size} seed={args.seed}")
    if cov["read_errors"]:
        emit(f"[WARN] {cov['read_errors']} source images failed to read (skipped)")
    emit("")

    all_pass = True

    # AC-1/2/3
    ac_ids = {1: "AC-1", 5: "AC-2", 6: "AC-3"}
    for cid in sorted(guarantee):
        name = out_names[cid]
        cov_n = cov["covered"].get(cid, 0)
        scope_n = cov["in_scope"].get(cid, 0)
        sub_n = cov["sub_min"].get(cid, 0)
        ok = (cov_n == scope_n) and scope_n > 0
        all_pass &= ok
        tag = ac_ids.get(cid, "AC-?")
        emit(f"[{'PASS' if ok else 'FAIL'}] {tag} {name:12} coverage {cov_n:,}/{scope_n:,} "
             f"in-scope (100% required); {sub_n:,} sub-{args.min_size}px skipped (expected)")

    # AC-4
    ok4 = val["mismatches"] == 0 and val["missing"] == 0
    all_pass &= ok4
    emit(f"[{'PASS' if ok4 else 'FAIL'}] AC-4 val byte-identity: {val['sampled']} sampled, "
         f"{val['mismatches']} mismatches, {val['missing']} missing (both must be 0)")

    # AC-5
    ok5 = not counts["drops"]
    all_pass &= ok5
    emit(f"[{'PASS' if ok5 else 'FAIL'}] AC-5 non-guarantee counts non-decreasing "
         f"({len(counts['drops'])} classes dropped):")
    for name, b, g, d in sorted(counts["rows"]):
        flag = "  <-- DROP" if d < 0 else ""
        emit(f"        {name:14} baseline={b:>8,}  gc={g:>8,}  delta={d:>+7,}{flag}")

    # AC-6
    ok6 = n_guar <= args.max_guarantee_tiles
    all_pass &= ok6
    emit(f"[{'PASS' if ok6 else 'FAIL'}] AC-6 guaranteed-tile count {n_guar:,} "
         f"(<= {args.max_guarantee_tiles:,} required)")

    # AC-7
    ok7 = cov["guar_tiles_empty"] == 0
    all_pass &= ok7
    emit(f"[{'PASS' if ok7 else 'FAIL'}] AC-7 guaranteed crops contain a target box: "
         f"{cov['guar_tiles_seen']:,} guarantee tiles, {cov['guar_tiles_empty']:,} empty "
         f"(0 required)")

    emit("")
    emit("=" * 70)
    emit(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    emit("=" * 70)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nsummary written → {args.output}")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
