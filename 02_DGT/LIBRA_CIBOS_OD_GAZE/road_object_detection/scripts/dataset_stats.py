"""
Automated dataset quality report for YOLO-format datasets.

Scans any dataset produced by tile-dataset (or the original BDD100K source)
and reports:

  • Image inventory  — total / labelled / background per split
  • Object counts    — total, per-image mean / median / p95 / max
  • Class distribution — count, %, imbalance ratio, rare-class warnings
  • Box geometry     — pixel width/height (p5/p50/p95/max), tiny-box count,
                       box area as % of image area
  • Comparison mode  — side-by-side class distribution between two datasets
                       (flag classes with > 20 % relative difference)

The report is printed to stdout and written as a JSON artifact to
  <dataset_dir>/stats.json   (override with --output)

JSON structure is designed for future mlflow.log_artifact() / log_metrics().

Usage
-----
dataset-stats --data-yaml data/bdd100k_r015/data.yaml

dataset-stats \\
    --data-yaml data/bdd100k_r015/data.yaml \\
    --compare   configs/data.yaml \\
    --workers   4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Worker (module-level — required for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _scan_label_file(path_s: str) -> tuple[int, list[int], list[float], list[float]]:
    """Parse one YOLO label file. Returns (n_objects, class_ids, norm_ws, norm_hs)."""
    text = Path(path_s).read_text().strip()
    if not text:
        return 0, [], [], []
    classes: list[int] = []
    ws: list[float] = []
    hs: list[float] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            classes.append(int(parts[0]))
            ws.append(float(parts[3]))
            hs.append(float(parts[4]))
        except ValueError:
            continue
    return len(classes), classes, ws, hs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _dist_stats(sorted_vals: list[float]) -> dict:
    if not sorted_vals:
        return {"mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(sum(sorted_vals) / len(sorted_vals), 1),
        "p5":   round(_percentile(sorted_vals, 5),  1),
        "p50":  round(_percentile(sorted_vals, 50), 1),
        "p95":  round(_percentile(sorted_vals, 95), 1),
        "max":  round(sorted_vals[-1], 1),
    }


def _detect_img_size(images_dir: Path) -> tuple[int, int] | None:
    from PIL import Image
    jpgs = sorted(images_dir.glob("*.jpg"))
    if not jpgs:
        return None
    with Image.open(jpgs[0]) as im:
        return im.width, im.height


def _labels_dir(images_dir: Path) -> Path:
    return Path(str(images_dir).replace("/images/", "/labels/", 1))


# ---------------------------------------------------------------------------
# Occlusion aggregation
# ---------------------------------------------------------------------------

def _scan_occlusion(labels_dir: Path, names: dict[str, str]) -> dict[str, dict] | None:
    """
    Aggregate `<labels_dir>/occlusion.csv` into per-class % occluded / %
    truncated. Returns `None` (not an empty dict) when the sidecar is absent
    — a missing sidecar is normal
    and callers must not treat it as "zero occlusion".

    `occlusion.csv` columns: `stem, box_index, occluded, truncated`, where
    `box_index` is the line index into `<labels_dir>/<stem>.txt`. Each row is
    joined to its class by reading that line of the label file.
    """
    csv_path = labels_dir / "occlusion.csv"
    if not csv_path.exists():
        return None

    # class id (as it appears at the front of a label line) for each
    # (stem, box_index) — read lazily, cached per stem to avoid re-reading
    # a label file once per box.
    cls_cache: dict[str, list[int]] = {}

    def _classes_for(stem: str) -> list[int]:
        if stem not in cls_cache:
            txt_path = labels_dir / f"{stem}.txt"
            classes: list[int] = []
            if txt_path.exists():
                for line in txt_path.read_text().splitlines():
                    parts = line.split()
                    classes.append(int(parts[0]) if parts else -1)
            cls_cache[stem] = classes
        return cls_cache[stem]

    totals: dict[int, int] = {}
    occ_counts: dict[int, int] = {}
    trc_counts: dict[int, int] = {}

    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            stem = row["stem"]
            box_index = int(row["box_index"])
            classes = _classes_for(stem)
            if box_index >= len(classes):
                continue
            cid = classes[box_index]
            totals[cid] = totals.get(cid, 0) + 1
            if row["occluded"].strip().lower() == "true":
                occ_counts[cid] = occ_counts.get(cid, 0) + 1
            if row["truncated"].strip().lower() == "true":
                trc_counts[cid] = trc_counts.get(cid, 0) + 1

    occlusion_stats: dict[str, dict] = {}
    for cid_s, name in names.items():
        cid = int(cid_s)
        n = totals.get(cid, 0)
        pct_occ = round(100.0 * occ_counts.get(cid, 0) / n, 2) if n > 0 else 0.0
        pct_trc = round(100.0 * trc_counts.get(cid, 0) / n, 2) if n > 0 else 0.0
        occlusion_stats[name] = {
            "n": n,
            "pct_occluded": pct_occ,
            "pct_truncated": pct_trc,
        }
    return occlusion_stats


# ---------------------------------------------------------------------------
# Core scanner
# ---------------------------------------------------------------------------

def _scan_split(
    images_dir: Path,
    names: dict[str, str],
    tiny_thresh: int,
    workers: int,
) -> dict:
    labels_dir = _labels_dir(images_dir)

    img_stems = {p.stem for p in images_dir.glob("*.jpg")}
    total_images = len(img_stems)
    if total_images == 0:
        print(f"  [WARN] no .jpg images found in {images_dir}")
        return {}

    img_size = _detect_img_size(images_dir)
    img_w, img_h = img_size if img_size else (None, None)

    if not labels_dir.exists():
        print(f"  [WARN] labels dir not found: {labels_dir}")
        label_files: list[Path] = []
    else:
        label_files = sorted(labels_dir.glob("*.txt"))

    label_stems = {p.stem for p in label_files}
    no_label_file = len(img_stems - label_stems)

    all_classes: list[int] = []
    all_ws: list[float] = []
    all_hs: list[float] = []
    objs_per_img: list[int] = []
    empty_labels = 0
    done = 0
    n = len(label_files)

    def _collect(n_obj: int, classes: list, ws: list, hs: list) -> None:
        nonlocal empty_labels
        objs_per_img.append(n_obj)
        if n_obj == 0:
            empty_labels += 1
        else:
            all_classes.extend(classes)
            all_ws.extend(ws)
            all_hs.extend(hs)

    if workers > 1 and n > 0:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_scan_label_file, str(p)) for p in label_files]
            for fut in as_completed(futures):
                _collect(*fut.result())
                done += 1
                if done % 10_000 == 0 or done == n:
                    print(f"    {done:,}/{n:,} label files …")
    else:
        for p in label_files:
            _collect(*_scan_label_file(str(p)))
            done += 1
            if done % 10_000 == 0 or done == n:
                print(f"    {done:,}/{n:,} label files …")

    objs_per_img.extend([0] * no_label_file)
    background = empty_labels + no_label_file

    class_counts = Counter(all_classes)
    total_objects = len(all_classes)
    classes_stats: dict[str, dict] = {}
    for cid_s, name in names.items():
        count = class_counts.get(int(cid_s), 0)
        pct = 100.0 * count / total_objects if total_objects > 0 else 0.0
        classes_stats[name] = {"count": count, "pct": round(pct, 2)}

    objs_per_img_sorted = sorted(objs_per_img)

    geometry: dict | None = None
    if img_w and img_h and all_ws:
        ws_px     = sorted(w * img_w for w in all_ws)
        hs_px     = sorted(h * img_h for h in all_hs)
        areas_pct = sorted(w * h * 100.0 for w, h in zip(all_ws, all_hs))
        tiny = sum(1 for w, h in zip(ws_px, hs_px) if w < tiny_thresh or h < tiny_thresh)
        geometry = {
            "img_size":  [img_w, img_h],
            "width_px":  _dist_stats(ws_px),
            "height_px": _dist_stats(hs_px),
            "tiny_boxes": {
                "count":        tiny,
                "pct":          round(100.0 * tiny / len(ws_px), 2),
                "threshold_px": tiny_thresh,
            },
            "area_pct": _dist_stats(areas_pct),
        }
    elif all_ws:
        ws_n  = sorted(all_ws)
        hs_n  = sorted(all_hs)
        ap_n  = sorted(w * h * 100.0 for w, h in zip(all_ws, all_hs))
        geometry = {
            "img_size":    None,
            "note":        "mixed image sizes — normalised stats only",
            "width_norm":  _dist_stats(ws_n),
            "height_norm": _dist_stats(hs_n),
            "area_pct":    _dist_stats(ap_n),
        }

    nonzero = [v["count"] for v in classes_stats.values() if v["count"] > 0]
    imbalance = round(max(nonzero) / min(nonzero), 1) if len(nonzero) >= 2 else None
    rare = [name for name, v in classes_stats.items() if 0 < v["count"] < 500]

    occlusion = _scan_occlusion(labels_dir, names)

    return {
        "images": {
            "total":          total_images,
            "labelled":       total_images - background,
            "background":     background,
            "background_pct": round(100.0 * background / total_images, 2),
        },
        "objects": {
            "total":     total_objects,
            "per_image": _dist_stats(objs_per_img_sorted),
        },
        "classes":         classes_stats,
        "geometry":        geometry,
        "imbalance_ratio": imbalance,
        "rare_classes":    rare,
        "occlusion":       occlusion,
        "coverage":        None,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

_SEP = "─" * 64


def _section(title: str) -> None:
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _print_split(split: str, s: dict) -> None:
    if not s:
        return

    _section(f"Split: {split}")

    im = s["images"]
    print(f"  images    total={im['total']:,}   "
          f"labelled={im['labelled']:,}   "
          f"background={im['background']:,} ({im['background_pct']:.1f} %)")

    ob = s["objects"]
    pi = ob["per_image"]
    print(f"  objects   total={ob['total']:,}   "
          f"per-image  mean={pi['mean']}  median={pi['p50']}  "
          f"p95={pi['p95']}  max={pi['max']}")

    print(f"\n  {'Class':<22} {'Count':>10}   {'%':>6}")
    print(f"  {'─' * 22} {'─' * 10}   {'─' * 6}")
    for name, v in sorted(s["classes"].items(), key=lambda x: -x[1]["count"]):
        print(f"  {name:<22} {v['count']:>10,}   {v['pct']:>5.1f} %")

    if s["imbalance_ratio"]:
        print(f"\n  imbalance ratio: {s['imbalance_ratio']}")
    for name in s["rare_classes"]:
        print(f"  [WARN] rare class (< 500 samples): {name}")

    g = s.get("geometry")
    if not g:
        return

    if g.get("note"):
        print(f"\n  Note: {g['note']}")

    size_str = f"{g['img_size'][0]}x{g['img_size'][1]}" if g.get("img_size") else "mixed"
    print(f"\n  Box geometry  (img {size_str},  tiny threshold {g.get('tiny_boxes', {}).get('threshold_px', '?')} px)")

    if g.get("width_px"):
        print(f"  {'':22} {'mean':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'max':>7}")
        wp = g["width_px"]
        hp = g["height_px"]
        print(f"  {'width  (px)':<22} {wp['mean']:>7} {wp['p5']:>7} {wp['p50']:>7} {wp['p95']:>7} {wp['max']:>7}")
        print(f"  {'height (px)':<22} {hp['mean']:>7} {hp['p5']:>7} {hp['p50']:>7} {hp['p95']:>7} {hp['max']:>7}")
    elif g.get("width_norm"):
        print(f"  {'':22} {'mean':>7} {'p5':>7} {'p50':>7} {'p95':>7} {'max':>7}")
        wn = g["width_norm"]
        hn = g["height_norm"]
        print(f"  {'width  (norm)':<22} {wn['mean']:>7} {wn['p5']:>7} {wn['p50']:>7} {wn['p95']:>7} {wn['max']:>7}")
        print(f"  {'height (norm)':<22} {hn['mean']:>7} {hn['p5']:>7} {hn['p50']:>7} {hn['p95']:>7} {hn['max']:>7}")

    ap = g.get("area_pct", {})
    if ap:
        print(f"  {'area   (% img)':<22} {ap['mean']:>7} {ap['p5']:>7} {ap['p50']:>7} {ap['p95']:>7}")

    if g.get("tiny_boxes"):
        tb = g["tiny_boxes"]
        tag = "[WARN]" if tb["pct"] > 5.0 else "[OK]  "
        print(f"\n  {tag} tiny boxes: {tb['count']:,}  ({tb['pct']:.1f} %)")

    _print_occlusion(s.get("occlusion"))
    _print_coverage(s.get("coverage"))


def _print_occlusion(occlusion: dict[str, dict] | None) -> None:
    print(f"\n  Occlusion  (from occlusion.csv sidecar)")
    if occlusion is None:
        print(f"  [n/a] occlusion.csv not found for this split")
        return
    rows = [(name, v) for name, v in occlusion.items() if v["n"] > 0]
    if not rows:
        print(f"  [n/a] no occluded/truncated data for any class")
        return
    print(f"  {'Class':<22} {'N':>10}   {'Occluded':>9}   {'Truncated':>10}")
    print(f"  {'─' * 22} {'─' * 10}   {'─' * 9}   {'─' * 10}")
    for name, v in sorted(rows, key=lambda x: -x[1]["pct_occluded"]):
        print(f"  {name:<22} {v['n']:>10,}   {v['pct_occluded']:>7.1f} %   {v['pct_truncated']:>8.1f} %")


def _print_coverage(coverage: dict[str, dict] | None) -> None:
    print(f"\n  Coverage  (tiling survival — % of source boxes reaching ≥ 1 tile)")
    if coverage is None:
        print(f"  [n/a] not computed — coverage requires tile-dataset (mode r) generation")
        return
    rows = [(name, v) for name, v in coverage.items() if v["n_source"] > 0]
    if not rows:
        print(f"  [n/a] no coverage data for any class")
        return
    print(f"  {'Class':<22} {'Source':>10}   {'Survived':>10}   {'Cov %':>7}")
    print(f"  {'─' * 22} {'─' * 10}   {'─' * 10}   {'─' * 7}")
    for name, v in sorted(rows, key=lambda x: x[1]["pct_survived"]):
        tag = "[WARN] " if v["pct_survived"] < 50.0 else ""
        print(f"  {name:<22} {v['n_source']:>10,}   {v['n_survived']:>10,}   {v['pct_survived']:>5.1f} % {tag}")


def _print_comparison(
    name_a: str,
    name_b: str,
    report_a: dict,
    report_b: dict,
    split: str,
) -> None:
    sa = report_a.get("splits", {}).get(split)
    sb = report_b.get("splits", {}).get(split)
    if not sa or not sb:
        return

    _section(
        f"Comparison [{split}]:  {Path(name_a).name}  vs  {Path(name_b).name}"
    )

    total_a = sa["objects"]["total"] or 1
    total_b = sb["objects"]["total"] or 1
    all_names = sorted(set(sa["classes"]) | set(sb["classes"]))

    print(f"  {'Class':<22} {'A count':>10} {'A %':>6}   {'B count':>10} {'B %':>6}   {'delta':>7}")
    print(f"  {'─' * 22} {'─' * 10} {'─' * 6}   {'─' * 10} {'─' * 6}   {'─' * 7}")
    for name in all_names:
        ca = sa["classes"].get(name, {}).get("count", 0)
        cb = sb["classes"].get(name, {}).get("count", 0)
        pa = 100.0 * ca / total_a
        pb = 100.0 * cb / total_b
        delta = pa - pb
        flag = "  !" if abs(delta) > 20 else ""
        print(
            f"  {name:<22} {ca:>10,} {pa:>5.1f} %   "
            f"{cb:>10,} {pb:>5.1f} %   {delta:>+6.1f} %{flag}"
        )


def _print_report(report: dict, compare_report: dict | None) -> None:
    print(f"\ndataset-stats")
    print(f"  dataset  : {report['dataset']}")
    print(f"  classes  : {report['nc']}  ({', '.join(report['names'].values())})")
    print(f"  generated: {report['generated_utc']}")

    for split, s in report["splits"].items():
        _print_split(split, s)

    if compare_report:
        for split in report["splits"]:
            _print_comparison(
                report["dataset"],
                compare_report["dataset"],
                report,
                compare_report,
                split,
            )


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def _write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"\nJSON written → {path}")


# ---------------------------------------------------------------------------
# Report builder (importable — used by `dataset-stats` CLI and `tile-dataset`)
# ---------------------------------------------------------------------------

def build_report(
    cfg: dict,
    root: Path,
    splits: list[str],
    workers: int,
    tiny_thresh: int = 16,
) -> dict:
    """
    Build the full stats report dict for a dataset (source or tiled).

    `cfg` is a parsed `data.yaml` dict (`path`, `train`/`val`, `nc`, `names`).
    `root` is the dataset root (normally `Path(cfg["path"])` — passed
    separately so callers, e.g. `tile-dataset`, can reuse an already-resolved
    path). Each split's dict gets `occlusion` (aggregated from
    `labels/<split>/occlusion.csv`, or `None` if that sidecar doesn't exist)
    and `coverage` (always `None` here — this script only ever sees output
    labels, never the tiling geometry needed to compute coverage; `tile-dataset`
    injects real coverage numbers after calling this function).
    """
    names = {str(k): v for k, v in cfg["names"].items()}

    report: dict = {
        "dataset":       str(root),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nc":            cfg["nc"],
        "names":         names,
        "splits":        {},
        "comparison":    None,
    }

    for split in splits:
        images_dir = root / cfg[split]
        print(f"\n── {split} ──  ({images_dir})")
        if not images_dir.exists():
            print(f"  [WARN] images dir not found: {images_dir}")
            continue
        report["splits"][split] = _scan_split(
            images_dir, names, tiny_thresh, workers,
        )

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataset-stats",
        description=(
            "Quality report for a YOLO-format dataset. "
            "Prints to stdout and writes a JSON artifact."
        ),
    )
    parser.add_argument(
        "--data-yaml", type=Path, required=True,
        help="data.yaml of the dataset to analyse",
    )
    parser.add_argument(
        "--compare", type=Path, default=None,
        help="Optional second data.yaml for side-by-side comparison",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"], choices=["train", "val"],
        help="Splits to analyse (default: train val)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="JSON output path (default: <dataset_dir>/stats.json)",
    )
    parser.add_argument(
        "--tiny-threshold", type=int, default=16,
        help="Flag boxes narrower or shorter than this many pixels (default: 16)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for label scanning (default: 1)",
    )

    args = parser.parse_args()

    if not args.data_yaml.exists():
        print(f"ERROR: {args.data_yaml} not found"); sys.exit(1)
    if args.compare and not args.compare.exists():
        print(f"ERROR: --compare {args.compare} not found"); sys.exit(1)

    cfg  = yaml.safe_load(args.data_yaml.read_text())
    root = Path(cfg["path"])

    out_path = args.output or (root / "stats.json")

    report = build_report(cfg, root, args.splits, args.workers, args.tiny_threshold)

    compare_report: dict | None = None
    if args.compare:
        cmp_cfg   = yaml.safe_load(args.compare.read_text())
        cmp_root  = Path(cmp_cfg["path"])
        cmp_names = {str(k): v for k, v in cmp_cfg["names"].items()}
        compare_report = {
            "dataset":       str(cmp_root),
            "generated_utc": report["generated_utc"],
            "nc":            cmp_cfg["nc"],
            "names":         cmp_names,
            "splits":        {},
            "comparison":    None,
        }
        for split in args.splits:
            cmp_images_dir = cmp_root / cmp_cfg[split]
            print(f"\n── {split} [compare] ──  ({cmp_images_dir})")
            if not cmp_images_dir.exists():
                print(f"  [WARN] images dir not found: {cmp_images_dir}")
                continue
            compare_report["splits"][split] = _scan_split(
                cmp_images_dir, cmp_names, args.tiny_threshold, args.workers,
            )
        report["comparison"] = compare_report

    _print_report(report, compare_report)
    _write_json(report, out_path)


if __name__ == "__main__":
    main()
