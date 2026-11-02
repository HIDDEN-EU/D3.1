"""
Occlusion-stratified detection evaluation.

It does the per-stratum class grouping and occlusion filtering, then defers all
per-pair / per-class arithmetic to the primitives in `metrics.py`.

Strata (overlap by design — a box can be both occluded and truncated):
  overall    — every GT box (no stratification; this is the AC-1 self-consistency target)
  visible    — occluded=False AND truncated=False
  occluded   — occluded=True
  truncated  — truncated=True

Stratification rule (contract "Stratification rule", rule (b)): stratify by ground-truth
flag only. For a stratum's class-C AP, `true_boxes` = only in-stratum GT of class C, but
`pred_boxes` = ALL class-C predictions on those images. A prediction matching only an
out-of-stratum GT box is therefore scored as a false positive, not ignored — occluded-
stratum precision reflects real false alarms on scenes containing occluded objects.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from road_object_detection.evaluation.metrics import (
    compute_average_precision,
    compute_iou,
    compute_map,
    match_detections,
)

STRATA = ["overall", "visible", "occluded", "truncated"]
# Class IDs for the VRU rollup: pedestrian, rider. Persons only.
#
# BDD100K annotates the *person* on a two-wheeler as `rider` and the vehicle under
# them as a separate `bicycle` / `motorcycle` box. Including `bicycle` therefore
# counted a cyclist twice (person + vehicle) while a motorcyclist counted once, and
# it swept in parked bicycles with no road user on them. Restricting the rollup to
# the two person classes makes every box in it a human, which is what the term
# vulnerable road *user* means.
#
# IDs 0 and 1 are stable across the 13/11/10/9-class spaces (pedestrian and rider
# lead the BDD100K order and no merge touches them).
VRU_CLASS_IDS = [0, 1]
# COCO-style IoU sweep for mAP@0.5:0.95 (0.50, 0.55, …, 0.95).
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
# Fixed recall grid (0.00 … 1.00) the per-class PR curves are resampled onto, so the
# same class overlays cleanly across runs in MLflow's Model Metrics panel.
PR_RECALL_GRID = [round(0.01 * i, 2) for i in range(101)]
COVERAGE_FLOOR = 0.005  # AC-2: reject if join miss rate exceeds this.
LOW_CONFIDENCE_FLOOR = 50  # AC-6(c): cells with fewer GT boxes are flagged, not failed.


def load_occlusion_flags(csv_path: Path) -> dict:
    """Load occlusion.csv into a {(stem, box_index): (occluded, truncated)} map.

    `.strip()` defends against the CRLF the propagation step's csv.writer emits
    (contract open question: ``"True\\r"`` must compare equal to ``"True"``).
    """
    flags: dict = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["stem"].strip(), int(row["box_index"]))
            flags[key] = (
                row["occluded"].strip() == "True",
                row["truncated"].strip() == "True",
            )
    return flags


def derive_test_slice(data: dict, seed: int, ratio: float) -> list:
    """Reproduce `_prepare_split`'s held-out *test* slice exactly: the first `ratio`
    fraction of the seed-shuffled, sorted val image list (which `_prepare_split`
    reserves and discards, keeping the remainder as the training-time val set)."""
    root = Path(data["path"])
    images = sorted((root / data["val"]).glob("*.jpg"))
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    n_test = int(len(shuffled) * ratio)
    return shuffled[:n_test]


def derive_val_slice(data: dict, seed: int, ratio: float) -> list:
    """Reproduce `_prepare_split`'s training-time *val* slice: the remaining `1-ratio`
    fraction of the seed-shuffled, sorted val image list (the complement of
    `derive_test_slice`). This is the data the model validated on during training — the
    clean slice for routine post-training occlusion-recall monitoring, leaving the
    held-out test slice pristine for final sign-off."""
    root = Path(data["path"])
    images = sorted((root / data["val"]).glob("*.jpg"))
    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    n_test = int(len(shuffled) * ratio)
    return shuffled[n_test:]


def _read_label(path: Path, w: int, h: int) -> list:
    """Read a YOLO label file into [(cls, [x1,y1,x2,y2])] in absolute pixels.

    List order is the box_index that joins to occlusion.csv."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:])
        boxes.append(
            (cls, [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        )
    return boxes


def _membership(flag) -> list:
    """Which strata a GT box belongs to, given its (occluded, truncated) flag (or None)."""
    out = ["overall"]
    if flag is None:
        return out  # unjoined box — counts in overall only, and toward the miss rate
    occluded, truncated = flag
    if not occluded and not truncated:
        out.append("visible")
    if occluded:
        out.append("occluded")
    if truncated:
        out.append("truncated")
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _iter_predictions(model, image_paths, *, conf, nms_iou, imgsz, device, batch_size):
    """Yield (image_path, Result) in order, predicting in fixed-size chunks.

    Passing a whole list to predict() stacks every image into ONE forward pass
    (OOMs past a few hundred images), so we chunk. res.path is relabelled
    "image0.jpg"… per chunk, hence we zip with the chunk's real paths for the stem.
    """
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i : i + batch_size]
        results = model.predict(
            source=[str(p) for p in chunk],
            stream=False, conf=conf, iou=nms_iou, imgsz=imgsz, device=device, verbose=False,
        )
        for p, res in zip(chunk, results):
            yield p, res


def _pr_points(detections: list, num_gt: int) -> dict:
    """Interpolated precision on the fixed PR_RECALL_GRID for one class.

    Reuses the exact cumulative-TP/FP logic of `metrics.compute_average_precision` (which
    is contract-frozen and returns only the scalar AP, not the curve), then resamples the
    monotonic precision envelope onto a fixed recall grid so curves are overlayable across
    runs. `detections` is the `(confidence, is_tp)` list the mAP pass already built; no
    extra inference. Returns {"recall": grid, "precision": [...]} aligned to the grid.
    """
    grid = PR_RECALL_GRID
    if num_gt == 0 or not detections:
        return {"recall": grid, "precision": [0.0] * len(grid)}
    ordered = sorted(detections, key=lambda d: d[0], reverse=True)
    recalls = [0.0]
    precisions = [1.0]
    tp = fp = 0
    for _, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / num_gt)
        precisions.append(tp / (tp + fp))
    # Monotonic non-increasing envelope (same as compute_average_precision).
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # Resample: p(q) = precision at the first cumulative point with recall >= q. Both the
    # grid and `recalls` are ascending, so the cursor j only moves forward.
    out = []
    j = 0
    for q in grid:
        while j < len(recalls) and recalls[j] < q:
            j += 1
        out.append(round(precisions[j], 6) if j < len(recalls) else 0.0)
    return {"recall": grid, "precision": out}


def evaluate(
    model,
    image_paths: list,
    occ_flags: dict,
    label_dir: Path,
    class_names: dict,
    *,
    conf: float,
    nms_iou: float,
    imgsz: int,
    device,
    batch_size: int = 16,
) -> dict:
    """Run stratified evaluation and return a JSON-ready result dict.

    Raises RuntimeError if the occlusion-flag join miss rate exceeds COVERAGE_FLOOR (AC-2).
    """
    n_classes = len(class_names)

    # --- Phase A: inference + GT collection (one pass over images) ---
    samples = []  # (preds_by_cls, gt_by_stratum_by_cls)
    gt_counts = {s: defaultdict(int) for s in STRATA}
    total_gt = 0
    missed = 0

    preds_stream = _iter_predictions(
        model, image_paths,
        conf=conf, nms_iou=nms_iou, imgsz=imgsz, device=device, batch_size=batch_size,
    )
    for img_path, res in preds_stream:
        stem = Path(img_path).stem
        h, w = res.orig_shape

        preds = defaultdict(list)
        for b in res.boxes:
            preds[int(b.cls[0])].append((b.xyxy[0].tolist(), float(b.conf[0])))

        gt_by_stratum = {s: defaultdict(list) for s in STRATA}
        for idx, (cls, box) in enumerate(_read_label(label_dir / f"{stem}.txt", w, h)):
            total_gt += 1
            flag = occ_flags.get((stem, idx))
            if flag is None:
                missed += 1
            for s in _membership(flag):
                gt_by_stratum[s][cls].append(box)
                gt_counts[s][cls] += 1
        samples.append((preds, gt_by_stratum))

    miss_rate = missed / total_gt if total_gt else 0.0
    print(f"[occ-eval] {len(samples)} images, {total_gt} GT boxes, join miss rate {miss_rate:.5f}")
    for s in STRATA:
        all_n = sum(gt_counts[s].values())
        vru_n = sum(gt_counts[s][c] for c in VRU_CLASS_IDS)
        print(f"[occ-eval]   {s:9s}: {all_n:>7d} GT (VRU {vru_n})")
    if miss_rate > COVERAGE_FLOOR:
        raise RuntimeError(
            f"occlusion-flag join miss rate {miss_rate:.5f} exceeds floor {COVERAGE_FLOOR} "
            f"(AC-2) — refusing to write output. {missed}/{total_gt} GT boxes had no "
            f"(stem, box_index) row in occlusion.csv."
        )

    # --- Phase B: per (stratum, IoU, class) AP via metrics.py primitives ---
    # ap[stratum][iou][cls]; transient detection lists kept only one (stratum, iou) at a time.
    ap = {s: {iou: {} for iou in IOU_THRESHOLDS} for s in STRATA}
    overall_dets50: dict = {}  # det_by_cls for (overall, IoU 0.5) — feeds the PR curves
    for s in STRATA:
        for iou in IOU_THRESHOLDS:
            det_by_cls = defaultdict(list)
            for preds, gt_by_stratum in samples:
                gts = gt_by_stratum[s]
                for cls in preds.keys() | gts.keys():
                    p = preds.get(cls, [])
                    g = gts.get(cls, [])
                    det = match_detections([b for b, _ in p], [sc for _, sc in p], g, iou)
                    if det:
                        det_by_cls[cls].extend(det)
            if s == "overall" and iou == 0.5:
                overall_dets50 = det_by_cls
            for cls in range(n_classes):
                ap[s][iou][cls] = compute_average_precision(det_by_cls.get(cls, []), gt_counts[s][cls])

    # --- Assemble results ---
    def scope_map(stratum, iou, ids):
        # Average only over classes that HAVE GT in this (stratum, scope): a class with
        # zero instances has undefined AP, not 0. This matches Ultralytics' convention
        # (absent classes are nan-excluded) and keeps rare strata comparable to overall.
        # On the full slice every class is present, so this equals the doc's all-N mean.
        return compute_map([ap[stratum][iou][c] for c in ids if gt_counts[stratum][c] > 0])

    all_ids = list(range(n_classes))

    def _cell(stratum, ids):
        gt_n = sum(gt_counts[stratum][c] for c in ids)
        return {
            "mAP50": round(scope_map(stratum, 0.5, ids), 6),
            "mAP50_95": round(
                sum(scope_map(stratum, iou, ids) for iou in IOU_THRESHOLDS) / len(IOU_THRESHOLDS), 6
            ),
            "gt_count": gt_n,
            "low_confidence": gt_n < LOW_CONFIDENCE_FLOOR,  # AC-6(c): flag sparse cells
        }

    results = {
        s: {
            "all_classes": _cell(s, all_ids),
            "VRU": _cell(s, VRU_CLASS_IDS),
            "per_class_ap50": {class_names[c]: round(ap[s][0.5][c], 6) for c in range(n_classes)},
        }
        for s in STRATA
    }

    # Per-class PR curves: the overall stratum at IoU 0.5 is the standard per-class PR
    # (every box is in `overall`, so true_boxes = all class-C GT, pred_boxes = all class-C
    # preds). Built from the detections the loop already produced — no extra inference.
    pr_curves = {
        class_names[c]: {
            **_pr_points(overall_dets50.get(c, []), gt_counts["overall"][c]),
            "ap50": round(ap["overall"][0.5][c], 6),
            "ap5095": round(
                sum(ap["overall"][iou][c] for iou in IOU_THRESHOLDS) / len(IOU_THRESHOLDS), 6
            ),
        }
        for c in range(n_classes)
    }

    return {
        "_miss_rate": round(miss_rate, 6),
        "_eval_images_n": len(samples),
        "_stratum_gt_counts": {
            s: {"all_classes": sum(gt_counts[s].values()),
                "VRU": sum(gt_counts[s][c] for c in VRU_CLASS_IDS)}
            for s in STRATA
        },
        "_protocol_commit": _git_commit(),
        "results": results,
        "_pr_curves": pr_curves,
    }


# --- AC-4 schema verification ---
REQUIRED_META = ["weights", "dataset", "split", "eval_images_n", "seed",
                 "iou_thresholds", "protocol_commit"]


def verify_schema(doc: dict, n_classes: int = 11) -> list:
    """Return a list of schema problems (empty list = AC-4 satisfied)."""
    problems = []
    meta = doc.get("metadata", {})
    for k in REQUIRED_META:
        if k not in meta:
            problems.append(f"metadata.{k} missing")
    results = doc.get("results", {})
    for s in STRATA:
        if s not in results:
            problems.append(f"results.{s} missing")
            continue
        for scope in ("all_classes", "VRU"):
            for metric in ("mAP50", "mAP50_95"):
                if metric not in results[s].get(scope, {}):
                    problems.append(f"results.{s}.{scope}.{metric} missing")
        pc = results[s].get("per_class_ap50", {})
        if len(pc) != n_classes:
            problems.append(f"results.{s}.per_class_ap50 has {len(pc)} entries, expected {n_classes}")
    return problems


# =============================================================================
# Count-based recall view.
#
# A plain "of the GT objects in this stratum, how many did the model find vs
# miss" answer, sidestepping the rule-(b) mAP confound of `evaluate` above. It is
# GT-anchored: it counts matched vs unmatched GT, with NO false-positive
# accounting and NO PR-curve integration. The `evaluate` mAP path above is
# untouched; this is a separate, additive code path.
# =============================================================================

RECALL_CONF_THRESHOLDS = [0.25, 0.001]  # headline (inference default) + comparison
RECALL_IOU = 0.5                        # a GT box is "found" at IoU >= 0.5, same class
GALLERY_CONF = 0.25                     # galleries use the deployment-realistic op-point only
GALLERY_N = 24                          # max images sampled per gallery
GALLERY_SLUGS = ["occluded_missed", "occluded_found",
                 "occluded_vru_missed", "occluded_vru_found"]
# Static webview viewer copied verbatim into each gallery (AC-11e: byte-identical).
WEBVIEW_SRC = Path(__file__).resolve().parents[2] / "scripts" / "webview"

# AC-7: surfaced verbatim in stdout and metadata.caveats. The keyword tokens
# (TILE-LEVEL / SOURCE-FLAGS / SPARSE-CELL) are what the acceptance check greps for.
CAVEATS = [
    "TILE-LEVEL: recall is per tile crop, not per source frame. An object spanning "
    "multiple tiles is counted once per tile. This is not the full-frame KPI-6 recall.",
    "SOURCE-FLAGS: occluded/truncated flags are propagated from the source frame. "
    "'truncated' means source-frame-edge truncation, not tile-edge clipping. The "
    "'visible' stratum is contaminated by tile-edge-clipped boxes that retain their "
    "source 'visible' flag.",
    "SPARSE-CELL: the truncated x VRU cell has 35 GT boxes; treat its recall estimate "
    "with caution (low_confidence: true).",
]


def _conf_key(conf: float) -> str:
    """Stable string key for a confidence threshold (0.25 -> '0.25', 0.001 -> '0.001')."""
    return f"{conf:g}"


def _recall_cell(total_gt: int, detected: int, conf: float) -> dict:
    """Build one recall cell. Pure — the AC-6 unit test exercises it directly.

    `low_confidence` mirrors the sibling contract's AC-6(c): cells with fewer than
    LOW_CONFIDENCE_FLOOR GT boxes are flagged, not suppressed.
    """
    return {
        "total_gt": total_gt,
        "detected": detected,
        "missed": total_gt - detected,
        "recall": round(detected / total_gt, 6) if total_gt else 0.0,
        "conf": conf,
        "low_confidence": total_gt < LOW_CONFIDENCE_FLOOR,
    }


def _match_gt_status(pred_boxes, pred_scores, true_boxes, iou_threshold):
    """Per-GT match status, mirroring `metrics.match_detections`'s greedy assignment.

    `match_detections` returns one flag per *prediction*; for the galleries we need to
    know which *GT box* each true positive claimed (to draw it and label its confidence).
    This replays the identical greedy loop — predictions in descending-confidence order,
    each claiming the highest-IoU unclaimed GT at or above threshold — but keyed on GT.

    Returns a list parallel to `true_boxes`: the matching prediction's confidence for a
    found (TP) box, or None for a missed (FN) box. By construction the count of non-None
    entries equals `match_detections`'s true-positive count on the same inputs (asserted
    by the caller), so the galleries reconcile exactly with the JSON counts (AC-11b).
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    gt_conf = [None] * len(true_boxes)
    for i in order:
        best_iou = 0.0
        best_j = -1
        for j, true_box in enumerate(true_boxes):
            if gt_conf[j] is not None:
                continue
            iou = compute_iou(pred_boxes[i], true_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            gt_conf[best_j] = pred_scores[i]
    return gt_conf


def _build_gallery_manifest(items: list, seed: int, n: int, weights_label: str):
    """Build a webview-compatible manifest dict for one gallery category.

    `items` is the full list of qualifying GT boxes (one dict per box with keys
    img_path, w, h, box, cls, name, conf). Up to `n` images are sampled
    deterministically (seeded shuffle of the sorted qualifying-image list), and each
    sampled image becomes one card drawing all of its qualifying boxes.

    Returns (manifest_dict, sorted_image_src_paths) or None if there are no items
    (AC-11a: empty categories produce no gallery). `available` == len(items) ties the
    artifact to the JSON count (AC-11b); top-level `conf` == 0.0 so the viewer's slider
    shows every box by default (AC-11d).
    """
    if not items:
        return None
    by_img: dict = defaultdict(list)
    dims: dict = {}
    for it in items:
        by_img[it["img_path"]].append(it)
        dims[it["img_path"]] = (it["w"], it["h"])
    qualifying = sorted(by_img)
    shuffled = qualifying[:]
    random.Random(seed).shuffle(shuffled)
    chosen = sorted(shuffled[:n])
    images = []
    for ip in chosen:
        w, h = dims[ip]
        dets = [
            {"xyxy": it["box"], "conf": it["conf"], "cls": it["cls"], "name": it["name"]}
            for it in by_img[ip]
        ]
        images.append({"file": f"images/{Path(ip).name}", "width": w, "height": h,
                       "tiles": [], "detections": dets})
    manifest = {
        "weights": weights_label,
        "conf": 0.0,
        "merge_iou": 0.0,
        "available": len(items),
        "sampled": len(chosen),
        "images": images,
    }
    return manifest, [str(ip) for ip in chosen]


def evaluate_recall_counts(
    model,
    image_paths: list,
    occ_flags: dict,
    label_dir: Path,
    class_names: dict,
    *,
    conf_thresholds: list,
    nms_iou: float,
    imgsz: int,
    device,
    weights_label: str,
    batch_size: int = 16,
    gallery_seed: int = 42,
    gallery_n: int = GALLERY_N,
) -> dict:
    """Count-based occlusion recall, plus found/missed visual galleries.

    Inference runs ONCE at the lowest requested threshold; higher thresholds are
    obtained by post-hoc confidence filtering (NMS ordering is threshold-independent,
    so this equals predicting at each threshold). Counts come from `match_detections`
    (the contract's count path); the galleries reuse the same greedy logic via
    `_match_gt_status`, asserted consistent.

    Raises RuntimeError if the occlusion-flag join miss rate exceeds COVERAGE_FLOOR (AC-2).
    """
    min_conf = min(conf_thresholds)
    conf_keys = [_conf_key(c) for c in conf_thresholds]
    key_to_conf = {_conf_key(c): c for c in conf_thresholds}

    # --- Phase A: inference (once, at min_conf) + GT collection ---
    samples = []  # (img_path, w, h, preds_by_cls[(box,conf)], gt_records[{cls,box,strata}])
    gt_counts = {s: defaultdict(int) for s in STRATA}
    total_gt = 0
    missed = 0

    for img_path, res in _iter_predictions(
        model, image_paths,
        conf=min_conf, nms_iou=nms_iou, imgsz=imgsz, device=device, batch_size=batch_size,
    ):
        stem = Path(img_path).stem
        h, w = res.orig_shape
        preds = defaultdict(list)
        for b in res.boxes:
            preds[int(b.cls[0])].append((b.xyxy[0].tolist(), float(b.conf[0])))

        gt_records = []
        for idx, (cls, box) in enumerate(_read_label(label_dir / f"{stem}.txt", w, h)):
            total_gt += 1
            flag = occ_flags.get((stem, idx))
            if flag is None:
                missed += 1
            strata = _membership(flag)
            for s in strata:
                gt_counts[s][cls] += 1
            gt_records.append({"cls": cls, "box": box, "strata": strata})
        samples.append((str(img_path), w, h, preds, gt_records))

    miss_rate = missed / total_gt if total_gt else 0.0
    for c in CAVEATS:
        print(f"[occ-recall] CAVEAT: {c}")
    print(f"[occ-recall] {len(samples)} images, {total_gt} GT boxes, "
          f"join miss rate {miss_rate:.5f}")
    for s in STRATA:
        all_n = sum(gt_counts[s].values())
        vru_n = sum(gt_counts[s][c] for c in VRU_CLASS_IDS)
        flag = "  [low_confidence]" if vru_n < LOW_CONFIDENCE_FLOOR else ""
        print(f"[occ-recall]   {s:9s}: {all_n:>7d} GT (VRU {vru_n}){flag}")
    if miss_rate > COVERAGE_FLOOR:
        raise RuntimeError(
            f"occlusion-flag join miss rate {miss_rate:.5f} exceeds floor {COVERAGE_FLOOR} "
            f"(AC-2) — refusing to write output. {missed}/{total_gt} GT boxes had no "
            f"(stem, box_index) row in occlusion.csv."
        )

    # --- Phase B: detected counts per (conf, stratum, class) via match_detections ---
    detected = {ck: {s: defaultdict(int) for s in STRATA} for ck in conf_keys}
    for ck in conf_keys:
        conf = key_to_conf[ck]
        for _img, _w, _h, preds, gt_records in samples:
            gt_sc = {s: defaultdict(list) for s in STRATA}
            for r in gt_records:
                for s in r["strata"]:
                    gt_sc[s][r["cls"]].append(r["box"])
            for s in STRATA:
                for cls, tboxes in gt_sc[s].items():
                    p = [(b, c) for b, c in preds.get(cls, []) if c >= conf]
                    det = match_detections([b for b, _ in p], [c for _, c in p], tboxes, RECALL_IOU)
                    detected[ck][s][cls] += sum(1 for _, is_tp in det if is_tp)

    # --- Assemble result cells ---
    all_ids = list(range(len(class_names)))

    def cell(ck, stratum, ids):
        total = sum(gt_counts[stratum][c] for c in ids)
        det = sum(detected[ck][stratum][c] for c in ids)
        return _recall_cell(total, det, key_to_conf[ck])

    results = {
        ck: {
            s: {"all_classes": cell(ck, s, all_ids), "VRU": cell(ck, s, VRU_CLASS_IDS)}
            for s in STRATA
        }
        for ck in conf_keys
    }

    # --- Galleries: occluded stratum at GALLERY_CONF, found vs missed ---
    galleries = _collect_galleries(
        samples, class_names, gallery_seed, gallery_n, weights_label, results,
    )

    return {
        "_miss_rate": round(miss_rate, 6),
        "_eval_images_n": len(samples),
        "_stratum_gt_counts": {
            s: {"all_classes": sum(gt_counts[s].values()),
                "VRU": sum(gt_counts[s][c] for c in VRU_CLASS_IDS)}
            for s in STRATA
        },
        "_caveats": CAVEATS,
        "_protocol_commit": _git_commit(),
        "results": results,
        "_galleries": galleries,
    }


def _collect_galleries(samples, class_names, seed, gallery_n, weights_label, results):
    """Build the four occluded found/missed gallery manifests (AC-11).

    Reuses `_match_gt_status` (same greedy as the count path) on the occluded GT of each
    image at GALLERY_CONF, then asserts the resulting found/missed totals equal the JSON
    `detected`/`missed` counts — guaranteeing AC-11b's `available == JSON count` holds by
    construction rather than by luck.
    """
    items = {slug: [] for slug in GALLERY_SLUGS}
    for img_path, w, h, preds, gt_records in samples:
        by_cls = defaultdict(list)
        for r in gt_records:
            if "occluded" in r["strata"]:
                by_cls[r["cls"]].append(r)
        for cls, recs in by_cls.items():
            p = [(b, c) for b, c in preds.get(cls, []) if c >= GALLERY_CONF]
            status = _match_gt_status(
                [b for b, _ in p], [c for _, c in p], [r["box"] for r in recs], RECALL_IOU,
            )
            for r, st in zip(recs, status):
                tag = "TP" if st is not None else "MISS"
                entry = {
                    "img_path": img_path, "w": w, "h": h, "box": r["box"], "cls": cls,
                    "name": f"{class_names[cls]} [occluded · {tag}]",
                    "conf": st if st is not None else 1.0,
                }
                found = st is not None
                items["occluded_found" if found else "occluded_missed"].append(entry)
                if cls in VRU_CLASS_IDS:
                    items["occluded_vru_found" if found else "occluded_vru_missed"].append(entry)

    # AC-11b reconciliation guard: gallery item totals must match the JSON counts.
    ck = _conf_key(GALLERY_CONF)
    occ = results[ck]["occluded"]
    expected = {
        "occluded_found": occ["all_classes"]["detected"],
        "occluded_missed": occ["all_classes"]["missed"],
        "occluded_vru_found": occ["VRU"]["detected"],
        "occluded_vru_missed": occ["VRU"]["missed"],
    }
    for slug, exp in expected.items():
        got = len(items[slug])
        assert got == exp, (
            f"gallery {slug} has {got} boxes but JSON count is {exp} — greedy match "
            f"divergence between _match_gt_status and match_detections"
        )

    return {
        slug: _build_gallery_manifest(items[slug], seed, gallery_n, weights_label)
        for slug in GALLERY_SLUGS
    }


# --- recall-view schema verification (AC-10) ---
REQUIRED_RECALL_META = ["weights", "dataset", "split", "eval_images_n", "seed",
                        "conf_thresholds", "protocol_commit", "miss_rate",
                        "stratum_gt_counts", "caveats"]
RECALL_CELL_KEYS = ["detected", "missed", "total_gt", "recall", "conf", "low_confidence"]


def write_galleries(galleries: dict, out_dir: Path, webview_src: Path = WEBVIEW_SRC) -> None:
    """Write each non-empty gallery as a self-contained webview directory (AC-11).

    Mirrors `scripts/evaluate_occlusion_recall.py`'s writer so the training-time hook and
    the CLI produce the same artifact layout: <slug>/manifest.js (JS global, file://-safe),
    the sampled tile images under <slug>/images/, and the three webview files copied
    verbatim from `webview_src`. Empty categories are skipped with a log line.
    """
    for slug in GALLERY_SLUGS:
        built = galleries.get(slug)
        if built is None:
            print(f"[occ-recall] gallery {slug}: 0 qualifying boxes — skipped")
            continue
        manifest, image_src_paths = built
        gdir = out_dir / slug
        (gdir / "images").mkdir(parents=True, exist_ok=True)
        for src in image_src_paths:
            shutil.copy(src, gdir / "images" / Path(src).name)
        (gdir / "manifest.js").write_text(
            "window.MANIFEST = " + json.dumps(manifest, indent=2, sort_keys=True) + ";\n"
        )
        for f in ("index.html", "style.css", "app.js"):
            shutil.copy(webview_src / f, gdir / f)
        print(f"[occ-recall] gallery {slug}: {manifest['sampled']}/{manifest['available']} "
              f"images sampled -> {gdir}")


def verify_recall_schema(doc: dict) -> list:
    """Return a list of schema problems for a recall artifact (empty list = AC-10 met)."""
    problems = []
    meta = doc.get("metadata", {})
    for k in REQUIRED_RECALL_META:
        if k not in meta:
            problems.append(f"metadata.{k} missing")
    results = doc.get("results", {})
    if not results:
        problems.append("results block missing or empty")
    for ck, by_stratum in results.items():
        for s in STRATA:
            if s not in by_stratum:
                problems.append(f"results.{ck}.{s} missing")
                continue
            for scope in ("all_classes", "VRU"):
                cell = by_stratum[s].get(scope)
                if cell is None:
                    problems.append(f"results.{ck}.{s}.{scope} missing")
                    continue
                for key in RECALL_CELL_KEYS:
                    if key not in cell:
                        problems.append(f"results.{ck}.{s}.{scope}.{key} missing")
    return problems
