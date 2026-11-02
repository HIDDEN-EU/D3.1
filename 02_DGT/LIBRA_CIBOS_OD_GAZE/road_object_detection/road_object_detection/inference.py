"""
Tiled inference — the ONE definition of how these weights are run on a full frame.

The model was fine-tuned on ~640x360 crops, so a full
frame is never fed to it whole: it is cut into a grid of ~640x360 tiles, each tile is
letterboxed and inferred, per-tile detections get NMS, are clipped to their tile
and lifted back to frame pixels, and a second class-wise NMS runs across tiles.
Every number that recipe depends on lives here, once.

Seams: on the default grid the tiles do not overlap and every box is clipped to its
tile, so boxes from different tiles never overlap and the cross-tile NMS removes
nothing — an object cut by a seam comes out as one box per tile. The merge only
acts when the caller passes overlapping `regions`. This is the behaviour every
tiled viewer in this repo has always had; changing it is a recipe change, made here.

Who imports it
--------------
- The edge service (../edge_inference) on the Jetson Orin, around its TensorRT engine. This is why
  the module is NumPy only: the base install of this package pulls nothing else, and
  `cv2` is imported lazily and never declared (the Orin uses JetPack's OpenCV; a pip
  `opencv-python` there breaks it).
- This repo's own viewers and evaluation, around ONNX Runtime or Ultralytics.

Backend-agnostic: `infer_tiled` takes `run(blob) -> raw`, a callable from the
letterboxed NCHW float32 blob to the graph's raw output `(1, 4 + nc, N)` (the
`onnx_nms=none` export). Whoever owns the engine owns `run`; nothing here touches a
model library.

Fidelity: `letterbox`, `postprocess` and `scale_boxes` reproduce Ultralytics 8.4.31's
`LetterBox(auto=False)`, `non_max_suppression` and `scale_boxes` — the code every
metric in this repo was measured through. `tests/test_inference_ultralytics_parity.py`
holds them to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Crop geometry the weights were fine-tuned on — a property of the weights, not a knob.
TILE_W, TILE_H = 640, 360

#: Network input side; the ONNX export is static `640x640` (`onnx_imgsz`).
IMGSZ = 640

#: Ultralytics' letterbox fill.
PAD_VALUE = 114

#: Per-tile NMS IoU — Ultralytics' predict/val default, what every eval here ran at.
TILE_IOU = 0.7

#: Cross-tile NMS IoU (the `--merge-iou` default of the tiled viewers). A no-op on
#: the non-overlapping default grid — see the module docstring.
MERGE_IOU = 0.5

#: Default confidence floor (Ultralytics' predict default). A runtime choice, not a
#: property of the weights: callers may tune it.
CONF = 0.25

#: Ultralytics' per-image caps: boxes entering NMS, detections leaving it.
MAX_NMS = 30000
MAX_DET = 300

#: Class offset for class-wise NMS in one pass (Ultralytics' `max_wh`).
_CLASS_OFFSET = 7680

Region = Tuple[int, int, int, int]
Run = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TiledDetections:
    """Detections in full-frame pixels, one row per box, after the cross-tile NMS."""

    boxes: np.ndarray       # (N, 4) float32 xyxy, frame pixels, clipped to the frame
    scores: np.ndarray      # (N,) float32
    class_ids: np.ndarray   # (N,) int64
    tiles: np.ndarray       # (N,) int64, index into `regions` the box came from
    regions: List[Region]   # the grid that was used (all of it, whatever `select` was)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


# --- geometry -----------------------------------------------------------------


def tile_regions(width: int, height: int, rows: int = 2, cols: int = 2) -> List[Region]:
    """(x0, y0, x1, y1) regions of a rows x cols grid that covers the frame exactly, row-major."""
    xs = [round(c * width / cols) for c in range(cols + 1)]
    ys = [round(r * height / rows) for r in range(rows + 1)]
    return [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(rows) for c in range(cols)]


def grid_regions(width: int, height: int, tile_w: int = TILE_W, tile_h: int = TILE_H) -> List[Region]:
    """Resolution-aware grid of ~tile_w x tile_h cells, no overlap.

    1280x720 -> 2x2, 1920x1080 -> 3x3. Below ~960x540 the frame rounds to ONE tile:
    tiling is then a no-op and the model sees the whole (small) frame."""
    cols = max(1, round(width / tile_w))
    rows = max(1, round(height / tile_h))
    return tile_regions(width, height, rows=rows, cols=cols)


# --- one tile -----------------------------------------------------------------


def letterbox(image: np.ndarray, size: int = IMGSZ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """BGR HxWx3 uint8 -> (1x3xSxS float32 RGB blob in [0, 1], gain, (pad_x, pad_y)).

    Ultralytics `LetterBox(new_shape=(size, size), auto=False)` + its predict
    preprocessing (BGR2RGB, /255, NCHW): aspect-preserving resize, centred pad of 114."""
    import cv2  # lazy: importing this module must not need OpenCV

    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = round(w * gain), round(h * gain)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    if (w, h) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=(PAD_VALUE,) * 3)
    blob = image[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob, dtype=np.float32) / 255.0, gain, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS, torchvision semantics: highest score first, drop IoU > threshold."""
    if not len(boxes):
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[~(iou > iou_threshold)]  # NaN (zero-area pair) counts as no overlap
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                iou_threshold: float) -> np.ndarray:
    """Class-wise NMS (boxes of different classes never suppress each other), kept indices by score."""
    if not len(boxes):
        return np.zeros(0, dtype=np.int64)
    # Same trick as torchvision/Ultralytics: shift each class into its own coordinate block.
    offset = class_ids.astype(boxes.dtype)[:, None] * (float(boxes.max()) + 1.0)
    return nms(boxes + offset, scores, iou_threshold)


def postprocess(raw: np.ndarray, conf: float = CONF, iou: float = TILE_IOU,
                max_det: int = MAX_DET) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw graph output (1, 4 + nc, N) -> (xyxy, scores, class_ids) in letterbox pixels.

    Ultralytics `non_max_suppression` for one image, single label per box: best class,
    keep score > conf, cap at MAX_NMS by score, class-wise NMS, cap at max_det."""
    pred = np.asarray(raw, dtype=np.float32)[0].T             # (N, 4 + nc)
    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(pred)), class_ids]
    keep = scores > conf
    pred, scores, class_ids = pred[keep], scores[keep], class_ids[keep]
    if len(scores) > MAX_NMS:
        top = np.argsort(-scores, kind="stable")[:MAX_NMS]
        pred, scores, class_ids = pred[top], scores[top], class_ids[top]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    i = nms(boxes + class_ids[:, None].astype(np.float32) * _CLASS_OFFSET, scores, iou)[:max_det]
    return boxes[i], scores[i], class_ids[i].astype(np.int64)


def scale_boxes(boxes: np.ndarray, gain: float, pad: Tuple[int, int],
                shape: Tuple[int, int]) -> np.ndarray:
    """Letterbox pixels -> source-image pixels, clipped to `shape` (h, w). Ultralytics `scale_boxes`."""
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] -= pad[0]
    out[:, [1, 3]] -= pad[1]
    out /= gain
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, shape[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, shape[0])
    return out


# --- the full frame -----------------------------------------------------------


def merge_tiles(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                iou: float = MERGE_IOU) -> np.ndarray:
    """Class-wise NMS over every tile's frame-pixel boxes; kept indices, ascending.

    Removes nothing unless the regions overlap (boxes are clipped to their tile)."""
    return np.sort(batched_nms(boxes, scores, class_ids, iou))


def infer_tiled(run: Run, frame: np.ndarray, conf: float = CONF,
                regions: Optional[Sequence[Region]] = None,
                select: Optional[Iterable[int]] = None,
                imgsz: int = IMGSZ, tile_iou: float = TILE_IOU,
                merge_iou: float = MERGE_IOU) -> TiledDetections:
    """Run the model over a full BGR frame, tile by tile, then NMS across tiles.

    `regions` defaults to `grid_regions(w, h)`. `select` restricts inference to those
    region indices; geometry and the `tiles` index stay stable whatever is selected.
    A frame with no detections returns an empty `TiledDetections` — it ran and found
    nothing; failure is the caller's `run` raising."""
    h, w = frame.shape[:2]
    regions = list(grid_regions(w, h) if regions is None else regions)
    active = set(range(len(regions))) if select is None else set(select)

    parts = []
    for ti, (x0, y0, x1, y1) in enumerate(regions):
        if ti not in active:
            continue
        crop = frame[y0:y1, x0:x1]
        blob, gain, pad = letterbox(crop, imgsz)
        boxes, scores, ids = postprocess(run(blob), conf, tile_iou)
        if not len(boxes):
            continue
        boxes = scale_boxes(boxes, gain, pad, crop.shape[:2])
        boxes[:, [0, 2]] += x0
        boxes[:, [1, 3]] += y0
        parts.append((boxes, scores, ids, np.full(len(boxes), ti, dtype=np.int64)))

    if not parts:
        return TiledDetections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                               np.zeros(0, np.int64), np.zeros(0, np.int64), regions)
    boxes, scores, ids, tiles = (np.concatenate(p) for p in zip(*parts))
    keep = merge_tiles(boxes, scores, ids, merge_iou)
    return TiledDetections(boxes[keep], scores[keep], ids[keep], tiles[keep], regions)
