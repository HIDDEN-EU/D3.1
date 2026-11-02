"""
Unit tests for `road_object_detection.inference` — the tiled-inference definition that
the edge service (../edge_inference) imports on the Jetson.

NumPy + OpenCV only: no model, no torch, no ONNX Runtime. `run` is a stub that returns
a hand-built raw graph output, so every number below is checkable by hand. Runnable
under pytest or standalone:

    python tests/test_inference.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from road_object_detection import inference as inf


def _raw(boxes_cxcywh, class_scores) -> np.ndarray:
    """Build a (1, 4 + nc, N) raw output from per-box cx,cy,w,h and per-class scores."""
    pred = np.concatenate([np.asarray(boxes_cxcywh, np.float32),
                           np.asarray(class_scores, np.float32)], axis=1)
    return pred.T[None]


def test_module_import_is_light():
    """The Orin imports this module: it must not drag in torch, ultralytics or cv2.

    Run in a fresh interpreter — other test files legitimately import the heavy stack."""
    code = ("import sys; import road_object_detection.inference; "
            "bad = [m for m in ('torch', 'torchvision', 'ultralytics', 'cv2', 'mlflow', "
            "'onnxruntime') if m in sys.modules]; print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True).stdout.strip()
    assert out == "", f"heavy modules leaked on import: {out}"


# --- geometry -----------------------------------------------------------------


def test_grid_is_resolution_aware():
    assert len(inf.grid_regions(1280, 720)) == 4      # 2x2
    assert len(inf.grid_regions(1920, 1080)) == 9     # 3x3
    assert inf.grid_regions(640, 360) == [(0, 0, 640, 360)]


def test_grid_covers_the_frame_exactly_without_overlap():
    for w, h in [(1280, 720), (1920, 1080), (1641, 923)]:
        regions = inf.grid_regions(w, h)
        mask = np.zeros((h, w), np.int32)
        for x0, y0, x1, y1 in regions:
            mask[y0:y1, x0:x1] += 1
        assert (mask == 1).all(), f"{w}x{h}: gap or overlap"


def test_grid_is_row_major():
    assert inf.grid_regions(1280, 720) == [
        (0, 0, 640, 360), (640, 0, 1280, 360),
        (0, 360, 640, 720), (640, 360, 1280, 720),
    ]


# --- one tile -----------------------------------------------------------------


def test_letterbox_tile_geometry():
    """640x360 tile -> 640x640: no resize, 140 px of 114 pad top and bottom."""
    tile = np.zeros((360, 640, 3), np.uint8)
    blob, gain, pad = inf.letterbox(tile)
    assert blob.shape == (1, 3, 640, 640) and blob.dtype == np.float32
    assert gain == 1.0 and pad == (0, 140)
    assert np.allclose(blob[0, :, :140], 114 / 255) and np.allclose(blob[0, :, 140:500], 0)


def test_letterbox_is_bgr_to_rgb():
    tile = np.zeros((360, 640, 3), np.uint8)
    tile[..., 0] = 255                                 # pure blue in BGR
    blob, _, _ = inf.letterbox(tile)
    assert blob[0, 2, 320, 320] == 1.0 and blob[0, 0, 320, 320] == 0.0


def test_nms_suppresses_strictly_above_threshold():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5], [20, 20, 30, 30]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    # boxes 0 and 1 overlap at IoU exactly 0.5
    assert inf.nms(boxes, scores, 0.5).tolist() == [0, 1, 2]   # 0.5 is NOT > 0.5
    assert inf.nms(boxes, scores, 0.4).tolist() == [0, 2]


def test_batched_nms_never_suppresses_across_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert sorted(inf.batched_nms(boxes, scores, np.array([0, 1]), 0.5).tolist()) == [0, 1]
    assert inf.batched_nms(boxes, scores, np.array([0, 0]), 0.5).tolist() == [0]


def test_postprocess_conf_is_strictly_greater():
    raw = _raw([[100, 100, 20, 20], [300, 300, 20, 20]], [[0.25, 0.0], [0.0, 0.3]])
    boxes, scores, ids = inf.postprocess(raw, conf=0.25)
    assert ids.tolist() == [1] and np.allclose(scores, [0.3])
    assert np.allclose(boxes, [[290, 290, 310, 310]])


def test_postprocess_empty():
    boxes, scores, ids = inf.postprocess(_raw([[1, 1, 1, 1]], [[0.1]]))
    assert boxes.shape == (0, 4) and len(scores) == 0 and len(ids) == 0


def test_scale_boxes_undoes_letterbox_and_clips():
    out = inf.scale_boxes(np.array([[10, 130, 650, 200]], np.float32), 1.0, (0, 140), (360, 640))
    assert np.allclose(out, [[10, 0, 640, 60]])


# --- the full frame -----------------------------------------------------------


def test_infer_tiled_lifts_boxes_to_frame_pixels():
    """A box at letterbox (320, 320) is the centre of every tile; the bottom-right tile
    puts it at frame (960, 540)."""
    calls = []

    def run(blob):
        calls.append(blob.shape)
        return _raw([[320, 320, 40, 40]], [[0.0, 0.9]])

    frame = np.zeros((720, 1280, 3), np.uint8)
    det = inf.infer_tiled(run, frame, select=[3])
    assert calls == [(1, 3, 640, 640)]
    assert len(det) == 1 and det.tiles.tolist() == [3] and det.class_ids.tolist() == [1]
    assert np.allclose(det.boxes, [[940, 520, 980, 560]])
    assert len(det.regions) == 4                     # geometry is the full grid regardless


def _run_sequence(outputs):
    it = iter(outputs)
    return lambda blob: next(it)


_EMPTY = _raw(np.zeros((0, 4)), np.zeros((0, 1)))


def test_seam_object_is_one_box_per_tile_on_the_default_grid():
    """Boxes are clipped to their tile and grid tiles do not overlap, so an object
    across the x=640 seam survives as two abutting boxes: the merge cannot see it.
    Pinned on purpose — changing it is a recipe change."""
    left = _raw([[620, 320, 60, 40]], [[0.9]])         # tile x 590..650 -> clipped 590..640
    right = _raw([[10, 320, 60, 40]], [[0.8]])         # tile x -20..40 -> clipped 0..40
    frame = np.zeros((720, 1280, 3), np.uint8)
    det = inf.infer_tiled(_run_sequence([left, right, _EMPTY, _EMPTY]), frame)
    order = np.argsort(det.boxes[:, 0])
    assert len(det) == 2
    assert np.allclose(det.boxes[order], [[590, 160, 640, 200], [640, 160, 680, 200]])


def test_merge_acts_when_regions_overlap():
    """Two overlapping regions that both see the same object -> one box, the stronger."""
    box = _raw([[320, 320, 40, 40]], [[0.9]])
    weaker = _raw([[300, 320, 40, 40]], [[0.8]])       # 20 px left in a region 20 px right
    frame = np.zeros((360, 1280, 3), np.uint8)
    regions = [(0, 0, 640, 360), (20, 0, 660, 360)]
    det = inf.infer_tiled(_run_sequence([box, weaker]), frame, regions=regions)
    assert len(det) == 1 and det.tiles.tolist() == [0]
    assert np.allclose(det.scores, [0.9])


def test_infer_tiled_nothing_found_is_empty_not_none():
    frame = np.zeros((720, 1280, 3), np.uint8)
    det = inf.infer_tiled(lambda blob: _raw([[1, 1, 1, 1]], [[0.0]]), frame)
    assert len(det) == 0 and det.boxes.shape == (0, 4) and len(det.regions) == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
