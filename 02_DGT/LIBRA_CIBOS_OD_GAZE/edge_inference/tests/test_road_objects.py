"""`RoadObjectDetector` — dgt's adapter around road-object-detection's tiling recipe.

No engine, no GPU. The TensorRT engine is replaced by a stub that returns a
hand-built raw output, so the arithmetic is checkable by hand. What is under test
is dgt's side only: the engine handed to `infer_tiled`, the conf threshold passed
through, pixel boxes normalised to the frame, `valid` semantics, labels. The
recipe itself (grid, letterbox, NMS) is road-object-detection's and is tested
there, against Ultralytics.

Skips without road-object-detection (the headless suite needs only numpy +
pyyaml + pytest); `pip install -e ../road_object_detection` to run it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pytest

pytest.importorskip("road_object_detection.inference")

import dgt.models.road_objects as road_objects  # noqa: E402
from dgt.contracts.detections import BDD100K_CLASSES, BDD100K_TAXONOMY  # noqa: E402
from dgt.models.road_objects import RoadObjectDetector  # noqa: E402

NC = len(BDD100K_CLASSES)


def raw(boxes_cxcywh, class_id: int = 2, score: float = 0.9) -> np.ndarray:
    """(1, 4 + nc, N) the way the engine emits it, every box the same class/score."""
    out = np.zeros((1, 4 + NC, len(boxes_cxcywh)), dtype=np.float32)
    for i, box in enumerate(boxes_cxcywh):
        out[0, :4, i] = box
        out[0, 4 + class_id, i] = score
    return out


@dataclass
class StubEngine:
    """Stand-in for `models/trt.TensorRTEngine`: records inputs, returns `output`."""

    output: np.ndarray
    blobs: List[np.ndarray] = field(default_factory=list)
    closed: bool = False

    def infer(self, blob: np.ndarray) -> List[np.ndarray]:
        self.blobs.append(blob)
        return [self.output]

    def close(self) -> None:
        self.closed = True


def detector(output: np.ndarray, conf: float = 0.25, tile: bool = True) -> RoadObjectDetector:
    """A detector around a stub engine — skips `__init__`, which opens a real engine."""
    d = RoadObjectDetector.__new__(RoadObjectDetector)
    d.engine_path = Path("road_objects_640x640_fp16.engine")
    d.conf_threshold = conf
    d.tile = tile
    d.class_names = BDD100K_CLASSES
    d.taxonomy = BDD100K_TAXONOMY
    d._engine = StubEngine(output)
    d._in_shape = (1, 3, 640, 640)
    d._out_shape = (1, 4 + NC, 8400)
    return d


def frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --- the recipe is imported, not copied -------------------------------------

@pytest.mark.parametrize("name", ["tile_regions", "grid_regions", "letterbox", "decode_raw",
                                  "nms", "TILE_W", "TILE_H", "PAD_VALUE"])
def test_the_tiling_recipe_is_not_redefined_in_dgt(name):
    """One definition, road_object_detection.inference. A copy here drifts."""
    assert not hasattr(road_objects, name)


def test_regions_are_the_shared_grid():
    from road_object_detection.inference import grid_regions

    d = detector(raw([]))
    assert d.regions_for(1280, 720) == grid_regions(1280, 720)
    assert len(d.regions_for(1920, 1080)) == 9


def test_tile_size_is_the_recipe_s():
    """The gaze crop is cut at the size the weights were fine-tuned on."""
    from road_object_detection.inference import TILE_H, TILE_W

    assert detector(raw([])).tile_size == (TILE_W, TILE_H)


def test_no_tile_is_one_whole_frame_region():
    d = detector(raw([]), tile=False)
    assert d.regions_for(1280, 720) == [(0, 0, 1280, 720)]
    d.detect(frame())
    assert len(d._engine.blobs) == 1


# --- detect -------------------------------------------------------------------

def test_720p_runs_the_engine_once_per_tile_with_its_input_shape():
    d = detector(raw([]))
    d.detect(frame())
    assert [b.shape for b in d._engine.blobs] == [(1, 3, 640, 640)] * 4


def test_boxes_are_normalised_to_the_full_frame():
    """A 40x40 box at letterbox (320, 320) is the centre of each 640x360 tile:
    frame (320, 180) for the top-left tile -> (300..340, 160..200) / (1280, 720)."""
    d = detector(raw([(320, 320, 40, 40)]))
    result = d.detect(frame(), regions=[(0, 0, 640, 360)])
    assert result.valid and len(result.detections) == 1
    det = result.detections[0]
    assert det.box == pytest.approx((300 / 1280, 160 / 720, 340 / 1280, 200 / 720))
    assert det.label == "car" and det.class_id == 2 and det.score == pytest.approx(0.9)
    assert result.taxonomy == BDD100K_TAXONOMY


def test_every_tile_contributes_in_frame_coordinates():
    d = detector(raw([(320, 320, 40, 40)]))
    result = d.detect(frame())
    centres = sorted(((b.box[0] + b.box[2]) / 2 * 1280, (b.box[1] + b.box[3]) / 2 * 720)
                     for b in result.detections)
    flat = [v for centre in centres for v in centre]
    assert flat == pytest.approx([320, 180, 320, 540, 960, 180, 960, 540])


def test_the_configured_conf_threshold_is_applied():
    assert len(detector(raw([(320, 320, 40, 40)], score=0.2), conf=0.25)
               .detect(frame()).detections) == 0
    assert len(detector(raw([(320, 320, 40, 40)], score=0.2), conf=0.1)
               .detect(frame()).detections) == 4


def test_nothing_found_is_valid_and_empty():
    result = detector(raw([])).detect(frame())
    assert result.valid and result.detections == []


def test_no_regions_is_invalid_not_empty():
    d = detector(raw([(320, 320, 40, 40)]))
    result = d.detect(frame(), regions=[])
    assert not result.valid and d._engine.blobs == []


@pytest.mark.parametrize("bad", [None, np.zeros((720, 1280), np.uint8),
                                 np.zeros((720, 1280, 4), np.uint8)])
def test_a_bad_frame_is_invalid(bad):
    assert not detector(raw([])).detect(bad).valid


def test_describe_reports_the_grid_and_the_recipe():
    line = detector(raw([])).describe((720, 1280, 3))
    assert "grid=2x2" in line and "recipe=road-object-detection" in line
    assert "NO-OP" in detector(raw([])).describe((360, 640, 3))


def test_close_closes_the_engine():
    d = detector(raw([]))
    d.close()
    assert d._engine.closed
