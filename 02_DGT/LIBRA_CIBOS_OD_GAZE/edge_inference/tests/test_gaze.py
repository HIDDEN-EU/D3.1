"""Gaze model: engine binding, the face-crop preprocessing, and the validity rule.

Headless. The engine is stubbed — nothing here needs TensorRT or a device. The
preprocessing tests need cv2 and skip without it (`_require_preprocessing()`).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from dgt.contracts.gaze import GazeResult
from testgaze import GazeTestPipeline
from dgt.calibration import Calibration
from dgt.models.gaze import (
    FACE_INPUT_SIZE,
    GazeEstimator,
    _is_degenerate,
    preprocess_face_crop,
)

def _require_preprocessing() -> None:
    """Skip unless the preprocessing can actually run."""
    pytest.importorskip("cv2")


FACE_ELEMS = 3 * FACE_INPUT_SIZE[0] * FACE_INPUT_SIZE[1]


class StubEngine:
    """Stands in for TensorRTEngine: declares shapes and returns a fixed point."""

    def __init__(self, input_shapes, output_shapes=((1, 2),), point=(0.25, 0.75)):
        self.engine_path = Path("stub_gaze.engine")
        self.input_names = [f"in{i}" for i in range(len(input_shapes))]
        self.input_shapes = [tuple(s) for s in input_shapes]
        self.output_names = ["out0"]
        self.output_shapes = [tuple(s) for s in output_shapes]
        self.point = point
        self.calls: List[Tuple[np.ndarray, ...]] = []
        self.closed = False

    def infer(self, *blobs):
        self.calls.append(blobs)
        return [np.array([list(self.point)], dtype=np.float32)]

    def close(self):
        self.closed = True


class StubCalibration:
    """dgt.calibration.Calibration's surface: `available` and `apply`."""

    def __init__(self, available=True, mapped=(6.0, 3.0)):
        self.available = available
        self.mapped = mapped

    def apply(self, x, y):
        return None if not self.available else self.mapped


def build(monkeypatch, engine):
    monkeypatch.setattr("dgt.models.gaze.TensorRTEngine", lambda _path: engine)
    return engine


def estimator(monkeypatch, engine=None, **kwargs):
    engine = engine or StubEngine([(1, 3, 112, 112)])
    build(monkeypatch, engine)
    return GazeEstimator(Path("stub_gaze.engine"), **kwargs), engine


# -- engine binding ---------------------------------------------------------


def test_the_face_input_is_bound(monkeypatch):
    gaze, _ = estimator(monkeypatch, StubEngine([(1, 3, 112, 112)]))
    assert gaze.describe().startswith("stub_gaze.engine")


def test_an_engine_with_two_inputs_is_rejected(monkeypatch):
    build(monkeypatch, StubEngine([(1, 3, 112, 112), (1, 1, 64, 64)]))
    with pytest.raises(ValueError, match="not the gaze model"):
        GazeEstimator(Path("stub_gaze.engine"))


def test_an_engine_whose_input_is_the_wrong_size_is_rejected(monkeypatch):
    build(monkeypatch, StubEngine([(1, 3, 96, 96)]))
    with pytest.raises(ValueError, match="not the gaze model"):
        GazeEstimator(Path("stub_gaze.engine"))


def test_an_engine_that_emits_more_than_a_point_is_rejected(monkeypatch):
    build(monkeypatch, StubEngine([(1, 3, 112, 112)], output_shapes=((1, 3),)))
    with pytest.raises(ValueError, match="one output of 2 values"):
        GazeEstimator(Path("stub_gaze.engine"))


# -- the validity rule ------------------------------------------------------


def test_no_box_is_an_explicit_non_answer(monkeypatch):
    gaze, engine = estimator(monkeypatch)
    result = gaze.estimate(np.zeros((480, 640, 3), np.uint8), None)
    assert result.valid is False and result.x is None and result.y is None
    assert gaze.last_raw is None
    assert engine.calls == [], "the engine must not run without a face"


@pytest.mark.parametrize("bbox", [(10, 10, 10, 50), (10, 10, 50, 10), (50, 50, 10, 10)])
def test_a_degenerate_box_never_reaches_the_engine(monkeypatch, bbox):
    gaze, engine = estimator(monkeypatch)
    assert gaze.estimate(np.zeros((480, 640, 3), np.uint8), bbox).valid is False
    assert engine.calls == []


def test_is_degenerate_accepts_a_real_box():
    assert _is_degenerate((10, 10, 50, 50)) is False


def test_a_frame_that_is_not_an_image_is_a_non_answer(monkeypatch):
    gaze, engine = estimator(monkeypatch)
    assert gaze.estimate(np.zeros((480, 640), np.uint8), (1, 1, 9, 9)).valid is False
    assert engine.calls == []


# -- preprocessing (needs cv2) ----------------------------------------------


def test_the_face_crop_is_chw_and_normalized_to_minus_one_to_one():
    _require_preprocessing()
    frame = np.full((480, 640, 3), 255, np.uint8)
    blob = preprocess_face_crop(frame, (100, 100, 300, 300))
    assert blob.shape == (3, 112, 112) and blob.dtype == np.float32
    assert np.allclose(blob, 1.0), "255 must map to +1"
    assert np.allclose(preprocess_face_crop(np.zeros((480, 640, 3), np.uint8),
                                            (100, 100, 300, 300)), -1.0)


def test_the_face_crop_converts_bgr_to_rgb():
    _require_preprocessing()
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 0] = 255  # pure BLUE in BGR
    blob = preprocess_face_crop(frame, (100, 100, 300, 300))
    # After BGR->RGB the blue channel is last.
    assert np.allclose(blob[2], 1.0) and np.allclose(blob[0], -1.0)


def test_a_degenerate_crop_returns_a_buffer_of_the_right_shape():
    _require_preprocessing()
    blob = preprocess_face_crop(np.zeros((480, 640, 3), np.uint8), (10, 10, 10, 50))
    assert blob.shape == (3, 112, 112) and blob.dtype == np.float32


def test_a_box_past_the_frame_edge_is_clipped():
    _require_preprocessing()
    frame = np.full((480, 640, 3), 255, np.uint8)
    blob = preprocess_face_crop(frame, (600, 400, 700, 500))
    assert blob.shape == (3, 112, 112) and np.allclose(blob, 1.0)


# -- the contract, end to end ----------------------------------------------


def test_without_calibration_a_real_face_is_still_not_a_measurement(monkeypatch):
    """The engine runs and the raw pair is kept, but nothing valid comes out.

    The raw output lives in the training-normalized space whose bounds were
    never logged, so there is no way to turn it into the [0, 1] image frame the
    contract promises. Reporting it as valid would be a fabricated measurement.
    """
    _require_preprocessing()
    gaze, engine = estimator(monkeypatch)
    result = gaze.estimate(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    assert result.valid is False and result.x is None and result.y is None
    assert gaze.last_raw == pytest.approx((0.25, 0.75))
    assert len(engine.calls) == 1, "the engine did run"


def test_an_unavailable_calibration_is_the_same_as_none(monkeypatch):
    _require_preprocessing()
    gaze, _ = estimator(monkeypatch, calibration=StubCalibration(available=False))
    result = gaze.estimate(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    assert result.valid is False


def test_a_calibrated_point_is_normalized_by_the_screen_grid(monkeypatch):
    _require_preprocessing()
    gaze, _ = estimator(monkeypatch, calibration=StubCalibration(mapped=(6.0, 3.0)),
                        screen_grid=(12.0, 6.0))
    result = gaze.estimate(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    assert result.valid is True
    assert (result.x, result.y) == pytest.approx((0.5, 0.5))


def test_confidence_is_none_because_the_model_emits_none(monkeypatch):
    _require_preprocessing()
    gaze, _ = estimator(monkeypatch, calibration=StubCalibration())
    result = gaze.estimate(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    assert result.confidence is None


def test_the_engine_gets_one_face_blob(monkeypatch):
    _require_preprocessing()
    gaze, engine = estimator(monkeypatch)
    gaze.estimate(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    (face,) = engine.calls[0]
    assert face.shape == (3, 112, 112) and face.size == FACE_ELEMS


def test_a_fitted_calibration_drives_the_result(monkeypatch):
    _require_preprocessing()
    identity = Calibration(np.eye(3))
    gaze, _ = estimator(monkeypatch, calibration=identity)
    result = gaze.estimate(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    assert result.valid is True
    assert (result.x, result.y) == pytest.approx((0.25, 0.75))


def test_close_releases_the_engine(monkeypatch):
    gaze, engine = estimator(monkeypatch)
    gaze.close()
    assert engine.closed is True


def test_it_works_as_a_context_manager(monkeypatch):
    gaze, engine = estimator(monkeypatch)
    with gaze:
        pass
    assert engine.closed is True


def test_describe_says_when_it_is_uncalibrated(monkeypatch):
    gaze, _ = estimator(monkeypatch)
    assert "UNCALIBRATED" in gaze.describe()


# -- the pipeline -----------------------------------------------------------


class StubBoxDetector:
    """A face detector that returns a scripted sequence of boxes."""

    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.index = 0
        self.closed = False

    def detect_face_box(self, frame):
        box = self.boxes[min(self.index, len(self.boxes) - 1)]
        self.index += 1
        return None if box is None else np.array(box, dtype=np.float32)

    def close(self):
        self.closed = True


class StubGazeModel:
    """A gaze model that emits a scripted sequence of raw points."""

    def __init__(self, points, valid=False):
        self.points = list(points)
        self.index = 0
        self.valid = valid
        self.last_raw = None
        self.seen = []
        self.closed = False

    def estimate(self, frame, bbox):
        self.seen.append(bbox)
        if bbox is None:
            self.last_raw = None
            return GazeResult(x=None, y=None, confidence=None, valid=False)
        point = self.points[min(self.index, len(self.points) - 1)]
        self.index += 1
        self.last_raw = point
        if not self.valid:
            return GazeResult(x=None, y=None, confidence=None, valid=False)
        return GazeResult(x=point[0], y=point[1], confidence=None, valid=True)

    def close(self):
        self.closed = True


def pipeline(boxes=((10, 10, 50, 50),), points=((0.4, 0.6),), valid=False):
    return GazeTestPipeline(StubBoxDetector(boxes), StubGazeModel(points, valid=valid))


def test_the_pipeline_requires_the_driver_frame():
    assert GazeTestPipeline.REQUIRED_FRAMES == ("driver",)


def test_tick_rejects_a_frame_it_does_not_own():
    with pytest.raises(ValueError, match="expects frames"):
        pipeline().tick(road=np.zeros((8, 8, 3), np.uint8))


def test_the_pipeline_stamps_the_envelope_on_the_contract():
    """The model leaves sync unset; stamping it is the pipeline's job."""
    gaze = pipeline()
    result = gaze.tick(driver=np.zeros((8, 8, 3), np.uint8), timestamp=1.5)
    assert result.sync.frame_id == 0 and result.sync.timestamp == 1.5
    assert result.gaze.sync is result.sync


def test_frame_ids_advance_monotonically():
    gaze = pipeline()
    ids = [gaze.tick(driver=np.zeros((8, 8, 3), np.uint8)).sync.frame_id
           for _ in range(3)]
    assert ids == [0, 1, 2]


def test_a_frame_with_no_face_never_reaches_the_gaze_model():
    gaze = GazeTestPipeline(StubBoxDetector([None]), StubGazeModel([(0.4, 0.6)]))
    result = gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert result.face_box is None and result.raw is None
    assert result.gaze.valid is False
    assert gaze.estimator.seen == [None], "the model was called with no box"
    assert gaze.no_face == 1 and gaze.faces == 0


def test_the_raw_pair_is_carried_beside_the_contract_not_inside_it():
    """Raw must never be mistaken for a measurement — hence its own field."""
    gaze = pipeline(points=((0.25, 0.75),))
    result = gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert result.raw == pytest.approx((0.25, 0.75))
    assert result.gaze.valid is False and result.gaze.x is None


def test_a_calibrated_model_counts_as_valid():
    gaze = pipeline(points=((0.25, 0.75),), valid=True)
    result = gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert result.gaze.valid is True and gaze.valid_results == 1


def test_the_face_rate_counts_boxes_not_valid_results():
    gaze = GazeTestPipeline(StubBoxDetector([(1, 1, 9, 9), None, (1, 1, 9, 9)]),
                            StubGazeModel([(0.4, 0.6)]))
    for _ in range(3):
        gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert gaze.faces == 2 and gaze.no_face == 1
    assert gaze.face_rate == pytest.approx(2 / 3)


def test_spread_needs_two_points_before_it_says_anything():
    gaze = pipeline()
    assert gaze.raw_std is None and gaze.raw_is_stuck is None
    gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert gaze.raw_std is None, "one point has no spread"


def test_an_unmoving_raw_output_is_reported_as_stuck():
    """The symptom of a broken input: a number that never changes."""
    gaze = pipeline(points=((0.5, 0.5),))
    for _ in range(5):
        gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert gaze.raw_is_stuck is True
    assert "STUCK" in gaze.summary()


def test_a_moving_raw_output_is_not_stuck():
    gaze = pipeline(boxes=[(1, 1, 9, 9)] * 4,
                    points=[(0.1, 0.9), (0.4, 0.5), (0.8, 0.2), (0.3, 0.7)])
    for _ in range(4):
        gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    assert gaze.raw_is_stuck is False
    assert gaze.raw_std[0] > 0 and gaze.raw_std[1] > 0
    assert "STUCK" not in gaze.summary()


def test_reset_clears_the_counters_and_the_spread():
    gaze = pipeline()
    for _ in range(3):
        gaze.tick(driver=np.zeros((8, 8, 3), np.uint8))
    gaze.reset()
    assert gaze.frames == 0 and gaze.faces == 0 and gaze.raw_std is None


def test_close_releases_both_models():
    gaze = pipeline()
    gaze.close()
    assert gaze.detector.closed is True and gaze.estimator.closed is True


def test_mean_latency_is_none_before_the_first_frame():
    assert pipeline().mean_inference_ms is None
