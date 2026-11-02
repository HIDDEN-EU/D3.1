"""The two-frame pipeline: frame ownership, the shared envelope, per-path cost.

Headless — every model is a stub, so nothing here needs an engine or a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from dgt.contracts.detections import Detection, DetectionResult
from dgt.contracts.gaze import GazeResult
from testroadgaze import RoadGazeFrameResult, RoadGazeTestPipeline


def frame(h: int = 32, w: int = 32) -> np.ndarray:
    return np.zeros((h, w, 3), np.uint8)


class StubRoadDetector:
    def __init__(self, valid=True, count=1):
        self.valid = valid
        self.count = count
        self.seen = []
        self.closed = False

    def detect(self, image):
        self.seen.append(image)
        detections = [
            Detection(label="car", class_id=0, score=0.9, box=(1.0, 1.0, 2.0, 2.0))
            for _ in range(self.count)
        ]
        return DetectionResult(detections=detections, valid=self.valid)

    def close(self):
        self.closed = True


class StubFaceDetector:
    def __init__(self, boxes=((1, 1, 9, 9),)):
        self.boxes = list(boxes)
        self.index = 0
        self.seen = []
        self.closed = False

    def detect_face_box(self, image):
        self.seen.append(image)
        box = self.boxes[min(self.index, len(self.boxes) - 1)]
        self.index += 1
        return None if box is None else np.array(box, dtype=np.float32)

    def close(self):
        self.closed = True


class StubGaze:
    def __init__(self, point=(0.3, 0.7), valid=False):
        self.point = point
        self.valid = valid
        self.last_raw = None
        self.seen = []
        self.closed = False

    def estimate(self, image, bbox):
        self.seen.append((image, bbox))
        if bbox is None:
            self.last_raw = None
            return GazeResult(x=None, y=None, confidence=None, valid=False)
        self.last_raw = self.point
        if not self.valid:
            return GazeResult(x=None, y=None, confidence=None, valid=False)
        return GazeResult(x=self.point[0], y=self.point[1],
                          confidence=None, valid=True)

    def close(self):
        self.closed = True


def build(road=None, face=None, gaze=None, **kwargs):
    return RoadGazeTestPipeline(
        road or StubRoadDetector(), face or StubFaceDetector(),
        gaze or StubGaze(), **kwargs,
    )


# -- frame ownership --------------------------------------------------------


def test_it_requires_both_frames():
    assert RoadGazeTestPipeline.REQUIRED_FRAMES == ("road", "driver")


def test_one_frame_alone_is_rejected():
    with pytest.raises(ValueError, match="missing=\\['driver'\\]"):
        build().tick(road=frame())


def test_an_extra_frame_is_rejected():
    with pytest.raises(ValueError, match="unexpected"):
        build().tick(road=frame(), driver=frame(), cabin=frame())


def test_each_model_gets_its_own_frame():
    """The whole point of two inputs: they must not be crossed."""
    road_img, driver_img = frame(8, 8), frame(16, 16)
    road, face, gaze = StubRoadDetector(), StubFaceDetector(), StubGaze()
    build(road, face, gaze).tick(road=road_img, driver=driver_img)
    assert road.seen[0].shape == (8, 8, 3)
    assert face.seen[0].shape == (16, 16, 3)
    assert gaze.seen[0][0].shape == (16, 16, 3), "gaze must read the DRIVER frame"


# -- the shared envelope ----------------------------------------------------


def test_one_tick_stamps_one_envelope_on_both_contracts():
    """The shared stamp is the only thing binding the pair together."""
    result = build().tick(road=frame(), driver=frame(), timestamp=4.5)
    assert result.sync.timestamp == 4.5 and result.sync.frame_id == 0
    assert result.road.sync is result.sync
    assert result.gaze.sync is result.sync


def test_frame_ids_advance_once_per_pair_not_once_per_frame():
    pipeline = build()
    ids = [pipeline.tick(road=frame(), driver=frame()).sync.frame_id
           for _ in range(3)]
    assert ids == [0, 1, 2]


# -- the caller's promise ---------------------------------------------------


def test_tick_reports_no_skew_because_there_is_none_to_report():
    assert build().tick(road=frame(), driver=frame()).frame_skew_ms is None


def test_tick_pair_records_the_skew_the_caller_measured():
    result = build().tick_pair(road=frame(), driver=frame(),
                               road_timestamp=10.0, driver_timestamp=10.025)
    assert result.frame_skew_ms == pytest.approx(25.0)


def test_the_envelope_takes_the_road_timestamp():
    result = build().tick_pair(road=frame(), driver=frame(),
                               road_timestamp=10.0, driver_timestamp=10.025)
    assert result.sync.timestamp == 10.0


def test_skew_is_absolute_whichever_frame_came_first():
    result = build().tick_pair(road=frame(), driver=frame(),
                               road_timestamp=10.030, driver_timestamp=10.0)
    assert result.frame_skew_ms == pytest.approx(30.0)


def test_skew_does_not_leak_into_the_next_tick():
    """A stale skew would misreport a pair the caller made no promise about."""
    pipeline = build()
    pipeline.tick_pair(road=frame(), driver=frame(),
                       road_timestamp=1.0, driver_timestamp=1.1)
    assert pipeline.tick(road=frame(), driver=frame()).frame_skew_ms is None


# -- the two contracts stay separate ----------------------------------------


def test_the_two_contracts_are_carried_side_by_side():
    result = build(gaze=StubGaze(valid=True)).tick(road=frame(), driver=frame())
    assert isinstance(result, RoadGazeFrameResult)
    assert result.road.valid is True and len(result.road.detections) == 1
    assert result.gaze.valid is True


def test_a_driver_frame_with_no_face_does_not_affect_the_road_path():
    """The two paths are independent — one failing must not blank the other."""
    result = build(face=StubFaceDetector([None])).tick(road=frame(), driver=frame())
    assert result.gaze.valid is False and result.face_box is None and result.raw is None
    assert result.road.valid is True, "the road detections stand on their own"


def test_an_invalid_road_result_does_not_affect_gaze():
    result = build(road=StubRoadDetector(valid=False),
                   gaze=StubGaze(valid=True)).tick(road=frame(), driver=frame())
    assert result.road.valid is False
    assert result.gaze.valid is True


def test_the_raw_pair_is_carried_beside_the_gaze_contract():
    result = build(gaze=StubGaze(point=(0.25, 0.75))).tick(road=frame(), driver=frame())
    assert result.raw == pytest.approx((0.25, 0.75))
    assert result.gaze.valid is False and result.gaze.x is None


# -- cost -------------------------------------------------------------------


def test_the_two_paths_are_timed_separately():
    """A combined number would hide which path got slower under contention."""
    result = build().tick(road=frame(), driver=frame())
    assert result.road_ms >= 0.0 and result.driver_ms >= 0.0
    assert result.inference_ms == pytest.approx(result.road_ms + result.driver_ms)


def test_means_are_none_before_the_first_frame():
    pipeline = build()
    assert pipeline.mean_road_ms is None
    assert pipeline.mean_driver_ms is None
    assert pipeline.mean_inference_ms is None
    assert pipeline.face_rate is None


def test_the_counters_track_each_path():
    pipeline = build(face=StubFaceDetector([(1, 1, 9, 9), None]),
                     gaze=StubGaze(valid=True))
    for _ in range(2):
        pipeline.tick(road=frame(), driver=frame())
    assert pipeline.frames == 2
    assert pipeline.road_valid == 2
    assert pipeline.faces == 1 and pipeline.no_face == 1
    assert pipeline.gaze_valid == 1
    assert pipeline.face_rate == pytest.approx(0.5)


def test_reset_clears_every_counter():
    pipeline = build()
    pipeline.tick(road=frame(), driver=frame())
    pipeline.reset()
    assert pipeline.frames == 0 and pipeline.road_valid == 0
    assert pipeline.mean_inference_ms is None


def test_close_releases_all_three_models():
    road, face, gaze = StubRoadDetector(), StubFaceDetector(), StubGaze()
    with build(road, face, gaze):
        pass
    assert road.closed and face.closed and gaze.closed


def test_the_summary_names_both_paths():
    pipeline = build()
    pipeline.tick(road=frame(), driver=frame())
    line = pipeline.summary()
    assert "road" in line and "driver" in line and "testroadgaze" in line
