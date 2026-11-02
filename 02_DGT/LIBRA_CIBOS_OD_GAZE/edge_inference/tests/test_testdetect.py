"""The pipeline that drives the road-object detector.

No engine, no GPU, no video. The detector is injected, so the pipeline is driven
here by a stub that returns fixed contracts — a test stand-in, which belongs in
`tests/` and must never exist inside `dgt/`.

The tiling geometry, letterbox, decode and NMS are road-object-detection's
(`road_object_detection.inference`) and tested there; the detector's adapter
around them is `tests/test_road_objects.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

from dgt.contracts.detections import (
    BDD100K_CLASSES,
    BDD100K_TAXONOMY,
    Detection,
    DetectionResult,
)
from testdetect import ObjectDetectionTestPipeline, RoadFrameResult


def frame(h: int = 720, w: int = 1280) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@dataclass
class StubDetector:
    """Stand-in detector. Records calls, returns whatever it is told to."""

    result: DetectionResult = field(
        default_factory=lambda: DetectionResult(detections=[], taxonomy=BDD100K_TAXONOMY,
                                                valid=True))
    calls: List[np.ndarray] = field(default_factory=list)
    closed: bool = False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self.calls.append(frame)
        return self.result

    def close(self) -> None:
        self.closed = True


# ------------------------------------------------------------------ pipeline

def test_the_pipeline_requires_a_road_frame():
    """REQUIRED_FRAMES is the frame-ownership contract: the caller supplies road."""
    assert ObjectDetectionTestPipeline.REQUIRED_FRAMES == ("road",)


def test_tick_rejects_the_wrong_frame_name():
    pipeline = ObjectDetectionTestPipeline(StubDetector())
    with pytest.raises(ValueError, match="expects frames"):
        pipeline.tick(camera=frame())


def test_tick_passes_the_frame_to_the_detector_and_stamps_the_envelope():
    stub = StubDetector()
    pipeline = ObjectDetectionTestPipeline(stub, source_rate_hz=15.0)
    result = pipeline.tick(road=frame(), timestamp=1.0)

    assert isinstance(result, RoadFrameResult)
    assert len(stub.calls) == 1 and stub.calls[0].shape == (720, 1280, 3)
    assert result.sync.frame_id == 0
    assert result.sync.timestamp == 1.0
    assert result.sync.source_rate_hz == 15.0


def test_the_envelope_reaches_the_detection_contract_too():
    """The detector leaves sync unset; the pipeline stamps it, so a DetectionResult
    handed downstream carries the frame it came from."""
    pipeline = ObjectDetectionTestPipeline(StubDetector())
    result = pipeline.tick(road=frame(), timestamp=2.0)
    assert result.objects.sync is result.sync


def test_frame_ids_are_monotonic():
    pipeline = ObjectDetectionTestPipeline(StubDetector())
    ids = [pipeline.tick(road=frame()).sync.frame_id for _ in range(3)]
    assert ids == [0, 1, 2]


def test_detections_are_counted_and_reported():
    found = DetectionResult(
        detections=[Detection(label="car", class_id=2, score=0.9, box=(0.1, 0.1, 0.2, 0.2)),
                    Detection(label="pedestrian", class_id=0, score=0.6,
                              box=(0.5, 0.5, 0.6, 0.7))],
        taxonomy=BDD100K_TAXONOMY, valid=True)
    pipeline = ObjectDetectionTestPipeline(StubDetector(result=found))
    pipeline.tick(road=frame())
    pipeline.tick(road=frame())
    assert (pipeline.frames, pipeline.detections, pipeline.invalid) == (2, 4, 0)


def test_an_invalid_result_is_counted_as_invalid_not_as_empty():
    """valid=False is 'no inference ran'. Counting it as 'nothing ahead' is the
    exact failure a safety signal cannot have."""
    pipeline = ObjectDetectionTestPipeline(
        StubDetector(result=DetectionResult(detections=[], taxonomy="", valid=False)))
    pipeline.tick(road=frame())
    assert (pipeline.invalid, pipeline.detections) == (1, 0)


def test_mean_latency_is_none_before_the_first_frame():
    assert ObjectDetectionTestPipeline(StubDetector()).mean_inference_ms is None


def test_latency_is_measured_per_frame():
    pipeline = ObjectDetectionTestPipeline(StubDetector())
    result = pipeline.tick(road=frame())
    assert result.inference_ms >= 0.0
    assert pipeline.mean_inference_ms is not None


def test_reset_clears_the_counters():
    pipeline = ObjectDetectionTestPipeline(StubDetector())
    pipeline.tick(road=frame())
    pipeline.reset()
    assert (pipeline.frames, pipeline.detections, pipeline.invalid) == (0, 0, 0)
    assert pipeline.mean_inference_ms is None


def test_closing_the_pipeline_closes_the_detector():
    """The engine holds GPU memory; a context manager must release it."""
    stub = StubDetector()
    with ObjectDetectionTestPipeline(stub) as pipeline:
        pipeline.tick(road=frame())
    assert stub.closed


def test_run_drives_the_pipeline_as_a_generator():
    stub = StubDetector()
    pipeline = ObjectDetectionTestPipeline(stub)
    results = list(pipeline.run(({"road": frame()} for _ in range(3))))
    assert [r.sync.frame_id for r in results] == [0, 1, 2]


def test_the_contract_class_list_matches_the_published_run():
    """9 names in the order the run's onnx_class_names tag records. The model
    emits indices, so a reordering silently mislabels every detection."""
    assert BDD100K_CLASSES == (
        "pedestrian", "rider", "car", "truck", "bus",
        "motorcycle", "bicycle", "traffic light", "traffic sign",
    )
