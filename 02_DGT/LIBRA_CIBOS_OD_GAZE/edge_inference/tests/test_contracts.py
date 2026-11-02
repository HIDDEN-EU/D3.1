"""The contracts are the boundary: they must be frozen, explicit and dependency-free."""
from __future__ import annotations

import dataclasses

import pytest

from dgt.contracts import (
    BDD100K_CLASSES,
    Detection,
    DetectionResult,
    FrameMeta,
    GazeResult,
    LandmarkResult,
)


def test_results_are_frozen():
    gaze = GazeResult(x=0.5, y=0.5, confidence=None, valid=True, sync=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gaze.x = 0.1


def test_no_answer_is_explicit_not_zero_filled():
    # A result defaults to invalid, so a partially-built one can never be
    # mistaken for a measurement.
    assert DetectionResult().valid is False
    assert LandmarkResult(points=None, valid=False).points is None


def test_detection_result_distinguishes_empty_scene_from_no_inference():
    empty_scene = DetectionResult(detections=[], valid=True)
    not_run = DetectionResult(detections=[], valid=False)
    assert empty_scene.valid and not not_run.valid


def test_taxonomy_matches_the_trained_model():
    # road-object-detection/configs/data.yaml: the 9-class variant
    assert len(BDD100K_CLASSES) == 9
    assert BDD100K_CLASSES[2] == "car"


def test_detection_contains_is_a_normalized_containment_test():
    d = Detection(label="car", class_id=2, score=0.9, box=(0.2, 0.2, 0.6, 0.6))
    assert d.contains(0.4, 0.4)
    assert d.contains(0.2, 0.2)  # inclusive on the edge
    assert not d.contains(0.7, 0.4)


def test_contracts_import_nothing_from_the_rest_of_dgt():
    import dgt.contracts.detections as detections
    import dgt.contracts.gaze as gaze
    import dgt.contracts.landmarks as landmarks

    for module in (detections, gaze, landmarks):
        source = open(module.__file__).read()
        assert "from .." not in source, f"{module.__name__} reaches outside contracts/"
        assert "import dgt" not in source, f"{module.__name__} reaches outside contracts/"


def test_frame_meta_is_the_sync_envelope():
    m = FrameMeta(frame_id=7, timestamp=1.5, source_rate_hz=21.0)
    assert (m.frame_id, m.timestamp, m.source_rate_hz) == (7, 1.5, 21.0)
