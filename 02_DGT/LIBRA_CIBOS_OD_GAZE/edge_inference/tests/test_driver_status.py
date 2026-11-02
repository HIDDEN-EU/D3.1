"""The DGT_DS contract: its key layout IS the reference in docs/DGT_DS.md."""
from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from dgt.contracts.driver_status import (
    EMOTION_CLASSES,
    AttendedObject,
    DgtMessage,
    DriverAttOD,
    DriverStatus,
    EmotionStatus,
    GazeArea,
    LevelOfAttention,
    MessageMetadata,
)

DOC = Path(__file__).resolve().parents[1] / "docs" / "DGT_DS.md"


def reference() -> dict:
    """The JSON block in docs/DGT_DS.md."""
    match = re.search(r"```json\n(.*?)\n```", DOC.read_text(), re.S)
    assert match, "docs/DGT_DS.md has no ```json reference block"
    return json.loads(match.group(1))


def keys(tree):
    """The key layout of a JSON tree, with `_comment` (documentation) dropped,
    a list of objects reduced to the layout of its first element, and any other
    value — a scalar or an array of numbers — a leaf. (The reference describes
    each leaf as a string such as "array[int] ...".)"""
    if isinstance(tree, dict):
        return {k: keys(v) for k, v in tree.items() if k != "_comment"}
    if isinstance(tree, list) and tree and isinstance(tree[0], dict):
        return [keys(tree[0])]
    return None


META = MessageMetadata("0.1.0", "DGT_DS", "id", "sender", 0, 1.0, "UTC")


def full_message() -> DgtMessage:
    return DgtMessage(
        metadata=META,
        driver_status=DriverStatus(
            timestamp=1.0,
            driver_att_od=DriverAttOD(
                frame_id="road_camera",
                gaze_area_px=GazeArea(center=(10, 20), radius_px=5),
                detections=[AttendedObject(2, "car", (1, 2, 3, 4), 0.9)],
            ),
            emotion_status=EmotionStatus(
                frame_id="driver_camera", label="neutral", class_id=4,
                confidence=0.7,
                probabilities={name: 0.05 for name in EMOTION_CLASSES} | {"neutral": 0.7},
            ),
            level_of_attention=LevelOfAttention("driver_camera", 80, 0.5),
            distracted=False, drowsy=False, confidence=0.5,
        ),
    )


def test_a_fully_populated_message_has_exactly_the_reference_layout():
    assert keys(full_message().to_dict()) == keys(reference())


def test_the_emotion_classes_match_the_reference():
    probabilities = reference()["driver_status"]["emotion_status"]["probabilities"]
    assert tuple(k for k in probabilities if k != "_comment") == EMOTION_CLASSES


def test_what_is_not_implemented_is_null_not_a_default():
    status = DriverStatus(timestamp=1.0).to_dict()
    for key in ("driver_att_od", "emotion_status", "level_of_attention",
                "distracted", "drowsy", "confidence"):
        assert key in status and status[key] is None, key
    # And it survives serialization as JSON null.
    assert json.loads(json.dumps(status))["emotion_status"] is None


def test_an_absent_gaze_area_is_null_and_empty_detections_stay_a_list():
    block = DriverAttOD(frame_id="road_camera", gaze_area_px=None).to_dict()
    assert block["gaze_area_px"] is None
    assert block["detections"] == []


def test_the_contract_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        META.sequence_number = 1  # type: ignore[misc]
