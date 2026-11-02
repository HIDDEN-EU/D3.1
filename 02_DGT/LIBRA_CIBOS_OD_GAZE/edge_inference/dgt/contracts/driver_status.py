"""DGT_DS, the message dgt sends (reference: docs/DGT_DS.md).

A block that is not measured is None, serialized as JSON null. Coordinates are
pixels in the named camera frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: `EmotionStatus.class_id` indexes this.
EMOTION_CLASSES: Tuple[str, ...] = (
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise",
)


@dataclass(frozen=True)
class MessageMetadata:
    """Attributes:
        send_timestamp: Unix seconds at send.
    """

    schema_version: str
    message_type: str
    message_id: str
    sender_id: str
    sequence_number: int
    send_timestamp: float
    time_reference: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "message_type": self.message_type,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "sequence_number": self.sequence_number,
            "send_timestamp": self.send_timestamp,
            "time_reference": self.time_reference,
        }


@dataclass(frozen=True)
class GazeArea:
    """The region of the road-camera plane the driver attends to (FUN-DGT-01)."""

    center: Tuple[int, int]
    radius_px: int

    def to_dict(self) -> Dict[str, Any]:
        return {"center": list(self.center), "radius_px": self.radius_px}


@dataclass(frozen=True)
class AttendedObject:
    """One road-camera detection; class_id / class_name are the model's own."""

    class_id: int
    class_name: str
    bbox_px: Tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox_px": list(self.bbox_px),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DriverAttOD:
    """Gaze area + detections in road-camera pixels.

    Attributes:
        gaze_area_px: None when not measured.
        detections: empty means none detected.
    """

    frame_id: str
    gaze_area_px: Optional[GazeArea]
    detections: List[AttendedObject] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "gaze_area_px": _opt(self.gaze_area_px),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass(frozen=True)
class EmotionStatus:
    """Categorical emotion and its full softmax."""

    frame_id: str
    label: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "label": self.label,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "probabilities": {name: self.probabilities[name] for name in EMOTION_CLASSES},
        }


@dataclass(frozen=True)
class LevelOfAttention:
    """The CTAG_ATTEN attentiveness score."""

    frame_id: str
    score: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"frame_id": self.frame_id, "score": self.score,
                "confidence": self.confidence}


@dataclass(frozen=True)
class DriverStatus:
    """The assessment of one frame.

    Attributes:
        timestamp: Unix seconds at capture of the assessed frame.
    """

    timestamp: float
    driver_att_od: Optional[DriverAttOD] = None
    emotion_status: Optional[EmotionStatus] = None
    level_of_attention: Optional[LevelOfAttention] = None
    distracted: Optional[bool] = None
    drowsy: Optional[bool] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "driver_att_od": _opt(self.driver_att_od),
            "emotion_status": _opt(self.emotion_status),
            "level_of_attention": _opt(self.level_of_attention),
            "distracted": self.distracted,
            "drowsy": self.drowsy,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DgtMessage:
    """One DGT_DS message: envelope + assessment."""

    metadata: MessageMetadata
    driver_status: DriverStatus

    def to_dict(self) -> Dict[str, Any]:
        return {"metadata": self.metadata.to_dict(),
                "driver_status": self.driver_status.to_dict()}


def _opt(block) -> Optional[Dict[str, Any]]:
    return None if block is None else block.to_dict()
