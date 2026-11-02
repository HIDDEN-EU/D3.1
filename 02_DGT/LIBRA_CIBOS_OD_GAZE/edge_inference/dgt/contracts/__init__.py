"""Frozen output shapes. Stdlib (and NumPy for landmarks) only; nothing else
from `dgt`. Every result carries `valid`: False means no answer."""
from .common import FrameMeta
from .detections import BDD100K_CLASSES, BDD100K_TAXONOMY, Detection, DetectionResult
from .driver_status import (
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
from .gaze import GazeResult
from .landmarks import LandmarkResult

__all__ = [
    "FrameMeta",
    "GazeResult",
    "LandmarkResult",
    "Detection",
    "DetectionResult",
    "BDD100K_CLASSES",
    "BDD100K_TAXONOMY",
    "DgtMessage",
    "MessageMetadata",
    "DriverStatus",
    "DriverAttOD",
    "GazeArea",
    "AttendedObject",
    "EmotionStatus",
    "LevelOfAttention",
    "EMOTION_CLASSES",
]
