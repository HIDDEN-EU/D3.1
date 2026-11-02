"""Road-object detection contract, for the forward road camera.

Labels are the model's own class names; `taxonomy` names the set. Boxes are
normalized [0, 1], origin top-left, the same convention as `GazeResult`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .common import FrameMeta

# The deployed model's classes in index order (road-object-detection
# configs/data.yaml, 9-class variant). A reference; the model is authoritative.
BDD100K_CLASSES: Tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)
BDD100K_TAXONOMY = "bdd100k-v2"


@dataclass(frozen=True)
class Detection:
    """One detected object.

    Attributes:
        label: Class name as the model reports it (e.g. "car").
        class_id: The model's integer class index.
        score: Detection confidence in [0, 1].
        box: (x1, y1, x2, y2), normalized [0, 1], top-left / bottom-right.
    """

    label: str
    class_id: int
    score: float
    box: Tuple[float, float, float, float]

    def contains(self, x: float, y: float) -> bool:
        """Is the normalized point (x, y) inside this box?"""
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass(frozen=True)
class DetectionResult:
    """One road frame's detections.

    Attributes:
        detections: Objects found this frame; empty with valid=True means none.
        taxonomy: Which label set `detections[*].label` belong to.
        valid: False means the detector did not run.
        sync: Frame id / timestamp / rate envelope.
    """

    detections: List[Detection] = field(default_factory=list)
    taxonomy: str = ""
    valid: bool = False
    sync: Optional[FrameMeta] = None
