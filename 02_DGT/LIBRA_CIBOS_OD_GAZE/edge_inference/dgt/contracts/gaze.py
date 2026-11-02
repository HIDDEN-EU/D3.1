"""Gaze output contract: a 2D gaze point, normalized [0, 1], origin top-left
(x -> right, y -> down)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .common import FrameMeta


@dataclass(frozen=True)
class GazeResult:
    """One frame's gaze estimate.

    Attributes:
        x: Gaze point x, normalized [0, 1], or None when `valid` is False.
        y: Gaze point y, normalized [0, 1], or None when `valid` is False.
        confidence: Estimator confidence in [0, 1], or None.
        valid: False means no estimate — do not read x/y.
        sync: Frame id / timestamp / rate envelope.
    """

    x: Optional[float]
    y: Optional[float]
    confidence: Optional[float]
    valid: bool
    sync: Optional[FrameMeta] = None
