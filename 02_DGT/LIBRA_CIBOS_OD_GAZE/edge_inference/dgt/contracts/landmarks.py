"""Facial-landmark output contract: 106 points (InsightFace 2d106det) in pixels
of the frame named by `image_size`. Point order carries no meaning downstream."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .common import FrameMeta

import numpy as np


@dataclass(frozen=True)
class LandmarkResult:
    """One frame's facial landmarks.

    Attributes:
        points: (N, 2) float array of pixel coordinates, or None when invalid.
        valid: False means no face / too few points — never read `points`.
        confidence: Detector confidence in [0, 1], or None if not reported.
        sync: Frame id / timestamp / rate envelope.
        image_size: (width, height) of the frame `points` are expressed in.
        source: Which landmark model produced these, e.g.
            "insightface_2d106det". Documentation only — no code branches on it.
    """

    points: Optional[np.ndarray]
    valid: bool
    confidence: Optional[float] = None
    sync: Optional[FrameMeta] = None
    image_size: Optional[Tuple[int, int]] = None
    source: str = ""
