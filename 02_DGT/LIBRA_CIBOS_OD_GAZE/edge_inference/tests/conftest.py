"""Shared fixtures. Everything here is headless: no camera, GPU, TensorRT or network."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dgt.contracts import FrameMeta

# The harness pipelines live in scripts/ (they left the package 2026-09-25), so
# the tests import them the way the scripts import each other: by module name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def eye_contour(cx: float, cy: float, width: float, opening: float) -> np.ndarray:
    """A 10-point eye contour with a known width and lid opening.

    Two canthi at +/- width/2 on the x axis, four upper-lid points at
    +opening/2 and four lower-lid points at -opening/2, so
    `eye_openness_ratio` must return opening / width.
    """
    xs = np.linspace(-width / 2 * 0.6, width / 2 * 0.6, 4)
    upper = np.stack([xs, np.full(4, opening / 2.0)], axis=1)
    lower = np.stack([xs, np.full(4, -opening / 2.0)], axis=1)
    canthi = np.array([[-width / 2.0, 0.0], [width / 2.0, 0.0]])
    return np.concatenate([canthi, upper, lower]) + np.array([cx, cy])


def landmarks_106(opening: float = 12.0, width: float = 40.0) -> np.ndarray:
    """A full (106, 2) 2d106det-shaped array whose EAR is `opening / width`.

    Only indices 33..42 and 87..96 are read by the EAR front-end; the rest are
    filled with a constant so an accidental read of them is obvious.
    """
    points = np.full((106, 2), 1000.0)
    points[33:43] = eye_contour(200.0, 300.0, width, opening)
    points[87:97] = eye_contour(400.0, 300.0, width, opening)
    return points


def rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a point set about its centroid (to test roll invariance)."""
    theta = math.radians(degrees)
    r = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    centre = points.mean(axis=0)
    return (points - centre) @ r.T + centre


def meta(frame_id: int = 0, timestamp: float = 0.0, rate: float = 30.0) -> FrameMeta:
    return FrameMeta(frame_id=frame_id, timestamp=timestamp, source_rate_hz=rate)


@pytest.fixture
def open_eyes() -> np.ndarray:
    """Landmarks with EAR = 0.30 — comfortably above the 0.20 threshold."""
    return landmarks_106(opening=12.0, width=40.0)


@pytest.fixture
def closed_eyes() -> np.ndarray:
    """Landmarks with EAR = 0.05 — comfortably below the 0.20 threshold."""
    return landmarks_106(opening=2.0, width=40.0)
