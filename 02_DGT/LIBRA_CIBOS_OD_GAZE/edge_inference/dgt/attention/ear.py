"""Eye Aspect Ratio from InsightFace 2d106det landmarks.

The corners are the two farthest-apart eye points and the lid gap is measured
perpendicular to that axis, so the result ignores point order and head roll.
Its scale differs from the classic 6-point EAR: calibrate `ear_threshold` for it.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

EYE1_IDX = list(range(33, 43))
EYE2_IDX = list(range(87, 97))

MIN_LANDMARKS = max(EYE2_IDX) + 1

_EPS = 1e-6


def eye_openness_ratio(eye_points: np.ndarray) -> float:
    """Opening / width for one eye's (N, 2) points; 0.0 if degenerate."""
    pts = np.asarray(eye_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return 0.0

    diffs = pts[:, None, :] - pts[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diffs, diffs)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    p0, p1 = pts[i], pts[j]

    axis = p1 - p0
    width = float(np.hypot(axis[0], axis[1]))
    if width < _EPS:
        return 0.0

    normal = np.array([-axis[1], axis[0]]) / width
    perp = (pts - p0) @ normal

    # Exclude the canthi by index: their perp is ~0 and rounding would sort them into a lid.
    lid = np.ones(pts.shape[0], dtype=bool)
    lid[[i, j]] = False
    lid_perp = perp[lid]
    upper = lid_perp[lid_perp > 0]
    lower = lid_perp[lid_perp < 0]
    top = float(upper.mean()) if upper.size else 0.0
    bottom = float(lower.mean()) if lower.size else 0.0
    opening = top - bottom
    return opening / (width + _EPS)


def ear_from_landmarks(landmarks: np.ndarray) -> Optional[float]:
    """Mean EAR over both eyes, or None (no measurement) when landmarks are missing."""
    if landmarks is None:
        return None
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < MIN_LANDMARKS:
        return None
    left = eye_openness_ratio(pts[EYE1_IDX])
    right = eye_openness_ratio(pts[EYE2_IDX])
    return float((left + right) / 2.0)
