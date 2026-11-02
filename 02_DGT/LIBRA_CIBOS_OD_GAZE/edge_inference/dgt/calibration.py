"""Gaze calibration: a homography from the camera gaze plane of the model to a target plane.

The procedure of D3.1 Section 3.6.1.2: nine targets in a 3x3 layout are shown in turn; for
each target the raw gaze outputs collected while the driver fixates it are reduced to their
component-wise median; a homography is fitted by least squares over the nine
correspondences; and every later frame is mapped through it.

Target positions are in normalised target-plane coordinates, [0, 1] with origin top-left,
so a mapped point already is a `GazeResult` x/y (use `screen_grid = (1, 1)`).

A calibration belongs to one physical installation: fit it on the unit it runs on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

#: The nine targets, row by row, inset from the plane edges.
DEFAULT_TARGETS: List[Point] = [
    (x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)
]


def fit_homography(raw: Sequence[Point], targets: Sequence[Point]) -> np.ndarray:
    """Least-squares homography (all points, no RANSAC) with `targets ~ H @ raw`."""
    import cv2

    src = np.asarray(raw, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(targets, dtype=np.float64).reshape(-1, 1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError(f"need >= 4 matching points, got {len(src)} raw and {len(dst)} targets")
    matrix, _ = cv2.findHomography(src, dst, 0)
    if matrix is None:
        raise ValueError("homography fit failed: the raw points are degenerate (collinear?)")
    return matrix


class Calibration:
    """A fitted homography; `available` and `apply(x, y)` are what `GazeEstimator` reads."""

    def __init__(self, matrix: Optional[np.ndarray] = None) -> None:
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)

    @property
    def available(self) -> bool:
        return self.matrix is not None

    def apply(self, x: float, y: float) -> Optional[Point]:
        """Map one raw gaze point: p = (H @ [x, y, 1]) / w."""
        if self.matrix is None:
            return None
        u, v, w = self.matrix @ np.array([x, y, 1.0])
        if abs(w) < 1e-12:
            return None
        return float(u / w), float(v / w)

    def save(self, path: Path) -> None:
        if self.matrix is None:
            raise ValueError("nothing to save: the calibration is not fitted")
        Path(path).write_text(json.dumps({"homography": self.matrix.tolist()}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(np.array(data["homography"], dtype=np.float64))


class CalibrationSession:
    """Collect raw gaze samples per target, then fit.

        session = CalibrationSession()
        for j, target in enumerate(session.targets):
            # show `target`, let the driver fixate it, then for each frame:
            session.add(j, gaze.last_raw)
        calibration = session.fit()
    """

    def __init__(self, targets: Iterable[Point] = DEFAULT_TARGETS) -> None:
        self.targets: List[Point] = [tuple(map(float, t)) for t in targets]
        self._samples: Dict[int, List[Point]] = {j: [] for j in range(len(self.targets))}

    def add(self, target_index: int, raw: Optional[Point]) -> None:
        """Record one frame's raw gaze output for a target; `None` (no face) is skipped."""
        if raw is not None:
            self._samples[target_index].append((float(raw[0]), float(raw[1])))

    def observations(self) -> List[Optional[Point]]:
        """Component-wise median of each target's samples (None for a target without any)."""
        out: List[Optional[Point]] = []
        for j in range(len(self.targets)):
            samples = self._samples[j]
            out.append(tuple(np.median(np.asarray(samples), axis=0)) if samples else None)
        return out

    def fit(self) -> Calibration:
        pairs = [(obs, t) for obs, t in zip(self.observations(), self.targets) if obs is not None]
        if len(pairs) < 4:
            raise ValueError(f"only {len(pairs)} targets have samples; need >= 4")
        raw, targets = zip(*pairs)
        return Calibration(fit_homography(raw, targets))
