"""PERCLOS: fraction of closed-eye frames over a rolling window.

Returns None until the window is `min_fill_frac` full. Feed only frames with a
face; a no-face frame is not "eyes open".
"""
from __future__ import annotations

import math
from typing import Optional

from .windows import RollingWindow


class Perclos:
    def __init__(
        self,
        ear_threshold: float = 0.20,
        window_seconds: float = 30.0,
        fps: float = 21.0,
        min_fill_frac: float = 1.0,
    ) -> None:
        if not 0.0 < min_fill_frac <= 1.0:
            raise ValueError("min_fill_frac must be in (0, 1]")
        self.ear_threshold = ear_threshold
        self.min_fill_frac = min_fill_frac
        self._window = RollingWindow(window_seconds, fps)
        self._min_frames = max(1, math.ceil(min_fill_frac * self._window.maxlen))

    def update(self, ear: float) -> Optional[float]:
        """Ingest one frame's EAR; return PERCLOS in [0, 1], or None while warming up."""
        self._window.append(1 if ear < self.ear_threshold else 0)
        return self.value

    @property
    def value(self) -> Optional[float]:
        closed = self._window.values()
        if len(closed) < self._min_frames:
            return None
        return sum(closed) / len(closed)

    @property
    def fill_frac(self) -> float:
        """Warm-up progress in [0, 1]."""
        return min(1.0, len(self._window) / self._min_frames)

    @property
    def is_ready(self) -> bool:
        return len(self._window) >= self._min_frames

    @property
    def is_full(self) -> bool:
        return self._window.is_full

    def reset(self) -> None:
        self._window.reset()
