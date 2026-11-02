"""Rolling FIFO window declared in seconds, sized by FPS."""
from __future__ import annotations

from collections import deque
from typing import List


class RollingWindow:
    def __init__(self, window_seconds: float, fps: float) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.window_seconds = window_seconds
        self.fps = fps
        self.maxlen = max(1, int(round(window_seconds * fps)))
        self._buf: deque = deque(maxlen=self.maxlen)

    def append(self, value) -> None:
        self._buf.append(value)

    def values(self) -> List:
        return list(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) == self.maxlen

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
