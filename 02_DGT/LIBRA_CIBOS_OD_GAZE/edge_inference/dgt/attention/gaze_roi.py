"""Where a gaze point lands on the road frame, and which region to infer there.

`region_containing` picks the fixed grid tile that holds a gaze point.

`GazeProjection` is a PROVISIONAL mapping read from config, not a measurement,
so anything built on it stays `valid=False`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# Pixel (x0, y0, x1, y1), x1/y1 exclusive.
Region = Tuple[int, int, int, int]


@dataclass(frozen=True)
class GazeProjection:
    """Affine map from gaze coordinates to road-frame normalized [0, 1]. PROVISIONAL."""

    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    clamp: bool = False

    def apply(self, point: Sequence[float]) -> Optional[Tuple[float, float]]:
        """Gaze (x, y) -> normalized frame (x, y), or None when it misses the frame."""
        if point is None or len(point) != 2:
            return None
        gx, gy = float(point[0]), float(point[1])
        if not (np.isfinite(gx) and np.isfinite(gy)):
            return None

        x = gx * self.scale_x + self.offset_x
        y = gy * self.scale_y + self.offset_y

        if self.clamp:
            return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x, y
        return None


def region_containing(regions: Sequence[Region], x: float, y: float,
                      width: int, height: int) -> Optional[int]:
    """Index of the region holding normalized (x, y), or None.

    Upper edges are inclusive only at the frame boundary, so 1.0 lands in the last tile.
    """
    if not regions or x is None or y is None:
        return None
    px, py = x * width, y * height
    for index, (x0, y0, x1, y1) in enumerate(regions):
        right = px < x1 or (x1 >= width and px <= x1)
        below = py < y1 or (y1 >= height and py <= y1)
        if x0 <= px and y0 <= py and right and below:
            return index
    return None


def dilate_region(region: Region, margin: float,
                  width: int, height: int) -> Region:
    """Grow a region by `margin` of its own size, clipped to the frame."""
    x0, y0, x1, y1 = region
    if margin <= 0:
        return region
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (
        int(max(0, round(x0 - dx))),
        int(max(0, round(y0 - dy))),
        int(min(width, round(x1 + dx))),
        int(min(height, round(y1 + dy))),
    )
