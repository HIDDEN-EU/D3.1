"""Shared contract primitives."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrameMeta:
    """Synchronisation envelope attached to every result.

    Attributes:
        frame_id: Monotonic frame counter; links results across streams.
        timestamp: Seconds on a monotonic clock (edge-local).
        source_rate_hz: Nominal producer rate, for staleness checks.
    """

    frame_id: int
    timestamp: float
    source_rate_hz: float
