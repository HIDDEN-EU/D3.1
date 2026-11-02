"""Pipeline base for the scripts/ harnesses; `tick()` advances exactly one frame."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Generic, Iterable, Iterator, Mapping, Optional, Tuple, TypeVar

import numpy as np

from dgt.contracts.common import FrameMeta


@dataclass(frozen=True)
class PipelineResult:
    """Base for a pipeline's per-frame output."""

    sync: FrameMeta


R = TypeVar("R", bound=PipelineResult)


class Pipeline(ABC, Generic[R]):
    name: ClassVar[str] = "pipeline"

    #: Frames the caller must pass; models that read their own camera are omitted.
    REQUIRED_FRAMES: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, source_rate_hz: float) -> None:
        self.source_rate_hz = source_rate_hz
        self._frame_id = -1

    def tick(self, *, timestamp: Optional[float] = None, **frames: np.ndarray) -> R:
        """Validate frames, stamp `FrameMeta`, and run one frame."""
        expected = set(self.REQUIRED_FRAMES)
        given = set(frames)
        if given != expected:
            missing = sorted(expected - given)
            unexpected = sorted(given - expected)
            raise ValueError(
                f"{self.name}.tick() expects frames {sorted(expected)}; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for key, frame in frames.items():
            if frame is None:
                raise ValueError(f"{self.name}.tick(): frame {key!r} is None")

        self._frame_id += 1
        sync = FrameMeta(
            frame_id=self._frame_id,
            timestamp=time.monotonic() if timestamp is None else timestamp,
            source_rate_hz=self.source_rate_hz,
        )
        return self._process(sync, **frames)

    @abstractmethod
    def _process(self, sync: FrameMeta, **frames: np.ndarray) -> R:
        """Run one frame with the envelope already stamped."""

    def run(
        self,
        frames: Optional[Iterable[Mapping[str, np.ndarray]]] = None,
        max_ticks: Optional[int] = None,
    ) -> Iterator[R]:
        """Yield results; `frames=None` is allowed only when `REQUIRED_FRAMES` is empty."""
        if frames is None:
            if self.REQUIRED_FRAMES:
                raise ValueError(
                    f"{self.name}.run() needs frames: this pipeline requires "
                    f"{sorted(self.REQUIRED_FRAMES)}"
                )
            count = 0
            while max_ticks is None or count < max_ticks:
                yield self.tick()
                count += 1
            return

        for count, batch in enumerate(frames):
            if max_ticks is not None and count >= max_ticks:
                return
            yield self.tick(**dict(batch))

    def reset(self) -> None:
        """Clear temporal state between sessions."""

    def close(self) -> None:
        """Release models and devices."""

    def __enter__(self) -> "Pipeline[R]":
        return self

    def __exit__(self, *exc_info) -> None:
        # A leaked CSI handle leaves nvarguscamerasrc unusable until device reset.
        self.close()
