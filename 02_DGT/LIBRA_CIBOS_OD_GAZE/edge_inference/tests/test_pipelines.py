"""The harness pipeline base (scripts/pipeline_base.py): frame ownership,
envelope stamping, lifecycle. The subclasses here are local to the test.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline_base import Pipeline, PipelineResult

FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


class _Echo(Pipeline[PipelineResult]):
    """A pipeline whose frame is supplied by the caller."""

    name = "echo"
    REQUIRED_FRAMES = ("a",)

    def _process(self, sync, *, a):
        return PipelineResult(sync=sync)


class _SelfSourced(Pipeline[PipelineResult]):
    """A pipeline whose model owns its own frame source (e.g. a camera)."""

    name = "self_sourced"
    REQUIRED_FRAMES = ()

    def __init__(self):
        super().__init__(10.0)
        self.closed = False

    def _process(self, sync):
        return PipelineResult(sync=sync)

    def close(self):
        self.closed = True


def test_frame_ids_are_monotonic_from_zero():
    p = _Echo(21.0)
    ids = [p.tick(a=FRAME).sync.frame_id for _ in range(3)]
    assert ids == [0, 1, 2]


def test_the_envelope_carries_the_source_rate():
    sync = _Echo(21.0).tick(a=FRAME, timestamp=1.5).sync
    assert sync.source_rate_hz == pytest.approx(21.0)
    assert sync.timestamp == pytest.approx(1.5)


@pytest.mark.parametrize(
    "kwargs", [{}, {"b": FRAME}, {"a": FRAME, "b": FRAME}, {"a": None}]
)
def test_wrong_frames_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _Echo(21.0).tick(**kwargs)


def test_run_requires_frames_unless_the_models_own_their_sources():
    with pytest.raises(ValueError):
        next(_Echo(21.0).run())
    assert len(list(_SelfSourced().run(max_ticks=3))) == 3


def test_run_iterates_supplied_frames():
    results = list(_Echo(21.0).run([{"a": FRAME}, {"a": FRAME}]))
    assert [r.sync.frame_id for r in results] == [0, 1]


def test_run_honours_max_ticks():
    results = list(_Echo(21.0).run([{"a": FRAME}] * 5, max_ticks=2))
    assert len(results) == 2


def test_context_manager_closes():
    with _SelfSourced() as p:
        p.tick()
    assert p.closed
