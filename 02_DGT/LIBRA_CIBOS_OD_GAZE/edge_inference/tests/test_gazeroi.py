"""Gaze-directed ROI: the projection geometry and the pipeline that uses it.

Headless — the models are stubs, so this needs no engine, GPU or camera.
"""
from __future__ import annotations

import numpy as np
import pytest

from dgt.attention.gaze_roi import (
    GazeProjection,
    dilate_region,
    region_containing,
)
from dgt.contracts.detections import BDD100K_TAXONOMY, Detection, DetectionResult
from dgt.contracts.gaze import GazeResult
from testgazeroi import GazeRoiPipeline

W, H = 1280, 720
GRID = [(0, 0, 640, 360), (640, 0, 1280, 360), (0, 360, 640, 720), (640, 360, 1280, 720)]


class StubDetector:
    tile_size = (640, 360)

    def __init__(self):
        self.called_with = None
        self.calls = 0

    def regions_for(self, width, height):
        return list(GRID)

    def detect(self, frame, regions=None):
        self.called_with = regions
        self.calls += 1
        if regions is not None and not len(regions):
            return DetectionResult(detections=[], taxonomy=BDD100K_TAXONOMY, valid=False)
        return DetectionResult(
            detections=[Detection(label="car", class_id=0, score=0.9,
                                  box=(0.1, 0.1, 0.2, 0.2))],
            taxonomy=BDD100K_TAXONOMY, valid=True)


class StubFace:
    def detect_face_box(self, frame):
        return (1, 1, 9, 9)


class StubGaze:
    def __init__(self, raw=(0.8, 0.6), valid=False, point=(0.0, 0.0)):
        self.last_raw = raw
        self._valid = valid
        self._point = point

    def estimate(self, frame, box):
        return GazeResult(x=self._point[0], y=self._point[1],
                          confidence=None, valid=self._valid)


def _frames():
    return np.zeros((H, W, 3), np.uint8), np.zeros((480, 640, 3), np.uint8)


def _run(gaze=None, **kwargs):
    detector = StubDetector()
    pipeline = GazeRoiPipeline(detector, StubFace(), gaze or StubGaze(), **kwargs)
    road, driver = _frames()
    return detector, pipeline.tick_pair(road, driver, 1.0, 1.0)


# -- projection --------------------------------------------------------------

def test_identity_projection_passes_a_point_through():
    assert GazeProjection().apply((0.25, 0.75)) == (0.25, 0.75)


def test_a_point_outside_the_frame_is_none_not_an_edge():
    """A driver looking away is a real state; pinning it to the edge would
    invent a fixation that did not happen."""
    assert GazeProjection().apply((1.4, 0.5)) is None
    assert GazeProjection().apply((-0.1, 0.5)) is None


def test_clamp_is_opt_in():
    assert GazeProjection(clamp=True).apply((1.4, 0.5)) == (1.0, 0.5)


def test_scale_and_offset_apply():
    p = GazeProjection(scale_x=0.5, scale_y=2.0, offset_x=0.25, offset_y=-0.2)
    assert p.apply((1.0, 0.5)) == (0.75, 0.8)


@pytest.mark.parametrize("bad", [None, (1.0,), (float("nan"), 0.5), (0.5, float("inf"))])
def test_a_nonfinite_or_malformed_point_is_none(bad):
    assert GazeProjection().apply(bad) is None


# -- region selection --------------------------------------------------------

@pytest.mark.parametrize("point,expected", [
    ((0.1, 0.1), 0), ((0.9, 0.1), 1), ((0.1, 0.9), 2), ((0.9, 0.9), 3),
])
def test_each_quadrant_selects_its_tile(point, expected):
    assert region_containing(GRID, point[0], point[1], W, H) == expected


def test_the_far_corner_stays_inside_the_last_tile():
    """Exactly 1.0 must not fall outside every region."""
    assert region_containing(GRID, 1.0, 1.0, W, H) == 3


def test_no_regions_is_none_not_a_crash():
    assert region_containing([], 0.5, 0.5, W, H) is None


def test_dilate_grows_and_clips_to_the_frame():
    assert dilate_region((640, 360, 1280, 720), 0.25, W, H) == (480, 270, 1280, 720)
    assert dilate_region((0, 0, 640, 360), 0.0, W, H) == (0, 0, 640, 360)


# -- the pipeline ------------------------------------------------------------

def test_only_the_looked_at_tile_is_inferred():
    """The whole point: one region in, not the full grid."""
    detector, result = _run(StubGaze(raw=(0.9, 0.9)))
    assert result.roi_index == 3
    assert len(detector.called_with) == 1


def test_the_chosen_region_is_the_dilated_tile():
    detector, result = _run(StubGaze(raw=(0.9, 0.9)), roi_margin=0.25)
    assert detector.called_with == ((480, 270, 1280, 720),)


def test_a_different_gaze_point_picks_a_different_tile():
    _, right = _run(StubGaze(raw=(0.9, 0.1)))
    _, left = _run(StubGaze(raw=(0.1, 0.1)))
    assert (right.roi_index, left.roi_index) == (1, 0)


def test_the_full_grid_is_reported_so_a_viewer_can_draw_it():
    _, result = _run()
    assert tuple(result.regions) == tuple(GRID)


def test_no_gaze_point_skips_rather_than_sweeping_the_frame():
    """'I don't know where they looked' must not become a full-frame sweep."""
    detector, result = _run(StubGaze(raw=None), on_no_gaze="skip")
    assert result.roi_index is None
    assert detector.called_with == ()
    assert result.objects.valid is False


def test_the_full_fallback_is_available_but_says_so():
    detector, result = _run(StubGaze(raw=None), on_no_gaze="full")
    assert result.roi_index is None
    assert tuple(detector.called_with) == tuple(GRID)
    assert result.objects.valid is True


def test_a_gaze_point_off_the_frame_is_not_a_tile():
    _, result = _run(StubGaze(raw=(1.5, 0.5)))
    assert result.roi_index is None and result.roi_point is None


def test_an_invalid_on_no_gaze_is_rejected_at_construction():
    with pytest.raises(ValueError, match="on_no_gaze"):
        GazeRoiPipeline(StubDetector(), StubFace(), StubGaze(), on_no_gaze="whatever")


# -- validity: the rule that matters -----------------------------------------

def test_roi_is_invalid_while_gaze_is_uncalibrated():
    _, result = _run(StubGaze(raw=(0.9, 0.9), valid=False))
    assert result.roi_index == 3 and result.roi_valid is False


def test_roi_is_invalid_even_with_valid_gaze_if_the_projection_is_untrusted():
    """A measured gaze through a placeholder mapping is still a guess."""
    _, result = _run(StubGaze(raw=None, valid=True, point=(0.9, 0.9)))
    assert result.roi_index == 3 and result.roi_valid is False


def test_roi_is_invalid_with_a_trusted_projection_but_invalid_gaze():
    _, result = _run(StubGaze(raw=(0.9, 0.9), valid=False), trust_projection=True)
    assert result.roi_valid is False


def test_roi_is_valid_only_when_gaze_and_projection_are_both_real():
    _, result = _run(StubGaze(raw=None, valid=True, point=(0.9, 0.9)),
                     trust_projection=True)
    assert result.roi_index == 3 and result.roi_valid is True


def test_skew_is_recorded_from_the_two_capture_times():
    detector = StubDetector()
    pipeline = GazeRoiPipeline(detector, StubFace(), StubGaze())
    road, driver = _frames()
    result = pipeline.tick_pair(road, driver, 1.000, 1.012)
    assert result.frame_skew_ms == pytest.approx(12.0)


def test_tick_without_a_pair_reports_no_skew():
    detector = StubDetector()
    pipeline = GazeRoiPipeline(detector, StubFace(), StubGaze())
    road, driver = _frames()
    assert pipeline.tick(road=road, driver=driver, timestamp=1.0).frame_skew_ms is None


def test_both_frames_are_required():
    pipeline = GazeRoiPipeline(StubDetector(), StubFace(), StubGaze())
    road, _ = _frames()
    with pytest.raises(ValueError):
        pipeline.tick(road=road, timestamp=1.0)


def test_the_frame_counter_advances_like_testroadgaze():
    """The scripts drive both pipelines with one loop shape."""
    pipeline = GazeRoiPipeline(StubDetector(), StubFace(), StubGaze())
    road, driver = _frames()
    assert pipeline.frames == 0
    for expected in (1, 2, 3):
        pipeline.tick_pair(road, driver, 1.0, 1.0)
        assert pipeline.frames == expected


def test_the_inferred_region_is_recorded_for_the_viewer():
    """A viewer must be able to show the pixels the detector actually saw,
    which is the dilated tile, not regions[roi_index]."""
    detector, result = _run(StubGaze(raw=(0.9, 0.9)), roi_margin=0.25)
    assert result.roi_region == (480, 270, 1280, 720)
    assert detector.called_with == (result.roi_region,)
    assert result.roi_region != GRID[result.roi_index]


def test_no_roi_region_when_there_is_no_tile():
    _, result = _run(StubGaze(raw=None), on_no_gaze="full")
    assert result.roi_region is None
