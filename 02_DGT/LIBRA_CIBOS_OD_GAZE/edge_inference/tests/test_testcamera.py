"""CameraTestPipeline — driven with synthetic frames, no camera."""
from __future__ import annotations

import numpy as np
import pytest

from testcamera import CameraFrameResult, CameraTestPipeline


def frame(h=48, w=64, c=3, value=128, dtype=np.uint8) -> np.ndarray:
    return np.full((h, w, c), value, dtype=dtype)


def test_it_reports_the_shape_it_was_given():
    result = CameraTestPipeline().tick(camera=frame(h=1080, w=1920))
    assert (result.width, result.height, result.channels) == (1920, 1080, 3)
    assert result.dtype == "uint8"


def test_a_grayscale_frame_is_one_channel():
    result = CameraTestPipeline().tick(camera=np.zeros((10, 20), dtype=np.uint8))
    assert (result.width, result.height, result.channels) == (20, 10, 1)


def test_mean_intensity_catches_a_black_stream():
    """Connected but black is the failure that looks healthy everywhere else."""
    assert CameraTestPipeline().tick(camera=frame(value=0)).mean_intensity == 0.0
    assert CameraTestPipeline().tick(camera=frame(value=200)).mean_intensity == pytest.approx(200.0)


def test_the_first_frame_has_no_rate_yet():
    result = CameraTestPipeline().tick(camera=frame(), timestamp=0.0)
    assert result.dt_s is None
    assert result.measured_fps is None


def test_it_measures_the_rate_actually_delivered_not_the_claimed_one():
    p = CameraTestPipeline(source_rate_hz=30.0, smoothing=1.0)  # no smoothing
    p.tick(camera=frame(), timestamp=0.0)
    result = p.tick(camera=frame(), timestamp=0.1)             # 10 fps in reality
    assert result.dt_s == pytest.approx(0.1)
    assert result.measured_fps == pytest.approx(10.0)
    assert p.source_rate_hz == pytest.approx(30.0)


def test_the_rate_is_smoothed_across_frames():
    p = CameraTestPipeline(smoothing=0.5)
    for i, t in enumerate([0.0, 0.1, 0.2, 0.3, 0.4]):
        result = p.tick(camera=frame(), timestamp=t)
    assert result.measured_fps == pytest.approx(10.0)


def test_a_duplicated_timestamp_does_not_divide_by_zero():
    p = CameraTestPipeline()
    p.tick(camera=frame(), timestamp=1.0)
    result = p.tick(camera=frame(), timestamp=1.0)
    assert result.dt_s == pytest.approx(0.0)
    assert result.measured_fps is None


def test_a_mid_stream_resolution_change_is_counted():
    """It silently invalidates anything holding pixel coordinates, so it is
    counted rather than ignored."""
    p = CameraTestPipeline()
    p.tick(camera=frame(h=480, w=640))
    p.tick(camera=frame(h=480, w=640))
    assert p.resolution_changes == 0
    p.tick(camera=frame(h=720, w=1280))
    assert p.resolution_changes == 1


def test_it_counts_frames_and_summarises():
    p = CameraTestPipeline(source_rate_hz=30.0, smoothing=1.0)
    for i in range(5):
        p.tick(camera=frame(), timestamp=i / 10.0)
    assert p.frames == 5
    summary = p.summary()
    assert "5 frames" in summary and "64x48x3" in summary and "~10.0 fps" in summary


def test_a_non_image_array_is_rejected():
    with pytest.raises(ValueError, match="2-D or 3-D"):
        CameraTestPipeline().tick(camera=np.zeros((10,)))


def test_it_honours_the_base_frame_contract():
    p = CameraTestPipeline()
    with pytest.raises(ValueError):
        p.tick()                          # camera= is required
    with pytest.raises(ValueError):
        p.tick(road=frame())              # wrong name
    assert isinstance(p.tick(camera=frame()), CameraFrameResult)


def test_run_drives_it_from_an_iterable():
    p = CameraTestPipeline()
    results = list(p.run([{"camera": frame()} for _ in range(3)]))
    assert [r.sync.frame_id for r in results] == [0, 1, 2]


def test_reset_clears_the_measurements():
    p = CameraTestPipeline()
    for i in range(3):
        p.tick(camera=frame(), timestamp=i / 10.0)
    p.reset()
    assert p.frames == 0 and p.resolution_changes == 0
    assert p.tick(camera=frame(), timestamp=0.0).measured_fps is None


@pytest.mark.parametrize("smoothing", [0.0, -0.1, 1.5])
def test_it_rejects_a_nonsense_smoothing_factor(smoothing):
    with pytest.raises(ValueError):
        CameraTestPipeline(smoothing=smoothing)
