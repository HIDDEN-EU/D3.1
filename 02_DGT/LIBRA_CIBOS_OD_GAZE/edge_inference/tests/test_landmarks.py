"""The landmark path: SCRFD/2d106det geometry, and the pipeline that drives it.

No engine, no GPU, no camera. The detector is injected, so the pipeline runs
here against a stub that returns fixed contracts — a test stand-in, which
belongs in `tests/` and must never exist inside `dgt/`.

The geometry tests carry the weight. Every failure they catch still produces 106
points that look like a face — a half-padding offset, an anchor grid off by a
stride, an inverse transform that forgot the crop centre — and EAR averages
whatever it is given. A picture cannot show that; arithmetic can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest
from conftest import landmarks_106

from dgt.attention.ear import ear_from_landmarks
from dgt.contracts.landmarks import LandmarkResult
from dgt.models.landmarks import (
    CROP_PAD,
    FPN_STRIDES,
    LANDMARK_SOURCE,
    anchor_centers,
    decode_landmarks,
    distance2bbox,
    face_crop_matrix,
    nms,
    points_to_sensor,
    scrfd_input_geometry,
)
from testperclos import DriverFrameResult, PerclosTestPipeline


def frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@dataclass
class StubLandmarks:
    """Stand-in landmark model. Returns a queue of contracts, records its calls."""

    results: List[LandmarkResult] = field(default_factory=list)
    calls: List[np.ndarray] = field(default_factory=list)
    closed: bool = False

    def detect(self, frame: np.ndarray) -> LandmarkResult:
        self.calls.append(frame)
        if not self.results:
            return LandmarkResult(points=None, valid=False, source=LANDMARK_SOURCE)
        return self.results[min(len(self.calls), len(self.results)) - 1]

    def close(self) -> None:
        self.closed = True


def valid_result(points: Optional[np.ndarray] = None) -> LandmarkResult:
    pts = landmarks_106() if points is None else points
    return LandmarkResult(points=pts, valid=True, confidence=0.9,
                          image_size=(640, 480), source=LANDMARK_SOURCE)


def invalid_result() -> LandmarkResult:
    return LandmarkResult(points=None, valid=False, image_size=(640, 480),
                          source=LANDMARK_SOURCE)


# ------------------------------------------------------- SCRFD input geometry

@pytest.mark.parametrize("w,h", [(640, 480), (1280, 720), (3280, 2464), (480, 640)])
def test_the_resized_frame_fits_inside_the_square_input(w, h):
    new_w, new_h, _ = scrfd_input_geometry(h, w, 640)
    assert new_w <= 640 and new_h <= 640
    assert new_w == 640 or new_h == 640      # one side touches; no wasted input


@pytest.mark.parametrize("w,h", [(640, 480), (1280, 720), (3280, 2464), (480, 640)])
def test_the_resize_preserves_aspect_ratio(w, h):
    new_w, new_h, _ = scrfd_input_geometry(h, w, 640)
    assert new_w / new_h == pytest.approx(w / h, rel=0.01)


def test_det_scale_takes_a_box_back_to_frame_pixels():
    """The whole decode hangs off this: boxes come out in INPUT pixels and are
    divided by det_scale to land on the frame."""
    new_w, new_h, det_scale = scrfd_input_geometry(720, 1280, 640)
    # A box covering the resized frame must cover the original frame again.
    assert (new_w / det_scale, new_h / det_scale) == pytest.approx((1280, 720), rel=0.01)


def test_a_square_frame_uses_the_whole_input():
    assert scrfd_input_geometry(640, 640, 640) == (640, 640, 1.0)


# --------------------------------------------------------------- anchor grid

def test_anchor_centres_step_by_the_stride_from_the_origin():
    """Not from the centre of the cell: SCRFD's anchors sit on the grid points,
    and a half-stride offset moves every box by four pixels at stride 8."""
    centers = anchor_centers(2, 3, 8, num_anchors=1)
    assert centers.tolist() == [[0, 0], [8, 0], [16, 0], [0, 8], [8, 8], [16, 8]]


def test_the_anchors_of_one_location_are_adjacent():
    """The heads flatten as (cell, anchor), so the duplicates must interleave —
    stacking them as (anchor, cell) would shuffle every prediction."""
    centers = anchor_centers(1, 2, 8, num_anchors=2)
    assert centers.tolist() == [[0, 0], [0, 0], [8, 0], [8, 0]]


@pytest.mark.parametrize("stride", FPN_STRIDES)
def test_the_grid_matches_the_640_input_the_published_onnx_declares(stride):
    """12800 / 3200 / 800 rows at strides 8 / 16 / 32 — the shapes in det_500m."""
    expected = {8: 12800, 16: 3200, 32: 800}[stride]
    grid = 640 // stride
    assert len(anchor_centers(grid, grid, stride, num_anchors=2)) == expected


# ------------------------------------------------------------- box decoding

def test_distance2bbox_reads_distances_outward_from_the_centre():
    points = np.array([[100.0, 100.0]])
    distance = np.array([[10.0, 20.0, 30.0, 40.0]])     # left, top, right, bottom
    assert distance2bbox(points, distance).tolist() == [[90.0, 80.0, 130.0, 140.0]]


def test_nms_keeps_the_best_box_and_drops_its_overlaps():
    boxes = np.array([[0.0, 0.0, 100.0, 100.0],
                      [5.0, 5.0, 105.0, 105.0],        # ~0.8 IoU with the first
                      [500.0, 500.0, 600.0, 600.0]])
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.4) == [0, 2]


def test_nms_returns_indices_best_first():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [500.0, 500.0, 510.0, 510.0]])
    assert nms(boxes, np.array([0.1, 0.99]), 0.4) == [1, 0]


def test_nms_on_nothing_is_empty_not_an_error():
    assert nms(np.empty((0, 4)), np.empty((0,)), 0.4) == []


# -------------------------------------------------------------- the face crop

def test_the_crop_puts_the_face_centre_in_the_middle():
    box = np.array([100.0, 200.0, 200.0, 340.0])       # centre (150, 270)
    matrix = face_crop_matrix(box, 192)
    centre = np.array([[150.0, 270.0]]) @ matrix[:, :2].T + matrix[:, 2]
    assert centre.ravel().tolist() == pytest.approx([96.0, 96.0])


def test_the_crop_pads_the_longer_side_by_the_trained_factor():
    """1.5x the longer side fills the crop. A tighter crop is off-distribution
    for these weights and moves every point."""
    box = np.array([0.0, 0.0, 100.0, 60.0])            # longer side 100
    matrix = face_crop_matrix(box, 192)
    assert matrix[0, 0] == pytest.approx(192 / (100 * CROP_PAD))


def test_the_crop_scale_follows_the_longer_side_whichever_it_is():
    wide = face_crop_matrix(np.array([0.0, 0.0, 100.0, 60.0]), 192)
    tall = face_crop_matrix(np.array([0.0, 0.0, 60.0, 100.0]), 192)
    assert wide[0, 0] == pytest.approx(tall[0, 0])


def test_points_survive_the_round_trip_through_the_crop():
    """The one test that would catch an inverse transform built the easy way."""
    matrix = face_crop_matrix(np.array([300.0, 150.0, 460.0, 350.0]), 192)
    sensor = np.array([[310.0, 160.0], [450.0, 340.0], [380.0, 250.0]], dtype=np.float32)
    in_crop = sensor @ matrix[:, :2].T + matrix[:, 2]
    assert points_to_sensor(in_crop, matrix) == pytest.approx(sensor, abs=1e-3)


def test_landmarks_map_back_into_the_frame_the_face_was_found_in():
    """Sensor space, not crop space — the contract says the points are pixels of
    the frame that was passed in, and EAR is a ratio so a wrong scale hides."""
    box = np.array([300.0, 150.0, 460.0, 350.0])
    matrix = face_crop_matrix(box, 192)
    centre_of_crop = decode_landmarks(np.zeros(212, dtype=np.float32), 192)
    back = points_to_sensor(centre_of_crop, matrix)
    assert back[0].tolist() == pytest.approx([380.0, 250.0])   # the box centre


# --------------------------------------------------------- landmark decoding

def test_decode_maps_the_networks_minus_one_to_one_range_onto_the_crop():
    raw = np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    assert decode_landmarks(raw, 192).tolist() == [[0.0, 0.0], [96.0, 96.0], [192.0, 192.0]]


def test_decode_gives_one_row_per_point():
    assert decode_landmarks(np.zeros((1, 212), dtype=np.float32), 192).shape == (106, 2)


def test_decode_does_not_write_through_to_the_engine_buffer():
    """The buffer is reused between frames; decoding in place would corrupt it."""
    raw = np.zeros((1, 212), dtype=np.float32)
    decode_landmarks(raw, 192)
    assert not raw.any()


# ------------------------------------------------------------- the pipeline

def test_the_pipeline_requires_a_driver_frame():
    pipeline = PerclosTestPipeline(StubLandmarks(), ear_threshold=0.2, window_seconds=1.0)
    with pytest.raises(ValueError, match="driver"):
        pipeline.tick(road=frame())


def test_a_frame_gives_landmarks_an_ear_and_a_stamped_envelope():
    stub = StubLandmarks([valid_result()])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0,
                                   source_rate_hz=10.0)
    result = pipeline.tick(driver=frame(), timestamp=1.5)

    assert isinstance(result, DriverFrameResult)
    assert result.landmarks.valid
    assert result.ear == pytest.approx(0.30)          # conftest's open eyes
    assert result.eyes_closed is False
    assert result.sync.frame_id == 0 and result.sync.timestamp == 1.5
    # The model never sees the envelope; the pipeline stamps it onto the contract.
    assert result.landmarks.sync is result.sync


def test_perclos_is_none_until_the_window_is_full():
    """A partial window is a different statistic, not a small-sample PERCLOS."""
    stub = StubLandmarks([valid_result(landmarks_106(opening=2.0))])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0,
                                   source_rate_hz=4.0)
    seen = [pipeline.tick(driver=frame()) for _ in range(3)]
    assert [r.perclos for r in seen] == [None, None, None]
    assert seen[-1].perclos_fill == pytest.approx(0.75)

    full = pipeline.tick(driver=frame())
    assert full.perclos == pytest.approx(1.0)         # four closed frames of four
    assert full.perclos_fill == pytest.approx(1.0)


def test_closed_eyes_read_as_closed_and_open_eyes_do_not():
    stub = StubLandmarks([valid_result(landmarks_106(opening=12.0)),
                          valid_result(landmarks_106(opening=2.0))])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0,
                                   source_rate_hz=2.0)
    first = pipeline.tick(driver=frame())
    second = pipeline.tick(driver=frame())
    assert (first.eyes_closed, second.eyes_closed) == (False, True)
    assert second.perclos == pytest.approx(0.5)
    assert pipeline.closed_frames == 1


def test_a_frame_with_no_face_is_not_a_frame_with_open_eyes():
    """The failure this prevents: a driver who leaves the camera reading as alert.
    A no-face frame must not enter the window at all."""
    stub = StubLandmarks([invalid_result()])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0,
                                   source_rate_hz=2.0)
    result = pipeline.tick(driver=frame())

    assert result.ear is None and result.eyes_closed is None
    assert result.perclos is None
    assert result.perclos_fill == 0.0                 # nothing was ingested
    assert pipeline.no_face == 1 and pipeline.faces == 0


def test_a_valid_result_with_too_few_points_is_still_no_measurement():
    """valid=True is the detector's claim; EAR reads fixed index ranges and says
    None when they are not there. Zero would be a closed eye."""
    stub = StubLandmarks([valid_result(np.zeros((40, 2)))])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0,
                                   source_rate_hz=2.0)
    result = pipeline.tick(driver=frame())
    assert result.ear is None and result.eyes_closed is None
    assert pipeline.no_face == 1


def test_the_face_rate_reports_how_much_of_the_window_was_measurable():
    stub = StubLandmarks([valid_result(), valid_result(), invalid_result(), invalid_result()])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=10.0)
    for _ in range(4):
        pipeline.tick(driver=frame())
    assert pipeline.face_rate == pytest.approx(0.5)


def test_reset_clears_the_window_as_well_as_the_counters():
    stub = StubLandmarks([valid_result(landmarks_106(opening=2.0))])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0,
                                   source_rate_hz=2.0)
    pipeline.tick(driver=frame())
    pipeline.tick(driver=frame())
    assert pipeline.perclos.value is not None

    pipeline.reset()
    assert pipeline.perclos.value is None
    assert (pipeline.frames, pipeline.faces, pipeline.closed_frames) == (0, 0, 0)


def test_closing_the_pipeline_closes_the_model():
    stub = StubLandmarks()
    with PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=1.0):
        pass
    assert stub.closed


def test_the_summary_says_warming_up_rather_than_a_number():
    stub = StubLandmarks([valid_result()])
    pipeline = PerclosTestPipeline(stub, ear_threshold=0.2, window_seconds=60.0)
    pipeline.tick(driver=frame())
    assert "warming up" in pipeline.summary()


# ------------------------------------------------- landmarks -> EAR, together

def test_landmarks_carried_through_a_crop_give_the_same_ear():
    """EAR is a ratio, so it must survive the crop round trip exactly — which is
    also why a broken inverse transform would not show up in the EAR alone."""
    points = landmarks_106(opening=12.0, width=40.0)
    matrix = face_crop_matrix(np.array([180.0, 280.0, 420.0, 320.0]), 192)
    in_crop = points @ matrix[:, :2].T + matrix[:, 2]
    assert ear_from_landmarks(points_to_sensor(in_crop, matrix)) == pytest.approx(
        ear_from_landmarks(points), rel=1e-5)
