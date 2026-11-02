"""The attentiveness signal math: rolling windows, EAR, PERCLOS."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import eye_contour, landmarks_106, rotate

from dgt.attention import (
    Perclos,
    RollingWindow,
    ear_from_landmarks,
    eye_openness_ratio,
)


# --- RollingWindow ---------------------------------------------------------

def test_window_length_scales_with_fps_not_a_frame_constant():
    assert RollingWindow(2.0, 30.0).maxlen == 60
    assert RollingWindow(2.0, 21.0).maxlen == 42


def test_window_is_fifo_and_resettable():
    w = RollingWindow(1.0, 3.0)  # maxlen 3
    for v in (1, 2, 3, 4):
        w.append(v)
    assert w.values() == [2, 3, 4]
    assert w.is_full
    w.reset()
    assert len(w) == 0 and not w.is_full


@pytest.mark.parametrize("seconds,fps", [(0.0, 30.0), (1.0, 0.0), (-1.0, 30.0)])
def test_window_rejects_nonsense_dimensions(seconds, fps):
    with pytest.raises(ValueError):
        RollingWindow(seconds, fps)


# --- EAR -------------------------------------------------------------------

def test_eye_openness_is_opening_over_width():
    contour = eye_contour(0.0, 0.0, width=40.0, opening=12.0)
    # abs=1e-6: the width carries an epsilon guard against a collapsed eye.
    assert eye_openness_ratio(contour) == pytest.approx(0.30, abs=1e-6)


@pytest.mark.parametrize("degrees", [0.0, 1.0, 7.0, 13.0, 30.0, 45.0, 60.0, 89.0,
                                     90.0, 127.0, 180.0, -17.0, -45.0, -90.0])
def test_ear_is_roll_invariant(degrees):
    """Head tilt must not change EAR — that is why the front-end is geometric.

    Swept rather than checked at one angle: this test used to try 30 degrees
    alone, which happened to be an orientation where the canthi rounded the
    harmless way. Every other angle was free to be wrong, and on NumPy 1.21 the
    same 30 degrees rounded the other way and the bug finally showed.
    """
    upright = landmarks_106(opening=12.0, width=40.0)
    tilted = rotate(upright, degrees)
    # abs=1e-6 against the nominal value: the width carries an epsilon guard
    # against a collapsed eye, so the exact result is 0.2999999925. Angle to
    # angle the agreement is far tighter, which is what this is really about.
    assert ear_from_landmarks(tilted) == pytest.approx(0.30, abs=1e-6)
    assert ear_from_landmarks(tilted) == pytest.approx(
        ear_from_landmarks(upright), abs=1e-9
    )


def test_a_canthus_is_never_counted_as_a_lid_point():
    """The regression itself, stated directly rather than through a roll angle.

    The canthi define the axis, so they sit at perpendicular distance zero. If
    one is sorted into a lid group it drags that group's mean to 4/5 of its
    true value and EAR reads 0.27 instead of 0.30 — a number that still looks
    entirely plausible, which is what made this survive so long.
    """
    contour = eye_contour(0.0, 0.0, width=40.0, opening=12.0)
    # Nudge both canthi off the axis by a few ulps, in opposite directions —
    # exactly what floating point does to them after a rotation.
    nudged = contour.copy()
    nudged[0, 1] += 1e-13
    nudged[1, 1] -= 1e-13
    assert eye_openness_ratio(nudged) == pytest.approx(0.30, abs=1e-6)


def test_ear_is_scale_invariant():
    """A face nearer the camera is not a more open eye."""
    small = landmarks_106(opening=12.0, width=40.0)
    large = small * 3.0
    # abs=1e-6, not tighter: the epsilon in `width + _EPS` has a slightly
    # smaller relative effect on a larger eye, so the two agree to ~5e-9 rather
    # than exactly. That residue is the guard, not the geometry.
    assert ear_from_landmarks(large) == pytest.approx(
        ear_from_landmarks(small), abs=1e-6
    )


def test_ear_averages_both_eyes():
    points = np.full((106, 2), 1000.0)
    points[33:43] = eye_contour(200.0, 300.0, width=40.0, opening=12.0)  # 0.30
    points[87:97] = eye_contour(400.0, 300.0, width=40.0, opening=4.0)   # 0.10
    assert ear_from_landmarks(points) == pytest.approx(0.20, abs=1e-6)


def test_ear_returns_none_rather_than_a_low_value_when_there_is_no_face():
    # None must never be confused with "eyes closed".
    assert ear_from_landmarks(None) is None
    assert ear_from_landmarks(np.zeros((96, 2))) is None
    assert ear_from_landmarks(np.zeros((106,))) is None


def test_collapsed_points_are_zero_not_an_exception():
    assert eye_openness_ratio(np.zeros((10, 2))) == 0.0
    assert eye_openness_ratio(np.zeros((2, 2))) == 0.0


# --- PERCLOS ---------------------------------------------------------------

def test_first_closed_frame_does_not_report_100_percent():
    """The regression this class exists to prevent.

    Dividing by frames BUFFERED (not window size) made one closed frame report
    PERCLOS = 1.0, which fed straight into a drowsiness verdict.
    """
    p = Perclos(ear_threshold=0.20, window_seconds=1.0, fps=10.0)
    assert p.update(0.05) is None
    assert p.value is None
    assert not p.is_ready


def test_perclos_reports_nothing_until_the_window_is_filled():
    p = Perclos(ear_threshold=0.20, window_seconds=1.0, fps=10.0)  # 10 frames
    for _ in range(9):
        assert p.update(0.30) is None
    assert p.update(0.30) == pytest.approx(0.0)
    assert p.is_ready and p.is_full


def test_perclos_is_the_closed_fraction_once_ready():
    p = Perclos(ear_threshold=0.20, window_seconds=1.0, fps=10.0)  # 10 frames
    for _ in range(3):
        p.update(0.05)   # closed
    for _ in range(6):
        p.update(0.30)   # open
    assert p.update(0.30) == pytest.approx(0.3)   # 3 closed of 10
    # An 11th frame evicts the oldest (closed) sample: the window slides.
    assert p.update(0.30) == pytest.approx(0.2)


def test_min_fill_frac_trades_warm_up_for_responsiveness():
    p = Perclos(ear_threshold=0.20, window_seconds=1.0, fps=10.0, min_fill_frac=0.5)
    for _ in range(4):
        assert p.update(0.05) is None
    assert p.update(0.05) == pytest.approx(1.0)  # 5 of 10 frames is enough now


def test_fill_frac_exposes_warm_up_progress():
    p = Perclos(ear_threshold=0.20, window_seconds=1.0, fps=10.0)
    assert p.fill_frac == 0.0
    for _ in range(5):
        p.update(0.30)
    assert p.fill_frac == pytest.approx(0.5)


def test_perclos_reset_clears_readiness():
    p = Perclos(ear_threshold=0.20, window_seconds=1.0, fps=10.0)
    for _ in range(10):
        p.update(0.30)
    assert p.is_ready
    p.reset()
    assert not p.is_ready and p.value is None


@pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
def test_perclos_rejects_a_nonsense_fill_fraction(frac):
    with pytest.raises(ValueError):
        Perclos(min_fill_frac=frac)
