"""Gaze calibration: per-target median, homography fit, apply, save/load (needs cv2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dgt.calibration import DEFAULT_TARGETS, Calibration, CalibrationSession

pytest.importorskip("cv2")

TRUE_H = np.array([[0.4, 0.05, 0.5], [-0.03, 0.35, 0.45], [0.02, -0.01, 1.0]])


def _inverse_map(point):
    """The raw gaze point that TRUE_H maps onto `point`."""
    u, v, w = np.linalg.inv(TRUE_H) @ np.array([point[0], point[1], 1.0])
    return u / w, v / w


def test_nine_targets_in_a_three_by_three_layout():
    assert len(DEFAULT_TARGETS) == 9
    assert sorted({x for x, _ in DEFAULT_TARGETS}) == sorted({y for _, y in DEFAULT_TARGETS})


def test_a_known_homography_is_recovered_from_the_nine_targets():
    session = CalibrationSession()
    rng = np.random.default_rng(0)
    for j, target in enumerate(session.targets):
        raw = np.array(_inverse_map(target))
        for _ in range(15):
            session.add(j, tuple(raw + rng.normal(0, 1e-3, 2)))
        session.add(j, tuple(raw + 5.0))  # one outlier: the median ignores it
    calibration = session.fit()
    for target in session.targets:
        assert calibration.apply(*_inverse_map(target)) == pytest.approx(target, abs=2e-3)


def test_frames_without_a_face_are_skipped():
    session = CalibrationSession()
    session.add(0, None)
    assert session.observations()[0] is None


def test_fewer_than_four_targets_cannot_be_fitted():
    session = CalibrationSession()
    for j in range(3):
        session.add(j, _inverse_map(session.targets[j]))
    with pytest.raises(ValueError, match="need >= 4"):
        session.fit()


def test_an_unfitted_calibration_is_unavailable():
    calibration = Calibration()
    assert calibration.available is False and calibration.apply(0.1, 0.2) is None


def test_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "calibration.json"
    Calibration(TRUE_H).save(path)
    assert np.allclose(Calibration.load(path).matrix, TRUE_H)
