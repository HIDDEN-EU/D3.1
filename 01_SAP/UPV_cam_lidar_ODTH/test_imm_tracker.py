"""
Unit tests for imm_tracker.py — IMM (CV + coordinated-turn) tracker.

No GPU/vision deps needed; imm_tracker.py only imports numpy/scipy.
"""
import numpy as np
import pytest

import imm_tracker as imm


def _det(x, y, yaw=0., label=0):
    return np.array([[x, y, 0., 4., 2., 1.5, yaw]]), np.array([label])


def _tracker(min_hits=1, max_age=3):
    t = imm.IMMMultiObjectTracker(max_age=max_age, min_hits=min_hits)
    t.reset()
    return t


# ── sanity: gates unchanged from track_and_visualize.py ──────────────────────

class TestMatchDist:
    def test_car_gate_18m(self):
        assert imm.MATCH_DIST[0] == 18.0

    def test_pedestrian_unchanged(self):
        assert imm.MATCH_DIST[8] == 2.0


# ── straight-line behaviour should match the old CV tracker ──────────────────

class TestStraightLineTracking:
    def test_single_id_across_fast_frames(self):
        tracker = _tracker(min_hits=1)
        ids_seen = set()
        for x in [0., 14., 28., 42.]:
            det, lbl = _det(x, 0.)
            tracker.update(det, lbl)
            for t in tracker.tracks:
                if t.time_since_update == 0:
                    ids_seen.add(t.id)
        assert len(ids_seen) == 1, f"Expected 1 track ID, got {ids_seen}"

    def test_no_track_explosion(self):
        tracker = _tracker(min_hits=1)
        for x in [0., 14., 28.]:
            tracker.update(*_det(x, 0.))
        assert len(tracker.tracks) <= 2

    def test_stationary_car_not_matched_25m_away(self):
        tracker = _tracker(min_hits=1)
        tracker.update(*_det(0., 0.))
        tracker.update(*_det(25., 0.))
        assert len(tracker.tracks) == 2

    def test_mode_stays_cv_dominant_on_straight_path(self):
        """No turning -> mu_cv should stay >= mu_ct throughout."""
        tracker = _tracker(min_hits=1)
        for x in [0., 10., 20., 30., 40.]:
            tracker.update(*_det(x, 0.))
        trk = tracker.tracks[0]
        assert trk.mu[0] >= trk.mu[1], f"Expected CV-dominant mode on straight path, got mu={trk.mu}"


# ── roundabout / turning motion — the actual improvement over pure CV ────────

def _circle_point(theta, R):
    """Coordinated-turn parametrization: starts at origin heading +x, curves left."""
    return R * np.sin(theta), R * (1 - np.cos(theta)), theta


class TestRoundaboutTracking:
    """
    A car goes around part of a roundabout: radius R=20m, turn rate
    omega=0.5 rad/s, dt=1s/frame (NuScenes-like 2Hz after subsampling) ->
    ~10 m/frame speed, quarter turn over ~pi/2 / 0.5 = ~3.14s.
    """
    R = 20.0
    OMEGA = 0.5
    N_FRAMES = 8

    def _drive_circle(self, tracker):
        pts = []
        for k in range(self.N_FRAMES):
            theta = k * self.OMEGA
            x, y, yaw = _circle_point(theta, self.R)
            pts.append((x, y, yaw))
            tracker.update(*_det(x, y, yaw))
        return pts

    def test_single_id_through_turn(self):
        """Track must not fragment into multiple IDs while turning."""
        tracker = _tracker(min_hits=1)
        ids_seen = set()
        for k in range(self.N_FRAMES):
            theta = k * self.OMEGA
            x, y, yaw = _circle_point(theta, self.R)
            tracker.update(*_det(x, y, yaw))
            for t in tracker.tracks:
                if t.time_since_update == 0:
                    ids_seen.add(t.id)
        assert len(ids_seen) == 1, f"Expected 1 track ID through the turn, got {ids_seen}"

    def test_mode_switches_to_ct_during_turn(self):
        """After several frames of sustained curving, mu_ct should dominate."""
        tracker = _tracker(min_hits=1)
        self._drive_circle(tracker)
        trk = tracker.tracks[0]
        assert trk.mu[1] > trk.mu[0], (
            f"Expected coordinated-turn mode to dominate after sustained turning, got mu={trk.mu}"
        )

    def test_occluded_prediction_beats_cv_extrapolation(self):
        """
        Core claim: once the IMM has 'learned' the turn, a predict-only step
        (simulating an occluded frame) should land closer to the true next
        point on the circle than naive constant-velocity extrapolation from
        the last known position+velocity.
        """
        tracker = _tracker(min_hits=1)
        pts = self._drive_circle(tracker)
        trk = tracker.tracks[0]

        # last known state before the "occluded" frame
        last_x, last_y = trk.x[0], trk.x[1]
        last_vx, last_vy = trk.x[2], trk.x[3]

        # ground truth next point on the circle
        theta_next = self.N_FRAMES * self.OMEGA
        true_x, true_y, _ = _circle_point(theta_next, self.R)

        # naive CV extrapolation (what the old pure-CV tracker would predict)
        cv_pred_x = last_x + last_vx * 1.0
        cv_pred_y = last_y + last_vy * 1.0
        cv_err = np.hypot(cv_pred_x - true_x, cv_pred_y - true_y)

        # IMM predict-only step (occlusion: no detection this frame)
        trk.predict()
        imm_err = np.hypot(trk.x[0] - true_x, trk.x[1] - true_y)

        assert imm_err < cv_err, (
            f"IMM prediction error ({imm_err:.2f}m) should beat CV extrapolation "
            f"error ({cv_err:.2f}m) through a turn"
        )

    def test_transitions_back_to_cv_after_straightening_out(self):
        """After the turn ends and the car drives straight again, mu_cv should recover."""
        tracker = _tracker(min_hits=1)
        pts = self._drive_circle(tracker)
        # drive straight, tangent to the circle at the last point, for a while
        last_x, last_y, last_yaw = pts[-1]
        vx, vy = 10 * np.cos(last_yaw), 10 * np.sin(last_yaw)
        x, y = last_x, last_y
        for _ in range(10):
            x, y = x + vx, y + vy
            tracker.update(*_det(x, y, last_yaw))
        trk = tracker.tracks[0]
        assert trk.mu[0] > trk.mu[1], (
            f"Expected CV mode to recover after straightening out, got mu={trk.mu}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
