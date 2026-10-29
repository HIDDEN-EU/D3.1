#!/usr/bin/env python3
"""
IMM (Interacting Multiple Model) tracker for BEV vehicle tracking.

Why: KalmanBoxTracker in track_and_visualize.py uses a pure constant-velocity
(CV) model — vx, vy are propagated linearly and never rotate with the
target's heading. That's fine on straightaways but wrong through roundabouts
and turns, where the true velocity vector curves; the predicted box keeps
shooting off along the old heading and either misses the gate (dropped
track / ID switch) or lags badly during occlusion.

This module runs two motion models per track and blends them each frame:
  - "cv" : constant velocity            (same dynamics as the original tracker)
  - "ct" : coordinated turn             (velocity vector rotates at the
                                          estimated turn rate omega — the
                                          textbook model for roundabouts /
                                          turning maneuvers)
Mode probabilities mu = [mu_cv, mu_ct] are updated every frame from each
model's measurement likelihood (Bar-Shalom & Li IMM estimator), so the
combined estimate automatically shifts weight to "ct" mid-turn and back to
"cv" once the vehicle straightens out again — unlike a single fixed model.

Drop-in replacement for MultiObjectTracker / KalmanBoxTracker — same public
interface (predict/update/get_state/_match/reset/get_future_trajectories),
same MATCH_DIST gates, so track_and_visualize.py can switch trackers with:

    from imm_tracker import IMMMultiObjectTracker as MultiObjectTracker
"""
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

# Per-track state history, one entry per tracker.update() call (native ~2Hz
# nuScenes keyframe cadence). See track_and_visualize.py's HIST_LEN for why
# 16 (8s) is enough for QCNet's 5s/50-step interpolated input window.
HIST_LEN = 16

# Same class-gate distances as track_and_visualize.py's MATCH_DIST.
MATCH_DIST = {0: 18.0, 1: 15.0, 2: 10.0, 3: 15.0, 4: 12.0,   # car/truck/cv/bus/trailer
              5: 2.5,  6:  5.0, 7:  5.0, 8:  2.0,  9:  2.0}  # barrier/moto/bike/ped/cone

_EPS_OMEGA = 1e-4   # below this, treat turn rate as ~0 to avoid a 0/0 in the CT model

# Robustness bounds (see changes.txt 2026-07-13 "runaway omega" entry).
# 3D detectors frequently flip a car box's yaw by ~180 deg between frames
# (front/tail confusion on a near-symmetric box). Without a bound, that looks
# to the filter like a huge one-frame turn, omega spikes, and — especially
# across a multi-frame occlusion where no measurement can correct it — the
# yaw state spirals (observed: -3.9 rad -> -14 rad over 5 frames on a real
# nuScenes roundabout scene). OMEGA_MAX caps how fast we ever believe a
# vehicle is actually turning; YAW_RESID_CLIP caps how much a single noisy
# yaw measurement can move the state in one update.
OMEGA_MAX = 2.0          # rad/s (~115 deg/s) — generous even for a tight, slow roundabout turn
YAW_RESID_CLIP = 1.0     # rad (~57 deg) — beyond this, treat the residual as a likely sensor artifact


# ══════════════════════════════════════════════════════════════════════════════
# Motion models — state layout shared by both: [x, y, vx, vy, omega, yaw]
# ══════════════════════════════════════════════════════════════════════════════

def _cv_transition(x: np.ndarray, dt: float):
    """Constant velocity. Linear, so returns (next_state, F)."""
    F = np.array([
        [1, 0, dt, 0,  0, 0],
        [0, 1, 0,  dt, 0, 0],
        [0, 0, 1,  0,  0, 0],
        [0, 0, 0,  1,  0, 0],
        [0, 0, 0,  0,  1, 0],
        [0, 0, 0,  0,  dt, 1],
    ], dtype=float)
    return F @ x, F


def _ct_transition(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Coordinated turn: velocity vector rotates at rate omega (rad/s).
    Nonlinear in omega — this is what lets the filter follow a roundabout
    instead of flying off tangent to it.
    """
    px, py, vx, vy, omega, yaw = x
    if abs(omega) < _EPS_OMEGA:
        nx, ny = px + vx * dt, py + vy * dt
        nvx, nvy = vx, vy
    else:
        s, c = np.sin(omega * dt), np.cos(omega * dt)
        nx = px + (vx * s - vy * (1 - c)) / omega
        ny = py + (vy * s + vx * (1 - c)) / omega
        nvx = vx * c - vy * s
        nvy = vx * s + vy * c
    return np.array([nx, ny, nvx, nvy, omega, yaw + omega * dt])


def _ct_jacobian(x: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian of _ct_transition (central to avoid asymmetry near omega=0)."""
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n); d[i] = eps
        f_plus  = _ct_transition(x + d, dt)
        f_minus = _ct_transition(x - d, dt)
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    return J


def _gauss_likelihood(y: np.ndarray, S: np.ndarray) -> float:
    """N(y; 0, S) — used to weight each IMM model by how well it explains the residual."""
    n = len(y)
    detS = np.linalg.det(S)
    if detS <= 0:
        return 1e-300
    invS = np.linalg.inv(S)
    norm = 1.0 / np.sqrt((2 * np.pi) ** n * detS)
    return float(norm * np.exp(-0.5 * y @ invS @ y))


# ══════════════════════════════════════════════════════════════════════════════
# IMM tracker
# ══════════════════════════════════════════════════════════════════════════════

class IMMBoxTracker:
    """
    Two-model IMM (CV + coordinated-turn) single-object tracker in BEV.

    State (per submodel) : [x, y, vx, vy, omega, yaw]  (6-D)
    Obs                   : [x, y, yaw]                 (3-D, same as KalmanBoxTracker)
    z, w, l, h tracked as constants; updated whenever a detection matches.

    self.x / self.P are the *combined* (IMM-mixed) state/covariance — this is
    what get_state()/_match() read, so callers don't need to know a mixture
    of models is running underneath.
    """
    _next_id = 1

    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=float)
    R = np.diag([0.5, 0.5, 0.1])                          # measurement noise (same as original)

    # Process noise per model. CT gets a much larger omega variance than CV
    # so it can actually track a changing turn rate through a roundabout;
    # CV's small omega variance keeps it committed to "not turning".
    Q_cv = np.diag([0.1, 0.1, 1.0, 1.0, 0.05, 0.01])
    Q_ct = np.diag([0.2, 0.2, 1.5, 1.5, 0.30, 0.05])

    # Markov mode-transition matrix, TRANS[i, j] = P(model j at k | model i at k-1).
    # Sticky (diagonal-heavy) so mode probabilities don't flicker frame to frame,
    # but switch within a handful of frames at the start/end of a maneuver.
    TRANS = np.array([
        [0.95, 0.05],
        [0.15, 0.85],
    ])

    def __init__(self, det7: np.ndarray, label: int):
        """det7 = [x, y, z_bottom, w, l, h, yaw] or [..., yaw, score]"""
        self.id    = IMMBoxTracker._next_id
        IMMBoxTracker._next_id += 1
        self.label = label
        self.hits  = 1
        self.age   = 0
        self.time_since_update = 0
        # detection score of the last matched detection (1.0 if not supplied)
        self.score = float(det7[7]) if len(det7) > 7 else 1.0

        self.z = float(det7[2])
        self.w = float(det7[3])
        self.l = float(det7[4])
        self.h = float(det7[5])

        x0, y0, yaw0 = float(det7[0]), float(det7[1]), float(det7[6])
        x_init = np.array([x0, y0, 0., 0., 0., yaw0])
        P_init = np.diag([1.0, 1.0, 10.0, 10.0, 1.0, 0.1])

        self.model_x = [x_init.copy(), x_init.copy()]   # [cv, ct]
        self.model_P = [P_init.copy(), P_init.copy()]
        self.mu = np.array([0.9, 0.1])                   # start biased toward "not turning"
        self._c = self.mu.copy()                         # predicted mode probs (set by predict())

        self.x = x_init.copy()
        self.P = P_init.copy()

        # (x, y, vx, vy, yaw, valid) per frame, oldest first — same layout as
        # KalmanBoxTracker.history (note: combined-state vx,vy,yaw, not omega).
        self.history = deque(maxlen=HIST_LEN)

    def record(self):
        """Append current combined state to history. Call once per track per
        frame, after predict()/update() for this cycle have both run."""
        self.history.append((self.x[0], self.x[1], self.x[2], self.x[3],
                              self.x[5], self.time_since_update == 0))

    # ------------------------------------------------------------------
    def predict(self, dt: float = 1.0) -> np.ndarray:
        # 1. Mixing: blend each model's prior state using the mode probabilities,
        #    so a track that was 80% "ct" last frame still starts its CV
        #    prediction from a state informed by the turning estimate.
        c = self.TRANS.T @ self.mu
        c = np.maximum(c, 1e-12)
        mix = (self.TRANS * self.mu[:, None]) / c[None, :]   # mix[i, j]

        mixed_x = [sum(mix[i, j] * self.model_x[i] for i in range(2)) for j in range(2)]
        mixed_P = []
        for j in range(2):
            Pj = np.zeros((6, 6))
            for i in range(2):
                d = (self.model_x[i] - mixed_x[j]).reshape(-1, 1)
                Pj += mix[i, j] * (self.model_P[i] + d @ d.T)
            mixed_P.append(Pj)

        # 2. Model-matched time update.
        x_cv, F_cv = _cv_transition(mixed_x[0], dt)
        P_cv = F_cv @ mixed_P[0] @ F_cv.T + self.Q_cv
        x_cv[4] = np.clip(x_cv[4], -OMEGA_MAX, OMEGA_MAX)

        x_ct = _ct_transition(mixed_x[1], dt)
        J_ct = _ct_jacobian(mixed_x[1], dt)
        P_ct = J_ct @ mixed_P[1] @ J_ct.T + self.Q_ct
        x_ct[4] = np.clip(x_ct[4], -OMEGA_MAX, OMEGA_MAX)

        self.model_x = [x_cv, x_ct]
        self.model_P = [P_cv, P_ct]
        self._c = c   # prior for mode-prob update if a measurement follows

        # No measurement yet this cycle -> mode probs are just the predicted ones.
        self.mu = c
        self._combine()

        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, det7: np.ndarray):
        z = np.array([det7[0], det7[1], det7[6]])

        likelihoods = np.empty(2)
        for j in range(2):
            xj, Pj = self.model_x[j], self.model_P[j]
            y = z - self.H @ xj
            y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi   # wrap angle residual
            y[2] = np.clip(y[2], -YAW_RESID_CLIP, YAW_RESID_CLIP)   # guard vs. heading-flip artifacts

            S = self.H @ Pj @ self.H.T + self.R
            K = Pj @ self.H.T @ np.linalg.inv(S)
            self.model_x[j] = xj + K @ y
            self.model_x[j][4] = np.clip(self.model_x[j][4], -OMEGA_MAX, OMEGA_MAX)
            self.model_P[j] = (np.eye(6) - K @ self.H) @ Pj
            likelihoods[j] = _gauss_likelihood(y, S)

        denom = float(self._c @ likelihoods)
        self.mu = self._c.copy() if denom < 1e-300 else (self._c * likelihoods) / denom
        self._combine()

        self.time_since_update = 0
        self.hits += 1
        self.score = float(det7[7]) if len(det7) > 7 else 1.0
        self.z, self.w, self.l, self.h = float(det7[2]), float(det7[3]), float(det7[4]), float(det7[5])

    def _combine(self):
        """Fuse the per-model states/covariances into self.x / self.P via mode probs."""
        self.x = sum(self.mu[j] * self.model_x[j] for j in range(2))
        P = np.zeros((6, 6))
        for j in range(2):
            d = (self.model_x[j] - self.x).reshape(-1, 1)
            P += self.mu[j] * (self.model_P[j] + d @ d.T)
        self.P = P

    def get_state(self) -> np.ndarray:
        """Return [x, y, z_bottom, w, l, h, yaw]"""
        return np.array([self.x[0], self.x[1], self.z, self.w, self.l, self.h, self.x[5]])

    # Uniform accessors so callers (e.g. JSON export) don't need to know the
    # per-model state layout differs from KalmanBoxTracker's.
    @property
    def vx_global(self): return self.x[2]
    @property
    def vy_global(self): return self.x[3]
    @property
    def yaw_rate(self):  return self.x[4]   # omega

    def predict_future(self, n_steps: int = 10, dt: float = 1.0) -> np.ndarray:
        """
        Simulate n_steps forward without mutating tracker state. Returns [n, 3] (x, y, z).
        Uses the currently-dominant model (cv vs ct) so a track mid-turn keeps
        curving in the projection instead of snapping to a straight line.
        """
        x = self.x.copy()
        use_ct = self.mu[1] > self.mu[0]
        out = np.empty((n_steps, 3))
        for i in range(n_steps):
            x = _ct_transition(x, dt) if use_ct else _cv_transition(x, dt)[0]
            out[i] = [x[0], x[1], self.z]
        return out


class IMMMultiObjectTracker:
    """
    Frame-by-frame tracker built on IMMBoxTracker. Call reset() at the start
    of each scene. Same interface/semantics as MultiObjectTracker.

    Returns per frame:
      det_boxes  [M,7],  det_labels  [M]   <- matched to a detection this frame
      pred_boxes [P,7],  pred_labels [P]   <- predicted while detection absent
    """

    def __init__(self, max_age: int = 3, min_hits: int = 2, tentative_grace: int = 0):
        self.max_age         = max_age
        self.min_hits        = min_hits
        self.tentative_grace = tentative_grace
        self.tracks: list[IMMBoxTracker] = []

    def reset(self):
        self.tracks = []
        IMMBoxTracker._next_id = 1

    # ------------------------------------------------------------------
    def update(self, det_boxes7: np.ndarray, det_labels: np.ndarray):
        for t in self.tracks:
            t.predict()

        matched, unmatched_dets, unmatched_trks = self._match(det_boxes7, det_labels)

        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(det_boxes7[d_idx])

        for d_idx in unmatched_dets:
            self.tracks.append(IMMBoxTracker(det_boxes7[d_idx], int(det_labels[d_idx])))

        out_det_b, out_det_l = [], []
        out_pred_b, out_pred_l = [], []
        to_del = []

        for i, t in enumerate(self.tracks):
            t.record()
            confirmed = t.hits >= self.min_hits
            if t.time_since_update == 0 and confirmed:
                out_det_b.append(t.get_state())
                out_det_l.append(t.label)
            elif 0 < t.time_since_update <= self.max_age and confirmed:
                out_pred_b.append(t.get_state())
                out_pred_l.append(t.label)

            if t.time_since_update > self.max_age:
                to_del.append(i)
            elif not confirmed and t.time_since_update > self.tentative_grace:
                to_del.append(i)

        for i in reversed(to_del):
            self.tracks.pop(i)

        def _arr(lst, dim):
            return np.array(lst, dtype=float) if lst else np.empty((0, dim), dtype=float)

        def _lbl(lst):
            return np.array(lst, dtype=int) if lst else np.empty(0, dtype=int)

        return _arr(out_det_b, 7), _lbl(out_det_l), _arr(out_pred_b, 7), _lbl(out_pred_l)

    # ------------------------------------------------------------------
    def get_future_trajectories(self, n_steps: int = 10):
        """Return [(track_id, positions [n,3], label), ...] for all confirmed tracks."""
        return [(t.id, t.predict_future(n_steps), t.label)
                for t in self.tracks if t.hits >= self.min_hits]

    # ------------------------------------------------------------------
    def _match(self, det_boxes7, det_labels):
        n_det = len(det_boxes7)
        n_trk = len(self.tracks)

        if n_trk == 0 or n_det == 0:
            return [], list(range(n_det)), list(range(n_trk))

        trk_xy = np.array([t.x[:2] for t in self.tracks])   # [T, 2]
        det_xy = det_boxes7[:, :2]                            # [D, 2]

        diff = det_xy[:, None, :] - trk_xy[None, :, :]        # [D, T, 2]
        cost = np.sqrt((diff ** 2).sum(axis=-1))               # [D, T]

        trk_lbl = np.array([t.label for t in self.tracks])
        cost[det_labels[:, None] != trk_lbl[None, :]] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)

        trk_speed = np.array([np.hypot(t.x[2], t.x[3]) for t in self.tracks])

        matched, matched_d, matched_t = [], set(), set()
        for d_idx, t_idx in zip(row_ind, col_ind):
            base_thr = MATCH_DIST.get(int(det_labels[d_idx]), 2.0)
            thr = min(base_thr + 1.5 * trk_speed[t_idx], 2.0 * base_thr)
            if cost[d_idx, t_idx] <= thr:
                matched.append((t_idx, d_idx))
                matched_d.add(d_idx)
                matched_t.add(t_idx)

        unmatched_dets = [d for d in range(n_det) if d not in matched_d]
        unmatched_trks = [t for t in range(n_trk) if t not in matched_t]
        return matched, unmatched_dets, unmatched_trks
