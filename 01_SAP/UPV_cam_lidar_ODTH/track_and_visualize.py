#!/usr/bin/env python3
"""
BEVFusion tracking visualizer with Kalman-filter occlusion prediction.

Detected boxes   → class-specific colours (same as run_detection.py)
Predicted boxes  → orange, shown ONLY while detection is absent

Usage (from /home/braulio/bevfusion, bevfusion conda env):

  # Run inference and render scenes 0, 1, 2:
  MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \\
      --scenes 0 1 2

  # Run inference, save raw results, render scenes 0-4:
  MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \\
      --save-results /tmp/bf_results.pkl --scenes 0-4

  # Re-render later from saved results (no GPU needed for inference):
  MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \\
      --results /tmp/bf_results.pkl --scenes 0 5 10

  --scenes accepts individual numbers and/or dash-ranges, e.g.  0 1 3-6 9
"""

import argparse, json, os, pickle, sys, uuid
from collections import deque
import cv2
import mmcv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmdet.apis import multi_gpu_test
from torchpack import distributed as dist
from torchpack.utils.config import configs as torchpack_configs

from imm_tracker import IMMMultiObjectTracker

# ── paths & model ──────────────────────────────────────────────────────────────
CONFIG     = "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
CHECKPOINT = "pretrained/bevfusion-det.pth"
VAL_PKL    = "data/nuscenes/nuscenes_infos_val.pkl"
OUT_DIR    = "tracked_scenes"
JSON_OUT_DIR = "tracked_scenes_json"

DEFAULT_SCORE_THR = 0.3
PC_RANGE = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

OBJECT_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
MOBILE_CLASSES = {"car", "truck", "bus", "trailer", "motorcycle", "bicycle", "pedestrian"}

# BEVFusion label -> ODT category string (see JSON template/ODT.json)
ODT_CATEGORY = {
    "car": "Car", "truck": "Truck", "construction_vehicle": "ConstructionVehicle",
    "bus": "Bus", "trailer": "Trailer", "barrier": "Barrier",
    "motorcycle": "Motorcycle", "bicycle": "Bicycle",
    "pedestrian": "Pedestrian", "traffic_cone": "TrafficCone",
}

SENDER_ID = "bevfusion_track_v1"
STEP_DT   = 0.5   # seconds per future-trajectory step (nuScenes keyframes are 2 Hz)

# class colour palette (RGB)
PALETTE = {
    "car":                  (255, 158,   0),
    "truck":                (255,  99,  71),
    "construction_vehicle": (233, 150,  70),
    "bus":                  (255,  69,   0),
    "trailer":              (255, 140,   0),
    "barrier":              (112, 128, 144),
    "motorcycle":           (255,  61,  99),
    "bicycle":              (220,  20,  60),
    "pedestrian":           (  0,   0, 230),
    "traffic_cone":         ( 47,  79,  79),
}

PRED_COLOR_RGB = (255, 165,   0)   # orange for all predicted boxes
PRED_COLOR_BGR = (  0, 165, 255)

# Per-track state history, one entry per tracker.update() call (native ~2Hz
# nuScenes keyframe cadence). 16 entries = 8s, comfortably more than the 5s
# window QCNet's pretrained model needs (see qcnet_predictor.py, which
# interpolates this to QCNet's native 10Hz/50-step training cadence).
HIST_LEN = 16

# Max BEV centre-to-centre distance (world frame) for det↔track matching.
# Base gate handles vehicles up to ~130 km/h at NuScenes 2Hz (≈18 m/step).
# A velocity-aware expansion (see _match) further widens the gate for fast tracks.
MATCH_DIST = {0: 18.0, 1: 15.0, 2: 10.0, 3: 15.0, 4: 12.0,  # car/truck/cv/bus/trailer
              5: 2.5,  6:  5.0, 7:  5.0, 8:  2.0,  9:  2.0}  # barrier/moto/bike/ped/cone

# video layout (1280×720)
FRAME_W, FRAME_H    = 1280, 720
CAM_COLS, CAM_ROWS  = 3, 2
CAM_W = FRAME_W // CAM_COLS
CAM_H = (FRAME_H * 2) // (CAM_ROWS * 3)
INFO_H = FRAME_H - CAM_H * CAM_ROWS
BEV_W  = INFO_H

CAM_GRID = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT",
]


# ══════════════════════════════════════════════════════════════════════════════
# Kalman filter tracker
# ══════════════════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Single-object Kalman filter in BEV.

    State  : [x, y, vx, vy, yaw, v_yaw]  (6-D)
    Obs    : [x, y, yaw]                  (3-D)
    z, w, l, h tracked as constants; updated whenever a detection matches.
    """
    _next_id = 1

    def __init__(self, det7: np.ndarray, label: int):
        """det7 = [x, y, z_bottom, w, l, h, yaw]"""
        self.id    = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.label = label
        self.hits  = 1
        self.age   = 0
        self.time_since_update = 0

        # constants updated on each detection match
        self.z = float(det7[2])
        self.w = float(det7[3])
        self.l = float(det7[4])
        self.h = float(det7[5])

        # state transition  (dt=1 normalised frame)
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0,  0,  0 ],
            [0, 1, 0,  dt, 0,  0 ],
            [0, 0, 1,  0,  0,  0 ],
            [0, 0, 0,  1,  0,  0 ],
            [0, 0, 0,  0,  1,  dt],
            [0, 0, 0,  0,  0,  1 ],
        ], dtype=float)

        # observation matrix  [x, y, yaw]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], dtype=float)

        self.Q = np.diag([0.1, 0.1, 1.0, 1.0, 0.01, 0.1])   # process noise
        self.R = np.diag([0.5, 0.5, 0.1])                     # measurement noise
        self.P = np.diag([1.0, 1.0, 10.0, 10.0, 0.1, 1.0])   # initial covariance

        x0, y0, yaw0 = float(det7[0]), float(det7[1]), float(det7[6])
        self.x = np.array([x0, y0, 0., 0., yaw0, 0.], dtype=float)

        # (x, y, vx, vy, yaw, valid) per frame, oldest first. valid=False marks
        # a coasted (undetected) frame — used to build QCNet's valid_mask.
        self.history = deque(maxlen=HIST_LEN)

    def record(self):
        """Append current state to history. Call once per track per frame,
        after predict()/update() for this cycle have both run."""
        self.history.append((self.x[0], self.x[1], self.x[2], self.x[3],
                              self.x[4], self.time_since_update == 0))

    # ------------------------------------------------------------------
    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, det7: np.ndarray):
        z = np.array([det7[0], det7[1], det7[6]])
        y = z - self.H @ self.x
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi   # wrap angle residual

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.z, self.w, self.l, self.h = float(det7[2]), float(det7[3]), float(det7[4]), float(det7[5])

    def get_state(self) -> np.ndarray:
        """Return [x, y, z_bottom, w, l, h, yaw]"""
        return np.array([self.x[0], self.x[1], self.z, self.w, self.l, self.h, self.x[4]])

    # Uniform accessors so callers (e.g. JSON export) don't need to know the
    # per-tracker state layout differs from IMMBoxTracker's.
    @property
    def vx_global(self): return self.x[2]
    @property
    def vy_global(self): return self.x[3]
    @property
    def yaw_rate(self):  return self.x[5]

    def predict_future(self, n_steps: int = 10) -> np.ndarray:
        """Simulate filter n_steps forward without mutating state. Returns [n, 3] (x, y, z)."""
        x = self.x.copy()
        out = np.empty((n_steps, 3))
        for i in range(n_steps):
            x = self.F @ x
            out[i] = [x[0], x[1], self.z]
        return out


class MultiObjectTracker:
    """
    Frame-by-frame tracker.  Call reset() at the start of each scene.

    Returns per frame:
      det_boxes  [M,7],  det_labels  [M]   ← matched to a detection this frame
      pred_boxes [P,7],  pred_labels [P]   ← predicted while detection absent
    """

    def __init__(self, max_age: int = 3, min_hits: int = 2, tentative_grace: int = 0):
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.tentative_grace = tentative_grace
        self.tracks: list[KalmanBoxTracker] = []

    def reset(self):
        self.tracks = []
        KalmanBoxTracker._next_id = 1

    # ------------------------------------------------------------------
    def update(self, det_boxes7: np.ndarray, det_labels: np.ndarray):
        """
        det_boxes7 : [N, 7]  (x, y, z_bottom, w, l, h, yaw)
        det_labels : [N]     int class indices

        Returns (det_boxes, det_labels, pred_boxes, pred_labels) — all np arrays.
        """
        # 1. predict all tracks one step forward
        for t in self.tracks:
            t.predict()

        # 2. match detections → tracks
        matched, unmatched_dets, unmatched_trks = self._match(det_boxes7, det_labels)

        # 3. update matched tracks
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(det_boxes7[d_idx])

        # 4. spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self.tracks.append(KalmanBoxTracker(det_boxes7[d_idx], int(det_labels[d_idx])))

        # 5. record history, collect outputs & prune dead tracks
        out_det_b, out_det_l = [], []
        out_pred_b, out_pred_l = [], []
        to_del = []

        for i, t in enumerate(self.tracks):
            t.record()
            confirmed = t.hits >= self.min_hits
            if t.time_since_update == 0 and confirmed:
                # matched this frame and has enough history → show as detected
                out_det_b.append(t.get_state())
                out_det_l.append(t.label)
            elif 0 < t.time_since_update <= self.max_age and confirmed:
                # missed this frame but was previously confirmed → show prediction
                out_pred_b.append(t.get_state())
                out_pred_l.append(t.label)

            # delete: aged out confirmed tracks, OR tentative tracks that exceed grace period
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
        """Return [(track_id, positions [n,3], label), ...] in global frame for all confirmed tracks."""
        return [(t.id, t.predict_future(n_steps), t.label)
                for t in self.tracks if t.hits >= self.min_hits]

    # ------------------------------------------------------------------
    def _match(self, det_boxes7, det_labels):
        n_det = len(det_boxes7)
        n_trk = len(self.tracks)

        if n_trk == 0 or n_det == 0:
            return [], list(range(n_det)), list(range(n_trk))

        trk_xy  = np.array([t.get_state()[:2] for t in self.tracks])  # [T, 2]
        det_xy  = det_boxes7[:, :2]                                    # [D, 2]

        diff = det_xy[:, None, :] - trk_xy[None, :, :]                # [D, T, 2]
        cost = np.sqrt((diff ** 2).sum(axis=-1))                       # [D, T]

        # block cross-class matches
        trk_lbl = np.array([t.label for t in self.tracks])
        cost[det_labels[:, None] != trk_lbl[None, :]] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)

        # Per-track gate: expand by estimated speed so fast vehicles get a larger
        # search radius while stationary ones stay tight.
        trk_speed = np.array([np.hypot(t.x[2], t.x[3]) for t in self.tracks])

        matched, matched_d, matched_t = [], set(), set()
        for d_idx, t_idx in zip(row_ind, col_ind):
            base_thr = MATCH_DIST.get(int(det_labels[d_idx]), 2.0)
            # Allow 1.5× the predicted displacement on top of the base gate,
            # capped at 2× base so we don't gate out on obviously wrong matches.
            thr = min(base_thr + 1.5 * trk_speed[t_idx], 2.0 * base_thr)
            if cost[d_idx, t_idx] <= thr:
                matched.append((t_idx, d_idx))
                matched_d.add(d_idx)
                matched_t.add(t_idx)

        unmatched_dets = [d for d in range(n_det) if d not in matched_d]
        unmatched_trks = [t for t in range(n_trk) if t not in matched_t]
        return matched, unmatched_dets, unmatched_trks


# ══════════════════════════════════════════════════════════════════════════════
# Ego-motion / coordinate helpers
# ══════════════════════════════════════════════════════════════════════════════

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """NuScenes quaternion [w,x,y,z] → 3×3 rotation matrix."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def build_lidar2global(info: dict) -> np.ndarray:
    """4×4 transform: LiDAR frame → global (world) frame."""
    R_l2e = _quat_to_mat(np.array(info["lidar2ego_rotation"]))
    t_l2e = np.array(info["lidar2ego_translation"])
    R_e2g = _quat_to_mat(np.array(info["ego2global_rotation"]))
    t_e2g = np.array(info["ego2global_translation"])

    R = R_e2g @ R_l2e
    t = R_e2g @ t_l2e + t_e2g
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def _z_rotation(R33: np.ndarray) -> float:
    """Extract the z-axis rotation angle (yaw) from a 3×3 rotation matrix."""
    return float(np.arctan2(R33[1, 0], R33[0, 0]))


def ego_pose_xy_yaw(info: dict):
    """Ego position (x, y) and heading (yaw, rad) in the global frame."""
    x, y = info["ego2global_translation"][:2]
    yaw = _z_rotation(_quat_to_mat(np.array(info["ego2global_rotation"])))
    return float(x), float(y), yaw


def boxes_lidar_to_global(boxes7: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform [N,7] boxes (x,y,z,w,l,h,yaw) from LiDAR → global frame."""
    if len(boxes7) == 0:
        return boxes7.copy()
    out = boxes7.copy()
    xyz = np.concatenate([boxes7[:, :3], np.ones((len(boxes7), 1))], axis=1)
    out[:, :3] = (T @ xyz.T).T[:, :3]
    out[:, 6]  = boxes7[:, 6] + _z_rotation(T[:3, :3])
    return out


def boxes_global_to_lidar(boxes7: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform [N,7] boxes from global → current LiDAR frame."""
    if len(boxes7) == 0:
        return boxes7.copy()
    return boxes_lidar_to_global(boxes7, np.linalg.inv(T))


def build_lidar2ego(info: dict) -> np.ndarray:
    """4×4 transform: LiDAR frame → ego frame (the frame the ODT JSON is expressed in)."""
    R = _quat_to_mat(np.array(info["lidar2ego_rotation"]))
    t = np.array(info["lidar2ego_translation"])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def lidar_to_ego(boxes7_lidar: np.ndarray, T_l2e: np.ndarray) -> np.ndarray:
    """Convert [N,7] box positions from LiDAR frame to ego frame."""
    if len(boxes7_lidar) == 0:
        return boxes7_lidar.copy()
    out = boxes7_lidar.copy()
    xyz = np.concatenate([boxes7_lidar[:, :3], np.ones((len(boxes7_lidar), 1))], axis=1)
    out[:, :3] = (T_l2e @ xyz.T).T[:, :3]
    out[:, 6]  = boxes7_lidar[:, 6] + _z_rotation(T_l2e[:3, :3])
    return out


def global_traj_to_ego(traj_global_xy: np.ndarray, T_l2g: np.ndarray, T_l2e: np.ndarray) -> np.ndarray:
    """Convert [N,2] global trajectory positions to ego frame [N,2]."""
    T_g2e = T_l2e @ np.linalg.inv(T_l2g)
    n = len(traj_global_xy)
    pts4 = np.concatenate([traj_global_xy, np.zeros((n, 1)), np.ones((n, 1))], axis=1)
    return (T_g2e @ pts4.T).T[:, :2]


def track_render_state(t, max_age: int, min_hits: int):
    """Classify a track the same way MultiObjectTracker/IMMMultiObjectTracker.update()
    does when it splits tracks into rendered "detected" vs "predicted" (coasted)
    boxes -- used so the JSON export mirrors exactly what the video shows.
    Returns "detected", "predicted", or None (not shown this frame)."""
    confirmed = t.hits >= min_hits
    if t.time_since_update == 0 and confirmed:
        return "detected"
    elif 0 < t.time_since_update <= max_age and confirmed:
        return "predicted"
    return None


def build_frame_json(info: dict, tracker, max_age: int, min_hits: int,
                     future_trajs_by_tid: dict, future_steps: int,
                     seq_num: int, prediction_source: str) -> dict:
    """Build one ODT JSON object (see 'JSON template/ODT.json') for a single frame,
    from the tracker's own current track list -- independent of --tracker/
    --future-predictor, which only affect what get_state()/get_future_trajectories()
    (or the QCNet lookup) return upstream.

    future_trajs_by_tid: {track_id: positions [n,3] global}, built once per frame
        in make_tracked_video() from whichever future-trajectory source is active
        (KF/IMM physics rollout, or QCNet top1/nearest-to-imm). Tracks missing from
        this dict (only possible in QCNet mode, e.g. a track absent from the
        offline QCNet dump) fall back to that track's own physics rollout so every
        mobile-class detection still gets a predicted_trajectory.
    """
    T_l2g = build_lidar2global(info)
    T_l2e = build_lidar2ego(info)
    T_g2e = T_l2e @ np.linalg.inv(T_l2g)
    R_g2e = T_g2e[:2, :2]

    ex, ey, eyaw = ego_pose_xy_yaw(info)
    timestamp = info.get("timestamp", 0.0) / 1e6   # microseconds -> seconds

    sender_state = {
        "agent_type": "ego",
        "position": {"x": ex, "y": ey, "z": float(info["ego2global_translation"][2])},
        "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": eyaw},
    }

    detections = []
    for t in tracker.tracks:
        state = track_render_state(t, max_age, min_hits)
        if state is None:
            continue

        state_g = t.get_state()[np.newaxis, :]
        ego_box = boxes_global_to_lidar(state_g, T_l2g)
        ego_box = lidar_to_ego(ego_box, T_l2e)[0]

        cat_str = OBJECT_CLASSES[t.label]
        odt_cat = ODT_CATEGORY.get(cat_str, "Unknown")

        vg = np.array([t.vx_global, t.vy_global])
        ve = R_g2e @ vg
        vx_ego, vy_ego = float(ve[0]), float(ve[1])

        det = {
            "track_id":  t.id,
            "category":  odt_cat,
            "score":     1.0,   # already filtered by --score-thr upstream
            "timestamp": timestamp,
            "x":         float(ego_box[0]),
            "y":         float(ego_box[1]),
            "z":         float(ego_box[2]),
            "length":    float(t.l),
            "width":     float(t.w),
            "height":    float(t.h),
            "heading":   float(ego_box[6]),
            "yaw_rate":  float(t.yaw_rate),
            "vx":        vx_ego,
            "vy":        vy_ego,
            "optional":  {"v": float(np.hypot(vx_ego, vy_ego)), "track_state": state},
        }

        if cat_str in MOBILE_CLASSES:
            traj_global_xyz = future_trajs_by_tid.get(t.id)
            src = prediction_source
            if traj_global_xyz is None:
                traj_global_xyz = t.predict_future(future_steps)
                src = f"{prediction_source}_fallback"

            traj_ego = global_traj_to_ego(traj_global_xyz[:, :2], T_l2g, T_l2e)
            det["prediction_source"] = src
            det["predicted_trajectory"] = [
                {"t": round(timestamp + STEP_DT * (i + 1), 3),
                 "x": round(float(traj_ego[i, 0]), 4),
                 "y": round(float(traj_ego[i, 1]), 4),
                 "z": 0.0}
                for i in range(len(traj_ego))
            ]

        detections.append(det)

    return {
        "metadata": {
            "schema_version":  "1.0",
            "message_type":    "Ego_3D_ODT",
            "message_id":      str(uuid.uuid4()),
            "sender_id":       SENDER_ID,
            "sequence_number": seq_num,
            "send_timestamp":  timestamp,
            "time_reference":  "UTC",
        },
        "sender_state": sender_state,
        "axes_convention": {"x": "forward", "y": "left", "z": "up"},
        "detections": detections,
    }


def select_qcnet_trajectories(frame_data: dict, mode_select: str,
                               physics_by_id: dict = None) -> list:
    """Turn one frame's raw QCNet output (tid -> dict(all_modes, probs, label, z),
    as saved by qcnet_infer.py) into the [(positions[n,3], label), ...] shape
    the renderer expects, by picking ONE of the 6 modes per track.

    mode_select:
      "top1"          -- QCNet's own highest-confidence mode (what a plain
                          deployment would show).
      "nearest-to-imm" -- the mode whose full trajectory is closest (mean L2)
                          to the IMM physics tracker's own rollout for that
                          same track/frame -- no ground truth involved, just a
                          consistency signal available at real inference time.
                          Validated in changes.txt (2026-07-15) to beat both
                          KF and IMM outright, including against independently
                          matched nuScenes ground truth. Falls back to top1
                          for any track missing from physics_by_id (e.g. a
                          fresh, not-yet-confirmed track).
    physics_by_id: tid -> [n_steps, 2] rollout from the CURRENT tracker replay
        (only needed/used for "nearest-to-imm"; must be an IMM tracker's
        rollout -- see the --tracker imm requirement enforced in main()).
    """
    out = []
    for tid, d in frame_data.items():
        all_modes, probs, label, z = d["all_modes"], d["probs"], d["label"], d["z"]
        ref = physics_by_id.get(tid) if (mode_select == "nearest-to-imm" and physics_by_id) else None
        if ref is not None:
            n = min(all_modes.shape[1], ref.shape[0])
            dists = np.linalg.norm(all_modes[:, :n] - ref[:n], axis=-1).mean(axis=1)
            sel = int(np.argmin(dists))
        else:
            sel = int(np.argmax(probs))
        xy = all_modes[sel]                              # [n_steps, 2]
        positions = np.concatenate([xy, np.full((len(xy), 1), z, dtype=np.float32)], axis=1)
        out.append((tid, positions, label))
    return out


def trajs_global_to_lidar(trajs_global: list, T_l2g: np.ndarray) -> list:
    """Convert [(track_id, positions [n,3], label), ...] from global to LiDAR frame."""
    if not trajs_global:
        return []
    T_g2l = np.linalg.inv(T_l2g)
    result = []
    for tid, positions, label in trajs_global:
        n = len(positions)
        pts4 = np.concatenate([positions, np.ones((n, 1))], axis=1)   # [n, 4]
        lidar = (T_g2l @ pts4.T).T[:, :3]                             # [n, 3]
        result.append((tid, lidar, label))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

_EDGES = [(0,1),(0,3),(0,4),(1,2),(1,5),(3,2),(3,7),(4,5),(4,7),(2,6),(5,6),(6,7)]


def _dashed_line(img, p1, p2, color, thickness=1, dash=8, gap=5):
    """Draw a dashed line in place on a BGR image."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    length = np.linalg.norm(p2 - p1)
    if length < 1:
        return
    d = (p2 - p1) / length
    pos, drawing = 0.0, True
    while pos < length:
        seg = dash if drawing else gap
        end = min(pos + seg, length)
        if drawing:
            a = tuple((p1 + d * pos).astype(int))
            b = tuple((p1 + d * end).astype(int))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        pos = end
        drawing = not drawing


def _boxes7_to_lidar(boxes7: np.ndarray) -> LiDARInstance3DBoxes:
    """Convert [N,7] array (x,y,z_bottom,w,l,h,yaw) to LiDARInstance3DBoxes."""
    return LiDARInstance3DBoxes(boxes7.astype(np.float32), box_dim=7)


def build_lidar2image(cam_info: dict) -> np.ndarray:
    R_s2l = np.array(cam_info["sensor2lidar_rotation"], dtype=np.float64)
    t_s2l = np.array(cam_info["sensor2lidar_translation"], dtype=np.float64)
    R_l2c = np.linalg.inv(R_s2l)
    t_l2c = t_s2l @ R_l2c.T
    rt = np.eye(4, dtype=np.float32)
    rt[:3, :3] = R_l2c.T
    rt[3,  :3] = -t_l2c.astype(np.float32)
    intr = np.eye(4, dtype=np.float32)
    intr[:3, :3] = cam_info["cam_intrinsic"]
    return (intr @ rt.T).astype(np.float32)


def draw_boxes_on_camera(img_bgr: np.ndarray,
                         boxes7: np.ndarray, labels: np.ndarray,
                         lidar2image: np.ndarray,
                         predicted: bool = False,
                         any_proj: bool = False) -> None:
    """Project 3-D boxes onto a BGR camera image (in-place)."""
    if len(boxes7) == 0:
        return
    bboxes = _boxes7_to_lidar(boxes7)
    corners = bboxes.corners        # [N, 8, 3]
    N = corners.shape[0]
    pts = np.concatenate([corners.reshape(-1, 3),
                          np.ones((N * 8, 1), dtype=np.float32)], axis=-1)
    proj = (pts @ lidar2image.T).reshape(N, 8, 4)

    # any_proj: keep box if *any* corner is in front of the camera (Fix 4).
    # Default (any_proj=False): require all 8 corners in front.
    keep = np.any(proj[..., 2] > 0, axis=1) if any_proj else np.all(proj[..., 2] > 0, axis=1)
    proj = proj[keep]; lbl = labels[keep]
    if len(proj) == 0:
        return

    order = np.argsort(-proj[..., 2].min(axis=1))
    proj = proj[order]; lbl = lbl[order]

    flat = proj.reshape(-1, 4)
    flat[:, 2] = np.clip(flat[:, 2], 1e-5, 1e5)
    flat[:, 0] /= flat[:, 2]
    flat[:, 1] /= flat[:, 2]
    pts2d = flat[:, :2].reshape(-1, 8, 2)

    H, W = img_bgr.shape[:2]
    for i in range(len(pts2d)):
        if predicted:
            bgr = PRED_COLOR_BGR
        else:
            bgr = tuple(int(c) for c in reversed(PALETTE[OBJECT_CLASSES[lbl[i]]]))

        for s, e in _EDGES:
            p1 = (int(np.clip(pts2d[i, s, 0], -10, W + 10)),
                  int(np.clip(pts2d[i, s, 1], -10, H + 10)))
            p2 = (int(np.clip(pts2d[i, e, 0], -10, W + 10)),
                  int(np.clip(pts2d[i, e, 1], -10, H + 10)))
            if predicted:
                _dashed_line(img_bgr, p1, p2, bgr, thickness=1)
            else:
                cv2.line(img_bgr, p1, p2, bgr, 2, cv2.LINE_AA)


def draw_trajectory_on_camera(img_bgr: np.ndarray,
                              future_trajs: list,
                              lidar2image: np.ndarray) -> None:
    """Project future trajectory centers (LiDAR frame) onto a BGR camera image (in-place)."""
    H, W = img_bgr.shape[:2]
    for _tid, positions, label in future_trajs:
        bgr = tuple(int(c) for c in reversed(PALETTE[OBJECT_CLASSES[label]]))
        pts2d = []
        for pos3 in positions:
            pt4 = np.array([pos3[0], pos3[1], pos3[2], 1.0], dtype=np.float32)
            proj = lidar2image @ pt4
            if proj[2] <= 0:
                pts2d.append(None)
                continue
            px = int(proj[0] / proj[2])
            py = int(proj[1] / proj[2])
            pts2d.append((px, py) if (0 <= px < W and 0 <= py < H) else None)

        prev = None
        for j, pt in enumerate(pts2d):
            if pt is None:
                prev = None
                continue
            radius = max(2, 5 - j // 2)
            cv2.circle(img_bgr, pt, radius, bgr, -1, cv2.LINE_AA)
            if prev is not None:
                _dashed_line(img_bgr, prev, pt, bgr, thickness=1)
            prev = pt


def render_bev(lidar_pts: np.ndarray,
               det_boxes7: np.ndarray, det_labels: np.ndarray,
               pred_boxes7: np.ndarray, pred_labels: np.ndarray,
               future_trajs: list = None,
               size: int = 480) -> np.ndarray:
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor="black")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_xlim(PC_RANGE[0], PC_RANGE[3])
    ax.set_ylim(PC_RANGE[1], PC_RANGE[4])
    ax.set_aspect(1); ax.set_axis_off()

    if lidar_pts is not None and len(lidar_pts):
        mask = ((lidar_pts[:, 0] > PC_RANGE[0]) & (lidar_pts[:, 0] < PC_RANGE[3]) &
                (lidar_pts[:, 1] > PC_RANGE[1]) & (lidar_pts[:, 1] < PC_RANGE[4]))
        ax.scatter(lidar_pts[mask, 0], lidar_pts[mask, 1],
                   s=0.2, c="white", alpha=0.5, linewidths=0)

    def _plot_boxes(boxes7, labels, ls, lw, alpha):
        if len(boxes7) == 0:
            return
        bboxes = _boxes7_to_lidar(boxes7)
        corners_xy = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for i in range(corners_xy.shape[0]):
            if ls == "--":
                c = np.array(PRED_COLOR_RGB) / 255.0
            else:
                c = np.array(PALETTE[OBJECT_CLASSES[labels[i]]]) / 255.0
            ax.plot(corners_xy[i, :, 0], corners_xy[i, :, 1],
                    ls, color=c, linewidth=lw, alpha=alpha)

    _plot_boxes(det_boxes7,  det_labels,  "-",  1.5, 1.0)
    _plot_boxes(pred_boxes7, pred_labels, "--", 1.0, 0.8)

    if future_trajs:
        n_steps = max((len(p) for _, p, _ in future_trajs), default=0)
        for _tid, positions, label in future_trajs:
            c = np.array(PALETTE[OBJECT_CLASSES[label]]) / 255.0
            xy = positions[:, :2]
            n = len(xy)
            ax.plot(xy[:, 0], xy[:, 1], ':', color=c, linewidth=0.9, alpha=0.55)
            for j in range(n):
                alpha = 0.85 * (1.0 - j / (n_steps + 1))
                size_pt = max(6.0 - j * 0.4, 1.5)
                ax.plot(xy[j, 0], xy[j, 1], 'o',
                        color=c, markersize=size_pt, alpha=alpha, zorder=5)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(size, size, 4)
    plt.close(fig)
    return rgba[:, :, :3]


def build_video_frame(cam_bgr: dict, bev_rgb: np.ndarray,
                      sample_idx: int, n_det: int, n_pred: int,
                      score_thr: float) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    for row in range(CAM_ROWS):
        for col in range(CAM_COLS):
            name = CAM_GRID[row * CAM_COLS + col]
            img  = cam_bgr.get(name)
            if img is None:
                continue
            cell = cv2.resize(img, (CAM_W, CAM_H))
            y0, x0 = row * CAM_H, col * CAM_W
            frame[y0:y0 + CAM_H, x0:x0 + CAM_W] = cell
            label = name.replace("CAM_", "").replace("_", " ")
            cv2.putText(frame, label, (x0 + 4, y0 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

    bev_bgr = cv2.cvtColor(cv2.resize(bev_rgb, (BEV_W, INFO_H)), cv2.COLOR_RGB2BGR)
    frame[FRAME_H - INFO_H:, FRAME_W - BEV_W:] = bev_bgr

    strip_w = FRAME_W - BEV_W
    frame[FRAME_H - INFO_H:, :strip_w] = (18, 18, 18)

    status = (f"Sample {sample_idx:04d}   |   "
              f"{n_det} detected   {n_pred} predicted  (conf > {score_thr})")
    cv2.putText(frame, status, (8, FRAME_H - INFO_H + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)

    # class colour legend
    lx, ly = 8, FRAME_H - INFO_H + 56
    for i, name in enumerate(OBJECT_CLASSES):
        bgr = tuple(int(c) for c in reversed(PALETTE[name]))
        cv2.rectangle(frame, (lx, ly - 10), (lx + 16, ly + 2), bgr, -1)
        cv2.putText(frame, name, (lx + 20, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        lx += 115
        if lx + 115 > strip_w:
            lx = 8; ly += 20

    # predicted legend entry
    cv2.rectangle(frame, (lx, ly - 10), (lx + 16, ly + 2), PRED_COLOR_BGR, -1)
    cv2.putText(frame, "predicted", (lx + 20, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# Video rendering with tracking
# ══════════════════════════════════════════════════════════════════════════════

def make_tracked_video(scene_infos: list, scene_results: list,
                       fps: int, score_thr: float, out_path: str,
                       max_age: int, min_hits: int,
                       future_steps: int = 10,
                       tentative_grace: int = 0,
                       any_proj: bool = False,
                       tracker_type: str = "kf",
                       future_predictor: str = "physics",
                       qcnet_futures_scene: dict = None,
                       qcnet_mode_select: str = "top1",
                       qcnet_dump: list = None,
                       json_dir: str = None,
                       prediction_source: str = "") -> None:
    """
    future_predictor: "physics" (default) uses the tracker's own KF/IMM
        rollout for the displayed future-trajectory line. "qcnet" instead
        looks up precomputed QCNet predictions from qcnet_futures_scene
        (produced offline by qcnet_infer.py, keyed by sample index — see
        --future-predictor/--qcnet-futures in the CLI). Independent of
        --tracker: kf/imm still does detection<->track association and
        occlusion-coasting (the orange "predicted box" while undetected);
        this only swaps the multi-step trajectory line.
    qcnet_mode_select: "top1" (default) or "nearest-to-imm" -- which of
        QCNet's 6 modes to draw. Only consulted when future_predictor ==
        "qcnet". See select_qcnet_trajectories()'s docstring.
    qcnet_dump: if a list is passed, each frame's track/ego state history
        is appended to it (for --dump-qcnet-input) so a separate QCNet-env
        process can run inference on it later — see qcnet_infer.py.
    json_dir: if set, one ODT-schema JSON file per frame (named after the
        sample token) is also written here — see build_frame_json().
    prediction_source: label stored in each JSON detection's
        "prediction_source" field (e.g. "kf", "imm", "qcnet_top1+imm").
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    if json_dir is not None:
        os.makedirs(json_dir, exist_ok=True)

    tracker_cls = IMMMultiObjectTracker if tracker_type == "imm" else MultiObjectTracker
    tracker = tracker_cls(max_age=max_age, min_hits=min_hits,
                          tentative_grace=tentative_grace)
    tracker.reset()

    if qcnet_dump is not None:
        ego_hist = deque(maxlen=HIST_LEN)
        prev_ego_xy = None

    n = len(scene_infos)
    for idx in tqdm(range(n), desc=os.path.basename(out_path), leave=False):
        info   = scene_infos[idx]
        result = scene_results[idx]

        # ── filter detections by score ────────────────────────────────────────
        boxes_np = result["boxes_3d"].tensor.numpy().copy()   # [N, 9]
        scores   = result["scores_3d"].numpy()
        labels   = result["labels_3d"].numpy()
        mask     = scores >= score_thr
        boxes_np = boxes_np[mask]
        labels   = labels[mask]

        # convert to 7-D (x,y,z_bottom, w,l,h, yaw) in LiDAR frame
        if len(boxes_np):
            boxes_np[:, 2] -= boxes_np[:, 5] / 2    # centre-z → bottom-z
            det_boxes7_lidar = boxes_np[:, :7].astype(np.float64)
        else:
            det_boxes7_lidar = np.empty((0, 7), dtype=np.float64)

        # ── ego-motion compensation: track in world frame ─────────────────────
        # BEVFusion outputs boxes in the ego/LiDAR frame, which shifts every
        # frame as the vehicle moves. Convert to global frame so the tracker
        # sees a fixed coordinate system, then convert predicted boxes back.
        T_l2g = build_lidar2global(info)
        det_boxes7_global = boxes_lidar_to_global(det_boxes7_lidar, T_l2g)

        det_b_g, det_l, pred_b_g, pred_l = tracker.update(det_boxes7_global, labels)

        if qcnet_dump is not None:
            ex, ey, eyaw = ego_pose_xy_yaw(info)
            evx, evy = (0.0, 0.0) if prev_ego_xy is None else \
                ((ex - prev_ego_xy[0]) / 0.5, (ey - prev_ego_xy[1]) / 0.5)
            ego_hist.append((ex, ey, evx, evy, eyaw, True))
            prev_ego_xy = (ex, ey)
            qcnet_dump.append(dict(
                sample_idx=idx,
                token=info.get("token"),
                ego_xy=(ex, ey),
                ego_history=list(ego_hist),
                tracks=[dict(id=t.id, label=int(t.label), z=t.z, history=list(t.history))
                        for t in tracker.tracks if t.hits >= min_hits],
            ))

        # future trajectories in global frame → convert to LiDAR frame
        if future_predictor == "qcnet":
            frame_data = (qcnet_futures_scene or {}).get(idx, {})
            physics_by_id = None
            if qcnet_mode_select == "nearest-to-imm":
                physics_by_id = {t.id: t.predict_future(future_steps)[:, :2]
                                  for t in tracker.tracks if t.hits >= min_hits}
            future_trajs_global = select_qcnet_trajectories(frame_data, qcnet_mode_select, physics_by_id)
        else:
            future_trajs_global = tracker.get_future_trajectories(future_steps)
        future_trajs_lidar  = trajs_global_to_lidar(future_trajs_global, T_l2g)

        if json_dir is not None:
            future_trajs_by_tid = {tid: positions for tid, positions, _ in future_trajs_global}
            frame_json = build_frame_json(info, tracker, max_age, min_hits,
                                          future_trajs_by_tid, future_steps,
                                          seq_num=idx, prediction_source=prediction_source)
            token = info.get("token", f"frame_{idx:04d}")
            with open(os.path.join(json_dir, f"{token}.json"), "w") as f:
                json.dump(frame_json, f, indent=2)

        det_b  = boxes_global_to_lidar(det_b_g,  T_l2g).astype(np.float32)
        pred_b = boxes_global_to_lidar(pred_b_g, T_l2g).astype(np.float32)

        # ── load LiDAR ────────────────────────────────────────────────────────
        try:
            lidar_pts = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 5)
        except Exception:
            lidar_pts = None

        # ── BEV ──────────────────────────────────────────────────────────────
        bev_rgb = render_bev(lidar_pts, det_b, det_l, pred_b, pred_l,
                             future_trajs=future_trajs_lidar)

        # ── cameras ──────────────────────────────────────────────────────────
        cam_bgr: dict = {}
        for cam_name, cam_info in info["cams"].items():
            img = mmcv.imread(cam_info["data_path"])
            if img is None:
                continue
            l2i = build_lidar2image(cam_info)
            if future_trajs_lidar:
                draw_trajectory_on_camera(img, future_trajs_lidar, l2i)
            if len(det_b):
                draw_boxes_on_camera(img, det_b, det_l, l2i, predicted=False, any_proj=any_proj)
            if len(pred_b):
                draw_boxes_on_camera(img, pred_b, pred_l, l2i, predicted=True, any_proj=any_proj)
            cam_bgr[cam_name] = img

        frame = build_video_frame(cam_bgr, bev_rgb, idx,
                                  len(det_b), len(pred_b), score_thr)
        writer.write(frame)

    writer.release()


# ══════════════════════════════════════════════════════════════════════════════
# Inference helpers (reused from run_detection.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> Config:
    torchpack_configs.load(config_path, recursive=True)
    return Config(recursive_eval(torchpack_configs), filename=config_path)


def build_inference_model(cfg: Config) -> torch.nn.Module:
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16 = cfg.get("fp16", None)
    if fp16 is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, CHECKPOINT, map_location="cpu")
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()
    return model


def split_by_scene(infos: list, results: list) -> list:
    scenes, cur_i, cur_r = [], [], []
    for info, res in zip(infos, results):
        if not info.get("prev_token"):
            if cur_i:
                scenes.append((cur_i, cur_r))
            cur_i, cur_r = [info], [res]
        else:
            cur_i.append(info); cur_r.append(res)
    if cur_i:
        scenes.append((cur_i, cur_r))
    return scenes


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_scene_tokens(tokens: list) -> list:
    """
    Parse mixed list like ['0', '1', '3-6', '9'] → [0,1,3,4,5,6,9].
    Dash-ranges are inclusive on both ends.
    """
    out = set()
    for tok in tokens:
        if '-' in tok:
            lo, hi = tok.split('-', 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", required=True, metavar="N",
                   help="Scene numbers to render.  Accepts individual values and "
                        "inclusive ranges, e.g.  0 1 3-6 9")
    p.add_argument("--results", default=None, metavar="PKL",
                   help="Path to pre-saved inference results pkl.  "
                        "If omitted, inference is run from scratch.")
    p.add_argument("--save-results", default=None, metavar="PKL",
                   help="Save raw inference results to this pkl for reuse.")
    p.add_argument("--score-thr",   type=float, default=DEFAULT_SCORE_THR)
    p.add_argument("--fps",         type=int,   default=5)
    p.add_argument("--max-age",     type=int,   default=3,
                   help="Frames a track can be invisible before deletion (default 3).")
    p.add_argument("--min-hits",    type=int,   default=2,
                   help="Detections needed before a new track is confirmed (default 2).")
    p.add_argument("--video-dir",   default=OUT_DIR)
    p.add_argument("--num-samples", type=int,   default=None,
                   help="Limit inference to first N samples (default: all 6019).")
    p.add_argument("--cameras", nargs="+", default=None, metavar="CAM",
                   help="Restrict the camera encoder to these channels, e.g. "
                        "--cameras CAM_FRONT (default: all 6 cameras, unchanged "
                        "behaviour). Only used when running fresh inference "
                        "(--results omitted); ignored with a pre-saved --results "
                        "bundle since the detections are already fixed.")
    p.add_argument("--future-steps", type=int, default=10,
                   help="Number of future steps to predict and visualize (default 10).")
    p.add_argument("--tentative-grace", type=int, default=0,
                   help="Missed frames a tentative (unconfirmed) track can survive before "
                        "deletion.  0=original behaviour, 1=Fix1 (one grace frame).")
    p.add_argument("--any-proj", action="store_true",
                   help="Fix 4: keep a box on a camera image if ANY corner is in front "
                        "of the lens (default requires ALL 8 corners in front).")
    p.add_argument("--tracker", choices=["kf", "imm"], default="kf",
                   help="kf = original constant-velocity Kalman tracker (default). "
                        "imm = 2-model IMM (CV + coordinated-turn), better through "
                        "roundabouts/turns — see imm_tracker.py.")
    p.add_argument("--future-predictor", choices=["physics", "qcnet"], default="physics",
                   help="physics = KF/IMM rollout (default). qcnet = precomputed "
                        "QCNet predictions loaded from --qcnet-futures (see "
                        "qcnet_infer.py; runs in the separate QCNet conda env). "
                        "Independent of --tracker, which still governs detection "
                        "association / occlusion coasting either way.")
    p.add_argument("--qcnet-futures", default=None, metavar="PKL",
                   help="Path to the futures pkl produced by qcnet_infer.py. "
                        "Required when --future-predictor qcnet.")
    p.add_argument("--qcnet-mode-select", choices=["top1", "nearest-to-imm"], default="top1",
                   help="Which of QCNet's 6 modes to draw (only with --future-predictor "
                        "qcnet). top1 (default) = QCNet's own highest-confidence mode. "
                        "nearest-to-imm = pick the mode closest to the IMM tracker's own "
                        "physics rollout each frame -- no ground truth used, validated in "
                        "changes.txt (2026-07-15) to beat both KF and IMM outright, "
                        "including against independently matched nuScenes ground truth. "
                        "Requires --tracker imm and a --qcnet-futures pkl produced by the "
                        "updated qcnet_infer.py (stores all 6 modes, not just top-1).")
    p.add_argument("--dump-qcnet-input", default=None, metavar="PKL",
                   help="Also dump per-frame track/ego history (needed by "
                        "qcnet_infer.py) for the rendered scenes to this pkl.")
    p.add_argument("--json-dir", default=JSON_OUT_DIR,
                   help="Directory for per-frame ODT-schema JSON output "
                        f"(default {JSON_OUT_DIR}/scene_XXX/<token>.json). "
                        "One file per frame, matching 'JSON template/ODT.json'.")
    p.add_argument("--no-json", action="store_true",
                   help="Skip JSON export entirely (video only).")
    return p.parse_args()


def main():
    args = parse_args()
    scene_ids = parse_scene_tokens(args.scenes)

    dist.init()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    # ── load config & dataset (always needed for data_infos) ─────────────────
    cfg = load_config(CONFIG)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.cameras:
        cfg.data.test.camera_channels = args.cameras
        print(f"[track] Restricting cameras to: {args.cameras}")

    if args.results is None:
        # limit samples if requested
        full_val = 6019
        if args.num_samples and args.num_samples < full_val:
            import tempfile
            with open(VAL_PKL, "rb") as f:
                vd = pickle.load(f)
            # Sort by timestamp first so the first N samples here are the same
            # as the first N samples the full dataset would produce — keeping
            # scene numbering consistent with bevfusion_scenes/.
            sorted_infos = sorted(vd["infos"], key=lambda x: x["timestamp"])
            mini = {"infos": sorted_infos[:args.num_samples], "metadata": vd["metadata"]}
            tmp_pkl = tempfile.mktemp(suffix=".pkl")
            with open(tmp_pkl, "wb") as f:
                pickle.dump(mini, f)
            cfg.data.test.ann_file = tmp_pkl
            print(f"[track] Limiting to {args.num_samples} samples (timestamp-sorted).")
        else:
            tmp_pkl = None

        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset, samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=True, shuffle=False,
        )

        print("[track] Loading model …")
        model = build_inference_model(cfg)
        print("[track] Running inference …")
        results = multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False)

        rank, _ = get_dist_info()
        if rank != 0:
            return

        # Save results bundled with their infos so reload is always aligned.
        # (The val pkl is not timestamp-sorted, so dataset.data_infos[:N] on a
        # reload would point to completely different samples than those used here.)
        infos = list(dataset.data_infos[:len(results)])
        if args.save_results:
            with open(args.save_results, "wb") as f:
                pickle.dump({"results": results, "infos": infos}, f)
            print(f"[track] Results bundle saved → {args.save_results}")

        if tmp_pkl:
            try: os.remove(tmp_pkl)
            except Exception: pass

    else:
        # Load pre-saved bundle (no GPU / model needed).
        # We still init dist + build dataset so torchpack is happy, but we
        # skip model loading entirely.
        dataset = build_dataset(cfg.data.test)
        rank, _ = get_dist_info()
        if rank != 0:
            return
        print(f"[track] Loading results bundle from {args.results} …")
        with open(args.results, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict) and "infos" in bundle:
            results = bundle["results"]
            infos   = bundle["infos"]
        else:
            # Backward-compat: plain results list (no saved infos → use dataset).
            # WARNING: may misalign if --num-samples was used during inference.
            results = bundle
            infos   = list(dataset.data_infos[:len(results)])
            print("[track] WARNING: bundle has no infos — alignment may be wrong "
                  "if inference used --num-samples.")

    rank, _ = get_dist_info()
    if rank != 0:
        return

    # ── split into scenes ─────────────────────────────────────────────────────
    scenes = split_by_scene(list(infos), results)
    print(f"[track] {len(infos)} samples → {len(scenes)} scenes.")

    # ── render requested scenes ───────────────────────────────────────────────
    os.makedirs(args.video_dir, exist_ok=True)
    valid = [si for si in scene_ids if si < len(scenes)]
    oob   = [si for si in scene_ids if si >= len(scenes)]
    if oob:
        print(f"[track] Warning: scene(s) {oob} out of range, skipping.")

    if args.future_predictor == "qcnet" and not args.qcnet_futures:
        raise ValueError("--future-predictor qcnet requires --qcnet-futures PATH "
                          "(see qcnet_infer.py to produce it).")
    if args.qcnet_mode_select == "nearest-to-imm":
        if args.future_predictor != "qcnet":
            raise ValueError("--qcnet-mode-select nearest-to-imm only applies with "
                              "--future-predictor qcnet.")
        if args.tracker != "imm":
            raise ValueError("--qcnet-mode-select nearest-to-imm requires --tracker imm "
                              "(the heuristic is validated specifically against the IMM "
                              "tracker's own rollout, not KF's -- see changes.txt 2026-07-15).")
    qcnet_futures_all = {}
    if args.qcnet_futures:
        with open(args.qcnet_futures, "rb") as f:
            qcnet_futures_all = pickle.load(f)
        print(f"[track] Loaded QCNet futures for {len(qcnet_futures_all)} scene(s) "
              f"from {args.qcnet_futures}")

    qcnet_dump_all = {}

    if args.future_predictor == "qcnet":
        prediction_source = f"qcnet_{args.qcnet_mode_select}+{args.tracker}"
    else:
        prediction_source = args.tracker

    if not args.no_json:
        os.makedirs(args.json_dir, exist_ok=True)

    for si in valid:
        scene_infos, scene_results = scenes[si]
        out_path = os.path.join(args.video_dir, f"scene_{si:03d}.mp4")
        print(f"[track] Rendering scene {si:03d}  ({len(scene_infos)} frames) → {out_path}")
        qcnet_dump = [] if args.dump_qcnet_input else None
        json_scene_dir = None if args.no_json else os.path.join(args.json_dir, f"scene_{si:03d}")
        make_tracked_video(scene_infos, scene_results,
                           args.fps, args.score_thr, out_path,
                           args.max_age, args.min_hits,
                           future_steps=args.future_steps,
                           tentative_grace=args.tentative_grace,
                           any_proj=args.any_proj,
                           tracker_type=args.tracker,
                           future_predictor=args.future_predictor,
                           qcnet_futures_scene=qcnet_futures_all.get(si),
                           qcnet_mode_select=args.qcnet_mode_select,
                           qcnet_dump=qcnet_dump,
                           json_dir=json_scene_dir,
                           prediction_source=prediction_source)
        if qcnet_dump is not None:
            qcnet_dump_all[si] = qcnet_dump
        print(f"         done → {out_path}"
              + ("" if args.no_json else f"  (+ JSON in {json_scene_dir}/)"))

    if args.dump_qcnet_input:
        with open(args.dump_qcnet_input, "wb") as f:
            pickle.dump(qcnet_dump_all, f)
        print(f"[track] QCNet input dump saved → {args.dump_qcnet_input}")

    print(f"\n[track] All done.  Videos in {args.video_dir}/"
          + ("" if args.no_json else f"  JSON in {args.json_dir}/"))


if __name__ == "__main__":
    main()
