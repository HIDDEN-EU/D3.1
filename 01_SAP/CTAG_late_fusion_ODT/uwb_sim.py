"""
uwb_sim.py - synthetic UC2 scenario: LiDAR tracker output + DENSO UWB messages with their imperfections.

Scenario (a few metres around a crossing, ego driving East and turning left, 24 s):
    VRU  Pedestrian carrying UWB tag 7: crosses the road, stops, walks on, turns, with stop-and-go.
    N1   pedestrian walking next to the VRU (1-1.5 m away, similar speed): the hard association case.
    N2   independent pedestrian crossing the scene.
    cars one parked, one driving ahead of the ego (pass-through objects).
Imperfections (all configurable):
    LiDAR  10 Hz, position/velocity noise, missed detections, false positives, the VRU is NOT detected during an
           occlusion window (hidden behind a van), a new LiDAR track id after the occlusion, and the ids of the VRU
           and of its neighbour are swapped at `swap_time`.
    UWB    1-10 Hz, position/velocity noise, latency + jitter (messages arrive out of order with respect to the
           LiDAR), multipath outliers, a dropout window.
Messages follow ODT.json (LiDAR, 'Ego_3D_ODT') and the 2D_ODT_UWB variant (x, y, vx, vy; no z, no size, no heading),
in the ego frame, each with the ego WGS84 pose at capture time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from UC2_fusion import MapFrame, rot2

ORIGIN_LLA = (39.4810, -0.3400, 10.0)          # Valencia
TAG_ID = 7
LIDAR_SENDER = "CTAG_lidar_ODTH"
UWB_SENDER = "DENSO_uwb_ODT"


@dataclass
class ScenarioConfig:
    seed: int = 0
    t0: float = 1_780_000_000.0
    duration: float = 24.0
    lidar_hz: float = 10.0
    uwb_hz: float = 10.0
    # LiDAR
    lidar_pos_sigma: float = 0.10
    lidar_vel_sigma: float = 0.30
    lidar_miss_prob: float = 0.05
    lidar_range: float = 45.0
    lidar_latency: float = 0.03
    lidar_fp_prob: float = 0.03                    # per frame: a short-lived false pedestrian appears
    vru_occlusions: Tuple[Tuple[float, float], ...] = ((8.0, 11.0),)    # seconds after t0, VRU not detected
    swap_time: Optional[float] = 14.0              # LiDAR ids of the VRU and the neighbour are swapped from here
    # UWB
    uwb_pos_sigma: float = 0.25
    uwb_vel_sigma: float = 0.30
    uwb_latency: float = 0.10
    uwb_jitter: float = 0.03
    uwb_outlier_prob: float = 0.02
    uwb_outlier_range: Tuple[float, float] = (2.0, 4.0)
    uwb_dropouts: Tuple[Tuple[float, float], ...] = ((16.0, 19.0),)
    uwb_bias: float = 0.0                          # constant offset between the tag and the body centre [m]


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------


class Traj:
    """Position/velocity over time, sampled finely and interpolated."""

    def __init__(self, t: np.ndarray, pos: np.ndarray, vel: np.ndarray):
        self.t, self.pos, self.vel = t, pos, vel

    def p(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.t, self.pos[:, 0]), np.interp(t, self.t, self.pos[:, 1])])

    def v(self, t: float) -> np.ndarray:
        return np.array([np.interp(t, self.t, self.vel[:, 0]), np.interp(t, self.t, self.vel[:, 1])])


def walk(start, segments, duration, tau=0.4, dt=0.01) -> Traj:
    """Piecewise target velocities [(t_end, vx, vy), ...] followed through a first-order lag (smooth stop and go)."""
    t = np.arange(0.0, duration + 1.0, dt)
    pos, vel = np.zeros((len(t), 2)), np.zeros((len(t), 2))
    pos[0] = start
    for k in range(1, len(t)):
        target = next(((vx, vy) for te, vx, vy in segments if t[k] < te), segments[-1][1:])
        vel[k] = vel[k - 1] + (np.array(target) - vel[k - 1]) * dt / tau
        pos[k] = pos[k - 1] + vel[k] * dt
    return Traj(t, pos, vel)


def ego_traj(duration, speed=2.5, start=(10.0, -5.0), dt=0.01):
    """Ego driving East, turning left between 6 and 14 s. Returns (t, pos, yaw)."""
    t = np.arange(0.0, duration + 1.0, dt)
    yaw = np.zeros(len(t))
    pos = np.zeros((len(t), 2))
    pos[0] = start
    for k in range(1, len(t)):
        yaw[k] = yaw[k - 1] + (0.05 * dt if 6.0 <= t[k] < 14.0 else 0.0)
        pos[k] = pos[k - 1] + speed * dt * np.array([math.cos(yaw[k]), math.sin(yaw[k])])
    return t, pos, yaw


@dataclass
class Actor:
    name: str
    category: str                                   # nuScenes-style name as the CTAG tracker publishes it
    size: Tuple[float, float, float]
    traj: Traj
    lidar_id: Callable[[float], int]
    heading0: float = 0.0


@dataclass
class Scenario:
    cfg: ScenarioConfig
    events: List[Tuple[float, dict]]                # (arrival time, message), sorted by arrival time
    actors: Dict[str, Actor]
    map: MapFrame
    _ego: Tuple[np.ndarray, np.ndarray, np.ndarray]

    @property
    def t_start(self) -> float:
        return self.cfg.t0

    @property
    def t_end(self) -> float:
        return self.cfg.t0 + self.cfg.duration

    def ego_pose(self, t: float) -> Tuple[np.ndarray, float]:
        te, pos, yaw = self._ego
        r = t - self.cfg.t0
        return np.array([np.interp(r, te, pos[:, 0]), np.interp(r, te, pos[:, 1])]), float(np.interp(r, te, yaw))

    def vru_map(self, t: float) -> np.ndarray:
        return self.actors["vru"].traj.p(t - self.cfg.t0)

    def vru_ego(self, t: float) -> np.ndarray:
        p, yaw = self.ego_pose(t)
        return rot2(-yaw) @ (self.vru_map(t) - p)

    def vru_lidar_id(self, t: float) -> int:
        return self.actors["vru"].lidar_id(t - self.cfg.t0)

    def in_windows(self, t: float, windows) -> bool:
        return any(a <= t - self.cfg.t0 < b for a, b in windows)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def make_actors(cfg: ScenarioConfig) -> Dict[str, Actor]:
    T = cfg.duration
    vru = walk((35.0, -7.0), [(5, 0.0, 1.4), (7, 0.0, 0.0), (12, 0.5, 1.2), (16, 1.2, 0.0), (20, 0.0, -1.0),
                              (1e9, -0.8, -0.5)], T)

    # neighbour: follows the VRU with a slowly varying offset (1-1.5 m away), so velocities are close but not equal
    t = vru.t
    off = np.stack([-1.0 + 0.4 * np.sin(0.5 * t), -0.9 + 0.3 * np.cos(0.4 * t)], axis=1)
    off_v = np.stack([0.2 * np.cos(0.5 * t), -0.12 * np.sin(0.4 * t)], axis=1)
    n1 = Traj(t, vru.pos + off, vru.vel + off_v)
    n2 = walk((20.0, 5.0), [(1e9, 1.1, 0.0)], T)

    swap = cfg.swap_time

    def vru_id(r: float) -> int:
        if r < 11.0:
            return 31
        if swap is not None and r >= swap:
            return 32
        return 47

    def n1_id(r: float) -> int:
        if swap is not None and r >= swap:
            return 47
        return 32

    static = lambda p: Traj(t, np.tile(p, (len(t), 1)), np.zeros((len(t), 2)))
    car_ahead = Traj(t, np.stack([25.0 + 2.5 * t, -5.0 + 0 * t], axis=1), np.tile([2.5, 0.0], (len(t), 1)))
    return {
        "vru": Actor("vru", "pedestrian", (0.6, 0.6, 1.75), vru, vru_id),
        "n1": Actor("n1", "pedestrian", (0.6, 0.6, 1.75), n1, n1_id),
        "n2": Actor("n2", "pedestrian", (0.6, 0.6, 1.75), n2, lambda r: 33),
        "parked": Actor("parked", "car", (4.5, 1.8, 1.5), static(np.array([38.0, -9.0])), lambda r: 40),
        "van": Actor("van", "truck", (5.2, 2.0, 2.2), static(np.array([36.0, -3.5])), lambda r: 41),
        "ahead": Actor("ahead", "car", (4.5, 1.8, 1.5), car_ahead, lambda r: 42),
    }


def _meta(sender: str, mtype: str, seq: int, t_send: float) -> dict:
    return {"schema_version": "1.0", "message_type": mtype, "message_id": f"{sender}-{seq}", "sender_id": sender,
            "sequence_number": seq, "send_timestamp": t_send, "time_reference": "UTC"}


def generate(cfg: Optional[ScenarioConfig] = None) -> Scenario:
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)
    mf = MapFrame(ORIGIN_LLA)
    ego = ego_traj(cfg.duration)
    actors = make_actors(cfg)
    sc = Scenario(cfg, [], actors, mf, ego)

    def sender_state(t: float) -> dict:
        p, yaw = sc.ego_pose(t)
        lat, lon, alt = mf.enu_to_lla(np.array([p[0], p[1], 0.0]))
        return {"agent_type": "ego", "position": {"latitude": lat, "longitude": lon, "altitude": alt},
                "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": math.degrees(yaw)}}

    def to_ego(t: float, p_map, v_map):
        pe, yaw = sc.ego_pose(t)
        return rot2(-yaw) @ (p_map - pe), rot2(-yaw) @ v_map, yaw

    events: List[Tuple[float, dict]] = []

    # ------------------------------------------------------------------ LiDAR
    last_heading = {k: a.heading0 for k, a in actors.items()}
    fps: List[dict] = []                                           # live false positives
    fp_next_id = 900
    seq = 0
    for k in range(int(cfg.duration * cfg.lidar_hz)):
        r = k / cfg.lidar_hz
        t = cfg.t0 + r
        dets = []
        for name, a in actors.items():
            pm, vm = a.traj.p(r), a.traj.v(r)
            pe_ego, ve_ego, yaw = to_ego(t, pm, vm)
            if np.hypot(*pe_ego) > cfg.lidar_range or rng.random() < cfg.lidar_miss_prob:
                continue
            if name == "vru" and sc.in_windows(t, cfg.vru_occlusions):
                continue
            if np.hypot(*vm) > 0.25:
                last_heading[name] = math.atan2(vm[1], vm[0])
            sp = cfg.lidar_pos_sigma + 0.004 * float(np.hypot(*pe_ego))
            pn = pe_ego + rng.normal(0, sp, 2)
            vn = ve_ego + rng.normal(0, cfg.lidar_vel_sigma, 2)
            h = last_heading[name] - yaw + rng.normal(0, 0.05)
            dets.append({"track_id": int(a.lidar_id(r)), "category": a.category,
                         "score": float(np.clip(rng.normal(0.82, 0.07), 0.35, 0.99)), "timestamp": t,
                         "x": float(pn[0]), "y": float(pn[1]), "z": float(a.size[2] / 2 + rng.normal(0, 0.03)),
                         "length": a.size[0], "width": a.size[1], "height": a.size[2], "heading": float(h),
                         "yaw_rate": 0.0, "vx": float(vn[0]), "vy": float(vn[1])})
        # false positives: static low-score pedestrians living ~0.6 s
        if rng.random() < cfg.lidar_fp_prob:
            fps.append({"x": rng.uniform(6, 30), "y": rng.uniform(-12, 12), "id": fp_next_id, "t_end": r + 0.6})
            fp_next_id += 1
        fps = [f for f in fps if f["t_end"] > r]
        for f in fps:
            dets.append({"track_id": f["id"], "category": "pedestrian", "score": float(rng.uniform(0.32, 0.5)),
                         "timestamp": t, "x": f["x"], "y": f["y"], "z": 0.9, "length": 0.6, "width": 0.6,
                         "height": 1.75, "heading": 0.0, "yaw_rate": 0.0, "vx": 0.0, "vy": 0.0})
        seq += 1
        msg = {"metadata": _meta(LIDAR_SENDER, "Ego_3D_ODT", seq, t + cfg.lidar_latency),
               "sender_state": sender_state(t),
               "reference_frame": {"coordinate_frame": "ego", "axes_convention": "x_forward_y_left_z_up"},
               "detections": dets}
        events.append((t + cfg.lidar_latency + abs(rng.normal(0, 0.005)), msg))

    # ------------------------------------------------------------------ UWB
    vru = actors["vru"]
    phase = rng.uniform(0.0, 1.0 / cfg.uwb_hz)
    seq = 0
    for k in range(int((cfg.duration - phase) * cfg.uwb_hz)):
        r = phase + k / cfg.uwb_hz
        t = cfg.t0 + r
        if sc.in_windows(t, cfg.uwb_dropouts):
            continue
        pm, vm = vru.traj.p(r), vru.traj.v(r)
        pe_ego, ve_ego, _ = to_ego(t, pm, vm)
        pe_ego = pe_ego + np.array([cfg.uwb_bias, 0.0]) + rng.normal(0, cfg.uwb_pos_sigma, 2)
        if rng.random() < cfg.uwb_outlier_prob:                      # multipath: a jump of a few metres
            a = rng.uniform(0, 2 * math.pi)
            pe_ego = pe_ego + rng.uniform(*cfg.uwb_outlier_range) * np.array([math.cos(a), math.sin(a)])
        ve_ego = ve_ego + rng.normal(0, cfg.uwb_vel_sigma, 2)
        seq += 1
        arrival = t + cfg.uwb_latency + abs(rng.normal(0, cfg.uwb_jitter))
        msg = {"metadata": _meta(UWB_SENDER, "Ego_2D_ODT_UWB", seq, arrival),
               "sender_state": sender_state(t),
               "reference_frame": {"coordinate_frame": "ego", "axes_convention": "x_forward_y_left_z_up"},
               "detections": [{"track_id": TAG_ID, "category": "pedestrian", "score": 0.95, "timestamp": t,
                               "x": float(pe_ego[0]), "y": float(pe_ego[1]),
                               "vx": float(ve_ego[0]), "vy": float(ve_ego[1]), "optional": {}}]}
        events.append((arrival, msg))

    events.sort(key=lambda e: e[0])
    sc.events = events
    return sc
