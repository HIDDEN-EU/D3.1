"""
UC2_fusion.py - UC2 fusion of the CTAG LiDAR detector/tracker with the DENSO UWB.

A VRU carries a UWB tag. The tag reports the VRU's position and velocity at 1-10 Hz (identity known, no
classification). The LiDAR module reports 3D boxes and track ids at its own rate (10-20 Hz). The module fuses both
into one "Ego_3D_ODT" message (ODT.json schema) at a fixed output rate:

    - the cooperative VRU (one per tag) is a Kalman-filter track fed by both sensors;
    - every other LiDAR object is passed through unchanged.

Pipeline
    push(msg)      any message, any time, any order (arrival order is not capture order)
        UWB    -> gate against the track (outlier rejection) -> Kalman update, or create the track of a new tag
        LiDAR  -> associate the frame to the tags (sticky link by LiDAR track id, then Hungarian on the
                  Mahalanobis distance of position+velocity, with ambiguity and ID-swap checks) -> Kalman update
    output(t)      predict every track to t, build the ODT message in the ego frame at t

Frames: everything is filtered in a local ENU "map" frame anchored at the first ego pose, using the WGS84 pose
carried by each message (sender_state), so ego motion between capture times is compensated. 2D only (x, y); heights
and box sizes come from the LiDAR, or class defaults when only the UWB sees the VRU. Roll/pitch are ignored.

Dependencies: numpy, scipy.
All noise / gate numbers in the config are placeholders to be calibrated on recorded data (NIS consistency).
"""

from __future__ import annotations

import bisect
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

log = logging.getLogger("UC2_fusion")

# =============================================================================
# 1. Configuration
# =============================================================================


@dataclass
class TagInfo:
    """What is known about the VRU that carries a tag (UWB tags are cooperative and self-identifying)."""
    category: str = "Pedestrian"
    size: Tuple[float, float, float] = (0.6, 0.6, 1.75)      # l, w, h used when the LiDAR does not see the VRU


DEFAULT_SIZE = {
    "Pedestrian": (0.6, 0.6, 1.75), "Cyclist": (1.8, 0.6, 1.7),
    "Motorcyclist": (2.1, 0.8, 1.6), "Scooter": (1.2, 0.6, 1.75),
}


@dataclass
class UWBSource:
    sender_id: str = "DENSO_uwb_ODT"
    pos_sigma: float = 0.30          # [m]   position std of one UWB sample (placeholder, to calibrate)
    vel_sigma: float = 0.50          # [m/s] velocity std
    has_velocity: bool = True        # False: ignore the sample's vx, vy
    trust: float = 0.95              # existence confidence of a fresh sample
    extrinsic: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, yaw of the UWB anchor in the ego frame,
    #                                                           only used if coordinate_frame == 'sensor'


@dataclass
class LidarSource:
    sender_id: str = "CTAG_lidar_ODTH"
    pos_sigma0: float = 0.10         # [m]   position std at 0 m range
    pos_sigma_per_m: float = 0.004   # [m/m] growth with range
    vel_sigma: float = 0.40          # [m/s]
    trust: float = 0.95
    min_score: float = 0.30          # LiDAR detections below are ignored
    extrinsic: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FusionConfig:
    uwb: UWBSource = field(default_factory=UWBSource)
    lidar: LidarSource = field(default_factory=LidarSource)
    tags: Dict[int, TagInfo] = field(default_factory=dict)    # tag id -> info; unknown tags get TagInfo()
    default_tag: TagInfo = field(default_factory=TagInfo)

    # motion model (constant velocity, white acceleration)
    process_acc: float = 1.2         # [m/s^2] std of the VRU acceleration (pedestrian, stop and go)
    init_vel_sigma: float = 1.0      # [m/s] velocity std when a track starts without velocity measurement
    replay_window_s: float = 1.5     # measurements older than (newest - window) can no longer be inserted

    # UWB outlier rejection (multipath)
    uwb_gate_prob: float = 0.9995    # chi-square gate (2 dof) on the position innovation
    uwb_max_rejects: int = 5         # consecutive rejected samples before the filter is restarted from the UWB
    # A LiDAR link is false if the UWB persistently disagrees with it: median NIS (2 dof, ~2 when they agree) of the UWB
    # samples of the last `span` s above `thr`, with >= `min` samples spanning >= `min_span_s`, and the last two UWB
    # samples consistent with each other (multipath outliers are not). The window grows with the UWB interval.
    link_nis_span_s: float = 1.0
    link_nis_min: int = 3
    link_nis_thr: float = 6.0
    link_nis_min_span_s: float = 0.4

    # LiDAR <-> tag association
    assoc_gate_prob: float = 0.99    # gate (4 dof, position+velocity) to create a link
    link_gate_prob: float = 0.999    # gate to keep an existing link
    assoc_max_dist_m: float = 4.0    # hard Euclidean cap on top of the gate
    ambiguity_margin: float = 3.0    # no new link if a second candidate is within this many d2 units of the best
    class_mismatch_penalty: float = 1.0   # d2 penalty when a VRU box has another VRU class than the tag's
    link_break_frames: int = 3       # consecutive frames outside the link gate before the link is dropped
    link_stale_s: float = 0.4        # the linked LiDAR id is absent for this long: the tag may link to another box
    challenge_frames: int = 2        # frames in which another box fits clearly better before the link is dropped

    # track life cycle
    uwb_stale_s: float = 2.5         # max age of a UWB sample still counted as fresh; the actual limit adapts:
    #                                  3 x the mean UWB interval observed (0.5 s at 10 Hz, this value at 1 Hz)
    lidar_lost_s: float = 0.5        # LiDAR older than this: the VRU is not seen by the LiDAR
    box_hold_s: float = 2.0          # keep the last LiDAR box size/height/heading this long after losing it
    uwb_drop_s: float = 10.0         # no UWB for this long: the track is deleted (LiDAR objects go back to pass-through)
    track_timeout_s: float = 5.0     # no measurement of any sensor for this long: the track is deleted
    score_tau_s: float = 1.5         # time constant of the score decay with the age of the last measurement

    # LiDAR pass-through
    emit_other_objects: bool = True
    lidar_max_age_s: float = 0.5     # LiDAR objects older than this vs the output time are not published
    coop_id_base: int = 1_000_000    # output id of a cooperative VRU = coop_id_base + tag id (LiDAR ids must be below)

    emit_provenance: bool = True
    ego_sender_id: str = "EGO"
    schema_version: str = "1.0"
    map_origin_lla: Optional[Tuple[float, float, float]] = None   # None -> first ego pose


# =============================================================================
# 2. Geometry (WGS84 <-> ENU, 2D poses)
# =============================================================================

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def lla_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    return np.array([(n + alt) * math.cos(lat) * math.cos(lon),
                     (n + alt) * math.cos(lat) * math.sin(lon),
                     (n * (1 - _WGS84_E2) + alt) * math.sin(lat)])


def ecef_to_lla(p: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = p
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1 - _WGS84_E2))
    for _ in range(5):
        n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
        alt = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1 - _WGS84_E2 * n / (n + alt)))
    n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(lon), r / math.cos(lat) - n


class MapFrame:
    """Local ENU tangent frame anchored at a fixed WGS84 origin for the session."""

    def __init__(self, origin_lla: Tuple[float, float, float]):
        self.origin_lla = origin_lla
        self.origin_ecef = lla_to_ecef(*origin_lla)
        lat, lon = math.radians(origin_lla[0]), math.radians(origin_lla[1])
        sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
        self.R = np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])

    def lla_to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        return self.R @ (lla_to_ecef(lat, lon, alt) - self.origin_ecef)

    def enu_to_lla(self, p: np.ndarray) -> Tuple[float, float, float]:
        return ecef_to_lla(self.R.T @ p + self.origin_ecef)


@dataclass
class Pose:
    """Ego pose in the map frame: ENU position (3) and yaw (rad, counter-clockwise from East)."""
    t: np.ndarray
    yaw: float

    @staticmethod
    def from_sender_state(ss: dict, mf: MapFrame) -> "Pose":
        p, o = ss["position"], ss["orientation"]
        return Pose(mf.lla_to_enu(float(p["latitude"]), float(p["longitude"]), float(p["altitude"])),
                    math.radians(float(o["yaw"])))

    def pos_to_map(self, p: np.ndarray) -> np.ndarray:
        return self.t[:2] + rot2(self.yaw) @ p

    def vec_to_map(self, v: np.ndarray) -> np.ndarray:
        return rot2(self.yaw) @ v

    def pos_to_ego(self, p: np.ndarray) -> np.ndarray:
        return rot2(-self.yaw) @ (p - self.t[:2])

    def vec_to_ego(self, v: np.ndarray) -> np.ndarray:
        return rot2(-self.yaw) @ v


class PoseBuffer:
    """Ego poses over time: interpolation, and a short extrapolation with the last ego velocity."""

    def __init__(self, keep_s: float = 10.0, max_extrap_s: float = 0.5):
        self.keep_s, self.max_extrap_s = keep_s, max_extrap_s
        self._t: List[float] = []
        self._p: List[Pose] = []

    def add(self, t: float, pose: Pose) -> None:
        i = bisect.bisect_left(self._t, t)
        if i < len(self._t) and abs(self._t[i] - t) < 1e-9:
            self._p[i] = pose
        else:
            self._t.insert(i, t)
            self._p.insert(i, pose)
        while len(self._t) > 2 and self._t[-1] - self._t[0] > self.keep_s:
            self._t.pop(0)
            self._p.pop(0)

    @staticmethod
    def _blend(a: Pose, b: Pose, w: float) -> Pose:
        return Pose(a.t + (b.t - a.t) * w, a.yaw + wrap_pi(b.yaw - a.yaw) * w)

    def at(self, t: float) -> Optional[Pose]:
        n = len(self._t)
        if n == 0:
            return None
        i = bisect.bisect_left(self._t, t)
        if i == 0:
            return self._p[0]
        if i == n:
            if n >= 2 and self._t[-1] > self._t[-2]:
                w = min(t - self._t[-1], self.max_extrap_s) / (self._t[-1] - self._t[-2])
                a, b = self._p[-2], self._p[-1]
                return Pose(b.t + (b.t - a.t) * w, b.yaw + wrap_pi(b.yaw - a.yaw) * w)
            return self._p[-1]
        w = (t - self._t[i - 1]) / max(self._t[i] - self._t[i - 1], 1e-9)
        return self._blend(self._p[i - 1], self._p[i], w)


# =============================================================================
# 3. Kalman filter with out-of-sequence measurements
# =============================================================================


@dataclass
class Meas:
    """A measurement of the VRU state [x, y, vx, vy] in the map frame at its capture time."""
    t: float
    kind: str                        # 'uwb' | 'lidar'
    z: np.ndarray                    # (4,)
    R: np.ndarray                    # (4, 4)
    has_vel: bool = True


class VRUFilter:
    """
    Constant-velocity Kalman filter, state [x, y, vx, vy], that accepts measurements in any order.

    The state is kept at the time of the newest measurement. The measurements of the last `window_s` seconds are
    stored together with a snapshot of the state before the oldest of them; a measurement that arrives late is
    inserted in time order and the stored ones are replayed from the snapshot (a handful of 4x4 updates).
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, t0: float, q_acc: float, window_s: float):
        self.q, self.window = q_acc ** 2, window_s
        self.t, self.x, self.P = t0, x0.copy(), P0.copy()
        self._snap = (t0, x0.copy(), P0.copy())
        self._hist: List[Meas] = []

    @staticmethod
    def _predict(x, P, dt, q):
        if dt <= 0.0:
            return x, P
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        d4, d3, d2 = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
        Q = q * np.array([[d4, 0, d3, 0], [0, d4, 0, d3], [d3, 0, d2, 0], [0, d3, 0, d2]])
        return F @ x, F @ P @ F.T + Q

    @staticmethod
    def _hz(m: Meas):
        if m.has_vel:
            return np.eye(4), m.z, m.R
        return np.eye(4)[:2], m.z[:2], m.R[:2, :2]

    @classmethod
    def _update(cls, x, P, m: Meas):
        H, z, R = cls._hz(m)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        IKH = np.eye(4) - K @ H
        return x + K @ (z - H @ x), IKH @ P @ IKH.T + K @ R @ K.T          # Joseph form

    def state_at(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """State estimate at time t (earlier than the newest measurement: replayed up to t, then predicted)."""
        if t >= self.t:
            return self._predict(self.x, self.P, t - self.t, self.q)
        tt, x, P = self._snap
        for m in self._hist:
            if m.t > t:
                break
            x, P = self._update(*self._predict(x, P, m.t - tt, self.q), m)
            tt = m.t
        return self._predict(x, P, t - tt, self.q)

    def d2(self, m: Meas, pos_only: bool = False) -> float:
        """Squared Mahalanobis distance between a measurement and the state predicted at its time."""
        x, P = self.state_at(m.t)
        if pos_only or not m.has_vel:
            H, z, R = np.eye(4)[:2], m.z[:2], m.R[:2, :2]
        else:
            H, z, R = np.eye(4), m.z, m.R
        y = z - H @ x
        return float(y @ np.linalg.solve(H @ P @ H.T + R, y))

    def add(self, m: Meas) -> bool:
        """Insert a measurement; False if it is older than the stored history."""
        if m.t < self._snap[0] - 1e-9:
            return False
        i = bisect.bisect_right([h.t for h in self._hist], m.t)
        self._hist.insert(i, m)
        if i == len(self._hist) - 1 and m.t >= self.t:
            self.x, self.P = self._update(*self._predict(self.x, self.P, m.t - self.t, self.q), m)
            self.t = m.t
        else:                                                              # out of sequence: replay
            tt, x, P = self._snap
            for h in self._hist:
                x, P = self._update(*self._predict(x, P, h.t - tt, self.q), h)
                tt = h.t
            self.t, self.x, self.P = tt, x, P
        while self._hist and self._hist[0].t < self.t - self.window:       # advance the snapshot
            h = self._hist.pop(0)
            tt, x, P = self._snap
            x, P = self._update(*self._predict(x, P, h.t - tt, self.q), h)
            self._snap = (h.t, x, P)
        return True


# =============================================================================
# 4. Internal objects
# =============================================================================

CANONICAL = {
    "car": "Car", "vehicle": "Car", "van": "Car", "truck": "Truck", "bus": "Bus", "trailer": "Truck",
    "construction_vehicle": "Truck", "pedestrian": "Pedestrian", "person": "Pedestrian", "rider": "Cyclist",
    "cyclist": "Cyclist", "bicycle": "Cyclist", "bike": "Cyclist", "motorcyclist": "Motorcyclist",
    "motorcycle": "Motorcyclist", "scooter": "Scooter", "e-scooter": "Scooter", "escooter": "Scooter",
    "other": "Unknown", "unknown": "Unknown",
}
VRU_CLASSES = {"Pedestrian", "Cyclist", "Motorcyclist", "Scooter"}


def canonical_category(raw) -> str:
    return CANONICAL.get(str(raw).strip().lower(), str(raw))


@dataclass
class LidarObj:
    key: int                         # lidar track id, or a negative number if the box is untracked
    id: Optional[int]
    category: str
    score: float
    t: float                         # capture time
    pos: np.ndarray                  # (2,) map
    vel: np.ndarray                  # (2,) map
    has_vel: bool
    heading: Optional[float]         # map frame
    size: Optional[Tuple[float, float, float]]
    z: Optional[float]               # ego-frame box centre height
    yaw_rate: float
    R: np.ndarray                    # (4, 4) measurement covariance

    def meas(self) -> Meas:
        return Meas(self.t, "lidar", np.array([self.pos[0], self.pos[1], self.vel[0], self.vel[1]]),
                    self.R, self.has_vel)


@dataclass
class LidarFrame:
    t: float
    objs: List[LidarObj]
    claimed: Dict[int, int] = field(default_factory=dict)      # obj key -> tag id fused with it


@dataclass
class CoopTrack:
    tag: int
    info: TagInfo
    filt: VRUFilter
    last_uwb_t: float
    last_lidar_t: float = -math.inf
    last_lidar: Optional[LidarObj] = None
    last_heading: Optional[float] = None
    link_id: Optional[int] = None
    link_miss: int = 0
    challenge: int = 0
    consec_rejects: int = 0
    n_uwb: int = 1
    n_lidar: int = 0
    n_uwb_rejected: int = 0
    ambiguous: bool = False
    nis: deque = field(default_factory=lambda: deque(maxlen=20))     # (time, NIS) of recent UWB samples
    prev_uwb: Optional[Meas] = None                                  # previous UWB sample (rejected or not)
    uwb_dt: Optional[float] = None                                   # mean interval between accepted UWB samples


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# =============================================================================
# 5. The fusion
# =============================================================================


class UC2Fusion:
    """
    Usage (event driven):
        fusion = UC2Fusion()
        for msg in incoming:                   # dicts (parsed JSON), in arrival order, from both sensors
            fusion.push(msg)
        out = fusion.output(t_out)             # call at the output rate (e.g. 10 Hz), t_out = Unix seconds
    or, per cycle:  out = fusion.fuse([lidar_msg, uwb_msg, ...], t_out)
    """

    def __init__(self, cfg: Optional[FusionConfig] = None):
        self.cfg = cfg or FusionConfig()
        self.map = MapFrame(self.cfg.map_origin_lla) if self.cfg.map_origin_lla else None
        self.poses = PoseBuffer()
        self.tracks: Dict[int, CoopTrack] = {}
        self.frame: Optional[LidarFrame] = None
        self.seq = 0
        self.stats: Dict[str, int] = {}
        c = self.cfg
        self._gate_uwb = float(chi2.ppf(c.uwb_gate_prob, df=2))
        self._gate_assoc = float(chi2.ppf(c.assoc_gate_prob, df=4))
        self._gate_link = float(chi2.ppf(c.link_gate_prob, df=4))

    # ------------------------------------------------------------------ helpers
    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    @staticmethod
    def _msg_time(msg: dict) -> float:
        ts = [_f(d.get("timestamp")) for d in msg.get("detections") or []]
        ts = [t for t in ts if t is not None]
        return max(ts) if ts else _f(msg.get("metadata", {}).get("send_timestamp"), 0.0)

    def _pose_of(self, msg: dict, t: float) -> Optional[Pose]:
        """Ego pose of a message: its own sender_state, else interpolated from the other messages."""
        ss = msg.get("sender_state")
        if ss:
            if self.map is None:
                p = ss["position"]
                self.map = MapFrame((float(p["latitude"]), float(p["longitude"]), float(p["altitude"])))
            pose = Pose.from_sender_state(ss, self.map)
            self.poses.add(t, pose)
            return pose
        return self.poses.at(t)

    @staticmethod
    def _to_ego_frame(msg: dict, src, x: float, y: float, vx: float, vy: float, heading: Optional[float]):
        """Numbers of a detection -> ego frame (identity if they already are)."""
        if msg.get("reference_frame", {}).get("coordinate_frame", "ego") != "sensor":
            return np.array([x, y]), np.array([vx, vy]), heading
        ex, ey, eyaw = src.extrinsic
        R = rot2(eyaw)
        return R @ np.array([x, y]) + np.array([ex, ey]), R @ np.array([vx, vy]), \
            (wrap_pi(heading + eyaw) if heading is not None else None)

    def _tag_info(self, tag: int) -> TagInfo:
        return self.cfg.tags.get(tag, self.cfg.default_tag)

    # ------------------------------------------------------------------ input
    def push(self, msg: dict) -> None:
        """Process one message from the UWB or the LiDAR module."""
        if not msg:
            return
        sid = msg.get("metadata", {}).get("sender_id")
        try:
            if sid == self.cfg.uwb.sender_id:
                self._on_uwb(msg)
            elif sid == self.cfg.lidar.sender_id:
                self._on_lidar(msg)
            else:
                log.warning("message from unknown sender_id %r ignored", sid)
                self._count("unknown_sender")
        except (KeyError, TypeError, ValueError) as e:
            log.error("message from %s rejected: %s", sid, e)
            self._count("rejected_msgs")

    # .................................................................. UWB
    def _on_uwb(self, msg: dict) -> None:
        cfg, u = self.cfg, self.cfg.uwb
        t_msg = self._msg_time(msg)
        pose = self._pose_of(msg, t_msg)
        if pose is None:
            self._count("uwb_no_pose")
            return
        for d in msg.get("detections") or []:
            tag = d.get("track_id")
            if tag is None:
                log.warning("UWB detection without tag id ignored")
                continue
            tag = int(tag)
            opt = d.get("optional") or {}
            vx, vy = _f(d.get("vx", opt.get("vx")), 0.0), _f(d.get("vy", opt.get("vy")), 0.0)
            p_ego, v_ego, _ = self._to_ego_frame(msg, u, float(d["x"]), float(d["y"]), vx, vy, None)
            t = float(d["timestamp"])
            pe = self._pose_of(msg, t) if msg.get("sender_state") is None else pose
            if pe is None:
                continue
            m = Meas(t, "uwb", np.concatenate([pe.pos_to_map(p_ego), pe.vec_to_map(v_ego)]),
                     np.diag([u.pos_sigma ** 2, u.pos_sigma ** 2, u.vel_sigma ** 2, u.vel_sigma ** 2]),
                     u.has_velocity)
            tr = self.tracks.get(tag)
            if tr is None:
                self.tracks[tag] = self._new_track(tag, m)
                continue
            nis = tr.filt.d2(m, pos_only=True)
            tr.nis.append((t, min(nis, 50.0)))
            span = max(cfg.link_nis_span_s, 1.5 * cfg.link_nis_min * (tr.uwb_dt or 0.0))   # 3 samples fit at 1 Hz
            recent = [(tt, v) for tt, v in tr.nis if tt >= t - span]
            long_enough = len(recent) >= cfg.link_nis_min and recent[-1][0] - recent[0][0] >= cfg.link_nis_min_span_s
            consistent = tr.prev_uwb is not None and self._uwb_consistent(tr.prev_uwb, m)
            tr.prev_uwb = m
            # persistent disagreement = high NIS AND the last two samples agree with each other (multipath
            # outliers jump in random directions; a false link makes every sample disagree in the same way)
            if tr.link_id is not None and long_enough and consistent and \
                    float(np.median([v for _, v in recent])) > cfg.link_nis_thr:             # the UWB says the linked box is not it
                log.warning("tag %d: UWB disagrees with the LiDAR link %s (median NIS %.1f): link dropped, "
                            "filter restarted", tag, tr.link_id, float(np.median([v for _, v in recent])))
                self._count("links_uwb_disagree")
                self.tracks[tag] = self._new_track(tag, m)
                continue
            if nis > self._gate_uwb:                                        # multipath outlier
                tr.consec_rejects += 1
                tr.n_uwb_rejected += 1
                self._count("uwb_outliers")
                if tr.consec_rejects >= cfg.uwb_max_rejects:                # the filter is lost: restart on the UWB
                    log.warning("tag %d: %d UWB samples rejected in a row, filter restarted", tag, tr.consec_rejects)
                    self.tracks[tag] = self._new_track(tag, m)
                continue
            tr.consec_rejects = 0
            if tr.filt.add(m):
                if t > tr.last_uwb_t:
                    dt = t - tr.last_uwb_t
                    tr.uwb_dt = dt if tr.uwb_dt is None else 0.7 * tr.uwb_dt + 0.3 * min(dt, 3.0 * tr.uwb_dt)
                tr.last_uwb_t = max(tr.last_uwb_t, t)
                tr.n_uwb += 1
            else:
                self._count("uwb_too_old")

    def _uwb_consistent(self, a: Meas, b: Meas) -> bool:
        """Do two consecutive UWB samples describe the same motion (position difference vs velocity * dt)?"""
        dt = b.t - a.t
        if dt <= 0.0:
            return False
        u = self.cfg.uwb
        move = 0.5 * (a.z[2:] + b.z[2:]) * dt if u.has_velocity else np.zeros(2)
        var = 2 * u.pos_sigma ** 2 + self.cfg.process_acc ** 2 * dt ** 4 / 4.0 + \
            ((u.vel_sigma * dt) ** 2 / 2.0 if u.has_velocity else (2.0 * dt) ** 2)
        return float(np.sum((b.z[:2] - a.z[:2] - move) ** 2) / var) < 9.2

    def _new_track(self, tag: int, m: Meas) -> CoopTrack:
        P0 = m.R.copy()
        if not m.has_vel:
            P0[2, 2] = P0[3, 3] = self.cfg.init_vel_sigma ** 2
        x0 = m.z.copy()
        if not m.has_vel:
            x0[2:] = 0.0
        return CoopTrack(tag, self._tag_info(tag), VRUFilter(x0, P0, m.t, self.cfg.process_acc,
                                                              self.cfg.replay_window_s), last_uwb_t=m.t,
                         nis=deque(maxlen=20), prev_uwb=m)

    # .................................................................. LiDAR
    def _on_lidar(self, msg: dict) -> None:
        cfg, l = self.cfg, self.cfg.lidar
        t_msg = self._msg_time(msg)
        pose = self._pose_of(msg, t_msg)
        if pose is None:
            self._count("lidar_no_pose")
            return
        objs: List[LidarObj] = []
        for i, d in enumerate(msg.get("detections") or []):
            score = float(d["score"])
            if score < l.min_score:
                continue
            h = _f(d.get("heading"))
            p_ego, v_ego, h = self._to_ego_frame(msg, l, float(d["x"]), float(d["y"]),
                                                 _f(d.get("vx"), 0.0), _f(d.get("vy"), 0.0), h)
            sp = l.pos_sigma0 + l.pos_sigma_per_m * float(np.hypot(*p_ego))
            lw, wd, ht = _f(d.get("length")), _f(d.get("width")), _f(d.get("height"))
            tid = d.get("track_id")
            objs.append(LidarObj(
                key=int(tid) if tid is not None else -(i + 1), id=int(tid) if tid is not None else None,
                category=canonical_category(d.get("category", "Unknown")), score=score, t=float(d["timestamp"]),
                pos=pose.pos_to_map(p_ego), vel=pose.vec_to_map(v_ego),
                has_vel=d.get("vx") is not None and d.get("vy") is not None,
                heading=wrap_pi(h + pose.yaw) if h is not None else None,
                size=(lw, wd, ht) if None not in (lw, wd, ht) else None, z=_f(d.get("z")),
                yaw_rate=_f(d.get("yaw_rate"), 0.0),
                R=np.diag([sp ** 2, sp ** 2, l.vel_sigma ** 2, l.vel_sigma ** 2])))
        frame = LidarFrame(t_msg, objs)
        if self.frame is None or t_msg >= self.frame.t:
            self.frame = frame
        self._associate(frame)

    def _penalty(self, tr: CoopTrack, o: LidarObj) -> Optional[float]:
        """Class compatibility between a tag and a LiDAR box: None = never, else a d2 penalty."""
        if o.category == tr.info.category:
            return 0.0
        if o.category in VRU_CLASSES or o.category == "Unknown":
            return self.cfg.class_mismatch_penalty
        return None

    def _associate(self, frame: LidarFrame) -> None:
        cfg = self.cfg
        tracks = list(self.tracks.values())
        if not tracks or not frame.objs:
            return
        objs = frame.objs
        d2 = np.full((len(tracks), len(objs)), np.inf)                     # raw Mahalanobis distances
        pen = np.full_like(d2, np.inf)
        for i, tr in enumerate(tracks):
            for j, o in enumerate(objs):
                p = self._penalty(tr, o)
                if p is None:
                    continue
                pen[i, j] = p
                d2[i, j] = tr.filt.d2(o.meas())
        matched: Dict[int, int] = {}                                       # track index -> object index
        taken: set = set()
        key_idx = {o.key: j for j, o in enumerate(objs)}
        owned = {tr.link_id for tr in tracks if tr.link_id is not None}

        # (1) sticky links: the box with the LiDAR track id the tag is already linked to
        for i, tr in enumerate(tracks):
            tr.ambiguous = False
            j = key_idx.get(tr.link_id) if tr.link_id is not None else None
            if j is None:
                continue
            if d2[i, j] <= self._gate_link:
                others = [d2[i, k] for k, ok in enumerate(objs)
                          if k != j and ok.key not in owned and np.isfinite(d2[i, k])]
                if others and min(others) + cfg.ambiguity_margin < d2[i, j]:   # another box fits much better:
                    tr.challenge += 1                                           # identity swap?
                else:
                    tr.challenge = 0
                if tr.challenge >= cfg.challenge_frames:
                    log.info("tag %d: link to lidar id %s dropped (a better box exists)", tr.tag, tr.link_id)
                    self._count("links_challenged")
                    tr.link_id, tr.challenge, tr.link_miss = None, 0, 0
                else:
                    matched[i] = j
                    taken.add(j)
            else:
                tr.link_miss += 1
                if tr.link_miss >= cfg.link_break_frames:
                    log.info("tag %d: link to lidar id %s lost (outside the gate)", tr.tag, tr.link_id)
                    self._count("links_lost")
                    tr.link_id, tr.link_miss, tr.challenge = None, 0, 0

        # (2) new links: Hungarian on the free tags x free boxes, skipping ambiguous pairs
        stale = lambda tr: tr.link_id is not None and tr.link_id not in key_idx \
            and frame.t - tr.last_lidar_t > cfg.link_stale_s                  # linked id vanished (occlusion, id change)
        free_t = [i for i in range(len(tracks)) if i not in matched and (tracks[i].link_id is None or stale(tracks[i]))]
        owned = {tr.link_id for tr in tracks if tr.link_id is not None and not stale(tr)}
        free_o = [j for j in range(len(objs)) if j not in taken and objs[j].key not in owned]
        if free_t and free_o:
            BIG = 1e6
            cost = np.full((len(free_t), len(free_o)), BIG)
            for a, i in enumerate(free_t):
                for b, j in enumerate(free_o):
                    if d2[i, j] <= self._gate_assoc and \
                            np.linalg.norm(objs[j].pos - self._pos_at(tracks[i], objs[j].t)) <= cfg.assoc_max_dist_m:
                        cost[a, b] = d2[i, j] + pen[i, j]
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] >= BIG:
                    continue
                i, j = free_t[a], free_o[b]
                row = sorted(d2[i, jj] for jj in free_o if jj != j and d2[i, jj] <= self._gate_assoc)
                col = sorted(d2[ii, j] for ii in free_t if ii != i and d2[ii, j] <= self._gate_assoc)
                best = d2[i, j]
                if (row and row[0] - best < cfg.ambiguity_margin) or (col and col[0] - best < cfg.ambiguity_margin):
                    tracks[i].ambiguous = True
                    self._count("assoc_ambiguous")
                    continue
                matched[i] = j
                taken.add(j)

        # (3) updates
        for i, j in matched.items():
            tr, o = tracks[i], objs[j]
            if tr.filt.add(o.meas()):
                tr.last_lidar_t = max(tr.last_lidar_t, o.t)
                tr.last_lidar = o
                tr.n_lidar += 1
                tr.link_id, tr.link_miss = o.id, 0
                if o.heading is not None:
                    tr.last_heading = o.heading
                frame.claimed[o.key] = tr.tag
            else:
                self._count("lidar_too_old")

    def _pos_at(self, tr: CoopTrack, t: float) -> np.ndarray:
        return tr.filt.state_at(t)[0][:2]

    # ------------------------------------------------------------------ output
    def output(self, t_out: float) -> dict:
        """ODT message in the ego frame at t_out with the cooperative VRUs and the other LiDAR objects."""
        cfg = self.cfg
        pose = self.poses.at(t_out)
        for tag, tr in list(self.tracks.items()):                          # life cycle
            last_any = max(tr.last_uwb_t, tr.last_lidar_t)
            if t_out - tr.last_uwb_t > cfg.uwb_drop_s or t_out - last_any > cfg.track_timeout_s:
                del self.tracks[tag]
        dets: List[dict] = []
        if pose is not None:
            for tag in sorted(self.tracks):
                dets.append(self._coop_detection(self.tracks[tag], pose, t_out))
            if cfg.emit_other_objects and self.frame is not None:
                for o in self.frame.objs:
                    if self.frame.claimed.get(o.key) in self.tracks or t_out - o.t > cfg.lidar_max_age_s:
                        continue
                    dets.append(self._passthrough_detection(o, pose, t_out))
        return self._message(dets, pose, t_out)

    def _coop_detection(self, tr: CoopTrack, pose: Pose, t_out: float) -> dict:
        cfg = self.cfg
        x, P = tr.filt.state_at(t_out)
        age_uwb = max(0.0, t_out - tr.last_uwb_t)
        age_lidar = max(0.0, t_out - tr.last_lidar_t)
        stale_s = cfg.uwb_stale_s if tr.uwb_dt is None else min(cfg.uwb_stale_s, max(0.5, 3.0 * tr.uwb_dt))
        uwb_fresh, lidar_fresh = age_uwb <= stale_s, age_lidar <= cfg.lidar_lost_s
        status = "fused" if uwb_fresh and lidar_fresh else "lidar_only" if lidar_fresh else \
            "uwb_only" if uwb_fresh else "coasting"
        lo = tr.last_lidar if age_lidar <= cfg.box_hold_s else None        # last LiDAR box, if recent enough
        size = lo.size if lo is not None and lo.size is not None else \
            DEFAULT_SIZE.get(tr.info.category, tr.info.size)
        speed = float(np.hypot(x[2], x[3]))
        if speed > 0.3:
            heading = math.atan2(x[3], x[2])                               # moving: along the velocity
        elif lo is not None and lo.heading is not None:
            heading = lo.heading
        else:
            heading = tr.last_heading if tr.last_heading is not None else math.atan2(x[3], x[2]) if speed > 0.05 else 0.0
        p_ego, v_ego = pose.pos_to_ego(x[:2]), pose.vec_to_ego(x[2:])
        z = lo.z if lo is not None and lo.z is not None else size[2] / 2.0
        s_uwb = cfg.uwb.trust * math.exp(-age_uwb / cfg.score_tau_s)
        s_lid = (tr.last_lidar.score * cfg.lidar.trust if tr.last_lidar is not None else 0.0) \
            * math.exp(-age_lidar / cfg.score_tau_s)
        det = {
            "track_id": cfg.coop_id_base + tr.tag, "category": tr.info.category,
            "score": round(float(1.0 - (1.0 - s_uwb) * (1.0 - s_lid)), 4), "timestamp": t_out,
            "x": round(float(p_ego[0]), 3), "y": round(float(p_ego[1]), 3), "z": round(float(z), 3),
            "length": round(float(size[0]), 3), "width": round(float(size[1]), 3), "height": round(float(size[2]), 3),
            "heading": round(wrap_pi(heading - pose.yaw), 4), "yaw_rate": 0.0,
            "vx": round(float(v_ego[0]), 3), "vy": round(float(v_ego[1]), 3),
            "optional": {"v": round(float(np.linalg.norm(v_ego)), 3)},
        }
        if cfg.emit_provenance:
            src = []
            if tr.last_uwb_t > -math.inf:
                src.append({"sender_id": cfg.uwb.sender_id, "source_track_id": tr.tag, "score": round(s_uwb, 4),
                            "capture_timestamp": tr.last_uwb_t})
            if tr.last_lidar is not None:
                src.append({"sender_id": cfg.lidar.sender_id, "source_track_id": tr.last_lidar.id,
                            "score": round(tr.last_lidar.score, 4), "capture_timestamp": tr.last_lidar_t})
            det["sources"] = src
            det["fusion_info"] = {
                "cooperative": True, "tag_id": tr.tag, "status": status,
                "lidar_track_id": tr.link_id if lidar_fresh else None,
                "age_uwb_s": round(age_uwb, 3), "age_lidar_s": round(age_lidar, 3) if age_lidar < 1e8 else None,
                "pos_std_m": round(float(math.sqrt(max(np.linalg.eigvalsh(P[:2, :2])))), 3),
                "size_measured": lo is not None and lo.size is not None,
                "link_ambiguous": tr.ambiguous,
            }
        return det

    def _passthrough_detection(self, o: LidarObj, pose: Pose, t_out: float) -> dict:
        dt = t_out - o.t
        p, v = o.pos + o.vel * dt, o.vel
        p_ego, v_ego = pose.pos_to_ego(p), pose.vec_to_ego(v)
        size = o.size or (1.0, 1.0, 1.5)
        det = {
            "track_id": o.id, "category": o.category, "score": round(o.score, 4), "timestamp": t_out,
            "x": round(float(p_ego[0]), 3), "y": round(float(p_ego[1]), 3),
            "z": round(float(o.z if o.z is not None else size[2] / 2.0), 3),
            "length": round(float(size[0]), 3), "width": round(float(size[1]), 3), "height": round(float(size[2]), 3),
            "heading": round(wrap_pi((o.heading if o.heading is not None else pose.yaw) - pose.yaw), 4),
            "yaw_rate": round(o.yaw_rate, 4), "vx": round(float(v_ego[0]), 3), "vy": round(float(v_ego[1]), 3),
            "optional": {"v": round(float(np.linalg.norm(v_ego)), 3)},
        }
        if self.cfg.emit_provenance:
            det["sources"] = [{"sender_id": self.cfg.lidar.sender_id, "source_track_id": o.id,
                               "score": round(o.score, 4), "capture_timestamp": o.t}]
            det["fusion_info"] = {"cooperative": False}
        return det

    def _message(self, dets: List[dict], pose: Optional[Pose], t: float) -> dict:
        self.seq += 1
        if pose is not None and self.map is not None:
            lla = self.map.enu_to_lla(pose.t)
            yaw = math.degrees(pose.yaw)
        else:
            lla, yaw = (0.0, 0.0, 0.0), 0.0
        return {
            "metadata": {"schema_version": self.cfg.schema_version, "message_type": "Ego_3D_ODT",
                         "message_id": str(uuid.uuid4()), "sender_id": self.cfg.ego_sender_id,
                         "sequence_number": self.seq, "send_timestamp": time.time(), "time_reference": "UTC"},
            "sender_state": {"agent_type": "ego",
                             "position": dict(zip(("latitude", "longitude", "altitude"), lla)),
                             "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": yaw}},
            "reference_frame": {"coordinate_frame": "ego", "axes_convention": "x_forward_y_left_z_up"},
            "detections": dets,
        }

    # ------------------------------------------------------------------ convenience
    def fuse(self, messages: Iterable[dict], t_out: Optional[float] = None) -> dict:
        """Push a batch of messages (in capture-time order) and publish at t_out (default: newest capture)."""
        msgs = [m for m in messages if m]
        for m in sorted(msgs, key=self._msg_time):
            self.push(m)
        if t_out is None:
            t_out = max((self._msg_time(m) for m in msgs), default=time.time())
        return self.output(t_out)
