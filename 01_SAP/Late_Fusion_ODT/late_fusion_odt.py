"""
late_fusion_odt.py — UPV late-fusion of object detections & tracks (HIDDEN, T3.x)

Fuses the three single-agent ODT streams of the ego vehicle into one
"Ego 3D ODT" message (ODT.json schema):

    DENSO_uwb_ODT        -> 2D_ODT_UWB.json  (x, y, vx, vy; z/size/heading optional)
    UPV_cam_lidar_ODTH   -> ODT.json         (full 3D boxes)
    CTAG_lidar_ODTH      -> ODT.json         (full 3D boxes)

Pipeline (one call to LateFusionODT.fuse() per output cycle):
    1. parse + validate   each message -> internal Detection objects
    2. spatial alignment  ego/sensor/world frame -> common local ENU "map" frame
                          (uses sender_state WGS84 pose, so ego motion between
                          the capture times of the three sensors is compensated)
    3. temporal alignment propagate every detection to the fusion time
                          (CTRV if yaw_rate known, else constant velocity),
                          inflating its covariance with the time gap
    4. association        source-by-source Hungarian assignment on Mahalanobis
                          distance with chi-square gating + class compatibility
    5. state fusion       information-weighted position/velocity, circular mean
                          heading (3D sources only), size from best 3D source,
                          noisy-OR existence score, weighted class vote
    6. track management   stable fused track_id across cycles (source-track
                          links first, then spatial matching to predicted tracks)
    7. output             ODT.json-compliant dict in the ego frame at fusion time,
                          plus an optional per-object "sources" provenance block

Dependencies: numpy, scipy (scipy only for the Hungarian solver; a greedy
fallback is used if it is missing).

All noise / trust numbers in DEFAULT_SOURCES are PLACEHOLDERS — calibrate them
per sensor on recorded data with ground truth (e.g. NIS / NEES consistency).
"""

from __future__ import annotations

import math
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import chi2 as _chi2
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

log = logging.getLogger("late_fusion_odt")

# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class SourceConfig:
    """Per-source characteristics. Keyed by metadata.sender_id."""
    name: str                         # human-readable, written into provenance
    kind: str                         # 'uwb_2d' | 'lidar_3d' | 'camlidar_3d'
    pos_sigma0: float                 # [m]   position std at 0 m range
    pos_sigma_per_m: float            # [m/m] std growth with range from ego
    vel_sigma: float                  # [m/s] velocity std
    trust: float = 1.0                # [0,1] multiplies score in existence fusion
    min_score: float = 0.3            # detections below are discarded on input
    priority: int = 0                 # association order (lower = first / anchor)
    has_3d_box: bool = True           # provides size + heading
    # Fixed sensor->ego extrinsic, only needed if the source publishes in
    # coordinate_frame == 'sensor'. (x, y, z, roll, pitch, yaw) in m / rad.
    extrinsic: Optional[Tuple[float, float, float, float, float, float]] = None
    # Category remapping specific to this source (applied before canonicalisation)
    category_map: Dict[str, str] = field(default_factory=dict)


# sender_id values are placeholders — replace with the real IDs agreed in the project.
DEFAULT_SOURCES: Dict[str, SourceConfig] = {
    "CTAG_lidar_ODTH": SourceConfig(
        name="CTAG_lidar_ODTH", kind="lidar_3d",
        pos_sigma0=0.10, pos_sigma_per_m=0.004, vel_sigma=0.4,
        trust=0.95, min_score=0.30, priority=0, has_3d_box=True),
    "UPV_cam_lidar_ODTH": SourceConfig(
        name="UPV_cam_lidar_ODTH", kind="camlidar_3d",
        pos_sigma0=0.15, pos_sigma_per_m=0.006, vel_sigma=0.5,
        trust=0.95, min_score=0.30, priority=1, has_3d_box=True),
    "DENSO_uwb_ODT": SourceConfig(
        name="DENSO_uwb_ODT", kind="uwb_2d",
        pos_sigma0=0.30, pos_sigma_per_m=0.010, vel_sigma=0.6,
        # A UWB tag is a cooperative, self-identifying device: when present it is
        # almost certainly a real object, even if occluded for lidar/camera.
        trust=0.90, min_score=0.20, priority=2, has_3d_box=False),
}


@dataclass
class FusionConfig:
    sources: Dict[str, SourceConfig] = field(default_factory=lambda: dict(DEFAULT_SOURCES))
    max_latency_s: float = 0.30        # drop detections older than this vs fusion time
    gate_prob: float = 0.99            # chi-square gate on 2D Mahalanobis distance
    max_gate_dist_m: float = 4.0       # hard Euclidean cap on top of the gate
    process_noise_acc: float = 2.0     # [m/s^2] for covariance inflation when propagating
    min_output_score: float = 0.35     # fused objects below this are not published
    single_source_penalty: float = 1.0 # multiply score of objects seen by one source only (1.0 = off)
    track_max_age_s: float = 0.5       # delete fused track if unseen for this long
    track_gate_m: float = 2.5          # spatial gate for re-associating to previous tracks
    state_min_rel_conf: float = 0.5    # member contributes to kinematics only if conf >= this * best conf
    emit_provenance: bool = True       # add "sources" list per detection (schema extension)
    ego_sender_id: str = "EGO"         # sender_id written into the output message
    schema_version: str = "1.0"
    # Map frame origin (WGS84). None -> first ego pose seen.
    map_origin_lla: Optional[Tuple[float, float, float]] = None


# =============================================================================
# 2. Categories
# =============================================================================

# Canonical output categories (what the ODT.json consumers see).
CANONICAL = {
    "car": "Car", "vehicle": "Car", "van": "Car", "suv": "Car",
    "truck": "Truck", "bus": "Bus", "trailer": "Truck",
    "pedestrian": "Pedestrian", "person": "Pedestrian", "rider": "Cyclist",
    "cyclist": "Cyclist", "bicycle": "Cyclist", "bike": "Cyclist",
    "motorcyclist": "Motorcyclist", "motorcycle": "Motorcyclist", "motor": "Motorcyclist",
    "scooter": "Scooter", "e-scooter": "Scooter", "escooter": "Scooter",
    "micromobility": "Scooter", "micro-mobility": "Scooter",
    "other": "Unknown", "unknown": "Unknown",
}

# Coarse groups for association gating.
GROUP = {
    "Car": "vehicle", "Truck": "vehicle", "Bus": "vehicle",
    "Pedestrian": "vru", "Cyclist": "vru", "Motorcyclist": "vru", "Scooter": "vru",
    "Unknown": "any",
}

# Default box sizes (l, w, h) and centre height for sources without 3D boxes (UWB).
DEFAULT_SIZE = {
    "Car": (4.5, 1.8, 1.5), "Truck": (8.0, 2.5, 3.2), "Bus": (12.0, 2.5, 3.2),
    "Pedestrian": (0.6, 0.6, 1.75), "Cyclist": (1.8, 0.6, 1.7),
    "Motorcyclist": (2.1, 0.8, 1.6), "Scooter": (1.2, 0.6, 1.75),
    "Unknown": (1.0, 1.0, 1.5),
}

# Within the VRU group, which class wins a vote when sources disagree is decided
# by the score-weighted vote; this matrix adds an association cost penalty
# (in Mahalanobis units) for plausible-but-different classes.
def class_penalty(a: str, b: str) -> Optional[float]:
    """Return extra association cost, or None if incompatible."""
    if a == b or "Unknown" in (a, b):
        return 0.0
    ga, gb = GROUP.get(a, "any"), GROUP.get(b, "any")
    if ga == gb:
        return 1.0          # e.g. lidar 'Pedestrian' vs UWB 'Scooter'
    return None             # vehicle vs VRU: never associate


def canonical_category(raw: str, src_map: Dict[str, str]) -> str:
    raw = src_map.get(raw, raw)
    return CANONICAL.get(str(raw).strip().lower(), str(raw) if str(raw) in GROUP else "Unknown")


# =============================================================================
# 3. Geometry helpers (WGS84 <-> ECEF <-> ENU, rotations)
# =============================================================================

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


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
    alt = r / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def _ecef_to_enu_rot(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), radians, body -> parent."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class MapFrame:
    """Local ENU tangent frame anchored at a fixed WGS84 origin for the session."""

    def __init__(self, origin_lla: Tuple[float, float, float]):
        self.origin_lla = origin_lla
        self.origin_ecef = lla_to_ecef(*origin_lla)
        self.R = _ecef_to_enu_rot(origin_lla[0], origin_lla[1])

    def lla_to_enu(self, lat: float, lon: float, alt: float) -> np.ndarray:
        return self.R @ (lla_to_ecef(lat, lon, alt) - self.origin_ecef)

    def enu_to_lla(self, p: np.ndarray) -> Tuple[float, float, float]:
        return ecef_to_lla(self.R.T @ p + self.origin_ecef)


@dataclass
class Pose:
    """Sender pose expressed in the map ENU frame."""
    t: np.ndarray        # (3,) position
    R: np.ndarray        # (3,3) body -> map
    yaw: float           # rad, CCW from East
    lla: Tuple[float, float, float]
    rpy_deg: Tuple[float, float, float]

    @staticmethod
    def from_sender_state(ss: dict, mf: MapFrame) -> "Pose":
        p, o = ss["position"], ss["orientation"]
        lla = (float(p["latitude"]), float(p["longitude"]), float(p["altitude"]))
        roll, pitch, yaw = (math.radians(float(o["roll"])),
                            math.radians(float(o["pitch"])),
                            math.radians(float(o["yaw"])))
        # Orientation is given in the ENU tangent at the SENDER's position; for the
        # few-hundred-metre ranges in HIDDEN the difference to the map-origin ENU
        # is negligible (<0.01 deg per 1 km), so it is not corrected here.
        return Pose(t=mf.lla_to_enu(*lla), R=rot_zyx(roll, pitch, yaw), yaw=yaw, lla=lla,
                    rpy_deg=(float(o["roll"]), float(o["pitch"]), float(o["yaw"])))


# =============================================================================
# 4. Internal representation
# =============================================================================

@dataclass
class Detection:
    source: str                     # sender_id
    src_track_id: Optional[int]
    category: str                   # canonical
    score: float
    t: float                        # capture timestamp
    pos: np.ndarray                 # (3,) map ENU
    vel: np.ndarray                 # (2,) map ENU, over ground
    P: np.ndarray                   # (2,2) position covariance (xy)
    Pv: np.ndarray                  # (2,2) velocity covariance
    heading: Optional[float] = None # rad, map frame
    yaw_rate: Optional[float] = None
    size: Optional[Tuple[float, float, float]] = None  # l, w, h
    has_z: bool = False
    extras: Dict[str, float] = field(default_factory=dict)  # optional a/ax/ay/w (ego-frame values passed through)
    t_capture: Optional[float] = None  # original sensor capture time (kept through propagation)


@dataclass
class FusedObject:
    members: List[Detection]
    category: str = "Unknown"
    score: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P: np.ndarray = field(default_factory=lambda: np.eye(2))
    heading: float = 0.0
    heading_measured: bool = False
    yaw_rate: float = 0.0
    size: Tuple[float, float, float] = (1.0, 1.0, 1.5)
    size_measured: bool = False
    track_id: Optional[int] = None


@dataclass
class FusedTrack:
    track_id: int
    pos: np.ndarray
    vel: np.ndarray
    t: float
    links: Dict[str, Optional[int]]   # sender_id -> last source track_id
    hits: int = 1


# =============================================================================
# 5. Parsing
# =============================================================================

class SchemaError(ValueError):
    pass


def _req(d: dict, key: str, ctx: str):
    if key not in d:
        raise SchemaError(f"{ctx}: missing required field '{key}'")
    return d[key]


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def detections_from_message(msg: dict, mf: MapFrame, cfg: FusionConfig,
                            ego_pose: Optional[Pose]) -> Tuple[List[Detection], Pose]:
    """Convert one ODT / 2D_ODT_UWB message into Detection objects in the map frame."""
    md = _req(msg, "metadata", "message")
    sender = _req(md, "sender_id", "metadata")
    scfg = cfg.sources.get(sender)
    if scfg is None:
        raise SchemaError(f"unknown sender_id '{sender}' — add it to FusionConfig.sources")

    sender_pose = Pose.from_sender_state(_req(msg, "sender_state", sender), mf)
    rf = msg.get("reference_frame", {}).get("coordinate_frame", "ego")

    # Body frame of the numbers in 'detections' -> map
    if rf == "ego":
        R_b, t_b = sender_pose.R, sender_pose.t
    elif rf == "sensor":
        if scfg.extrinsic is None:
            raise SchemaError(f"{sender}: coordinate_frame='sensor' but no extrinsic configured")
        ex, ey, ez, er, ep, eyaw = scfg.extrinsic
        R_se = rot_zyx(er, ep, eyaw)
        R_b = sender_pose.R @ R_se
        t_b = sender_pose.t + sender_pose.R @ np.array([ex, ey, ez])
    elif rf == "world":
        # Schema does not define the world origin; we assume ENU at the map origin.
        R_b, t_b = np.eye(3), np.zeros(3)
    else:
        raise SchemaError(f"{sender}: unsupported coordinate_frame '{rf}'")
    yaw_b = math.atan2(R_b[1, 0], R_b[0, 0])

    out: List[Detection] = []
    for i, d in enumerate(msg.get("detections", []) or []):
        ctx = f"{sender}.detections[{i}]"
        try:
            score = float(_req(d, "score", ctx))
            if score < scfg.min_score:
                continue
            opt = d.get("optional") or {}
            # 3D_ODT puts z/size/heading at top level, 2D_ODT_UWB under 'optional'
            def g(k):
                return d.get(k, opt.get(k))

            x, y = float(_req(d, "x", ctx)), float(_req(d, "y", ctx))
            z = _f(g("z"))
            p_body = np.array([x, y, z if z is not None else 0.0])
            pos = R_b @ p_body + t_b
            if z is None:  # no height info: put the object on the sender's ground plane
                pos[2] = np.nan

            v_body = np.array([_f(_req(d, "vx", ctx), 0.0), _f(_req(d, "vy", ctx), 0.0), 0.0])
            vel = (R_b @ v_body)[:2]

            rng = math.hypot(x, y) if rf != "world" else (
                float(np.linalg.norm((pos - ego_pose.t)[:2])) if ego_pose else 0.0)
            sp = scfg.pos_sigma0 + scfg.pos_sigma_per_m * rng
            heading = _f(g("heading"))
            l, w, h = _f(g("length")), _f(g("width")), _f(g("height"))
            size = (l, w, h) if None not in (l, w, h) else None
            # Anisotropic covariance for 3D boxes: larger along the box length
            if heading is not None and size is not None:
                hm = heading + yaw_b
                c, s = math.cos(hm), math.sin(hm)
                Rh = np.array([[c, -s], [s, c]])
                P = Rh @ np.diag([(sp * (1 + 0.1 * size[0])) ** 2, sp ** 2]) @ Rh.T
            else:
                P = np.eye(2) * sp ** 2

            cat = canonical_category(_req(d, "category", ctx), scfg.category_map)
            out.append(Detection(
                source=sender,
                src_track_id=d.get("track_id"),
                category=cat,
                score=score,
                t=float(_req(d, "timestamp", ctx)),
                t_capture=float(_req(d, "timestamp", ctx)),
                pos=pos,
                vel=vel,
                P=P,
                Pv=np.eye(2) * scfg.vel_sigma ** 2,
                heading=wrap_pi(heading + yaw_b) if heading is not None else None,
                yaw_rate=_f(g("yaw_rate")),
                size=size,
                has_z=z is not None,
                extras={k: float(opt[k]) for k in ("a", "ax", "ay", "w") if k in opt},
            ))
        except SchemaError as e:
            log.warning("skipping detection: %s", e)
    return out, sender_pose


# =============================================================================
# 6. Temporal alignment
# =============================================================================

def propagate(det: Detection, t_target: float, q_acc: float) -> Detection:
    """Move a detection to t_target (CTRV if yaw_rate known, else CV)."""
    dt = t_target - det.t
    if abs(dt) < 1e-6:
        return det
    vx, vy = det.vel
    w = det.yaw_rate or 0.0
    if abs(w) > 1e-3 and det.heading is not None:
        v = math.hypot(vx, vy)
        th = math.atan2(vy, vx) if v > 0.2 else det.heading
        dx = v / w * (math.sin(th + w * dt) - math.sin(th))
        dy = v / w * (-math.cos(th + w * dt) + math.cos(th))
        th2 = th + w * dt
        vel = np.array([v * math.cos(th2), v * math.sin(th2)])
        heading = wrap_pi(det.heading + w * dt)
    else:
        dx, dy = vx * dt, vy * dt
        vel = det.vel.copy()
        heading = det.heading
    pos = det.pos.copy()
    pos[0] += dx
    pos[1] += dy
    # Covariance: P + dt^2 Pv + white-acceleration term
    q = (q_acc ** 2) * (dt ** 4) / 4.0
    P = det.P + (dt ** 2) * det.Pv + np.eye(2) * q
    Pv = det.Pv + np.eye(2) * (q_acc ** 2) * (dt ** 2)
    return Detection(**{**det.__dict__, "pos": pos, "vel": vel, "P": P, "Pv": Pv,
                        "heading": heading, "t": t_target})


# =============================================================================
# 7. Association
# =============================================================================

def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    if cost.size == 0:
        return []
    if _HAVE_SCIPY:
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    pairs, used_r, used_c = [], set(), set()  # greedy fallback
    for idx in np.argsort(cost, axis=None):
        i, j = np.unravel_index(idx, cost.shape)
        if i not in used_r and j not in used_c:
            pairs.append((int(i), int(j)))
            used_r.add(i)
            used_c.add(j)
    return pairs


def _info(d: Detection, cfg: "FusionConfig") -> np.ndarray:
    """Information matrix of a detection, down-weighted by its confidence so a
    low-score (possibly false) detection cannot drag a confident object."""
    w = max(1e-3, d.score * cfg.sources[d.source].trust)
    return w * np.linalg.inv(d.P)


class _Cluster:
    """Running fused estimate used during association."""

    def __init__(self, d: Detection, cfg: "FusionConfig"):
        self.cfg = cfg
        self.members = [d]
        self.pos = d.pos[:2].copy()
        self.P = np.linalg.inv(_info(d, cfg))
        self.category = d.category
        self.sources = {d.source}

    def add(self, d: Detection):
        self.members.append(d)
        self.sources.add(d.source)
        I1, I2 = np.linalg.inv(self.P), _info(d, self.cfg)
        self.P = np.linalg.inv(I1 + I2)
        self.pos = self.P @ (I1 @ self.pos + I2 @ d.pos[:2])
        if self.category == "Unknown":
            self.category = d.category


def associate(dets_by_source: Dict[str, List[Detection]], cfg: FusionConfig) -> List[_Cluster]:
    gate = _chi2.ppf(cfg.gate_prob, df=2) if _HAVE_SCIPY else 9.21
    BIG = 1e6
    order = sorted(dets_by_source, key=lambda s: cfg.sources[s].priority)
    clusters: List[_Cluster] = []
    for src in order:
        dets = dets_by_source[src]
        if not dets:
            continue
        if not clusters:
            clusters = [_Cluster(d, cfg) for d in dets]
            continue
        cost = np.full((len(dets), len(clusters)), BIG)
        for i, d in enumerate(dets):
            for j, c in enumerate(clusters):
                if src in c.sources:
                    continue  # one detection per source per object
                pen = class_penalty(d.category, c.category)
                if pen is None:
                    continue
                diff = d.pos[:2] - c.pos
                if np.linalg.norm(diff) > cfg.max_gate_dist_m:
                    continue
                S = d.P + c.P
                m2 = float(diff @ np.linalg.solve(S, diff))
                if m2 <= gate:
                    cost[i, j] = m2 + pen
        matched = set()
        for i, j in _assign(cost):
            if cost[i, j] < BIG:
                clusters[j].add(dets[i])
                matched.add(i)
        clusters.extend(_Cluster(dets[i], cfg) for i in range(len(dets)) if i not in matched)
    return clusters


# =============================================================================
# 8. State fusion
# =============================================================================

def fuse_cluster(c: _Cluster, cfg: FusionConfig) -> FusedObject:
    m = c.members
    fo = FusedObject(members=m)

    # --- position / velocity: information-weighted, using only members whose
    #     confidence is comparable to the best one (weak members — possible
    #     clutter — still support existence but do not move the state)
    conf = [d.score * cfg.sources[d.source].trust for d in m]
    kin = [d for d, c in zip(m, conf) if c >= cfg.state_min_rel_conf * max(conf)]
    I_sum = np.zeros((2, 2))
    Iv_sum = np.zeros((2, 2))
    xi = np.zeros(2)
    vi = np.zeros(2)
    for d in kin:
        w = max(1e-3, d.score * cfg.sources[d.source].trust)
        I = _info(d, cfg)
        Iv = w * np.linalg.inv(d.Pv)
        I_sum += I
        xi += I @ d.pos[:2]
        Iv_sum += Iv
        vi += Iv @ d.vel
    fo.P = np.linalg.inv(I_sum)
    xy = fo.P @ xi
    fo.vel = np.linalg.inv(Iv_sum) @ vi

    # --- z: mean of sources that measured it
    zs = [d.pos[2] for d in kin if d.has_z] or [d.pos[2] for d in m if d.has_z]
    z = float(np.mean(zs)) if zs else float("nan")

    # --- category: score*trust weighted vote; 'Unknown' only if nothing else
    votes: Dict[str, float] = {}
    for d in m:
        w = d.score * cfg.sources[d.source].trust
        votes[d.category] = votes.get(d.category, 0.0) + (0.01 * w if d.category == "Unknown" else w)
    fo.category = max(votes, key=votes.get)

    # --- existence score: noisy-OR over independent sources
    miss = 1.0
    for d in m:
        miss *= 1.0 - min(1.0, d.score * cfg.sources[d.source].trust)
    fo.score = 1.0 - miss
    if len(c.sources) == 1:
        fo.score *= cfg.single_source_penalty

    # --- heading: circular mean of 3D sources, resolving the 180 deg ambiguity
    box_members = [d for d in m if d.heading is not None and cfg.sources[d.source].has_3d_box]
    if box_members:
        ref = max(box_members, key=lambda d: d.score).heading
        s = cth = 0.0
        for d in box_members:
            h = d.heading
            if abs(wrap_pi(h - ref)) > math.pi / 2:
                h = wrap_pi(h + math.pi)
            w = d.score * cfg.sources[d.source].trust
            s += w * math.sin(h)
            cth += w * math.cos(h)
        fo.heading = math.atan2(s, cth)
        fo.heading_measured = True
        # If the object is clearly moving, align heading with velocity direction
        if np.linalg.norm(fo.vel) > 1.0:
            vh = math.atan2(fo.vel[1], fo.vel[0])
            if abs(wrap_pi(fo.heading - vh)) > math.pi / 2:
                fo.heading = wrap_pi(fo.heading + math.pi)
    elif np.linalg.norm(fo.vel) > 0.5:
        fo.heading = math.atan2(fo.vel[1], fo.vel[0])
    else:
        hs = [d.heading for d in m if d.heading is not None]
        fo.heading = hs[0] if hs else 0.0

    yrs = [(d.yaw_rate, d.score) for d in m if d.yaw_rate is not None]
    fo.yaw_rate = (sum(y * s for y, s in yrs) / sum(s for _, s in yrs)) if yrs else 0.0

    # --- size: best-scoring 3D source; else category default
    sized = [d for d in m if d.size is not None]
    if sized:
        fo.size = max(sized, key=lambda d: d.score).size
        fo.size_measured = True
    else:
        fo.size = DEFAULT_SIZE.get(fo.category, DEFAULT_SIZE["Unknown"])

    fo.pos = np.array([xy[0], xy[1], z])  # z may be NaN (UWB-only) -> resolved at output
    return fo


# =============================================================================
# 9. Track management (stable fused IDs)
# =============================================================================

class TrackManager:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.tracks: Dict[int, FusedTrack] = {}
        self._next_id = 1

    def update(self, objs: List[FusedObject], t: float):
        taken: set = set()
        unassigned: List[FusedObject] = []

        # (a) reuse ID through source-track links (most robust when sources track well)
        for o in objs:
            votes: Dict[int, int] = {}
            for d in o.members:
                if d.src_track_id is None:
                    continue
                for tid, tr in self.tracks.items():
                    if tr.links.get(d.source) == d.src_track_id:
                        votes[tid] = votes.get(tid, 0) + 1
            cand = sorted(votes, key=lambda k: -votes[k])
            cand = [c for c in cand if c not in taken]
            if cand:
                o.track_id = cand[0]
                taken.add(cand[0])
            else:
                unassigned.append(o)

        # (b) spatial matching to predicted remaining tracks
        free = [tid for tid in self.tracks if tid not in taken]
        if unassigned and free:
            cost = np.full((len(unassigned), len(free)), 1e6)
            for i, o in enumerate(unassigned):
                for j, tid in enumerate(free):
                    tr = self.tracks[tid]
                    pred = tr.pos[:2] + tr.vel * (t - tr.t)
                    dist = float(np.linalg.norm(o.pos[:2] - pred))
                    if dist < self.cfg.track_gate_m:
                        cost[i, j] = dist
            for i, j in _assign(cost):
                if cost[i, j] < 1e6:
                    unassigned[i].track_id = free[j]
                    taken.add(free[j])

        # (c) new IDs
        for o in objs:
            if o.track_id is None:
                o.track_id = self._next_id
                self._next_id += 1

        # update / create
        for o in objs:
            links = {d.source: d.src_track_id for d in o.members}
            tr = self.tracks.get(o.track_id)
            if tr is None:
                self.tracks[o.track_id] = FusedTrack(o.track_id, o.pos.copy(), o.vel.copy(), t, links)
            else:
                tr.pos, tr.vel, tr.t = o.pos.copy(), o.vel.copy(), t
                tr.links.update(links)
                tr.hits += 1

        # prune
        for tid in [k for k, tr in self.tracks.items() if t - tr.t > self.cfg.track_max_age_s]:
            del self.tracks[tid]


# =============================================================================
# 10. Main class
# =============================================================================

class LateFusionODT:
    """
    Usage:
        fusion = LateFusionODT()
        out_msg = fusion.fuse([denso_msg, upv_msg, ctag_msg])   # dicts (parsed JSON)
        json.dumps(out_msg)
    Call fuse() at the output rate with the latest message from each source
    (missing / stale sources are simply skipped).
    """

    def __init__(self, cfg: Optional[FusionConfig] = None):
        self.cfg = cfg or FusionConfig()
        self.map: Optional[MapFrame] = (MapFrame(self.cfg.map_origin_lla)
                                        if self.cfg.map_origin_lla else None)
        self.tracks = TrackManager(self.cfg)
        self.seq = 0
        self.last_ego_pose: Optional[Pose] = None
        self.stats: Dict[str, int] = {}

    # ------------------------------------------------------------------ public
    def fuse(self, messages: Iterable[dict], t_fusion: Optional[float] = None) -> dict:
        messages = [m for m in messages if m]
        # Keep only the newest message per sender
        latest: Dict[str, dict] = {}
        for m in messages:
            sid = m.get("metadata", {}).get("sender_id")
            if sid is None:
                log.warning("message without sender_id ignored")
                continue
            if sid not in latest or m["metadata"].get("sequence_number", 0) >= \
                    latest[sid]["metadata"].get("sequence_number", 0):
                latest[sid] = m

        if self.map is None and latest:
            ss = next(iter(latest.values()))["sender_state"]["position"]
            self.map = MapFrame((float(ss["latitude"]), float(ss["longitude"]), float(ss["altitude"])))

        dets_by_source: Dict[str, List[Detection]] = {}
        poses: List[Tuple[float, Pose]] = []
        for sid, m in latest.items():
            try:
                dets, pose = detections_from_message(m, self.map, self.cfg, self.last_ego_pose)
            except (SchemaError, KeyError, TypeError, ValueError) as e:
                log.error("message from %s rejected: %s", sid, e)
                continue
            dets_by_source[sid] = dets
            t_msg = max((d.t for d in dets), default=_f(m["metadata"].get("send_timestamp"), 0.0))
            poses.append((t_msg, pose))

        if t_fusion is None:
            all_t = [d.t for ds in dets_by_source.values() for d in ds]
            t_fusion = max(all_t) if all_t else time.time()

        # Ego pose at fusion time = pose attached to the most recent message
        # (sender_state has no timestamp of its own — see notes).
        if poses:
            self.last_ego_pose = max(poses, key=lambda p: p[0])[1]
        ego = self.last_ego_pose

        # Drop stale, propagate the rest
        for sid in list(dets_by_source):
            kept = []
            for d in dets_by_source[sid]:
                if t_fusion - d.t > self.cfg.max_latency_s:
                    self.stats["stale"] = self.stats.get("stale", 0) + 1
                    continue
                kept.append(propagate(d, t_fusion, self.cfg.process_noise_acc))
            dets_by_source[sid] = kept

        clusters = associate(dets_by_source, self.cfg)
        objs = [fuse_cluster(c, self.cfg) for c in clusters]
        objs = [o for o in objs if o.score >= self.cfg.min_output_score]
        self.tracks.update(objs, t_fusion)
        return self._to_message(objs, ego, t_fusion)

    # ----------------------------------------------------------------- output
    def _to_message(self, objs: List[FusedObject], ego: Optional[Pose], t: float) -> dict:
        self.seq += 1
        R_inv = ego.R.T if ego else np.eye(3)
        t0 = ego.t if ego else np.zeros(3)
        yaw0 = ego.yaw if ego else 0.0

        dets_out = []
        for o in sorted(objs, key=lambda o: o.track_id):
            p = o.pos.copy()
            if np.isnan(p[2]):
                p[2] = t0[2]  # ground plane of ego; z then lifted below
                p_ego = R_inv @ (p - t0)
                # No height measured (UWB-only): box centre at h/2 above the ego
                # reference plane (assumes flat road and ego origin on the ground).
                p_ego[2] = o.size[2] / 2.0
            else:
                p_ego = R_inv @ (p - t0)
            v_ego = (R_inv @ np.array([o.vel[0], o.vel[1], 0.0]))[:2]
            heading_ego = wrap_pi(o.heading - yaw0)
            det = {
                "track_id": int(o.track_id),
                "category": o.category,
                "score": round(float(o.score), 4),
                "timestamp": t,
                "x": round(float(p_ego[0]), 3),
                "y": round(float(p_ego[1]), 3),
                "z": round(float(p_ego[2]), 3),
                "length": round(float(o.size[0]), 3),
                "width": round(float(o.size[1]), 3),
                "height": round(float(o.size[2]), 3),
                "heading": round(float(heading_ego), 4),
                "yaw_rate": round(float(o.yaw_rate), 4),
                "vx": round(float(v_ego[0]), 3),
                "vy": round(float(v_ego[1]), 3),
                "optional": {"v": round(float(np.linalg.norm(v_ego)), 3)},
            }
            if self.cfg.emit_provenance:
                # Proposed schema extension — needed downstream for CA-impact / XAI.
                det["sources"] = [{
                    "sender_id": d.source,
                    "source_track_id": d.src_track_id,
                    "score": round(d.score, 4),
                    "capture_timestamp": d.t_capture,
                } for d in o.members]
                det["fusion_info"] = {
                    "num_sources": len({d.source for d in o.members}),
                    "pos_std_m": round(float(math.sqrt(max(np.linalg.eigvalsh(o.P)))), 3),
                    "size_measured": o.size_measured,
                    "heading_measured": o.heading_measured,
                }
            dets_out.append(det)

        ss = {
            "agent_type": "ego",
            "position": dict(zip(("latitude", "longitude", "altitude"),
                                 ego.lla if ego else (0.0, 0.0, 0.0))),
            "orientation": dict(zip(("roll", "pitch", "yaw"),
                                    ego.rpy_deg if ego else (0.0, 0.0, 0.0))),
        }
        return {
            "metadata": {
                "schema_version": self.cfg.schema_version,
                "message_type": "Ego_3D_ODT",
                "message_id": str(uuid.uuid4()),
                "sender_id": self.cfg.ego_sender_id,
                "sequence_number": self.seq,
                "send_timestamp": time.time(),
                "time_reference": "UTC",
            },
            "sender_state": ss,
            "reference_frame": {"coordinate_frame": "ego", "axes_convention": "x_forward_y_left_z_up"},
            "detections": dets_out,
        }
