"""
evaluate.py - replay a synthetic UC2 scenario through the fusion and score it against the ground truth.

The messages are delivered in ARRIVAL order (capture time + latency + jitter), the fusion publishes every
1/out_hz seconds. Errors are measured in the ego frame at the publication time, where a consumer (collision
avoidance) uses the object. Baselines show what a consumer would get without the fusion:
    UWB only    the latest UWB sample, moved to the ego frame (ego motion compensated, but no prediction)
    LiDAR only  the latest box that really belongs to the VRU (oracle association: it uses the simulator's truth to
                pick it), valid for 0.5 s
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from UC2_fusion import FusionConfig, Pose, UC2Fusion, rot2
from uwb_sim import LIDAR_SENDER, TAG_ID, UWB_SENDER, Scenario


@dataclass
class Run:
    t: np.ndarray
    gt: np.ndarray                       # (N, 2) ego frame
    fused: np.ndarray                    # (N, 2) nan if the VRU is not published
    uwb_hold: np.ndarray
    lidar_hold: np.ndarray
    status: List[Optional[str]]
    fused_id: List[Optional[int]]
    link_ok: np.ndarray                  # bool: published LiDAR link is the true VRU box
    link_wrong: np.ndarray               # bool: published LiDAR link is another object
    ghost: np.ndarray                    # bool: another box within 0.6 m of the fused VRU (duplicate)
    occluded: np.ndarray                 # bool: LiDAR cannot see the VRU
    uwb_down: np.ndarray                 # bool: UWB samples are not arriving
    outputs: List[dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


def run(sc: Scenario, cfg: Optional[FusionConfig] = None, out_hz: float = 10.0, extra_delay: float = 0.0,
        keep_outputs: bool = False, seed: int = 123) -> Run:
    rng = np.random.default_rng(seed)
    events = sc.events
    if extra_delay > 0:                                              # stress the reordering
        events = sorted(((a + rng.uniform(0, extra_delay), m) for a, m in events), key=lambda e: e[0])
    fusion = UC2Fusion(cfg)
    cc = fusion.cfg
    n = int((sc.cfg.duration - 1.0) * out_hz)
    ts = sc.t_start + 1.0 + np.arange(n) / out_hz
    nan2 = np.full(2, np.nan)
    gt, fused, uwb_h, lid_h = (np.full((n, 2), np.nan) for _ in range(4))
    status, ids, outs = [], [], []
    link_ok, link_wrong, ghost = (np.zeros(n, bool) for _ in range(3))
    last_uwb = last_lidar = None                                      # (capture t, map position)
    ei = 0
    for k, t_out in enumerate(ts):
        while ei < len(events) and events[ei][0] <= t_out:
            _, msg = events[ei]
            ei += 1
            fusion.push(msg)
            sid = msg["metadata"]["sender_id"]
            pose = Pose.from_sender_state(msg["sender_state"], sc.map)
            if sid == UWB_SENDER:
                d = msg["detections"][0]
                last_uwb = (d["timestamp"], pose.pos_to_map(np.array([d["x"], d["y"]])))
            elif sid == LIDAR_SENDER:
                for d in msg["detections"]:
                    if d["track_id"] == sc.vru_lidar_id(d["timestamp"]):
                        last_lidar = (d["timestamp"], pose.pos_to_map(np.array([d["x"], d["y"]])))
        ego_p, ego_yaw = sc.ego_pose(t_out)
        to_ego = lambda pm: rot2(-ego_yaw) @ (pm - ego_p)
        gt[k] = sc.vru_ego(t_out)
        if last_uwb is not None:
            uwb_h[k] = to_ego(last_uwb[1])
        if last_lidar is not None and t_out - last_lidar[0] <= 0.5:
            lid_h[k] = to_ego(last_lidar[1])
        out = fusion.output(t_out)
        if keep_outputs:
            outs.append(out)
        coop = [d for d in out["detections"] if d["track_id"] == cc.coop_id_base + TAG_ID]
        if coop:
            c = coop[0]
            fused[k] = (c["x"], c["y"])
            status.append(c["fusion_info"]["status"])
            ids.append(c["track_id"])
            lid = c["fusion_info"]["lidar_track_id"]
            if lid is not None:
                (link_ok if lid == sc.vru_lidar_id(t_out - 0.06) else link_wrong)[k] = True
            ghost[k] = any(d["track_id"] < cc.coop_id_base and
                           np.hypot(d["x"] - c["x"], d["y"] - c["y"]) < 0.6 for d in out["detections"])
        else:
            status.append(None)
            ids.append(None)
    occ = np.array([sc.in_windows(t, sc.cfg.vru_occlusions) for t in ts])
    down = np.array([sc.in_windows(t - sc.cfg.uwb_latency - 0.1, sc.cfg.uwb_dropouts) for t in ts])
    return Run(ts, gt, fused, uwb_h, lid_h, status, ids, link_ok, link_wrong, ghost, occ, down, outs, fusion.stats)


def _rmse(est: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    e = np.linalg.norm(est - gt, axis=1)
    if mask is not None:
        e = e[mask]
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")


def _avail(est: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    ok = np.isfinite(est[:, 0])
    return float(np.mean(ok[mask] if mask is not None else ok)) if (mask is None or mask.any()) else float("nan")


def summarize(r: Run) -> Dict[str, float]:
    valid = [i for i in r.fused_id if i is not None]
    linked = r.link_ok | r.link_wrong
    return {
        "fused_rmse": _rmse(r.fused, r.gt), "uwb_rmse": _rmse(r.uwb_hold, r.gt), "lidar_rmse": _rmse(r.lidar_hold, r.gt),
        "fused_avail": _avail(r.fused), "uwb_avail": _avail(r.uwb_hold), "lidar_avail": _avail(r.lidar_hold),
        "fused_rmse_occl": _rmse(r.fused, r.gt, r.occluded), "uwb_rmse_occl": _rmse(r.uwb_hold, r.gt, r.occluded),
        "lidar_avail_occl": _avail(r.lidar_hold, r.occluded), "fused_avail_occl": _avail(r.fused, r.occluded),
        "fused_rmse_uwbdown": _rmse(r.fused, r.gt, r.uwb_down), "uwb_rmse_uwbdown": _rmse(r.uwb_hold, r.gt, r.uwb_down),
        "lidar_rmse_uwbdown": _rmse(r.lidar_hold, r.gt, r.uwb_down),
        "id_changes": float(sum(1 for a, b in zip(valid, valid[1:]) if a != b)),
        "link_correct": float(r.link_ok.sum() / max(linked.sum(), 1)), "linked_frac": float(linked.mean()),
        "ghost_frac": float(r.ghost.mean()), "uwb_outliers": float(r.stats.get("uwb_outliers", 0)),
    }
