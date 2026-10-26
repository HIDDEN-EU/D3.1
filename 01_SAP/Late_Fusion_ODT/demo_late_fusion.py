"""
Synthetic end-to-end test for late_fusion_odt.py.

Scenario (Valencia, ego driving East at 10 m/s, 10 Hz, 3 s):
  GT1  Car         ahead in lane, 8 m/s          -> CTAG + UPV
  GT2  Pedestrian  crossing from the right       -> CTAG + UPV + DENSO(UWB)
  GT3  Scooter     occluded behind parked van    -> DENSO(UWB) only
  GT4  Parked van  on the right                  -> CTAG + UPV
  + UPV produces one low-score false positive per frame at a random place.
Each source has its own latency (CTAG 0 ms, UPV 50 ms, UWB 100 ms), noise
and its own track IDs. UWB uses the 2D_ODT_UWB schema (z/size under 'optional',
mostly absent). We check: one fused object per GT, stable IDs, position error.
"""
import json
import math
import numpy as np

from late_fusion_odt import LateFusionODT, MapFrame, rot_zyx

rng = np.random.default_rng(0)
ORIGIN = (39.4810, -0.3400, 10.0)
mf = MapFrame(ORIGIN)
EGO_V = 10.0

GT = {
    1: dict(cat="Car", p0=np.array([25.0, 0.0]), v=np.array([8.0, 0.0]), size=(4.5, 1.8, 1.5)),
    2: dict(cat="Pedestrian", p0=np.array([30.0, -6.0]), v=np.array([0.0, 1.4]), size=(0.6, 0.6, 1.75)),
    3: dict(cat="Scooter", p0=np.array([45.0, -5.0]), v=np.array([0.0, 2.0]), size=(1.2, 0.6, 1.75)),
    4: dict(cat="Car", p0=np.array([40.0, -3.5]), v=np.array([0.0, 0.0]), size=(5.0, 2.0, 2.2)),
}
VISIBLE = {"CTAG_lidar_ODTH": [1, 2, 4], "UPV_cam_lidar_ODTH": [1, 2, 4], "DENSO_uwb_ODT": [2, 3]}
LATENCY = {"CTAG_lidar_ODTH": 0.0, "UPV_cam_lidar_ODTH": 0.05, "DENSO_uwb_ODT": 0.10}
NOISE = {"CTAG_lidar_ODTH": 0.10, "UPV_cam_lidar_ODTH": 0.18, "DENSO_uwb_ODT": 0.30}
TID_OFFSET = {"CTAG_lidar_ODTH": 100, "UPV_cam_lidar_ODTH": 500, "DENSO_uwb_ODT": 900}
UWB_CAT = {2: "pedestrian", 3: "e-scooter"}   # UWB tags announce their user type
T0 = 1_780_000_000.0


def ego_state(t):
    return np.array([EGO_V * (t - T0), 0.0, 0.0]), 0.0  # ENU pos, yaw


def gt_pos(k, t):
    g = GT[k]
    return g["p0"] + g["v"] * (t - T0)


def make_msg(sender, t_cap, seq):
    ego_p, ego_yaw = ego_state(t_cap)
    lat, lon, alt = mf.enu_to_lla(ego_p)
    R = rot_zyx(0, 0, ego_yaw)
    dets = []
    for k in VISIBLE[sender]:
        g = GT[k]
        pw = np.array([*gt_pos(k, t_cap), 0.0]) - ego_p
        pb = R.T @ pw + rng.normal(0, NOISE[sender], 3) * [1, 1, 0.3]
        vb = R.T @ np.array([*g["v"], 0.0]) + rng.normal(0, 0.2, 3)
        heading = math.atan2(g["v"][1], g["v"][0]) if np.linalg.norm(g["v"]) > 0 else 0.0
        heading += rng.normal(0, 0.05) + (math.pi if (sender == "UPV_cam_lidar_ODTH" and k == 4) else 0)
        if sender == "DENSO_uwb_ODT":
            dets.append({"track_id": TID_OFFSET[sender] + k, "category": UWB_CAT[k],
                         "score": 0.95, "timestamp": t_cap,
                         "x": float(pb[0]), "y": float(pb[1]),
                         "vx": float(vb[0]), "vy": float(vb[1]), "optional": {}})
        else:
            dets.append({"track_id": TID_OFFSET[sender] + k, "category": g["cat"],
                         "score": float(np.clip(rng.normal(0.85, 0.05), 0, 1)), "timestamp": t_cap,
                         "x": float(pb[0]), "y": float(pb[1]), "z": float(g["size"][2] / 2 + pb[2]),
                         "length": g["size"][0], "width": g["size"][1], "height": g["size"][2],
                         "heading": float(heading), "yaw_rate": 0.0,
                         "vx": float(vb[0]), "vy": float(vb[1])})
    if sender == "UPV_cam_lidar_ODTH":   # clutter
        dets.append({"track_id": None, "category": "Pedestrian", "score": 0.32, "timestamp": t_cap,
                     "x": float(rng.uniform(5, 60)), "y": float(rng.uniform(-10, 10)), "z": 0.9,
                     "length": 0.6, "width": 0.6, "height": 1.7, "heading": 0.0, "yaw_rate": 0.0,
                     "vx": 0.0, "vy": 0.0})
    return {
        "metadata": {"schema_version": "1.0", "message_type": "Ego_3D_ODT" if "UWB" not in sender.upper() else "Ego_2D_ODT_UWB",
                     "message_id": f"{sender}-{seq}", "sender_id": sender, "sequence_number": seq,
                     "send_timestamp": t_cap + 0.02, "time_reference": "UTC"},
        "sender_state": {"agent_type": "ego",
                         "position": {"latitude": lat, "longitude": lon, "altitude": alt},
                         "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": math.degrees(ego_yaw)}},
        "reference_frame": {"coordinate_frame": "ego", "axes_convention": "x_forward_y_left_z_up"},
        "detections": dets,
    }


def main():
    fusion = LateFusionODT()
    errs, id_hist, counts = {k: [] for k in GT}, {k: set() for k in GT}, []
    out = None
    for step in range(30):
        t = T0 + 0.1 * step
        msgs = [make_msg(s, t - LATENCY[s], step) for s in VISIBLE]
        out = fusion.fuse(msgs, t_fusion=t)
        counts.append(len(out["detections"]))
        ego_p, _ = ego_state(t)
        for d in out["detections"]:
            pw = np.array([d["x"], d["y"]]) + ego_p[:2]
            same = [k for k in GT if GT[k]["cat"] == d["category"]]
            if not same:
                continue
            k = min(same, key=lambda k: np.linalg.norm(gt_pos(k, t) - pw))
            e = np.linalg.norm(gt_pos(k, t) - pw)
            if e < 2.0:
                errs[k].append(e)
                id_hist[k].add(d["track_id"])

    print("objects published per cycle:", counts)
    print(f"{'GT':<4}{'class':<12}{'frames':>7}{'mean err [m]':>14}{'fused IDs':>12}")
    for k in GT:
        print(f"{k:<4}{GT[k]['cat']:<12}{len(errs[k]):>7}{np.mean(errs[k]):>14.3f}{str(sorted(id_hist[k])):>12}")
    print("\nLast output message (truncated to 2 detections):")
    o = dict(out)
    o["detections"] = out["detections"][:2]
    print(json.dumps(o, indent=2)[:2500])


if __name__ == "__main__":
    main()
