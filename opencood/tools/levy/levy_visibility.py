"""
Who can see each ground-truth vehicle: LiDAR points inside its box from the ego and
from each collaborator inside the communication range, per frame of a split.

Uses the raw point clouds and CLEAN poses, the same comm_range filter as the
intermediate dataset, and the ground-truth corners already stored by
``inference.py --save_levy`` (``<prefix>_geometry.npz``), so ``(frame, gt_id)``
joins the ``_gt.csv`` tables of any model evaluated on the same split.

    python opencood/tools/levy/levy_visibility.py --model_dir <eval dir> \
        --geometry <prefix>_geometry.npz --split val --out <csv>
"""

import argparse
import csv
import importlib
import math

import numpy as np

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.utils.pcd_utils import mask_ego_points
from opencood.utils.transformation_utils import x1_to_x2


def points_in_boxes(points, corners, z_margin=0.1):
    """(n_boxes,) count of points inside each oriented box; corners (n, 8, 3)."""
    counts = np.zeros(len(corners), dtype=int)
    if not len(points) or not len(corners):
        return counts
    xy = points[:, :2]
    z = points[:, 2]
    for b, c in enumerate(corners):
        centre = c[:4, :2].mean(0)
        reach = np.linalg.norm(c[0, :2] - c[2, :2]) / 2 + 0.5
        near = np.abs(xy - centre).max(1) < reach
        if not near.any():
            continue
        p = xy[near] - c[0, :2]
        e1, e2 = c[1, :2] - c[0, :2], c[3, :2] - c[0, :2]
        u = p @ e1 / (e1 @ e1)
        v = p @ e2 / (e2 @ e2)
        zz = z[near]
        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) \
            & (zz >= c[:, 2].min() - z_margin) & (zz <= c[:, 2].max() + z_margin)
        counts[b] = int(inside.sum())
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="any eval dir with the m1 config")
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    class Opt:
        model_dir = args.model_dir
    hypes = yaml_utils.load_yaml(None, Opt)
    if "yaml_parser" in hypes:
        lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(lib, hypes["yaml_parser"])(hypes)
    if args.split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
    hypes["noise_setting"] = {"add_noise": False}
    dataset = build_dataset(hypes, visualize=False, train=False)
    comm_range = hypes["comm_range"]
    geo = np.load(args.geometry)

    rows = []
    n = len(dataset)
    for idx in range(n):
        key = f"gt_{idx}"
        if key not in geo:
            continue
        corners = geo[key]
        if not len(corners):
            continue
        base = dataset.retrieve_base_data(idx)
        ego = next(c for c in base.values() if c["ego"])
        ego_pose = ego["params"]["lidar_pose"]
        ego_pts = None
        collab = []
        for cav in base.values():
            pose = cav["params"]["lidar_pose"]
            if math.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1]) > comm_range:
                continue
            pts = mask_ego_points(cav["lidar_np"])[:, :3]
            t = x1_to_x2(pose, ego_pose)
            pts = pts @ t[:3, :3].T + t[:3, 3]
            cnt = points_in_boxes(pts, corners)
            if cav["ego"]:
                ego_pts = cnt
            else:
                collab.append(cnt)
        collab = np.stack(collab) if collab else np.zeros((0, len(corners)), dtype=int)
        for g in range(len(corners)):
            rows.append({
                "frame": idx, "gt_id": g,
                "ego_pts": int(ego_pts[g]),
                "collab_pts_max": int(collab[:, g].max()) if len(collab) else 0,
                "collab_pts_sum": int(collab[:, g].sum()) if len(collab) else 0,
                "n_collab": len(collab),
                "n_collab_seeing": int((collab[:, g] > 0).sum()) if len(collab) else 0,
            })
        if idx % 200 == 0:
            print(f"{idx}/{n}", flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
