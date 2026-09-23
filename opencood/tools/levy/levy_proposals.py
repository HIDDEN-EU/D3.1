"""
Turn the Levy vehicle mask into detections where the detector produced none.

The mask says "a vehicle covers this cell" from the fused feature map alone, so
its value is in the gaps: cells it flags that no predicted box already covers.
Each connected group of such cells becomes one proposal box, placed at the
group's centroid and turned along its principal axis, with the anchor's vehicle
size. Proposals are ranked below every real detection, so they can only add to
the tail of the precision-recall curve.

Recorded during inference (``--save_proposals``) and scored afterwards with
``levy_proposal_ap.py``.
"""

import os

import numpy as np
import torch

from opencood.tools.levy.levy_cells import _polygon_mask
from opencood.utils import box_utils, common_utils

# A group smaller than this is noise, not a vehicle: a car covers roughly
# 4.5 x 2 m, i.e. about 15 cells on the 0.8 m grid.
MIN_CELLS = 6
# Cells this close to an existing box are treated as already detected.
DILATE_CELLS = 1


def _principal_yaw(rows, cols, dx, dy):
    """Orientation of the flagged group, from the spread of its cells."""
    x = cols * dx
    y = rows * dy
    x = x - x.mean()
    y = y - y.mean()
    if len(x) < 3:
        return 0.0
    cov = np.cov(np.stack([x, y]))
    if not np.all(np.isfinite(cov)):
        return 0.0
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    return float(np.arctan2(major[1], major[0]))


class ProposalRecorder:
    """Collects mask proposals over a run."""

    def __init__(self, pc_range, det_range, anchor_args, min_cells=MIN_CELLS):
        self.pc_range = pc_range
        self.det_range = det_range
        self.size = (anchor_args["h"], anchor_args["w"], anchor_args["l"])  # hwl
        self.z = float((pc_range[2] + pc_range[5]) / 2)
        self.min_cells = min_cells
        self.rows = []

    def add(self, frame, use_cav, infer_result, uncertainty):
        from scipy.ndimage import binary_dilation, label

        if uncertainty.get("vehicle_mask") is None:
            raise ValueError("--save_proposals needs levy.vehicle_mask enabled")
        mask = uncertainty["vehicle_mask"][0, 0].cpu().numpy()
        H, W = mask.shape
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        dx, dy = (x_max - x_min) / W, (y_max - y_min) / H

        pred = infer_result["pred_box_tensor"]
        pred_corners = (np.zeros((0, 8, 3), np.float32) if pred is None
                        else common_utils.torch_tensor_to_numpy(pred).astype(np.float32))
        covered = _polygon_mask(pred_corners, (H, W), self.pc_range)
        if DILATE_CELLS:
            covered = binary_dilation(covered, np.ones((2 * DILATE_CELLS + 1,) * 2))

        xs = x_min + (np.arange(W) + 0.5) * dx
        ys = y_min + (np.arange(H) + 0.5) * dy
        in_range = ((ys >= self.det_range[1]) & (ys <= self.det_range[4]))[:, None] & (
            (xs >= self.det_range[0]) & (xs <= self.det_range[3]))[None, :]

        free = mask & ~covered & in_range
        groups, n = label(free)
        if os.environ.get("LEVY_PROP_DEBUG"):
            sizes = np.bincount(groups.ravel())[1:] if n else np.zeros(0)
            print(f"[proposals] frame {frame}: flagged {mask.sum()} cells, "
                  f"{free.sum()} outside detections, {n} groups, "
                  f"{(sizes >= self.min_cells).sum()} of size >= {self.min_cells}")
        if n == 0:
            return
        prob = uncertainty["vehicle_prob"][0, 0].cpu().numpy()
        for g in range(1, n + 1):
            rows, cols = np.nonzero(groups == g)
            if len(rows) < self.min_cells:
                continue
            self.rows.append({
                "frame": frame,
                "use_cav": use_cav,
                "x": x_min + (cols.mean() + 0.5) * dx,
                "y": y_min + (rows.mean() + 0.5) * dy,
                "yaw": _principal_yaw(rows, cols, dx, dy),
                "n_cells": len(rows),
                "prob": float(prob[rows, cols].mean()),
                "prob_max": float(prob[rows, cols].max()),
            })

    def corners(self, rows):
        """(N, 8, 3) corners of the given proposal rows, as the evaluation wants."""
        if not len(rows):
            return np.zeros((0, 8, 3), np.float32)
        boxes = np.stack([
            rows["x"], rows["y"], np.full(len(rows["x"]), self.z),
            np.full(len(rows["x"]), self.size[0]),
            np.full(len(rows["x"]), self.size[1]),
            np.full(len(rows["x"]), self.size[2]), rows["yaw"],
        ], 1)
        return box_utils.boxes_to_corners_3d(torch.tensor(boxes, dtype=torch.float32),
                                             order="hwl").numpy()

    def write(self, path_prefix):
        if not self.rows:
            return []
        out = {k: np.array([r[k] for r in self.rows]) for k in self.rows[0]}
        out["box_size_hwl"] = np.array(self.size)
        out["box_z"] = np.array([self.z])
        path = f"{path_prefix}_proposals.npz"
        np.savez_compressed(path, **out)
        return [path]


@torch.no_grad()
def record_frame(recorder, levy_fusion, frame, use_cav, infer_result, **_):
    if levy_fusion.uncertainty_output is not None:
        recorder.add(frame, use_cav, infer_result, levy_fusion.uncertainty_output)
