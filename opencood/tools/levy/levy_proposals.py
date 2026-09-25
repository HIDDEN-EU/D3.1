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


def _pred_corners(infer_result):
    pred = infer_result.get("pred_box_tensor")
    if pred is None:
        return np.zeros((0, 8, 3), np.float32)
    return common_utils.torch_tensor_to_numpy(pred).astype(np.float32)


def undetected_cells(mask, pred_corners, pc_range, det_range):
    """(H, W) bool: flagged cells inside det_range that no (dilated) predicted box covers."""
    from scipy.ndimage import binary_dilation

    H, W = mask.shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    dx, dy = (x_max - x_min) / W, (y_max - y_min) / H

    covered = _polygon_mask(pred_corners, (H, W), pc_range)
    if DILATE_CELLS:
        covered = binary_dilation(covered, np.ones((2 * DILATE_CELLS + 1,) * 2))

    xs = x_min + (np.arange(W) + 0.5) * dx
    ys = y_min + (np.arange(H) + 0.5) * dy
    in_range = ((ys >= det_range[1]) & (ys <= det_range[4]))[:, None] & (
        (xs >= det_range[0]) & (xs <= det_range[3]))[None, :]

    return mask & ~covered & in_range


def group_proposals(free, prob, pc_range, min_cells=MIN_CELLS):
    """One row per connected group of ``free`` cells with at least ``min_cells`` cells."""
    from scipy.ndimage import label

    H, W = free.shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    dx, dy = (x_max - x_min) / W, (y_max - y_min) / H

    groups, n = label(free)
    found = []
    for g in range(1, n + 1):
        rows, cols = np.nonzero(groups == g)
        if len(rows) < min_cells:
            continue
        found.append({
            "x": x_min + (cols.mean() + 0.5) * dx,
            "y": y_min + (rows.mean() + 0.5) * dy,
            "yaw": _principal_yaw(rows, cols, dx, dy),
            "n_cells": len(rows),
            "prob": float(prob[rows, cols].mean()),
            "prob_max": float(prob[rows, cols].max()),
        })
    return found


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
        if uncertainty.get("vehicle_mask") is None:
            raise ValueError("--save_proposals needs levy.vehicle_mask enabled")
        mask = uncertainty["vehicle_mask"][0, 0].cpu().numpy()
        free = undetected_cells(mask, _pred_corners(infer_result), self.pc_range, self.det_range)
        prob = uncertainty["vehicle_prob"][0, 0].cpu().numpy()
        found = group_proposals(free, prob, self.pc_range, self.min_cells)
        if os.environ.get("LEVY_PROP_DEBUG"):
            print(f"[proposals] frame {frame}: flagged {mask.sum()} cells, "
                  f"{free.sum()} outside detections, "
                  f"{len(found)} groups of size >= {self.min_cells}")
        for row in found:
            self.rows.append({"frame": frame, "use_cav": use_cav, **row})

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


@torch.no_grad()
def vis_overlay(levy_fusion, infer_result, pc_range, det_range, anchor_args,
                min_cells=MIN_CELLS):
    """What the vehicle mask adds to one frame, in metric coordinates, for drawing.

    ``cell_corners`` (N, 4, 2) are the flagged cells no detection covers, ``in_gt``
    (N,) says which of them lie inside a ground-truth box, and ``proposal_corners``
    (M, 8, 3) / ``proposal_prob`` (M,) are the proposals built from them.
    """
    uncertainty = levy_fusion.uncertainty_output
    if uncertainty is None or uncertainty.get("vehicle_mask") is None:
        return None
    mask = uncertainty["vehicle_mask"][0, 0].cpu().numpy()
    H, W = mask.shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    dx, dy = (x_max - x_min) / W, (y_max - y_min) / H

    free = undetected_cells(mask, _pred_corners(infer_result), pc_range, det_range)
    rows, cols = np.nonzero(free)
    x0 = x_min + cols * dx
    y0 = y_min + rows * dy
    cell_corners = np.stack([
        np.stack([x0, y0], 1), np.stack([x0 + dx, y0], 1),
        np.stack([x0 + dx, y0 + dy], 1), np.stack([x0, y0 + dy], 1),
    ], 1)

    gt = infer_result.get("gt_box_tensor")
    gt_corners = (np.zeros((0, 8, 3), np.float32) if gt is None
                  else common_utils.torch_tensor_to_numpy(gt).astype(np.float32))
    in_gt = _polygon_mask(gt_corners, (H, W), pc_range)[rows, cols]

    prob = uncertainty["vehicle_prob"][0, 0].cpu().numpy()
    found = group_proposals(free, prob, pc_range, min_cells)
    recorder = ProposalRecorder(pc_range, det_range, anchor_args, min_cells)
    table = {k: np.array([r[k] for r in found]) for k in ("x", "y", "yaw", "prob")}
    return {
        "cell_corners": cell_corners,
        "in_gt": in_gt,
        "proposal_corners": recorder.corners(table if found else []),
        "proposal_prob": table["prob"],
    }
