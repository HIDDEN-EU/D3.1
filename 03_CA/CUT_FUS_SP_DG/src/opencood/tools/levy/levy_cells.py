"""
Cell-level capture: do the Levy signals light up where vehicles actually are?

For every frame this samples bird's-eye-view cells of the level-0 grid and stores
each cell's signals together with two labels:

* ``is_vehicle`` -- a ground-truth vehicle covers the cell. Cells within
  ``MARGIN_CELLS`` of a vehicle are dropped rather than called background, since a
  cell on a vehicle's edge is only partly covered.
Each cell also carries ``det_score``: the detector's own classification score at
that cell (the strongest anchor), which is the baseline a Levy cell mask has to
beat -- evidence is built from the per-agent foreground heads, so the two may say
much the same thing.

* ``pred_tp`` -- whether a predicted box covers the cell, and whether that box is
  a true positive (1), a false positive (0), or no box covers it (-1). This is
  the cell-level view of the box study: cells under a phantom detection.

All vehicle cells are kept (up to ``MAX_PER_CLASS``) and as many background cells
are drawn at random, so the stored file stays small; every reported rate is
computed on the stored rows, not on the whole grid.
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.path import Path
from scipy.ndimage import binary_dilation

from opencood.tools.levy.levy_collect import LEVEL_KEYS, frame_identity, match_boxes
from opencood.utils import common_utils

# Cells this far (in level-0 cells) from a vehicle are neither vehicle nor background.
MARGIN_CELLS = 2
MAX_PER_CLASS = 400
TP_IOU = 0.7


def _polygon_mask(corners, hw, pc_range):
    """Binary (H, W) mask of the cells whose centre lies inside any box."""
    H, W = hw
    x_min, y_min, _, x_max, y_max, _ = pc_range
    dx, dy = (x_max - x_min) / W, (y_max - y_min) / H
    mask = np.zeros((H, W), dtype=bool)
    for box in corners:
        poly = box[:4, :2]
        c0 = int(np.floor((poly[:, 0].min() - x_min) / dx))
        c1 = int(np.ceil((poly[:, 0].max() - x_min) / dx))
        r0 = int(np.floor((poly[:, 1].min() - y_min) / dy))
        r1 = int(np.ceil((poly[:, 1].max() - y_min) / dy))
        c0, c1 = max(c0, 0), min(c1 + 1, W)
        r0, r1 = max(r0, 0), min(r1 + 1, H)
        if c0 >= c1 or r0 >= r1:
            continue
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        xs = x_min + (cols.ravel() + 0.5) * dx
        ys = y_min + (rows.ravel() + 0.5) * dy
        inside = Path(poly).contains_points(np.stack([xs, ys], 1))
        mask[rows.ravel()[inside], cols.ravel()[inside]] = True
    return mask


def _box_index_map(corners, hw, pc_range):
    """(H, W) int map: index of the last box covering each cell, -1 if none."""
    H, W = hw
    out = np.full((H, W), -1, dtype=np.int32)
    for k, box in enumerate(corners):
        m = _polygon_mask(box[None], hw, pc_range)
        out[m] = k
    return out


class ScoreMapHook:
    """Keeps the last classification map the model produced, via a forward hook."""

    def __init__(self, model):
        self.last = None
        head = getattr(model, "cls_head", None)
        if head is None:
            raise ValueError("model has no cls_head; cannot record the detector score map")
        head.register_forward_hook(self._store)

    def _store(self, module, inputs, output):
        # (B, anchors, H, W) logits -> per-cell probability of the strongest anchor
        self.last = torch.sigmoid(output.detach()).max(dim=1, keepdim=True).values


class CellRecorder:
    """Accumulates sampled cells over the run and writes one npz."""

    def __init__(self, pc_range, det_range, seed=0):
        self.pc_range = pc_range
        self.det_range = det_range
        self.rng = np.random.default_rng(seed)
        self.rows = []

    def add(self, frame, dataset, use_cav, batch_data, infer_result, uncertainty,
            det_score_map=None):
        maps = uncertainty["per_scale"]
        H, W = maps[0]["evidence"].shape[-2:]
        gt = infer_result["gt_box_tensor"]
        pred = infer_result["pred_box_tensor"]
        gt_corners = (np.zeros((0, 8, 3), np.float32) if gt is None
                      else common_utils.torch_tensor_to_numpy(gt).astype(np.float32))
        pred_corners = (np.zeros((0, 8, 3), np.float32) if pred is None
                        else common_utils.torch_tensor_to_numpy(pred).astype(np.float32))

        vehicle = _polygon_mask(gt_corners, (H, W), self.pc_range)
        near = binary_dilation(vehicle, np.ones((2 * MARGIN_CELLS + 1,) * 2))

        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        dx, dy = (x_max - x_min) / W, (y_max - y_min) / H
        xs = x_min + (np.arange(W) + 0.5) * dx
        ys = y_min + (np.arange(H) + 0.5) * dy
        in_range = ((ys >= self.det_range[1]) & (ys <= self.det_range[4]))[:, None] & (
            (xs >= self.det_range[0]) & (xs <= self.det_range[3]))[None, :]
        # Cells with a single contributor are kept too: disagreement is zero
        # there, but evidence is not (Lambda = tau * exp(S) of that one agent),
        # so an evidence-only rule can still act on them. ``fused_n_valid`` on
        # each row says which regime it is, and the analysis filters on it.
        usable = in_range & (uncertainty["n_valid"][0, 0] >= 1).cpu().numpy()

        pos = np.argwhere(vehicle & usable)
        neg = np.argwhere(~near & usable)
        if len(pos) == 0 or len(neg) == 0:
            return
        if len(pos) > MAX_PER_CLASS:
            pos = pos[self.rng.choice(len(pos), MAX_PER_CLASS, replace=False)]
        neg = neg[self.rng.choice(len(neg), min(len(pos), len(neg)), replace=False)]
        rows = np.concatenate([pos, neg])
        label = np.concatenate([np.ones(len(pos), np.int8), np.zeros(len(neg), np.int8)])

        # Which predicted box covers each sampled cell, and was it a true positive?
        pred_tp = np.full(len(rows), -1, np.int8)
        if len(pred_corners):
            idx = _box_index_map(pred_corners, (H, W), self.pc_range)[rows[:, 0], rows[:, 1]]
            _, tp, _ = match_boxes(pred, infer_result["pred_score"], gt, (TP_IOU,))
            covered = idx >= 0
            pred_tp[covered] = tp[TP_IOU][idx[covered]].astype(np.int8)

        scenario, timestamp = frame_identity(dataset, frame)
        record = {
            "frame": np.full(len(rows), frame, np.int32),
            "use_cav": np.full(len(rows), use_cav, np.int8),
            "scenario": np.array([scenario] * len(rows)),
            "modalities": np.array(["".join(batch_data["ego"].get("agent_modality_list", []))] * len(rows)),
            "is_vehicle": label,
            "pred_tp": pred_tp,
            # Per-frame totals, so the true share of vehicle cells on the grid can be
            # recovered from this balanced sample (needed for live precision).
            "n_usable_cells": np.full(len(rows), int(usable.sum()), np.int32),
            "n_vehicle_cells": np.full(len(rows), int((vehicle & usable).sum()), np.int32),
            "x": (x_min + (rows[:, 1] + 0.5) * dx).astype(np.float32),
            "y": (y_min + (rows[:, 0] + 0.5) * dy).astype(np.float32),
        }
        record["dist"] = np.hypot(record["x"], record["y"]).astype(np.float32)
        if det_score_map is not None:
            grid = det_score_map.float()
            if grid.shape[-2:] != (H, W):
                grid = F.interpolate(grid, size=(H, W), mode="nearest")
            record["det_score"] = (
                grid[0, 0].cpu().numpy()[rows[:, 0], rows[:, 1]].astype(np.float32))
        record["fused_n_valid"] = (
            uncertainty["n_valid"][0, 0].cpu().numpy()[rows[:, 0], rows[:, 1]].astype(np.int8))
        for level, m in enumerate(maps):
            for key in LEVEL_KEYS + ("n_valid",):
                if key not in m:
                    continue
                grid = m[key]
                if grid.shape[-2:] != (H, W):
                    grid = F.interpolate(grid.float(), size=(H, W), mode="nearest")
                record[f"L{level}_{key}"] = (
                    grid[0, 0].float().cpu().numpy()[rows[:, 0], rows[:, 1]].astype(np.float32))
        for key in ("evidence", "log_evidence", "disagreement", "sigma",
                    "disagreement_cosine", "sigma_cosine", "vehicle_prob"):
            if uncertainty.get(key) is not None:
                record[f"fused_{key}"] = (
                    uncertainty[key][0, 0].float().cpu().numpy()[rows[:, 0], rows[:, 1]].astype(np.float32))
        self.rows.append(record)

    def write(self, path_prefix):
        if not self.rows:
            return []
        out = {k: np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]}
        path = f"{path_prefix}_cells.npz"
        np.savez_compressed(path, **out)
        return [path]


@torch.no_grad()
def record_frame(recorder, levy_fusion, score_hook=None, **kwargs):
    if levy_fusion.uncertainty_output is not None:
        recorder.add(
            uncertainty=levy_fusion.uncertainty_output,
            det_score_map=None if score_hook is None else score_hook.last,
            **kwargs,
        )
