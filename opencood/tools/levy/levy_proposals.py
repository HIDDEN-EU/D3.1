"""
Turn the Levy vehicle mask into detections where the detector produced none.

The mask says "a vehicle covers this cell" from the fused feature map alone, so
its value is in the gaps: cells it flags that no predicted box already covers.
Each connected group of such cells becomes one proposal box, placed at the
group's centroid and turned along its principal axis, with the anchor's vehicle
size. Proposals are ranked below every real detection, so they can only add to
the tail of the precision-recall curve.

Recorded during inference (``--save_proposals``) and scored afterwards with
``levy_proposal_ap.py``. Given the model, the recorder also keeps each group's
cells (``*_proposal_cells.npz``) and the detector's own regressed box at the
group's strongest anchor, for the refinements in ``levy_proposal_refine.py``.
"""

import json
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


def _cover_mask(corners, hw, pc_range):
    """(H, W) bool: cells whose centre lies inside any box. Same result as
    ``levy_cells._polygon_mask`` for the convex box footprints, but all boxes at
    once: each box's bounding window of cells is tested against its four edges
    (inside = on the same side of every edge)."""
    H, W = hw
    mask = np.zeros((H, W), dtype=bool)
    if len(corners) == 0:
        return mask
    x_min, y_min, _, x_max, y_max, _ = pc_range
    dx, dy = (x_max - x_min) / W, (y_max - y_min) / H
    poly = np.asarray(corners)[:, :4, :2].astype(np.float64)  # (N, 4, 2)
    c0 = np.floor((poly[..., 0].min(1) - x_min) / dx).astype(int)
    c1 = np.ceil((poly[..., 0].max(1) - x_min) / dx).astype(int) + 1
    r0 = np.floor((poly[..., 1].min(1) - y_min) / dy).astype(int)
    r1 = np.ceil((poly[..., 1].max(1) - y_min) / dy).astype(int) + 1
    size = max(int((c1 - c0).max()), int((r1 - r0).max()), 1)
    step = np.arange(size)
    cols = c0[:, None, None] + step[None, None, :]  # (N, 1, S)
    rows = r0[:, None, None] + step[None, :, None]  # (N, S, 1)
    px = x_min + (cols + 0.5) * dx
    py = y_min + (rows + 0.5) * dy
    edge = np.roll(poly, -1, axis=1) - poly  # (N, 4, 2)
    cross = (edge[:, :, 0, None, None] * (py[:, None] - poly[:, :, 1, None, None])
             - edge[:, :, 1, None, None] * (px[:, None] - poly[:, :, 0, None, None]))  # (N, 4, S, S)
    inside = (cross >= 0).all(1) | (cross <= 0).all(1)
    inside &= ((cols < c1[:, None, None]) & (cols >= 0) & (cols < W)
               & (rows < r1[:, None, None]) & (rows >= 0) & (rows < H))
    n, r, c = np.nonzero(inside)
    mask[rows[n, r, 0], cols[n, 0, c]] = True
    return mask


def _dilate_square(mask, radius):
    """Binary dilation by a (2r+1) x (2r+1) square, as separable shifts (zero border)."""
    out = mask
    for _ in range(radius):
        rows = out.copy()
        rows[1:] |= out[:-1]
        rows[:-1] |= out[1:]
        out = rows.copy()
        out[:, 1:] |= rows[:, :-1]
        out[:, :-1] |= rows[:, 1:]
    return out


_IN_RANGE = {}


def _in_range(hw, pc_range, det_range):
    """(H, W) bool: cell centres inside det_range (cached per grid)."""
    key = (hw, tuple(pc_range), tuple(det_range))
    if key not in _IN_RANGE:
        H, W = hw
        x_min, y_min, _, x_max, y_max, _ = pc_range
        xs = x_min + (np.arange(W) + 0.5) * (x_max - x_min) / W
        ys = y_min + (np.arange(H) + 0.5) * (y_max - y_min) / H
        _IN_RANGE[key] = ((ys >= det_range[1]) & (ys <= det_range[4]))[:, None] & (
            (xs >= det_range[0]) & (xs <= det_range[3]))[None, :]
    return _IN_RANGE[key]


def undetected_cells(mask, pred_corners, pc_range, det_range):
    """(H, W) bool: flagged cells inside det_range that no (dilated) predicted box covers."""
    covered = _cover_mask(pred_corners, mask.shape, pc_range)
    if DILATE_CELLS:
        covered = _dilate_square(covered, DILATE_CELLS)
    return mask & ~covered & _in_range(mask.shape, pc_range, det_range)


def group_proposals(free, prob, pc_range, min_cells=MIN_CELLS):
    """One row per connected group of ``free`` cells with at least ``min_cells`` cells."""
    from scipy.ndimage import find_objects, label

    H, W = free.shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    dx, dy = (x_max - x_min) / W, (y_max - y_min) / H

    groups, n = label(free)
    sizes = np.bincount(groups.ravel(), minlength=n + 1)
    found = []
    # Only groups big enough to be a proposal, each searched inside its own bounding box.
    for g, box in enumerate(find_objects(groups), start=1):
        if box is None or sizes[g] < min_cells:
            continue
        rows, cols = np.nonzero(groups[box] == g)
        rows = rows + box[0].start
        cols = cols + box[1].start
        found.append({
            "x": x_min + (cols.mean() + 0.5) * dx,
            "y": y_min + (rows.mean() + 0.5) * dy,
            "yaw": _principal_yaw(rows, cols, dx, dy),
            "n_cells": len(rows),
            "prob": float(prob[rows, cols].mean()),
            "prob_max": float(prob[rows, cols].max()),
            # private: the group's cells, written to the separate cells file
            "_rows": rows,
            "_cols": cols,
        })
    return found


class DetectorMapHook:
    """Keeps the detection head's last class logits and box regression, via forward hooks."""

    def __init__(self, model):
        self.cls = self.reg = None
        for name, attr in (("cls_head", "cls"), ("reg_head", "reg")):
            head = getattr(model, name, None)
            if head is None:
                raise ValueError(f"model has no {name}; cannot record the detector's boxes")
            head.register_forward_hook(self._store(attr))

    def _store(self, attr):
        def hook(module, inputs, output):
            setattr(self, attr, output.detach())
        return hook

    @torch.no_grad()
    def anchors(self, anchor_box):
        """(prob (N,), boxes (N, 7) xyzhwlr) for every anchor, in the post-processor's order."""
        from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor

        prob = torch.sigmoid(self.cls.permute(0, 2, 3, 1)).reshape(-1)
        boxes = VoxelPostprocessor.delta_to_boxes3d(self.reg, anchor_box)[0]
        return prob.cpu().numpy(), boxes.cpu().numpy()


def strongest_anchor(rows, cols, mask_hw, anchor_hwa, pc_range, prob, boxes):
    """The detector's anchor with the highest class probability inside a group's cells."""
    H, W = mask_hw
    Ha, Wa, A = anchor_hwa
    x_min, y_min, _, x_max, y_max, _ = pc_range
    xs = x_min + (cols + 0.5) * (x_max - x_min) / W
    ys = y_min + (rows + 0.5) * (y_max - y_min) / H
    ca = np.clip(((xs - x_min) / ((x_max - x_min) / Wa)).astype(int), 0, Wa - 1)
    ra = np.clip(((ys - y_min) / ((y_max - y_min) / Ha)).astype(int), 0, Ha - 1)
    cells = np.unique(ra * Wa + ca)
    idx = (cells[:, None] * A + np.arange(A)[None]).ravel()
    k = idx[int(np.argmax(prob[idx]))]
    return float(prob[k]), boxes[k]


def group_elongation(xy, w):
    """sqrt(major / minor eigenvalue) of a group's probability-weighted cell spread."""
    if len(xy) < 3:
        return 1.0
    c = np.average(xy, axis=0, weights=w)
    ev = np.linalg.eigvalsh(np.cov((xy - c).T, aweights=w))
    return float(np.sqrt(max(ev.max(), 1e-6) / max(ev.min(), 1e-6)))


def proposal_features(n_cells, prob, prob_max, det_prob, elong, box_xy):
    """The quality model's inputs for one detector-box proposal, in the order of
    ``levy_proposal_refine.FEATURES`` (``fit_frac`` is 1 for a detector box)."""
    return [np.log(n_cells), prob, prob_max, det_prob, 1.0, elong,
            np.log1p(np.hypot(box_xy[0], box_xy[1]))]


class ProposalSelector:
    """Live mask proposals: in every uncovered group of flagged cells, the detector's
    own regressed box at the group's strongest anchor, kept when its learned quality
    score passes the threshold fitted on validation (``levy_proposal_refine.py
    --write_selector``). Kept boxes get score ``score_scale * quality``, below every
    re-scored detection, so they can only extend the tail of the ranking."""

    def __init__(self, params_path, model, pc_range, det_range):
        with open(params_path) as f:
            self.params = json.load(f)
        self.mean = np.array(self.params["mean"])
        self.scale = np.array(self.params["scale"])
        self.coef = np.array(self.params["coef"])
        self.intercept = float(self.params["intercept"])
        self.threshold = float(self.params["threshold"])
        self.score_scale = float(self.params.get("score_scale", 0.05))
        self.min_cells = int(self.params.get("min_cells", MIN_CELLS))
        self.max_cells = int(self.params.get("max_group_cells", 1 << 30))
        self.detector = DetectorMapHook(model)
        self.pc_range = pc_range
        self.det_range = det_range
        self.n_groups = self.n_kept = self.n_frames = 0

    @torch.no_grad()
    def __call__(self, uncertainty, infer_result, anchor_box):
        """(corners (K, 8, 3), quality (K,)) of the kept proposals for this frame."""
        self.n_frames += 1
        empty = (np.zeros((0, 8, 3), np.float32), np.zeros(0))
        if uncertainty is None or uncertainty.get("vehicle_mask") is None or self.detector.cls is None:
            return empty
        mask = uncertainty["vehicle_mask"][0, 0].cpu().numpy()
        prob = uncertainty["vehicle_prob"][0, 0].cpu().numpy()
        free = undetected_cells(mask, _pred_corners(infer_result), self.pc_range, self.det_range)
        found = [g for g in group_proposals(free, prob, self.pc_range, self.min_cells)
                 if g["n_cells"] <= self.max_cells]
        self.n_groups += len(found)
        if not found:
            return empty
        det_prob, det_box = self.detector.anchors(anchor_box)
        H, W = mask.shape
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        dx, dy = (x_max - x_min) / W, (y_max - y_min) / H
        boxes, X = [], []
        for g in found:
            r, c = g["_rows"], g["_cols"]
            p, box = strongest_anchor(r, c, mask.shape, tuple(anchor_box.shape[:3]),
                                      self.pc_range, det_prob, det_box)
            xy = np.stack([x_min + (c + 0.5) * dx, y_min + (r + 0.5) * dy], 1)
            X.append(proposal_features(g["n_cells"], g["prob"], g["prob_max"], p,
                                       group_elongation(xy, prob[r, c].astype(float)), box[:2]))
            boxes.append(box)
        logit = ((np.array(X) - self.mean) / self.scale) @ self.coef + self.intercept
        quality = 1.0 / (1.0 + np.exp(-logit))
        keep = quality >= self.threshold
        self.n_kept += int(keep.sum())
        if not keep.any():
            return empty
        corners = box_utils.boxes_to_corners_3d(
            torch.tensor(np.stack(boxes)[keep], dtype=torch.float32), order="hwl").numpy()
        return corners, quality[keep]

    def summary(self):
        f = max(self.n_frames, 1)
        return (f"Levy proposals: {self.n_kept} kept of {self.n_groups} groups "
                f"({self.n_kept / f:.2f} per frame, quality >= {self.threshold:.3f})")


class ProposalRecorder:
    """Collects mask proposals over a run."""

    def __init__(self, pc_range, det_range, anchor_args, min_cells=MIN_CELLS, model=None):
        self.pc_range = pc_range
        self.det_range = det_range
        self.size = (anchor_args["h"], anchor_args["w"], anchor_args["l"])  # hwl
        self.z = float((pc_range[2] + pc_range[5]) / 2)
        self.min_cells = min_cells
        self.rows = []
        self.cells = []  # (rows, cols, prob) per proposal, aligned with self.rows
        self.grid = None
        self.detector = DetectorMapHook(model) if model is not None else None

    def add(self, frame, use_cav, infer_result, uncertainty, batch_data=None):
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
        self.grid = mask.shape
        det = None
        if self.detector is not None and found and self.detector.cls is not None:
            anchor_box = batch_data["ego"]["anchor_box"]
            det = self.detector.anchors(anchor_box)
            anchor_hwa = tuple(anchor_box.shape[:3])
        for row in found:
            r, c = row.pop("_rows"), row.pop("_cols")
            if det is not None:
                p, box = strongest_anchor(r, c, mask.shape, anchor_hwa, self.pc_range, *det)
                row.update({"det_prob": p, **{f"det_{k}": float(v) for k, v in
                                             zip(("x", "y", "z", "h", "w", "l", "yaw"), box)}})
            self.rows.append({"frame": frame, "use_cav": use_cav, **row})
            self.cells.append((r.astype(np.int16), c.astype(np.int16),
                               prob[r, c].astype(np.float32)))

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
        # Ragged per-proposal cells in their own file (levy_proposal_ap keeps only
        # the per-proposal columns of the main one).
        lengths = np.array([len(r) for r, _, _ in self.cells])
        cells_path = f"{path_prefix}_proposal_cells.npz"
        np.savez_compressed(
            cells_path,
            offset=np.concatenate([[0], np.cumsum(lengths)]),
            rows=np.concatenate([r for r, _, _ in self.cells]),
            cols=np.concatenate([c for _, c, _ in self.cells]),
            prob=np.concatenate([p for _, _, p in self.cells]),
            grid=np.array(self.grid), pc_range=np.array(self.pc_range, dtype=float),
        )
        return [path, cells_path]


@torch.no_grad()
def record_frame(recorder, levy_fusion, frame, use_cav, infer_result, batch_data=None, **_):
    if levy_fusion.uncertainty_output is not None:
        recorder.add(frame, use_cav, infer_result, levy_fusion.uncertainty_output,
                     batch_data=batch_data)


@torch.no_grad()
def vis_overlay(levy_fusion, infer_result, pc_range, det_range, anchor_args,
                min_cells=MIN_CELLS, proposals=None):
    """What the vehicle mask adds to one frame, in metric coordinates, for drawing.

    ``cell_corners`` (N, 4, 2) are the flagged cells no detection covers, ``in_gt``
    (N,) says which of them lie inside a ground-truth box, and ``proposal_corners``
    (M, 8, 3) / ``proposal_prob`` (M,) are the proposals built from them -- or, when
    ``proposals`` = (corners, quality) is given, those (e.g. a ProposalSelector's).
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

    if proposals is not None:
        return {"cell_corners": cell_corners, "in_gt": in_gt,
                "proposal_corners": proposals[0], "proposal_prob": proposals[1]}
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
