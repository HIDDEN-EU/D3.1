"""
Collect Levy trust-map statistics during inference.

Two CSVs per run:

* frames: one row per frame, with detection counts and every per-level signal
  reduced over that level's overlap mask (see ``overlap_masked_stats``), plus
  the combined level-0 trust map reduced over the combined mask.
* boxes: one row per predicted box, with its score, IoU / TP flags and every
  signal sampled at the BEV cell under the box centre.

Analyse them with ``opencood/tools/levy_analysis.py``.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from opencood.models.fuse_modules.levy_pyramid_fuse import (
    LevyPyramidFusion,
    overlap_masked_stats,
)
from opencood.utils import common_utils
from opencood.utils.transformation_utils import x1_to_x2

# Per-level maps reduced / sampled when present in ``per_scale[l]``.
LEVEL_KEYS = (
    "evidence",
    "log_evidence",
    "disagreement",
    "sigma",
    "disagreement_cosine",
    "sigma_cosine",
    "disagreement_decoded",
)
# Combined level-0 maps; 'evidence' is None under combined_mode='decoded'.
FUSED_KEYS = ("evidence", "disagreement", "sigma")
STATS = ("mean", "std", "max", "q50", "q90", "q99")
IOU_THRESHOLDS = (0.3, 0.5, 0.7)


def get_levy_fusion(model):
    """The model's LevyPyramidFusion, or None if it does not use one."""
    fusion = getattr(model, "pyramid_backbone", None)
    return fusion if isinstance(fusion, LevyPyramidFusion) else None


def frame_identity(dataset, idx):
    """
    (scenario, timestamp) of dataset index ``idx`` for OPV2V-style datasets.

    Returns empty strings when the dataset does not expose ``len_record`` /
    ``scenario_database`` or ``idx`` is out of range.
    """
    if isinstance(dataset, Subset):
        idx = dataset.indices[idx]
        dataset = dataset.dataset
    len_record = getattr(dataset, "len_record", None)
    database = getattr(dataset, "scenario_database", None)
    if not len_record or database is None or not 0 <= idx < len_record[-1]:
        return "", ""

    scenario_index = next(i for i, end in enumerate(len_record) if idx < end)
    timestamp_index = idx - (len_record[scenario_index - 1] if scenario_index else 0)
    scenario_database = database[scenario_index]
    timestamp = dataset.return_timestamp_key(scenario_database, timestamp_index)
    first_cav = next(iter(scenario_database.values()))
    scenario = os.path.basename(
        os.path.dirname(os.path.dirname(first_cav[timestamp]["yaml"]))
    )
    return scenario, timestamp


def match_boxes(pred_box, pred_score, gt_box, iou_thresholds=IOU_THRESHOLDS):
    """
    Per-box TP flags, replicating ``eval_utils.caluclate_tp_fp``.

    Returns
    -------
    best_iou : np.ndarray
        (N,) highest IoU of each prediction with any ground-truth box.
    tp : dict
        {iou_thresh: (N,) int array}, in the original prediction order, using the
        same greedy score-descending matching (a GT box is consumed once matched).
    """
    n_pred = 0 if pred_box is None else pred_box.shape[0]
    if n_pred == 0:
        n_gt_ = 0 if gt_box is None else gt_box.shape[0]
        return (
            np.zeros(0),
            {t: np.zeros(0, dtype=int) for t in iou_thresholds},
            {t: np.zeros(n_gt_, dtype=int) for t in iou_thresholds},
        )

    det_polygons = list(
        common_utils.convert_format(common_utils.torch_tensor_to_numpy(pred_box))
    )
    gt_polygons = list(
        common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt_box))
    )
    order = np.argsort(-common_utils.torch_tensor_to_numpy(pred_score))

    # One IoU matrix per frame, reused for every threshold. Polygon IoU is the
    # expensive part and does not depend on the matching order, so recomputing
    # it per threshold dominated runtime once a low score threshold left many
    # predictions standing.
    n_gt = len(gt_polygons)
    iou = np.zeros((n_pred, n_gt))
    if n_gt:
        for k in range(n_pred):
            iou[k] = common_utils.compute_iou(det_polygons[k], gt_polygons)
    best_iou = iou.max(axis=1) if n_gt else np.zeros(n_pred)

    tp, gt_hit = {}, {}
    for thresh in iou_thresholds:
        flags = np.zeros(n_pred, dtype=int)
        hit = np.zeros(n_gt, dtype=int)
        # `free` replaces popping from a list: same greedy order, but the GT
        # index survives so a per-GT detected/missed flag falls out of the pass.
        free = np.ones(n_gt, dtype=bool)
        for k in order:
            if not free.any():
                break
            row = np.where(free, iou[k], -1.0)
            j = int(np.argmax(row))
            if row[j] >= thresh:
                flags[k] = 1
                hit[j] = 1
                free[j] = False
        tp[thresh] = flags
        gt_hit[thresh] = hit
    return best_iou, tp, gt_hit


def pose_misalignment(batch_data, n_agents):
    """
    Ego-frame misalignment of every collaborator under pose noise.

    Returns a list of 4x4 ``E_j = T_noisy @ inv(T_clean)``, one per collaborator
    j = 1..n_agents-1, where T maps agent j into the ego frame. A point that agent
    j sees at ego-frame location p (clean) is fused at ``E_j p``, so
    ``||(E_j - I) p||`` is the feature displacement there. Empty when the batch
    carries no clean poses or there is a single agent.
    """
    ego = batch_data["ego"]
    if "lidar_pose" not in ego or "lidar_pose_clean" not in ego:
        return []
    noisy = common_utils.torch_tensor_to_numpy(ego["lidar_pose"]).astype(np.float64)
    clean = common_utils.torch_tensor_to_numpy(ego["lidar_pose_clean"]).astype(
        np.float64
    )
    errors = []
    for j in range(1, min(n_agents, len(noisy))):
        t_noisy = x1_to_x2(noisy[j], noisy[0])
        t_clean = x1_to_x2(clean[j], clean[0])
        errors.append(t_noisy @ np.linalg.inv(t_clean))
    return errors


def _cell_index(xy, pc_range, hw):
    """Row / column of ego-frame points on a BEV grid spanning ``pc_range``."""
    H, W = hw
    x_min, y_min, _, x_max, y_max, _ = pc_range
    col = np.floor((xy[:, 0] - x_min) / (x_max - x_min) * W).astype(int)
    row = np.floor((xy[:, 1] - y_min) / (y_max - y_min) * H).astype(int)
    return np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)


def _sample(bev_map, xy, pc_range):
    """Values of a (B, 1, H, W) map (batch 0) at ego-frame points."""
    grid = bev_map[0, 0].float().cpu().numpy()
    row, col = _cell_index(xy, pc_range, grid.shape)
    return grid[row, col]


def _mask_for(mask, target):
    """Nearest-resample a level's (B, 1, h, w) mask to ``target``'s grid."""
    if mask.shape[-2:] == target.shape[-2:]:
        return mask
    return F.interpolate(mask.float(), size=target.shape[-2:], mode="nearest").bool()


def _reduce(prefix, bev_map, mask, record):
    stats = overlap_masked_stats(bev_map[0:1], mask[0:1])
    for stat in STATS:
        record[f"{prefix}_{stat}"] = stats[stat]


class LevyRecorder:
    """Accumulates per-frame and per-box rows for one inference setting."""

    def __init__(self, pc_range):
        self.pc_range = pc_range
        self.frames = []
        self.boxes = []
        # One row per GROUND-TRUTH box, for false-negative analysis: a missed
        # object has no prediction, so it cannot appear in self.boxes.
        self.gt = []
        # Corner geometry per frame, so that a re-ranking experiment can redo the
        # greedy matching instead of reusing TP flags that were assigned in the
        # original score order (a different order can hand a GT box to a
        # different prediction).
        self.geometry = {}

    def add(self, frame, dataset, use_cav, batch_data, infer_result, uncertainty):
        pred_box = infer_result["pred_box_tensor"]
        pred_score = infer_result["pred_score"]
        gt_box = infer_result["gt_box_tensor"]
        n_pred = 0 if pred_box is None else int(pred_box.shape[0])
        n_gt = 0 if gt_box is None else int(gt_box.shape[0])
        best_iou, tp, gt_hit = match_boxes(pred_box, pred_score, gt_box)

        self.geometry[f"pred_{frame}"] = (
            np.zeros((0, 8, 3), dtype=np.float32)
            if pred_box is None
            else common_utils.torch_tensor_to_numpy(pred_box).astype(np.float32)
        )
        self.geometry[f"gt_{frame}"] = (
            np.zeros((0, 8, 3), dtype=np.float32)
            if gt_box is None
            else common_utils.torch_tensor_to_numpy(gt_box).astype(np.float32)
        )

        scenario, timestamp = frame_identity(dataset, frame)
        modalities = batch_data["ego"].get("agent_modality_list", [])
        identity = {"frame": frame, "use_cav": use_cav}

        frame_row = dict(identity)
        frame_row.update(
            {
                "scenario": scenario,
                "timestamp": timestamp,
                "n_agents": int(batch_data["ego"]["record_len"][0]),
                "modalities": "".join(modalities),
                "n_pred": n_pred,
                "n_gt": n_gt,
            }
        )
        misalignment = pose_misalignment(batch_data, frame_row["n_agents"])
        if misalignment:
            frame_row["pose_err_xy"] = max(
                float(np.hypot(e[0, 3], e[1, 3])) for e in misalignment
            )
            frame_row["pose_err_yaw"] = max(
                abs(float(np.degrees(np.arctan2(e[1, 0], e[0, 0]))))
                for e in misalignment
            )
        for thresh in IOU_THRESHOLDS:
            n_tp = int(tp[thresh].sum())
            tag = int(round(thresh * 100))
            frame_row[f"tp{tag}"] = n_tp
            frame_row[f"fp{tag}"] = n_pred - n_tp

        for level, maps in enumerate(uncertainty["per_scale"]):
            mask = maps["overlap_mask"]
            frame_row[f"L{level}_n_cells"] = int(mask[0].sum())
            frame_row[f"L{level}_coverage"] = float(mask[0].float().mean())
            for key in LEVEL_KEYS:
                if key in maps:
                    _reduce(
                        f"L{level}_{key}",
                        maps[key],
                        _mask_for(mask, maps[key]),
                        frame_row,
                    )

        fused_mask = uncertainty["overlap_mask"]
        frame_row["fused_n_cells"] = int(fused_mask[0].sum())
        frame_row["fused_coverage"] = float(fused_mask[0].float().mean())
        for key in FUSED_KEYS:
            if uncertainty.get(key) is not None:
                _reduce(f"fused_{key}", uncertainty[key], fused_mask, frame_row)
        self.frames.append(frame_row)

        if n_gt > 0:
            gt_xy = common_utils.torch_tensor_to_numpy(gt_box).mean(axis=1)[:, :2]
            gt_cols = {}
            for level, maps in enumerate(uncertainty["per_scale"]):
                gt_cols[f"L{level}_n_valid"] = _sample(
                    maps["n_valid"], gt_xy, self.pc_range
                )
                for key in LEVEL_KEYS:
                    if key in maps:
                        gt_cols[f"L{level}_{key}"] = _sample(
                            maps[key], gt_xy, self.pc_range
                        )
            gt_cols["fused_n_valid"] = _sample(
                uncertainty["n_valid"], gt_xy, self.pc_range
            )
            for key in FUSED_KEYS:
                if uncertainty.get(key) is not None:
                    gt_cols[f"fused_{key}"] = _sample(
                        uncertainty[key], gt_xy, self.pc_range
                    )
            for g in range(n_gt):
                row = dict(identity)
                row.update(
                    {
                        "gt_id": g,
                        "x": float(gt_xy[g, 0]),
                        "y": float(gt_xy[g, 1]),
                        "dist": float(np.hypot(gt_xy[g, 0], gt_xy[g, 1])),
                        "n_pred": n_pred,
                    }
                )
                for thresh in IOU_THRESHOLDS:
                    row[f"hit{round(thresh * 100)}"] = int(gt_hit[thresh][g])
                for name, values in gt_cols.items():
                    row[name] = (
                        int(values[g]) if name.endswith("n_valid") else float(values[g])
                    )
                self.gt.append(row)

        if n_pred == 0:
            return

        corners = common_utils.torch_tensor_to_numpy(pred_box)
        xy = corners.mean(axis=1)[:, :2]
        scores = common_utils.torch_tensor_to_numpy(pred_score)
        columns = {}
        for level, maps in enumerate(uncertainty["per_scale"]):
            columns[f"L{level}_n_valid"] = _sample(maps["n_valid"], xy, self.pc_range)
            for key in LEVEL_KEYS:
                if key in maps:
                    columns[f"L{level}_{key}"] = _sample(maps[key], xy, self.pc_range)
        if misalignment:
            points = np.concatenate(
                [xy, np.zeros((n_pred, 1)), np.ones((n_pred, 1))], 1
            )
            columns["pose_disp"] = np.max(
                [
                    np.linalg.norm(((e - np.eye(4)) @ points.T)[:2], axis=0)
                    for e in misalignment
                ],
                axis=0,
            )
        columns["fused_n_valid"] = _sample(uncertainty["n_valid"], xy, self.pc_range)
        for key in FUSED_KEYS:
            if uncertainty.get(key) is not None:
                columns[f"fused_{key}"] = _sample(uncertainty[key], xy, self.pc_range)

        for k in range(n_pred):
            box_row = dict(identity)
            box_row.update(
                {
                    "box_id": k,
                    "x": float(xy[k, 0]),
                    "y": float(xy[k, 1]),
                    "dist": float(np.hypot(xy[k, 0], xy[k, 1])),
                    "score": float(scores[k]),
                    "best_iou": float(best_iou[k]),
                }
            )
            for thresh in IOU_THRESHOLDS:
                box_row[f"tp{int(round(thresh * 100))}"] = int(tp[thresh][k])
            for name, values in columns.items():
                value = values[k]
                box_row[name] = int(value) if name.endswith("n_valid") else float(value)
            self.boxes.append(box_row)

    def write(self, path_prefix):
        """Write ``<prefix>_frames.csv``, ``<prefix>_boxes.csv`` and geometry."""
        written = []
        if self.geometry:
            path = f"{path_prefix}_geometry.npz"
            np.savez_compressed(path, **self.geometry)
            written.append(path)
        for suffix, rows in (
            ("frames", self.frames),
            ("boxes", self.boxes),
            ("gt", self.gt),
        ):
            if not rows:
                continue
            path = f"{path_prefix}_{suffix}.csv"
            fieldnames = list(dict.fromkeys(key for row in rows for key in row))
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(rows)
            written.append(path)
        return written


@torch.no_grad()
def record_frame(recorder, levy_fusion, **kwargs):
    """Add a frame if the fusion published a trust map on its last forward."""
    if levy_fusion.uncertainty_output is not None:
        recorder.add(uncertainty=levy_fusion.uncertainty_output, **kwargs)
