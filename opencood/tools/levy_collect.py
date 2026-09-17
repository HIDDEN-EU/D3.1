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
        return np.zeros(0), {t: np.zeros(0, dtype=int) for t in iou_thresholds}

    det_polygons = list(
        common_utils.convert_format(common_utils.torch_tensor_to_numpy(pred_box))
    )
    gt_polygons = list(
        common_utils.convert_format(common_utils.torch_tensor_to_numpy(gt_box))
    )
    order = np.argsort(-common_utils.torch_tensor_to_numpy(pred_score))

    best_iou = np.zeros(n_pred)
    if gt_polygons:
        for k in range(n_pred):
            best_iou[k] = np.max(common_utils.compute_iou(det_polygons[k], gt_polygons))

    tp = {}
    for thresh in iou_thresholds:
        flags = np.zeros(n_pred, dtype=int)
        remaining = list(gt_polygons)
        for k in order:
            if not remaining:
                break
            ious = common_utils.compute_iou(det_polygons[k], remaining)
            if np.max(ious) >= thresh:
                flags[k] = 1
                remaining.pop(int(np.argmax(ious)))
        tp[thresh] = flags
    return best_iou, tp


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

    def add(self, frame, dataset, use_cav, batch_data, infer_result, uncertainty):
        pred_box = infer_result["pred_box_tensor"]
        pred_score = infer_result["pred_score"]
        gt_box = infer_result["gt_box_tensor"]
        n_pred = 0 if pred_box is None else int(pred_box.shape[0])
        n_gt = 0 if gt_box is None else int(gt_box.shape[0])
        best_iou, tp = match_boxes(pred_box, pred_score, gt_box)

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
        """Write ``<prefix>_frames.csv`` and ``<prefix>_boxes.csv``."""
        written = []
        for suffix, rows in (("frames", self.frames), ("boxes", self.boxes)):
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
