# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.utils import common_utils
from opencood.tools import levy_metrics
from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d, nms_rotated
from opencood.utils.camera_utils import indices_to_depth
from sklearn.metrics import mean_squared_error

def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict



def inference_no_fusion(batch_data, model, dataset, single_gt=False):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    single_gt : bool
        if True, only use ego agent's label.
        else, use all agent's merged labels.
    """
    output_dict_ego = OrderedDict()
    if single_gt:
        batch_data = {'ego': batch_data['ego']}
        
    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process_no_fusion(batch_data,  # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    return return_dict

def inference_no_fusion_w_uncertainty(batch_data, model, dataset):
    """
    Model inference for no fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict_ego = OrderedDict()

    output_dict_ego['ego'] = model(batch_data['ego'])
    # output_dict only contains ego
    # but batch_data havs all cavs, because we need the gt box inside.

    pred_box_tensor, pred_score, gt_box_tensor, uncertainty_tensor = \
        dataset.post_process_no_fusion_uncertainty(batch_data, # only for late fusion dataset
                             output_dict_ego)

    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor, \
                    "uncertainty_tensor" : uncertainty_tensor}

    return return_dict


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)
    
    return_dict = {"pred_box_tensor" : pred_box_tensor, \
                    "pred_score" : pred_score, \
                    "gt_box_tensor" : gt_box_tensor}
    if "depth_items" in output_dict['ego']:
        return_dict.update({"depth_items" : output_dict['ego']['depth_items']})
    # Only present when the fusion backbone is LevyPyramidFusion with
    # levy.compute_uncertainty on; absent for plain PyramidFusion.
    if output_dict['ego'].get("levy_uncertainty", None) is not None:
        return_dict.update({"levy_uncertainty" : output_dict['ego']['levy_uncertainty']})
    
    return return_dict


def inference_intermediate_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    return_dict = inference_early_fusion(batch_data, model, dataset)
    return return_dict


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    pred_np = torch_tensor_to_numpy(pred_tensor)
    gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)


def depth_metric(depth_items, grid_conf):
    # depth logdit: [N, D, H, W]
    # depth gt indices: [N, H, W]
    depth_logit, depth_gt_indices = depth_items
    depth_pred_indices = torch.argmax(depth_logit, 1)
    depth_pred = indices_to_depth(depth_pred_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    depth_gt = indices_to_depth(depth_gt_indices, *grid_conf['ddiscr'], mode=grid_conf['mode']).flatten()
    rmse = mean_squared_error(depth_gt.cpu(), depth_pred.cpu(), squared=False)
    return rmse


def fix_cavs_box(pred_box_tensor, gt_box_tensor, pred_score, batch_data):
    """
    Fix the missing pred_box and gt_box for ego and cav(s).
    Args:
        pred_box_tensor : tensor
            shape (N1, 8, 3), may or may not include ego agent prediction, but it should include
        gt_box_tensor : tensor
            shape (N2, 8, 3), not include ego agent in camera cases, but it should include
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    Returns:
        pred_box_tensor : tensor
            shape (N1+?, 8, 3)
        gt_box_tensor : tensor
            shape (N2+1, 8, 3)
    """
    if pred_box_tensor is None or gt_box_tensor is None:
        return pred_box_tensor, gt_box_tensor, pred_score, 0
    # prepare cav's boxes

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()
        
    extent = [2.45, 1.06, 0.75]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_tensor = torch.tensor(np.concatenate(box_list, axis=0), device=pred_box_tensor.device)
    
    pred_box_tensor_ = torch.cat((cav_box_tensor, pred_box_tensor), dim=0)
    gt_box_tensor_ = torch.cat((cav_box_tensor, gt_box_tensor), dim=0)

    pred_score_ = torch.cat((torch.ones(N, device=pred_score.device), pred_score))

    gt_score_ = torch.ones(gt_box_tensor_.shape[0], device=pred_box_tensor.device)
    gt_score_[N:] = 0.5

    keep_index = nms_rotated(pred_box_tensor_,
                            pred_score_,
                            0.01)
    pred_box_tensor = pred_box_tensor_[keep_index]
    pred_score = pred_score_[keep_index]

    keep_index = nms_rotated(gt_box_tensor_,
                            gt_score_,
                            0.01)
    gt_box_tensor = gt_box_tensor_[keep_index]

    return pred_box_tensor, gt_box_tensor, pred_score, N


def get_cav_box(batch_data):
    """
    Args:
        batch_data : dict
            batch_data['lidar_pose'] and batch_data['record_len'] for putting ego's pred box and gt box
    """

    # if key only contains "ego", like intermediate fusion
    if 'record_len' in batch_data['ego']:
        lidar_pose =  batch_data['ego']['lidar_pose'].cpu().numpy()
        N = batch_data['ego']['record_len']
        relative_t = get_relative_transformation(lidar_pose) # [N, 4, 4], cav_to_ego, T_ego_cav
        agent_modality_list = batch_data['ego']['agent_modality_list']

    # elif key contains "ego", "641", "649" ..., like late fusion
    else:
        relative_t = []
        agent_modality_list = []
        for cavid, cav_data in batch_data.items():
            relative_t.append(cav_data['transformation_matrix'])
            agent_modality_list.append(cav_data['modality_name'])
        N = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

        

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3) # [8, 3]
    ego_box[..., 2] -= 1.2 # hard coded now

    box_list = [ego_box]
    
    for i in range(1, N):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)


    return cav_box_np, agent_modality_list


# ---------------------------------------------------------------------------
# Levy spatial trust map (opencood.models.fuse_modules.levy_pyramid_fuse)
#
# These helpers are no-ops unless the fusion backbone is LevyPyramidFusion with
# levy.compute_uncertainty enabled -- a plain PyramidFusion run never puts
# 'levy_uncertainty' in the output dict, so the caller simply gets None back.
# ---------------------------------------------------------------------------

# Channels of the aggregated (multi-scale) trust map that carry a magnitude and
# are worth summarising statistically.
LEVY_CHANNELS = ('evidence', 'disagreement', 'sigma')

# Panels of the BEV render. 'coverage' leads because it is the mask the other
# three have to be read through -- disagreement and sigma are structurally 0
# wherever fewer than two agents contribute.
LEVY_VIS_CHANNELS = ('coverage',) + LEVY_CHANNELS


def _levy_map_to_numpy(tensor, batch_idx=0):
    """[B, 1, H, W] tensor -> [H, W] float32 array for batch element ``batch_idx``."""
    return tensor[batch_idx, 0].detach().float().cpu().numpy()


def save_levy_uncertainty(levy_uncertainty, timestamp, save_path):
    """
    Dump one frame's Levy trust map to a compressed .npz.

    Stores the aggregated maps ('evidence', 'disagreement', 'sigma',
    'sigma_calibrated') plus each pyramid level's own maps under a '_l{i}'
    suffix, so you can tell which scale drives the signal.

    Parameters
    ----------
    levy_uncertainty : dict or None
        ``output_dict['levy_uncertainty']``. None (plain PyramidFusion) is a no-op.
    timestamp : int
        Frame index, used for the filename.
    save_path : str
        Directory to write ``%04d_levy.npz`` into.
    """
    if levy_uncertainty is None:
        return

    arrays = {}
    for key in ('coverage', 'evidence', 'disagreement', 'sigma', 'sigma_calibrated'):
        if levy_uncertainty.get(key, None) is not None:
            arrays[key] = _levy_map_to_numpy(levy_uncertainty[key])

    for level, scale_map in enumerate(levy_uncertainty.get('per_scale', [])):
        for key in ('evidence', 'disagreement', 'sigma', 'coverage'):
            arrays[f'{key}_l{level}'] = _levy_map_to_numpy(scale_map[key])

    np.savez_compressed(
        os.path.join(save_path, '%04d_levy.npz' % timestamp), **arrays)


def summarize_levy_uncertainty(levy_uncertainty):
    """
    Per-frame summary statistics of the aggregated trust map.

    Reported in two blocks, because the two populations answer different
    questions and pooling them is misleading:

    ``grid``
        every cell at least one agent contributes to. The ego is warped by the
        identity, so in practice this is its entire BEV extent -- these numbers
        describe the whole plane, mostly empty road.
    ``overlap``
        cells where two or more agents contribute. tr Sigma is identically zero
        below that (the variance of a single sample), so this is the only
        population in which disagreement and sigma mean anything at all. None
        for a frame with no collaborator overlap.

    Returns
    -------
    dict or None
        ``{'coverage': {...}, 'grid': {channel: {...}}, 'overlap': {...} or None}``
        where each channel carries mean / p50 / p90 / p99 / max.
        None when there is no Levy uncertainty to summarise.
    """
    if levy_uncertainty is None:
        return None

    coverage = _levy_map_to_numpy(levy_uncertainty['coverage'])
    channels = {key: _levy_map_to_numpy(levy_uncertainty[key])
                for key in LEVY_CHANNELS}

    def block(mask):
        if not mask.any():
            return None
        out = {}
        for key, values in channels.items():
            selected = values[mask]
            out[key] = {
                'mean': float(selected.mean()),
                'p50': float(np.percentile(selected, 50)),
                'p90': float(np.percentile(selected, 90)),
                'p99': float(np.percentile(selected, 99)),
                'max': float(selected.max()),
            }
        return out

    covered = coverage >= 1
    overlap = coverage >= 2

    return {
        'coverage': {
            'covered_frac': float(covered.mean()),
            'overlap_frac': float(overlap.mean()),
            'mean_agents': float(coverage[covered].mean()) if covered.any() else 0.0,
            'max_agents': float(coverage.max()),
        },
        'grid': block(covered),
        'overlap': block(overlap),
    }


# Blueprint Sec. 8 "Sensing Triage": 0.20 ~= the p90 of the overlap-cell sigma
# distribution for the m1_base_levy epoch-17 checkpoint (levy_stats.json:
# overlap.sigma.p90 = 0.211).
LEVY_SPARSIFY_DEFAULT_SIGMA = 0.20


def _levy_grid_indices(xy, pc_range, hw):
    """
    Map metric ``(x, y)`` coordinates to ``(row, col)`` indices on the BEV trust
    map.

    Mirrors :func:`visualize_levy_uncertainty`: the map is ``[H(y), W(x)]`` with
    row 0 at ``y_min`` and column 0 at ``x_min``.

    Parameters
    ----------
    xy : np.ndarray
        ``(K, 2)`` metric coordinates.
    pc_range : list
        ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    hw : tuple
        ``(H, W)`` of the trust map.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``row`` and ``col`` int arrays, clipped to the grid.
    """
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    H, W = hw
    x = np.asarray(xy[:, 0], dtype=np.float64)
    y = np.asarray(xy[:, 1], dtype=np.float64)
    col = np.floor((x - x_min) / (x_max - x_min) * W).astype(np.int64)
    row = np.floor((y - y_min) / (y_max - y_min) * H).astype(np.int64)
    return np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)


def levy_sigma_under_boxes(pred_box_tensor, levy_uncertainty, pc_range,
                           channel='sigma', reduce='mean'):
    """
    Sample the Levy spatial trust map under every predicted 3D box.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor or None
        ``(N, 8, 3)`` corner tensor in the ego metric frame. ``None`` yields an
        empty array.
    levy_uncertainty : dict or None
        ``output_dict['levy_uncertainty']``. ``None`` (plain PyramidFusion)
        yields an empty array.
    pc_range : list
        ``[x_min, y_min, z_min, x_max, y_max, z_max]`` of the BEV grid
        (``hypes['postprocess']['gt_range']``).
    channel : str
        Which map to read -- ``'sigma'`` (raw feature-space ``sigma_hat``, the
        blueprint's ``\\hat{\\sigma}``) or ``'sigma_calibrated'``.
    reduce : str
        ``'center'`` looks up the single cell under the box centre; ``'mean'`` /
        ``'max'`` reduce over every cell the box's xy footprint covers.

    Returns
    -------
    np.ndarray
        ``(N,)`` float array of per-box sigma; empty when there is nothing to
        sample.
    """
    if pred_box_tensor is None or levy_uncertainty is None:
        return np.zeros(0, dtype=np.float64)
    levy_map = levy_uncertainty.get(channel, None)
    if levy_map is None:
        return np.zeros(0, dtype=np.float64)
    if isinstance(levy_map, np.ndarray):
        levy_map = np.squeeze(levy_map)
    else:
        levy_map = _levy_map_to_numpy(levy_map)
    H, W = levy_map.shape[-2:]

    boxes = pred_box_tensor.detach().cpu().numpy()  # (N, 8, 3)
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    centers = boxes[:, :, :2].mean(axis=1)  # (N, 2)
    if reduce == 'center':
        row, col = _levy_grid_indices(centers, pc_range, (H, W))
        return levy_map[row, col].astype(np.float64)

    out = np.zeros(boxes.shape[0], dtype=np.float64)
    for n in range(boxes.shape[0]):
        r, c = _levy_grid_indices(boxes[n, :, :2], pc_range, (H, W))
        patch = levy_map[r.min():r.max() + 1, c.min():c.max() + 1]
        if patch.size == 0:  # degenerate box, fall back to the centre cell
            rr, cc = _levy_grid_indices(centers[n:n + 1], pc_range, (H, W))
            out[n] = levy_map[rr[0], cc[0]]
        else:
            out[n] = patch.max() if reduce == 'max' else patch.mean()
    return out


def sparsify_predictions_by_levy(pred_box_tensor, pred_score, levy_uncertainty,
                                 pc_range,
                                 sigma_threshold=LEVY_SPARSIFY_DEFAULT_SIGMA,
                                 mode='attenuate', attenuation=0.5,
                                 channel='sigma', reduce='mean',
                                 direction='low'):
    """
    Sparsification Filter (blueprint Sec. 8, "Sensing Triage").

    Look up the Levy sigma map under each predicted box and flag the boxes that
    sit on the wrong side of ``sigma_threshold``, then either drop them or lower
    their confidence.

    ``direction`` picks which side is "wrong":

    * ``'low'``  -- flag ``sigma_hat < sigma_threshold``. The error diagnostic on
      the m1_base_levy checkpoint shows sigma is *inverted*: high sigma marks the
      correct boxes and the low-sigma band is the noisy background where the
      false positives live, so this is the direction that actually removes FPs.
    * ``'high'`` -- flag ``sigma_hat > sigma_threshold``. The original blueprint
      behaviour (down-weight the high-uncertainty zone).

    Parameters
    ----------
    pred_box_tensor : torch.Tensor or None
        ``(N, 8, 3)``.
    pred_score : torch.Tensor or None
        ``(N,)``.
    levy_uncertainty : dict or None
        ``None`` (plain PyramidFusion) makes this a no-op passthrough.
    pc_range : list
        ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
    sigma_threshold : float
        High-uncertainty cut on ``sigma_hat``.
    mode : str
        ``'attenuate'`` -> ``score *= attenuation`` for flagged boxes;
        ``'drop'``      -> flagged boxes are removed.
    attenuation : float
        Multiplicative confidence penalty in ``'attenuate'`` mode.
    channel, reduce : str
        Passed to :func:`levy_sigma_under_boxes`.
    direction : str
        ``'low'`` -> flag ``sigma_hat < sigma_threshold``;
        ``'high'`` -> flag ``sigma_hat > sigma_threshold``.

    Returns
    -------
    pred_box_tensor, pred_score
        Same types as the inputs (possibly fewer rows; both ``None`` if every
        box was dropped).
    info : dict
        ``{'sigma', 'flagged', 'n_total', 'n_flagged', 'mode',
        'sigma_threshold', 'direction'}``.
    """
    info = {'sigma': np.zeros(0, dtype=np.float64),
            'flagged': np.zeros(0, dtype=bool),
            'n_total': 0, 'n_flagged': 0, 'mode': mode,
            'sigma_threshold': float(sigma_threshold),
            'direction': direction}
    if pred_box_tensor is None or levy_uncertainty is None \
            or pred_box_tensor.shape[0] == 0:
        return pred_box_tensor, pred_score, info

    sigma = levy_sigma_under_boxes(pred_box_tensor, levy_uncertainty, pc_range,
                                   channel=channel, reduce=reduce)
    if direction == 'low':
        flagged = sigma < float(sigma_threshold)
    elif direction == 'high':
        flagged = sigma > float(sigma_threshold)
    else:
        raise ValueError(f"unknown sparsify direction {direction!r}")
    info.update({'sigma': sigma, 'flagged': flagged,
                 'n_total': int(sigma.shape[0]),
                 'n_flagged': int(flagged.sum())})

    if not flagged.any():
        return pred_box_tensor, pred_score, info

    if mode == 'drop':
        keep = torch.as_tensor(~flagged, device=pred_box_tensor.device)
        pred_box_tensor = pred_box_tensor[keep]
        pred_score = pred_score[keep] if pred_score is not None else None
        if pred_box_tensor.shape[0] == 0:
            return None, None, info
    elif mode == 'attenuate':
        if pred_score is not None:
            penalty = torch.ones_like(pred_score)
            penalty[torch.as_tensor(flagged, device=pred_score.device)] = \
                float(attenuation)
            pred_score = pred_score * penalty
    else:
        raise ValueError(f"unknown sparsify mode {mode!r}")

    return pred_box_tensor, pred_score, info


def levy_match_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, iou_thresh):
    """
    Per-box true/false-positive labels, indexed by the *input* box order.

    Mirrors :func:`opencood.utils.eval_utils.caluclate_tp_fp` exactly -- greedy
    matching in descending score order, each ground-truth box consumed by at
    most one detection -- but keeps the correspondence back to the original box
    indices, which caluclate_tp_fp loses when it appends to score-sorted lists.
    That correspondence is what lets a per-box quantity such as the Levy sigma
    be paired with whether the box was actually right.

    Returns
    -------
    np.ndarray
        ``(N,)`` bool, True where the box is a true positive at ``iou_thresh``.
    """
    if pred_box_tensor is None or pred_box_tensor.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    det_boxes = torch_tensor_to_numpy(pred_box_tensor)
    det_score = torch_tensor_to_numpy(pred_score)
    is_tp = np.zeros(det_boxes.shape[0], dtype=bool)

    if gt_box_tensor is None or gt_box_tensor.shape[0] == 0:
        return is_tp
    gt_boxes = torch_tensor_to_numpy(gt_box_tensor)

    det_polygon_list = list(common_utils.convert_format(det_boxes))
    gt_polygon_list = list(common_utils.convert_format(gt_boxes))

    for idx in np.argsort(-det_score):
        if len(gt_polygon_list) == 0:
            break
        ious = common_utils.compute_iou(det_polygon_list[idx], gt_polygon_list)
        if np.max(ious) < iou_thresh:
            continue
        is_tp[idx] = True
        gt_polygon_list.pop(int(np.argmax(ious)))

    return is_tp


# The canonical implementations live in levy_metrics, which is the leaf module
# of the Levy tooling; these aliases keep the private names this file already
# used. One definition, so the diagnostic here and the offline harnesses cannot
# drift apart.
_average_ranks = levy_metrics.average_ranks
_auroc = levy_metrics.auroc


def _trapezoid(y, x):
    """Area under y(x) by the trapezoid rule; used for the AUSE integral."""
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def levy_error_metrics(sigma, is_tp, sigma_threshold, n_steps=101,
                       direction='low'):
    """
    Does sigma predict detection error?

    Treats a false positive as the event to be caught. Reports the separation of
    the two sigma distributions, the ranking quality (AUROC, AUSE), and what the
    operating threshold actually buys.

    AUROC is the decisive number: 0.5 means sigma carries no information about
    whether a box is wrong, above 0.5 means high sigma marks false positives,
    and below 0.5 means the signal is *inverted* -- high sigma marks the boxes
    that are right, so a ``direction='low'`` filter (flag sigma below the
    threshold) is the one that removes false positives.

    AUSE is the area between the sigma-ordered and oracle-ordered
    sparsification curves: progressively discard the boxes the filter would
    flag first -- most-uncertain for ``direction='high'``, least-uncertain for
    ``direction='low'`` -- and watch the false-positive rate of what is left.
    0 is a perfect ranking; larger is worse.

    ``direction`` also selects which side of ``sigma_threshold`` the reported
    operating point (``at_threshold``) flags.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    is_tp = np.asarray(is_tp, dtype=bool)
    n = sigma.shape[0]
    if n == 0:
        return None

    is_fp = ~is_tp
    n_fp = int(is_fp.sum())
    base_rate = n_fp / n

    def dist(mask):
        if not mask.any():
            return None
        values = sigma[mask]
        return {'n': int(mask.sum()), 'mean': float(values.mean()),
                'p50': float(np.percentile(values, 50)),
                'p90': float(np.percentile(values, 90))}

    # Sparsification curves: drop the most-uncertain fraction, keep the rest.
    fractions = np.linspace(0.0, 0.95, n_steps)

    def curve(order):
        errors = is_fp[order]
        return np.array([errors[int(round(f * n)):].mean()
                         if int(round(f * n)) < n else 0.0 for f in fractions])

    # Order the boxes the way the filter would flag them first: highest sigma
    # first for 'high', lowest sigma first for 'low'.
    flag_order_key = sigma if direction == 'low' else -sigma
    by_sigma = curve(np.argsort(flag_order_key, kind='mergesort'))
    by_oracle = curve(np.argsort(-is_fp.astype(np.float64), kind='mergesort'))

    if direction == 'low':
        flagged = sigma < float(sigma_threshold)
    else:
        flagged = sigma > float(sigma_threshold)
    n_flagged = int(flagged.sum())
    precision = float(is_fp[flagged].mean()) if n_flagged else float('nan')

    return {
        'n_boxes': n, 'n_tp': int(is_tp.sum()), 'n_fp': n_fp,
        'base_fp_rate': base_rate,
        'sigma_tp': dist(is_tp), 'sigma_fp': dist(is_fp),
        'auroc': _auroc(sigma, is_fp),
        'ause': _trapezoid(by_sigma - by_oracle, fractions),
        'percentiles': {f'p{q}': float(np.percentile(sigma, q))
                        for q in (10, 25, 50, 75, 90, 99)},
        'at_threshold': {
            'sigma_threshold': float(sigma_threshold),
            'direction': direction,
            'flagged_frac': n_flagged / n,
            'precision': precision,
            'lift': precision / base_rate if base_rate > 0 else float('nan'),
            'recall': float(flagged[is_fp].mean()) if n_fp else float('nan'),
        },
    }


def visualize_levy_uncertainty(levy_uncertainty, pc_range, save_path,
                               infer_result=None, left_hand=False):
    """
    Render the coverage map and the three trust-map channels as stacked BEV
    heatmaps.

    Coverage comes first deliberately: it is the exact per-cell agent count, and
    disagreement and sigma are structurally zero wherever it is below 2, so the
    dark regions of those two panels are only interpretable against it.

    Ground-truth and predicted box centres are overlaid so that a disagreement
    or evidence structure can be read against the objects in the scene.

    Parameters
    ----------
    levy_uncertainty : dict or None
        ``output_dict['levy_uncertainty']``. None (plain PyramidFusion) is a no-op.
    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax], used to put the maps in metres.
    save_path : str
        Output .png path.
    infer_result : dict, optional
        Carries 'gt_box_tensor' / 'pred_box_tensor' for the overlay.
    left_hand : bool
        OPV2V / V2XSet are left-hand (CARLA). Inverts the y axis so the render
        matches the BEV lidar visualizations written next to it; the underlying
        data placement is unaffected.
    """
    if levy_uncertainty is None:
        return

    from matplotlib import pyplot as plt

    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    # The BEV feature grid is [ny, nx] (point_pillar_scatter.py: "It put y axis
    # (in lidar frame) as image height"), so row 0 is y_min and column 0 is
    # x_min. With origin='lower' this extent places the data at its true metric
    # position -- the same frame the box-centre scatter is drawn in.
    extent = [x_min, x_max, y_min, y_max]

    def box_centers(tensor):
        if tensor is None or tensor.shape[0] == 0:
            return None
        return tensor.detach().cpu().numpy()[:, :, :2].mean(axis=1)

    infer_result = infer_result or {}
    gt_xy = box_centers(infer_result.get('gt_box_tensor', None))
    pred_xy = box_centers(infer_result.get('pred_box_tensor', None))

    width = (x_max - x_min) / 40
    height = (y_max - y_min) / 40
    fig, axes = plt.subplots(len(LEVY_VIS_CHANNELS), 1,
                             figsize=(width, height * len(LEVY_VIS_CHANNELS) * 1.25))

    for ax, key in zip(np.atleast_1d(axes), LEVY_VIS_CHANNELS):
        values = _levy_map_to_numpy(levy_uncertainty[key])
        if key == 'coverage':
            # An agent count, so show it whole rather than clipped: the step
            # from 1 to 2 agents is the boundary the other panels depend on.
            vmax = max(float(values.max()), 1.0)
            cmap = 'cividis'
        else:
            # Evidence spans orders of magnitude in 'logit' score_mode; the
            # other two are heavy tailed. Clip the colour scale at p99 so a
            # handful of cells cannot wash the map out.
            positive = values[values > 0]
            vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
            vmax = max(vmax, 1e-12)
            cmap = 'viridis'
        image = ax.imshow(values, origin='lower', extent=extent, aspect='equal',
                          cmap=cmap, vmin=0.0, vmax=vmax)
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)

        # Match the vis_ BEV renders: Canvas_BEV_heading_right leaves y alone for
        # left-hand data and draws rows downward, so y_min belongs at the top.
        # Flip the *view*, never the data -- inverting the axis carries the
        # heatmap and the box overlay together, whereas flipping the extent
        # would mirror the heatmap out from under the markers.
        ax.set_ylim(y_max, y_min) if left_hand else ax.set_ylim(y_min, y_max)

        if gt_xy is not None:
            ax.scatter(gt_xy[:, 0], gt_xy[:, 1], s=8, facecolors='none',
                       edgecolors='#00ff00', linewidths=0.6, label='gt')
        if pred_xy is not None:
            ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=8, marker='x',
                       c='#ff2d2d', linewidths=0.6, label='pred')

        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
