# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
from collections import OrderedDict

import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy
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
        OPV2V / V2XSet are left-hand (CARLA); flip y so the render matches the
        BEV lidar visualizations written next to it.
    """
    if levy_uncertainty is None:
        return

    from matplotlib import pyplot as plt

    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    extent = [x_min, x_max, y_max, y_min] if left_hand \
        else [x_min, x_max, y_min, y_max]

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
