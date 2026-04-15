"""
-*- coding: utf-8 -*-
Author: George Panaretou

Dual-ego heterogeneous inference.

The ego vehicle is treated as TWO virtual agents sharing the same pose:
  - Agent 0: ego lidar   (m1)
  - Agent 1: ego camera  (m2)  [only when use_cav=2]

GT boxes are collected from ALL real vehicles in the scene, so the
evaluation reflects full-scene coverage even though only the ego's
sensors contribute to the collaborative feature extraction.

use_cav=1  →  single-modality ego (lidar only, baseline)
use_cav=2  →  dual-modality ego   (lidar + camera fusion)

Example command (intermediate fusion):
    python opencood/tools/inference_heter_dual_ego.py \
        --model_dir opencood/logs_HEAL/Pyramid_collaboration_m1_base/final_infer \
        --range 102.4,102.4 \
        --use_cav "[1,2]"
"""

import argparse
import os
import time
from typing import OrderedDict
import importlib
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Subset
import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils, my_vis, simple_vis
from opencood.utils.common_utils import update_dict
torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(description="Dual-ego heterogeneous inference")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to saved model checkpoint directory')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='Interval for saving BEV visualizations')
    parser.add_argument('--save_npy', action='store_true',
                        help='Whether to save prediction/gt results as .npy files')
    parser.add_argument('--range', type=str, default="204.8,102.4",
                        help="Detection range, e.g. '102.4,102.4' for [-102.4,+102.4] in x and y")
    parser.add_argument('--no_score', action='store_true',
                        help="Do not include prediction scores in result dict")
    parser.add_argument('--use_cav', type=str, default="[1,2]",
                        help="1=lidar-only ego, 2=lidar+camera dual ego")
    parser.add_argument('--note', default="", type=str, help="Optional suffix appended to output filenames")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [
            x_min, y_min,
            hypes['postprocess']['anchor_args']['cav_lidar_range'][2],
            x_max, y_max,
            hypes['postprocess']['anchor_args']['cav_lidar_range'][5]
        ]

        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        hypes = update_dict(hypes, {
            "mapping_dict": {
                "m1": "m1",
                "m2": "m2",
                "m3": "m3",
                "m4": "m4"
            }
        })

        # Reload anchor boxes after range update
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']

    # Left-hand coordinate system: OPV2V / V2XSet
    # Right-hand coordinate system: V2X-Sim 2.0 / DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False
    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"Resumed from epoch {resume_epoch}.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    np.random.seed(303)

    # Switch fusion core_method to the dual-ego inference dataset
    if opt.fusion_method == 'intermediate':
        hypes['fusion']['core_method'] += 'dualegoinfer'

    hypes['comm_range'] = 180

    # Keep the _in_order assignment so that real vehicles in the scene have
    # consistent modality labels for GT box generation.
    hypes['heter']['assignment_path'] = hypes['heter']['assignment_path'].replace(
        ".json", "_in_order.json"
    )

    hypes = update_dict(hypes, {"ego_modality": 'm1'})

    use_cav_list = eval(opt.use_cav)
    # Clamp to valid range for dual ego (1 = lidar only, 2 = lidar + camera)
    use_cav_list = [min(max(int(v), 1), 2) for v in use_cav_list]
    use_cav_list = sorted(set(use_cav_list))  # deduplicate

    for use_cav in use_cav_list:
        hypes['use_cav'] = use_cav

        print('Building Dataset')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        data_loader = DataLoader(
            opencood_dataset,
            batch_size=1,
            num_workers=4,
            collate_fn=opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False
        )

        result_stat = {
            0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        }

        modality_label = 'lidar_only' if use_cav == 1 else 'lidar_camera'
        infer_info = opt.fusion_method + opt.note + f"_dual_ego_{modality_label}"

        for i, batch_data in enumerate(data_loader):
            print(f"{infer_info}_{i}")
            if batch_data is None:
                continue

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(
                        batch_data, model, opencood_dataset
                    )
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(
                        batch_data, model, opencood_dataset
                    )
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(
                        batch_data, model, opencood_dataset, single_gt=True
                    )
                else:
                    raise NotImplementedError(
                        f"fusion_method '{opt.fusion_method}' is not supported for dual-ego inference. "
                        "Use 'intermediate', 'no', or 'single'."
                    )

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']

                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5)
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7)

                if not opt.no_score:
                    infer_result.update({'score_tensor': pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                    infer_result.update({
                        "cav_box_np": cav_box_np,
                        "agent_modality_list": agent_modality_list
                    })

                if (i % opt.save_vis_interval == 0) and \
                        (pred_box_tensor is not None or gt_box_tensor is not None):
                    vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                    simple_vis.visualize(
                        infer_result,
                        batch_data['ego']['origin_lidar'][0],
                        hypes['postprocess']['gt_range'],
                        vis_save_path,
                        method='bev',
                        left_hand=left_hand
                    )

            torch.cuda.empty_cache()

        _, ap50, ap70 = eval_utils.eval_final_results(
            result_stat, opt.model_dir, infer_info
        )


if __name__ == '__main__':
    main()
