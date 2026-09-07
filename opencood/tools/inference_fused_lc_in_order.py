"""
Incremental-collaborator evaluation for the LiDAR+Camera *fused* encoder
(model core_method `fused_pyramid_collab`, every agent = sensor_type
`lidar_camera` / `m1`).

This is a copy of `inference_heter_in_order.py` specialised for the homogeneous
fused-agent setting:
  * every collaborator is mapped to m1 (fused lidar+camera), so the sweep is
    "1 fused agent -> 2 -> 3 -> 4 fused agents" (ego always m1),
  * no open3d / visualization dependency (eval only needs boxes),
  * uses the `_in_order.json` assignment (defines the order agents are added)
    and comm_range 180, exactly like the standard HEAL in-order eval.

Usage:
    python opencood/tools/inference_fused_lc_in_order.py \
        --model_dir opencood/logs/HEAL_m1_based/stage_fused_lc/m1_lidar_camera \
        --range 102.4,51.2 --use_cav "[1,2,3,4]"
"""

import argparse
import importlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict
from opencood.visualization import simple_vis   # matplotlib-based BEV vis (no open3d)

torch.multiprocessing.set_sharing_strategy('file_system')


def test_parser():
    parser = argparse.ArgumentParser(description="fused lidar-camera in-order eval")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='model dir; its config.yaml + checkpoint are used')
    parser.add_argument('--fusion_method', type=str, default='intermediate',
                        help='intermediate (default) or no / single')
    parser.add_argument('--range', type=str, default="102.4,51.2",
                        help="eval range as x_max,y_max -> [-x_max,x_max]x[-y_max,y_max]. "
                             "Default matches the fused model's training range.")
    parser.add_argument('--use_cav', type=str, default="[1,2,3,4]",
                        help="list of collaborator counts to evaluate")
    parser.add_argument('--no_score', action='store_true')
    parser.add_argument('--dataset_subset', action='store_true',
                        help="evaluate on a small slice (debug only)")
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help="save a BEV png every N frames; 0 disables vis")
    parser.add_argument('--note', default="", type=str)
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['intermediate', 'no', 'single']

    hypes = yaml_utils.load_yaml(None, opt)

    # ---- range / anchor rebuild (same mechanism as inference_heter_in_order) ----
    x_max = float(opt.range.split(',')[0])
    y_max = float(opt.range.split(',')[1])
    opt.note += f"_{x_max}_{y_max}"
    z_min = hypes['postprocess']['anchor_args']['cav_lidar_range'][2]
    z_max = hypes['postprocess']['anchor_args']['cav_lidar_range'][5]
    new_cav_range = [-x_max, -y_max, z_min, x_max, y_max, z_max]

    hypes = update_dict(hypes, {
        "cav_lidar_range": new_cav_range,
        "lidar_range": new_cav_range,
        "gt_range": new_cav_range,
    })

    # Every collaborator -> m1 (fused lidar+camera). This is the whole point of
    # "use only this type of agent".
    hypes = update_dict(hypes, {
        "mapping_dict": {"m1": "m1", "m2": "m1", "m3": "m1", "m4": "m1"},
    })

    # rebuild anchors for the (possibly) new range
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = None
    for name, func in yaml_utils_lib.__dict__.items():
        if name == hypes["yaml_parser"]:
            parser_func = func
    hypes = parser_func(hypes)

    hypes['validate_dir'] = hypes['test_dir']
    assert "test" in hypes['validate_dir']

    # left-hand coord for BEV vis: OPV2V / V2XSet
    left_hand = ("OPV2V" in hypes['test_dir']) or ("V2XSET" in hypes['test_dir'])

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    np.random.seed(303)

    if opt.fusion_method == 'intermediate':
        hypes['fusion']['core_method'] += 'infer'          # -> intermediateheterinfer
    hypes['comm_range'] = 180
    hypes['heter']['assignment_path'] = \
        hypes['heter']['assignment_path'].replace(".json", "_in_order.json")
    hypes = update_dict(hypes, {"ego_modality": 'm1'})
    hypes['heter'].setdefault('lidar_channels_dict', {'m3': 32})

    for use_cav in eval(opt.use_cav):
        hypes['use_cav'] = use_cav

        print(f'\n===== Dataset building for use_cav={use_cav} =====')
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)
        opencood_dataset.force_all_same_agent_modality = True
        if opt.dataset_subset:
            opencood_dataset = Subset(opencood_dataset, range(1200, 1305))

        data_loader = DataLoader(opencood_dataset,
                                 batch_size=1,
                                 num_workers=4,
                                 collate_fn=opencood_dataset.collate_batch_test,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)

        result_stat = {
            0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
            0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
            0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
        }

        infer_info = opt.fusion_method + opt.note + f"_use_cav{use_cav}"

        for i, batch_data in enumerate(data_loader):
            if batch_data is None:
                continue
            if i % 20 == 0:
                print(f"{infer_info}_{i}")

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'intermediate':
                    infer_result = inference_utils.inference_intermediate_fusion(
                        batch_data, model, opencood_dataset)
                elif opt.fusion_method == 'no':
                    infer_result = inference_utils.inference_no_fusion(
                        batch_data, model, opencood_dataset)
                elif opt.fusion_method == 'single':
                    infer_result = inference_utils.inference_no_fusion(
                        batch_data, model, opencood_dataset, single_gt=True)

                pred_box_tensor = infer_result['pred_box_tensor']
                gt_box_tensor = infer_result['gt_box_tensor']
                pred_score = infer_result['pred_score']

                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                               gt_box_tensor, result_stat, iou)

                # ---- BEV visualization (matplotlib; no open3d) ----
                if opt.save_vis_interval > 0 and (i % opt.save_vis_interval == 0) \
                        and (pred_box_tensor is not None or gt_box_tensor is not None):
                    if not opt.no_score:
                        infer_result.update({'score_tensor': pred_score})
                    if getattr(opencood_dataset, "heterogeneous", False):
                        cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                        infer_result.update({"cav_box_np": cav_box_np,
                                             "agent_modality_list": agent_modality_list})
                    vis_dir = os.path.join(opt.model_dir, f'vis_{infer_info}')
                    os.makedirs(vis_dir, exist_ok=True)
                    simple_vis.visualize(infer_result,
                                         batch_data['ego']['origin_lidar'][0],
                                         hypes['postprocess']['gt_range'],
                                         os.path.join(vis_dir, 'bev_%05d.png' % i),
                                         method='bev',
                                         left_hand=left_hand)
            torch.cuda.empty_cache()

        _, ap50, ap70 = eval_utils.eval_final_results(
            result_stat, opt.model_dir, infer_info)
        print(f"[use_cav={use_cav}]  AP@0.50 = {ap50:.4f}   AP@0.70 = {ap70:.4f}")


if __name__ == '__main__':
    main()
