# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import importlib
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import inference_utils, train_utils
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='interval of saving visualization')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--pos_std', type=float, default=0.0,
                        help='std (metres) of the zero-mean Gaussian noise added '
                             'to every agent\'s x/y localisation before fusion '
                             '(blueprint Protocol 2). 0 = clean poses.')
    parser.add_argument('--rot_std', type=float, default=0.0,
                        help='std (degrees) of the zero-mean Gaussian noise '
                             'added to every agent\'s yaw before fusion '
                             '(blueprint Protocol 2). 0 = clean poses.')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--use_cav', type=int, default=None,
                        help='evaluate with exactly this many collaborators, as '
                             'inference_heter_in_order.py does. Switches the '
                             'dataset to its *infer variant, which uses every '
                             'CAV to generate ground truth but lets only the '
                             'first --use_cav of them contribute features. '
                             'Without it the natural per-frame CAV count (up to '
                             'max_cav) participates, which is a different -- and '
                             'on OPV2V harder -- operating point than the one '
                             "HEAL's published tables use.")
    parser.add_argument('--lidar_corrupt', type=float, nargs='?', const=-1.0,
                        default=None, metavar='RATIO',
                        help='enable LiDAR point dropout at this ratio, e.g. '
                             '--lidar_corrupt 0.5. Bare --lidar_corrupt keeps '
                             "the config's own dropout_ratio. Overrides the "
                             'config; both dataset variants honour it '
                             'identically, so it no longer depends on --use_cav.')
    parser.add_argument('--no_lidar_corrupt', action='store_true',
                        help='force LiDAR dropout off, overriding a config that '
                             'enables it.')
    parser.add_argument('--assignment', type=str, default=None,
                        help='override heter.assignment_path with this json, to '
                             'evaluate a specific alliance composition (e.g. a '
                             'fixed LiDAR+camera pair). Pair with --use_cav to '
                             'fix how many of the listed agents collaborate.')
    parser.add_argument('--comm_range', type=float, default=None,
                        help='override comm_range (metres). The config ships 70, '
                             'at which ~40%% of OPV2V test frames have no '
                             'collaborator within range and the ego runs alone; '
                             'inference_heter_in_order.py uses 180, which is why '
                             "HEAL's use_cav tables get consistent agent counts. "
                             'Set it when the alliance composition itself is the '
                             'variable under test.')
    parser.add_argument('--in_order_assign', action='store_true',
                        help="swap in the *_in_order.json modality assignment, "
                             "comm_range=180 and ego_modality=m1, as "
                             "inference_heter_in_order.py does. Required to "
                             "reproduce HEAL's published use_cav tables. NOTE "
                             "the in-order assignment for OPV2V is LiDAR-only "
                             "(m1/m3), so it removes all camera agents.")
    parser.add_argument('--max_frames', type=int, default=None,
                        help='stop after this many frames. Smoke-testing only -- '
                             'the AP/FP it reports are on a truncated split and '
                             'are not comparable to a full run.')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single'] 

    hypes = yaml_utils.load_yaml(None, opt)

    if 'heter' in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(',')[0]), eval(opt.range.split(',')[0])
        y_min, y_max = -eval(opt.range.split(',')[1]), eval(opt.range.split(',')[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [x_min, y_min, hypes['postprocess']['anchor_args']['cav_lidar_range'][2], \
                            x_max, y_max, hypes['postprocess']['anchor_args']['cav_lidar_range'][5]]

        # replace all appearance
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range
        })

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

        
    
    hypes['validate_dir'] = hypes['test_dir']
    if "OPV2V" in hypes['test_dir'] or "v2xsim" in hypes['test_dir']:
        assert "test" in hypes['validate_dir']
    
    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = True if ("OPV2V" in hypes['test_dir'] or "V2XSET" in hypes['test_dir']) else False

    print(f"Left hand visualizing: {left_hand}")

    if 'box_align' in hypes.keys():
        hypes['box_align']['val_result'] = hypes['box_align']['test_result']

    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # setting noise
    np.random.seed(303)

    # Pose Noise (blueprint Protocol 2). Misaligning the agents is what
    # manufactures ghost boxes: each agent's BEV features land on the wrong
    # cells, so features for a real object elsewhere pile up on empty ground.
    # The dataset applies this per agent per frame in add_noise_data_dict().
    if opt.pos_std > 0 or opt.rot_std > 0:
        hypes['noise_setting'] = {
            'add_noise': True,
            'args': {'pos_std': opt.pos_std, 'rot_std': opt.rot_std,
                     'pos_mean': 0.0, 'rot_mean': 0.0},
        }
        print(f"[noise] pose noise on: pos_std={opt.pos_std:g} m, "
              f"rot_std={opt.rot_std:g} deg (zero-mean Gaussian, per agent)")
    else:
        hypes['noise_setting'] = {'add_noise': False}
    
    # Fixed collaborator count. Mirrors inference_heter_in_order.py: the dataset
    # core_method gains an 'infer' suffix, which selects the subclass that reads
    # hypes['use_cav'] and skips feature extraction for agents past that index
    # while still stacking their objects into the ground truth.
    if opt.use_cav is not None:
        _fusion_cm = hypes['fusion']['core_method']
        if not _fusion_cm.endswith('infer'):
            hypes['fusion']['core_method'] = _fusion_cm + 'infer'
        hypes['use_cav'] = opt.use_cav
        opt.note += f"_use_cav{opt.use_cav}"
        print(f"[eval] fixed collaborator count: use_cav={opt.use_cav} "
              f"(dataset {hypes['fusion']['core_method']})")

    # The rest of inference_heter_in_order.py's setup. Kept behind its own flag
    # because it is NOT cosmetic: the *_in_order.json assignment is a different
    # experiment, not a reordering of the same one -- it contains only m1/m3
    # LiDAR agents, so it removes every camera agent from the split.
    if opt.comm_range is not None:
        print(f"[eval] comm_range {hypes.get('comm_range')} -> {opt.comm_range}")
        hypes['comm_range'] = opt.comm_range
        opt.note += f"_cr{opt.comm_range:g}"

    if opt.assignment is not None:
        if not os.path.exists(opt.assignment):
            raise SystemExit(f"--assignment: no such file {opt.assignment}")
        hypes['heter']['assignment_path'] = opt.assignment
        print(f"[eval] modality assignment: {opt.assignment}")

    if opt.in_order_assign:
        hypes['comm_range'] = 180
        hypes['heter']['assignment_path'] = \
            hypes['heter']['assignment_path'].replace(".json", "_in_order.json")
        hypes = update_dict(hypes, {"ego_modality": 'm1'})
        opt.note += "_inorder"
        print(f"[eval] in-order assignment: "
              f"{hypes['heter']['assignment_path']}, comm_range=180, ego=m1")

    # LiDAR dropout. Both dataset variants now read the same key, so this is a
    # plain config override rather than a second, independent switch. The
    # datasets announce the resolved setting themselves at construction.
    if opt.lidar_corrupt is not None and opt.no_lidar_corrupt:
        raise SystemExit("--lidar_corrupt and --no_lidar_corrupt are mutually "
                         "exclusive.")
    if opt.no_lidar_corrupt:
        hypes.setdefault('lidar_corrupt', {})['enabled'] = False
    elif opt.lidar_corrupt is not None:
        _lc = hypes.setdefault('lidar_corrupt', {})
        _lc['enabled'] = True
        if opt.lidar_corrupt >= 0:
            _lc['dropout_ratio'] = opt.lidar_corrupt
        elif 'dropout_ratio' not in _lc:
            raise SystemExit("bare --lidar_corrupt needs a dropout_ratio in the "
                             "config; pass --lidar_corrupt RATIO instead.")
        opt.note += f"_corrupt{_lc['dropout_ratio']:g}"

    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    # Pose noise is drawn inside __getitem__, which runs in the worker processes.
    # A single np.random.seed() in the parent is inherited by every forked worker,
    # so they share one stream and which draw lands on which frame depends on
    # scheduling -- two runs with identical --pos_std then differ. Seeding each
    # worker deterministically makes a noise level reproducible across runs.
    def _seed_worker(worker_id):
        np.random.seed(303 + worker_id)

    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False,
                            worker_init_fn=_seed_worker)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note

    for i, batch_data in enumerate(data_loader):
        if opt.max_frames is not None and i >= opt.max_frames:
            print(f"[eval] --max_frames {opt.max_frames} reached; stopping early. "
                  f"The AP/FP below are on a truncated split.")
            break
        print(f"{infer_info}_{i}")
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if opt.fusion_method == 'late':
                infer_result = inference_utils.inference_late_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'early':
                infer_result = inference_utils.inference_early_fusion(batch_data,
                                                        model,
                                                        opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'no_w_uncertainty':
                infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                model,
                                                                opencood_dataset)
            elif opt.fusion_method == 'single':
                infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                model,
                                                                opencood_dataset,
                                                                single_gt=True)
            else:
                raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                        'fusion is supported.')

            pred_box_tensor = infer_result['pred_box_tensor']
            gt_box_tensor = infer_result['gt_box_tensor']
            pred_score = infer_result['pred_score']
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
            if opt.save_npy:
                npy_save_path = os.path.join(opt.model_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path)

            if not opt.no_score:
                infer_result.update({'score_tensor': pred_score})

            if getattr(opencood_dataset, "heterogeneous", False):
                cav_box_np, agent_modality_list = inference_utils.get_cav_box(batch_data)
                infer_result.update({"cav_box_np": cav_box_np, \
                                     "agent_modality_list": agent_modality_list})

            if (i % opt.save_vis_interval == 0) and (pred_box_tensor is not None or gt_box_tensor is not None):
                vis_save_path_root = os.path.join(opt.model_dir, f'vis_{infer_info}')
                if not os.path.exists(vis_save_path_root):
                    os.makedirs(vis_save_path_root)

                # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                # simple_vis.visualize(infer_result,
                #                     batch_data['ego'][
                #                         'origin_lidar'][0],
                #                     hypes['postprocess']['gt_range'],
                #                     vis_save_path,
                #                     method='3d',
                #                     left_hand=left_hand)
                 
                vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                simple_vis.visualize(infer_result,
                                    batch_data['ego'][
                                        'origin_lidar'][0],
                                    hypes['postprocess']['gt_range'],
                                    vis_save_path,
                                    method='bev',
                                    left_hand=left_hand)
        torch.cuda.empty_cache()

    _, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir, infer_info)

if __name__ == '__main__':
    main()
