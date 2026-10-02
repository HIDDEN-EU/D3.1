"""
-*- coding: utf-8 -*-
Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
License: TDG-Attribution-NonCommercial-NoDistrib

Incrementally increase heterogeneous agents in order.

Actual collaborator m1 -> m1+m2 -> m1+m2+m3 -> m1+m2+m3+m4

Ego is always m1

commrange is 180 (large enough)

For Intermediate Fusion, we will switch to IntermediateHeterinferFusionDataset
"""

import argparse
import importlib
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml import yaml_utils
from opencood.tools import inference_utils, train_utils
from opencood.tools.levy import (
    levy_agent_dev,
    levy_cells,
    levy_collect,
    levy_proposals,
    levy_rescore,
)
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict
from opencood.utils.perf_utils import PerfProfiler
from opencood.visualization import simple_vis

torch.multiprocessing.set_sharing_strategy("file_system")


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Continued training path"
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate",
        help="no, no_w_uncertainty, late, early or intermediate",
    )
    parser.add_argument(
        "--save_vis_interval",
        type=int,
        default=40,
        help="interval of saving visualization",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="whether to save prediction and gt resultin npy file",
    )
    parser.add_argument(
        "--range",
        type=str,
        default="204.8,102.4",
        help="detection range is [-204.8, +204.8, -102.4, +102.4]",
    )
    parser.add_argument(
        "--no_score", action="store_true", help="whether print the score of prediction"
    )
    parser.add_argument(
        "--use_cav",
        type=str,
        default="[1,2,3,4]",
        help="evaluate with real collaborator number",
    )
    parser.add_argument(
        "--lidar_degrade",
        action="store_true",
        help="whether to degrade lidar. {m1:32, m3:16} and {m1:16, m3:16}",
    )
    parser.add_argument("--note", default="", type=str, help="any other thing?")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="evaluate on test_dir or validate_dir. 'val' builds the in-order "
        "modality assignment for the validation scenarios on the fly",
    )
    parser.add_argument(
        "--save_levy",
        action="store_true",
        help="write per-frame and per-box Levy trust-map statistics to "
        "model_dir/levy_<info>_{frames,boxes}.csv (intermediate fusion with "
        "fusion_backbone.core_method: levy_pyramid)",
    )
    parser.add_argument(
        "--pose_noise",
        type=str,
        default=None,
        help="list of Gaussian pose-noise std, e.g. [0.2,0.4,0.6]; each value is "
        "used as both pos_std (m) and rot_std (deg) with zero mean, as in "
        "inference_w_noise.py. Every agent's pose is perturbed (seed 303 per level)",
    )
    parser.add_argument(
        "--measure_perf",
        action="store_true",
        help="measure inference FPS and peak GPU memory, save results to model_dir",
    )
    parser.add_argument(
        "--dataset_subset",
        type=bool,
        default=False,
        help="whether to use subset for dataset eval",
    )
    parser.add_argument(
        "--save_cells",
        action="store_true",
        help="also write sampled per-BEV-cell rows (vehicle vs background) to "
        "model_dir/levy_<info>_cells.npz; implies --save_levy",
    )
    parser.add_argument(
        "--save_agent_dev",
        action="store_true",
        help="dump per-agent deviations at cells with 3+ contributing agents to "
        "model_dir/levy_<info>_agentdev.npz; needs levy.agent_deviation: true",
    )
    parser.add_argument(
        "--save_proposals",
        action="store_true",
        help="turn the Levy vehicle mask into boxes where no detection was made and "
        "write them to model_dir/levy_<info>_proposals.npz; implies --save_levy",
    )
    parser.add_argument(
        "--vis_levy_mask",
        action="store_true",
        help="draw the Levy vehicle mask's undetected cells (orange = inside a GT box, "
        "magenta = not) and its proposals on the vis frames; needs levy.vehicle_mask",
    )
    parser.add_argument(
        "--levy_rescore",
        type=str,
        nargs="+",
        default=None,
        help="one or more box_rescore_params*.json (levy_rescore.py): replace each box's "
        "score with the logistic probability of a true positive (model B: Levy signals, "
        "C: Levy signals + detection score, both for >= 2 observing agents; S: evidence + "
        "detection score for 1 agent); boxes no model covers keep the detector score",
    )
    parser.add_argument(
        "--levy_proposals",
        type=str,
        default=None,
        help="proposal_selector*.json (levy_proposal_refine.py --write_selector): add the "
        "detector's own box at each uncovered Levy-mask group whose learned quality passes "
        "the selector's threshold, scored below every detection; needs levy.vehicle_mask",
    )
    parser.add_argument(
        "--levy_gateway",
        action="store_true",
        help="enable the pre-transmission Levy feature gateway (camera agents only, "
        "level 0, tau=0.5): each camera agent multiplies its own features by "
        "(1 - exp(-tau*exp(gamma*S)))^k before fusion; overrides "
        "model.args.fusion_backbone.levy.gateway",
    )
    # Defaults chosen on the validation split (use_cav 4 sweep over k in
    # {0.25, 0.5, 1, 2} x gamma in {2, 4, 8}); k=2, gamma=8 was best on every IoU.
    parser.add_argument(
        "--levy_gateway_gamma",
        type=float,
        default=8.0,
        help="temperature gamma for --levy_gateway",
    )
    parser.add_argument(
        "--levy_gateway_k",
        type=float,
        default=2.0,
        help="gate power k for --levy_gateway (0 disables the gate)",
    )
    parser.add_argument(
        "--levy_gateway_fn",
        type=str,
        default="levy",
        choices=["levy", "sigmoid"],
        help="gate curve for --levy_gateway: 'levy' uses --levy_gateway_k/_gamma; "
        "'sigmoid' multiplies by g_min + (1 - g_min) * sigmoid((x - t) / T) on the raw "
        "occupancy logit x, set by --levy_gateway_floor/_center/_temp",
    )
    parser.add_argument(
        "--levy_gateway_floor", type=float, default=0.5,
        help="g_min for --levy_gateway_fn sigmoid",
    )
    parser.add_argument(
        "--levy_gateway_center", type=float, default=-3.0,
        help="t (raw logit) for --levy_gateway_fn sigmoid",
    )
    parser.add_argument(
        "--levy_gateway_temp", type=float, default=1.0,
        help="T (logits) for --levy_gateway_fn sigmoid",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers; fewer workers use less RAM when runs share the machine",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="only run the first N frames (quick sanity run)",
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()

    assert opt.fusion_method in [
        "late",
        "early",
        "intermediate",
        "no",
        "no_w_uncertainty",
        "single",
    ]

    hypes = yaml_utils.load_yaml(None, opt)

    if opt.levy_gateway:
        levy_cfg = hypes["model"]["args"]["fusion_backbone"].setdefault("levy", {})
        levy_cfg["gateway"] = {
            "enable": True,
            "gate_fn": opt.levy_gateway_fn,
            "gate_power": opt.levy_gateway_k,
            "temp_gamma": opt.levy_gateway_gamma,
            "rate_tau": 0.5,
            "gate_floor": opt.levy_gateway_floor,
            "gate_center": opt.levy_gateway_center,
            "gate_temp": opt.levy_gateway_temp,
            "levels": [0],
            "cameras_only": True,
        }

    if "heter" in hypes:
        # hypes['heter']['lidar_channels'] = 16
        # opt.note += "_16ch"

        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"

        new_cav_range = [
            x_min,
            y_min,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max,
            y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]

        # replace all appearance
        hypes = update_dict(
            hypes,
            {
                "cav_lidar_range": new_cav_range,
                "lidar_range": new_cav_range,
                "gt_range": new_cav_range,
            },
        )

        hypes = update_dict(
            hypes, {"mapping_dict": {"m1": "m1", "m2": "m4", "m3": "m3", "m4": "m4"}}
        )

        # reload anchor
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                parser_func = func
        hypes = parser_func(hypes)

    if opt.split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
        if "OPV2V" in hypes["test_dir"] or "v2xsim" in hypes["test_dir"]:
            assert "test" in hypes["validate_dir"]
    else:
        opt.note += "_val"

    # This is used in visualization
    # left hand: OPV2V, V2XSet
    # right hand: V2X-Sim 2.0 and DAIR-V2X
    left_hand = (
        True
        if ("OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"])
        else False
    )

    print(f"Left hand visualizing: {left_hand}")

    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    saved_path = opt.model_dir
    resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    print(f"resume from {resume_epoch} epoch.")
    opt.note += f"_epoch{resume_epoch}"

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    levy_fusion = None
    if opt.save_cells or opt.save_proposals or opt.save_agent_dev:
        opt.save_levy = True
    if opt.save_levy or opt.vis_levy_mask or opt.levy_rescore or opt.levy_proposals:
        levy_fusion = levy_collect.get_levy_fusion(model)
        if opt.fusion_method != "intermediate" or levy_fusion is None:
            raise ValueError(
                "--save_levy needs --fusion_method intermediate and a model with "
                "fusion_backbone.core_method: levy_pyramid"
            )
        if not levy_fusion.compute_uncertainty:
            raise ValueError("--save_levy needs levy.compute_uncertainty: true")

    if opt.measure_perf:
        profiler = PerfProfiler(use_cuda=torch.cuda.is_available())
        profiler.hook_model(model)
        if hasattr(model, "pyramid_backbone"):
            profiler.patch_callable(model.pyramid_backbone, "forward_collab", "fusion")
        elif hasattr(model, "fusion_net"):
            profiler.patch_callable(model.fusion_net, "forward", "fusion")

    # setting noise
    np.random.seed(303)

    if opt.fusion_method == "intermediate":
        hypes["fusion"]["core_method"] += "infer"
    hypes["comm_range"] = 180
    if opt.split == "test":
        hypes["heter"]["assignment_path"] = hypes["heter"]["assignment_path"].replace(
            ".json", "_in_order.json"
        )
    else:
        # The shipped *_in_order.json only lists the test scenarios; validation
        # scenarios missing from it would silently fall back to all-m1.
        val_assignment = os.path.join(
            opt.model_dir,
            os.path.basename(hypes["heter"]["assignment_path"]).replace(
                ".json", "_validate_in_order.json"
            ),
        )
        write_in_order_assignment(hypes["validate_dir"], val_assignment)
        hypes["heter"]["assignment_path"] = val_assignment
    hypes = update_dict(hypes, {"ego_modality": "m1"})

    if opt.lidar_degrade:
        lidar_dict1 = {"m1": 32, "m3": 16}
        lidar_dict2 = {"m1": 16, "m3": 16}
        opt.use_cav = "[4]"
        use_cav_and_lidar_config_pair = [(4, lidar_dict1), (4, lidar_dict2)]
    else:
        lidar_dict0 = {"m3": 32}
        use_cav_and_lidar_config_pair = [(x, lidar_dict0) for x in eval(opt.use_cav)]

    noise_levels = [None] if opt.pose_noise is None else eval(opt.pose_noise)
    settings = [
        (use_cav, lidar_config, noise)
        for use_cav, lidar_config in use_cav_and_lidar_config_pair
        for noise in noise_levels
    ]

    for use_cav, lidar_config, noise in settings:
        hypes["use_cav"] = use_cav
        if noise is None:
            hypes.pop("noise_setting", None)
        else:
            np.random.seed(303)
            hypes["noise_setting"] = OrderedDict(
                add_noise=True,
                args={
                    "pos_std": noise,
                    "rot_std": noise,
                    "pos_mean": 0,
                    "rot_mean": 0,
                },
            )
            print(f"Pose noise: pos_std={noise} m, rot_std={noise} deg")
        if lidar_config is not None:
            hypes["heter"]["lidar_channels_dict"] = lidar_config
            print(hypes["heter"]["lidar_channels_dict"])

        # build dataset for each noise setting
        print("Dataset Building")
        opencood_dataset = build_dataset(hypes, visualize=True, train=False)

        if opt.measure_perf:
            profiler.patch_callable(
                opencood_dataset, "post_process", "post_process", cpu_only=True
            )

        opencood_dataset.force_all_same_agent_modality = True
        # Only the DataLoader iterates the Subset; post-processing needs the dataset.
        loader_dataset = opencood_dataset
        if opt.dataset_subset:
            print("Using Subset of DATASET!")
            loader_dataset = Subset(opencood_dataset, range(1200, 1305))
        if opt.max_frames is not None:
            print(f"Using only the first {opt.max_frames} frames!")
            loader_dataset = Subset(
                loader_dataset, range(min(opt.max_frames, len(loader_dataset)))
            )

        data_loader = DataLoader(
            loader_dataset,
            batch_size=1,
            num_workers=opt.num_workers,
            collate_fn=opencood_dataset.collate_batch_test,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

        # Create the dictionary for evaluation
        result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        infer_info = opt.fusion_method + opt.note + f"_use_cav{use_cav}"
        if noise is not None:
            infer_info += f"_noise{noise}_{noise}_0_0"
        if opt.lidar_degrade:
            infer_info += f"_m1_{lidar_config['m1']}_m3_{lidar_config['m3']}"

        rescorer = (
            levy_rescore.BoxRescorer(
                opt.levy_rescore, hypes["postprocess"]["anchor_args"]["cav_lidar_range"]
            )
            if opt.levy_rescore
            else None
        )
        selector = (
            levy_proposals.ProposalSelector(
                opt.levy_proposals,
                model,
                hypes["postprocess"]["anchor_args"]["cav_lidar_range"],
                hypes["postprocess"]["gt_range"],
            )
            if opt.levy_proposals
            else None
        )
        levy_recorder = (
            levy_collect.LevyRecorder(
                hypes["postprocess"]["anchor_args"]["cav_lidar_range"],
                hypes["postprocess"]["gt_range"],
            )
            if levy_fusion is not None and opt.save_levy
            else None
        )
        agent_dev_recorder = (
            levy_agent_dev.AgentDeviationRecorder(
                hypes["postprocess"]["anchor_args"]["cav_lidar_range"],
                hypes["postprocess"]["gt_range"],
            )
            if levy_fusion is not None and opt.save_agent_dev
            else None
        )
        score_hook = (
            levy_cells.ScoreMapHook(model)
            if levy_fusion is not None and opt.save_cells
            else None
        )
        proposal_recorder = (
            levy_proposals.ProposalRecorder(
                hypes["postprocess"]["anchor_args"]["cav_lidar_range"],
                hypes["postprocess"]["gt_range"],
                hypes["postprocess"]["anchor_args"],
                model=model,
            )
            if levy_fusion is not None and opt.save_proposals
            else None
        )
        cell_recorder = (
            levy_cells.CellRecorder(
                hypes["postprocess"]["anchor_args"]["cav_lidar_range"],
                hypes["postprocess"]["gt_range"],
            )
            if levy_fusion is not None and opt.save_cells
            else None
        )

        if opt.measure_perf:
            WARMUP_ITERS = 5
            perf_timings = []
            torch.cuda.reset_peak_memory_stats()
            use_cuda_events = torch.cuda.is_available()
            if use_cuda_events:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)

        for i, batch_data in enumerate(data_loader):
            # we can just save the batch_data to file. For perfomance test.
            # intermediate dataset
            # command: python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs_HEAL/Pyramid_collaboration_m1_base/final_infer --range 102.4,102.4
            # import pickle
            # print(batch_data['ego']['agent_modality_list'])
            # collabortor = "".join(batch_data['ego']['agent_modality_list'])
            # save_dir = f"opencood/logs_HEAL/FLOPs_calc/{collabortor}_online"
            # with open(os.path.join(save_dir, 'input.pkl'), 'wb') as f:
            #     pickle.dump(batch_data, f)
            # break

            # late fusion dataset
            # command: python opencood/tools/inference_heter_in_order.py --model_dir opencood/logs_HEAL/HEAL_late_final --fusion_method late --use_cav [4] --range 102.4,102.4
            # import pickle
            # for i_, (cav_id, cav_content) in enumerate(batch_data.items()):
            #     modality_name = cav_content['modality_name']
            #     print(cav_id, modality_name)
            #     save_dir = f"opencood/logs_HEAL/FLOPs_calc/{modality_name}_single"
            #     with open(os.path.join(save_dir, 'input.pkl'), 'wb') as f:
            #         pickle.dump({'ego':cav_content}, f)
            #     if i_ >= 4:
            #         break
            # raise

            if i % 10 == 0:
                print(f"{infer_info}_{i}")
            if batch_data is None:
                continue

            if opt.measure_perf and i == WARMUP_ITERS:
                profiler.reset()

            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)

                if opt.measure_perf:
                    if use_cuda_events:
                        torch.cuda.synchronize()
                        starter.record()
                    else:
                        _t0 = time.time()

                if opt.fusion_method == "late":
                    infer_result = inference_late_fusion_heter_in_order(
                        batch_data, model, opencood_dataset, use_cav
                    )
                elif opt.fusion_method == "intermediate":
                    if levy_fusion is not None:
                        # never record a trust map left over from a previous frame
                        levy_fusion.uncertainty_output = None
                    infer_result = inference_utils.inference_intermediate_fusion(
                        batch_data, model, opencood_dataset
                    )
                elif opt.fusion_method == "no":
                    infer_result = inference_utils.inference_no_fusion(
                        batch_data, model, opencood_dataset
                    )
                elif opt.fusion_method == "single":
                    infer_result = inference_utils.inference_no_fusion(
                        batch_data, model, opencood_dataset, single_gt=True
                    )
                else:
                    raise NotImplementedError(
                        "Only single, no, no_w_uncertainty, early, late and intermediate"
                        "fusion is supported."
                    )

                if opt.measure_perf:
                    if use_cuda_events:
                        ender.record()
                        torch.cuda.synchronize()
                        elapsed = starter.elapsed_time(ender) / 1000.0  # ms -> s
                    else:
                        elapsed = time.time() - _t0
                    if i >= WARMUP_ITERS:
                        perf_timings.append(elapsed)

                # of shape [N, 8, 3]
                # N: number of objects detected
                # [8, 3]: coordinates of the 8 bounding box corners, in 3-dimensions
                if rescorer is not None:
                    infer_result["pred_score"] = rescorer(
                        levy_fusion.uncertainty_output, infer_result
                    )
                n_detections = None
                if selector is not None:
                    corners, quality = selector(
                        levy_fusion.uncertainty_output,
                        infer_result,
                        batch_data["ego"]["anchor_box"],
                    )
                    infer_result["levy_proposals"] = (corners, quality)
                    det_box = infer_result["pred_box_tensor"]
                    n_detections = 0 if det_box is None else det_box.shape[0]
                    if len(corners):
                        new_box = torch.as_tensor(corners, device=device)
                        new_score = torch.as_tensor(
                            selector.score_scale * quality, dtype=torch.float32, device=device
                        )
                        if det_box is None:
                            infer_result["pred_box_tensor"] = new_box
                            infer_result["pred_score"] = new_score
                        else:
                            infer_result["pred_box_tensor"] = torch.cat(
                                [det_box, new_box.to(det_box.dtype)]
                            )
                            infer_result["pred_score"] = torch.cat(
                                [infer_result["pred_score"],
                                 new_score.to(infer_result["pred_score"].dtype)]
                            )
                pred_box_tensor = infer_result["pred_box_tensor"]
                gt_box_tensor = infer_result["gt_box_tensor"]
                pred_score = infer_result["pred_score"]

                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3
                )
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5
                )
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7
                )

                if levy_recorder is not None:
                    kw = dict(
                        frame=i,
                        dataset=loader_dataset,
                        use_cav=use_cav,
                        batch_data=batch_data,
                        infer_result=infer_result,
                    )
                    levy_collect.record_frame(levy_recorder, levy_fusion, **kw)
                    if cell_recorder is not None:
                        levy_cells.record_frame(
                            cell_recorder, levy_fusion, score_hook=score_hook, **kw)
                    if proposal_recorder is not None:
                        levy_proposals.record_frame(proposal_recorder, levy_fusion, **kw)
                    if agent_dev_recorder is not None:
                        levy_agent_dev.record_frame(agent_dev_recorder, levy_fusion, **kw)

                if not opt.no_score:
                    infer_result.update({"score_tensor": pred_score})

                if getattr(opencood_dataset, "heterogeneous", False):
                    cav_box_np, agent_modality_list = inference_utils.get_cav_box(
                        batch_data
                    )
                    infer_result.update(
                        {
                            "cav_box_np": cav_box_np,
                            "agent_modality_list": agent_modality_list,
                        }
                    )

                if (i % opt.save_vis_interval == 0) and (
                    pred_box_tensor is not None or gt_box_tensor is not None
                ):
                    vis_save_path_root = os.path.join(
                        opt.model_dir, f"vis_{infer_info}"
                    )
                    if not os.path.exists(vis_save_path_root):
                        os.makedirs(vis_save_path_root)

                    vis_save_path = os.path.join(vis_save_path_root, "bev_%05d.png" % i)
                    vis_result = infer_result
                    if n_detections is not None:
                        # draw the real detections as usual and the proposals in the overlay
                        vis_result = dict(infer_result)
                        if infer_result["pred_box_tensor"] is not None:
                            vis_result["pred_box_tensor"] = infer_result["pred_box_tensor"][:n_detections]
                            if infer_result.get("score_tensor") is not None:
                                vis_result["score_tensor"] = infer_result["score_tensor"][:n_detections]
                    if opt.vis_levy_mask:
                        vis_result["levy_mask_overlay"] = levy_proposals.vis_overlay(
                            levy_fusion,
                            vis_result,
                            hypes["postprocess"]["anchor_args"]["cav_lidar_range"],
                            hypes["postprocess"]["gt_range"],
                            hypes["postprocess"]["anchor_args"],
                            proposals=infer_result.get("levy_proposals"),
                        )
                    simple_vis.visualize(
                        vis_result,
                        batch_data["ego"]["origin_lidar"][0],
                        hypes["postprocess"]["gt_range"],
                        vis_save_path,
                        method="bev",
                        left_hand=left_hand,
                    )
            torch.cuda.empty_cache()

        if opt.measure_perf:
            peak_mem_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else 0.0
            )

            if len(perf_timings) > 0:
                total_time = sum(perf_timings)
                num_samples = len(perf_timings)
                fps = num_samples / total_time
                mean_ms = (total_time / num_samples) * 1000
                std_ms = float(np.std(perf_timings)) * 1000
            else:
                fps = mean_ms = std_ms = 0.0
                num_samples = 0

            perf_lines = [
                f"Perf measurement for: {infer_info}",
                f"Warmup samples skipped: {WARMUP_ITERS}",
                f"Samples measured: {num_samples}",
                f"FPS (samples/sec): {fps:.3f}",
                f"Mean latency (ms): {mean_ms:.3f}",
                f"Std latency (ms):  {std_ms:.3f}",
                f"Peak GPU memory allocated (MB): {peak_mem_mb:.1f}",
                f"Timing backend: {'cuda_event' if use_cuda_events else 'time.time'}",
            ]
            extra = {}
            if hasattr(model, "pyramid_backbone"):
                extra["fusion"] = model.pyramid_backbone
            elif hasattr(model, "fusion_net"):
                extra["fusion"] = model.fusion_net
            perf_lines.extend(profiler.report_weight_lines(model, extra_modules=extra))
            perf_lines.extend(profiler.report_lines(mean_ms))
            for line in perf_lines:
                print(line)

            perf_file = os.path.join(opt.model_dir, f"perf_{infer_info}.txt")
            with open(perf_file, "w") as f:
                f.write("\n".join(perf_lines) + "\n")
            print(f"Perf results saved to {perf_file}")

        _, ap50, ap70 = eval_utils.eval_final_results(
            result_stat, opt.model_dir, infer_info
        )

        if rescorer is not None:
            print(rescorer.summary())
        if selector is not None:
            print(selector.summary())

        if levy_recorder is not None:
            paths = levy_recorder.write(os.path.join(opt.model_dir, f"levy_{infer_info}"))
            if cell_recorder is not None:
                paths += cell_recorder.write(os.path.join(opt.model_dir, f"levy_{infer_info}"))
            if proposal_recorder is not None:
                paths += proposal_recorder.write(os.path.join(opt.model_dir, f"levy_{infer_info}"))
            if agent_dev_recorder is not None:
                paths += agent_dev_recorder.write(os.path.join(opt.model_dir, f"levy_{infer_info}"))
            for path in paths:
                print(f"Levy statistics saved to {path}")


def write_in_order_assignment(root_dir, out_path, modalities=("m1", "m2", "m3", "m4")):
    """
    Build the in-order modality assignment for every scenario under ``root_dir``.

    Same rule as the shipped ``*_in_order.json`` (checked against all OPV2V and
    V2XSet test scenarios): cav ids sorted as the base dataset sorts them, a
    negative (RSU) id moved last, then m1, m2, m3, m4 assigned cyclically.
    """
    assignment = {}
    for scenario in sorted(os.listdir(root_dir)):
        scenario_dir = os.path.join(root_dir, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        cav_ids = sorted(
            x
            for x in os.listdir(scenario_dir)
            if os.path.isdir(os.path.join(scenario_dir, x))
        )
        if not cav_ids:
            continue
        if int(cav_ids[0]) < 0:
            cav_ids = cav_ids[1:] + [cav_ids[0]]
        assignment[scenario] = {
            cav_id: modalities[k % len(modalities)] for k, cav_id in enumerate(cav_ids)
        }
    with open(out_path, "w") as f:
        json.dump(assignment, f, indent=4)
    print(f"In-order assignment for {len(assignment)} scenarios written to {out_path}")


def inference_late_fusion_heter_in_order(batch_data, model, dataset, use_cav):
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

    # ['ego', "650", "659", ...]  keys in batch_data is in order
    for i_, (cav_id, cav_content) in enumerate(batch_data.items()):
        if i_ >= use_cav:
            break
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data, output_dict
    )

    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
    }
    return return_dict


if __name__ == "__main__":
    main()
