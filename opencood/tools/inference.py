# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>, Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import json
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
    parser.add_argument('--save_levy', action='store_true',
                        help='report Levy spatial trust map (evidence, '
                             'disagreement, sigma) statistics over every frame, '
                             'and dump the maps as npz plus BEV heatmaps every '
                             '--save_vis_interval frames. Silently does nothing '
                             'unless the fusion backbone is LevyPyramidFusion '
                             'with levy.compute_uncertainty enabled.')
    parser.add_argument('--levy_diag', action='store_true',
                        help='measure whether sigma_hat actually predicts '
                             'detection error: per-box sigma split by '
                             'true/false positive, AUROC, AUSE and what the '
                             'sparsification threshold buys. Run this before '
                             'tuning --levy_sigma_thresh. No-op unless the '
                             'backbone is LevyPyramidFusion.')
    parser.add_argument('--levy_diag_iou', type=float, default=0.5,
                        help='IoU threshold at which --levy_diag calls a box a '
                             'true positive (default 0.5)')
    parser.add_argument('--levy_sparsify', action='store_true',
                        help='apply the Levy Sparsification Filter: predicted '
                             'boxes whose footprint on the sigma trust map '
                             'exceeds --levy_sigma_thresh are dropped or '
                             'down-weighted before AP is computed. No-op unless '
                             'the fusion backbone is LevyPyramidFusion.')
    parser.add_argument('--levy_sigma_thresh', type=float, default=None,
                        help='high-uncertainty sigma_hat cut for '
                             '--levy_sparsify. Default: levy.sparsify.'
                             'sigma_threshold from the yaml, else 0.20 '
                             '(~p90 of overlap cells).')
    parser.add_argument('--levy_sparsify_mode', type=str, default=None,
                        choices=['attenuate', 'drop'],
                        help="action for a flagged box (default: yaml, "
                             "else 'attenuate').")
    parser.add_argument('--levy_sparsify_attenuation', type=float, default=None,
                        help='confidence multiplier for flagged boxes in '
                             'attenuate mode (default: yaml, else 0.5).')
    parser.add_argument('--levy_sparsify_reduce', type=str, default=None,
                        choices=['mean', 'max', 'center'],
                        help='how to reduce sigma over a box footprint '
                             '(default: yaml, else mean).')
    parser.add_argument('--range', type=str, default="102.4,102.4",
                        help="detection range is [-102.4, +102.4, -102.4, +102.4]")
    parser.add_argument('--no_score', action='store_true',
                        help="whether print the score of prediction")
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    opt = parser.parse_args()
    return opt


def report_levy_stats(levy_stats, levy_npy_path):
    """
    Average the per-frame Levy trust-map statistics and report them.

    Percentiles are taken within each frame and then averaged over frames, so
    they describe a typical frame rather than the pooled cell distribution.

    Two tables are printed. 'grid' covers every cell at least one agent
    contributes to -- since the ego is warped by the identity, that is its whole
    BEV extent, so those numbers are dominated by empty road. 'overlap' covers
    cells with two or more contributing agents, which is the only population
    where disagreement and sigma are not structurally zero. Read the second one.
    """
    if not levy_stats:
        print("[levy] --save_levy was set but the model emitted no "
              "'levy_uncertainty'. The fusion backbone is not LevyPyramidFusion "
              "(or levy.compute_uncertainty is off) -- nothing was saved.")
        return

    def average_block(name):
        frames = [s[name] for s in levy_stats if s.get(name) is not None]
        if not frames:
            return None, 0
        block = {}
        for key in inference_utils.LEVY_CHANNELS:
            rows = [f[key] for f in frames if key in f]
            if rows:
                block[key] = {stat: float(np.mean([r[stat] for r in rows]))
                              for stat in ('mean', 'p50', 'p90', 'p99', 'max')}
        return block, len(frames)

    coverage = {
        key: float(np.mean([s['coverage'][key] for s in levy_stats]))
        for key in ('covered_frac', 'overlap_frac', 'mean_agents')
    }
    # The busiest cell anywhere in the split, not an average of per-frame maxima.
    coverage['max_agents'] = float(np.max([s['coverage']['max_agents']
                                           for s in levy_stats]))
    summary = {'frames': len(levy_stats), 'coverage': coverage}

    print(f"\n[levy] spatial trust map over {len(levy_stats)} frames")
    print(f"[levy] coverage: {coverage['covered_frac']:.1%} of the BEV grid seen "
          f"by >=1 agent, {coverage['overlap_frac']:.1%} by >=2; "
          f"{coverage['mean_agents']:.2f} agents per covered cell on average, "
          f"{coverage['max_agents']:.0f} agents on the busiest cell in the split")

    for name, caption in (('grid', 'all covered cells (>=1 agent)'),
                          ('overlap', 'overlap only (>=2 agents)')):
        block, n_frames = average_block(name)
        summary[name] = block
        summary[f'{name}_frames'] = n_frames
        if block is None:
            print(f"\n  {name}: no frame had any such cell")
            continue
        print(f"\n  {name} -- {caption}, {n_frames}/{len(levy_stats)} frames")
        print(f"  {'channel':<14}{'mean':>12}{'p50':>12}{'p90':>12}"
              f"{'p99':>12}{'max':>12}")
        for key in inference_utils.LEVY_CHANNELS:
            if key not in block:
                continue
            row = block[key]
            print(f"  {key:<14}" + "".join(f"{row[stat]:>12.4g}" for stat in
                                           ('mean', 'p50', 'p90', 'p99', 'max')))

    stats_path = os.path.join(levy_npy_path, 'levy_stats.json')
    if not os.path.exists(levy_npy_path):
        os.makedirs(levy_npy_path)
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[levy] per-frame maps in {levy_npy_path}, summary in {stats_path}\n")


def report_levy_error_diagnostic(levy_diag, sigma_threshold, iou, out_dir):
    """
    Print and persist the sigma-vs-error diagnostic.

    The headline is AUROC. Sigma is only usable as a filter if high sigma marks
    boxes that are *wrong*; 0.5 means it carries no such information, and below
    0.5 means it marks the boxes that are right, so any threshold on it removes
    true positives and costs AP.
    """
    if not levy_diag['sigma']:
        print("[levy] --levy_diag was set but no Levy uncertainty was produced "
              "-- nothing to diagnose.")
        return

    sigma = np.concatenate(levy_diag['sigma'])
    is_tp = np.concatenate(levy_diag['is_tp'])
    m = inference_utils.levy_error_metrics(sigma, is_tp, sigma_threshold)
    if m is None:
        print("[levy] --levy_diag: no predicted boxes to diagnose.")
        return

    print(f"\n[levy] error diagnostic @ IoU {iou:g} -- does sigma predict error?")
    print(f"  {m['n_boxes']} boxes: {m['n_tp']} TP, {m['n_fp']} FP "
          f"(base FP rate {m['base_fp_rate']:.1%})")
    for name, key in (('sigma | TP', 'sigma_tp'), ('      | FP', 'sigma_fp')):
        d = m[key]
        if d is None:
            continue
        print(f"  {name}: mean {d['mean']:.4f}   p50 {d['p50']:.4f}   "
              f"p90 {d['p90']:.4f}   (n={d['n']})")

    auroc = m['auroc']
    if auroc == auroc:  # not NaN
        if auroc > 0.55:
            verdict = "high sigma marks false positives -- usable"
        elif auroc < 0.45:
            verdict = "INVERTED: high sigma marks CORRECT boxes"
        else:
            verdict = "no signal: sigma is uninformative about correctness"
        print(f"  AUROC(sigma -> FP) = {auroc:.4f}   [{verdict}]")
    print(f"  AUSE = {m['ause']:.4f}   (0 = perfect ranking, larger is worse)")

    pct = m['percentiles']
    print("  box sigma percentiles: "
          + "  ".join(f"{k} {v:.4f}" for k, v in pct.items()))

    t = m['at_threshold']
    print(f"  at sigma > {t['sigma_threshold']:g}: flags {t['flagged_frac']:.1%} "
          f"of boxes; {t['precision']:.1%} of those are FP vs a "
          f"{m['base_fp_rate']:.1%} base rate (lift x{t['lift']:.2f}); "
          f"catches {t['recall']:.1%} of all FP")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    path = os.path.join(out_dir, 'levy_error_diagnostic.json')
    with open(path, 'w') as f:
        json.dump({'iou': float(iou), **m}, f, indent=2)
    print(f"  written to {path}\n")


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

    print('Creating Model')
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
    
    # build dataset for each noise setting
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    # opencood_dataset_subset = Subset(opencood_dataset, range(640,2100))
    # data_loader = DataLoader(opencood_dataset_subset,
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=4,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    
    infer_info = opt.fusion_method + opt.note

    # Levy spatial trust map. Stays empty -- and every levy branch below stays
    # inert -- when the fusion backbone is not LevyPyramidFusion, since only
    # LevyPyramidFusion puts 'levy_uncertainty' in the model output.
    levy_stats = []
    levy_npy_path = os.path.join(opt.model_dir, f'levy_{infer_info}')

    # Levy Sparsification Filter. Defaults come from the yaml
    # (model.args.fusion_backbone.levy.sparsify), CLI flags override them.
    try:
        _levy_cfg = hypes['model']['args']['fusion_backbone'].get('levy', {}) or {}
    except (KeyError, TypeError):
        _levy_cfg = {}
    _sp_cfg = _levy_cfg.get('sparsify', {}) or {}
    sparsify_on = opt.levy_sparsify or bool(_sp_cfg.get('enable', False))
    sparsify_thresh = opt.levy_sigma_thresh if opt.levy_sigma_thresh is not None \
        else float(_sp_cfg.get('sigma_threshold',
                               inference_utils.LEVY_SPARSIFY_DEFAULT_SIGMA))
    sparsify_mode = opt.levy_sparsify_mode or _sp_cfg.get('mode', 'attenuate')
    sparsify_atten = opt.levy_sparsify_attenuation \
        if opt.levy_sparsify_attenuation is not None \
        else float(_sp_cfg.get('attenuation', 0.5))
    sparsify_reduce = opt.levy_sparsify_reduce or _sp_cfg.get('reduce', 'mean')
    sparsify_channel = _sp_cfg.get('channel', 'sigma')
    levy_sparsify_tally = {'frames': 0, 'boxes': 0, 'flagged': 0}
    # Per-box sigma paired with whether the box was actually correct. Collected
    # before the sparsification filter runs, so it characterises sigma itself
    # rather than sigma after its own threshold has reshuffled the scores.
    levy_diag = {'sigma': [], 'is_tp': []}
    if sparsify_on:
        print(f"[levy] sparsification filter armed: {sparsify_mode}, "
              f"sigma_hat > {sparsify_thresh:g} (channel={sparsify_channel}, "
              f"reduce={sparsify_reduce}"
              + (f", x{sparsify_atten:g})" if sparsify_mode == 'attenuate'
                 else ")"))

    for i, batch_data in enumerate(data_loader):
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

            # None for a plain PyramidFusion backbone, which makes --save_levy
            # a no-op rather than an error.
            levy_uncertainty = infer_result.get('levy_uncertainty', None)
            if opt.save_levy and levy_uncertainty is not None:
                # Summaries are a few floats, so they are worth taking on every
                # frame; the maps themselves are only dumped on the vis interval.
                frame_stats = inference_utils.summarize_levy_uncertainty(
                    levy_uncertainty)
                if frame_stats is not None:
                    levy_stats.append(frame_stats)

                if i % opt.save_vis_interval == 0:
                    if not os.path.exists(levy_npy_path):
                        os.makedirs(levy_npy_path)
                    inference_utils.save_levy_uncertainty(levy_uncertainty, i,
                                                          levy_npy_path)

            if opt.levy_diag and levy_uncertainty is not None \
                    and pred_box_tensor is not None:
                levy_diag['sigma'].append(
                    inference_utils.levy_sigma_under_boxes(
                        pred_box_tensor, levy_uncertainty,
                        hypes['postprocess']['gt_range'],
                        channel=sparsify_channel, reduce=sparsify_reduce))
                levy_diag['is_tp'].append(
                    inference_utils.levy_match_tp_fp(
                        pred_box_tensor, pred_score, gt_box_tensor,
                        opt.levy_diag_iou))

            # Sparsification Filter: prune / down-weight boxes that fall in
            # high-sigma zones of the Levy trust map, then let AP see the result.
            if sparsify_on and levy_uncertainty is not None:
                pred_box_tensor, pred_score, sp_info = \
                    inference_utils.sparsify_predictions_by_levy(
                        pred_box_tensor, pred_score, levy_uncertainty,
                        hypes['postprocess']['gt_range'],
                        sigma_threshold=sparsify_thresh, mode=sparsify_mode,
                        attenuation=sparsify_atten, channel=sparsify_channel,
                        reduce=sparsify_reduce)
                infer_result['pred_box_tensor'] = pred_box_tensor
                infer_result['pred_score'] = pred_score
                levy_sparsify_tally['frames'] += 1
                levy_sparsify_tally['boxes'] += sp_info['n_total']
                levy_sparsify_tally['flagged'] += sp_info['n_flagged']

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

            if opt.save_levy and levy_uncertainty is not None \
                    and (i % opt.save_vis_interval == 0):
                levy_vis_path_root = os.path.join(opt.model_dir,
                                                  f'levy_vis_{infer_info}')
                if not os.path.exists(levy_vis_path_root):
                    os.makedirs(levy_vis_path_root)

                inference_utils.visualize_levy_uncertainty(
                    levy_uncertainty,
                    hypes['postprocess']['gt_range'],
                    os.path.join(levy_vis_path_root, 'levy_%05d.png' % i),
                    infer_result=infer_result,
                    left_hand=left_hand)

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

    if opt.save_levy:
        report_levy_stats(levy_stats, levy_npy_path)

    if opt.levy_diag:
        report_levy_error_diagnostic(levy_diag, sparsify_thresh,
                                     opt.levy_diag_iou, levy_npy_path)

    if sparsify_on and levy_sparsify_tally['frames']:
        t = levy_sparsify_tally
        frac = t['flagged'] / max(t['boxes'], 1)
        verb = 'dropped' if sparsify_mode == 'drop' else \
            f"down-weighted (x{sparsify_atten:g})"
        print(f"\n[levy] sparsification filter: {t['flagged']}/{t['boxes']} "
              f"predicted boxes {verb} over {t['frames']} frames ({frac:.1%}), "
              f"sigma_hat > {sparsify_thresh:g}")

if __name__ == '__main__':
    main()
