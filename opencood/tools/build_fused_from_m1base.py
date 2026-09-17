"""
Build the initial checkpoint for a LiDAR+Camera BEV fusion encoder
(``fused_pyramid_collab`` / :class:`FusedPyramidCollab`) from a HEAL stage1
LiDAR-only collaboration base (``heter_pyramid_collab`` / ``m1``).

This is the LiDAR+Camera analogue of ``heal_tools.py merge``: it lets the new
per-agent fused encoder start from the LiDAR weights that were already trained
under HEAL, instead of from scratch.

Weight mapping (source -> target)
---------------------------------
    encoder_<m>.*        -> encoder_lidar_<m>.*     PointPillar pillar-VFE
    backbone_<m>.*        -> backbone_<m>.*          ResNet BEV backbone (identical shape)
    pyramid_backbone.*    -> pyramid_backbone.*      PyramidFusion collaboration base
    shrink_conv.*         -> shrink_conv.*
    {cls,reg,dir}_head.*  -> {cls,reg,dir}_head.*

Left at random initialisation (trained from scratch in stage "fused"):
    encoder_camera_<m>.*  LiftSplatShoot camera BEV encoder
    fusion_conv_<m>.*     1x1 conv fusing concat([lidar_bev, camera_bev]): 128 -> 64

``opencood/tools/train.py --model_dir <output_dir>`` loads the produced
``net_epoch1.pth`` with ``strict=False`` (see ``train_utils.load_saved_model``),
so the un-mapped camera/fusion modules keep their fresh init and training
resumes from epoch 1.

Usage
-----
    python opencood/tools/build_fused_from_m1base.py \
        --m1_base_dir opencood/logs/HEAL_m1_based/stage1/m1_base \
        --output_dir  opencood/logs/HEAL_m1_based/stage_fused_lc/m1_lidar_camera
"""

import argparse
import os
from collections import OrderedDict

import torch

from opencood.tools.heal_tools import get_model_path_from_dir


def build_parser():
    parser = argparse.ArgumentParser(
        description="Init a lidar+camera fused encoder from a HEAL stage1 lidar base."
    )
    parser.add_argument('--m1_base_dir', type=str, required=True,
                        help='HEAL stage1 lidar collaboration base log dir '
                             '(contains net_epoch_bestval_at*.pth / net_epoch*.pth).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Destination log dir for the fused stage. '
                             'net_epoch1.pth is written here.')
    parser.add_argument('--src_modality', type=str, default='m1',
                        help="Modality name of the lidar encoder in the stage1 "
                             "checkpoint (default: m1).")
    parser.add_argument('--dst_modality', type=str, default=None,
                        help="Modality name of the fused encoder in the target "
                             "model (default: same as --src_modality).")
    return parser.parse_args()


def main():
    opt = build_parser()
    dst_modality = opt.dst_modality or opt.src_modality

    src_path = get_model_path_from_dir(opt.m1_base_dir)
    print(f"Loading HEAL stage1 lidar base: {src_path}")
    src_state = torch.load(src_path, map_location='cpu')

    encoder_prefix = f"encoder_{opt.src_modality}."
    new_encoder_prefix = f"encoder_lidar_{dst_modality}."

    out_state = OrderedDict()
    renamed, passthrough = 0, 0
    for key, value in src_state.items():
        if key.startswith(encoder_prefix):
            new_key = new_encoder_prefix + key[len(encoder_prefix):]
            renamed += 1
        else:
            new_key = key
            passthrough += 1
        out_state[new_key] = value

    if renamed == 0:
        raise RuntimeError(
            f"No keys starting with '{encoder_prefix}' found in {src_path}. "
            f"Available top-level modules: "
            f"{sorted({k.split('.')[0] for k in src_state})}"
        )

    os.makedirs(opt.output_dir, exist_ok=True)
    out_path = os.path.join(opt.output_dir, 'net_epoch1.pth')
    torch.save(out_state, out_path)

    print(f"  renamed   {renamed:4d} tensors  {encoder_prefix}* -> {new_encoder_prefix}*")
    print(f"  copied    {passthrough:4d} tensors  (backbone / pyramid_backbone / "
          f"shrink_conv / heads)")
    print(f"Wrote {out_path}  ({len(out_state)} tensors total)")
    print("\nNext:")
    print(f"  cp <fused-config>.yaml {opt.output_dir}/config.yaml")
    print(f"  python opencood/tools/train.py -y <fused-config>.yaml "
          f"--model_dir {opt.output_dir}")


if __name__ == '__main__':
    main()
