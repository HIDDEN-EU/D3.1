#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# HEAL extra stage: train a LiDAR+Camera BEV fusion encoder on top of the
# stage1 LiDAR collaboration base (opencood/logs/HEAL_m1_based/stage1/m1_base).
#
# Pipeline
#   1. copy the fused-stage config into a fresh log dir as config.yaml
#   2. build net_epoch1.pth for FusedPyramidCollab by transferring the
#      pretrained LiDAR weights (VFE + BEV backbone + PyramidFusion base +
#      detection heads) from m1_base; camera branch + fusion conv stay fresh
#   3. train.py --model_dir <log dir>  (reads config.yaml, resumes from ep 1,
#      strict=False so the fresh camera modules keep their init)
#
# Run from the repo root, in the conda env that has opencood + spconv.
# ---------------------------------------------------------------------------
set -euo pipefail

M1_BASE_DIR="opencood/logs/HEAL_m1_based/stage1/m1_base"
CONFIG="opencood/hypes_yaml/opv2v/CameraLidarFusion/fused_pyramid_from_m1base.yaml"
OUT_DIR="opencood/logs/HEAL_m1_based/stage_fused_lc/m1_lidar_camera"

mkdir -p "${OUT_DIR}"
cp "${CONFIG}" "${OUT_DIR}/config.yaml"

python opencood/tools/build_fused_from_m1base.py \
    --m1_base_dir "${M1_BASE_DIR}" \
    --output_dir  "${OUT_DIR}"

python opencood/tools/train.py \
    -y "${CONFIG}" \
    --model_dir "${OUT_DIR}"
