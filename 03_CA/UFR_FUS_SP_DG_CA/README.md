# OPUS on CoHFF

This repository is a fork of the official [OPUS / OPUS-V2 repository](https://github.com/jbwang1997/OPUS)
by Jiabao Wang et al. ([OPUS paper](https://arxiv.org/pdf/2409.09350)). It is **not** the official release and is not
affiliated with the original authors. The model, the nuScenes configs and most of the code are theirs; refer to
the upstream repository for the method description, pretrained models and results.

The changes in this fork:

- Single-agent training and evaluation of OPUS-V2 on the CoHFF (OPV2V) occupancy dataset, with
  optional evaluation on merged multi-agent GT (see [CoHFF](#cohff-opv2v)).
- An optional `mask_filter` in the OPUS-V2 head, which excludes out-of-taxonomy voxels from the loss.
- Extra options in `train.py` / `val.py` (work dir, wandb, distributed timeout, `--override`, `--metrics-out`).

No multi-agent collaboration code is included (the camera-LiDAR fusion models are upstream's).

## Environment

The environment follows upstream (PyTorch 1.13.1 + CUDA 11.6):

```
conda create -n opus python=3.8
conda activate opus
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.28.2
pip install spconv-cu116  # change the cuda version
pip install torch-scatter -f https://data.pyg.org/whl/torch-{Version}+${CUDA}.html # change the torch and cuda version
pip install fvcore einops
```

Compile the CUDA extensions:

```
cd models/csrc
python setup.py build_ext --inplace
```

## nuScenes

Unchanged from upstream, apart from the shared script options below.

### Dataset

Download [nuScenes](https://www.nuscenes.org/nuscenes) to `data/nuscenes`, and optionally
[Occ3D-nuScenes](https://tsinghua-mars-lab.github.io/Occ3D/) to `data/nuscenes/gts` and
[OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) to `data/nuscenes/occupancy`.
Then generate the info files:

```
python scripts/preprocess_nuscenes.py
```

### Training

Download the mmdet3d
[pre-trained weights](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth)
into `pretrain/`, then run

```
python train.py --config configs/opusv2_nusc-occ3d/opusv2-t_r50_704x256_8f_nusc-occ3d_50e.py   # single GPU
bash dist_train.sh 8 configs/opusv2_nusc-occ3d/opusv2-t_r50_704x256_8f_nusc-occ3d_50e.py       # 8 GPUs
```

The batch size per GPU is scaled automatically. The fusion models need the DAL-tiny weights; see upstream.

### Evaluation

```
python val.py --config <config> --weights path/to/checkpoint
bash dist_val.sh 8 <config> path/to/checkpoint
```

## CoHFF (OPV2V)

OPUS-V2 can be trained on the CoHFF occupancy dataset (built on OPV2V) in a single-agent setting: every agent is treated as an independent ego vehicle, and no information is exchanged between agents.

Install `numba` (needed for GT voxelization):

```
pip install numba
```

### Dataset

Place the dataset in `data/OPV2V` (or change `dataset_root` in the config):

```
data/OPV2V
├── train / validate / test
│   └── <scene>/<agent>/<frame>_camera{0-3}.png, <frame>.yaml
└── surround
    └── train / validate / test
        └── <scene>/<agent>/<frame>_semantic.pcd   % semantic point cloud GT
```

The scene splits are listed in the config. The per-frame infos are cached in `data/cohff_opusv2_{train,val,test}_infos.pkl` on first use (set `rebuild_cache=True` in the config to regenerate them).

Dataset options (in `data.{train,val,test}` of the config):

- `ego_perspectives`: `'first'` uses only the first agent of each scene as ego, `'all'` uses every agent.
- `use_merged_gt`: use the merged multi-agent GT (`surround/<split>/<scene>/merged_gt/<frame>_ego<id>_merged_semantic.npz`) instead of the single-agent GT, for training and/or evaluation. Egos without a merged GT file are skipped.
- Multiple frames can be used by setting `num_frames` in the config (past frames). Future frames can be loaded with `next_sweeps_num` in `LoadCOHFFMVImageWithSweeps`.

### Training

Download the R101-DCN FCOS3D pre-trained weights and put them at `pretrain/r101_dcn_fcos3d_pretrain.pth`, then run

```
bash dist_train.sh 8 configs/opusv2_cohff/opusv2-t_r101_800x576_1f_cohff.py
```

Head options (in `model.pts_bbox_head` / `model.train_cfg.pts`):

- `mask_filter=True`: voxels outside the CoHFF taxonomy (`mask_camera == False`) are excluded from the classification and point losses, but still used for matching.
- `rare_weights_scale`: multiplier of the rare-class point weight (the CoHFF config uses 2).

`train.py` additionally supports `--work_dir`, `--wandb_id`, `--wandb_name` and `--wandb_tags` (the latter require a `WandbLoggerHook` in `log_config.hooks`), and a configurable `dist_timeout_minutes` in the config.

### Evaluation

mIoU is computed on the 12 CoHFF classes at 100x100x8 resolution. The GT grids are voxelized in parallel and cached.

```
bash dist_val.sh 8 configs/opusv2_cohff/opusv2-t_r101_800x576_1f_cohff.py path/to/checkpoints
```

`dist_val.sh` takes an optional temporary directory as 4th argument (for gathering multi-GPU results) and forwards all further arguments to `val.py`, e.g. `--metrics-out metrics.json`.

To evaluate on the merged GT with an existing config:

```
bash dist_val.sh 8 configs/opusv2_cohff/opusv2-t_r101_800x576_1f_cohff.py path/to/checkpoints "" --override data.val.use_merged_gt=True
```

## Citation

If this is useful for your research, please cite the original OPUS paper:

```
@inproceedings{wang2024opus,
  title={Opus: occupancy prediction using a sparse set},
  author={Wang, Jiabao and Liu, Zhaojiang and Meng, Qiang and Yan, Liujiang and Wang, Ke and Yang, Jie and Liu, Wei and Hou, Qibin and Cheng, Mingming},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## License and acknowledgement

Released under the MIT License of the original OPUS repository (see [LICENSE](LICENSE)).
The code builds on [OPUS](https://github.com/jbwang1997/OPUS), [SparseBEV](https://github.com/MCG-NJU/SparseBEV)
and [SparseOcc](https://github.com/MCG-NJU/SparseOcc).
