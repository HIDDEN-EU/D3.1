# BEVFusion Detection + Tracking + Trajectory Prediction

This repo runs [BEVFusion](https://github.com/mit-han-lab/bevfusion) (camera+LiDAR
3D object detection) on nuScenes, tracks the detections across frames, and predicts
each tracked object's future trajectory. For every scene you render you get:

- **A video** (`scene_XXX.mp4`) — all 6 camera views + a bird's-eye-view (BEV) panel,
  with detected/coasted 3D boxes and predicted future trajectories drawn in.
- **Per-frame JSON files** (`scene_XXX/<sample_token>.json`) in the ODT schema
  (see [`json_template/ODT.json`](json_template/ODT.json)) — one file per frame,
  with every tracked object's pose, velocity, and predicted future trajectory in
  the ego vehicle's frame.

Trajectory prediction is selectable per run:

| Choice | Flag(s) | What it is |
|---|---|---|
| **KF** | `--tracker kf` (default) | Constant-velocity Kalman filter. Simple, fast, good on straight roads. |
| **IMM** | `--tracker imm` | 2-model Interacting Multiple Model (constant-velocity + coordinated-turn). More accurate through roundabouts/turns than plain KF. |
| **QCNet + IMM** | `--tracker imm --future-predictor qcnet --qcnet-mode-select nearest-to-imm` | [QCNet](https://github.com/ZikangZhou/QCNet) (CVPR'23), a learned, map-aware, multi-modal trajectory predictor, with its output mode re-ranked by consistency with the IMM tracker's own rollout. In our internal evaluation (independently matched against nuScenes ground truth) this combination beat both plain KF and IMM. |

Detection and KF/IMM tracking run in one conda environment (`bevfusion`); QCNet
needs a second, incompatible environment (`QCNet`) — see the install steps below
for why, and how the two talk to each other.

---

## Repository layout

```
configs/                BEVFusion model configs (detector architecture, training/eval)
mmdet3d/                 Modified mmdet3d/mmdetection3d package BEVFusion is built on
tools/                   Upstream BEVFusion tools (create_data.py, test.py, train.py,
                         download_pretrained.sh, ...)
docker/                  Reference Dockerfile from upstream BEVFusion
track_and_visualize.py  Main entry point: detection -> tracking -> video + JSON
imm_tracker.py           IMM (CV + coordinated-turn) tracker
test_imm_tracker.py      Unit tests for imm_tracker.py
nuscenes_map_adapter.py  nuScenes HD map -> QCNet's map schema
qcnet_data_adapter.py    Track history -> QCNet's input tensors (2Hz -> 10Hz interp.)
qcnet_predictor.py       Loads the QCNet checkpoint and runs inference
qcnet_infer.py           CLI entry point for the QCNet inference stage (runs in the
                         separate `QCNet` conda env)
json_template/ODT.json  Reference schema/template for the per-frame JSON output
setup.py, setup.cfg      BEVFusion package build files
```

Not included here (see install steps below for how to get them):
nuScenes dataset itself, pretrained model weights, and the QCNet repo/checkpoint —
all too large to check into git.

---

## Requirements

- Linux, an NVIDIA GPU (tested on a single RTX A5000; any CUDA 11-class GPU should work)
- Miniconda/Anaconda
- ~15 GB disk for pretrained weights + nuScenes `v1.0-trainval` metadata/maps
  (full nuScenes sensor data is ~300+ GB if you want more than a handful of scenes)

---

## 1. Install BEVFusion

```bash
git clone <this-repo-url> bevfusion
cd bevfusion

conda create -n bevfusion python=3.8 -y
conda activate bevfusion

# CUDA 11.7 toolchain via conda-forge — needed to compile BEVFusion's custom
# CUDA ops even if your system CUDA is a different version (system gcc is
# often too new for CUDA 11.7's nvcc otherwise).
conda install -c conda-forge cudatoolkit-dev=11.7 gxx_linux-64=11.4 -y

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install mmcv-full==1.7.0 mmdet==2.28.2
pip install nuscenes-devkit torchpack numba timm opencv-python matplotlib tqdm

pip install -v -e .   # compiles mmdet3d's custom CUDA ops
```

**Known install gotcha (conda + CUDA 11.7 on newer Ubuntu):** if the build fails
linking against `libm.so`/`libc.so` looking for CentOS-style paths
(`/x86_64-conda-linux-gnu/sysroot/lib64/...`), your conda sysroot linker scripts
need to point at your actual system libs instead
(`/lib/x86_64-linux-gnu/libm.so` / `libc.so` on Debian/Ubuntu). Edit
`<conda-env>/x86_64-conda-linux-gnu/sysroot/usr/lib64/lib{m,c}.so` (they're just
linker scripts, not real binaries) to reference the correct path, then re-run
`pip install -v -e .`.

Everything else needed to build cleanly (a missing extension entry, an int32/int64
mismatch against this PyTorch version, an mmcv/mmdet3d sparse-conv registration
conflict, and a dead `flash_attn` import) is already fixed in this repo's
`setup.py`/`mmdet3d/` — you shouldn't need to touch that code.

---

## 2. Prepare the nuScenes dataset

1. Download nuScenes `v1.0-trainval` (Full dataset + Map expansion pack) from
   [nuscenes.org](https://www.nuscenes.org/download) and extract it under
   `data/nuscenes/` so you have `data/nuscenes/v1.0-trainval/`, `data/nuscenes/samples/`,
   `data/nuscenes/sweeps/`, `data/nuscenes/maps/`.

2. Generate the info pkls:

   ```bash
   conda run -n bevfusion python tools/create_data.py nuscenes \
       --root-path data/nuscenes --out-dir data/nuscenes --extra-tag nuscenes
   ```

   **Known gotcha:** the converter (`tools/data_converter/nuscenes_converter.py`)
   saves train/val info files to `nuscenes_infos_{train,val}_radar.pkl` in the
   current directory rather than `data/nuscenes/nuscenes_infos_{train,val}.pkl`
   (an upstream naming/path quirk, not specific to this fork). Symlink the
   expected names to what actually got generated rather than trying to
   regenerate them differently:

   ```bash
   ln -s $(pwd)/nuscenes_infos_train_radar.pkl data/nuscenes/nuscenes_infos_train.pkl
   ln -s $(pwd)/nuscenes_infos_val_radar.pkl   data/nuscenes/nuscenes_infos_val.pkl
   ```

---

## 3. Download pretrained detection weights

```bash
conda run -n bevfusion bash tools/download_pretrained.sh
```

This fetches `pretrained/bevfusion-det.pth` (camera+LiDAR detector, used by
`track_and_visualize.py` by default) plus the camera-only/LiDAR-only variants.

---

## 4. (Optional) Install QCNet — only needed for the QCNet+IMM predictor

QCNet needs `torch 2.0.1+cu118`/PyTorch Geometric, which conflicts with the
`bevfusion` env's `torch 1.13.1+cu117` (whose custom CUDA ops are compiled
against it) — they can't share one environment. If you only want KF or IMM,
skip this section entirely.

```bash
git clone https://github.com/ZikangZhou/QCNet.git
# commit used during development: 55cacb418cbbce3753119c1f157360e66993d0d0
cd QCNet
conda env create -f environment.yml -n QCNet
conda run -n QCNet pip install nuscenes-devkit
cd ..
```

Download the pretrained checkpoint (trained on Argoverse 2 — see
[the note on domain gap](#choosing-a-predictor) below) into `QCNet/checkpoints/`:

```bash
mkdir -p QCNet/checkpoints
conda run -n bevfusion gdown 1OKBytt6N6BdRa9FWmS7F1-YvF0YectBv \
    -O QCNet/checkpoints/QCNet_AV2.ckpt
```

---

## Running: detection + tracking + video + JSON

All commands run from the repo root. `MASTER_HOST` is required by
`torchpack`'s distributed init even for a single GPU.

`--scenes` accepts individual scene numbers and/or dash-ranges, e.g. `0 1 3-6 9`.
Scene numbers are assigned in the order scenes appear in the val split (or in
your `--num-samples`-limited subset, timestamp-sorted).

### KF (default) or IMM

```bash
# Run inference fresh and render scenes 0-2 (KF tracker, the default):
MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \
    --scenes 0-2 --save-results bf_results.pkl

# Same, but with the IMM tracker:
MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \
    --scenes 0-2 --results bf_results.pkl --tracker imm \
    --video-dir tracked_scenes_imm --json-dir tracked_scenes_imm_json
```

Output: `tracked_scenes/scene_000.mp4` (or wherever `--video-dir` points) and
`tracked_scenes_json/scene_000/<sample_token>.json` per frame (or wherever
`--json-dir` points). Pass `--no-json` to skip JSON and only render video.

### QCNet + IMM

QCNet runs as a separate 3-stage pipeline (dump → QCNet-env inference → render),
because of the environment split described above:

```bash
# Stage 1 (bevfusion env): dump track/ego history for scenes 0-2
MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \
    --scenes 0-2 --results bf_results.pkl --tracker imm --no-json \
    --video-dir /tmp/discard --dump-qcnet-input qcnet_input.pkl

# Stage 2 (QCNet env): run QCNet on the dumped histories
conda run -n QCNet python qcnet_infer.py \
    --dump qcnet_input.pkl --out qcnet_futures.pkl \
    --dataroot data/nuscenes --future-steps 10

# Stage 3 (bevfusion env): render using QCNet's predictions, re-ranked by
# consistency with the IMM rollout
MASTER_HOST=localhost:29500 conda run -n bevfusion python track_and_visualize.py \
    --scenes 0-2 --results bf_results.pkl --tracker imm \
    --future-predictor qcnet --qcnet-mode-select nearest-to-imm \
    --qcnet-futures qcnet_futures.pkl \
    --video-dir tracked_scenes_qcnet --json-dir tracked_scenes_qcnet_json
```

`--qcnet-mode-select top1` (QCNet's own highest-confidence mode, no IMM
involved) is also available but was not the best performer in our evaluation —
see below.

<a name="choosing-a-predictor"></a>
### Choosing a predictor

Internally we evaluated all three against independently matched nuScenes
ground truth (not just each tracker's own continuation) across 20 scenes.
Average displacement error (ADE) ordering, best to worst:
**QCNet + nearest-to-IMM re-ranking < IMM ≈ KF < QCNet top-1**.

QCNet is trained on Argoverse 2, not nuScenes (no native nuScenes datamodule
exists upstream), so there is a real sensor/map/cadence domain gap; top-1
(QCNet's own confidence ranking) suffers from that. But QCNet's 6-mode output
already contains a trajectory that beats KF/IMM at every horizon — picking the
mode closest to the IMM tracker's own rollout (`--qcnet-mode-select
nearest-to-imm`, no ground truth involved, valid at real inference time)
recovers most of that and is the recommended setting if you enable QCNet at
all. Plain KF or IMM remain solid, much simpler choices if you'd rather not
run a second environment/model.

---

## Output formats

**Video** (`scene_XXX.mp4`, 1280×720): 6 camera views in a grid, plus a BEV
panel and a status/legend strip. Solid boxes = matched detection this frame;
dashed orange boxes = coasted prediction while occluded; dotted lines =
predicted future trajectory.

**JSON** (`scene_XXX/<sample_token>.json`): one file per frame, schema in
[`json_template/ODT.json`](json_template/ODT.json). Key fields per detection:

- `track_id`, `category`, `x`/`y`/`z`/`length`/`width`/`height`/`heading` (ego
  vehicle frame, meters/radians), `vx`/`vy`/`yaw_rate`
- `optional.track_state`: `"detected"` (matched this frame) or `"predicted"`
  (coasting through occlusion)
- `predicted_trajectory`: list of `{t, x, y, z}` future waypoints (0.5s apart,
  matching nuScenes' 2Hz keyframe rate)
- `prediction_source`: which tracker/predictor produced the trajectory, e.g.
  `"kf"`, `"imm"`, `"qcnet_nearest-to-imm+imm"` (or `"..._fallback"` if a track
  was missing from a QCNet futures dump and fell back to its own physics
  rollout)

---

## License

BEVFusion/mmdet3d code is under the Apache 2.0 license (see [`LICENSE`](LICENSE)).
QCNet (not included here — see install step 4) has its own license; check
[github.com/ZikangZhou/QCNet](https://github.com/ZikangZhou/QCNet) if you use it.
