# Edge inference service

The DGT edge service for the NVIDIA Jetson Orin Nano with JetPack 6.2 (D3.1 Sections 3.6 and 3.7). All processing is local to the device and no frame is written to storage.

## Layout

- `dgt/sources.py`: frame sources, including the IMX219 CSI camera pipeline (3280×2464 at 21 fps).
- `dgt/models/`: TensorRT runner, face detection and landmarks, face-crop gaze estimator, road-object detector.
- `dgt/calibration.py`: nine-target gaze calibration (per-target median, homography fit, apply).
- `dgt/attention/`: gaze-directed road region, eye-closure signals.
- `dgt/engine_build.py`: `dgt-build-engines`, ONNX to FP16 TensorRT engines on the device.
- `dgt/daemon.py`: `dgt-daemon`, publishes the DGT_DS message ([docs/DGT_DS.md](docs/DGT_DS.md)) on ROS 2.
- `scripts/`: test harnesses for each pipeline stage.
- `deploy/`: systemd units and the Janus gateway configuration (no recording, no STUN or TURN).
- `Dockerfile`, `compose.yaml`: container deployment.

## Install on the Jetson

```bash
python3 -m venv --system-site-packages .venv   # keeps JetPack OpenCV, TensorRT and ROS 2 visible
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Do not install `opencv-python` on the device, because it has no GStreamer support.

## Models and engines

Put the ONNX paths in `dgt/config/models.yaml`, then build the engines on the device:

```bash
dgt-build-engines --engine-dir engines
```

The face detector and the landmark model are `det_500m` and `2d106det` from the InsightFace `buffalo_s` pack. Get them from InsightFace under its licence. The gaze model comes from `../gaze_estimation` and the road detector from `../road_object_detection` (`export-onnx` in each folder).

## Run

```bash
python scripts/testgaze.py --face-engine engines/det_500m_fp16.engine \
    --landmark-engine engines/2d106det_fp16.engine --gaze-engine engines/gaze_face_fp16.engine \
    --source csi --calibration calibration/calibration.json
python scripts/testgazeroi.py --help    # gaze-directed road-object detection
cp .env.example .env && dgt-daemon      # DGT_DS on ROS 2 (source ROS 2 Humble first)
```

For Docker: `docker compose build`, `docker compose run --rm engine-builder`, `docker compose up -d`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests need no camera, GPU or TensorRT.

## Licence

MIT. See [LICENSE](LICENSE). Third-party models are not redistributed.

The edge service imports `road_object_detection.inference`, which is AGPL-3.0. A distribution that contains both, such as the Docker image, falls under AGPL-3.0 as a whole.
