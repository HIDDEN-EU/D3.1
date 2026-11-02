# LIBRA_CIBOS_OD_GAZE — gaze estimation and gaze-area object detection

Software of the DGT sub-component that estimates where the driver looks and what the driver looks at (D3.1 Sections 3.2, 3.3, 3.6 and 3.7). It outputs the driver attention object and location (Driver att OD).

| Folder | Content | D3.1 |
| --- | --- | --- |
| [gaze_estimation](gaze_estimation) | Generic face-crop Vision Transformer that regresses the gaze point: training, evaluation, ONNX export | 3.2 |
| [road_object_detection](road_object_detection) | YOLO11 road-scene detector on a 9-class BDD100K variant: dataset tool, training, evaluation, tiled inference | 3.3 |
| [edge_inference](edge_inference) | Edge service for the NVIDIA Jetson Orin Nano: camera capture, TensorRT engines, gaze calibration, gaze-directed road detection, ROS 2 output | 3.6, 3.7 |
| [driver_monitoring_tool](driver_monitoring_tool) | The calibration and driver-monitoring web tool (LIBRA, CIBOS) | 3.6 |

The gaze model that LIBRA deploys is not released. `gaze_estimation` gives a public model with the same interface: a 112×112 face crop in, a gaze point out.

No training data, recorded footage or third-party model weights are in this repository. Each folder states where its weights come from.

## Licence

Each folder has its own licence:

- `gaze_estimation`, `edge_inference`: MIT.
- `road_object_detection`: AGPL-3.0, because it uses Ultralytics YOLO11.
- `driver_monitoring_tool`: to be confirmed by CIBOS.
