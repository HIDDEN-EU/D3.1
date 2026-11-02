# LIBRA_camera_EM — facial emotion recognition

Training, evaluation and inference package for the facial emotion recognition of the DGT (D3.1 Section 3.4). The model classifies a face crop of the driver into seven classes: angry, disgust, fear, happy, neutral, sad, surprise. No final model is selected yet.

## Layout

- `emotion_recognition/models/`: classifiers (EfficientNet, ResNet, SwinFace backbone).
- `emotion_recognition/data/`: manifest loading, splits and transforms for each dataset.
- `emotion_recognition/train.py`, `evaluate.py`: training and evaluation entry points.
- `emotion_recognition/inference.py`: `predict(model, face_crop, transform)` returns the class and its softmax confidence.
- `scripts/prepare_<dataset>.py`: writes the manifest CSV for one dataset.
- `configs/rafdb_efficientnet_b0.json`: example experiment configuration.

## Install

```bash
pip install -e ".[dev]"
```

## Datasets

The package supports FER2013, FER+, KDEF, RAF-DB and an AffectNet subset. The datasets are not redistributed. Get each one from its original source under its own licence, then write its manifest:

```bash
python scripts/prepare_rafdb.py --root data/raf-db
```

The docstring at the top of each `prepare_*.py` script gives the folder layout that the script expects.

## Train and evaluate

```bash
train --config configs/rafdb_efficientnet_b0.json
test --config checkpoints/rafdb_efficientnet_b0/experiment.json
```

The metrics are accuracy and macro F1. MLflow logs to `./mlruns` unless `MLFLOW_TRACKING_URI` is set. The SwinFace backbone needs the public SwinFace checkpoint (<https://github.com/lxq1000/SwinFace>) in `model.weights_path`.

## Tests

```bash
pytest tests -q
```

## Weights

The weights of the selected model will be published after the model selection.

## Licence

MIT. See [LICENSE](LICENSE). `emotion_recognition/models/swin_backbone.py` is a copy of the SwinFace backbone, which is MIT-licensed; its notices are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

The code licence does not cover model weights. Weights trained on RAF-DB, KDEF or AffectNet are for research use only, under the terms of those datasets.
