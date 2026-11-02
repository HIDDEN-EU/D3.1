# Gaze estimation — generic face-crop model

A generic gaze estimation model for the DGT (D3.1 Section 3.2). A Vision Transformer encodes the face crop of the driver, and a small regression head outputs a two-dimensional gaze point on a target plane. The face crop is the only input.

This is a public reference implementation. It is not the gaze model that LIBRA deploys, which is not released. It has the same input and output interface, so the edge service (`../edge_inference`) runs either one.

## Layout

- `gaze_estimation/model.py`: `GazeViT`, a timm Vision Transformer at 112×112 input with a regression head. Only the last transformer blocks are fine-tuned.
- `gaze_estimation/data.py`: manifest reader, target normalisation, face-crop dataset.
- `gaze_estimation/train.py`, `evaluate.py`, `export_onnx.py`: training, evaluation, ONNX export.
- `configs/example.yaml`: example configuration.

## Install

```bash
pip install -e ".[dev]"
```

## Data format

A CSV manifest with one row per face crop: `image_path,subject,x,y`. The `image_path` is relative to the manifest folder. The `x, y` columns hold the gaze target on the target plane. The split is subject-independent: list the subjects of each split in the configuration. The targets are normalised to [-1, 1] between the 10th and the 90th percentile of the training targets.

## Train, evaluate, export

```bash
gaze-train configs/example.yaml
gaze-evaluate runs/example/best.pt --split test
gaze-export-onnx runs/example/best.pt --output gaze_face.onnx
```

The loss is the mean Euclidean distance between the predicted and the true gaze point. The ONNX model has one input `face_img` (B×3×112×112, RGB in [-1, 1]) and one output `gaze` (B×2).

## Tests

```bash
pytest -q
```

## Data and weights

No training data and no trained weights are released. The LIBRA gaze dataset contains facial images of the participants (D3.1 Section 3.2.4).

## Licence

MIT. See [LICENSE](LICENSE).
