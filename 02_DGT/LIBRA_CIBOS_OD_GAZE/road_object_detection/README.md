# Road-scene object detection

Training, evaluation and inference package for the road-scene detector of the DGT (D3.1 Section 3.3). The detector is a YOLO11 model fine-tuned on a 9-class variant of BDD100K: pedestrian, rider, car, truck, bus, motorcycle, bicycle, traffic light, traffic sign. The package also contains the tool that builds the variant from the original BDD100K dataset.

## Layout

- `road_object_detection/inference.py`: NumPy-only tiled inference. Returns a box, a class and a confidence for each detection. The edge service (`../edge_inference`) imports it.
- `road_object_detection/training/`, `config/`, `evaluation/`: training, experiment schema, metrics.
- `scripts/`: dataset conversion and tiling, evaluation, ONNX export, viewers.
- `configs/recipes/data_bdd100k_9class.yaml`: class merges and drops for the 9-class variant.
- `configs/experiments/yolo11s_bdd100k_9class.json`: training configuration.

## Install

```bash
pip install -e ".[train,test]"     # training, evaluation, tests
pip install -e ".[export]"         # ONNX export
```

## Build the dataset

Get BDD100K from its original source under its own licence. The variant is not redistributed.

```bash
python scripts/bdd100k.py convert --aggregate data/bdd100k/labels/det_20/det_train.json --output data/bdd100k/labels/100k/train
python scripts/bdd100k.py convert --aggregate data/bdd100k/labels/det_20/det_val.json --output data/bdd100k/labels/100k/val
tile-dataset --data-yaml configs/recipes/data_bdd100k_9class.yaml --output data/bdd100k_variant_9class \
    --mode r --num-tiles 6 --min-visibility 0.30 --min-size 16 \
    --guarantee-classes 1,5,6 --max-background-ratio 0.15 --seed 42 --workers 4
```

Set `path` in the recipe to your BDD100K root. The experiment configuration reads the generated `data/bdd100k_variant_9class/data.yaml`.

## Train, evaluate, export

```bash
train configs/experiments/yolo11s_bdd100k_9class.json
evaluate-model --help
export-onnx --weights runs/train/<exp>/weights/best.pt --dry-run
```

The validation crops are split into validation and held-out test at seed 42 with a ratio of 0.3. MLflow logs to `./mlruns` unless `MLFLOW_TRACKING_URI` is set.

## Tests

```bash
pytest tests -q
```

## Weights

The fine-tuned weights are not in this repository. A download link will be added on release.

## Licence

AGPL-3.0. See [LICENSE](LICENSE). The package uses Ultralytics YOLO11, which is AGPL-3.0, so the code and the fine-tuned weights are AGPL-3.0. The weights are also subject to the BDD100K terms.
