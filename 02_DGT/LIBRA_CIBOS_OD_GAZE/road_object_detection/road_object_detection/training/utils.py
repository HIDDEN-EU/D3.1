"""
Training preparation utilities.

Helpers that sit between the raw dataset config and the Ultralytics trainer:
  - val/test split file generation (_prepare_split)

Called once per run from train.run() and operates on a temp directory.
"""

import random
from pathlib import Path

import yaml

from road_object_detection.config import Experiment


def resolve_dataset_yaml(yaml_path: Path) -> dict:
    """Load a data.yaml, resolving a tile-dataset recipe to its generated dataset.

    Experiments point `dataset.data_yaml` at a recipe under `configs/recipes/` (a yaml
    that declares `class_merges`). The generated tiled dataset's own `data.yaml` —
    written by `tile-dataset` at `<recipe.path>/data.yaml` — is the post-merge file that
    YOLO and the evaluators actually consume. Plain YOLO yamls (no `class_merges`) are
    returned as-is for the rare pre-merge baselines.
    """
    data = yaml.safe_load(yaml_path.read_text())
    if "class_merges" in data:
        output_root = Path(data["path"])
        generated = output_root / "data.yaml"
        if not generated.exists():
            raise FileNotFoundError(
                f"{yaml_path} is a tile-dataset recipe but its generated dataset "
                f"was not found at {generated}. Run tile-dataset first:\n\n"
                f"    tile-dataset --data-yaml {yaml_path} --output {output_root} ...\n"
            )
        data = yaml.safe_load(generated.read_text())
    return data


def generated_data_yaml(yaml_path: Path) -> Path:
    """Path to the post-merge data.yaml the evaluators consume: `<recipe.path>/data.yaml`
    when the yaml declares `class_merges`, else the yaml itself. (The eval CLIs reject a
    raw recipe, so the post-training reminder must hand the developer this resolved path.)"""
    data = yaml.safe_load(yaml_path.read_text())
    if "class_merges" in data:
        return Path(data["path"]) / "data.yaml"
    return yaml_path


def _prepare_split(experiment: Experiment, tmp_dir: Path) -> Path:
    """
    Write a temporary data.yaml with the val/test split and return its path.

    The held-out *test* slice (the first `val_test_ratio` of the seed-shuffled val list)
    is reserved and discarded here; the remaining fraction becomes the training-time val
    set. `road_object_detection.evaluation.occlusion_eval.derive_val_slice` reproduces
    the same remaining slice for post-training evaluation.
    """
    cfg = experiment.dataset
    yaml_path = cfg.data_yaml.resolve()
    data = resolve_dataset_yaml(yaml_path)

    root = Path(data["path"])
    val_images = sorted((root / data["val"]).glob("*.jpg"))

    if not val_images:
        raise RuntimeError(f"No val images found under {root / data['val']}")

    rng = random.Random(cfg.split_seed)
    shuffled = val_images[:]
    rng.shuffle(shuffled)

    n_test = int(len(shuffled) * cfg.val_test_ratio)
    val_imgs = shuffled[n_test:]

    val_txt = tmp_dir / "val_split.txt"
    val_txt.write_text("\n".join(str(p) for p in val_imgs))

    split_yaml = tmp_dir / "split_data.yaml"
    split_yaml.write_text(yaml.dump({**data, "val": str(val_txt)}))

    return split_yaml
