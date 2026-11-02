"""
Train from an experiment JSON config.

Usage
-----
train --config configs/experiments/fer2013_baseline.json
train --config configs/experiments/kdef_frontal.json
"""

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.pytorch

from emotion_recognition.config import ExperimentConfig
from emotion_recognition.training import Trainer, MLflowCallback


def _flatten_config(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict into dotted keys for mlflow.log_params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_config(v, key + "."))
        else:
            out[key] = str(v)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", required=True, type=Path, help="Path to experiment JSON file"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig.from_json(args.config)
    print(f"Experiment : {cfg.name}")
    print(f"Dataset    : {cfg.data.dataset}  root={cfg.data.root}")
    print(f"Model      : {cfg.model.backbone}  pretrained={cfg.model.pretrained}")
    print(
        f"Epochs     : {cfg.training.epochs}  lr={cfg.training.lr}  warmup={cfg.training.warmup_epochs}"
    )

    # MLflow tracking — URI comes from MLFLOW_TRACKING_URI env var
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        print("WARNING: MLFLOW_TRACKING_URI is not set — logging to local ./mlruns/")
    experiment_name = cfg.mlflow.experiment_name or cfg.name
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        uri = mlflow.get_tracking_uri()
        print(
            f"\nERROR: Cannot connect to MLflow server at {uri}\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            f"Start the server with:\n"
            f"  mlflow server --host 127.0.0.1 --port 5000\n"
            f"Or unset MLFLOW_TRACKING_URI to log locally to ./mlruns/\n"
        )
        raise SystemExit(1) from None
    print(f"MLflow         : {mlflow.get_tracking_uri()}  experiment={experiment_name}")

    # Resolve tags: auto-derive from config fields + custom static tags
    flat = _flatten_config(
        {k: v for k, v in cfg.model_dump(mode="json").items() if k != "mlflow"}
    )
    tags = {k: flat[k] for k in cfg.mlflow.auto_tag_fields if k in flat}
    tags.update(cfg.mlflow.tags)

    with mlflow.start_run(run_name=cfg.name, tags=tags):
        mlflow.log_params(flat)
        mlflow.log_artifact(str(args.config))

        model, loaders, callbacks = cfg.to_runtime()
        callbacks.append(MLflowCallback())
        checkpoint_dir = cfg.training.checkpoint_dir / cfg.name
        print(f"Splits     : { {k: len(v.dataset) for k, v in loaders.items()} }")
        print(f"Device     : {cfg.training.device}  AMP={cfg.training.amp}")
        print(f"Checkpoint : {checkpoint_dir}/{cfg.training.checkpoint_name}")
        print(f"Callbacks  : {[type(cb).__name__ for cb in callbacks]}")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_json(checkpoint_dir / "experiment.json")

        trainer = Trainer(
            model,
            loaders,
            cfg.training,
            callbacks=callbacks,
            checkpoint_dir=checkpoint_dir,
        )
        trainer.fit()

        trainer.load_best()
        trainer.evaluate("test")

        mlflow.log_artifact(str(checkpoint_dir / cfg.training.checkpoint_name))
        mlflow.pytorch.log_model(trainer.model, "model")


if __name__ == "__main__":
    main()
