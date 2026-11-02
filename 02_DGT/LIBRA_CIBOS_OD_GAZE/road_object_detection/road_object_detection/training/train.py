"""
Training entrypoint.

Usage:
    train configs/experiments/tiled_v030_mz16_no_ov_notrain_gc_bgcap015_20ep_sgd.json
"""

import os
import sys
import tempfile
from pathlib import Path

import mlflow
import torch
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO
from ultralytics import settings as yolo_settings

from road_object_detection.config import Experiment
from road_object_detection.training.post_eval import run_and_log_occlusion_recall
from road_object_detection.training.utils import _prepare_split, generated_data_yaml


def _stash_batch_previews(trainer) -> None:
    """Move train_batch*/val_batch* previews out of save_dir's top level before the
    MLflow callback globs it, so these low-value images are excluded from MLflow but
    preserved locally under save_dir/batch_previews/."""
    dest = Path(trainer.save_dir) / "batch_previews"
    moved = 0
    previews = list(Path(trainer.save_dir).glob("train_batch*.jpg")) + list(
        Path(trainer.save_dir).glob("val_batch*.jpg")
    )
    for f in previews:
        dest.mkdir(exist_ok=True)
        f.rename(dest / f.name)
        moved += 1
    if moved:
        print(
            f"[train] stashed {moved} batch-preview images into {dest}/ (excluded from MLflow)"
        )


def run(config_path: Path) -> None:
    experiment = Experiment.from_json(config_path)
    hp = experiment.hyperparameters
    hw = experiment.hardware
    mlcfg = experiment.mlflow

    load_dotenv(
        find_dotenv(usecwd=True)
    )  # find .env from cwd, not the install location

    yolo_settings.update({"mlflow": True})

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlcfg.experiment_name
    mlflow.set_experiment(mlcfg.experiment_name)

    print(f"\n[train] Experiment : {experiment.meta.name}")
    print(f"        {experiment.meta.description}")

    data_yaml_path = experiment.dataset.data_yaml.resolve()

    with tempfile.TemporaryDirectory() as tmp:
        split_yaml = _prepare_split(experiment, Path(tmp))

        mlflow.start_run(run_name=experiment.meta.name)
        run_id = mlflow.active_run().info.run_id
        if mlcfg.tags:
            mlflow.set_tags(mlcfg.tags)

        mlflow.log_artifact(str(config_path.resolve()), artifact_path="config")
        mlflow.log_artifact(str(data_yaml_path), artifact_path="config")

        model = YOLO(experiment.model.weights)

        # Keep low-value batch-preview images out of MLflow. Ultralytics' MLflow
        # callback logs every top-level *.jpg in save_dir; this custom on_train_end
        # runs *before* it (user callbacks precede appended integration callbacks) and
        # moves train_batch*/val_batch* into a subdir, which glob("*") won't sweep —
        # so they stay on disk locally but are excluded from the MLflow artifacts.
        model.add_callback("on_train_end", _stash_batch_previews)

        if hp.optimizer == "auto":
            print(
                "[train] WARNING: optimizer='auto' — Ultralytics will choose the "
                "optimizer, lr0 and momentum automatically and IGNORE the configured "
                f"learning_rate={hp.learning_rate} / momentum={hp.momentum}. "
                "Set hyperparameters.optimizer to 'SGD' or 'AdamW' to take control."
            )
        else:
            print(
                f"[train] optimizer={hp.optimizer}  lr0={hp.learning_rate}  lrf={hp.lrf}  "
                f"momentum={hp.momentum}  weight_decay={hp.weight_decay}  "
                f"warmup_epochs={hp.warmup_epochs}  cos_lr={hp.cos_lr}  rect={hp.rect}  "
                f"compile={hp.compile}  amp={hp.amp}  tf32={hp.tf32}  close_mosaic={hp.close_mosaic}  seed={hp.seed}"
            )

        if hp.tf32:
            # Enable TF32 matmul on Ampere+ tensor cores. Matters most when amp=False:
            # recovers most of the fp32 speed while keeping fp32's exponent range, so it
            # cannot overflow the way fp16 (amp=True) did. Accuracy-neutral.
            torch.set_float32_matmul_precision("high")

        results = model.train(
            data=str(split_yaml),
            epochs=hp.epochs,
            batch=hp.batch_size,
            imgsz=experiment.model.imgsz,
            optimizer=hp.optimizer,
            lr0=hp.learning_rate,
            lrf=hp.lrf,
            momentum=hp.momentum,
            weight_decay=hp.weight_decay,
            warmup_epochs=hp.warmup_epochs,
            cos_lr=hp.cos_lr,
            rect=hp.rect,
            compile=hp.compile,
            amp=hp.amp,
            close_mosaic=hp.close_mosaic,
            patience=hp.patience,
            seed=hp.seed,
            device=hw.device,
            workers=hw.workers,
            project=str(experiment.project_dir.resolve()),
            name=experiment.meta.name,
        )

    run_dir = Path(results.save_dir)
    experiment.to_json(run_dir / "experiment.json")

    # Post-training occlusion-recall eval, logged into the same MLflow run. Ultralytics'
    # MLflow callback closes the run at train end, so the hook resumes it by run_id.
    # Non-fatal: a post-eval failure must never discard a completed training run.
    try:
        run_and_log_occlusion_recall(experiment, run_dir, run_id, hw.device)
    except Exception as e:  # noqa: BLE001 — never let post-eval kill a finished run
        print(f"[post-eval] occlusion-recall eval failed (non-fatal): {e}")

    print(f"\n[train] Done. Artifacts at: {run_dir}")

    # Reminder: val-slice recall was just logged; the held-out TEST sign-off is manual.
    # Run this on your CHOSEN model (not every run, to avoid leakage into the test slice).
    best = run_dir / "weights" / "best.pt"
    gen_yaml = generated_data_yaml(data_yaml_path)
    print(
        "\n[next] Val-slice occlusion recall was logged to this MLflow run. For the\n"
        "       held-out TEST sign-off on this checkpoint (recall + mAP, logged to a\n"
        "       separate '<run>_test' MLflow run), copy-paste:\n\n"
        f"         evaluate-model \\\n"
        f"             --weights   {best} \\\n"
        f"             --data-yaml {gen_yaml}\n"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: train <config.json>")
        sys.exit(1)
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
