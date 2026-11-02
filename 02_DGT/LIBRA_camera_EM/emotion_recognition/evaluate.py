"""
Evaluate a saved checkpoint using its experiment config.

Usage
-----
test --config checkpoints/fer2013_baseline/experiment.json
test --config checkpoints/kdef_frontal/experiment.json --split val
test --config checkpoints/fer2013_baseline/experiment.json --run-id <mlflow_run_id>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch

from .config import ExperimentConfig
from .data import build_loaders
from .evaluation import Evaluator
from .models import FERClassifier
from .utils.plots import plot_confusion_matrix, plot_per_class_metrics

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a FERClassifier checkpoint.")
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to experiment.json saved during training",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--run-id",
        default="",
        help="Existing MLflow run ID to attach results to (from training)",
    )
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def load_model(checkpoint: Path, cfg: ExperimentConfig, device: str) -> FERClassifier:
    model = FERClassifier(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        dropout=cfg.model.dropout,
        pretrained=False,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.2%}")
    return model


def print_summary(results: dict, classes: list[str]) -> None:
    report = results["report"]
    print(f"\n{'─'*48}")
    print(f"  Overall accuracy : {results['accuracy']:.2%}")
    print(f"  Macro F1         : {report['macro avg']['f1-score']:.4f}")
    print(f"  Macro precision  : {report['macro avg']['precision']:.4f}")
    print(f"  Macro recall     : {report['macro avg']['recall']:.4f}")
    print(
        f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    )
    print(f"  {'─'*44}")
    for cls in classes:
        m = report.get(cls, {})
        print(
            f"  {cls:<12} {m.get('precision',0):>10.4f} "
            f"{m.get('recall',0):>10.4f} {m.get('f1-score',0):>10.4f} "
            f"{m.get('support',0):>10.0f}"
        )
    print(f"{'─'*48}\n")


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ExperimentConfig.from_json(args.config)
    print(f"Experiment : {cfg.name}")
    print(f"Dataset    : {cfg.data.dataset.value}  backbone={cfg.model.backbone}")

    ckpt_dir = args.config.parent
    checkpoint = ckpt_dir / cfg.training.checkpoint_name

    loaders = build_loaders(
        dataset=cfg.data.dataset,
        root=cfg.data.root,
        batch_size=cfg.data.batch_size,
        num_workers=args.workers,
        split_cfg=cfg.data.split,
        angles=cfg.data.angles,
        genders=cfg.data.genders,
        sessions=cfg.data.sessions,
    )
    loader = loaders[args.split]
    print(f"Split '{args.split}': {len(loader.dataset)} samples")

    model = load_model(checkpoint, cfg, device)
    evaluator = Evaluator(model=model, loader=loader, device=device, classes=CLASSES)
    results = evaluator.run()
    print_summary(results, CLASSES)

    mlflow.set_experiment(cfg.mlflow.experiment_name or cfg.name)

    run_id = args.run_id or None
    with mlflow.start_run(run_id=run_id, run_name=cfg.name):
        mlflow.log_metrics(
            {
                "test_accuracy": results["accuracy"],
                "test_macro_f1": results["report"]["macro avg"]["f1-score"],
                "test_macro_prec": results["report"]["macro avg"]["precision"],
                "test_macro_rec": results["report"]["macro avg"]["recall"],
            }
        )

        fig_cm = plot_confusion_matrix(
            results["confusion_matrix"], class_names=CLASSES, normalize=True
        )
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        fig_cm_counts = plot_confusion_matrix(
            results["confusion_matrix"], class_names=CLASSES, normalize=False
        )
        mlflow.log_figure(fig_cm_counts, "confusion_matrix_counts.png")
        plt.close(fig_cm_counts)

        fig_per = plot_per_class_metrics(results["report"], class_names=CLASSES)
        mlflow.log_figure(fig_per, "per_class_metrics.png")
        plt.close(fig_per)


if __name__ == "__main__":
    main()
