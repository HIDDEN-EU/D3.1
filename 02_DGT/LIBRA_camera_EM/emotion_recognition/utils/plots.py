"""
Visualization utilities.

All functions
  - accept plain data (no model, no loader)
  - return a ``matplotlib.figure.Figure`` — the caller decides whether to
    save, show, or log to MLflow (``mlflow.log_figure(fig, artifact_path)``)
  - accept an optional ``save_path`` for convenience

Usage
-----
>>> from emotion_recognition.utils.plots import (
...     plot_training_curves, plot_confusion_matrix, plot_per_class_metrics
... )
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history: list[dict] | str | Path,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot train/val loss and accuracy from training history.

    Parameters
    ----------
    history :
        Either a list of epoch dicts (from ``Trainer.history``) or a path to
        the ``history.csv`` file saved by the trainer.
    save_path :
        If given, the figure is saved to this path.
    """
    import csv

    if isinstance(history, (str, Path)):
        with open(history, newline="") as f:
            rows = list(csv.DictReader(f))
        history = [{k: float(v) if k != "epoch" else int(v)
                    for k, v in r.items()} for r in rows]

    epochs     = [r["epoch"]      for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]
    train_acc  = [r["train_acc"]  for r in history]
    val_acc    = [r["val_acc"]    for r in history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training history", fontsize=13)

    # Loss
    ax_loss.plot(epochs, train_loss, label="train")
    ax_loss.plot(epochs, val_loss,   label="val", linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, [a * 100 for a in train_acc], label="train")
    ax_acc.plot(epochs, [a * 100 for a in val_acc],   label="val", linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Heatmap of a confusion matrix.

    Parameters
    ----------
    cm :
        Raw counts confusion matrix (shape ``[C, C]``).
    normalize :
        If ``True``, row-normalise so each cell shows recall per class.
    save_path :
        If given, the figure is saved to this path.
    """
    data   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fmt    = ".2f" if normalize else "d"
    title  = "Confusion matrix (row-normalised)" if normalize else "Confusion matrix (counts)"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True, fmt=fmt,
        xticklabels=class_names, yticklabels=class_names,
        cmap="Blues", linewidths=0.5,
        vmin=0, vmax=1 if normalize else None,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Per-class metrics bar chart ───────────────────────────────────────────────

def plot_per_class_metrics(
    report: dict,
    class_names: list[str],
    metrics: list[str] = ("precision", "recall", "f1-score"),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Grouped bar chart of per-class precision, recall, and F1.

    Parameters
    ----------
    report :
        ``output_dict=True`` result from ``sklearn.metrics.classification_report``.
    class_names :
        Class names in label order.
    metrics :
        Which metrics to include in the plot.
    save_path :
        If given, the figure is saved to this path.
    """
    x      = np.arange(len(class_names))
    n      = len(metrics)
    width  = 0.8 / n
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [report.get(cls, {}).get(metric, 0.0) for cls in class_names]
        ax.bar(x + (i - n / 2 + 0.5) * width, values, width=width,
               label=metric, color=color, alpha=0.85)

    # Macro averages as horizontal lines
    for metric, color in zip(metrics, colors):
        macro = report.get("macro avg", {}).get(metric, 0.0)
        ax.axhline(macro, linestyle="--", color=color, alpha=0.5, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics  (dashed = macro average)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
