"""
Evaluator
---------
Runs inference on any DataLoader and computes classification metrics via sklearn.

Design notes
------------
- ``run()`` has no side effects: it only returns a results dict.
  Saving, plotting, and logging are the caller's responsibility.
  This keeps the class trivially wrappable by MLflow, a CLI script, or a notebook.
- Plots are not created here; pass the returned dict to ``utils.plots``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class Evaluator:
    """Run inference and compute metrics on a single split.

    Parameters
    ----------
    model :
        Trained model. Switched to eval mode automatically.
    loader :
        DataLoader for the split to evaluate.
    device :
        ``"cuda"`` / ``"cpu"`` / ``"mps"``. Auto-detected if ``None``.
    classes :
        Ordered list of class names matching the label indices.
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: Optional[str] = None,
        classes: list[str] = CLASSES,
    ) -> None:
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else
                "cpu"
            )
        self.model   = model.to(device).eval()
        self.loader  = loader
        self.device  = device
        self.classes = classes

    def run(self) -> dict:
        """Collect predictions and compute all metrics.

        Returns
        -------
        dict with keys:
            ``accuracy``        – float, overall top-1 accuracy
            ``report``          – dict from sklearn classification_report
                                  (keys: class names + "macro avg" etc.)
            ``confusion_matrix``– np.ndarray, shape (C, C), raw counts
            ``confusion_matrix_norm`` – np.ndarray, row-normalised (recall per class)
            ``all_preds``       – np.ndarray of predicted label indices
            ``all_labels``      – np.ndarray of true label indices
        """
        all_preds:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in tqdm(self.loader, desc="Evaluating", unit="batch"):
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                preds  = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        report = classification_report(
            y_true, y_pred,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        return {
            "accuracy":             report["accuracy"],
            "report":               report,
            "confusion_matrix":     cm,
            "confusion_matrix_norm": cm_norm,
            "all_preds":            y_pred,
            "all_labels":           y_true,
        }
