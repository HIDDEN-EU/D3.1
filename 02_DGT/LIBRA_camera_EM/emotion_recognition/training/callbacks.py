"""
Training callbacks.

Each callback receives an :class:`EpochResult` after every epoch and returns
``True`` to stop training or ``False`` to continue.

Adding a new callback requires no changes to ``Trainer`` — just append it to
``TrainConfig.callbacks``.

MLflow note
-----------
A future ``MLflowCallback`` can live here and log metrics on every
``on_epoch_end`` call without touching ``Trainer`` internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpochResult:
    """Snapshot of one training epoch passed to every callback."""
    epoch:      int
    train_loss: float
    train_acc:  float
    val_loss:   float
    val_acc:    float


class Callback:
    """Base class — override ``on_epoch_end`` to implement logic."""

    def on_epoch_end(self, result: EpochResult) -> bool:
        """Return ``True`` to stop training."""
        return False


# ── Built-in callbacks ────────────────────────────────────────────────────────

class EarlyStoppingCallback(Callback):
    """Stop when a monitored metric shows no improvement for ``patience`` epochs.

    Parameters
    ----------
    monitor :
        ``"val_loss"`` (lower is better) or ``"val_acc"`` (higher is better).
    patience :
        Number of epochs with no improvement before stopping.
    min_delta :
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> None:
        if monitor not in ("val_loss", "val_acc"):
            raise ValueError("monitor must be 'val_loss' or 'val_acc'")
        self.monitor   = monitor
        self.patience  = patience
        self.min_delta = min_delta
        self._best:    float = float("inf") if monitor == "val_loss" else float("-inf")
        self._counter: int   = 0

    def on_epoch_end(self, result: EpochResult) -> bool:
        current = result.val_loss if self.monitor == "val_loss" else result.val_acc
        improved = (
            current < self._best - self.min_delta
            if self.monitor == "val_loss"
            else current > self._best + self.min_delta
        )
        if improved:
            self._best   = current
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            print(
                f"\n[EarlyStopping] {self.monitor} has not improved for "
                f"{self.patience} epochs (best={self._best:.4f}). Stopping."
            )
            return True
        return False


class OverfitDetectorCallback(Callback):
    """Stop when the model shows clear overfitting signs.

    Overfitting is detected when **both** conditions hold for ``patience``
    consecutive epochs:
      1. val_loss   is increasing  (or not improving by ``min_delta``)
      2. train_loss is decreasing  (model keeps fitting training data)

    This is stricter than ``EarlyStoppingCallback`` on ``val_loss`` alone —
    it avoids stopping during a temporary val plateau while the model is
    still learning on the training set.

    Parameters
    ----------
    patience :
        Consecutive epochs both conditions must hold before stopping.
    min_delta :
        Minimum change in val_loss to count as "increasing".
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-3) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self._prev_val_loss:   float | None = None
        self._prev_train_loss: float | None = None
        self._counter: int = 0

    def on_epoch_end(self, result: EpochResult) -> bool:
        stop = False

        if self._prev_val_loss is not None and self._prev_train_loss is not None:
            val_worsening   = result.val_loss   > self._prev_val_loss   - self.min_delta
            train_improving = result.train_loss < self._prev_train_loss

            if val_worsening and train_improving:
                self._counter += 1
            else:
                self._counter = 0

            if self._counter >= self.patience:
                gap = result.val_loss - result.train_loss
                print(
                    f"\n[OverfitDetector] Overfitting for {self.patience} consecutive "
                    f"epochs (train_loss={result.train_loss:.4f}, "
                    f"val_loss={result.val_loss:.4f}, gap={gap:.4f}). Stopping."
                )
                stop = True

        self._prev_val_loss   = result.val_loss
        self._prev_train_loss = result.train_loss
        return stop


class MLflowCallback(Callback):
    """Log per-epoch metrics to an active MLflow run.

    Must be used inside a ``with mlflow.start_run():`` context in the caller.
    Logs ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc`` at each epoch.
    """

    def on_epoch_end(self, result: EpochResult) -> bool:
        import mlflow
        mlflow.log_metrics(
            {
                "train_loss": result.train_loss,
                "train_acc":  result.train_acc,
                "val_loss":   result.val_loss,
                "val_acc":    result.val_acc,
            },
            step=result.epoch,
        )
        return False
