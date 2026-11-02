"""
Trainer
-------
Self-contained training loop for FERClassifier.

Features
--------
- Mixed-precision training (torch.amp)
- Gradient clipping
- OneCycleLR scheduler
- Per-class accuracy tracking
- Best-checkpoint saving (highest val accuracy)
- Early stopping
- Two-phase fine-tuning: frozen backbone warmup → full unfreeze

Usage
-----
>>> from emotion_recognition.config  import ExperimentConfig
>>> cfg = ExperimentConfig.from_json("configs/experiments/fer2013_baseline.json")
>>> model, loaders, callbacks = cfg.to_runtime()
>>> trainer = Trainer(
...     model, loaders, cfg.training,
...     callbacks=callbacks,
...     checkpoint_dir=cfg.training.checkpoint_dir / cfg.name,
... )
>>> history = trainer.fit()
"""
from __future__ import annotations

import time
from pathlib import Path

from .callbacks import Callback, EpochResult

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """Encapsulates the training loop for :class:`~emotion_recognition.models.FERClassifier`.

    Parameters
    ----------
    model :
        The classifier to train.
    loaders :
        Dict with keys ``"train"``, ``"val"`` (and optionally ``"test"``).
    config :
        :class:`~emotion_recognition.config.TrainingConfig` instance.
    callbacks :
        List of :class:`~emotion_recognition.training.Callback` objects run
        after every epoch. Any callback returning ``True`` stops training.
    class_weights :
        Optional tensor of per-class loss weights (e.g. from dataset imbalance).
    checkpoint_dir :
        Directory for checkpoints. Overrides ``config.checkpoint_dir`` when
        provided (used to inject the experiment name, e.g. ``checkpoints/run1``).
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        config,                              # TrainingConfig from emotion_recognition.config
        *,
        callbacks: list[Callback] | None = None,
        class_weights: torch.Tensor | None = None,
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        # Resolve device
        if config.device == "auto":
            device = (
                "cuda" if torch.cuda.is_available() else
                "mps"  if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            device = config.device

        self._device          = device
        self._callbacks       = callbacks or []
        self._checkpoint_dir  = Path(checkpoint_dir) if checkpoint_dir is not None else config.checkpoint_dir

        self.model   = model.to(device)
        self.loaders = loaders
        self.cfg     = config

        # AMP scaler (no-op on CPU/MPS)
        use_amp = config.amp and device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.amp_ctx = torch.amp.autocast(
            device_type=device if device != "mps" else "cpu",
            enabled=use_amp,
        )

        # Loss
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=config.label_smoothing,
        )

        # Optimizer & scheduler are built in fit() after phase transitions
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self._best_val_acc: float = 0.0
        self._patience_counter: int = 0
        self.history: list[dict] = []

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self) -> list[dict]:
        """Run the full training schedule.

        Returns
        -------
        List of per-epoch metric dicts.
        """
        cfg = self.cfg
        epoch_bar = tqdm(range(1, cfg.epochs + 1), desc="Epochs", unit="epoch")
        for epoch in epoch_bar:
            # Phase transition: unfreeze after warmup
            if epoch == 1 and cfg.warmup_epochs > 0:
                print("Phase 1 — head only (backbone frozen)")
                self.model.freeze_backbone()
                self._build_optimizer_and_scheduler(phase=1)
            elif epoch == cfg.warmup_epochs + 1:
                print("Phase 2 — full model (backbone unfrozen)")
                self.model.unfreeze_backbone()
                self._build_optimizer_and_scheduler(phase=2)

            t0 = time.perf_counter()
            train_metrics = self._run_epoch("train")
            val_metrics   = self._run_epoch("val")
            elapsed = time.perf_counter() - t0

            row = {
                "epoch":             epoch,
                "train_loss":        train_metrics["loss"],
                "train_acc":         train_metrics["acc"],
                "val_loss":          val_metrics["loss"],
                "val_acc":           val_metrics["acc"],
                "val_per_class_acc": val_metrics["per_class_acc"],
                "lr":                self.optimizer.param_groups[0]["lr"],
                "time_s":            elapsed,
            }
            self.history.append(row)
            self._log(row)
            epoch_bar.set_postfix(
                val_loss=f"{val_metrics['loss']:.4f}",
                val_acc=f"{val_metrics['acc']:.2%}",
                best=f"{self._best_val_acc:.2%}",
            )

            # Checkpoint best model
            if val_metrics["acc"] > self._best_val_acc:
                self._best_val_acc = val_metrics["acc"]
                self._save_checkpoint(epoch, val_metrics)
                self._patience_counter = 0
            else:
                self._patience_counter += 1

            if self._patience_counter >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs).")
                break

            # Run user-defined callbacks
            result = EpochResult(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_acc=train_metrics["acc"],
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["acc"],
            )
            if any(cb.on_epoch_end(result) for cb in self._callbacks):
                break

        print(f"\nBest val accuracy: {self._best_val_acc:.2%}")
        return self.history

    def evaluate(self, split: str = "test") -> dict:
        """Run evaluation on a given split and print per-class results."""
        if split not in self.loaders:
            raise KeyError(f"Loader for split {split!r} not provided.")
        metrics = self._run_epoch(split)
        print(f"\n{'─'*40}")
        print(f"{split.upper()} RESULTS")
        print(f"  Loss : {metrics['loss']:.4f}")
        print(f"  Acc  : {metrics['acc']:.2%}")
        print("  Per-class accuracy:")
        for cls, acc in zip(CLASSES, metrics["per_class_acc"]):
            print(f"    {cls:<10s} {acc:.2%}")
        return metrics

    def load_best(self) -> None:
        """Restore the best checkpoint saved during training."""
        path = self._checkpoint_dir / self.cfg.checkpoint_name
        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
              f"(val_acc={ckpt['val_acc']:.2%})")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_optimizer_and_scheduler(self, phase: int) -> None:
        lr = self.cfg.lr if phase == 2 else self.cfg.lr * 5  # larger LR for head-only warmup
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=self.cfg.weight_decay,
        )
        steps_per_epoch = len(self.loaders["train"])
        if phase == 1:
            remaining = self.cfg.warmup_epochs
        else:
            remaining = self.cfg.epochs - self.cfg.warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=remaining,
            pct_start=self.cfg.pct_start,
            div_factor=self.cfg.div_factor,
            final_div_factor=self.cfg.final_div_factor,
        )

    def _run_epoch(self, split: str) -> dict:
        training = split == "train"
        self.model.train(training)
        loader = self.loaders[split]

        total_loss = 0.0
        correct    = 0
        total      = 0
        class_correct = torch.zeros(len(CLASSES))
        class_total   = torch.zeros(len(CLASSES))

        pbar = tqdm(loader, desc=f"{split:>5}", unit="batch", leave=False)
        with torch.set_grad_enabled(training):
            for images, labels in pbar:
                images = images.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)

                with self.amp_ctx:
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()

                preds = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                for c in range(len(CLASSES)):
                    mask = labels == c
                    class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    class_total[c]   += mask.sum().item()

                pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.2%}")

        per_class_acc = (class_correct / class_total.clamp(min=1)).tolist()
        return {
            "loss":          total_loss / total,
            "acc":           correct / total,
            "per_class_acc": per_class_acc,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict) -> None:
        path = self._checkpoint_dir / self.cfg.checkpoint_name
        torch.save({
            "epoch":     epoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "val_acc":   metrics["acc"],
            "val_loss":  metrics["loss"],
        }, path)

    @staticmethod
    def _log(row: dict) -> None:
        per_cls = " | ".join(
            f"{c[:3]}={a:.0%}"
            for c, a in zip(CLASSES, row["val_per_class_acc"])
        )
        print(
            f"[{row['epoch']:>3}] "
            f"loss {row['train_loss']:.4f}/{row['val_loss']:.4f}  "
            f"acc {row['train_acc']:.2%}/{row['val_acc']:.2%}  "
            f"lr {row['lr']:.2e}  "
            f"({row['time_s']:.0f}s)  "
            f"{per_cls}"
        )
