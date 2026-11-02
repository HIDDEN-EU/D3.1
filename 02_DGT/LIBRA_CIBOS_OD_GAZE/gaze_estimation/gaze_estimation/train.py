"""Train GazeViT on a face-crop manifest.

    gaze-train configs/example.yaml

Loss: mean Euclidean distance between predicted and true gaze point (normalised space).
Optimiser: Adam over the trainable parameters only, ReduceLROnPlateau on the validation
loss, and early stopping. The checkpoint with the lowest validation loss is kept as
`<output_dir>/best.pt`, and per-epoch metrics go to `<output_dir>/metrics.json`.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ExperimentConfig
from .data import FaceGazeDataset, compute_bounds, read_manifest, select_subjects
from .model import build_model


def euclidean_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean distance between predicted and true points."""
    return torch.linalg.vector_norm(pred - target, dim=1).mean()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record a validation loss; True when training should stop."""
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for faces, targets in loader:
            faces, targets = faces.to(device), targets.to(device)
            loss = euclidean_loss(model(faces), targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(faces)
            count += len(faces)
    return total / max(count, 1)


def train(cfg: ExperimentConfig) -> Path:
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = read_manifest(cfg.data.manifest)
    train_samples = select_subjects(samples, cfg.data.train_subjects)
    val_samples = select_subjects(samples, cfg.data.val_subjects)
    bounds = compute_bounds(train_samples, cfg.data.bounds_quantile)

    size = cfg.data.image_size
    loaders = {
        "train": DataLoader(FaceGazeDataset(train_samples, bounds, size), cfg.train.batch_size,
                            shuffle=True, num_workers=cfg.train.num_workers, drop_last=True),
        "val": DataLoader(FaceGazeDataset(val_samples, bounds, size), cfg.train.batch_size,
                          num_workers=cfg.train.num_workers),
    }

    model = build_model(cfg.model, img_size=size).to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    stopper = EarlyStopping(cfg.train.patience, cfg.train.min_delta)

    best_path = out / "best.pt"
    history = []
    best = float("inf")
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = run_epoch(model, loaders["train"], device, optimizer)
        val_loss = run_epoch(model, loaders["val"], device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  lr {lr:.2e}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(best_path, model, cfg, bounds, epoch, val_loss)
        if stopper.step(val_loss):
            print(f"early stop at epoch {epoch}")
            break

    (out / "metrics.json").write_text(json.dumps(
        {"best_val_loss": best, "bounds": bounds.tolist(), "history": history}, indent=2))
    print(f"best val loss {best:.4f} -> {best_path}")
    return best_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    train(ExperimentConfig.load(ap.parse_args().config))


if __name__ == "__main__":
    main()
