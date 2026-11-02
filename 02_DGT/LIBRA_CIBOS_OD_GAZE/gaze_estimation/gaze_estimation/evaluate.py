"""Evaluate a checkpoint on the test (or validation) subjects.

    gaze-evaluate runs/example/best.pt --split test

Reports the mean, median and 90th-percentile Euclidean error in the normalised [-1, 1]
space, and the mean error in target-plane units (the inverse of the normalisation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import FaceGazeDataset, denormalise, read_manifest, select_subjects


def error_metrics(pred: np.ndarray, target: np.ndarray, bounds: np.ndarray) -> dict:
    err = np.linalg.norm(pred - target, axis=1)
    err_units = np.linalg.norm(denormalise(pred, bounds) - denormalise(target, bounds), axis=1)
    return {
        "n": int(len(err)),
        "mean_error_norm": float(err.mean()),
        "median_error_norm": float(np.median(err)),
        "p90_error_norm": float(np.quantile(err, 0.9)),
        "mean_error_units": float(err_units.mean()),
    }


@torch.no_grad()
def evaluate(checkpoint: Path, split: str = "test", manifest: Path | None = None,
             batch_size: int = 64, device: str = "cpu") -> dict:
    model, cfg, bounds = load_checkpoint(checkpoint, device)
    subjects = {"train": cfg.data.train_subjects, "val": cfg.data.val_subjects,
                "test": cfg.data.test_subjects}[split]
    samples = select_subjects(read_manifest(manifest or cfg.data.manifest), subjects)
    if not samples:
        raise ValueError(f"no samples for the {split} subjects {subjects}")
    loader = DataLoader(FaceGazeDataset(samples, bounds, cfg.data.image_size), batch_size)
    preds, targets = [], []
    for faces, target in loader:
        preds.append(model(faces.to(device)).cpu().numpy())
        targets.append(target.numpy())
    return error_metrics(np.concatenate(preds), np.concatenate(targets), bounds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--manifest", type=Path, default=None, help="override the config manifest")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.split, args.manifest, device=args.device),
                     indent=2))


if __name__ == "__main__":
    main()
