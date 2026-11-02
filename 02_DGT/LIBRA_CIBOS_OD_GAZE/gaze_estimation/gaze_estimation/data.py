"""Manifest, target normalisation and the face-crop dataset.

The manifest is a CSV with one row per face crop:

    image_path,subject,x,y
    crops/s01/000001.jpg,s01,0.32,1.10

`image_path` is relative to the manifest folder (or absolute). `x, y` is the gaze target
on the target plane, in any fixed unit. Targets are normalised per axis to [-1, 1] between
the `q` and `1 - q` quantiles of the TRAIN split; the same bounds are applied to every split
and saved with the checkpoint.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    subject: str
    x: float
    y: float


def read_manifest(path: Path) -> List[Sample]:
    path = Path(path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = {"image_path", "subject", "x", "y"} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    root = path.parent
    return [
        Sample(
            image_path=(root / r["image_path"]) if not Path(r["image_path"]).is_absolute()
            else Path(r["image_path"]),
            subject=str(r["subject"]),
            x=float(r["x"]),
            y=float(r["y"]),
        )
        for r in rows
    ]


def select_subjects(samples: Sequence[Sample], subjects: Sequence[str]) -> List[Sample]:
    wanted = set(subjects)
    return [s for s in samples if s.subject in wanted]


def compute_bounds(samples: Sequence[Sample], q: float = 0.1) -> np.ndarray:
    """[[x_min, x_max], [y_min, y_max]] at the q and 1 - q quantiles."""
    if not samples:
        raise ValueError("cannot compute bounds of an empty split")
    xy = np.array([[s.x, s.y] for s in samples], dtype=np.float64)
    lo = np.quantile(xy, q, axis=0)
    hi = np.quantile(xy, 1.0 - q, axis=0)
    if np.any(hi - lo <= 0):
        raise ValueError("degenerate bounds: the train targets do not vary on an axis")
    return np.stack([lo, hi], axis=1)


def normalise(xy: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return 2.0 * (np.asarray(xy, dtype=np.float64) - lo) / (hi - lo) - 1.0


def denormalise(xy: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (np.asarray(xy, dtype=np.float64) + 1.0) / 2.0 * (hi - lo) + lo


def load_face(path: Path, image_size: int = 112) -> np.ndarray:
    """Face crop -> (3, S, S) float32 RGB in [-1, 1]: (x / 255 - 0.5) / 0.5."""
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - 0.5) / 0.5).transpose(2, 0, 1).copy()


class FaceGazeDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], bounds: np.ndarray, image_size: int = 112) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.targets = normalise([[s.x, s.y] for s in self.samples], bounds).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        face = load_face(self.samples[idx].image_path, self.image_size)
        return torch.from_numpy(face), torch.from_numpy(self.targets[idx])
