"""
Generic manifest-based dataset and sampling utilities.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_LABEL: dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}


class ManifestDataset(Dataset):
    """Dataset backed by a list of (path, label) records.

    Consumed by :func:`~emotion_recognition.data.loader.build_loaders` after
    the manifest CSV has been partitioned into train / val / test splits.

    Parameters
    ----------
    records :
        List of ``(absolute_image_path, integer_label)`` tuples.
    transform :
        Optional callable applied to each PIL image before returning.
    """

    def __init__(
        self,
        records: list[tuple[str, int]],
        transform=None,
    ) -> None:
        self._records = records
        self.targets = [label for _, label in records]
        self.transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        path, label = self._records[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def compute_sample_weights(targets: list[int]) -> Tensor:
    """Return per-sample inverse-frequency weights for WeightedRandomSampler.

    Parameters
    ----------
    targets :
        Integer class label for each sample in the dataset.

    Returns
    -------
    Tensor of shape ``(N,)`` where each value is ``1 / class_count``.
    """
    t = torch.tensor(targets, dtype=torch.long)
    counts = torch.bincount(t)
    weights = 1.0 / counts.float()
    return weights[t]
