"""
Single-image inference: one face crop in, one emotion class and its confidence out.

Usage
-----
    from emotion_recognition.data import rafdb_transforms
    from emotion_recognition.inference import predict

    label, confidence = predict(model, face_crop, rafdb_transforms("test"))

``model`` is any trained classifier of this package (``FERClassifier`` or
``SwinFaceClassifier``) that returns logits over the seven classes of
``EMOTIONS``. ``transform`` must be the evaluation transform of the dataset the
model was trained on, so the crop is resized and normalised the same way.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .data.base import EMOTIONS


def _to_pil(face_crop: Image.Image | np.ndarray) -> Image.Image:
    """Accept a PIL image or an RGB ``uint8`` array of shape (H, W, 3) or (H, W)."""
    if isinstance(face_crop, Image.Image):
        return face_crop.convert("RGB")
    return Image.fromarray(np.asarray(face_crop, dtype=np.uint8)).convert("RGB")


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    face_crop: Image.Image | np.ndarray,
    transform: Callable,
    device: str = "cpu",
) -> np.ndarray:
    """Return the softmax probabilities over ``EMOTIONS`` for one face crop."""
    model.eval()
    x = transform(_to_pil(face_crop)).unsqueeze(0).to(device)
    logits = model(x)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def predict(
    model: nn.Module,
    face_crop: Image.Image | np.ndarray,
    transform: Callable,
    device: str = "cpu",
) -> tuple[str, float]:
    """Return ``(emotion, confidence)``: the top class and its softmax probability."""
    probs = predict_proba(model, face_crop, transform, device)
    idx = int(probs.argmax())
    return EMOTIONS[idx], float(probs[idx])
