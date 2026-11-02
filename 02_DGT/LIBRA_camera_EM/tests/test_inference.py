import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from emotion_recognition.data.base import EMOTIONS
from emotion_recognition.inference import predict, predict_proba


class _FixedLogits(nn.Module):
    """Stub classifier: always favours one class."""

    def __init__(self, winner: int) -> None:
        super().__init__()
        self.winner = winner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], len(EMOTIONS))
        logits[:, self.winner] = 5.0
        return logits


_TF = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


def test_predict_returns_top_class_and_softmax_confidence():
    crop = np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8)
    label, conf = predict(_FixedLogits(3), crop, _TF)
    assert label == EMOTIONS[3]
    expected = np.exp(5.0) / (np.exp(5.0) + len(EMOTIONS) - 1)
    assert abs(conf - expected) < 1e-5


def test_predict_proba_sums_to_one_for_grayscale_input():
    crop = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
    probs = predict_proba(_FixedLogits(0), crop, _TF)
    assert probs.shape == (len(EMOTIONS),)
    assert abs(probs.sum() - 1.0) < 1e-5
