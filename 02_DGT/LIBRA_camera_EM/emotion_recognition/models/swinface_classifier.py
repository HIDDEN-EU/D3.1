"""
SwinFaceClassifier — SwinFace backbone adapted for emotion recognition.

Uses the SwinTransformer backbone pretrained on face recognition (MS1M-RetinaFace).
The pretrained weights are loaded from a checkpoint produced by the SwinFace project
(checkpoint_step_79999_gpu_0.pt).

The backbone's forward() returns (local_features, global_features, embedding[B, 512]).
Only the 512-dim face embedding is used here; local/global spatial maps are discarded.

Interface is compatible with FERClassifier so Trainer works without changes:
    - freeze_backbone() / unfreeze_backbone(from_layer=0)
    - trainable_params()
    - forward(x) → logits [B, num_classes]
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .swin_backbone import SwinTransformer

# Embedding dimension produced by the pretrained SwinFace backbone
_SWINFACE_EMBED_DIM = 512


class SwinFaceClassifier(nn.Module):
    """SwinFace backbone + thin classification head for emotion recognition.

    Parameters
    ----------
    num_classes : int
        Number of output emotion classes (default 7).
    weights_path : str or Path
        Path to the SwinFace pretrained checkpoint
        (``checkpoint_step_79999_gpu_0.pt``).  The checkpoint must contain
        a ``"state_dict_backbone"`` key with the SwinTransformer weights.
    dropout : float
        Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        weights_path: str | Path | None = None,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        if weights_path is None:
            raise ValueError(
                "weights_path is required for SwinFaceClassifier. "
                "Download the checkpoint from the SwinFace Google Drive and set "
                "'weights_path' in your experiment config."
            )

        # Backbone — num_classes=512 means the feature MLP projects to 512-dim
        self.backbone = SwinTransformer(num_classes=_SWINFACE_EMBED_DIM)

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(ckpt["state_dict_backbone"])

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(_SWINFACE_EMBED_DIM, num_classes),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, feat = self.backbone(x)   # discard local/global maps
        return self.head(feat)

    # ── Fine-tuning helpers (called by Trainer) ───────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze the entire backbone; leave the head trainable.

        Called by Trainer at epoch 1 during the warmup phase.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        """Unfreeze backbone stages from ``from_layer`` onward, plus head.

        Available stage indices: 0, 1, 2, 3  (SwinTransformer has 4 stages).
        Called by Trainer after the warmup phase with ``from_layer=0``
        (full unfreeze).

        Parameters
        ----------
        from_layer : int
            Index of the first backbone stage to unfreeze.  0 = full unfreeze.
        """
        for layer in list(self.backbone.layers)[from_layer:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        for p in self.backbone.feature.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
