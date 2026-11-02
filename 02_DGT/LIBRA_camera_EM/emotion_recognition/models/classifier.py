"""
FERClassifier
-------------
Thin wrapper around a torchvision backbone with a task-specific head.

Supported backbones
-------------------
- ``efficientnet_b0``  (default, ~5 M params, good speed/accuracy trade-off)
- ``efficientnet_b2``  (~9 M params, higher accuracy)
- ``resnet50``         (~25 M params, strong baseline)
- ``resnet18``         (~11 M params, lightweight)

Channel handling
----------------
Pretrained backbones expect 3-channel RGB input.  FER2013 images are
grayscale; pass ``in_channels=1`` to insert a learned 1→3 projection
instead of converting images in the transform pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

_BACKBONE_REGISTRY: dict[str, tuple] = {
    "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, 1280),
    "efficientnet_b2": (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, 1408),
    "resnet50":        (models.resnet50,        ResNet50_Weights.DEFAULT,        2048),
    "resnet18":        (models.resnet18,        ResNet18_Weights.DEFAULT,        512),
}


class FERClassifier(nn.Module):
    """Facial Emotion Recognition classifier.

    Parameters
    ----------
    num_classes : int
        Number of output emotion categories (default 7).
    backbone : str
        One of the keys in ``_BACKBONE_REGISTRY``.
    pretrained : bool
        Load ImageNet weights for the backbone (default ``True``).
    in_channels : int
        ``1`` for grayscale input (inserts a learned projection to 3ch),
        ``3`` for standard RGB (default).
    dropout : float
        Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        backbone: Literal[
            "efficientnet_b0", "efficientnet_b2", "resnet50", "resnet18"
        ] = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.4,
        weights_path: Optional[Path] = None,
    ) -> None:
        super().__init__()

        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone {backbone!r}. "
                f"Choose from {list(_BACKBONE_REGISTRY)}"
            )

        factory, weights, feat_dim = _BACKBONE_REGISTRY[backbone]
        self.backbone_name = backbone

        # ── Channel adapter (grayscale → 3-channel) ───────────────────────────
        if in_channels == 1:
            self.channel_adapter: nn.Module = nn.Conv2d(
                1, 3, kernel_size=3, padding=1, bias=False
            )
            nn.init.kaiming_normal_(self.channel_adapter.weight, mode="fan_out")
        else:
            self.channel_adapter = nn.Identity()

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = factory(weights=weights if pretrained else None)
        self._replace_head(backbone, feat_dim, num_classes, dropout)

        # ── Optional custom weights (e.g. AffectNet pretrained) ───────────────
        # Loaded after head replacement so head weights are always re-initialised.
        # strict=False allows source checkpoints with a different number of classes.
        if weights_path is not None:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(
                f"Loaded custom weights from {weights_path}  "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    # ── Head replacement ──────────────────────────────────────────────────────

    def _replace_head(
        self,
        backbone: str,
        feat_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )
        if backbone.startswith("efficientnet"):
            self.backbone.classifier = head
        else:
            # ResNet — replace fc
            self.backbone.fc = head

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_adapter(x)
        return self.backbone(x)

    # ── Fine-tuning helpers ───────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters; only the head trains."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze the head
        head_attr = "classifier" if self.backbone_name.startswith("efficientnet") else "fc"
        for p in getattr(self.backbone, head_attr).parameters():
            p.requires_grad = True

    def unfreeze_backbone(self, from_layer: int = 0) -> None:
        """Unfreeze the full backbone (or from a given child module index)."""
        children = list(self.backbone.children())
        for child in children[from_layer:]:
            for p in child.parameters():
                p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total  = sum(p.numel() for p in self.parameters())
        train  = self.trainable_params()
        return (
            f"FERClassifier(backbone={self.backbone_name!r}, "
            f"params={total/1e6:.1f}M, trainable={train/1e6:.1f}M)"
        )
