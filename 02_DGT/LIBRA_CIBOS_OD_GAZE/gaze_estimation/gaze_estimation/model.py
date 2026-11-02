"""GazeViT: a Vision Transformer encoder over the face crop and a small regression head.

The encoder is any timm Vision Transformer, built for 112x112 input. Only the last
`n_trainable_blocks` transformer blocks (and the final norm) are trained; the rest stay
frozen. The head maps the pooled embedding to two values, the gaze point in the
normalised [-1, 1] target space.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class GazeViT(nn.Module):
    def __init__(
        self,
        name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        img_size: int = 112,
        n_trainable_blocks: int = 4,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            name, pretrained=pretrained, img_size=img_size, num_classes=0, **encoder_kwargs
        )
        embed_dim = self.encoder.num_features
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 2),
        )
        self.freeze(n_trainable_blocks)

    def freeze(self, n_trainable_blocks: int) -> None:
        """Freeze the encoder except its last `n_trainable_blocks` blocks and final norm."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        blocks = list(self.encoder.blocks)
        for block in blocks[len(blocks) - n_trainable_blocks:] if n_trainable_blocks else []:
            for p in block.parameters():
                p.requires_grad = True
        if n_trainable_blocks:
            for p in self.encoder.norm.parameters():
                p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, face_img: torch.Tensor) -> torch.Tensor:
        """face_img: (B, 3, H, W) in [-1, 1] -> gaze: (B, 2)."""
        return self.head(self.encoder(face_img))


def build_model(cfg, pretrained: bool | None = None, img_size: int = 112) -> GazeViT:
    """GazeViT from a `ModelConfig`; `pretrained` overrides the config (False when loading a checkpoint)."""
    return GazeViT(
        name=cfg.name,
        pretrained=cfg.pretrained if pretrained is None else pretrained,
        img_size=img_size,
        n_trainable_blocks=cfg.n_trainable_blocks,
        head_hidden_dim=cfg.head_hidden_dim,
        dropout=cfg.dropout,
        **cfg.encoder_kwargs,
    )
