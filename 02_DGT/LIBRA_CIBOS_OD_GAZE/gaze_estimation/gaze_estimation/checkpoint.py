"""Save and load a trained model with everything inference needs: config and target bounds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .model import GazeViT, build_model


def save_checkpoint(path: Path, model: GazeViT, cfg: ExperimentConfig, bounds: np.ndarray,
                    epoch: int, val_loss: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.model_dump(mode="json"),
            "bounds": bounds.tolist(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu"):
    """-> (model in eval mode, ExperimentConfig, bounds)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ExperimentConfig.model_validate(ckpt["config"])
    model = build_model(cfg.model, pretrained=False, img_size=cfg.data.image_size)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, np.asarray(ckpt["bounds"], dtype=np.float64)
