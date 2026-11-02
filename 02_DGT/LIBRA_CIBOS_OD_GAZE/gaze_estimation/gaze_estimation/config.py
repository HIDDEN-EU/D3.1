"""Experiment configuration (YAML or JSON), validated with pydantic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "vit_small_patch16_224"
    pretrained: bool = True
    n_trainable_blocks: int = Field(4, ge=0)
    head_hidden_dim: int = Field(256, gt=0)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    #: Extra keyword arguments for `timm.create_model` (e.g. depth, drop_path_rate).
    encoder_kwargs: Dict[str, Any] = {}


class DataConfig(BaseModel):
    manifest: Path
    image_size: int = Field(112, gt=0)
    bounds_quantile: float = Field(0.1, ge=0.0, lt=0.5)
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str] = []

    @model_validator(mode="after")
    def subjects_are_disjoint(self) -> "DataConfig":
        """Subject-independent split: no subject may appear in two splits."""
        splits = {"train": self.train_subjects, "val": self.val_subjects,
                  "test": self.test_subjects}
        seen: dict[str, str] = {}
        for split, subjects in splits.items():
            for subject in subjects:
                if subject in seen:
                    raise ValueError(f"subject {subject!r} is in both {seen[subject]} and {split}")
                seen[subject] = split
        return self


class TrainConfig(BaseModel):
    epochs: int = Field(50, gt=0)
    batch_size: int = Field(64, gt=0)
    lr: float = Field(1e-4, gt=0)
    patience: int = Field(8, gt=0)
    min_delta: float = Field(1e-4, ge=0)
    num_workers: int = Field(4, ge=0)
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("runs/example")


class ExperimentConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    data: DataConfig
    train: TrainConfig = TrainConfig()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        text = path.read_text()
        raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        return cls.model_validate(raw)
