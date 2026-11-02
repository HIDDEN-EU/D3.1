"""
Experiment configuration via Pydantic v2.

Experiments are defined as JSON files and loaded with:

    cfg = ExperimentConfig.from_json("configs/experiments/fer2013_baseline.json")

``ExperimentConfig.to_runtime()`` converts the schema into the concrete runtime
objects (model, loaders, TrainConfig, callbacks) consumed by ``Trainer``.

MLflow note
-----------
When MLflow is added, log the full config with:
    mlflow.log_params(cfg.model_dump(mode="json"))
or log the JSON file itself as an artifact:
    mlflow.log_artifact(config_path)
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetName(str, Enum):
    """Supported training datasets.

    Inherits from ``str`` so members compare equal to their string values and
    serialise transparently to/from JSON and experiment configs::

        DatasetName.FER2013 == "fer2013"  # True
    """
    FER2013   = "fer2013"
    KDEF      = "kdef"
    RAFDB     = "rafdb"
    FERPLUS   = "ferplus"
    AFFECTNET = "affectnet"


# ── Sub-configs ───────────────────────────────────────────────────────────────


class SplitConfig(BaseModel):
    """Controls how the manifest is partitioned into train / val / test.

    strategy = "source"  (default)
        Respects the ``source_split`` column written by the prepare scripts.
        - ``source_split == "train"`` rows form the train pool.
        - ``source_split == "test"``  rows become the test set unchanged.
        - ``source_split == "val"``   rows (KDEF only) become val directly;
          ``val_ratio`` is ignored for those datasets.
        - For datasets without an official val split (FER2013, RAF-DB) a
          stratified fraction of the train pool is carved out as val.

    strategy = "ratio"
        Ignores ``source_split`` entirely — pools **all** rows and re-splits
        from scratch using ``test_ratio`` then ``val_ratio``.  Use this when
        you want to merge the original train+test sets or define custom splits.
    """

    strategy: Literal["source", "ratio"] = "source"
    val_ratio: float = Field(0.10, ge=0, lt=1)
    test_ratio: float = Field(
        0.15,
        ge=0,
        lt=1,
        description="Fraction for test; only used with strategy='ratio'",
    )
    seed: int = 42
    stratified: bool = True


class AugmentationConfig(BaseModel):
    """Training-time augmentation parameters.

    All numeric fields default to ``None``, which means "use the per-dataset
    hardcoded default".  Set a field explicitly in the experiment JSON to
    override it — the value is then logged to MLflow as part of the config.
    """

    horizontal_flip: bool = True
    rotation_degrees: Optional[float] = None
    crop_scale_min: Optional[float] = None
    # FER2013-specific
    autocontrast_p: float = 0.2
    # RGB datasets (KDEF, RAF-DB)
    color_jitter_brightness: Optional[float] = None
    color_jitter_contrast: Optional[float] = None
    color_jitter_saturation: Optional[float] = None
    color_jitter_hue: float = 0.05
    random_grayscale_p: float = 0.05
    random_erasing_p: Optional[float] = None


class DataConfig(BaseModel):
    dataset: DatasetName
    root: Path
    batch_size: int = Field(64, gt=0)
    num_workers: int = Field(2, ge=0)
    use_weighted_sampler: bool = True

    split: SplitConfig = Field(default_factory=SplitConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # KDEF-specific filters — ignored for FER2013 / RAF-DB
    angles: Optional[list[Literal["FL", "HL", "S", "HR", "FR"]]] = None
    genders: Optional[list[Literal["F", "M"]]] = None
    sessions: Optional[list[Literal["A", "B"]]] = None

    @field_validator("root")
    @classmethod
    def root_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Dataset root does not exist: {v}")
        return v


class ModelConfig(BaseModel):
    backbone: Literal["efficientnet_b0", "efficientnet_b2", "resnet50", "resnet18", "swinface"] = (
        "efficientnet_b0"
    )
    num_classes: int = Field(7, gt=0)
    pretrained: bool = True
    in_channels: int = Field(3, ge=1, le=3)
    dropout: float = Field(0.4, ge=0.0, lt=1.0)
    # Path to a checkpoint to initialise the backbone from.
    # Required when backbone == "swinface"; optional for EfficientNet/ResNet
    # (use to load AffectNet-pretrained weights with strict=False).
    weights_path: Optional[Path] = None
    # Informational field logged to MLflow — does not change model behaviour.
    pretrained_source: Literal["imagenet", "affectnet", "custom", "none"] = "imagenet"

    @model_validator(mode="after")
    def swinface_requires_weights_path(self) -> "ModelConfig":
        if self.backbone == "swinface" and self.weights_path is None:
            raise ValueError(
                "weights_path is required when backbone='swinface'. "
                "Set it to the path of checkpoint_step_79999_gpu_0.pt."
            )
        return self


class TrainingConfig(BaseModel):
    epochs: int = Field(30, gt=0)
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    grad_clip: float = Field(1.0, gt=0)
    label_smoothing: float = Field(0.1, ge=0, lt=1)
    warmup_epochs: int = Field(
        3, ge=0, description="Frozen-backbone epochs before full unfreeze"
    )
    patience: int = Field(
        10, gt=0, description="Val-acc early-stop patience (built-in)"
    )

    # OneCycleLR scheduler
    pct_start: float = Field(0.2, gt=0, lt=1)
    div_factor: float = Field(10.0, gt=1)
    final_div_factor: float = Field(1e4, gt=1)

    # Hardware
    amp: bool = True
    device: str = Field("auto", description="cuda/mps/cpu or auto-detect")
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_name: str = "best_model.pt"


class CallbacksConfig(BaseModel):
    overfit_detector: bool = True
    overfit_patience: int = Field(3, ge=0, description="0 = disabled")
    early_stopping: bool = False
    es_monitor: Literal["val_loss", "val_acc"] = "val_loss"
    es_patience: int = Field(5, gt=0)
    es_min_delta: float = Field(1e-4, ge=0)

    @model_validator(mode="after")
    def overfit_patience_nonzero_when_enabled(self) -> "CallbacksConfig":
        if self.overfit_detector and self.overfit_patience == 0:
            raise ValueError(
                "overfit_patience must be > 0 when overfit_detector is True. "
                "Set overfit_detector=false to disable."
            )
        return self


# ── MLflow settings ───────────────────────────────────────────────────────────


class MLflowConfig(BaseModel):
    experiment_name: str = Field(
        "", description="MLflow experiment name; empty = ExperimentConfig.name"
    )
    tags: dict[str, str] = Field(
        default_factory=dict, description="Custom static tags (e.g. team, use_case)"
    )
    auto_tag_fields: list[str] = Field(
        default_factory=lambda: [
            "data.dataset",
            "model.backbone",
            "training.epochs",
            "training.lr",
        ],
        description="Dotted config paths auto-promoted as run tags; resolved from the flat config dump",
    )


# ── Top-level experiment ──────────────────────────────────────────────────────


class ExperimentConfig(BaseModel):
    name: str
    data: DataConfig
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    callbacks: CallbacksConfig = CallbacksConfig()
    mlflow: MLflowConfig = MLflowConfig()

    # ── I/O ───────────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(self, path: str | Path) -> None:
        """Serialise this config to a JSON file (useful for logging artifacts)."""
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    # ── Runtime factory ───────────────────────────────────────────────────────

    def to_runtime(self):
        """Build runtime objects from this config.

        Returns
        -------
        tuple of (model, loaders, callbacks_list)

        Pass ``cfg.training`` directly to ``Trainer`` along with the callbacks
        and the resolved checkpoint directory::

            model, loaders, callbacks = cfg.to_runtime()
            trainer = Trainer(
                model, loaders, cfg.training,
                callbacks=callbacks,
                checkpoint_dir=cfg.training.checkpoint_dir / cfg.name,
            )
        """
        from emotion_recognition.training import (
            EarlyStoppingCallback,
            OverfitDetectorCallback,
        )
        from emotion_recognition.data import build_loaders
        from emotion_recognition.models import FERClassifier, SwinFaceClassifier
        from emotion_recognition.data.transforms import swinface_transforms

        if self.model.backbone == "swinface":
            model = SwinFaceClassifier(
                num_classes=self.model.num_classes,
                weights_path=self.model.weights_path,
                dropout=self.model.dropout,
            )
            train_tf = swinface_transforms("train", self.data.dataset, self.data.augmentation)
            eval_tf  = swinface_transforms("test",  self.data.dataset, self.data.augmentation)
        else:
            model = FERClassifier(
                num_classes=self.model.num_classes,
                backbone=self.model.backbone,
                pretrained=self.model.pretrained,
                in_channels=self.model.in_channels,
                dropout=self.model.dropout,
                weights_path=self.model.weights_path,
            )
            train_tf = None
            eval_tf  = None

        loaders = build_loaders(
            dataset=self.data.dataset,
            root=self.data.root,
            batch_size=self.data.batch_size,
            num_workers=self.data.num_workers,
            use_weighted_sampler=self.data.use_weighted_sampler,
            split_cfg=self.data.split,
            angles=self.data.angles,
            genders=self.data.genders,
            sessions=self.data.sessions,
            augmentation=self.data.augmentation,
            train_transform=train_tf,
            eval_transform=eval_tf,
        )

        cb_cfg = self.callbacks
        callbacks = []
        if cb_cfg.overfit_detector and cb_cfg.overfit_patience > 0:
            callbacks.append(OverfitDetectorCallback(patience=cb_cfg.overfit_patience))
        if cb_cfg.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=cb_cfg.es_monitor,
                    patience=cb_cfg.es_patience,
                    min_delta=cb_cfg.es_min_delta,
                )
            )

        return model, loaders, callbacks
