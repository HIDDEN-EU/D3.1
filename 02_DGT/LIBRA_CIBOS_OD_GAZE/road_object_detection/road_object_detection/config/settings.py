"""
Experiment configuration for the object detection module.

Each training run is described by a single Experiment instance loaded from a
JSON file.  That file is the ground truth for the run and is saved to MLflow
as an artifact so every run is fully reproducible.

Usage:
    experiment = Experiment.from_json(Path("configs/experiments/tiled_v030_mz16_no_ov_notrain_gc_bgcap015_20ep_sgd.json"))
"""

from pathlib import Path

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    weights: str = Field(default="yolo11s.pt")
    imgsz: int = Field(default=640)


class DatasetConfig(BaseModel):
    data_yaml: Path = Field(default=Path("configs/data.yaml"))
    val_test_ratio: float = Field(default=0.3)
    split_seed: int = Field(default=42)


class HyperparameterConfig(BaseModel):
    epochs: int = Field(default=10)
    batch_size: int = Field(default=16)
    learning_rate: float = Field(default=0.01)  # -> Ultralytics lr0; honored only when optimizer != "auto"
    patience: int = Field(default=20)
    # Defaults below mirror Ultralytics' own defaults, so configs that omit them
    # behave exactly as before. optimizer="auto" makes Ultralytics pick the
    # optimizer/lr0/momentum itself and IGNORE learning_rate + momentum.
    optimizer: str = Field(default="auto")  # "auto" | "SGD" | "AdamW" | ...
    lrf: float = Field(default=0.01)         # final LR fraction (lr_end = lr0 * lrf)
    momentum: float = Field(default=0.937)
    weight_decay: float = Field(default=0.0005)
    warmup_epochs: float = Field(default=3.0)
    cos_lr: bool = Field(default=False)
    seed: int = Field(default=0)
    rect: bool = Field(default=False)  # rectangular training: letterbox to min stride-multiple instead of square imgsz
    compile: bool = Field(default=False)  # torch.compile kernel fusion: faster, accuracy-neutral
    amp: bool = Field(default=True)  # mixed-precision (fp16). Set False if training diverges to NaN (overflow)
    tf32: bool = Field(default=False)  # TF32 matmul on Ampere+ tensor cores: recovers most fp32 speed, keeps fp32 exponent range (no fp16-style overflow), accuracy-neutral. Recommended whenever amp=False.
    close_mosaic: int = Field(default=10)  # disable mosaic augmentation for the final N epochs (settle on real images)


class HardwareConfig(BaseModel):
    device: int = Field(default=0)
    workers: int = Field(default=8)


class MLflowConfig(BaseModel):
    experiment_name: str = Field(default="road-object-detection")
    tags: dict[str, str] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Post-training evaluation logged to MLflow after the run.

    `occlusion_recall` runs the accepted count-based recall view on the best checkpoint and logs
    found/missed/recall per occlusion stratum × {all, VRU} plus the found/missed webview
    galleries. On by default; set `occlusion_recall: false` to skip the extra pass
    (e.g. smoke runs). Auto-skips silently when the dataset has no occlusion.csv sidecar.
    """
    occlusion_recall: bool = Field(default=True)
    eval_on: str = Field(default="val")  # "val" (clean monitoring) | "test" (held-out)
    conf: list[float] = Field(default_factory=lambda: [0.25, 0.001])
    nms_iou: float = Field(default=0.7)
    gallery_n: int = Field(default=24)
    max_images: int | None = Field(default=None)  # cap eval slice (smoke only)


class ExperimentMeta(BaseModel):
    name: str
    description: str = Field(default="")


class Experiment(BaseModel):
    meta: ExperimentMeta
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    hyperparameters: HyperparameterConfig = HyperparameterConfig()
    hardware: HardwareConfig = HardwareConfig()

    mlflow: MLflowConfig = MLflowConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    # Ultralytics writes weights, plots, and logs to <project_dir>/<meta.name>/
    project_dir: Path = Field(default=Path("runs/train"))

    @classmethod
    def from_json(cls, path: Path) -> "Experiment":
        return cls.model_validate_json(path.read_text())

    def to_json(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))
