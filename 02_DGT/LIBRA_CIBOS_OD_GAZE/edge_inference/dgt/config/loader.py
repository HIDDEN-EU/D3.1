"""YAML -> frozen dataclasses: `thresholds.yaml` (signal and model tunables),
`models.yaml` (weights sources), `service.yaml` (dgt-daemon settings).

Required keys raise; any other missing key falls back to the default here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_THRESHOLDS = _CONFIG_DIR / "thresholds.yaml"
_DEFAULT_MODELS = _CONFIG_DIR / "models.yaml"
_DEFAULT_SERVICE = _CONFIG_DIR / "service.yaml"


@dataclass(frozen=True)
class PerclosParams:
    window_seconds: float
    min_fill_frac: float


@dataclass(frozen=True)
class RoadObjectParams:

    conf_threshold: float
    tile: bool


@dataclass(frozen=True)
class FaceLandmarkParams:

    det_threshold: float
    nms_threshold: float


@dataclass(frozen=True)
class GazeParams:
    screen_grid_cols: float
    screen_grid_rows: float
    roi_scale_x: float = 1.0
    roi_scale_y: float = 1.0
    roi_offset_x: float = 0.0
    roi_offset_y: float = 0.0
    roi_margin: float = 0.15


@dataclass(frozen=True)
class Thresholds:
    ear_threshold: float
    perclos: PerclosParams
    road_objects: RoadObjectParams
    face_landmarks: FaceLandmarkParams
    gaze: GazeParams


@dataclass(frozen=True)
class ServiceConfig:

    schema_version: str
    message_type: str
    time_reference: str
    road_frame_id: str


@dataclass(frozen=True)
class ModelSource:
    """Where an ONNX artifact lives: a local `path` or a download `url`. Exactly ONE is set.

    `sha256`, when given, is checked after download (and for a local file).
    """

    path: Optional[Path] = None
    url: Optional[str] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.url is None):
            raise ValueError(
                "ModelSource needs exactly one of path / url; "
                f"got path={self.path!r}, url={self.url!r}"
            )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSource":
        path = d.get("path")
        return cls(
            path=Path(path) if path else None,
            url=d.get("url") or None,
            sha256=d.get("sha256") or None,
        )

    @property
    def is_remote(self) -> bool:
        return self.url is not None

    def describe(self) -> str:
        """One-line provenance, for logs."""
        return str(self.path) if self.path is not None else str(self.url)


@dataclass(frozen=True)
class ModelConfig:
    """One model's entry in models.yaml: where it is, plus any runtime knobs."""

    source: ModelSource
    params: Dict[str, Any] = field(default_factory=dict)
    #: The model's inputs in order, for the on-device engine build.
    input_shapes: Tuple[Tuple[int, ...], ...] = ()


def load_thresholds(path: Optional[Path] = None) -> Thresholds:
    """Read thresholds.yaml into `Thresholds`. `ear_threshold` is required."""
    raw = yaml.safe_load((path or _DEFAULT_THRESHOLDS).read_text()) or {}
    perclos_raw = raw.get("perclos", {}) or {}
    road_raw = raw.get("road_objects", {}) or {}
    face_raw = raw.get("face_landmarks", {}) or {}
    gaze_raw = raw.get("gaze", {}) or {}
    return Thresholds(
        ear_threshold=float(raw["ear_threshold"]),
        perclos=PerclosParams(
            window_seconds=float(perclos_raw.get("window_seconds", 30.0)),
            min_fill_frac=float(perclos_raw.get("min_fill_frac", 1.0)),
        ),
        road_objects=RoadObjectParams(
            conf_threshold=float(road_raw.get("conf_threshold", 0.25)),
            tile=bool(road_raw.get("tile", True)),
        ),
        face_landmarks=FaceLandmarkParams(
            det_threshold=float(face_raw.get("det_threshold", 0.5)),
            nms_threshold=float(face_raw.get("nms_threshold", 0.4)),
        ),
        gaze=GazeParams(
            screen_grid_cols=float(gaze_raw.get("screen_grid_cols", 1)),
            screen_grid_rows=float(gaze_raw.get("screen_grid_rows", 1)),
            roi_scale_x=float(gaze_raw.get("roi_scale_x", 1.0)),
            roi_scale_y=float(gaze_raw.get("roi_scale_y", 1.0)),
            roi_offset_x=float(gaze_raw.get("roi_offset_x", 0.0)),
            roi_offset_y=float(gaze_raw.get("roi_offset_y", 0.0)),
            roi_margin=float(gaze_raw.get("roi_margin", 0.15)),
        ),
    )


def load_models_config(path: Optional[Path] = None) -> Dict[str, ModelConfig]:
    """Read models.yaml into `{model_name: ModelConfig}`. An entry whose source
    names no artifact is skipped."""
    raw = yaml.safe_load((path or _DEFAULT_MODELS).read_text()) or {}
    out: Dict[str, ModelConfig] = {}
    for name, entry in raw.items():
        entry = entry or {}
        source_raw = entry.get("source") or {}
        if not any(source_raw.get(k) for k in ("path", "url")):
            continue
        out[str(name)] = ModelConfig(
            source=ModelSource.from_dict(source_raw),
            params=dict(entry.get("params") or {}),
            input_shapes=tuple(tuple(int(d) for d in shape)
                               for shape in entry.get("input_shapes") or ()),
        )
    return out


def load_service(path: Optional[Path] = None) -> ServiceConfig:
    """Read service.yaml into `ServiceConfig`. The `message` fields are required."""
    raw = yaml.safe_load((path or _DEFAULT_SERVICE).read_text()) or {}
    message = raw["message"]
    return ServiceConfig(
        schema_version=str(message["schema_version"]),
        message_type=str(message["message_type"]),
        time_reference=str(message["time_reference"]),
        road_frame_id=str(raw.get("road_frame_id", "road_camera")),
    )
