"""Road-object detector: TensorRT engine -> `DetectionResult`, boxes normalized to [0, 1].

Tiling is part of the model: the weights were fine-tuned on 640x360 crops of 1280x720 frames.
How the frame is tiled, letterboxed, decoded and NMS'd is NOT defined here — it is
`road_object_detection.inference` (../road_object_detection), the one
definition the model's own evaluation runs through. This module only hands it the
engine and turns its pixel boxes into the contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..contracts.detections import (
    BDD100K_CLASSES,
    BDD100K_TAXONOMY,
    Detection,
    DetectionResult,
)
from .trt import TensorRTEngine

Region = Tuple[int, int, int, int]


class RoadObjectDetector:
    """Runs the road-object engine over a road frame; `tile=False` is off-distribution for these weights."""

    def __init__(
        self,
        engine_path: Path,
        conf_threshold: float,
        tile: bool = True,
        class_names: Sequence[str] = BDD100K_CLASSES,
        taxonomy: str = BDD100K_TAXONOMY,
    ) -> None:
        self.engine_path = Path(engine_path)
        self.conf_threshold = float(conf_threshold)
        self.tile = bool(tile)
        self.class_names = tuple(class_names)
        self.taxonomy = taxonomy

        self._engine = TensorRTEngine(self.engine_path)
        self._in_shape = self._engine.input_shapes[0]
        self._out_shape = self._engine.output_shapes[0]

        expected = 4 + len(self.class_names)
        if self._out_shape[1] != expected:
            raise ValueError(
                f"{self.engine_path.name} emits {self._out_shape[1]} channels but "
                f"{len(self.class_names)} class names are configured (expected "
                f"{expected} = 4 box + {len(self.class_names)} classes). This engine "
                f"is not the model these labels describe; every box would be "
                f"mislabelled rather than rejected."
            )

    @property
    def input_size(self) -> int:
        """Square input edge of the engine."""
        return int(self._in_shape[-1])

    @property
    def tile_size(self) -> Tuple[int, int]:
        """(width, height) of the crop the weights were fine-tuned on."""
        from road_object_detection.inference import TILE_H, TILE_W  # lazy: see module doc

        return int(TILE_W), int(TILE_H)

    def regions_for(self, width: int, height: int) -> List[Region]:
        """Regions `detect` would infer for a frame of this size."""
        if not self.tile:
            return [(0, 0, int(width), int(height))]
        from road_object_detection.inference import grid_regions  # lazy: see module doc

        return grid_regions(int(width), int(height))

    def detect(self, frame: np.ndarray,
               regions: Optional[Sequence[Region]] = None) -> DetectionResult:
        """Detections normalized to the full frame; empty `regions` or a bad frame returns `valid=False`."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return DetectionResult(detections=[], taxonomy=self.taxonomy, valid=False)

        height, width = frame.shape[:2]
        if regions is None:
            regions = self.regions_for(width, height)
        elif not len(regions):
            return DetectionResult(detections=[], taxonomy=self.taxonomy, valid=False)

        from road_object_detection.inference import infer_tiled

        # Ran and found nothing is valid and empty; an engine failure raises.
        found = infer_tiled(self._infer, frame, conf=self.conf_threshold,
                            regions=regions, imgsz=self.input_size)
        detections = [
            Detection(
                label=self._label(int(class_id)),
                class_id=int(class_id),
                score=float(score),
                box=(
                    float(np.clip(box[0] / width, 0.0, 1.0)),
                    float(np.clip(box[1] / height, 0.0, 1.0)),
                    float(np.clip(box[2] / width, 0.0, 1.0)),
                    float(np.clip(box[3] / height, 0.0, 1.0)),
                ),
            )
            for box, score, class_id in zip(found.boxes, found.scores, found.class_ids)
        ]
        return DetectionResult(detections=detections, taxonomy=self.taxonomy, valid=True)

    def _label(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _infer(self, blob: np.ndarray) -> np.ndarray:
        return self._engine.infer(blob)[0]

    def close(self) -> None:
        self._engine.close()

    def describe(self, frame_shape: Optional[Tuple[int, ...]] = None) -> str:
        """One line for logs; with a frame shape, reports the actual grid (below ~960x540 tiling is a no-op)."""
        from road_object_detection import inference

        tiled = f"tiled={self.tile}"
        if self.tile and frame_shape is not None:
            height, width = frame_shape[:2]
            regions = inference.grid_regions(width, height)
            cols = len({r[0] for r in regions})
            rows = len({r[1] for r in regions})
            tiled = f"tiled={self.tile} grid={cols}x{rows}"
            if len(regions) == 1:
                tiled += (f" (NO-OP: {width}x{height} rounds to one tile; "
                          f"tiling needs ~{inference.TILE_W + inference.TILE_W // 2}x"
                          f"{inference.TILE_H + inference.TILE_H // 2} or larger)")
        return (f"{self.engine_path.name} in={self._in_shape} out={self._out_shape} "
                f"conf={self.conf_threshold} tile_iou={inference.TILE_IOU} "
                f"merge_iou={inference.MERGE_IOU} {tiled} nc={len(self.class_names)} "
                f"recipe=road-object-detection {_recipe_version()}")


def _recipe_version() -> str:
    """Installed road-object-detection version: the tiling recipe this process runs."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("road-object-detection")
    except PackageNotFoundError:
        return "unknown"
