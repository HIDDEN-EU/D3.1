"""Gaze estimation: face crop -> gaze point.

The engine is the face-crop gaze model of ../gaze_estimation, exported to ONNX with one input
(`face_img`, 1x3x112x112) and one output (`gaze`, 1x2). The raw output is in the normalised
training space, not image coordinates; a result is valid only after calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ..contracts.gaze import GazeResult
from .trt import TensorRTEngine

#: Fixed by the trained weights; checked against the engine's input shape.
FACE_INPUT_SIZE = (112, 112)


def preprocess_face_crop(
    frame: np.ndarray,
    bbox: Optional[Sequence[int]],
    face_input_size: Tuple[int, int] = FACE_INPUT_SIZE,
) -> np.ndarray:
    """BGR frame + face box -> (3, H, W) float32 RGB in [-1, 1], as in training.

    The crop is the raw box (clipped to the frame), resized, then (x / 255 - 0.5) / 0.5.
    A missing or empty box gives a zero blob.
    """
    import cv2

    width, height = face_input_size
    if bbox is None:
        return np.zeros((3, height, width), np.float32)
    frame_h, frame_w = frame.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_w, x1), min(frame_h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((3, height, width), np.float32)
    crop = cv2.resize(frame[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    return ((crop / 255.0 - 0.5) / 0.5).transpose(2, 0, 1).copy()


def _is_degenerate(bbox: Optional[Sequence[int]]) -> bool:
    """True for a missing box or one that slices an empty crop."""
    if bbox is None:
        return True
    x_origin, y_origin, x_end, y_end = (int(v) for v in bbox)
    return x_end <= x_origin or y_end <= y_origin


class GazeEstimator:
    """Face-crop gaze engine; valid only once `calibration` maps the raw point onto the
    target plane, which `screen_grid` (cols, rows) then normalises to [0, 1]."""

    def __init__(
        self,
        gaze_engine_path: Path,
        calibration: Optional[object] = None,
        screen_grid: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self._engine = TensorRTEngine(gaze_engine_path)
        self.calibration = calibration
        self.screen_grid = (float(screen_grid[0]), float(screen_grid[1]))
        #: Last raw engine output: training-normalized, not a measurement.
        self.last_raw: Optional[Tuple[float, float]] = None

        self._bind_input()
        self._bind_output()

    def _bind_input(self) -> None:
        face_elems = 3 * FACE_INPUT_SIZE[0] * FACE_INPUT_SIZE[1]
        sizes = [int(np.prod(shape)) for shape in self._engine.input_shapes]
        if sizes != [face_elems]:
            raise ValueError(
                f"{self._engine.engine_path.name}: expected one input of {face_elems} "
                f"elements (3x{FACE_INPUT_SIZE[0]}x{FACE_INPUT_SIZE[1]} face crop); this "
                f"engine declares {self._engine.input_shapes}. It is not the gaze model "
                f"this class describes."
            )

    def _bind_output(self) -> None:
        sizes = [int(np.prod(shape)) for shape in self._engine.output_shapes]
        if len(sizes) != 1 or sizes[0] != 2:
            raise ValueError(
                f"{self._engine.engine_path.name}: expected one output of 2 values "
                f"(a gaze point), got shapes {self._engine.output_shapes}."
            )

    def estimate(self, frame: np.ndarray, bbox: Optional[Sequence[int]]) -> GazeResult:
        """Gaze point for one frame; every no-answer path is `valid=False` with x/y None."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            self.last_raw = None
            return GazeResult(x=None, y=None, confidence=None, valid=False)

        if _is_degenerate(bbox):
            self.last_raw = None
            return GazeResult(x=None, y=None, confidence=None, valid=False)

        face = preprocess_face_crop(frame, [int(v) for v in bbox])
        raw = np.asarray(self._engine.infer(face)[0], dtype=np.float32).reshape(-1)

        self.last_raw = (float(raw[0]), float(raw[1]))

        mapped = None
        if self.calibration is not None and getattr(self.calibration, "available", False):
            mapped = self.calibration.apply(self.last_raw[0], self.last_raw[1])
        if mapped is None:
            return GazeResult(x=None, y=None, confidence=None, valid=False)

        cols, rows = self.screen_grid
        return GazeResult(
            x=float(mapped[0]) / cols,
            y=float(mapped[1]) / rows,
            # The model emits no confidence.
            confidence=None,
            valid=True,
        )

    def close(self) -> None:
        self._engine.close()

    def describe(self) -> str:
        state = "calibrated" if (
            self.calibration is not None
            and getattr(self.calibration, "available", False)
        ) else "UNCALIBRATED (every result valid=False)"
        return (f"{self._engine.engine_path.name} face{FACE_INPUT_SIZE} -> grid "
                f"{self.screen_grid[0]:g}x{self.screen_grid[1]:g} — {state}")

    def __enter__(self) -> "GazeEstimator":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
