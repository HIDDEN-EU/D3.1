"""Face detection (SCRFD) + 106-point facial landmarks (2d106det).

Points are returned in the pixel space of the input frame.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..contracts.landmarks import LandmarkResult
from .trt import TensorRTEngine

LANDMARK_SOURCE = "insightface_2d106det"

DET_MEAN, DET_STD = 127.5, 128.0

#: 2d106det takes raw 0..255 RGB.
LMK_MEAN, LMK_STD = 0.0, 1.0

#: Fixed by the trained weights: crop = 1.5x the box's longer side.
CROP_PAD = 1.5

#: Ordered finest first, matching output order.
FPN_STRIDES = (8, 16, 32)

#: Heads are bound by channel count: engine binding order need not match ONNX.
SCORE_CHANNELS, BBOX_CHANNELS = 1, 4


def scrfd_input_geometry(frame_h: int, frame_w: int, size: int) -> Tuple[int, int, float]:
    """(new_w, new_h, det_scale); SCRFD letterboxes at the top-left, not centred."""
    im_ratio = float(frame_h) / float(frame_w)
    if im_ratio > 1.0:
        new_h = size
        new_w = int(size / im_ratio)
    else:
        new_w = size
        new_h = int(size * im_ratio)
    return new_w, new_h, float(new_h) / float(frame_h)


def anchor_centers(height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
    """(H*W*A, 2) centres in input pixels, row-major with a location's anchors adjacent."""
    centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)
    return centers


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Anchor centres + (left, top, right, bottom) distances -> (x0, y0, x1, y1)."""
    x0 = points[:, 0] - distance[:, 0]
    y0 = points[:, 1] - distance[:, 1]
    x1 = points[:, 0] + distance[:, 2]
    y1 = points[:, 1] + distance[:, 3]
    return np.stack([x0, y0, x1, y1], axis=-1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Greedy NMS, kept indices highest score first; the `+ 1` pixel-inclusive areas are intentional."""
    if len(boxes) == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(x0[i], x0[rest])
        yy0 = np.maximum(y0[i], y0[rest])
        xx1 = np.minimum(x1[i], x1[rest])
        yy1 = np.minimum(y1[i], y1[rest])
        inter = np.maximum(0.0, xx1 - xx0 + 1) * np.maximum(0.0, yy1 - yy0 + 1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


def face_crop_matrix(bbox: np.ndarray, size: int) -> np.ndarray:
    """2x3 affine mapping the padded face box onto a size x size crop."""
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    width, height = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    scale = size / (max(width, height) * CROP_PAD)
    return np.array(
        [[scale, 0.0, size / 2.0 - cx * scale],
         [0.0, scale, size / 2.0 - cy * scale]],
        dtype=np.float32,
    )


def decode_landmarks(raw: np.ndarray, size: int) -> np.ndarray:
    """Engine output (1, 2N) in [-1, 1] -> (N, 2) crop pixels."""
    pred = np.asarray(raw, dtype=np.float32).reshape(-1, 2).copy()
    pred += 1.0
    pred *= size // 2
    return pred


def points_to_sensor(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Crop-space points back into frame pixels."""
    linear, offset = matrix[:, :2], matrix[:, 2]
    return (np.asarray(points, dtype=np.float32) - offset) @ np.linalg.inv(linear).T


def _blob(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    """BGR HWC uint8 -> NCHW float32 RGB, mean-subtracted and scaled."""
    rgb = np.asarray(image)[:, :, ::-1].astype(np.float32)
    chw = ((rgb - mean) / std).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(chw, dtype=np.float32)


class FaceLandmarkDetector:
    """SCRFD face engine then 2d106det landmark engine on a frame."""

    def __init__(
        self,
        face_engine_path: Path,
        landmark_engine_path: Path,
        det_threshold: float,
        nms_threshold: float,
    ) -> None:
        self.det_threshold = float(det_threshold)
        self.nms_threshold = float(nms_threshold)

        self._face = TensorRTEngine(face_engine_path)
        self._landmark = TensorRTEngine(landmark_engine_path)

        self.face_input_size = self._square_input(self._face, "face detector")
        self.landmark_input_size = self._square_input(self._landmark, "landmark model")

        self._heads = self._bind_face_outputs()
        self._centers: Dict[int, np.ndarray] = {
            stride: anchor_centers(self.face_input_size // stride,
                                   self.face_input_size // stride,
                                   stride, anchors)
            for stride, (_, _, anchors) in self._heads.items()
        }
        self.landmark_count = self._bind_landmark_output()

    @staticmethod
    def _square_input(engine: TensorRTEngine, role: str) -> int:
        if len(engine.input_names) != 1:
            raise ValueError(
                f"{engine.engine_path.name}: expected one input for the {role}, "
                f"got {engine.input_names}"
            )
        shape = engine.input_shapes[0]
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(
                f"{engine.engine_path.name}: the {role} wants an NCHW 3-channel "
                f"input, got {shape}"
            )
        if shape[2] <= 0 or shape[3] <= 0:
            raise ValueError(
                f"{engine.engine_path.name}: the {role} input is {shape} — the spatial "
                f"dimensions are still dynamic. The published ONNX declares them dynamic, "
                f"so the engine must be built with a fixed optimization profile "
                f"(the anchor grid is laid out from a known input size)."
            )
        if shape[2] != shape[3]:
            raise ValueError(
                f"{engine.engine_path.name}: the {role} input is {shape[2]}x{shape[3]}; "
                f"this code assumes a square input, as the published ONNX is built"
            )
        return int(shape[2])

    def _bind_face_outputs(self) -> Dict[int, Tuple[int, int, int]]:
        """{stride: (score index, bbox index, anchors per location)}, bound by shape."""
        by_channels: Dict[int, List[Tuple[int, int]]] = {SCORE_CHANNELS: [], BBOX_CHANNELS: []}
        for index, shape in enumerate(self._face.output_shapes):
            channels = int(shape[-1])
            if channels in by_channels:
                by_channels[channels].append((int(np.prod(shape)) // channels, index))

        scores = sorted(by_channels[SCORE_CHANNELS], reverse=True)
        boxes = sorted(by_channels[BBOX_CHANNELS], reverse=True)
        if len(scores) != len(FPN_STRIDES) or len(boxes) != len(FPN_STRIDES):
            raise ValueError(
                f"{self._face.engine_path.name}: expected {len(FPN_STRIDES)} score and "
                f"{len(FPN_STRIDES)} bbox outputs for strides {FPN_STRIDES}, got shapes "
                f"{self._face.output_shapes}. This engine is not the SCRFD model this "
                f"decode describes."
            )

        heads: Dict[int, Tuple[int, int, int]] = {}
        for stride, (score_len, score_i), (bbox_len, bbox_i) in zip(FPN_STRIDES, scores, boxes):
            cells = (self.face_input_size // stride) ** 2
            if score_len != bbox_len or score_len % cells:
                raise ValueError(
                    f"{self._face.engine_path.name}: stride {stride} emits {score_len} "
                    f"score and {bbox_len} bbox rows, which is not a whole number of "
                    f"anchors over a {self.face_input_size // stride}^2 grid. The engine's "
                    f"input size and this anchor layout disagree."
                )
            heads[stride] = (score_i, bbox_i, score_len // cells)
        return heads

    def _bind_landmark_output(self) -> int:
        shape = self._landmark.output_shapes[0]
        values = int(np.prod(shape))
        if len(self._landmark.output_names) != 1 or values % 2:
            raise ValueError(
                f"{self._landmark.engine_path.name}: expected one output of 2 values per "
                f"point, got {self._landmark.output_shapes}"
            )
        count = values // 2
        if count != 106:
            raise ValueError(
                f"{self._landmark.engine_path.name} emits {count} points, not the 106 of "
                f"2d106det (a 68-point/3D model emits 3309 values and needs a mean shape "
                f"this package does not carry). `attention/ear.py` reads fixed index "
                f"ranges, so every EAR from another point set would be meaningless "
                f"rather than wrong-looking."
            )
        return count

    def detect(self, frame: np.ndarray) -> LandmarkResult:
        """106 landmarks in frame pixels; no face -> `valid=False`, never an empty array."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return LandmarkResult(points=None, valid=False, source=LANDMARK_SOURCE)

        height, width = frame.shape[:2]
        image_size = (width, height)

        boxes, scores = self._detect_faces(frame)
        if not len(boxes):
            return LandmarkResult(points=None, valid=False, image_size=image_size,
                                  source=LANDMARK_SOURCE)

        best = int(np.argmax(scores))
        points = self._landmarks_for(frame, boxes[best])
        return LandmarkResult(
            points=points,
            valid=True,
            confidence=float(scores[best]),
            image_size=image_size,
            source=LANDMARK_SOURCE,
        )

    def detect_face_box(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """The highest-scoring face box in frame pixels, or None."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return None
        boxes, scores = self._detect_faces(frame)
        if not len(boxes):
            return None
        return boxes[int(np.argmax(scores))]

    def _detect_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Face boxes and scores in frame pixels, NMS applied."""
        import cv2  # heavy import: `import dgt.models` must not need OpenCV

        size = self.face_input_size
        height, width = frame.shape[:2]
        new_w, new_h, det_scale = scrfd_input_geometry(height, width, size)

        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(frame, (new_w, new_h),
                                            interpolation=cv2.INTER_LINEAR)
        outputs = self._face.infer(_blob(canvas, DET_MEAN, DET_STD))

        boxes_all: List[np.ndarray] = []
        scores_all: List[np.ndarray] = []
        for stride, (score_i, bbox_i, _) in self._heads.items():
            scores = outputs[score_i].reshape(-1)
            keep = scores >= self.det_threshold
            if not keep.any():
                continue
            # Distances are in stride units, not pixels.
            deltas = outputs[bbox_i].reshape(-1, 4)[keep] * stride
            boxes_all.append(distance2bbox(self._centers[stride][keep], deltas))
            scores_all.append(scores[keep])

        if not boxes_all:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Top-left letterbox: undo with scale only, no offset.
        boxes = np.concatenate(boxes_all) / det_scale
        scores = np.concatenate(scores_all)
        keep = nms(boxes, scores, self.nms_threshold)
        return boxes[keep], scores[keep]

    def _landmarks_for(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """The 106 points for one face box, in frame pixels."""
        import cv2

        size = self.landmark_input_size
        matrix = face_crop_matrix(bbox, size)
        crop = cv2.warpAffine(frame, matrix, (size, size), borderValue=0.0)
        raw = self._landmark.infer(_blob(crop, LMK_MEAN, LMK_STD))[0]
        return points_to_sensor(decode_landmarks(raw, size), matrix)

    def close(self) -> None:
        """Release both engines."""
        self._face.close()
        self._landmark.close()

    def describe(self) -> str:
        """One line for logs."""
        strides = ",".join(f"{s}x{a}" for s, (_, _, a) in sorted(self._heads.items()))
        return (f"{self._face.engine_path.name} in={self.face_input_size} strides={strides} "
                f"det={self.det_threshold} nms={self.nms_threshold} -> "
                f"{self._landmark.engine_path.name} in={self.landmark_input_size} "
                f"pts={self.landmark_count}")
