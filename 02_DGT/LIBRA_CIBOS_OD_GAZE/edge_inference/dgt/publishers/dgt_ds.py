"""Road detections -> one DGT_DS message.

Everything not produced yet (gaze area, emotion, attention, the verdicts) is
None, serialized as null.
"""
from __future__ import annotations

import time
import uuid
from typing import Callable, Optional, Tuple

from ..config.loader import ServiceConfig
from ..contracts.detections import DetectionResult
from ..contracts.driver_status import (
    AttendedObject,
    DgtMessage,
    DriverAttOD,
    DriverStatus,
    MessageMetadata,
)


def to_pixels(box: Tuple[float, float, float, float], width: int,
              height: int) -> Tuple[int, int, int, int]:
    """A normalized (x1, y1, x2, y2) box as integer pixels, clamped to the frame."""

    def px(value: float, size: int) -> int:
        return int(round(min(max(value, 0.0), 1.0) * size))

    x1, y1, x2, y2 = box
    return px(x1, width), px(y1, height), px(x2, width), px(y2, height)


def assemble_status(objects: DetectionResult, width: int, height: int,
                    capture_ts: float, road_frame_id: str) -> DriverStatus:
    """`driver_att_od` is None when the detector did not run (`valid=False`)."""
    att_od: Optional[DriverAttOD] = None
    if objects.valid:
        att_od = DriverAttOD(
            frame_id=road_frame_id,
            gaze_area_px=None,
            detections=[
                AttendedObject(class_id=int(d.class_id), class_name=d.label,
                               bbox_px=to_pixels(d.box, width, height),
                               confidence=float(d.score))
                for d in objects.detections
            ],
        )
    return DriverStatus(timestamp=capture_ts, driver_att_od=att_od)


class MessageEnvelope:
    """Wraps each `DriverStatus` in metadata. `sequence_number` counts from 0
    per process; `message_id` is a fresh UUID; `send_timestamp` is taken here."""

    def __init__(self, config: ServiceConfig, sender_id: str,
                 clock: Callable[[], float] = time.time) -> None:
        if not sender_id:
            raise ValueError("sender_id is empty: set DGT_SENDER_ID")
        self._config = config
        self._sender_id = sender_id
        self._clock = clock
        self._sequence = 0

    def wrap(self, status: DriverStatus) -> DgtMessage:
        metadata = MessageMetadata(
            schema_version=self._config.schema_version,
            message_type=self._config.message_type,
            message_id=str(uuid.uuid4()),
            sender_id=self._sender_id,
            sequence_number=self._sequence,
            send_timestamp=self._clock(),
            time_reference=self._config.time_reference,
        )
        self._sequence += 1
        return DgtMessage(metadata=metadata, driver_status=status)
