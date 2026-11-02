#!/usr/bin/env python3
"""Drive `RoadGazeTestPipeline` over a road source and a driver source at once.

    python scripts/testroadgaze.py \\
        --road-engine engines/road_objects_640x640.engine \\
        --face-engine engines/det_500m_fp16.engine \\
        --landmark-engine engines/2d106det_fp16.engine \\
        --gaze-engine engines/gaze_face_fp16.engine \\
        --road-source media/clip_1.mp4 \\
        --driver-source 0 \\
        --frames 600 --debug

The two sources are read back to back; the reported skew is between those
reads, not true camera sync. The road clip loops if shorter than `--frames`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgt.config.loader import load_thresholds  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from typing import Optional, Protocol, Sequence, Tuple  # noqa: E402
from dgt.contracts.common import FrameMeta  # noqa: E402
from dgt.contracts.detections import DetectionResult  # noqa: E402
from dgt.contracts.gaze import GazeResult  # noqa: E402
from pipeline_base import Pipeline, PipelineResult  # noqa: E402


class _RoadDetector(Protocol):
    def detect(self, frame: np.ndarray) -> DetectionResult: ...


class _FaceBoxDetector(Protocol):
    def detect_face_box(self, frame: np.ndarray) -> Optional[np.ndarray]: ...


class _GazeEstimator(Protocol):
    last_raw: Optional[Tuple[float, float]]

    def estimate(self, frame: np.ndarray,
                 bbox: Optional[Sequence[int]]) -> GazeResult: ...


@dataclass(frozen=True)
class RoadGazeFrameResult(PipelineResult):
    """Paired road and gaze contracts sharing one envelope; never merged."""

    road: DetectionResult
    gaze: GazeResult
    raw: Optional[Tuple[float, float]]
    face_box: Optional[Tuple[int, int, int, int]]
    road_ms: float
    driver_ms: float
    frame_skew_ms: Optional[float] = None

    @property
    def inference_ms(self) -> float:
        return self.road_ms + self.driver_ms


class RoadGazeTestPipeline(Pipeline[RoadGazeFrameResult]):
    """Road-object detection and gaze over a paired road/driver frame."""

    name = "testroadgaze"
    REQUIRED_FRAMES = ("road", "driver")

    def __init__(
        self,
        road_detector: _RoadDetector,
        face_detector: _FaceBoxDetector,
        gaze_estimator: _GazeEstimator,
        source_rate_hz: float = 30.0,
    ) -> None:
        super().__init__(source_rate_hz=source_rate_hz)
        self.road_detector = road_detector
        self.face_detector = face_detector
        self.gaze_estimator = gaze_estimator
        self.frames = 0
        self.road_valid = 0
        self.faces = 0
        self.no_face = 0
        self.gaze_valid = 0
        self._road_ms = 0.0
        self._driver_ms = 0.0
        # Set by tick_pair(), consumed by the next _process().
        self._pending_skew_ms: Optional[float] = None

    def tick_pair(
        self,
        *,
        road: np.ndarray,
        driver: np.ndarray,
        road_timestamp: float,
        driver_timestamp: float,
    ) -> RoadGazeFrameResult:
        """`tick()` for two separately captured frames; stamps the road timestamp."""
        self._pending_skew_ms = abs(road_timestamp - driver_timestamp) * 1000.0
        return self.tick(road=road, driver=driver, timestamp=road_timestamp)

    def _process(self, sync: FrameMeta, *, road: np.ndarray,
                 driver: np.ndarray) -> RoadGazeFrameResult:
        skew_ms, self._pending_skew_ms = self._pending_skew_ms, None

        started = time.perf_counter()
        road_result = self.road_detector.detect(road)
        road_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        box = self.face_detector.detect_face_box(driver)
        gaze_result = self.gaze_estimator.estimate(
            driver, None if box is None else [int(v) for v in box]
        )
        driver_ms = (time.perf_counter() - started) * 1000.0

        self.frames += 1
        self._road_ms += road_ms
        self._driver_ms += driver_ms
        if road_result.valid:
            self.road_valid += 1
        if box is None:
            self.no_face += 1
        else:
            self.faces += 1
        if gaze_result.valid:
            self.gaze_valid += 1

        raw = getattr(self.gaze_estimator, "last_raw", None)

        return RoadGazeFrameResult(
            sync=sync,
            road=replace(road_result, sync=sync),
            gaze=replace(gaze_result, sync=sync),
            raw=None if raw is None else (float(raw[0]), float(raw[1])),
            face_box=None if box is None else tuple(int(v) for v in box),
            road_ms=road_ms,
            driver_ms=driver_ms,
            frame_skew_ms=skew_ms,
        )

    def reset(self) -> None:
        self.frames = self.road_valid = 0
        self.faces = self.no_face = self.gaze_valid = 0
        self._road_ms = self._driver_ms = 0.0
        self._pending_skew_ms = None

    def close(self) -> None:
        for model in (self.gaze_estimator, self.face_detector, self.road_detector):
            closer = getattr(model, "close", None)
            if callable(closer):
                closer()

    @property
    def mean_road_ms(self) -> Optional[float]:
        return None if self.frames == 0 else self._road_ms / self.frames

    @property
    def mean_driver_ms(self) -> Optional[float]:
        return None if self.frames == 0 else self._driver_ms / self.frames

    @property
    def mean_inference_ms(self) -> Optional[float]:
        if self.frames == 0:
            return None
        return (self._road_ms + self._driver_ms) / self.frames

    @property
    def face_rate(self) -> Optional[float]:
        return None if self.frames == 0 else self.faces / self.frames

    def summary(self) -> str:
        road = self.mean_road_ms
        driver = self.mean_driver_ms
        both = self.mean_inference_ms
        return (
            f"{self.name}: {self.frames} frames, road valid {self.road_valid}, "
            f"faces {self.faces}/{self.frames}, gaze valid {self.gaze_valid}, "
            f"road {'n/a' if road is None else f'{road:.1f}'} ms + "
            f"driver {'n/a' if driver is None else f'{driver:.1f}'} ms = "
            f"{'n/a' if both is None else f'{both:.1f}'} ms mean"
        )


from debug_screen import add_debug_args, open_debug_screen  # noqa: E402
from dgt.sources import open_source  # noqa: E402


PANE_HEIGHT = 480
GAZE_PANEL_WIDTH = 360
STATUS_HEIGHT = 38
TRAIL_LENGTH = 32


def _fit(frame: np.ndarray, height: int) -> np.ndarray:
    """Scale a frame to `height`, keeping its aspect ratio."""
    import cv2

    h, w = frame.shape[:2]
    return cv2.resize(frame, (max(int(round(w * height / h)), 1), height))


def _draw_gaze_panel(height: int, width: int, result: RoadGazeFrameResult,
                     trail) -> np.ndarray:
    """Gaze dot in its own panel: uncalibrated raw values are not road positions."""
    import cv2

    panel = np.full((height, width, 3), 24, np.uint8)
    margin = 46
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin

    calibrated = result.gaze.valid
    accent = (0, 255, 0) if calibrated else (0, 165, 255)

    cv2.rectangle(panel, (x0, y0), (x1, y1), (70, 70, 70), 1)
    for fraction in (0.25, 0.5, 0.75):
        gx = int(x0 + (x1 - x0) * fraction)
        gy = int(y0 + (y1 - y0) * fraction)
        shade = (55, 55, 55) if fraction != 0.5 else (85, 85, 85)
        cv2.line(panel, (gx, y0), (gx, y1), shade, 1)
        cv2.line(panel, (x0, gy), (x1, gy), shade, 1)

    def to_panel(point):
        px = x0 + (x1 - x0) * min(max(point[0], 0.0), 1.0)
        py = y0 + (y1 - y0) * min(max(point[1], 0.0), 1.0)
        return int(px), int(py)

    for index, past in enumerate(trail):
        weight = (index + 1) / max(len(trail), 1)
        colour = tuple(int(channel * weight * 0.55) for channel in accent)
        cv2.circle(panel, to_panel(past), 2, colour, -1)

    point = ((result.gaze.x, result.gaze.y) if calibrated else result.raw)
    if point is None:
        cv2.putText(panel, "no face", (x0 + 8, (y0 + y1) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 2, cv2.LINE_AA)
    else:
        centre = to_panel(point)
        cv2.circle(panel, centre, 9, accent, 2)
        cv2.circle(panel, centre, 3, accent, -1)
        outside = not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0)
        if outside:
            cv2.putText(panel, "OFF-RANGE", (x0 + 8, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"x {point[0]:+.3f}  y {point[1]:+.3f}",
                    (x0 - 34, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)

    title = "GAZE (calibrated)" if calibrated else "GAZE - RAW, uncalibrated"
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                accent, 2, cv2.LINE_AA)
    if not calibrated:
        cv2.putText(panel, "not a screen position", (12, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
    return panel


def annotate(road_frame: np.ndarray, driver_frame: np.ndarray,
             result: RoadGazeFrameResult, trail=()) -> np.ndarray:
    """One canvas: road, driver, gaze panel and a status bar."""
    import cv2

    road = _fit(road_frame, PANE_HEIGHT)
    road_h, road_w = road.shape[:2]
    for detection in result.road.detections:
        # Boxes are normalized [0, 1].
        bx1, by1, bx2, by2 = detection.box
        p0 = (int(bx1 * road_w), int(by1 * road_h))
        p1 = (int(bx2 * road_w), int(by2 * road_h))
        cv2.rectangle(road, p0, p1, (0, 200, 255), 2)
        cv2.putText(road, f"{detection.label} {detection.score:.2f}",
                    (p0[0], max(p0[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(road, f"ROAD  {len(result.road.detections)} boxes  "
                      f"{result.road_ms:.0f} ms", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    driver = _fit(driver_frame, PANE_HEIGHT)
    if result.face_box is not None:
        # Face box is in driver-frame pixels.
        scale = PANE_HEIGHT / driver_frame.shape[0]
        fx1, fy1, fx2, fy2 = (int(v * scale) for v in result.face_box)
        cv2.rectangle(driver, (fx1, fy1), (fx2, fy2), (0, 200, 0), 2)
    cv2.putText(driver, f"DRIVER  {'face' if result.face_box else 'NO FACE'}  "
                        f"{result.driver_ms:.0f} ms", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    panel = _draw_gaze_panel(PANE_HEIGHT, GAZE_PANEL_WIDTH, result, trail)

    width = road.shape[1] + driver.shape[1] + panel.shape[1]
    canvas = np.zeros((PANE_HEIGHT + STATUS_HEIGHT, width, 3), np.uint8)
    offset = 0
    for pane in (road, driver, panel):
        canvas[:PANE_HEIGHT, offset:offset + pane.shape[1]] = pane
        offset += pane.shape[1]
        if offset < width:
            cv2.line(canvas, (offset - 1, 0), (offset - 1, PANE_HEIGHT), (90, 90, 90), 1)

    skew = ("skew n/a" if result.frame_skew_ms is None
            else f"read skew {result.frame_skew_ms:.1f} ms")
    fps = 1000.0 / result.inference_ms if result.inference_ms > 0 else 0.0
    cv2.putText(canvas, f"frame {result.sync.frame_id}   {skew}   "
                        f"road {result.road_ms:.0f} + driver {result.driver_ms:.0f} "
                        f"= {result.inference_ms:.0f} ms   {fps:.1f} fps",
                (12, PANE_HEIGHT + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--road-engine", type=Path, required=True)
    ap.add_argument("--face-engine", type=Path, required=True)
    ap.add_argument("--landmark-engine", type=Path, required=True)
    ap.add_argument("--gaze-engine", type=Path, required=True)
    ap.add_argument("--road-source", required=True,
                    help="road video file, camera index, or GStreamer pipeline")
    ap.add_argument("--driver-source", required=True,
                    help="driver video file, camera index, or GStreamer pipeline")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--rate", type=float, default=30.0,
                    help="rate the envelope claims for BOTH sources")
    ap.add_argument("--no-loop-road", action="store_true",
                    help="stop when the road source ends instead of looping it")
    add_debug_args(ap)
    ap.add_argument("--save-dir", type=Path, default=Path("/tmp/roadgaze"))
    ap.add_argument("--save-every", type=int, default=30,
                    help="write every Nth frame as a JPEG; 0 = none")
    args = ap.parse_args()

    import cv2

    from dgt.models.gaze import GazeEstimator
    from dgt.models.landmarks import FaceLandmarkDetector
    from dgt.models.road_objects import RoadObjectDetector

    thresholds = load_thresholds()
    try:
        road_detector = RoadObjectDetector(
            engine_path=args.road_engine,
            conf_threshold=thresholds.road_objects.conf_threshold,
            tile=thresholds.road_objects.tile,
        )
        face_detector = FaceLandmarkDetector(
            face_engine_path=args.face_engine,
            landmark_engine_path=args.landmark_engine,
            det_threshold=thresholds.face_landmarks.det_threshold,
            nms_threshold=thresholds.face_landmarks.nms_threshold,
        )
        gaze = GazeEstimator(
            args.gaze_engine,
            screen_grid=(thresholds.gaze.screen_grid_cols,
                         thresholds.gaze.screen_grid_rows),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[roadgaze] {exc}", file=sys.stderr)
        return 1

    print(f"[roadgaze] road: {road_detector.describe()}")
    print(f"[roadgaze] driver: {face_detector.describe()}")
    print(f"[roadgaze] {gaze.describe()}")

    try:
        road_capture = open_source(args.road_source)
        driver_capture = open_source(args.driver_source)
    except FileNotFoundError as exc:
        print(f"[roadgaze] {exc}", file=sys.stderr)
        return 1
    for name, capture, source in (("road", road_capture, args.road_source),
                                  ("driver", driver_capture, args.driver_source)):
        if not capture.isOpened():
            print(f"[roadgaze] cannot open {name} source: {source}", file=sys.stderr)
            return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    screen = open_debug_screen(args, "roadgaze")

    skews: List[float] = []
    trail = deque(maxlen=TRAIL_LENGTH)
    road_loops = 0
    stopped: Optional[str] = None

    with RoadGazeTestPipeline(road_detector, face_detector, gaze,
                              source_rate_hz=args.rate) as pipeline:
        try:
            while pipeline.frames < args.frames:
                ok_road, road_frame = road_capture.read()
                if not ok_road:
                    if args.no_loop_road:
                        stopped = "road source ended"
                        break
                    road_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_road, road_frame = road_capture.read()
                    road_loops += 1
                    if not ok_road:
                        stopped = "road source unreadable after rewind"
                        break
                road_read_at = time.monotonic()

                ok_driver, driver_frame = driver_capture.read()
                if not ok_driver:
                    stopped = "driver source ended"
                    break
                driver_read_at = time.monotonic()

                result = pipeline.tick_pair(
                    road=road_frame, driver=driver_frame,
                    road_timestamp=road_read_at, driver_timestamp=driver_read_at,
                )
                if result.frame_skew_ms is not None:
                    skews.append(result.frame_skew_ms)

                point = ((result.gaze.x, result.gaze.y) if result.gaze.valid
                         else result.raw)
                if point is not None:
                    trail.append(point)

                saving = args.save_every > 0 and result.sync.frame_id % args.save_every == 0
                if saving or screen is not None:
                    shown = annotate(road_frame, driver_frame, result, trail)
                    if saving:
                        cv2.imwrite(
                            str(args.save_dir / f"{result.sync.frame_id:05d}.jpg"), shown)
                    if screen is not None:
                        ok_jpeg, buf = cv2.imencode(".jpg", shown,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok_jpeg:
                            screen.publish(buf.tobytes())

                if pipeline.frames % 60 == 0:
                    print(f"[roadgaze] {pipeline.frames} frames, "
                          f"road {result.road_ms:.1f} ms + "
                          f"driver {result.driver_ms:.1f} ms, "
                          f"{len(result.road.detections)} boxes, "
                          f"skew {result.frame_skew_ms:.1f} ms")
        finally:
            road_capture.release()
            driver_capture.release()
            if screen is not None:
                screen.close()
        frames = pipeline.frames

        if frames == 0:
            print("[roadgaze] no frames decoded — check both sources", file=sys.stderr)
            return 1

        if stopped:
            print(f"[roadgaze] stopped early: {stopped}")
        print(f"\n[roadgaze] {pipeline.summary()}")
        both = pipeline.mean_inference_ms
        print(f"[roadgaze] combined {both:.1f} ms -> {1000.0 / both:.1f} fps ceiling")
        if road_loops:
            print(f"[roadgaze] road clip looped {road_loops}x — the scene is not "
                  f"continuous across a wrap")
        if skews:
            skew_arr = np.array(skews)
            print(f"[roadgaze] read skew: mean {skew_arr.mean():.1f} ms  "
                  f"p95 {np.percentile(skew_arr, 95):.1f} ms  max {skew_arr.max():.1f} ms")
            print("[roadgaze] NOTE: that is the gap between the two read() calls in "
                  "THIS harness, not evidence about a two-camera unit.")

        summary = {
            "road_source": args.road_source,
            "driver_source": args.driver_source,
            "frames": frames,
            "road_valid": pipeline.road_valid,
            "frames_with_a_face": pipeline.faces,
            "face_rate": round(pipeline.face_rate, 3),
            "gaze_valid": pipeline.gaze_valid,
            "road_loops": road_loops,
            "stopped_early": stopped,
            "latency_ms": {
                "road_mean": round(pipeline.mean_road_ms, 2),
                "driver_mean": round(pipeline.mean_driver_ms, 2),
                "combined_mean": round(both, 2),
                "fps_ceiling": round(1000.0 / both, 1),
            },
            "read_skew_ms": None if not skews else {
                "mean": round(float(np.mean(skews)), 2),
                "p95": round(float(np.percentile(skews, 95)), 2),
                "max": round(float(np.max(skews)), 2),
            },
        }
        (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if args.save_every > 0:
            print(f"[roadgaze] frames written to {args.save_dir}/")
        print(f"[roadgaze] summary -> {args.save_dir / 'summary.json'}")

        if pipeline.gaze_valid == 0:
            print("[roadgaze] NOTE: no valid gaze. Without a calibration matrix that "
                  "is the designed behaviour — see dgt/models/gaze.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
