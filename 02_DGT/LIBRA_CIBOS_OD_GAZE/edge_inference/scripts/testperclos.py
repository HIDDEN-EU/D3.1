#!/usr/bin/env python3
"""Drive `PerclosTestPipeline` over a driver video and save annotated frames.

    dgt-build-engines --engine-dir engines
    python scripts/testperclos.py \\
        --face-engine engines/det_500m.engine \\
        --landmark-engine engines/2d106det.engine \\
        --source media/driver.mp4 --frames 900 --save-dir /tmp/perclos
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgt.attention.ear import EYE1_IDX, EYE2_IDX  # noqa: E402
from dgt.config.loader import load_thresholds  # noqa: E402
from debug_screen import add_debug_args, open_debug_screen  # noqa: E402
from dgt.sources import open_source  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from typing import Optional, Protocol  # noqa: E402
from dgt.attention.ear import ear_from_landmarks  # noqa: E402
from dgt.attention.perclos import Perclos  # noqa: E402
from dgt.contracts.common import FrameMeta  # noqa: E402
from dgt.contracts.landmarks import LandmarkResult  # noqa: E402
from pipeline_base import Pipeline, PipelineResult  # noqa: E402


class _LandmarkDetector(Protocol):
    def detect(self, frame: np.ndarray) -> LandmarkResult: ...


@dataclass(frozen=True)
class DriverFrameResult(PipelineResult):
    """Per-frame landmarks, EAR and PERCLOS; None means unavailable, never zero."""

    landmarks: LandmarkResult
    ear: Optional[float]
    eyes_closed: Optional[bool]
    perclos: Optional[float]
    perclos_fill: float
    inference_ms: float


class PerclosTestPipeline(Pipeline[DriverFrameResult]):
    """Driver frame -> landmarks -> EAR -> PERCLOS."""

    name = "testperclos"
    REQUIRED_FRAMES = ("driver",)

    def __init__(
        self,
        detector: _LandmarkDetector,
        ear_threshold: float,
        window_seconds: float,
        min_fill_frac: float = 1.0,
        source_rate_hz: float = 21.0,
    ) -> None:
        super().__init__(source_rate_hz)
        self.detector = detector
        self.ear_threshold = float(ear_threshold)
        self.perclos = Perclos(
            ear_threshold=self.ear_threshold,
            window_seconds=window_seconds,
            fps=source_rate_hz,
            min_fill_frac=min_fill_frac,
        )
        self.frames = 0
        self.faces = 0
        self.no_face = 0
        self.closed_frames = 0
        self._total_ms = 0.0

    def _process(self, sync: FrameMeta, *, driver: np.ndarray) -> DriverFrameResult:
        started = time.perf_counter()
        result = self.detector.detect(driver)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.frames += 1
        self._total_ms += elapsed_ms

        # A valid result can still have too few points; EAR is then None.
        ear = ear_from_landmarks(result.points) if result.valid else None
        if ear is None:
            self.no_face += 1
            closed = None
        else:
            self.faces += 1
            closed = ear < self.ear_threshold
            self.closed_frames += int(closed)
            self.perclos.update(ear)

        return DriverFrameResult(
            sync=sync,
            landmarks=replace(result, sync=sync),
            ear=ear,
            eyes_closed=closed,
            perclos=self.perclos.value,
            perclos_fill=self.perclos.fill_frac,
            inference_ms=elapsed_ms,
        )

    @property
    def mean_inference_ms(self) -> Optional[float]:
        return None if self.frames == 0 else self._total_ms / self.frames

    @property
    def face_rate(self) -> Optional[float]:
        """Fraction of frames with a measurable EAR."""
        return None if self.frames == 0 else self.faces / self.frames

    def summary(self) -> str:
        mean = self.mean_inference_ms
        perclos = self.perclos.value
        return (
            f"{self.name}: {self.frames} frames, {self.faces} with a face, "
            f"{self.no_face} without, {self.closed_frames} eyes-closed, "
            f"PERCLOS {'warming up' if perclos is None else f'{perclos:.1%}'} "
            f"(window {self.perclos.fill_frac:.0%} filled), "
            f"{'n/a' if mean is None else f'{mean:.1f}'} ms mean"
        )

    def reset(self) -> None:
        self.perclos.reset()
        self.frames = 0
        self.faces = 0
        self.no_face = 0
        self.closed_frames = 0
        self._total_ms = 0.0

    def close(self) -> None:
        closer = getattr(self.detector, "close", None)
        if callable(closer):
            closer()


EYE_POINTS = set(EYE1_IDX) | set(EYE2_IDX)


def annotate(frame: np.ndarray, result: DriverFrameResult, threshold: float) -> np.ndarray:
    """Draw landmarks (eye points in red) and the frame's EAR/PERCLOS."""
    import cv2

    out = frame.copy()
    if result.landmarks.valid and result.landmarks.points is not None:
        for index, (x, y) in enumerate(result.landmarks.points):
            on_eye = index in EYE_POINTS
            cv2.circle(out, (int(round(x)), int(round(y))), 2 if on_eye else 1,
                       (0, 0, 255) if on_eye else (0, 200, 0), -1)

    ear = "no face" if result.ear is None else f"{result.ear:.3f}"
    state = "" if result.eyes_closed is None else ("  CLOSED" if result.eyes_closed else "  open")
    perclos = (f"warming up {result.perclos_fill:.0%}" if result.perclos is None
               else f"{result.perclos:.1%}")
    for row, text in enumerate((f"EAR {ear} (thr {threshold:.2f}){state}",
                                f"PERCLOS {perclos}")):
        cv2.putText(out, text, (12, 30 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def histogram(values: np.ndarray, bins: int = 20) -> str:
    """Text histogram of the observed EAR."""
    counts, edges = np.histogram(values, bins=bins)
    widest = max(counts.max(), 1)
    lines = []
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(40 * count / widest))
        lines.append(f"    {low:5.3f}-{high:5.3f} {count:6d} {bar}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face-engine", type=Path, required=True,
                    help="det_500m .engine built on THIS device")
    ap.add_argument("--landmark-engine", type=Path, required=True,
                    help="2d106det .engine built on THIS device")
    ap.add_argument("--source", required=True, help="video file, camera index, or a "
                                                    "GStreamer pipeline")
    ap.add_argument("--frames", type=int, default=900,
                    help="default is 900: at 30 fps that is one full 30 s PERCLOS window")
    ap.add_argument("--det-threshold", type=float, default=None,
                    help="override config/thresholds.yaml face_landmarks.det_threshold")
    ap.add_argument("--nms-threshold", type=float, default=None,
                    help="override config/thresholds.yaml face_landmarks.nms_threshold")
    ap.add_argument("--ear-threshold", type=float, default=None,
                    help="override config/thresholds.yaml ear_threshold (PROVISIONAL "
                         "and uncalibrated — see --ear-histogram)")
    ap.add_argument("--window", type=float, default=None,
                    help="override config/thresholds.yaml perclos.window_seconds")
    ap.add_argument("--rate", type=float, default=30.0,
                    help="source rate the envelope claims; it also converts the PERCLOS "
                         "window from seconds to frames")
    ap.add_argument("--ear-histogram", action="store_true",
                    help="print the observed EAR distribution (calibration aid)")
    ap.add_argument("--save-dir", type=Path, default=Path("/tmp/perclos"))
    ap.add_argument("--save-every", type=int, default=15,
                    help="write every Nth frame as a JPEG; 0 = none")
    add_debug_args(ap)
    args = ap.parse_args()

    import cv2

    from dgt.models.landmarks import FaceLandmarkDetector

    thresholds = load_thresholds()
    face = thresholds.face_landmarks
    ear_threshold = args.ear_threshold if args.ear_threshold is not None else thresholds.ear_threshold
    window = args.window if args.window is not None else thresholds.perclos.window_seconds

    try:
        detector = FaceLandmarkDetector(
            face_engine_path=args.face_engine,
            landmark_engine_path=args.landmark_engine,
            det_threshold=args.det_threshold if args.det_threshold is not None
            else face.det_threshold,
            nms_threshold=args.nms_threshold if args.nms_threshold is not None
            else face.nms_threshold,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[perclos] {exc}", file=sys.stderr)
        return 1
    print(f"[perclos] {detector.describe()}")
    print(f"[perclos] ear_threshold={ear_threshold} window={window}s @ {args.rate} fps "
          f"-> {int(round(window * args.rate))} frames")

    try:
        capture = open_source(args.source)
    except FileNotFoundError as exc:
        print(f"[perclos] {exc}", file=sys.stderr)
        return 1
    if not capture.isOpened():
        print(f"[perclos] cannot open source: {args.source}", file=sys.stderr)
        return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    screen = open_debug_screen(args, "perclos")
    ears: list[float] = []
    latencies: list[float] = []
    perclos_series: list[float] = []

    with PerclosTestPipeline(
        detector,
        ear_threshold=ear_threshold,
        window_seconds=window,
        min_fill_frac=thresholds.perclos.min_fill_frac,
        source_rate_hz=args.rate,
    ) as pipeline:
        while pipeline.frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            result = pipeline.tick(driver=frame, timestamp=time.monotonic())
            latencies.append(result.inference_ms)
            if result.ear is not None:
                ears.append(result.ear)
            if result.perclos is not None:
                perclos_series.append(result.perclos)
            saving = args.save_every > 0 and result.sync.frame_id % args.save_every == 0
            if saving or screen is not None:
                shown = annotate(frame, result, ear_threshold)
                if saving:
                    cv2.imwrite(str(args.save_dir / f"{result.sync.frame_id:05d}.jpg"), shown)
                if screen is not None:
                    ok_jpeg, buf = cv2.imencode(".jpg", shown,
                                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok_jpeg:
                        screen.publish(buf.tobytes())
            if pipeline.frames % 60 == 0:
                print(f"[perclos] {pipeline.frames} frames, "
                      f"{result.inference_ms:.0f} ms last, "
                      f"window {result.perclos_fill:.0%}")

        capture.release()
        if screen is not None:
            screen.close()
        if pipeline.frames == 0:
            print("[perclos] no frames decoded — check the source", file=sys.stderr)
            return 1

        lat = np.array(latencies)
        ear = np.array(ears) if ears else np.array([])
        summary = {
            "face_engine": str(args.face_engine),
            "landmark_engine": str(args.landmark_engine),
            "source": args.source,
            "frames": pipeline.frames,
            "frames_with_a_face": pipeline.faces,
            "frames_without_a_face": pipeline.no_face,
            "face_rate": round(pipeline.face_rate, 3),
            "det_threshold": detector.det_threshold,
            "nms_threshold": detector.nms_threshold,
            "ear_threshold": ear_threshold,
            "ear": {} if not ear.size else {
                "mean": round(float(ear.mean()), 4),
                "p5": round(float(np.percentile(ear, 5)), 4),
                "p50": round(float(np.percentile(ear, 50)), 4),
                "p95": round(float(np.percentile(ear, 95)), 4),
                "min": round(float(ear.min()), 4),
                "max": round(float(ear.max()), 4),
            },
            "eyes_closed_frames": pipeline.closed_frames,
            "window_seconds": window,
            "window_filled": pipeline.perclos.is_full,
            "perclos": None if pipeline.perclos.value is None
            else round(pipeline.perclos.value, 4),
            "perclos_max": None if not perclos_series else round(max(perclos_series), 4),
            "latency_ms": {"mean": round(float(lat.mean()), 1),
                           "p50": round(float(np.percentile(lat, 50)), 1),
                           "p95": round(float(np.percentile(lat, 95)), 1),
                           "max": round(float(lat.max()), 1)},
            "fps_single_stream": round(1000.0 / float(lat.mean()), 1),
        }
        (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        print(f"\n[perclos] {pipeline.summary()}")
        if ear.size:
            print(f"[perclos] EAR p5 {summary['ear']['p5']}  median "
                  f"{summary['ear']['p50']}  p95 {summary['ear']['p95']}")
            if args.ear_histogram:
                print("[perclos] EAR distribution:")
                print(histogram(ear))
        print(f"[perclos] latency mean {summary['latency_ms']['mean']} ms  "
              f"p95 {summary['latency_ms']['p95']} ms  -> "
              f"{summary['fps_single_stream']} fps")
        if args.save_every > 0:
            print(f"[perclos] frames written to {args.save_dir}/  (review them by eye: the "
                  f"red points must be on the eyelids)")
        print(f"[perclos] summary -> {args.save_dir}/summary.json")

        if not pipeline.perclos.is_full:
            print(f"[perclos] NOTE: the {window:.0f}s window never filled — PERCLOS is "
                  f"still a non-answer. Run at least "
                  f"{int(round(window * args.rate))} frames for one.")

        if pipeline.faces == 0:
            print("[perclos] ZERO frames with a measurable face — the engines ran but "
                  "produced nothing usable.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
