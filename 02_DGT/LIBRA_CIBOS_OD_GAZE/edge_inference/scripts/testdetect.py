#!/usr/bin/env python3
"""Drive `ObjectDetectionTestPipeline` over a video and save annotated frames (headless, no accuracy metrics).

    # 1. build the engines on this device (engines are device-specific)
    dgt-build-engines --engine-dir engines

    # 2. run it, then look at the frames
    python scripts/testdetect.py --engine engines/road_objects_640x640_fp16.engine \\
        --source media/clip.mp4 --frames 60 --save-dir /tmp/detect
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Run from a clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgt.config.loader import load_thresholds  # noqa: E402
from dgt.contracts.detections import DetectionResult  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from typing import Optional, Protocol  # noqa: E402
from dgt.contracts.common import FrameMeta  # noqa: E402
from pipeline_base import Pipeline, PipelineResult  # noqa: E402
from debug_screen import add_debug_args, open_debug_screen, publish_frame  # noqa: E402
from dgt.sources import open_source  # noqa: E402


class _Detector(Protocol):
    """Detector interface; a Protocol so this never imports TensorRT."""

    def detect(self, frame: np.ndarray) -> DetectionResult: ...


@dataclass(frozen=True)
class RoadFrameResult(PipelineResult):
    """One road frame's detections (`objects.valid=False` means no inference ran) and latency."""

    objects: DetectionResult
    inference_ms: float


class ObjectDetectionTestPipeline(Pipeline[RoadFrameResult]):
    """Road frame in, `DetectionResult` out."""

    name = "testdetect"
    REQUIRED_FRAMES = ("road",)

    def __init__(self, detector: _Detector, source_rate_hz: float = 30.0) -> None:
        super().__init__(source_rate_hz)
        self.detector = detector
        self.frames = 0
        self.detections = 0
        self.invalid = 0
        self._total_ms = 0.0

    def _process(self, sync: FrameMeta, *, road: np.ndarray) -> RoadFrameResult:
        import time

        started = time.perf_counter()
        result = self.detector.detect(road)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.frames += 1
        if result.valid:
            self.detections += len(result.detections)
        else:
            self.invalid += 1
        self._total_ms += elapsed_ms

        # The detector leaves `sync` unset.
        return RoadFrameResult(
            sync=sync,
            objects=replace(result, sync=sync),
            inference_ms=elapsed_ms,
        )

    @property
    def mean_inference_ms(self) -> Optional[float]:
        """Mean detector latency, or None before the first frame."""
        return None if self.frames == 0 else self._total_ms / self.frames

    def summary(self) -> str:
        mean = self.mean_inference_ms
        rate = "n/a" if not mean else f"{1000.0 / mean:.1f}"
        return (
            f"{self.name}: {self.frames} frames, {self.detections} detections, "
            f"{self.invalid} invalid, {'n/a' if mean is None else f'{mean:.1f}'} ms mean "
            f"(~{rate} fps)"
        )

    def reset(self) -> None:
        self.frames = 0
        self.detections = 0
        self.invalid = 0
        self._total_ms = 0.0

    def close(self) -> None:
        closer = getattr(self.detector, "close", None)
        if callable(closer):
            closer()


# Vulnerable road users, drawn in red.
VRU = {"pedestrian", "rider", "motorcycle", "bicycle"}


def annotate(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Draw the normalized boxes onto the frame."""
    import cv2

    out = frame.copy()
    height, width = frame.shape[:2]
    for det in result.detections:
        x0, y0, x1, y1 = det.box
        p0 = (int(x0 * width), int(y0 * height))
        p1 = (int(x1 * width), int(y1 * height))
        colour = (0, 0, 255) if det.label in VRU else (0, 200, 0)
        cv2.rectangle(out, p0, p1, colour, 2)
        label = f"{det.label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p0[0], p0[1] - th - 4), (p0[0] + tw, p0[1]), colour, -1)
        cv2.putText(out, label, (p0[0], p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", type=Path, required=True,
                    help="a .engine built on THIS device from the published ONNX")
    ap.add_argument("--source", required=True, help="video file, or a GStreamer pipeline")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--conf", type=float, default=None,
                    help="override config/thresholds.yaml road_objects.conf_threshold")
    ap.add_argument("--no-tile", action="store_true",
                    help="run the whole frame instead of tiling — for comparison only; "
                         "these weights were fine-tuned on 640x360 crops")
    ap.add_argument("--rate", type=float, default=30.0, help="source rate the envelope claims")
    ap.add_argument("--save-dir", type=Path, default=Path("/tmp/detect"))
    ap.add_argument("--save-every", type=int, default=1,
                    help="write every Nth frame as a JPEG; 0 = none")
    add_debug_args(ap)
    args = ap.parse_args()

    import cv2

    from dgt.models.road_objects import RoadObjectDetector

    params = load_thresholds().road_objects
    try:
        detector = RoadObjectDetector(
            engine_path=args.engine,
            conf_threshold=args.conf if args.conf is not None else params.conf_threshold,
            tile=params.tile and not args.no_tile,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[detect] {exc}", file=sys.stderr)
        return 1
    print(f"[detect] {detector.describe()}")

    try:
        capture = open_source(args.source)
    except FileNotFoundError as exc:
        print(f"[detect] {exc}", file=sys.stderr)
        return 1
    if not capture.isOpened():
        print(f"[detect] cannot open source: {args.source}", file=sys.stderr)
        return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    screen = open_debug_screen(args, "detect")
    per_class: dict[str, int] = {}
    latencies: list[float] = []
    empty = 0

    with ObjectDetectionTestPipeline(detector, source_rate_hz=args.rate) as pipeline:
        while pipeline.frames < args.frames:
            ok, frame = capture.read()
            if not ok:
                break
            if pipeline.frames == 0:
                # The tiling grid depends on the first frame's size.
                print(f"[detect] {detector.describe(frame.shape)}")
            result = pipeline.tick(road=frame, timestamp=time.monotonic())
            latencies.append(result.inference_ms)

            for det in result.objects.detections:
                per_class[det.label] = per_class.get(det.label, 0) + 1
            if result.objects.valid and not result.objects.detections:
                empty += 1
            saving = args.save_every > 0 and result.sync.frame_id % args.save_every == 0
            if saving or screen is not None:
                view = annotate(frame, result.objects)
                if saving:
                    cv2.imwrite(str(args.save_dir / f"{result.sync.frame_id:04d}.jpg"), view)
                publish_frame(screen, view)
            if pipeline.frames % 20 == 0:
                print(f"[detect] {pipeline.frames} frames, {result.inference_ms:.0f} ms last")

        capture.release()
        if screen is not None:
            screen.close()
        if pipeline.frames == 0:
            print("[detect] no frames decoded — check the source", file=sys.stderr)
            return 1

        lat = np.array(latencies)
        vru = sum(n for k, n in per_class.items() if k in VRU)
        summary = {
            "engine": str(args.engine), "source": args.source,
            "frames": pipeline.frames, "tiled": detector.tile,
            "conf": detector.conf_threshold,
            "detections": pipeline.detections, "per_class": per_class,
            "vru_detections": int(vru),
            "frames_with_no_detection": empty,
            "invalid_results": pipeline.invalid,
            "latency_ms": {"mean": round(float(lat.mean()), 1),
                           "p50": round(float(np.percentile(lat, 50)), 1),
                           "p95": round(float(np.percentile(lat, 95)), 1),
                           "max": round(float(lat.max()), 1)},
            "fps_single_stream": round(1000.0 / float(lat.mean()), 1),
        }
        (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        print(f"\n[detect] {pipeline.summary()}")
        print(f"[detect] {summary['vru_detections']} VRU, {empty} frames with none, "
              f"{pipeline.invalid} invalid")
        for name in sorted(per_class, key=lambda k: -per_class[k]):
            mark = "  <- VRU" if name in VRU else ""
            print(f"[detect]   {name:15s} {per_class[name]:5d}{mark}")
        print(f"[detect] latency mean {summary['latency_ms']['mean']} ms  "
              f"p95 {summary['latency_ms']['p95']} ms  -> "
              f"{summary['fps_single_stream']} fps")
        if args.save_every > 0:
            print(f"[detect] frames written to {args.save_dir}/  (review them by eye)")
        print(f"[detect] summary -> {args.save_dir}/summary.json")

        # Zero detections means a wrong engine or broken decoding.
        if pipeline.detections == 0:
            print("[detect] ZERO detections across every frame — the engine ran but "
                  "produced nothing usable.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
