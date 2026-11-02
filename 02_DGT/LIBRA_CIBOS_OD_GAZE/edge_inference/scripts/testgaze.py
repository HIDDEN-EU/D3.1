#!/usr/bin/env python3
"""Drive `GazeEstimator` over a driver video and report what the engine emits.

    # 1. build the engines ON THIS DEVICE from the ONNX named in models.yaml
    dgt-build-engines --engine-dir engines

    # 2. run it
    python scripts/testgaze.py \\
        --face-engine engines/det_500m_fp16.engine \\
        --landmark-engine engines/2d106det_fp16.engine \\
        --gaze-engine engines/gaze_face_fp16.engine \\
        --source media/driver.mp4 --frames 600 --save-dir /tmp/gaze

    # 3. to watch it remotely, add --debug and open http://<this-host>:8090/

Without --calibration every result is valid=False by design.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Run straight from a clone, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgt.config.loader import load_thresholds  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from typing import Optional, Protocol, Sequence, Tuple  # noqa: E402
from dgt.contracts.common import FrameMeta  # noqa: E402
from dgt.contracts.gaze import GazeResult  # noqa: E402
from pipeline_base import Pipeline, PipelineResult  # noqa: E402


class _FaceBoxDetector(Protocol):

    def detect_face_box(self, frame: np.ndarray) -> Optional[np.ndarray]: ...


class _GazeEstimator(Protocol):

    last_raw: Optional[Tuple[float, float]]

    def estimate(self, frame: np.ndarray,
                 bbox: Optional[Sequence[int]]) -> GazeResult: ...


@dataclass(frozen=True)
class GazeFrameResult(PipelineResult):
    """One driver frame: face box, gaze contract, and the raw (uncalibrated) pair."""

    gaze: GazeResult
    raw: Optional[Tuple[float, float]]
    face_box: Optional[Tuple[int, int, int, int]]
    inference_ms: float


class GazeTestPipeline(Pipeline[GazeFrameResult]):
    """Face box then gaze, over one driver frame at a time."""

    name = "testgaze"
    REQUIRED_FRAMES = ("driver",)

    def __init__(
        self,
        detector: _FaceBoxDetector,
        estimator: _GazeEstimator,
        source_rate_hz: float = 30.0,
    ) -> None:
        super().__init__(source_rate_hz=source_rate_hz)
        self.detector = detector
        self.estimator = estimator
        self.frames = 0
        self.faces = 0
        self.no_face = 0
        self.valid_results = 0
        self._total_ms = 0.0
        # Running moments, so a long session stores no per-frame list.
        self._raw_n = 0
        self._raw_sum = [0.0, 0.0]
        self._raw_sumsq = [0.0, 0.0]

    def _process(self, sync: FrameMeta, *, driver: np.ndarray) -> GazeFrameResult:
        started = time.perf_counter()
        box = self.detector.detect_face_box(driver)
        result = self.estimator.estimate(
            driver, None if box is None else [int(v) for v in box]
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.frames += 1
        self._total_ms += elapsed_ms
        if box is None:
            self.no_face += 1
        else:
            self.faces += 1
        if result.valid:
            self.valid_results += 1

        raw = getattr(self.estimator, "last_raw", None)
        if raw is not None:
            self._raw_n += 1
            for axis in (0, 1):
                self._raw_sum[axis] += float(raw[axis])
                self._raw_sumsq[axis] += float(raw[axis]) ** 2

        return GazeFrameResult(
            sync=sync,
            # The model leaves `sync` unset.
            gaze=replace(result, sync=sync),
            raw=None if raw is None else (float(raw[0]), float(raw[1])),
            face_box=None if box is None else tuple(int(v) for v in box),
            inference_ms=elapsed_ms,
        )

    def reset(self) -> None:
        self.frames = self.faces = self.no_face = self.valid_results = 0
        self._total_ms = 0.0
        self._raw_n = 0
        self._raw_sum = [0.0, 0.0]
        self._raw_sumsq = [0.0, 0.0]

    def close(self) -> None:
        for model in (self.estimator, self.detector):
            closer = getattr(model, "close", None)
            if callable(closer):
                closer()

    @property
    def mean_inference_ms(self) -> Optional[float]:
        """Mean per-frame latency so far, or None before the first frame."""
        return None if self.frames == 0 else self._total_ms / self.frames

    @property
    def face_rate(self) -> Optional[float]:
        """Fraction of frames that yielded a face box, or None before any."""
        return None if self.frames == 0 else self.faces / self.frames

    @property
    def raw_std(self) -> Optional[Tuple[float, float]]:
        """Per-axis std of the raw output, or None with fewer than two points."""
        if self._raw_n < 2:
            return None
        out = []
        for axis in (0, 1):
            mean = self._raw_sum[axis] / self._raw_n
            variance = max(self._raw_sumsq[axis] / self._raw_n - mean * mean, 0.0)
            out.append(math.sqrt(variance))
        return (out[0], out[1])

    @property
    def raw_is_stuck(self) -> Optional[bool]:
        """True when the raw output never moved (a broken input or engine)."""
        spread = self.raw_std
        return None if spread is None else max(spread) < 1e-6

    def summary(self) -> str:
        mean = self.mean_inference_ms
        spread = self.raw_std
        if spread is None:
            movement = "raw spread n/a"
        elif self.raw_is_stuck:
            movement = "raw output STUCK — suspect the face crop or the engine"
        else:
            movement = f"raw std x {spread[0]:.4f} y {spread[1]:.4f}"
        return (
            f"{self.name}: {self.frames} frames, {self.faces} with a face, "
            f"{self.no_face} without, {self.valid_results} valid, {movement}, "
            f"{'n/a' if mean is None else f'{mean:.1f}'} ms mean"
        )


# Importable because scripts/ is sys.path[0] when a harness runs.
from debug_screen import add_debug_args, open_debug_screen  # noqa: E402
from dgt.sources import open_source  # noqa: E402


def annotate(frame: np.ndarray, result: GazeFrameResult) -> np.ndarray:
    """Draw the face box plus the raw and calibrated gaze."""
    import cv2

    out = frame.copy()
    if result.face_box is not None:
        x1, y1, x2, y2 = result.face_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)

    raw = result.raw
    raw_text = "raw --" if raw is None else f"raw x {raw[0]:+.4f}  y {raw[1]:+.4f}"
    if result.gaze.valid:
        state = f"gaze x {result.gaze.x:.3f}  y {result.gaze.y:.3f}  (calibrated)"
        colour = (0, 255, 0)
    else:
        state = "valid=False — uncalibrated, NOT a measurement"
        colour = (0, 165, 255)

    cv2.putText(out, raw_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, state, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                colour, 2, cv2.LINE_AA)
    return out


def load_calibration(path: Optional[Path]):
    """The fitted homography of dgt/calibration.py, or None."""
    if path is None:
        return None
    from dgt.calibration import Calibration

    return Calibration.load(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face-engine", type=Path, required=True,
                    help="det_500m .engine built on THIS device")
    ap.add_argument("--landmark-engine", type=Path, required=True,
                    help="2d106det .engine built on THIS device")
    ap.add_argument("--gaze-engine", type=Path, required=True,
                    help="face-crop gaze .engine built on THIS device")
    ap.add_argument("--source", required=True, help="video file, camera index, or a "
                                                    "GStreamer pipeline")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--rate", type=float, default=30.0,
                    help="source rate the envelope claims")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="the calibration JSON (dgt/calibration.py); without "
                         "it every result is valid=False by design")
    ap.add_argument("--det-threshold", type=float, default=None,
                    help="override config/thresholds.yaml face_landmarks.det_threshold")
    ap.add_argument("--nms-threshold", type=float, default=None,
                    help="override config/thresholds.yaml face_landmarks.nms_threshold")
    add_debug_args(ap)
    ap.add_argument("--save-dir", type=Path, default=Path("/tmp/gaze"))
    ap.add_argument("--save-every", type=int, default=15,
                    help="write every Nth frame as a JPEG; 0 = none")
    args = ap.parse_args()

    import cv2

    from dgt.models.gaze import GazeEstimator
    from dgt.models.landmarks import FaceLandmarkDetector

    thresholds = load_thresholds()
    face = thresholds.face_landmarks

    try:
        detector = FaceLandmarkDetector(
            face_engine_path=args.face_engine,
            landmark_engine_path=args.landmark_engine,
            det_threshold=args.det_threshold if args.det_threshold is not None
            else face.det_threshold,
            nms_threshold=args.nms_threshold if args.nms_threshold is not None
            else face.nms_threshold,
        )
        gaze = GazeEstimator(
            args.gaze_engine,
            calibration=load_calibration(args.calibration),
            screen_grid=(thresholds.gaze.screen_grid_cols,
                         thresholds.gaze.screen_grid_rows),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[gaze] {exc}", file=sys.stderr)
        return 1

    print(f"[gaze] {detector.describe()}")
    print(f"[gaze] {gaze.describe()}")

    try:
        capture = open_source(args.source)
    except FileNotFoundError as exc:
        print(f"[gaze] {exc}", file=sys.stderr)
        return 1
    if not capture.isOpened():
        print(f"[gaze] cannot open source: {args.source}", file=sys.stderr)
        return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    screen = open_debug_screen(args, "gaze")

    raws: List[Sequence[float]] = []

    with GazeTestPipeline(detector, gaze, source_rate_hz=args.rate) as pipeline:
        try:
            while pipeline.frames < args.frames:
                ok, frame = capture.read()
                if not ok:
                    break
                result = pipeline.tick(driver=frame, timestamp=time.monotonic())
                if result.raw is not None:
                    raws.append(result.raw)

                saving = args.save_every > 0 and result.sync.frame_id % args.save_every == 0
                if saving or screen is not None:
                    shown = annotate(frame, result)
                    if saving:
                        cv2.imwrite(
                            str(args.save_dir / f"{result.sync.frame_id:05d}.jpg"), shown)
                    if screen is not None:
                        ok_jpeg, buf = cv2.imencode(".jpg", shown,
                                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok_jpeg:
                            screen.publish(buf.tobytes())

                if pipeline.frames % 60 == 0:
                    print(f"[gaze] {pipeline.frames} frames, "
                          f"{result.inference_ms:.1f} ms last, raw {result.raw}")
        finally:
            capture.release()
            if screen is not None:
                screen.close()

        frames = pipeline.frames

    if frames == 0:
        print("[gaze] no frames decoded — check the source", file=sys.stderr)
        return 1

    print(f"\n[gaze] {pipeline.summary()}")

    summary = {
        "face_engine": str(args.face_engine),
        "landmark_engine": str(args.landmark_engine),
        "gaze_engine": str(args.gaze_engine),
        "source": args.source,
        "frames": frames,
        "frames_with_a_face": pipeline.faces,
        "frames_without_a_face": pipeline.no_face,
        "face_rate": round(pipeline.face_rate, 3),
        "valid_results": pipeline.valid_results,
        "calibrated": args.calibration is not None,
        "mean_inference_ms": round(pipeline.mean_inference_ms, 2),
        "raw_is_stuck": pipeline.raw_is_stuck,
    }

    if not raws:
        print("[gaze] NO raw output — the engine never ran, no face was found",
              file=sys.stderr)
        summary["raw"] = None
    else:
        arr = np.array(raws, dtype=np.float64)
        for axis, name in ((0, "x"), (1, "y")):
            column = arr[:, axis]
            print(f"[gaze] raw {name}: mean {column.mean():+.4f} "
                  f"std {column.std():.4f} "
                  f"range [{column.min():+.4f}, {column.max():+.4f}]")
        summary["raw"] = {
            name: {
                "mean": round(float(arr[:, axis].mean()), 4),
                "std": round(float(arr[:, axis].std()), 4),
                "min": round(float(arr[:, axis].min()), 4),
                "max": round(float(arr[:, axis].max()), 4),
            }
            for axis, name in ((0, "x"), (1, "y"))
        }
        verdict = ("IS STUCK — the same value every frame; suspect the "
                   "face crop or the engine" if pipeline.raw_is_stuck
                   else "varies across frames")
        print(f"[gaze] raw output {verdict}")

    if args.save_every > 0:
        print(f"[gaze] frames written to {args.save_dir}/")
    (args.save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[gaze] summary -> {args.save_dir / 'summary.json'}")

    if pipeline.valid_results == 0:
        print("[gaze] NOTE: no valid result. Without a calibration matrix that is "
              "the designed behaviour — the raw output above cannot be mapped into "
              "the [0, 1] the GazeResult contract promises. See dgt/models/gaze.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
