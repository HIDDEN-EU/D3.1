#!/usr/bin/env python3
"""Point a video source at `CameraTestPipeline` and report what arrives (headless).

    # a laptop streaming to this Jetson over UDP
    python scripts/testcamera.py --source 'udp://@:5000'

    # a file you scp'd across
    python scripts/testcamera.py --source clip.mp4 --frames 200

    # a USB camera plugged into the Jetson
    python scripts/testcamera.py --source 0

    # save a few frames to eyeball, since there is no display
    python scripts/testcamera.py --source 'udp://@:5000' --save-dir /tmp/frames
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from a clone without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass  # noqa: E402
from typing import Optional, Tuple  # noqa: E402
import numpy as np  # noqa: E402
from dgt.contracts.common import FrameMeta  # noqa: E402
from pipeline_base import Pipeline, PipelineResult  # noqa: E402
from debug_screen import add_debug_args, open_debug_screen, publish_frame  # noqa: E402
from dgt.sources import open_source  # noqa: E402


@dataclass(frozen=True)
class CameraFrameResult(PipelineResult):
    """What one frame looked like on arrival; `measured_fps` is the real rate, not the claimed one."""

    width: int
    height: int
    channels: int
    dtype: str
    mean_intensity: float
    dt_s: Optional[float]
    measured_fps: Optional[float]


class CameraTestPipeline(Pipeline[CameraFrameResult]):
    """Reports on the frames handed to it; no model."""

    name = "testcamera"
    REQUIRED_FRAMES = ("camera",)

    def __init__(self, source_rate_hz: float = 30.0, smoothing: float = 0.1) -> None:
        """`source_rate_hz` is the claimed rate; `smoothing` is the EMA factor for the measured one."""
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        super().__init__(source_rate_hz)
        self.smoothing = smoothing
        self.frames = 0
        self.resolution_changes = 0
        self._shape: Optional[Tuple[int, int, int]] = None
        self._last_t: Optional[float] = None
        self._fps: Optional[float] = None

    def _process(self, sync: FrameMeta, *, camera: np.ndarray) -> CameraFrameResult:
        if camera.ndim not in (2, 3):
            raise ValueError(
                f"{self.name}: expected a 2-D or 3-D frame, got shape {camera.shape}"
            )
        height, width = camera.shape[:2]
        channels = 1 if camera.ndim == 2 else camera.shape[2]

        shape = (height, width, channels)
        if self._shape is not None and shape != self._shape:
            # A mid-stream resolution change invalidates held pixel coordinates.
            self.resolution_changes += 1
        self._shape = shape
        self.frames += 1

        dt: Optional[float] = None
        if self._last_t is not None:
            dt = sync.timestamp - self._last_t
            if dt > 0:
                instant = 1.0 / dt
                self._fps = (
                    instant
                    if self._fps is None
                    else (1 - self.smoothing) * self._fps + self.smoothing * instant
                )
        self._last_t = sync.timestamp

        mean_intensity = float(np.asarray(camera[::8, ::8], dtype=np.float64).mean())

        return CameraFrameResult(
            sync=sync,
            width=width,
            height=height,
            channels=channels,
            dtype=str(camera.dtype),
            mean_intensity=mean_intensity,
            dt_s=dt,
            measured_fps=self._fps,
        )

    @property
    def measured_fps(self) -> Optional[float]:
        """Smoothed arrival rate, or None before the second frame."""
        return self._fps

    def summary(self) -> str:
        shape = "none" if self._shape is None else f"{self._shape[1]}x{self._shape[0]}x{self._shape[2]}"
        fps = "n/a" if self._fps is None else f"{self._fps:.1f}"
        return (
            f"{self.name}: {self.frames} frames, {shape}, ~{fps} fps measured "
            f"(claimed {self.source_rate_hz:g}), {self.resolution_changes} resolution changes"
        )

    def reset(self) -> None:
        self.frames = 0
        self.resolution_changes = 0
        self._shape = None
        self._last_t = None
        self._fps = None


NO_FRAMES_HINT = """
No frames arrived. Isolate one layer at a time — in this order:

  1. Does the script itself work? (no network involved)
       python scripts/testcamera.py --frames 30 --source \\
         'videotestsrc ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false'

  2. Are packets landing? (tcpdump is not installed on JetPack)
       python -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);\\
       s.bind(('',5000));s.settimeout(5);d,a=s.recvfrom(4096);print(len(d),a,d[:8].hex())"
     MPEG-TS starts with 47. Stop it before retrying — it holds the port.

  3. Which decoders does this box have?
       gst-inspect-1.0 | grep -iE "h264|jpegdec|nvv4l2"
     JetPack has nvv4l2decoder (hardware) but NOT avdec_h264.

  4. Can GStreamer alone decode it?  -v prints the negotiated caps at every pad,
     which is where a rejected profile shows up.
       gst-launch-1.0 -v udpsrc port=5000 caps="video/mpegts,systemstream=(boolean)true,packetsize=(int)188" \\
         ! tsdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! 'video/x-raw,format=BGRx' ! fakesink

The sender must be started BEFORE this script: OpenCV blocks on preroll and
errors out if no buffer arrives. See the README for the verified sender command
(-pix_fmt yuv420p and repeat-headers=1 are both required).
"""


def overlay(frame, result):
    import cv2

    out = frame.copy()
    fps = "n/a" if result.measured_fps is None else f"{result.measured_fps:.1f}"
    text = (f"frame {result.sync.frame_id}  {result.width}x{result.height}  "
            f"fps {fps}  mean {result.mean_intensity:.1f}")
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        required=True,
        help="device index (0), file path, URL (udp://@:5000, rtsp://...), "
        "or a full GStreamer pipeline ending in appsink",
    )
    parser.add_argument("--frames", type=int, default=100, help="stop after N frames (0 = forever)")
    parser.add_argument("--print-every", type=int, default=10, help="print one line every N frames")
    parser.add_argument("--save-dir", type=Path, help="write a JPEG every --print-every frames")
    parser.add_argument("--gst", action="store_true", help="force the GStreamer backend")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="give up after N seconds with no frame"
    )
    add_debug_args(parser)
    args = parser.parse_args(argv)

    import cv2

    try:
        cap = open_source(args.source, gst=args.gst)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not cap.isOpened():
        print(f"could not open source: {args.source}", file=sys.stderr)
        print(NO_FRAMES_HINT, file=sys.stderr)
        return 2

    claimed = cap.get(cv2.CAP_PROP_FPS) or 30.0
    backend = cap.getBackendName()
    print(f"opened {args.source!r} via {backend} (source claims {claimed:.1f} fps)")
    if backend == "GSTREAMER" and "appsink" not in args.source:
        print(
            "NOTE: OpenCV chose the GStreamer backend, which does not understand "
            "FFmpeg URL syntax. Pass a full GStreamer pipeline ending in appsink.",
            file=sys.stderr,
        )
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    screen = open_debug_screen(args, "camera")
    pipeline = CameraTestPipeline(source_rate_hz=claimed)
    deadline = time.monotonic() + args.timeout
    try:
        with pipeline:
            while args.frames == 0 or pipeline.frames < args.frames:
                ok, frame = cap.read()
                if not ok:
                    if pipeline.frames == 0 and time.monotonic() < deadline:
                        continue
                    print("stream ended" if pipeline.frames else "no frames received")
                    if not pipeline.frames:
                        print(NO_FRAMES_HINT, file=sys.stderr)
                    break
                deadline = time.monotonic() + args.timeout

                result = pipeline.tick(camera=frame)
                if screen is not None:
                    publish_frame(screen, overlay(frame, result))
                if result.sync.frame_id % args.print_every == 0:
                    fps = "  n/a" if result.measured_fps is None else f"{result.measured_fps:5.1f}"
                    print(
                        f"frame {result.sync.frame_id:5d}  "
                        f"{result.width}x{result.height}x{result.channels} {result.dtype}  "
                        f"fps {fps}  mean {result.mean_intensity:6.1f}"
                    )
                    if args.save_dir:
                        cv2.imwrite(str(args.save_dir / f"{result.sync.frame_id:06d}.jpg"), frame)
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        if screen is not None:
            screen.close()

    print(pipeline.summary())
    measured = pipeline.measured_fps
    if measured is not None and measured < claimed * 0.7:
        print(
            f"NOTE: arriving at ~{measured:.1f} fps against a claimed {claimed:.1f}. "
            "Every window in dgt/attention is scaled by fps, so it covers less "
            "real time than configured."
        )
    elif measured is not None and measured > claimed * 2:
        print(
            f"NOTE: arriving at ~{measured:.0f} fps against a claimed {claimed:.1f} — "
            "a non-live source (a file, videotestsrc) runs as fast as it can be "
            "read, so this is a decode rate, not a delivery rate."
        )
    return 0 if pipeline.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
