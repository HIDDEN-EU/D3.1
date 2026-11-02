"""dgt-daemon: road frames -> TensorRT -> one DGT_DS message per frame on ROS 2.
Run by deploy/dgt.service; exits cleanly on SIGTERM.

ROS 2 is the only output, so the daemon refuses to start when rclpy is not importable (ROS not sourced)
rather than burn the camera and GPU publishing nothing.

    {prefix}/status   std_msgs/String, the DGT_DS JSON (docs/DGT_DS.md)

Environment (or `.env` in the working directory):
    DGT_SENDER_ID              DGT_DS metadata.sender_id (required)
    DGT_ROAD_SOURCE            road frame source (required)
    ROAD_OBJECTS_ENGINE_PATH   road-object engine (required)
    DGT_ROAD_LOOP              "true" loops a file source
    ROS_TOPIC_PREFIX           default /dgt
    ROS_QOS_BEST_EFFORT        default true; false if a RELIABLE subscriber
                               receives nothing
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Mapping, Optional

from .config.loader import load_service, load_thresholds

REQUIRED = ("DGT_SENDER_ID", "DGT_ROAD_SOURCE", "ROAD_OBJECTS_ENGINE_PATH")


def _log(message: str) -> None:
    print(f"[dgt-daemon] {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class UnitSettings:
    sender_id: str
    road_source: str
    road_engine: Path
    road_loop: bool = False
    ros_topic_prefix: str = "/dgt"
    ros_qos_best_effort: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "UnitSettings":
        missing = [name for name in REQUIRED if not env.get(name)]
        if missing:
            raise RuntimeError(f"not set: {', '.join(missing)} (environment or .env)")
        return cls(
            sender_id=env["DGT_SENDER_ID"],
            road_source=env["DGT_ROAD_SOURCE"],
            road_engine=Path(env["ROAD_OBJECTS_ENGINE_PATH"]),
            road_loop=_flag(env.get("DGT_ROAD_LOOP"), False),
            ros_topic_prefix=env.get("ROS_TOPIC_PREFIX") or "/dgt",
            ros_qos_best_effort=_flag(env.get("ROS_QOS_BEST_EFFORT"), True),
        )


def _flag(value: Optional[str], default: bool) -> bool:
    if not value:
        return default
    return value.lower() in ("1", "true", "yes")


def load_env_file(path: Path) -> None:
    """Load a dotenv file into os.environ without overriding what is set."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _log(f"{path} ignored: python-dotenv is not installed (the `daemon` extra)")
        return
    load_dotenv(path, override=False)


def run(settings: UnitSettings, publisher, stop: Event,
        max_frames: Optional[int] = None) -> int:
    """`publisher` is anything with `publish(json_str)`; the caller owns it."""
    import cv2

    from .models.road_objects import RoadObjectDetector
    from .publishers.dgt_ds import MessageEnvelope, assemble_status
    from .sources import open_source

    config = load_service()
    envelope = MessageEnvelope(config, settings.sender_id)
    capture = open_source(settings.road_source)
    detector = None
    sent = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open road source: {settings.road_source}")
        params = load_thresholds().road_objects
        detector = RoadObjectDetector(
            engine_path=settings.road_engine,
            conf_threshold=params.conf_threshold,
            tile=params.tile,
        )
        _log(f"road: {detector.describe()}")
        _log("running." if max_frames is None else f"running for {max_frames} frames.")
        while not stop.is_set() and (max_frames is None or sent < max_frames):
            ok, frame = capture.read()
            if not ok and settings.road_loop:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                _log("road source ended.")
                break
            capture_ts = time.time()
            height, width = frame.shape[:2]
            status = assemble_status(detector.detect(frame), width, height, capture_ts,
                                     config.road_frame_id)
            publisher.publish(json.dumps(envelope.wrap(status).to_dict(),
                                         separators=(",", ":")))
            sent += 1
    finally:
        # The capture first: a leaked handle can leave nvarguscamerasrc unusable.
        capture.release()
        if detector is not None:
            detector.close()
        _log(f"stopped. messages={sent}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Headless DGT runtime: road frames "
                                                 "-> DGT_DS messages on ROS 2.")
    parser.add_argument("--frames", type=int, default=None, metavar="N",
                        help="exit after N messages (smoke tests)")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="dotenv file (default ./.env); the environment wins")
    args = parser.parse_args(argv)

    load_env_file(args.env_file)
    try:
        settings = UnitSettings.from_env(os.environ)
    except RuntimeError as exc:
        _log(str(exc))
        return 1

    stop = Event()

    def _handle(signum, _frame):
        _log(f"received {signal.Signals(signum).name}, stopping.")
        stop.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    from .publishers.ros_publisher import DgtRosPublisher

    publisher = DgtRosPublisher(topic_prefix=settings.ros_topic_prefix,
                                best_effort=settings.ros_qos_best_effort)
    if not publisher.enabled:
        _log("rclpy is not importable, and ROS 2 is this daemon's only output: "
             "source /opt/ros/humble/setup.bash first, in a venv created with "
             "--system-site-packages.")
        return 1
    _log(f"publishing {publisher.topic} (std_msgs/String, {publisher.qos_name}, depth 1).")
    try:
        return run(settings, publisher, stop, args.frames)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        _log(str(exc))
        return 1
    finally:
        publisher.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
