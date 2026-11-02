"""Open a frame source — the only capture code in the package, shared by the
daemon and the harnesses in scripts/.

A source is a device index ("0"), a GStreamer pipeline ending in `appsink`, a
URL, a video file, or "csi" for the driver-facing CSI camera. cv2 is imported on use.
"""
from __future__ import annotations

from pathlib import Path


def csi_pipeline(
    sensor_id: int = 0,
    width: int = 3280,
    height: int = 2464,
    fps: int = 21,
    flip_method: int = 0,
) -> str:
    """GStreamer pipeline for an IMX219 CSI camera on a Jetson, ending in a BGR `appsink`.

    The default mode, 3280x2464 at 21 fps, matches the resolution of the training data.
    `drop=true max-buffers=1` keeps only the newest frame, so a slow consumer never
    reads stale frames.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def open_source(source: str, gst: bool = False):
    """A `cv2.VideoCapture` for `source`; the caller checks `isOpened()`.

    Pipelines go to the GStreamer backend explicitly: JetPack's OpenCV cannot
    parse them otherwise. A missing file raises instead of reaching GStreamer,
    which would misreport it as a pipeline error.
    """
    import cv2

    if source == "csi":
        source = csi_pipeline()
    if gst or "appsink" in source:
        return cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if "://" not in source and not Path(source).is_file():
        raise FileNotFoundError(f"no such file: {source}")
    return cv2.VideoCapture(source)
