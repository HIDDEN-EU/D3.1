"""`--debug`: serve annotated frames as MJPEG at http://<jetson>:<port>/. Off by default."""
from __future__ import annotations

import argparse
import socket
from typing import Optional

import numpy as np

DEFAULT_PORT = 8090


def add_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--debug", action="store_true",
                        help="serve the live debug screen (MJPEG) to a browser; "
                             "off by default")
    parser.add_argument("--debug-port", type=int, default=DEFAULT_PORT, metavar="PORT",
                        help=f"port for --debug (default {DEFAULT_PORT})")


def _lan_address() -> str:
    """Outward-facing address; connecting a UDP socket sends no packet."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            return s.getsockname()[0]
    except OSError:
        return "<this-host>"


def open_debug_screen(args: argparse.Namespace, tag: str) -> Optional["DebugScreen"]:
    """A running screen when `--debug` was given, else None."""
    if not getattr(args, "debug", False):
        return None
    screen = DebugScreen(args.debug_port)
    print(f"[{tag}] debug screen: http://{_lan_address()}:{args.debug_port}/")
    return screen


def publish_frame(screen: Optional["DebugScreen"], frame: np.ndarray,
                  quality: int = 80) -> None:
    """JPEG-encode `frame` and publish it; no-op when screen is None."""
    if screen is None:
        return
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        screen.publish(buf.tobytes())


class DebugScreen:
    """MJPEG-over-HTTP server; latest frame only, so slow viewers drop frames."""

    def __init__(self, port: int) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0
        screen = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seen = -1
                while True:
                    with screen._cond:
                        screen._cond.wait_for(lambda: screen._seq != seen, timeout=5.0)
                        frame, seen = screen._frame, screen._seq
                    if frame is None:
                        continue
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: %d\r\n\r\n" % len(frame))
                        self.wfile.write(frame + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return

        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()

    def close(self) -> None:
        self._httpd.shutdown()
