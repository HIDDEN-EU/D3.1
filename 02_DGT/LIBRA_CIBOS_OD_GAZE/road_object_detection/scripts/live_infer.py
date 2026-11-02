"""
Live tiled-inference MJPEG feed — watch the model run in real time from a browser.

Loops a road/dashcam video at real-time pace (simulating a live camera), runs the
same resolution-aware tiled inference the model was trained for, burns boxes + a live
FPS/latency/detection overlay onto each frame, and serves the result as an MJPEG stream
(multipart/x-mixed-replace) over HTTP — no display or X11 needed on this headless box.

Open from your laptop:  http://<host>:8088/   (or port-forward 8088 and use localhost)

A producer thread does inference as fast as the GPU allows and (optionally) paces to the
source fps so it looks like a real camera; the overlay reports the model's true capability
(infer FPS = 1/latency) separately from the displayed stream fps.

This is a long-running server — start it yourself and Ctrl+C to stop.

Usage
-----
live-infer --source road_clip.mp4
live-infer --source road_clip.mp4 --port 8088 --conf 0.3 --no-pace   # run flat-out
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from ultralytics import YOLO

from road_object_detection import inference
from scripts.infer_test_tiled import _grid_regions, _infer_tiled, _print_grid, _resolve_tiles

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/train/tiled-v030-mz16-5ep/weights/best.pt"
PALETTE = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
    (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
    (144, 153, 70),
]


class _Feed:
    """Holds the latest encoded JPEG frame; producer writes, HTTP handlers read."""

    def __init__(self):
        self.jpeg = None
        self.cond = threading.Condition()
        self.running = True

    def publish(self, jpeg_bytes):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.cond.notify_all()

    def latest(self, last_id):
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg is not None and id(self.jpeg) != last_id, timeout=5)
            return self.jpeg, id(self.jpeg)


def _draw_overlay(frame, dets, regions, infer_ms, stream_fps, active=None):
    for i, (x0, y0, x1, y1) in enumerate(regions):
        on = active is None or i in active
        cv2.rectangle(frame, (x0, y0), (x1, y1), (70, 70, 70) if on else (35, 35, 35), 1)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        c = PALETTE[d["cls"] % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        lab = f'{d["name"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), c, -1)
        cv2.putText(frame, lab, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    tiles_str = f"{len(regions)} tiles" if active is None else f"{len(active)}/{len(regions)} tiles"
    hud = f"infer {1000.0/infer_ms:4.1f} FPS ({infer_ms:5.1f} ms) | stream {stream_fps:4.1f} fps | {len(dets)} dets | {tiles_str} | cuda"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 120), 1, cv2.LINE_AA)
    return frame


def _producer(feed, args, select):
    model = YOLO(str(args.weights))
    cap = cv2.VideoCapture(str(args.source))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = 1.0 / (args.max_fps or src_fps)
    regions = None
    stream_fps = 0.0
    last_pub = time.perf_counter()
    print(f"[live] streaming {args.source.name} @ http://0.0.0.0:{args.port}/  "
          f"(src {src_fps:.0f}fps, pace={'off' if args.no_pace else f'{args.max_fps or src_fps:.0f}fps'})")
    while feed.running:
        ok, frame = cap.read()
        if not ok:                       # loop the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if regions is None:
            h, w = frame.shape[:2]
            regions = _grid_regions(w, h)
        t0 = time.perf_counter()
        _, dets = _infer_tiled(model, frame, args.conf, args.imgsz, args.merge_iou,
                               regions=regions, select=select)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        stream_fps = 0.9 * stream_fps + 0.1 * (1.0 / max(1e-3, now - last_pub))
        last_pub = now
        _draw_overlay(frame, dets, regions, infer_ms, stream_fps, active=select)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            feed.publish(buf.tobytes())
        if not args.no_pace:
            slack = frame_interval - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    cap.release()


_PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>live-infer</title><style>body{margin:0;background:#111;color:#ccc;
font-family:system-ui}h1{font-size:15px;padding:8px 12px;margin:0;background:#1b1f27}
img{display:block;max-width:100%;margin:0 auto}</style></head>
<body><h1>Tiled live inference &mdash; overlay shows infer FPS / latency / detections
(RTX&nbsp;3090, not Jetson Orin)</h1><img src="/stream"></body></html>"""


def _make_handler(feed):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(_PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_id = None
            try:
                while feed.running:
                    jpeg, last_id = feed.latest(last_id)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="Live tiled-inference MJPEG feed (looped video).")
    ap.add_argument("--source", type=Path, required=True, help="road/dashcam video to loop")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--merge-iou", type=float, default=inference.MERGE_IOU)
    ap.add_argument("--max-fps", type=float, default=None,
                    help="pace cap (default = source fps); ignored with --no-pace")
    ap.add_argument("--no-pace", action="store_true", help="run flat-out to measure max FPS")
    ap.add_argument("--tiles", type=int, nargs="+", default=None, metavar="IDX",
                    help="tile indices to annotate (row-major, 0-based); default: all tiles")
    args = ap.parse_args()

    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}")
    if not args.source.exists():
        sys.exit(f"source not found: {args.source}")

    # Probe resolution up front so a bad --tiles value fails here on the main
    # thread, not silently inside the daemon producer.
    probe = cv2.VideoCapture(str(args.source))
    ok, frame0 = probe.read()
    probe.release()
    if not ok:
        sys.exit(f"could not read frames from {args.source}")
    regions0 = _grid_regions(frame0.shape[1], frame0.shape[0])
    select = _resolve_tiles(args.tiles, len(regions0))
    _print_grid(regions0, select)

    feed = _Feed()
    prod = threading.Thread(target=_producer, args=(feed, args, select), daemon=True)
    prod.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), _make_handler(feed))
    print(f"[live] open http://<host>:{args.port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[live] stopping")
    finally:
        feed.running = False
        server.shutdown()


if __name__ == "__main__":
    main()
