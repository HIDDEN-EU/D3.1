"""
Tiled video inference pipeline.

Runs the fine-tuned road-object detector over a video with the same 2×2 tiled
inference the model was trained for (see scripts/infer_test_tiled.py): each frame
is split into tiles, inferred per tile, boxes are lifted to full-frame coords and
merged across tile seams with class-wise NMS. Emits:

  - <output>/<stem>_annotated.mp4   — boxes drawn per frame (visual QA)
  - <output>/<stem>_detections.jsonl — one JSON record per processed frame:
      {"frame": int, "timestamp": float_seconds, "detections": [
          {"name": str, "cls": int, "conf": float, "xyxy": [x1,y1,x2,y2]}, ...]}

The JSONL (bbox + class + confidence + timestamp per frame) is the machine-readable
stream the gaze-correlation service consumes;
the MP4 is for eyeballing.

Note: full-frame inference would downsample away the small/distant objects tiling
recovers, so this tiles by default — matching production tiled inference.

Usage
-----
infer-video --source path/to/road_clip.mp4
infer-video --source clip.mp4 --weights runs/train/.../best.pt --conf 0.25 --stride 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from road_object_detection import inference
from scripts.infer_test_tiled import _grid_regions, _infer_tiled, _print_grid, _resolve_tiles

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs/train/tiled-v030-mz16-5ep/weights/best.pt"
WEBVIEW_SRC = Path(__file__).resolve().parent / "webview_video"

# Per-class BGR colours (cycled).
PALETTE = [
    (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
    (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
    (144, 153, 70),
]


def _draw(frame, dets, regions, show_tiles, active=None):
    if show_tiles:
        for i, (x0, y0, x1, y1) in enumerate(regions):
            on = active is None or i in active
            cv2.rectangle(frame, (x0, y0), (x1, y1),
                          (90, 90, 90) if on else (40, 40, 40), 1)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        c = PALETTE[d["cls"] % len(PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f'{d["name"]} {d["conf"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), c, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiled (2×2) video inference pipeline.")
    ap.add_argument("--source", type=Path, required=True, help="input video path")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "output/infer_video")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--merge-iou", type=float, default=inference.MERGE_IOU)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    ap.add_argument("--no-tiles-overlay", action="store_true", help="don't draw tile grid")
    ap.add_argument("--tiles", type=int, nargs="+", default=None, metavar="IDX",
                    help="tile indices to annotate (row-major, 0-based); default: all tiles")
    args = ap.parse_args()

    if not args.weights.exists():
        sys.exit(f"weights not found: {args.weights}")
    if not args.source.exists():
        sys.exit(f"source video not found: {args.source}")

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        sys.exit(f"could not open video: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = max(1.0, src_fps / args.stride)
    print(f"[infer-video] {args.source.name}  {w}x{h}  {src_fps:.1f}fps  ~{n_frames} frames  "
          f"stride={args.stride}")

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.source.stem
    raw_path = args.output / f"{stem}_annotated_raw.mp4"   # cv2 mp4v (not browser-playable)
    mp4_path = args.output / f"{stem}_annotated.mp4"        # H.264 after ffmpeg transcode
    jsonl_path = args.output / f"{stem}_detections.jsonl"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    print(f"[infer-video] loading {args.weights}")
    model = YOLO(str(args.weights))
    regions = _grid_regions(w, h)  # resolution-aware: ~640×360 tiles regardless of frame size
    select = _resolve_tiles(args.tiles, len(regions))
    _print_grid(regions, select)

    frame_idx = 0
    latencies = []        # per-frame pipeline latency (ms): tiling + inference + cross-tile NMS
    frames_meta = []      # light per-frame record for the dashboard timeline: [{f, t, n}]
    class_counts = Counter()
    with open(jsonl_path, "w") as jf:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.stride == 0:
                t0 = time.perf_counter()
                _, dets = _infer_tiled(model, frame, args.conf, args.imgsz, args.merge_iou,
                                       regions=regions, select=select)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                ts = round(frame_idx / src_fps, 4)
                jf.write(json.dumps({
                    "frame": frame_idx, "timestamp": ts,
                    "detections": [
                        {"name": d["name"], "cls": d["cls"], "conf": d["conf"], "xyxy": d["xyxy"]}
                        for d in dets
                    ],
                }) + "\n")
                for d in dets:
                    class_counts[d["name"]] += 1
                frames_meta.append({"f": frame_idx, "t": ts, "n": len(dets)})
                writer.write(_draw(frame, dets, regions, not args.no_tiles_overlay, active=select))
                if len(frames_meta) % 50 == 0:
                    print(f"[infer-video]   {len(frames_meta)} frames processed")
                if args.max_frames and len(frames_meta) >= args.max_frames:
                    break
            frame_idx += 1

    cap.release()
    writer.release()

    processed = len(frames_meta)
    if not processed:
        raw_path.unlink(missing_ok=True)
        sys.exit("[infer-video] no frames processed")

    # cv2 here can't encode H.264, so transcode the mp4v output to browser-playable H.264.
    playable = True
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path)],
            check=True,
        )
        raw_path.unlink(missing_ok=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raw_path.replace(mp4_path)  # keep mp4v; browser may not play it
        playable = False

    bench = latencies[1:] if len(latencies) > 1 else latencies  # drop frame-0 GPU warmup
    lat_sorted = sorted(bench)
    pct = lambda p: round(lat_sorted[min(len(lat_sorted) - 1, int(p / 100 * len(lat_sorted)))], 2)
    mean_ms = sum(bench) / len(bench)
    tiles = len(select) if select is not None else len(regions)  # tiles actually inferred
    total_dets = sum(class_counts.values())
    device_str = str(getattr(model, "device", "cpu"))
    stats = {
        "device": device_str,
        "resolution": f"{w}x{h}",
        "tiles_per_frame": tiles,
        "grid_tiles": len(regions),
        "annotated_tiles": "all" if select is None else sorted(select),
        "tile_size": f"{regions[0][2]-regions[0][0]}x{regions[0][3]-regions[0][1]}",
        "source_fps": round(src_fps, 2),
        "stride": args.stride,
        "frames_processed": processed,
        "warmup_frame_excluded": len(latencies) > 1,
        "pipeline_fps": round(1000.0 / mean_ms, 2),
        "mean_latency_ms": round(mean_ms, 2),
        "p50_latency_ms": pct(50),
        "p90_latency_ms": pct(90),
        "per_tile_inference_ms": round(mean_ms / tiles, 2),
        "total_detections": total_dets,
        "detections_per_frame": round(total_dets / processed, 2),
        "per_class": dict(class_counts.most_common()),
    }
    (args.output / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # self-contained dashboard (data.js global avoids file:// fetch restrictions)
    out_fps = max(1.0, src_fps / args.stride)
    data = {"video": mp4_path.name, "out_fps": out_fps, "playable": playable,
            "stats": stats, "frames": frames_meta}
    (args.output / "data.js").write_text("window.VIDEO_DATA = " + json.dumps(data) + ";\n")
    for f in ("index.html", "style.css", "app.js"):
        shutil.copy(WEBVIEW_SRC / f, args.output / f)

    print(f"\n[infer-video] done — {processed} frames, {total_dets} dets "
          f"({stats['detections_per_frame']}/frame)")
    print(f"[infer-video]   pipeline {stats['pipeline_fps']} FPS  (mean {stats['mean_latency_ms']} ms/frame "
          f"= {tiles} tiles × {stats['per_tile_inference_ms']} ms)  on {device_str}")
    print(f"[infer-video]   NOTE: measured on this GPU, not the Jetson Orin target (OD-2).")
    if not playable:
        print(f"[infer-video]   WARNING: ffmpeg transcode failed — {mp4_path.name} is mp4v; "
              f"the browser may not play it (use VLC).")
    print(f"[infer-video]   dashboard: {args.output / 'index.html'}")
    print(f"[infer-video]   files: {mp4_path.name}, {jsonl_path.name}, {stem}_stats.json")


if __name__ == "__main__":
    main()
