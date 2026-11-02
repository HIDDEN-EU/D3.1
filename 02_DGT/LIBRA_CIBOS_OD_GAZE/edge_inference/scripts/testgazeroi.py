#!/usr/bin/env python3
"""Gaze-directed road inference: infer only the region the driver is looking at.

    python scripts/testgazeroi.py \
        --road-engine engines/road_objects_640x640_fp16.engine \
        --face-engine engines/det_500m_fp16.engine \
        --landmark-engine engines/2d106det_fp16.engine \
        --gaze-engine engines/gaze_face_fp16.engine \
        --road-source road.mp4 \
        --driver-source "$SRC" --debug --frames 900

The region is the fixed grid tile (640x360, the weights' tile size) holding the gaze point.

Uncalibrated: the highlighted region is NOT where the driver looked (roi_valid=False).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgt.config.loader import load_thresholds  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import List, Optional, Protocol, Sequence, Tuple  # noqa: E402
from dgt.attention.gaze_roi import (  # noqa: E402
    GazeProjection, Region, dilate_region, region_containing,
)
from dgt.contracts.common import FrameMeta  # noqa: E402
from dgt.contracts.detections import DetectionResult  # noqa: E402
from dgt.contracts.gaze import GazeResult  # noqa: E402
from pipeline_base import Pipeline, PipelineResult  # noqa: E402


class _RoadDetector(Protocol):
    def detect(self, frame: np.ndarray,
               regions: Optional[Sequence[Region]] = None) -> DetectionResult: ...
    def regions_for(self, width: int, height: int) -> List[Region]: ...
    @property
    def tile_size(self) -> Tuple[int, int]: ...


class _GazeModel(Protocol):
    def estimate(self, frame: np.ndarray, box) -> GazeResult: ...


class _FaceModel(Protocol):
    def detect_face_box(self, frame: np.ndarray): ...


@dataclass(frozen=True)
class GazeRoiResult(PipelineResult):
    """One tick: gaze, and detections from the chosen region (boxes normalized to the whole frame).

    `roi_region` is the pixels the detector saw; `roi_index` is the grid tile, `grid` mode only.
    """

    gaze: GazeResult
    objects: DetectionResult
    roi_index: Optional[int]
    roi_region: Optional[Region]
    roi_point: Optional[Tuple[float, float]]
    regions: Tuple[Region, ...]
    roi_valid: bool
    road_ms: float
    driver_ms: float
    frame_skew_ms: Optional[float] = None


class GazeRoiPipeline(Pipeline):
    """Road + driver frames -> detections from the tile the driver looked at."""

    name = "gazeroi"
    REQUIRED_FRAMES = ("road", "driver")

    def __init__(
        self,
        detector: _RoadDetector,
        face: _FaceModel,
        gaze: _GazeModel,
        projection: Optional[GazeProjection] = None,
        source_rate_hz: float = 30.0,
        on_no_gaze: str = "skip",
        roi_margin: float = 0.0,
        trust_projection: bool = False,
    ) -> None:
        """`trust_projection` is the only way `roi_valid` can become True."""
        super().__init__(source_rate_hz=source_rate_hz)
        if on_no_gaze not in ("skip", "full"):
            raise ValueError(f"on_no_gaze must be 'skip' or 'full', got {on_no_gaze!r}")
        self._detector = detector
        self._face = face
        self._gaze = gaze
        self._projection = projection or GazeProjection()
        self._on_no_gaze = on_no_gaze
        self._roi_margin = float(roi_margin)
        self._trust_projection = bool(trust_projection)
        self._pending_skew_ms: Optional[float] = None
        self.frames = 0

    def tick_pair(self, road: np.ndarray, driver: np.ndarray,
                  road_timestamp: float, driver_timestamp: float) -> GazeRoiResult:
        """`tick()` for separately captured frames; records skew, stamps with the road timestamp."""
        self._pending_skew_ms = abs(road_timestamp - driver_timestamp) * 1000.0
        return self.tick(road=road, driver=driver, timestamp=road_timestamp)

    def _process(self, sync: FrameMeta, *, road: np.ndarray,
                 driver: np.ndarray) -> GazeRoiResult:
        skew_ms, self._pending_skew_ms = self._pending_skew_ms, None
        self.frames += 1

        started = time.perf_counter()
        box = self._face.detect_face_box(driver)
        gaze = self._gaze.estimate(driver, box)
        driver_ms = (time.perf_counter() - started) * 1000.0

        height, width = road.shape[:2]
        regions = tuple(self._detector.regions_for(width, height))

        point = self._point_for(gaze)
        index = roi_region = None
        if point is not None:
            index = region_containing(regions, point[0], point[1], width, height)
            if index is not None:
                roi_region = dilate_region(regions[index], self._roi_margin, width, height)

        if roi_region is None:
            chosen = regions if self._on_no_gaze == "full" else ()
        else:
            chosen = (roi_region,)

        started = time.perf_counter()
        objects = self._detector.detect(road, regions=chosen)
        road_ms = (time.perf_counter() - started) * 1000.0

        return GazeRoiResult(
            sync=sync,
            gaze=gaze,
            objects=objects,
            roi_index=index,
            roi_region=roi_region,
            roi_point=point,
            regions=regions,
            roi_valid=bool(roi_region is not None and gaze.valid and self._trust_projection),
            road_ms=road_ms,
            driver_ms=driver_ms,
            frame_skew_ms=skew_ms,
        )

    def _point_for(self, gaze: GazeResult) -> Optional[Tuple[float, float]]:
        """Projected point from valid gaze, else from the raw (uncalibrated) pair."""
        if gaze is None:
            return None
        if gaze.valid:
            return self._projection.apply((gaze.x, gaze.y))
        raw = getattr(self._gaze, "last_raw", None)
        return self._projection.apply(raw) if raw is not None else None

    def close(self) -> None:
        self._pending_skew_ms = None
        for model in (self._detector, self._face, self._gaze):
            closer = getattr(model, "close", None)
            if closer is not None:
                closer()


from debug_screen import add_debug_args, open_debug_screen  # noqa: E402
from dgt.sources import open_source  # noqa: E402

SCREEN_W = 1280
INSET = 0.18


def _inset(canvas, image, corner, label, colour, fraction=INSET):
    """Draw `image` as a labelled inset in `corner` ("tr" or "br") of `canvas`."""
    import cv2

    if image is None or image.size == 0:
        return
    h, w = canvas.shape[:2]
    iw = int(w * fraction)
    ih = max(1, int(image.shape[0] * iw / image.shape[1]))
    if ih >= h - 40:
        ih = h // 3
        iw = max(1, int(image.shape[1] * ih / image.shape[0]))
    thumb = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA)

    margin = 12
    x1 = w - margin
    x0 = x1 - iw
    if corner == "tr":
        y0 = margin + 34            # clear of the banner
        y1 = y0 + ih
    else:
        y1 = h - margin
        y0 = y1 - ih
    if x0 < 0 or y0 < 0 or y1 > h or x1 > w:
        return
    canvas[y0:y1, x0:x1] = thumb
    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
    cv2.rectangle(canvas, (x0, y0 - 20), (x1, y0), colour, -1)
    cv2.putText(canvas, label, (x0 + 6, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)


def render(road, driver, result, class_colour, inset=INSET):
    """Road frame with grid, chosen tile, boxes, and driver/tile-crop insets."""
    import cv2

    canvas = road.copy()
    h, w = canvas.shape[:2]

    for i, (x0, y0, x1, y1) in enumerate(result.regions):
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (70, 70, 70), 1)
        cv2.putText(canvas, str(i), (x0 + 6, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 70), 1, cv2.LINE_AA)

    # Crop before the overlays are drawn on it.
    tile_crop = None
    if result.roi_region is not None:
        rx0, ry0, rx1, ry1 = result.roi_region
        tile_crop = road[ry0:ry1, rx0:rx1].copy()
        cv2.rectangle(canvas, (rx0, ry0), (rx1 - 1, ry1 - 1), (0, 200, 255), 3)

    for det in result.objects.detections:
        bx0, by0, bx1, by1 = det.box
        p0 = (int(bx0 * w), int(by0 * h))
        p1 = (int(bx1 * w), int(by1 * h))
        colour = class_colour(det.label)
        cv2.rectangle(canvas, p0, p1, colour, 2)
        cv2.putText(canvas, f"{det.label} {det.score:.2f}", (p0[0], max(14, p0[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if tile_crop is not None:
            rx0, ry0, _, _ = result.roi_region
            cv2.rectangle(tile_crop, (p0[0] - rx0, p0[1] - ry0),
                          (p1[0] - rx0, p1[1] - ry0), colour, 2)

    if result.roi_point is not None:
        gx, gy = result.roi_point
        centre = (int(gx * w), int(gy * h))
        cv2.circle(canvas, centre, 18, (0, 0, 255), 2)
        cv2.drawMarker(canvas, centre, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

    scale = SCREEN_W / w
    canvas = cv2.resize(canvas, (SCREEN_W, int(h * scale)))
    sh, sw = canvas.shape[:2]

    banner = ("UNCALIBRATED - the tile is NOT a measurement of where the driver looked"
              if not result.roi_valid else "calibrated")
    cv2.rectangle(canvas, (0, 0), (sw, 32), (0, 0, 0), -1)
    cv2.putText(canvas, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 165, 255) if not result.roi_valid else (0, 255, 0), 1, cv2.LINE_AA)

    _inset(canvas, driver, "tr", "driver", (0, 0, 255), inset)
    if result.roi_region is None:
        tile_label = "no region - gaze gave no point"
    elif result.roi_index is not None:
        tile_label = f"tile {result.roi_index} - inferred here"
    else:
        rx0, ry0, rx1, ry1 = result.roi_region
        tile_label = f"gaze crop {rx1 - rx0}x{ry1 - ry0} - inferred here"
    _inset(canvas, tile_crop, "br", tile_label, (0, 200, 255), inset)
    return canvas


def main(argv=None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--road-engine", type=Path, required=True)
    ap.add_argument("--face-engine", type=Path, required=True)
    ap.add_argument("--landmark-engine", type=Path, required=True)
    ap.add_argument("--gaze-engine", type=Path, required=True)
    ap.add_argument("--road-source", required=True)
    ap.add_argument("--driver-source", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--on-no-gaze", choices=("skip", "full"), default="skip",
                    help="what to infer when gaze gives no point (default: skip, "
                         "an explicit non-answer)")
    ap.add_argument("--no-loop-road", action="store_true")
    add_debug_args(ap)
    ap.add_argument("--save-dir", type=Path, default=Path("/tmp/gazeroi"))
    ap.add_argument("--save-every", type=int, default=30,
                    help="write every Nth frame as a JPEG; 0 = none")
    ap.add_argument("--inset", type=float, default=INSET, metavar="FRACTION",
                    help=f"inset width as a fraction of the screen "
                         f"(default {INSET}); 0 hides both insets")
    args = ap.parse_args(argv)

    from dgt.models.gaze import GazeEstimator
    from dgt.models.landmarks import FaceLandmarkDetector
    from dgt.models.road_objects import RoadObjectDetector

    thresholds = load_thresholds()
    try:
        road = RoadObjectDetector(
            engine_path=args.road_engine,
            conf_threshold=(args.conf if args.conf is not None
                            else thresholds.road_objects.conf_threshold),
            tile=thresholds.road_objects.tile,
        )
        face = FaceLandmarkDetector(
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
        print(f"[gazeroi] {exc}", file=sys.stderr)
        return 1
    projection = GazeProjection(
        scale_x=thresholds.gaze.roi_scale_x, scale_y=thresholds.gaze.roi_scale_y,
        offset_x=thresholds.gaze.roi_offset_x, offset_y=thresholds.gaze.roi_offset_y,
    )

    print(f"[gazeroi] road: {road.describe()}")
    print(f"[gazeroi] driver: {face.describe()}")
    print(f"[gazeroi] {gaze.describe()}")
    print(f"[gazeroi] projection {projection} — PLACEHOLDER, roi_valid stays False")
    print(f"[gazeroi] roi_margin={thresholds.gaze.roi_margin}")

    try:
        road_cap = open_source(args.road_source)
        driver_cap = open_source(args.driver_source)
    except FileNotFoundError as exc:
        print(f"[gazeroi] {exc}", file=sys.stderr)
        return 1
    for cap, name in ((road_cap, args.road_source), (driver_cap, args.driver_source)):
        if not cap.isOpened():
            print(f"[gazeroi] cannot open source: {name}", file=sys.stderr)
            return 1

    args.save_dir.mkdir(parents=True, exist_ok=True)
    screen = open_debug_screen(args, "gazeroi")

    palette = {}

    def class_colour(label):
        if label not in palette:
            rng = np.random.default_rng(abs(hash(label)) % (2 ** 32))
            palette[label] = tuple(int(v) for v in rng.integers(60, 255, 3))
        return palette[label]

    per_tile = {}
    per_class = {}
    road_ms, driver_ms, skipped = [], [], 0

    pipeline = GazeRoiPipeline(
        road, face, gaze, projection=projection, source_rate_hz=args.rate,
        on_no_gaze=args.on_no_gaze, roi_margin=thresholds.gaze.roi_margin,
    )
    try:
        while pipeline.frames < args.frames:
            ok_r, road_frame = road_cap.read()
            t_road = time.monotonic()
            if not ok_r:
                if args.no_loop_road:
                    break
                road_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            ok_d, driver_frame = driver_cap.read()
            t_driver = time.monotonic()
            if not ok_d:
                break

            result = pipeline.tick_pair(road_frame, driver_frame, t_road, t_driver)
            road_ms.append(result.road_ms)
            driver_ms.append(result.driver_ms)
            if result.roi_region is None:
                skipped += 1
            elif result.roi_index is not None:
                per_tile[result.roi_index] = per_tile.get(result.roi_index, 0) + 1
            for det in result.objects.detections:
                per_class[det.label] = per_class.get(det.label, 0) + 1

            view = render(road_frame, driver_frame, result, class_colour, args.inset)
            if screen:
                ok_jpeg, buf = cv2.imencode(".jpg", view,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpeg:
                    screen.publish(buf.tobytes())
            if args.save_every > 0 and result.sync.frame_id % args.save_every == 0:
                cv2.imwrite(str(args.save_dir / f"{result.sync.frame_id:04d}.jpg"), view)
            if pipeline.frames % 60 == 0:
                print(f"[gazeroi] {pipeline.frames} frames, region {result.roi_region}, "
                      f"road {result.road_ms:.0f} ms + driver {result.driver_ms:.0f} ms, "
                      f"{len(result.objects.detections)} boxes")
    finally:
        road_cap.release()
        driver_cap.release()
        pipeline.close()
        if screen:
            screen.close()

    n = max(1, len(road_ms))
    print(f"\n[gazeroi] {pipeline.frames} frames, {skipped} with no gaze point")
    print(f"[gazeroi] tiles chosen: {dict(sorted(per_tile.items()))}")
    print(f"[gazeroi] detections: {dict(sorted(per_class.items())) or 'none'}")
    print(f"[gazeroi] road {sum(road_ms)/n:.1f} ms + driver {sum(driver_ms)/n:.1f} ms "
          f"= {(sum(road_ms)+sum(driver_ms))/n:.1f} ms mean "
          f"-> {1000.0/max(1e-6,(sum(road_ms)+sum(driver_ms))/n):.1f} fps ceiling")
    if args.save_every > 0:
        print(f"[gazeroi] frames written to {args.save_dir}/")
    print("[gazeroi] NOTE: roi_valid=False on every frame. The region follows gaze, but "
          "the gaze->road mapping is a placeholder — see dgt/attention/gaze_roi.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
