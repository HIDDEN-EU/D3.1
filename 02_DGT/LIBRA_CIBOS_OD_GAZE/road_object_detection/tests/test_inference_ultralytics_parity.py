"""
Parity: `road_object_detection.inference` against Ultralytics itself.

The NumPy re-statement of letterbox / NMS / box scaling is only worth having if it is
the same arithmetic every metric in this repo was measured through. These tests run
both on the same inputs and compare. They need the `[train]` extra (torch +
ultralytics) and skip without it — the NumPy-only tests are in `test_inference.py`.

Heavy imports stay inside the tests so collecting this file does not load torch
into the process for the files collected after it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_object_detection import inference as inf


@pytest.mark.parametrize("shape", [(360, 640), (720, 1280), (300, 500), (361, 641)])
def test_letterbox_matches_ultralytics(shape):
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import LetterBox

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    ref = LetterBox(new_shape=(inf.IMGSZ, inf.IMGSZ), auto=False)(image=image)
    ref_blob = np.ascontiguousarray(ref[:, :, ::-1].transpose(2, 0, 1)[None],
                                    dtype=np.float32) / 255.0
    blob, _, _ = inf.letterbox(image)
    assert blob.shape == ref_blob.shape
    assert np.array_equal(blob, ref_blob)


@pytest.mark.parametrize("seed", range(5))
def test_postprocess_matches_ultralytics_nms(seed):
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics.utils.nms import non_max_suppression

    rng = np.random.default_rng(seed)
    n, nc = 2000, 9
    xy = rng.uniform(0, 640, (n, 2))
    wh = rng.uniform(4, 200, (n, 2))
    scores = rng.uniform(0, 1, (n, nc)) ** 4               # mostly low, some confident
    raw = np.concatenate([xy, wh, scores], 1).T[None].astype(np.float32)

    ref = non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=inf.CONF,
                              iou_thres=inf.TILE_IOU, max_det=inf.MAX_DET)[0].numpy()
    boxes, conf, ids = inf.postprocess(raw)

    assert len(boxes) == len(ref)
    a = np.concatenate([boxes, conf[:, None], ids[:, None]], 1)
    key = lambda m: np.lexsort((m[:, 0], m[:, 1], m[:, 4]))  # noqa: E731
    assert np.allclose(a[key(a)], ref[key(ref)], atol=1e-4)


def test_scale_boxes_matches_ultralytics():
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics.utils.ops import scale_boxes

    rng = np.random.default_rng(0)
    tile = (360, 640)
    boxes = rng.uniform(-20, 660, (100, 4)).astype(np.float32)
    ref = scale_boxes((inf.IMGSZ, inf.IMGSZ), torch.from_numpy(boxes.copy()), tile).numpy()
    _, gain, pad = inf.letterbox(np.zeros((*tile, 3), np.uint8))
    assert np.allclose(inf.scale_boxes(boxes, gain, pad, tile), ref, atol=1e-4)


def test_cross_tile_merge_matches_torchvision():
    """Replaces `torchvision.ops.batched_nms` in the tiled viewers."""
    torchvision = pytest.importorskip("torchvision")
    import torch

    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1200, (500, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(4, 150, (500, 2))], 1).astype(np.float32)
    scores = rng.uniform(0, 1, 500).astype(np.float32)
    ids = rng.integers(0, 9, 500)
    ref = torchvision.ops.batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
                                      torch.from_numpy(ids), inf.MERGE_IOU).numpy()
    assert inf.merge_tiles(boxes, scores, ids).tolist() == sorted(ref.tolist())
