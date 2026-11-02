"""
Unit tests for the count-based occlusion-recall view.

Pure-logic tests — no model, no dataset, no GPU. Runnable under pytest or standalone:

    python tests/test_occlusion_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_object_detection.evaluation import occlusion_eval as oe
from road_object_detection.evaluation.metrics import match_detections


def test_low_confidence_flag_below_floor():
    """AC-6: a cell with < LOW_CONFIDENCE_FLOOR GT boxes is flagged.

    The contract names the truncated x VRU cell (35 GT) as the single sparse cell.
    """
    assert oe.LOW_CONFIDENCE_FLOOR == 50
    cell = oe._recall_cell(total_gt=35, detected=10, conf=0.25)
    assert cell["low_confidence"] is True


def test_low_confidence_flag_at_floor():
    """AC-6: a cell with exactly LOW_CONFIDENCE_FLOOR GT boxes is NOT flagged."""
    cell = oe._recall_cell(total_gt=50, detected=10, conf=0.25)
    assert cell["low_confidence"] is False


def test_recall_cell_identity_and_recall():
    """AC-3: detected + missed == total_gt, and recall == detected / total."""
    cell = oe._recall_cell(total_gt=200, detected=80, conf=0.25)
    assert cell["detected"] + cell["missed"] == cell["total_gt"]
    assert cell["recall"] == 0.4
    assert cell["conf"] == 0.25


def test_recall_cell_empty_stratum():
    """A zero-GT cell has recall 0.0 and is flagged low_confidence (0 < 50)."""
    cell = oe._recall_cell(total_gt=0, detected=0, conf=0.001)
    assert cell["recall"] == 0.0
    assert cell["missed"] == 0
    assert cell["low_confidence"] is True


def test_match_gt_status_agrees_with_match_detections():
    """AC-11b guard: _match_gt_status's TP count must equal match_detections's TP count
    on the same inputs (the galleries reconcile with the JSON counts by construction)."""
    preds = [[0, 0, 10, 10], [20, 20, 30, 30], [0, 0, 9, 9], [100, 100, 110, 110]]
    scores = [0.9, 0.8, 0.7, 0.6]
    truth = [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]]

    det = match_detections(preds, scores, truth, oe.RECALL_IOU)
    md_tp = sum(1 for _, is_tp in det if is_tp)

    status = oe._match_gt_status(preds, scores, truth, oe.RECALL_IOU)
    gt_tp = sum(1 for s in status if s is not None)

    assert gt_tp == md_tp
    # GT box 0 and 1 are exact matches -> found; GT box 2 has no overlapping pred -> missed.
    assert status[0] is not None and status[1] is not None and status[2] is None
    # A found box carries the matching prediction's confidence.
    assert status[0] == 0.9


def test_match_gt_status_no_predictions_all_missed():
    """No predictions -> every GT box is a miss (FN)."""
    truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
    status = oe._match_gt_status([], [], truth, oe.RECALL_IOU)
    assert status == [None, None]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
