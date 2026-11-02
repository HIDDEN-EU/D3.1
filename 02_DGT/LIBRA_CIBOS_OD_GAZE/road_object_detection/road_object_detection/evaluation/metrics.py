"""
Detection evaluation metrics.

All functions operate on plain numeric values — no Ultralytics dependency.
Boxes are expected in [x1, y1, x2, y2] absolute-pixel format.
"""

from typing import Sequence


Detection = tuple  # (confidence: float, is_true_positive: bool)


def compute_iou(
    box_pred: Sequence[float],
    box_true: Sequence[float],
) -> float:
    """Compute IoU between one predicted box and one ground truth box.

    Parameters
    ----------
    box_pred, box_true:
        Bounding boxes as [x1, y1, x2, y2] in absolute pixel coordinates.

    Returns
    -------
    float
        IoU in [0, 1]. Returns 0.0 if either box has zero area.
    """
    px1, py1, px2, py2 = box_pred
    gx1, gy1, gx2, gy2 = box_true

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_pred = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    area_true = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area_pred + area_true - intersection

    if union == 0.0:
        return 0.0

    return intersection / union


def compute_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP).

    Returns 0.0 when there are no predictions (TP + FP == 0).
    """
    total = tp + fp
    return tp / total if total > 0 else 0.0


def compute_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN).

    Returns 0.0 when there are no ground truth objects (TP + FN == 0).
    """
    total = tp + fn
    return tp / total if total > 0 else 0.0


def match_detections(
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Sequence[float],
    true_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> list:
    """Match predicted boxes of a single class to ground truth boxes by IoU.

    Implements the standard greedy assignment used before computing AP: predictions
    are considered in descending confidence order and each is matched to the
    highest-IoU ground truth box not already claimed, provided that IoU meets the
    threshold. This decides which predictions are true positives.

    Parameters
    ----------
    pred_boxes:
        Predicted boxes as [x1, y1, x2, y2] in absolute pixel coordinates.
    pred_scores:
        Confidence score for each predicted box (same order as ``pred_boxes``).
    true_boxes:
        Ground truth boxes as [x1, y1, x2, y2] in absolute pixel coordinates.
    iou_threshold:
        Minimum IoU for a prediction to count as a true positive (default 0.5).

    Returns
    -------
    list of (float, bool)
        One ``(confidence, is_true_positive)`` tuple per prediction, sorted by
        descending confidence. A ``False`` flag is a false positive. Ground truth
        boxes left unmatched are false negatives (use ``len(true_boxes)`` minus the
        number of true positives, or pass ``len(true_boxes)`` to
        :func:`compute_average_precision`).
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    matched_truth = [False] * len(true_boxes)
    results = []

    for i in order:
        best_iou = 0.0
        best_j = -1
        for j, true_box in enumerate(true_boxes):
            if matched_truth[j]:
                continue
            iou = compute_iou(pred_boxes[i], true_box)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        is_tp = best_iou >= iou_threshold and best_j >= 0
        if is_tp:
            matched_truth[best_j] = True
        results.append((pred_scores[i], is_tp))

    return results


def compute_average_precision(
    detections: Sequence[Detection],
    num_ground_truth: int,
) -> float:
    """Average Precision — area under the Precision–Recall curve for one class.

    Uses the all-points interpolation (COCO / VOC2010+): the precision envelope is
    made monotonically non-increasing before integrating, so the result is the exact
    area under the interpolated PR curve rather than the legacy 11-point sample.

    Parameters
    ----------
    detections:
        Sequence of ``(confidence, is_true_positive)`` tuples for a single class,
        typically the output of :func:`match_detections`. Order does not matter —
        they are sorted by confidence internally.
    num_ground_truth:
        Total number of ground truth boxes for this class (TP + FN). This anchors
        recall; missed detections are accounted for even though they never appear
        as predictions.

    Returns
    -------
    float
        AP in [0, 1]. Returns 0.0 if there are no ground truth boxes.
    """
    if num_ground_truth == 0:
        return 0.0
    if not detections:
        return 0.0

    ordered = sorted(detections, key=lambda d: d[0], reverse=True)

    # Cumulative TP / FP counts walking down the confidence-ranked predictions.
    recalls = [0.0]
    precisions = [1.0]
    tp = 0
    fp = 0
    for _, is_tp in ordered:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / num_ground_truth)
        precisions.append(tp / (tp + fp))

    # Make precision monotonically non-increasing (interpolation envelope).
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate precision over the recall axis.
    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap


def compute_map(average_precisions: Sequence[float]) -> float:
    """Mean Average Precision — the mean of per-class AP values.

    The headline metric for this project (``mAP@0.5`` gates KPI-6). The IoU threshold
    and any class subsetting (e.g. the VRU group rollup) are decided by the caller
    when building each AP; this function only averages them.

    Parameters
    ----------
    average_precisions:
        One AP value per class, e.g. the outputs of
        :func:`compute_average_precision`.

    Returns
    -------
    float
        Mean of the provided AP values, or 0.0 if the sequence is empty.
    """
    aps = list(average_precisions)
    return sum(aps) / len(aps) if aps else 0.0
