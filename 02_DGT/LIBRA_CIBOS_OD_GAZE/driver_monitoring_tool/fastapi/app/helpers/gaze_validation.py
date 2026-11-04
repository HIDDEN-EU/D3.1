"""Validation of incoming gaze messages before they enter the processing queue."""

import math

def parse_gaze_point(msg) -> tuple[dict | None, str | None]:
    """Extract x/y from a geometry_msgs/PointStamped message.
    Returns (point, None) if valid, or (None, reason) if not."""
    point = msg.get("point") if isinstance(msg, dict) else None
    if not isinstance(point, dict):
        return None, "missing 'point' field"
    values = {}

    for axis in ("x", "y"):
        raw = point.get(axis)
        if raw is None:
            return None, f"{axis} is null or missing"
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, f"{axis} is not numeric ({raw!r})"
        if math.isnan(value):
            return None, f"{axis} is NaN"
        if math.isinf(value):
            return None, f"{axis} is {'+' if value > 0 else '-'}inf"
        values[axis] = value

    return values, None

def is_zero_point(point: dict) -> bool:
    """(0, 0) is a valid gaze point, but a long run of them usually means the
    predictor is emitting faulty data."""
    return point["x"] == 0.0 and point["y"] == 0.0