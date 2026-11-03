"""A single on-screen reference point and the gaze predictions collected for it."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import numpy.typing as npt

@dataclass
class CalibrationPoint:
    """A target point paired with the raw gaze predictions recorded while it was shown."""
    target_point: npt.NDArray[np.float32]
    prediction_points: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )

    def add_prediction(self, point: npt.NDArray[np.float32]):
        """Append a (1, 2) gaze prediction to the collected samples."""
        self.prediction_points = np.append(self.prediction_points, point, axis=0)

    def get_aggregated_point(self) -> Optional[npt.NDArray[np.float32]]:
        """Return the per-axis median of the samples as a (1, 2) array, or None if empty."""
        if self.prediction_points.shape[0] == 0:
            return None
        return np.array([np.median(self.prediction_points, axis=0)], dtype=np.float32)

    def reset(self) -> None:
        """Discard all collected predictions."""
        self.prediction_points = np.empty((0, 2), dtype=np.float32)

    @property
    def sample_count(self) -> int:
        """Number of predictions collected so far."""
        return int(self.prediction_points.shape[0])