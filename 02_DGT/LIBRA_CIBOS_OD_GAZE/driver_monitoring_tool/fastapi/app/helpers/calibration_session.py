"""Calibration session: collects gaze samples per reference point and computes
the homography that maps raw gaze predictions onto screen coordinates.

TODO: works separately from ../edge_inference/dgt/calibration.py for now; combine the two."""

from app.helpers.errors import point_not_found
import numpy as np
import numpy.typing as npt
import pickle
from app.helpers.calibration_point import CalibrationPoint
import cv2
import time

from app.config import Settings

class CalibrationSession:
    """Tracks the calibration points, the active point and the resulting matrix."""
    FOCUS_DELAY = 1.0  # seconds to ignore gaze samples after a point is (re)selected

    def __init__(self):
        self.data: list[CalibrationPoint] = []
        self.current_point: CalibrationPoint | None = None
        self.transformation_matrix = None
        self._focus_until = 0.0
        self._ignore = False

    def intialize_points(self, points: npt.NDArray[np.float32]):
        """Replace the session's points with fresh CalibrationPoints for the given targets."""
        self.data = [CalibrationPoint(target_point=p) for p in points]

    def find_current_point(self, target_point: npt.NDArray[np.float32]) -> CalibrationPoint | None:
        """Return the point whose target exactly matches target_point, or None."""
        return next(
            (p for p in self.data if np.array_equal(p.target_point, target_point)),
            None
        )

    def set_current_point(self, target_point):
        """Make target_point the active point, clear its samples and start the focus delay.

        Raises PointNotFoundError if no reference point matches.
        """
        point = self.find_current_point(target_point=target_point)
        if point is None:
            self.current_point = None
            raise point_not_found(target_point)
        self.current_point = point
        self.current_point.reset()
        self.wait_for_focus()

    def wait_for_focus(self):
        """Ignore incoming samples for FOCUS_DELAY seconds while the user's eyes settle."""
        self._focus_until = time.monotonic() + self.FOCUS_DELAY

    @property
    def idle(self) -> bool:
        """True while still inside the focus delay window."""
        return time.monotonic() < self._focus_until

    def set_ignore(self, value: bool) -> None:
        """Pause (True) or resume (False) sample collection."""
        self._ignore = value

    def add_predicted_point(self, predicted_point: npt.NDArray[np.float32]):
        """Record a gaze sample for the active point unless paused, idle or no point is set."""
        if self._ignore:
            return

        if self.current_point is None:
            return

        if self.idle:
            return

        self.current_point.add_prediction(point=predicted_point)

    def calculate_transformation_matrix(self):
        """Compute and store the homography from aggregated gaze points to target points.

        Raises ValueError if points are missing samples or the homography is invalid.
        """
        if not self.data:
            raise ValueError("No calibration points have been initialized.")

        aggregated_points = [d.get_aggregated_point() for d in self.data]
        missing = [i for i, p in enumerate(aggregated_points) if p is None]
        if missing:
            raise ValueError(
                f"No gaze samples were collected for calibration point(s) {missing};"
            )

        target_points = np.array([d.target_point for d in self.data], dtype=np.float32)
        reference_points = np.array(aggregated_points, dtype=np.float32)
        self.transformation_matrix, _ = cv2.findHomography(reference_points, target_points, method=0)
        if self.transformation_matrix is None or not np.isfinite(self.transformation_matrix).all():
            self.transformation_matrix = None
            raise ValueError("Could not compute a valid homography from the collected points.")

        return self.transformation_matrix

    def save_transformation_matrix(self):
        """Pickle the computed matrix to the configured calibration file path."""
        if self.transformation_matrix is None:
            raise ValueError("No transformation matrix to save; calculate it first.")
        s = Settings()
        with open(s.calibration_matrix_path, 'wb') as f:
            pickle.dump(self.transformation_matrix, f)