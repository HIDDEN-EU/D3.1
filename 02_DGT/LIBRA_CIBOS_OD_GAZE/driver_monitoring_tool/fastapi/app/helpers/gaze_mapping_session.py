"""Gaze-mapping session: applies the saved calibration homography to gaze
samples collected for each reference point to predict screen positions."""

import logging
from typing import List, Optional

import cv2
import numpy as np
import numpy.typing as npt

from app.helpers.calibration_file import load_calibration_matrix
from app.helpers.calibration_point import CalibrationPoint
from app.helpers.errors import point_not_found
from app.config import Settings

logger = logging.getLogger(__name__)


class GazeMappingSession:
    """Tracks reference points and maps their aggregated gaze through the calibration matrix."""
    data: List[CalibrationPoint]
    current_point: Optional[CalibrationPoint]

    def __init__(self):
        self.data = []
        self.current_point = None
        self.transformation_matrix = None
        self._matrix_problem_reported = False

    def find_current_point(self, target_point: npt.NDArray[np.float32]) -> Optional[CalibrationPoint]:
        """Return the point whose target exactly matches target_point, or None."""
        return next(
            (p for p in self.data if np.array_equal(p.target_point, target_point)),
            None
        )

    def set_current_point(self, target_point: npt.NDArray[np.float32]):
        """Make target_point the active point; raises PointNotFoundError if unknown."""
        point = self.find_current_point(target_point=target_point)
        if point is None:
            self.current_point = None
            raise point_not_found(target_point)
        self.current_point = point

    def add_predicted_point(self, predicted_point: npt.NDArray[np.float32]):
        """Record a gaze sample for the active point, if one is set."""
        if self.current_point is None:
            return
        self.current_point.add_prediction(point=predicted_point)

    def get_prediction(self) -> Optional[npt.NDArray[np.float32]]:
        """Return the active point's median gaze mapped to screen coordinates, or None.

        Lazily loads the calibration matrix on first use.
        """
        if self.transformation_matrix is None:
            self.import_transformation_matrix()

        if self.transformation_matrix is None or self.current_point is None:
            return None

        agg_point = self.current_point.get_aggregated_point()
        if agg_point is None:
            return None

        agg_point = np.array([agg_point], dtype=np.float32)
        return cv2.perspectiveTransform(agg_point, self.transformation_matrix)[0][0]

    def end_of_session(self) -> bool:
        """True when the active point is the last reference point."""
        if not self.data or self.current_point is None:
            return False
        return np.array_equal(self.data[-1].target_point, self.current_point.target_point)

    def import_transformation_matrix(self):
        """Load the calibration matrix, or None if it's missing or unusable,
        so a bad file doesn't turn every prediction into a 500."""
        path = Settings().calibration_matrix_path
        matrix, problem = load_calibration_matrix(path)
        if problem:
            self._report_matrix_problem(problem)
        self.transformation_matrix = matrix

    def _report_matrix_problem(self, msg):
        """Log a matrix loading problem only once per session to avoid log spam."""
        if not self._matrix_problem_reported:
            logger.warning("%s", msg)
            self._matrix_problem_reported = True
