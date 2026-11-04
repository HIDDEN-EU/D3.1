"""Background consumer of the gaze queue that feeds samples into the active
calibration or gaze-mapping session and exposes session control to the routes."""

import asyncio
import logging
from app.helpers.errors import NoActiveSessionError
import numpy as np
from app.helpers.calibration_point import CalibrationPoint
from app.helpers.calibration_session import CalibrationSession
from app.helpers.gaze_mapping_session import GazeMappingSession

logger = logging.getLogger(__name__)

class DataProcessor:
    """Owns the processing task and the current session (calibration or mapping)."""
    def __init__(self, data_queue):
        self.processor_task: asyncio.Task | None = None
        self.queue = data_queue
        self.session: CalibrationSession | GazeMappingSession | None = None

    def start(self):
        """Flush stale samples and (re)start the processing task.

        A running task is replaced rather than rejected, since a client may leave
        without calling stop (e.g. browser tab closed). Samples queued while stopped
        are dropped so they aren't attributed to the new point.
        """
        task = self.processor_task
        if task is not None and not task.done():
            task.cancel()

        while not self.queue.empty():
            self.queue.get_nowait()
        self.processor_task = asyncio.create_task(self.processor())

        return

    async def stop(self):
        """Cancel the processing task and wait for it to finish.

        No-op when already stopped, so repeated stop requests are safe.
        """
        task = self.processor_task
        if task is None or task.done():
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        return

    async def pause(self):
        """Stop recording samples in the calibration session until the next point update.

        Pausing while stopped is allowed, as no samples are being collected anyway.
        """
        if not isinstance(self.session, CalibrationSession):
            raise NoActiveSessionError("No active calibration session; call /points/reference/initialize first.")

        self.session.set_ignore(True)

    def status(self):
        """Return True while the processing task is running."""
        return bool(self.processor_task) and not self.processor_task.done()

    async def processor(self):
        """Loop forever, pulling samples from the queue and processing each one.

        A sample that fails is logged and skipped so it doesn't end the session.
        """
        while True:
            data = await self.queue.get()
            if data is None:
                continue
            try:
                self.process(data)
            except Exception:
                logger.exception('Failed to process sample %r', data)

    def process(self, data):
        """Forward one {"x", "y"} sample to the active session."""
        if self.session is None:
            return
        x = data["x"]
        y = data["y"]
        self.session.add_predicted_point(np.array([[x, y]], dtype=np.float32))

    def reset_points(self, calibration: bool) -> CalibrationSession | GazeMappingSession:
        """Replace the current session with a new calibration or gaze-mapping session."""
        self.session = CalibrationSession() if calibration else GazeMappingSession()
        return self.session

    def initialize_ref_points(self, calibration, n_cols = 12, n_rows = 6, num_points = 10):
        """Start processing, create a new session and return its reference points.

        Calibration uses a fixed 3x3 grid; gaze mapping uses num_points random points.
        """
        self.start()
        session = self.reset_points(calibration=calibration)

        if calibration:
            points = self._get_initial_points(n_cols=n_cols, n_rows=n_rows)
        else:
            self.num_points = num_points
            points = self._get_random_points(n_cols=n_cols, n_rows=n_rows)

        session.data = [CalibrationPoint(target_point=point) for point in points]
        return points

    def update_ref_point(self, point):
        """Switch the session's active point to the given {x_value, y_value} target."""
        if self.session is None:
            raise NoActiveSessionError("No active session; call /points/reference/initialize first.")

        if isinstance(self.session, CalibrationSession):
            self.session.set_ignore(False)

        rx = float(point['x_value'])
        ry = float(point['y_value'])
        self.session.set_current_point(np.array([rx, ry]))

    async def make_calculations(self):
        """Return the mapped gaze point for the active point; stops after the last point."""
        if not isinstance(self.session, GazeMappingSession):
            return None

        if self.session.end_of_session():
            await self.stop()

        return self.session.get_prediction()

    async def calculate_transformation_matrix(self):
        """Stop processing and compute the calibration homography."""
        if not isinstance(self.session, CalibrationSession):
            raise ValueError(
                "Transformation matrix can only be calculated during a calibration session."
            )

        await self.stop()
        return self.session.calculate_transformation_matrix()

    def save_transformation_matrix(self):
        """Persist the calibration session's homography to disk."""
        if not isinstance(self.session, CalibrationSession):
            raise ValueError(
                "Transformation matrix can only be saved during a calibration session."
            )
        self.session.save_transformation_matrix()

    def _get_initial_points(self, n_cols, n_rows):
        """Return the 9 calibration targets: corners, edge midpoints and centre of the grid."""
        initial_points_arr = np.array([
                [0, 0],
                [n_cols // 2, 0],
                [n_cols, 0],
                [0, n_rows // 2],
                [n_cols // 2, n_rows // 2],
                [n_cols, n_rows // 2],
                [0, n_rows],
                [n_cols // 2, n_rows],
                [n_cols, n_rows],
            ], dtype=np.float32)

        return initial_points_arr

    def _get_random_points(self, n_cols, n_rows):
        """Return num_points uniformly random targets within the grid bounds."""
        x_vals = np.random.uniform(0, n_cols, self.num_points)
        y_vals = np.random.uniform(0, n_rows, self.num_points)
        initial_points_arr = np.stack((x_vals, y_vals), axis=1).astype(np.float32)

        return initial_points_arr
