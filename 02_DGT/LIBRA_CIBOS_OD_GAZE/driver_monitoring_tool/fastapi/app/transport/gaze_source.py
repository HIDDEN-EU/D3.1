"""Base class for gaze data producers (rosbridge subscriber, simulation)."""

import asyncio
import logging
import time

from app.transport.connection_status import ConnectionStatus
from app.helpers.gaze_validation import is_zero_point, parse_gaze_point
from app.helpers.bounded_queue import put_drop_oldest

logger = logging.getLogger(__name__)


class GazeSource:
    """
    Shared status tracking and gaze validation for anything that feeds gaze
    points into the processing queue. Subclasses implement start(), call
    _on_subscribed() once data starts flowing and pass each raw gaze message
    to _handle_publish().
    """
    INVALID_THRESHOLD = 10

    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self._running = False

        self._status = ConnectionStatus.DISCONNECTED
        self._last_error = None
        self._last_message_time = None
        self._consecutive_invalid = 0

    async def start(self):
        """Run the source until stopped; must be implemented by subclasses."""
        raise NotImplementedError

    async def stop(self):
        """Signal the start() loop to exit."""
        self._running = False

    @property
    def status(self):
        """Current ConnectionStatus."""
        return self._status

    def _set_status(self, status, error=None):
        """Update the status and the last error (cleared when error is None)."""
        self._status = status
        self._last_error = error

    def is_healthy(self) -> bool:
        """True when subscribed and receiving valid data."""
        return self._status == ConnectionStatus.SUBSCRIBED

    def get_status_report(self) -> dict:
        """Return status, last error, time since last valid message and health."""
        return {
            "status": self._status.value,
            "last_error": self._last_error,
            "seconds_since_last_message": (
                time.monotonic() - self._last_message_time
                if self._last_message_time else None
            ),
            "healthy": self.is_healthy(),
        }

    def _on_subscribed(self) -> None:
        """Mark the source as subscribed and reset the invalid-message counter."""
        self._consecutive_invalid = 0
        self._set_status(ConnectionStatus.SUBSCRIBED)

    def _handle_publish(self, msg) -> None:
        """Validate a raw gaze message and enqueue it, tracking invalid/zero runs.

        Zero points are kept, but a long run of them still marks the source unhealthy.
        """
        point, reason = parse_gaze_point(msg)
        if point is None:
            self._record_invalid(reason or "Invalid data")
            return

        self._last_message_time = time.monotonic()
        put_drop_oldest(self.queue, point)

        if is_zero_point(point):
            self._record_invalid("x and y are zero")
            return

        if self._status == ConnectionStatus.INVALID_DATA:
            self._set_status(ConnectionStatus.SUBSCRIBED)
        self._consecutive_invalid = 0

    def _record_invalid(self, reason: str) -> None:
        """Count an invalid message; switch to INVALID_DATA once INVALID_THRESHOLD is hit."""
        self._consecutive_invalid += 1
        if self._consecutive_invalid == self.INVALID_THRESHOLD:
            logger.warning("%s: receiving invalid messages (%s)", self._source_name(), reason)
            self._set_status(ConnectionStatus.INVALID_DATA, f"Receiving invalid data: {reason}")

    def _source_name(self) -> str:
        """Name used in log messages; defaults to the class name."""
        return type(self).__name__
