import asyncio
import json
import logging
import time

import numpy as np
import websockets

from app.transport.connection_status import ConnectionStatus
from app.config import Settings

logger = logging.getLogger(__name__)

class RosbridgePublisher:
    """
    Keeps a persistent, auto-reconnecting connection to rosbridge and
    advertises a topic for the lifetime of the app, so consumers can
    subscribe to it at any point. publish_matrix() sends the calibration
    matrix once a calibration is completed.
    """

    def __init__(self, rosbridge_url, topic, message_type="std_msgs/Float64MultiArray"):
        self.rosbridge_url = rosbridge_url
        self.topic = topic
        self.message_type = message_type
        self._running = False
        self._websocket = None

        self._status = ConnectionStatus.DISCONNECTED
        self._last_error = None
        self._last_publish_time = None

    @property
    def status(self):
        return self._status

    def _set_status(self, status, error=None):
        self._status = status
        self._last_error = error

    def is_healthy(self) -> bool:
        return self._status == ConnectionStatus.SUBSCRIBED and self._websocket is not None

    def get_status_report(self) -> dict:
        return {
            "status": self._status.value,
            "last_error": self._last_error,
            "seconds_since_last_publish": (
                time.monotonic() - self._last_publish_time
                if self._last_publish_time else None
            ),
            "healthy": self.is_healthy(),
        }

    async def start(self):
        """
        Background task: connects to rosbridge, advertises the topic,
        and keeps the connection alive, reconnecting (and re-advertising)
        automatically if it drops.
        """
        self._running = True
        s = Settings()

        while self._running:
            try:
                self._set_status(ConnectionStatus.CONNECTING)
                logger.info("Connecting to rosbridge: %s", self.rosbridge_url)

                async with websockets.connect(self.rosbridge_url) as websocket:
                    self._websocket = websocket
                    self._set_status(ConnectionStatus.CONNECTED)
                    logger.info("Connected to rosbridge")

                    await websocket.send(json.dumps({
                        "op": "advertise",
                        "topic": self.topic,
                        "type": self.message_type,
                    }))
                    self._set_status(ConnectionStatus.SUBSCRIBED)  # reused to mean "advertised"
                    logger.info("Advertised %s as %s", self.topic, self.message_type)

                    while self._running:
                        try:
                            await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue

            except asyncio.CancelledError:
                self._set_status(ConnectionStatus.DISCONNECTED)
                break

            except Exception as e:
                self._set_status(ConnectionStatus.ERROR, error=str(e))
                logger.error("Rosbridge publisher error: %s", e)
                self._websocket = None

                if self._running:
                    await asyncio.sleep(s.ros_reconnect_delay)

        self._websocket = None
        self._set_status(ConnectionStatus.DISCONNECTED)

    async def publish_matrix(self, matrix: np.ndarray | None):
        """Publish a homography as std_msgs/Float64MultiArray"""
        if matrix is None:
            raise ValueError("No matrix to publish")

        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"Found invalid homography (shape={matrix.shape}) - not publishing")

        if self._websocket is None:
            raise RuntimeError(f"Cannot publish to {self.topic}: not connected to rosbridge")

        await self._websocket.send(json.dumps({
            "op": "publish",
            "topic": self.topic,
            "msg": {"data": matrix.ravel().tolist()},
        }))
        self._last_publish_time = time.monotonic()

    async def stop(self):
        self._running = False