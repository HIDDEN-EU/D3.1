"""Rosbridge subscriber that streams gaze points from a ROS topic into the queue."""

import asyncio
import json
import logging

import websockets

from app.transport.connection_status import ConnectionStatus
from app.transport.gaze_source import GazeSource
from app.config import Settings

logger = logging.getLogger(__name__)

class RosbridgeSubscriber(GazeSource):
    """Gaze source backed by a rosbridge websocket subscription."""
    def __init__(self, rosbridge_url, topic, message_type, queue):
        super().__init__(queue)
        self.rosbridge_url = rosbridge_url
        self.topic = topic
        self.message_type = message_type

    async def start(self):
        """Background task: connect, subscribe and forward messages, reconnecting on failure."""
        self._running = True
        s = Settings()
        while self._running:
            try:
                self._set_status(ConnectionStatus.CONNECTING)

                async with websockets.connect(self.rosbridge_url) as websocket:
                    self._set_status(ConnectionStatus.CONNECTED)

                    await websocket.send(json.dumps({
                        "op": "subscribe",
                        "topic": self.topic,
                        "type": self.message_type,
                    }))

                    self._on_subscribed()

                    async for message in websocket:
                        if not self._running:
                            break
                        self._handle_message(message)

                    if self._running:
                        raise ConnectionError("rosbridge closed the connection")

            except asyncio.CancelledError:
                self._set_status(ConnectionStatus.DISCONNECTED)
                break

            except Exception as e:
                self._set_status(ConnectionStatus.ERROR, error=str(e))
                logger.error("Rosbridge error: %s", e)

                if self._running:
                    await asyncio.sleep(s.ros_reconnect_delay)

        self._set_status(ConnectionStatus.DISCONNECTED)

    def _handle_message(self, message) -> None:
        """Decode a rosbridge frame and pass "publish" payloads to _handle_publish()."""
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            self._record_invalid("message is not valid JSON")
            return
        if not isinstance(data, dict):
            self._record_invalid("message is not a JSON object")
            return
        if data.get("op") == "publish":
            self._handle_publish(data.get("msg"))

    def _source_name(self) -> str:
        """Use the topic name in log messages."""
        return self.topic
