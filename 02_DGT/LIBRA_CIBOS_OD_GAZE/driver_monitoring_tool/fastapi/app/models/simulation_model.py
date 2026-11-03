"""Simulated gaze predictor used in place of rosbridge for local development
and for exercising the invalid-data path via SIMULATION_FAULT."""

import asyncio
import logging
import random

import numpy as np

from app.config import Settings
from app.transport.connection_status import ConnectionStatus
from app.transport.gaze_source import GazeSource

logger = logging.getLogger(__name__)

# Values emitted by SIMULATION_FAULT, for testing the invalid-data path.
FAULT_VALUES = {
    "nan": np.nan,
    "inf": np.inf,
    "zero": 0.0,
}

class GazePredictorSimulation(GazeSource):
    """Gaze source that emits random (or deliberately faulty) points every 0.2s."""
    def __init__(self, queue: asyncio.Queue):
        super().__init__(queue)
        self.fault = Settings().simulation_fault

    async def start(self):
        """Background task: fake a connection, then publish points until stopped, retrying on errors."""
        self._running = True
        while self._running:
            try:
                self._set_status(ConnectionStatus.CONNECTING)

                # Mimics the startup delay of a real rosbridge connection.
                await asyncio.sleep(3)
                self._on_subscribed()

                while self._running:
                    await self._gaze_prediction_simulator()

            except asyncio.CancelledError:
                break

            except Exception as e:
                self._set_status(ConnectionStatus.ERROR, error=str(e))
                logger.exception("Simulation error")

                if self._running:
                    await asyncio.sleep(10)

        self._set_status(ConnectionStatus.DISCONNECTED)

    async def _gaze_prediction_simulator(self):
        """Publish one simulated PointStamped-shaped message, then wait 0.2s."""
        x, y = self._get_point()
        self._handle_publish({'point': {'x': x, 'y': y}})
        await asyncio.sleep(0.2)

    def _get_point(self) -> tuple[float, float]:
        """Return the configured fault value for both axes, or a random point in [-1, 1]."""
        if self.fault in FAULT_VALUES:
            value = FAULT_VALUES[self.fault]
            return value, value
        return random.uniform(-1, 1), random.uniform(-1, 1)
