"""Connection states shared by gaze sources and the rosbridge publisher."""

from enum import Enum

class ConnectionStatus(str, Enum):
    """Lifecycle state of a transport connection, reported in health checks."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    ERROR = "error"
    INVALID_DATA = "invalid_data"