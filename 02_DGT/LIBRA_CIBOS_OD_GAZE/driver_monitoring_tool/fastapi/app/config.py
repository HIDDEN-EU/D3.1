from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from the environment and .env.

    Unknown keys are rejected, so typos in .env fail at startup.
    publish_in_simulation is off by default: simulated gaze is random, so a
    calibration computed from it is noise and publishing it would overwrite a
    real calibration. simulation_fault makes the simulator emit only invalid
    points, to test the invalid-data status.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    simulation_mode: bool = False
    publish_in_simulation: bool = False
    simulation_fault: Literal["none", "nan", "inf", "zero"] = "none"
    calibration_matrix_path: str = "calibration.pkl"
    host: str = "0.0.0.0"
    port: int = 8000
    rosbridge_url: str = "ws://192.168.1.203:9090"
    ros_topic: str = "/dgt/gaze/raw"
    ros_message_type: str = "geometry_msgs/PointStamped"
    ros_calibration_topic: str = "/dgt/calibration_matrix"
    ros_calibration_message_type: str = "std_msgs/Float64MultiArray"
    ros_reconnect_delay: int = 3
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

