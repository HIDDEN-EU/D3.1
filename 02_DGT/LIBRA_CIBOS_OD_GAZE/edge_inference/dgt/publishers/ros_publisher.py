"""ROS 2 publisher for DGT_DS messages — the daemon's only output.

A publisher-only node. DGT_DS is a nested JSON document (docs/DGT_DS.md), not
a point, so it goes out as `std_msgs/String` on `{prefix}/status`. That needs no
custom message package, and rosbridge relays it as-is.
"""
from __future__ import annotations


class DgtRosPublisher:
    """Publisher-only ROS 2 node for one DGT_DS message per frame.

    Publishes std_msgs/String on {prefix}/status, the DGT_DS JSON in `data`.

    If rclpy is not importable (ROS not sourced, or a non-Jetson dev machine)
    the instance stays disabled and `publish` is a no-op; the daemon refuses to
    run on a disabled publisher, since it has no other output. No executor is
    spun — a publisher-only node does not need one.
    """

    def __init__(self, topic_prefix: str = "/dgt", best_effort: bool = True):
        self.enabled = False
        self.node = None
        self.topic = f"{topic_prefix}/status"

        try:
            import rclpy
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from std_msgs.msg import String
        except ImportError:
            return

        self._rclpy = rclpy
        self._string = String

        rclpy.init()
        self.node = rclpy.create_node("dgt_publisher")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT if best_effort else ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.status_publisher = self.node.create_publisher(String, self.topic, qos)
        self.enabled = True
        self.qos_name = "BEST_EFFORT" if best_effort else "RELIABLE"

    def publish(self, message_json: str) -> None:
        if self.enabled:
            msg = self._string()
            msg.data = message_json
            self.status_publisher.publish(msg)

    def shutdown(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
            self._rclpy.shutdown()
        self.enabled = False
