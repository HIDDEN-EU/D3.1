#!/usr/bin/env bash
# Source ROS 2 Humble, then become the command. Without the source rclpy is not
# importable and dgt-daemon exits: ROS is its only output. `exec` matters, as
# in deploy/dgt.service: the SIGTERM from `docker stop` must reach the daemon,
# or the capture handle is never released.
set -e
# setup.bash reads unset variables, so it is sourced before any `set -u`.
source /opt/ros/humble/setup.bash
exec "$@"
