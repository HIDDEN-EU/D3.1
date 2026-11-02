"""dgt — the T3.2 DGT runtime for the HIDDEN project.

Driver Gaze Tracking & status monitoring: consumes model inference outputs
(gaze, road objects, emotion) and produces driver-attentiveness signals. It is
an INFERENCE layer — it never trains, and it contains no stand-in, dummy or
simulated data path.

Deliberately import-light: `import dgt` pulls nothing. Reach for what you need
(`dgt.contracts`, `dgt.attention`, `dgt.models`, `dgt.publishers`).

What runs on a device is `dgt-daemon` (`dgt.daemon`), which publishes the
DGT_DS message described in docs/DGT_DS.md on ROS 2.
"""

__version__ = "0.1.0"
