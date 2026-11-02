"""The daemon and its publishers with stand-ins: no engine, GPU, camera or ROS."""
from __future__ import annotations

import json
import sys
import types
import uuid

import pytest

from dgt.config.loader import load_service
from dgt.contracts.detections import Detection, DetectionResult
from dgt.daemon import UnitSettings, main
from dgt.publishers.dgt_ds import MessageEnvelope, assemble_status, to_pixels
from dgt.publishers.ros_publisher import DgtRosPublisher
from dgt.sources import open_source

CONFIG = load_service()
CAR = Detection(label="car", class_id=2, score=0.9, box=(0.25, 0.5, 0.5, 1.0))


def status(objects):
    return assemble_status(objects, 1280, 720, 1_700_000_000.25, "road_camera").to_dict()


def test_detections_reach_the_wire_in_pixels_with_the_capture_time():
    s = status(DetectionResult(detections=[CAR], taxonomy="bdd100k-v2", valid=True))
    assert s["timestamp"] == 1_700_000_000.25
    assert s["driver_att_od"] == {
        "frame_id": "road_camera",
        "gaze_area_px": None,
        "detections": [{"class_id": 2, "class_name": "car",
                        "bbox_px": [320, 360, 640, 720], "confidence": 0.9}],
    }
    for key in ("emotion_status", "level_of_attention", "distracted", "drowsy",
                "confidence"):
        assert s[key] is None


def test_a_detector_that_did_not_run_is_null_not_an_empty_road():
    assert status(DetectionResult(valid=False))["driver_att_od"] is None


def test_no_objects_is_a_real_empty_list():
    s = status(DetectionResult(detections=[], taxonomy="bdd100k-v2", valid=True))
    assert s["driver_att_od"]["detections"] == []


def test_boxes_are_clamped_to_the_frame():
    assert to_pixels((-0.1, 0.0, 1.2, 1.0), 100, 50) == (0, 0, 100, 50)


def test_the_envelope_counts_up_and_never_repeats_an_id():
    envelope = MessageEnvelope(CONFIG, "dgt-test", clock=lambda: 5.0)
    empty = assemble_status(DetectionResult(valid=False), 1, 1, 1.0, "road_camera")
    metas = [envelope.wrap(empty).metadata for _ in range(3)]
    assert [m.sequence_number for m in metas] == [0, 1, 2]
    assert len({m.message_id for m in metas}) == 3
    uuid.UUID(metas[0].message_id)
    assert (metas[0].message_type, metas[0].sender_id, metas[0].send_timestamp) == \
        ("DGT_DS", "dgt-test", 5.0)


def test_an_empty_sender_id_is_refused():
    with pytest.raises(ValueError, match="DGT_SENDER_ID"):
        MessageEnvelope(CONFIG, "")


def test_unit_settings_name_every_missing_variable():
    with pytest.raises(RuntimeError, match="DGT_SENDER_ID.*DGT_ROAD_SOURCE.*ROAD_OBJECTS"):
        UnitSettings.from_env({})


def test_the_daemon_exits_nonzero_without_its_environment(monkeypatch, tmp_path):
    for name in ("DGT_SENDER_ID", "DGT_ROAD_SOURCE", "ROAD_OBJECTS_ENGINE_PATH"):
        monkeypatch.delenv(name, raising=False)
    assert main(["--env-file", str(tmp_path / "absent.env")]) == 1


def test_a_missing_file_source_fails_before_reaching_gstreamer(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "absent.mp4"))


def test_unit_settings_read_the_ros_variables():
    env = {"DGT_SENDER_ID": "s", "DGT_ROAD_SOURCE": "r", "ROAD_OBJECTS_ENGINE_PATH": "e"}
    defaults = UnitSettings.from_env(env)
    assert (defaults.ros_topic_prefix, defaults.ros_qos_best_effort) == ("/dgt", True)
    custom = UnitSettings.from_env({**env, "ROS_TOPIC_PREFIX": "/hidden/dgt",
                                    "ROS_QOS_BEST_EFFORT": "false"})
    assert (custom.ros_topic_prefix, custom.ros_qos_best_effort) == ("/hidden/dgt", False)


def test_the_daemon_refuses_to_run_without_ros(monkeypatch, tmp_path):
    """ROS is the only output: no rclpy means exit 1, not a silent run."""
    monkeypatch.setenv("DGT_SENDER_ID", "dgt-test")
    monkeypatch.setenv("DGT_ROAD_SOURCE", str(tmp_path / "road.mp4"))
    monkeypatch.setenv("ROAD_OBJECTS_ENGINE_PATH", str(tmp_path / "road.engine"))
    monkeypatch.setitem(sys.modules, "rclpy", None)  # makes `import rclpy` raise
    monkeypatch.setattr("signal.signal", lambda *_: None)
    assert main(["--env-file", str(tmp_path / "absent.env")]) == 1


@pytest.fixture
def fake_rclpy(monkeypatch):
    """Just enough of rclpy / std_msgs to record what the publisher does."""
    log = {"published": [], "calls": []}

    class Publisher:
        def publish(self, msg):
            log["published"].append(msg.data)

    class Node:
        def create_publisher(self, msg_type, topic, qos):
            log["calls"].append(("create_publisher", msg_type.__name__, topic, qos))
            return Publisher()

        def destroy_node(self):
            log["calls"].append(("destroy_node",))

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: log["calls"].append(("init",))
    rclpy.shutdown = lambda: log["calls"].append(("shutdown",))
    rclpy.create_node = lambda name: log["calls"].append(("create_node", name)) or Node()
    qos = types.ModuleType("rclpy.qos")
    qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST="KEEP_LAST")
    qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT="BEST_EFFORT",
                                                  RELIABLE="RELIABLE")
    qos.QoSProfile = lambda **kw: kw
    std_msgs = types.ModuleType("std_msgs")
    msg = types.ModuleType("std_msgs.msg")

    class String:
        data = ""

    msg.String = String
    for name, module in (("rclpy", rclpy), ("rclpy.qos", qos),
                         ("std_msgs", std_msgs), ("std_msgs.msg", msg)):
        monkeypatch.setitem(sys.modules, name, module)
    return log


def test_the_publisher_sends_the_json_on_prefix_status(fake_rclpy):
    publisher = DgtRosPublisher(topic_prefix="/dgt", best_effort=True)
    assert publisher.enabled and publisher.topic == "/dgt/status"
    publisher.publish('{"ok":true}')
    assert fake_rclpy["published"] == ['{"ok":true}']
    _, msg_type, topic, qos = fake_rclpy["calls"][2]
    assert (msg_type, topic) == ("String", "/dgt/status")
    assert qos == {"reliability": "BEST_EFFORT", "history": "KEEP_LAST", "depth": 1}


def test_the_publisher_shuts_down_once(fake_rclpy):
    publisher = DgtRosPublisher()
    publisher.shutdown()
    publisher.shutdown()
    publisher.publish("{}")
    assert [c for c in fake_rclpy["calls"] if c[0] in ("destroy_node", "shutdown")] == \
        [("destroy_node",), ("shutdown",)]
    assert fake_rclpy["published"] == []


def test_without_rclpy_the_publisher_is_disabled_not_broken(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", None)
    publisher = DgtRosPublisher()
    assert not publisher.enabled
    publisher.publish("{}")
    publisher.shutdown()
