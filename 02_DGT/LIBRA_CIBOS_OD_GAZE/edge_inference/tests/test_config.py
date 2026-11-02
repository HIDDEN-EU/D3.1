"""The shipped YAMLs must load, and the code defaults must not drift from them."""
from __future__ import annotations

from pathlib import Path

import pytest

from dgt.config.loader import ModelSource, load_models_config, load_service, load_thresholds


def test_shipped_thresholds_load_completely():
    t = load_thresholds()
    assert t.ear_threshold == pytest.approx(0.20)  # geometric EAR, not the 0.25 MediaPipe one
    assert t.perclos.window_seconds == pytest.approx(30.0)
    assert t.perclos.min_fill_frac == pytest.approx(1.0)
    assert t.road_objects.conf_threshold == pytest.approx(0.25)
    assert t.face_landmarks.det_threshold == pytest.approx(0.50)
    assert t.face_landmarks.nms_threshold == pytest.approx(0.40)


def test_only_ear_threshold_is_required(tmp_path: Path):
    stripped = tmp_path / "thresholds.yaml"
    stripped.write_text("ear_threshold: 0.18\n")
    t = load_thresholds(stripped)
    assert t.ear_threshold == pytest.approx(0.18)
    assert t.perclos.window_seconds == pytest.approx(30.0)     # code fallback
    assert t.face_landmarks.det_threshold == pytest.approx(0.5)  # code fallback


def test_missing_ear_threshold_raises(tmp_path: Path):
    empty = tmp_path / "thresholds.yaml"
    empty.write_text("perclos: {window_seconds: 10.0}\n")
    with pytest.raises(KeyError):
        load_thresholds(empty)


def test_shipped_models_yaml_names_no_source():
    """Model files are obtained per deployment, so the shipped file names none and
    loads to an empty mapping."""
    assert load_models_config() == {}


def test_model_source_needs_exactly_one_mode():
    with pytest.raises(ValueError):
        ModelSource()
    with pytest.raises(ValueError):
        ModelSource(path=Path("a.onnx"), url="https://example.org/a.onnx")


def test_model_source_describes_its_provenance():
    assert ModelSource(path=Path("/w/a.onnx")).describe() == "/w/a.onnx"
    remote = ModelSource(url="https://example.org/a.onnx", sha256="ab")
    assert remote.is_remote and remote.describe() == "https://example.org/a.onnx"


def test_models_config_reads_a_configured_source(tmp_path: Path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "object_detection:\n"
        "  source: {url: https://example.org/road.onnx, sha256: ab12}\n"
        "  input_shapes: [[1, 3, 640, 640]]\n"
        "emotion:\n"
        "  source: {path: null}\n"
    )
    configs = load_models_config(cfg)
    assert set(configs) == {"object_detection"}  # emotion has no source -> skipped
    assert configs["object_detection"].source.url == "https://example.org/road.onnx"
    assert configs["object_detection"].input_shapes == ((1, 3, 640, 640),)


# -- input_shapes: what the engine builder reads -----------------------------

def _models_yaml():
    import yaml

    from dgt.config import loader

    return yaml.safe_load(loader._DEFAULT_MODELS.read_text())


def test_every_model_declares_input_shapes():
    """The builder skips an ONNX with no declared shape rather than guessing
    one, so a model added without this silently never gets an engine."""
    for name, entry in _models_yaml().items():
        assert entry.get("input_shapes"), f"{name} has no input_shapes"


def test_input_shapes_are_lists_of_positive_ints():
    """TensorRT pins min=opt=max from these; a -1 left in would not build."""
    for name, entry in _models_yaml().items():
        for shape in entry["input_shapes"]:
            assert len(shape) == 4, f"{name}: {shape} is not NCHW"
            assert all(isinstance(d, int) and d > 0 for d in shape), \
                f"{name}: {shape} has a non-positive dimension"


def test_shipped_service_config_loads():
    s = load_service()
    assert (s.message_type, s.time_reference, s.road_frame_id) == \
        ("DGT_DS", "UTC", "road_camera")


def test_service_message_fields_are_required(tmp_path: Path):
    stripped = tmp_path / "service.yaml"
    stripped.write_text("road_frame_id: r\n")
    with pytest.raises(KeyError):
        load_service(stripped)
