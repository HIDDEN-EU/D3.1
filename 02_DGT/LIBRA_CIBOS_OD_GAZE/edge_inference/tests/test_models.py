"""ONNX resolution for the engine build, and the import-time guards.

None of this needs torch, ultralytics, TensorRT or a camera.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dgt.config.loader import ModelSource
from dgt.config.loader import ModelConfig
from dgt.engine_build import resolve_onnx, sha256sum


# --- import-time leanness --------------------------------------------------

@pytest.mark.parametrize(
    "statement",
    [
        "import dgt",
        "import dgt.contracts",
        "import dgt.attention",
        "import dgt.config.loader",
        "import dgt.models",
        "import dgt.calibration",
        "import dgt.engine_build",
        # The road-object model touches TensorRT, CUDA and OpenCV — all of them
        # inside functions, so the module still imports on a bare dev box.
        "import dgt.models.road_objects",
        # Same for the landmark path: TensorRT, CUDA and OpenCV are all reached
        # inside functions, so a dev box with numpy alone still imports these.
        "import dgt.models.trt",
        "import dgt.models.landmarks",
        # The daemon reaches models, cv2, dotenv and rclpy inside functions only.
        "import dgt.contracts.driver_status",
        "import dgt.sources",
        "import dgt.daemon",
        # rclpy is reached inside the publisher's constructor only.
        "import dgt.publishers",
        "import dgt.publishers.dgt_ds",
        "import dgt.publishers.ros_publisher",
    ],
)
def test_modules_import_without_any_heavy_dependency(statement):
    """A Jetson with only numpy+pyyaml must be able to import all of this; the
    heavy deps are pulled in when a model is USED, not imported."""
    code = (
        f"{statement}; import sys; "
        # `cuda` is deliberately NOT checked: cuda-python ships a .pth
        # (_cuda_bindings_redirector) that runs at interpreter startup, so it is
        # in sys.modules of a BARE python wherever the package is installed.
        # Asserting on it would fail on the Jetson for a reason unrelated to dgt.
        "leaked = [m for m in ('torch','ultralytics','mlflow','cv2',"
        "'tensorrt','rclpy') "
        "if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dgt_ships_no_pipelines():
    """The harness pipelines are development tools and live in scripts/. If a
    `dgt/pipelines` comes back, a harness has crept into the deployed wheel."""
    import dgt

    # .py files, not the directory: a checkout that predates the move keeps a
    # stale __pycache__ there, which ships nothing and imports nothing.
    assert not list((Path(dgt.__file__).parent / "pipelines").glob("*.py"))


def test_lazy_exports_are_reachable_and_typos_still_raise():
    import dgt.models as models

    assert models.GazeEstimator.__name__ == "GazeEstimator"
    with pytest.raises(AttributeError):
        models.NoSuchModel


# --- ONNX resolution ---------------------------------------------------------

def test_a_local_path_is_returned_as_is(tmp_path: Path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"not really a model")
    assert resolve_onnx(ModelConfig(ModelSource(path=onnx)), tmp_path) == onnx


def test_a_missing_local_path_fails_loudly(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_onnx(ModelConfig(ModelSource(path=tmp_path / "absent.onnx")), tmp_path)


def test_a_sha256_mismatch_is_rejected(tmp_path: Path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"bytes")
    good = ModelConfig(ModelSource(path=onnx, sha256=sha256sum(onnx)))
    assert resolve_onnx(good, tmp_path) == onnx
    with pytest.raises(ValueError, match="sha256"):
        resolve_onnx(ModelConfig(ModelSource(path=onnx, sha256="00" * 32)), tmp_path)


def test_a_cached_download_is_reused(tmp_path: Path):
    cached = tmp_path / "road.onnx"
    cached.write_bytes(b"cached")
    source = ModelSource(url="https://example.org/models/road.onnx?download=1")
    assert resolve_onnx(ModelConfig(source), tmp_path) == cached
