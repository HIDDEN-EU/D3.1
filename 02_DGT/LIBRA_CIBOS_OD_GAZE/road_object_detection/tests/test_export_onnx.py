"""
Unit tests for the ONNX export CLI's pure helpers.

No model, no ONNX runtime, no network — `scripts/export_onnx.py` keeps every heavy
import inside a function precisely so this stays true. Runnable under pytest or
standalone:

    python tests/test_export_onnx.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import export_onnx as ex


def test_module_import_is_light():
    """Importing the CLI must not drag in torch/ultralytics/onnx.

    The helpers are unit-testable and `--help` is instant only while this holds.
    """
    for heavy in ("torch", "ultralytics", "onnx", "onnxruntime", "cv2"):
        assert heavy not in sys.modules, f"{heavy} leaked into sys.modules on import"


def test_imgsz_single_value_is_square():
    assert ex._parse_imgsz("640") == (640, 640)


def test_imgsz_pair_is_read_as_height_by_width():
    """384x640 is the 640x360 tile padded up to the next stride multiple, not 640x384."""
    assert ex._parse_imgsz("384x640") == (384, 640)


def test_imgsz_rejects_non_stride_multiples():
    """A non-multiple of 32 silently reshapes inside YOLO; refuse it at the edge."""
    for bad in ("360", "640x360", "100"):
        try:
            ex._parse_imgsz(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def test_imgsz_rejects_garbage():
    for bad in ("", "abc", "640x640x640", "640xN"):
        try:
            ex._parse_imgsz(bad)
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def test_sha256_matches_hashlib():
    """The digest is what dgt's gate compares against, so it must be the plain
    file hash — not a hash of anything we add around it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact.onnx"
        payload = b"weights" * 5000
        path.write_bytes(payload)
        assert ex._sha256(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_reads_in_chunks_without_truncating():
    """The real artifact is ~20 MB, read in 1 MiB chunks; prove the loop terminates
    on the exact boundary rather than dropping the tail."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "big.onnx"
        payload = b"\x00" * (1 << 20) + b"tail"
        path.write_bytes(payload)
        assert ex._sha256(path) == hashlib.sha256(payload).hexdigest()


def test_require_names_the_missing_modules():
    try:
        ex._require("a_module_that_does_not_exist")
    except SystemExit as exc:
        assert "a_module_that_does_not_exist" in str(exc)
        assert "pip install" in str(exc)
        return
    raise AssertionError("_require should have exited")


def test_require_passes_on_stdlib():
    ex._require("json", "pathlib")


def test_defaults_are_the_documented_ones():
    """These are the deployment contract, not preferences: opset 17 is what
    TensorRT 10.3 on JetPack 6.2 parses, and 640x640 is what rect=False training
    letterboxed to."""
    assert ex.DEFAULT_OPSET == 17
    assert ex._parse_imgsz(ex.DEFAULT_IMGSZ) == (640, 640)
    assert ex.DEFAULT_TILE == "640x360"


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 — standalone runner reports, not raises
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
