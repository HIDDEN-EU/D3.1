"""Model classes: one per model, each returning a contract.

Imports are lazy (PEP 562), so a device only imports the backends it uses.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "RoadObjectDetector": "road_objects",
    "FaceLandmarkDetector": "landmarks",
    "GazeEstimator": "gaze",
    "TensorRTEngine": "trt",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, name)


def __dir__():
    return __all__
