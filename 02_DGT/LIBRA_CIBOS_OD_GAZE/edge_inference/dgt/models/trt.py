"""Minimal TensorRT 10 runner: deserialize an engine, bind I/O by name, run one batch.

Engines are valid only for the GPU / JetPack / TensorRT that built them, so they are built
on-device (`dgt-build-engines`). TensorRT and cuda-python exist only on the Jetson and are
imported on use.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def _check(result):
    """cuda-python returns (err, *values); raise on a non-zero error, return the values."""
    err, *values = result if isinstance(result, tuple) else (result,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA call failed with error {err}")
    if not values:
        return None
    return values[0] if len(values) == 1 else values


class TensorRTEngine:
    """One loaded engine; names and shapes are in I/O order, as `infer()` takes and returns."""

    def __init__(self, engine_path: Path) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f"no TensorRT engine at {self.engine_path}. An engine is valid only "
                f"for the exact GPU / JetPack / TensorRT that built it, so build it "
                f"here from the ONNX:\n"
                f"  dgt-build-engines --engine-dir engines"
            )

        try:
            import tensorrt as trt
            from cuda import cudart
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "running a TensorRT engine needs TensorRT (JetPack) and cuda-python "
                "(pip install 'cuda-python~=12.6')"
            ) from exc
        self._cudart = cudart

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(
                f"could not deserialize {self.engine_path} — it was almost certainly "
                f"built on different hardware or a different TensorRT. Rebuild it here."
            )
        self._context = self._engine.create_execution_context()
        self._stream = _check(cudart.cudaStreamCreate())

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: List[Tuple[int, ...]] = []
        self.output_shapes: List[Tuple[int, ...]] = []
        self._host: dict = {}
        self._device: dict = {}
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                # Dynamic batch: run with batch 1 at the profile's fixed spatial size.
                shape = tuple(1 if d < 0 else d for d in shape)
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self._context.set_input_shape(name, shape)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host = np.zeros(shape, dtype=dtype)
            device = _check(cudart.cudaMalloc(host.nbytes))
            self._context.set_tensor_address(name, int(device))
            self._host[name] = host
            self._device[name] = device
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self.input_shapes.append(shape)
            else:
                self.output_names.append(name)
                self.output_shapes.append(shape)

    def infer(self, *blobs: np.ndarray) -> List[np.ndarray]:
        """One forward pass; inputs are cast to the engine dtype, outputs are copies."""
        if len(blobs) != len(self.input_names):
            raise ValueError(
                f"{self.engine_path.name} takes {len(self.input_names)} input(s) "
                f"{self.input_names}, got {len(blobs)}"
            )
        cudart = self._cudart
        h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        for name, blob in zip(self.input_names, blobs):
            host = self._host[name]
            host[...] = np.asarray(blob, dtype=host.dtype).reshape(host.shape)
            _check(cudart.cudaMemcpyAsync(self._device[name], host.ctypes.data,
                                          host.nbytes, h2d, self._stream))
        if not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError(f"{self.engine_path.name}: inference failed")
        for name in self.output_names:
            host = self._host[name]
            _check(cudart.cudaMemcpyAsync(host.ctypes.data, self._device[name],
                                          host.nbytes, d2h, self._stream))
        _check(cudart.cudaStreamSynchronize(self._stream))
        return [self._host[name].copy() for name in self.output_names]

    def close(self) -> None:
        if getattr(self, "_device", None):
            for device in self._device.values():
                self._cudart.cudaFree(device)
            self._device = {}
            self._cudart.cudaStreamDestroy(self._stream)
        self._context = None
        self._engine = None

    def describe(self) -> str:
        ins = ", ".join(f"{n}{s}" for n, s in zip(self.input_names, self.input_shapes))
        outs = ", ".join(f"{n}{s}" for n, s in zip(self.output_names, self.output_shapes))
        return f"{self.engine_path.name} in=[{ins}] out=[{outs}]"
