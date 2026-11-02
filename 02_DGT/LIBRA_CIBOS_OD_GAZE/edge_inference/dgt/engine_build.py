"""dgt-build-engines: models.yaml ONNX -> FP16 TensorRT engines, on the device.

An engine is valid only for the exact GPU / JetPack / TensorRT that built it, so it is built
here and never shipped. Each entry of models.yaml with a source is resolved (a local path,
or a URL downloaded into the engine directory), checked against its sha256 when one is
given, and built to `<engine-dir>/<onnx stem>_fp16.engine`.

    dgt-build-engines --engine-dir engines
    dgt-build-engines --engine-dir engines --models gaze_estimator --force
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

from .config.loader import ModelConfig, load_models_config


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_onnx(model: ModelConfig, cache_dir: Path) -> Path:
    """Local path, or the cached download of `url`; verified against `sha256` if set."""
    source = model.source
    if source.path is not None:
        onnx = Path(source.path)
        if not onnx.is_file():
            raise FileNotFoundError(f"no ONNX at {onnx}")
    else:
        onnx = cache_dir / Path(source.url.split("?")[0]).name
        if not onnx.is_file():
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"[build] downloading {source.url}")
            urllib.request.urlretrieve(source.url, onnx)
    if source.sha256 and sha256sum(onnx) != source.sha256.lower():
        raise ValueError(f"{onnx}: sha256 does not match models.yaml")
    return onnx


def build_engine(onnx: Path, engine: Path, input_shapes: Sequence[Sequence[int]],
                 workspace_mb: int = 1024) -> None:
    """Parse the ONNX and serialize an FP16 engine. Dynamic inputs get a fixed profile
    at `input_shapes` (batch 1)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"{onnx}: ONNX parse failed\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    dynamic = [network.get_input(i) for i in range(network.num_inputs)
               if any(d < 0 for d in network.get_input(i).shape)]
    if dynamic:
        if len(input_shapes) != network.num_inputs:
            raise ValueError(f"{onnx}: has dynamic inputs; models.yaml must give "
                             f"input_shapes for all {network.num_inputs} input(s)")
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            shape = tuple(int(d) for d in input_shapes[i])
            profile.set_shape(network.get_input(i).name, shape, shape, shape)
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"{onnx}: engine build failed")
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(bytes(serialized))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="dgt-build-engines", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-dir", type=Path, default=Path("engines"))
    ap.add_argument("--models-config", type=Path, default=None,
                    help="models.yaml (default: the packaged dgt/config/models.yaml)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="build only these entries (default: every entry with a source)")
    ap.add_argument("--force", action="store_true", help="rebuild existing engines")
    args = ap.parse_args(argv)

    models = load_models_config(args.models_config)
    if args.models:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            ap.error(f"no source in models.yaml for: {', '.join(unknown)}")
        models = {name: models[name] for name in args.models}
    if not models:
        print("[build] models.yaml names no source; fill in the paths first", file=sys.stderr)
        return 1

    for name, model in models.items():
        onnx = resolve_onnx(model, args.engine_dir / "onnx")
        engine = args.engine_dir / f"{onnx.stem}_fp16.engine"
        if engine.is_file() and not args.force:
            print(f"[build] {name}: {engine} exists (use --force to rebuild)")
            continue
        print(f"[build] {name}: {onnx} -> {engine}")
        build_engine(onnx, engine, model.input_shapes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
