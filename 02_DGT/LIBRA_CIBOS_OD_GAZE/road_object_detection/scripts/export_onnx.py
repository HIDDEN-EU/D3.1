"""
CLI: export-onnx — publish a trained checkpoint as a deployable ONNX artifact.

Training produces a `.pt`. The Jetson Orin runs a TensorRT engine. ONNX is the
handoff between the two: it is portable, whereas an engine is valid only for the
exact GPU / JetPack / TensorRT that built it, so the engine is always built
on-device and never shipped. That makes the `.pt -> .onnx` step the last one that
belongs to THIS repo — the consumer (the edge service in ../edge_inference) fetches the ONNX and
builds the engine itself.

The export is logged back into the SAME MLflow run that produced the weights, so
one run holds the metrics, the `.pt` they were measured on, and the `.onnx` that
actually ships. Without that, the checkpoint that was validated and the artifact
that runs can diverge with nothing on record saying so.

What the consumer cannot infer and this command therefore records as run tags:

  sha256              hash of the ONNX FILE. `dgt`'s fetch_models.sh reads this
                      exact tag name and refuses to run on a mismatch.
  onnx_graph_sha256   hash of the graph with metadata stripped. Ultralytics stamps
                      the export time into the ONNX, so `sha256` changes on every
                      export while the model does not; this one identifies the
                      model itself, and is what tells a re-publish apart from a
                      genuine change.
  onnx_class_names    the 10 names in index order — the model emits indices only.
  onnx_preprocess     letterbox geometry, channel order, scaling.
  onnx_tile_size      the tile the model was TRAINED on. It was fine-tuned on
                      640x360 crops, so a consumer that feeds it a full 1280x720
                      frame downsamples away the small objects tiling existed to
                      recover. Not a hint — a precondition.
  onnx_nms            whether NMS is inside the graph or is the caller's job.

Usage
-----
# from a local checkpoint, logging into the run that produced it
export-onnx --weights runs/train/<exp>/weights/best.pt --run-id <mlflow run id>

# fetch the checkpoint from the run as well
export-onnx --run-id <mlflow run id>

# verify the graph against the checkpoint, then register a new model version
export-onnx --run-id <id> --verify-image frame.jpg \
            --register hidden-road-object-detection

# produce the file only, touch nothing remote
export-onnx --weights runs/train/<exp>/weights/best.pt --dry-run

Environment — a separate `onnx` env, NOT the training env
------------------------------------------------------
The export runs on CPU and needs onnx/onnxslim/onnxruntime, whose wheels can pull
`numpy` down a major version. In the training env, a resolver walking
over its numpy to satisfy a once-per-release export is not a trade worth making.
So this command lives in its own env:

    conda create -n onnx python=3.11 -y
    conda activate onnx
    cd road_object_detection
    pip install torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu
    pip install -e ".[export]"

CPU torch on purpose: every path here (the export, and the --verify-image parity
check) runs on CPU, so the CUDA wheel is ~2.5 GB of download that never executes.

torchvision MUST come from the same index and in the same command. Installing only
torch from the CPU index leaves ultralytics to pull torchvision from PyPI, which
ships the CUDA build — the versions match, the builds do not, and the first import
dies with `RuntimeError: operator torchvision::nms does not exist`. If that has
already happened, repair it in place without disturbing torch:

    pip install --force-reinstall --no-deps torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

DEFAULT_OPSET = 17          # TensorRT 10.3 (JetPack 6.2) parses 17 without a plugin
DEFAULT_IMGSZ = "640x640"   # square, because training ran rect=False at imgsz=640
DEFAULT_TILE = "640x360"    # the crop geometry the model was fine-tuned on
DEFAULT_OUTPUT = Path("output/export")

# Ultralytics' fixed inference preprocessing. There is no mean/std normalisation —
# recording it as a tag means the consumer never has to guess or read our source.
PREPROCESS = "letterbox(pad=114) -> BGR2RGB -> /255 -> NCHW float32"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_imgsz(text: str) -> tuple[int, int]:
    """'640' -> (640, 640); '384x640' -> (384, 640), read as HxW."""
    parts = text.lower().split("x")
    try:
        dims = [int(p) for p in parts]
    except ValueError:
        raise SystemExit(f"--imgsz must be N or HxW, got {text!r}")
    if len(dims) == 1:
        dims *= 2
    if len(dims) != 2:
        raise SystemExit(f"--imgsz must be N or HxW, got {text!r}")
    for d in dims:
        if d % 32:
            raise SystemExit(f"--imgsz dimensions must be multiples of 32 (stride), got {d}")
    return dims[0], dims[1]


def _require(*modules: str) -> None:
    """Fail with the exact install line rather than letting Ultralytics pip-install
    into the conda env behind the user's back."""
    missing = []
    for m in modules:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        raise SystemExit(
            f"[export] missing export requirement(s): {', '.join(missing)}\n"
            f"[export] these live in the `onnx` env, not the training env (see the module "
            f"docstring for why). Either:\n"
            f"    conda activate onnx\n"
            f"[export] or, if that env does not exist yet:\n"
            f"    conda create -n onnx python=3.11 -y && conda activate onnx\n"
            f"    pip install torch==2.7.1 torchvision==0.22.1 "
            f"--index-url https://download.pytorch.org/whl/cpu\n"
            f"    pip install -e \".[export]\""
        )


def _load_dotenv() -> str | None:
    """Find .env from the working directory, not the console script's install path."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    return os.environ.get("MLFLOW_TRACKING_URI")


def _fetch_weights(run_id: str, artifact: str, dest: Path) -> Path:
    """Download the checkpoint from its MLflow run."""
    import mlflow

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[export] fetching {artifact} from run {run_id}")
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact, dst_path=str(dest)
    )
    return Path(local)


def _export(weights: Path, hw: tuple[int, int], opset: int, batch: int,
            nms: bool, simplify: bool) -> tuple[Path, dict]:
    """Run the Ultralytics export and return (onnx path, model facts)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names  # authoritative: read off the checkpoint, not a data.yaml
    print(f"[export] {weights}  nc={len(names)}  imgsz={hw[0]}x{hw[1]}  opset={opset}  "
          f"batch={batch}  nms={'embedded' if nms else 'none'}")

    produced = model.export(
        format="onnx",
        imgsz=[hw[0], hw[1]],
        opset=opset,
        batch=batch,
        nms=nms,
        simplify=simplify,
        # Static shapes: the Orin runs a single stream, and dynamic axes cost real
        # throughput in TensorRT for no benefit here.
        dynamic=False,
        # fp32 on purpose. Precision is a TensorRT build-time choice (`--fp16`);
        # baking fp16 into the ONNX would make that decision unavailable on-device.
        half=False,
        device="cpu",
    )
    return Path(produced), {"names": names}


def _graph_io(onnx_path: Path) -> dict:
    """Input/output names and shapes, read back from the file that will ship."""
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    def shape_of(value) -> str:
        dims = []
        for d in value.type.tensor_type.shape.dim:
            dims.append(str(d.dim_value) if d.dim_value else (d.dim_param or "?"))
        return "x".join(dims)

    inputs = {i.name: shape_of(i) for i in model.graph.input}
    outputs = {o.name: shape_of(o) for o in model.graph.output}

    # Identity of the MODEL, as opposed to identity of the file. Ultralytics stamps a
    # wall-clock `date` into metadata_props, so two exports of one checkpoint produce
    # different file hashes with identical graphs. Hashing the graph with metadata
    # stripped gives a digest that is stable across re-exports, which is what lets a
    # re-publish be told apart from a genuine model change.
    del model.metadata_props[:]
    graph_sha = hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    return {"inputs": inputs, "outputs": outputs, "graph_sha256": graph_sha}


def _letterbox(image, hw: tuple[int, int]):
    """Ultralytics-equivalent letterbox: preserve aspect, pad to hw with 114."""
    import cv2
    import numpy as np

    h, w = hw
    ih, iw = image.shape[:2]
    r = min(h / ih, w / iw)
    nh, nw = round(ih * r), round(iw * r)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0


def _verify(weights: Path, onnx_path: Path, image: Path, hw: tuple[int, int]) -> float:
    """Run both graphs on one real frame and return the max absolute difference.

    This tests the EXPORT, not the preprocessing: the same tensor goes into both,
    so any divergence is the ONNX graph itself. A silently wrong export is the
    failure this exists to catch — found later, on-device, it looks like a
    TensorRT bug and costs a day.
    """
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    frame = cv2.imread(str(image))
    if frame is None:
        raise SystemExit(f"[export] cannot read --verify-image: {image}")
    tensor = _letterbox(frame, hw)

    torch_model = YOLO(str(weights)).model.float().eval()
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(tensor))
    ref = ref[0] if isinstance(ref, (list, tuple)) else ref
    ref = ref.cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]

    if ref.shape != got.shape:
        raise SystemExit(f"[export] VERIFY FAILED: shape {ref.shape} (torch) != {got.shape} (onnx)")
    return float(np.abs(ref - got).max())


def _log_to_mlflow(run_id: str, onnx_path: Path, tags: dict, register: str | None,
                   force: bool) -> bool:
    """Attach the artifact + its metadata to the run that produced the weights.

    Returns False when a requested registry write was refused but the publish itself
    succeeded — the caller reports that as a partial success, not a crash.
    """
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    run_tags = client.get_run(run_id).data.tags
    existing = run_tags.get("sha256")
    if existing and existing != tags["sha256"] and not force:
        prior_graph = run_tags.get("onnx_graph_sha256")
        if prior_graph and prior_graph == tags["onnx_graph_sha256"]:
            print(f"[export] re-publishing the same model: file hash changed "
                  f"({existing[:12]}... -> {tags['sha256'][:12]}...) but the graph is "
                  f"identical, so this is Ultralytics' export timestamp, not a new model.")
        else:
            raise SystemExit(
                f"[export] run {run_id} already carries sha256={existing[:16]}... for a "
                f"different artifact.\n[export] a run's `sha256` tag is what dgt verifies "
                f"against, so overwriting it silently repoints the consumer.\n"
                + (f"[export] its graph digest {prior_graph[:16]}... also differs from this "
                   f"export's {tags['onnx_graph_sha256'][:16]}..., so the MODEL really did "
                   f"change.\n" if prior_graph else
                   f"[export] that artifact predates graph-digest tagging, so this cannot "
                   f"tell a re-export apart from a new model. Note the file hash changes on "
                   f"EVERY export: Ultralytics stamps a wall-clock date into the ONNX "
                   f"metadata.\n")
                + f"[export] re-run with --force if that is what you mean."
            )

    artifact_dir = "onnx"
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(onnx_path), artifact_path=artifact_dir)
        mlflow.set_tags(tags)
    print(f"[export] logged {artifact_dir}/{onnx_path.name} + {len(tags)} tags to run {run_id}")

    if register:
        uri = f"runs:/{run_id}/{artifact_dir}/{onnx_path.name}"
        # NOT mlflow.register_model(): since MLflow 3 that resolves a runs:/ URI to a
        # "logged model" — an MLmodel-flavoured directory — and fails with "Unable to
        # find a logged_model" on a plain artifact. Giving the ONNX a flavour would mean
        # a second copy of a 38 MB file that nothing reads: the consumer fetches the file
        # by artifact path and builds a TensorRT engine from it, never mlflow.pyfunc.
        # The registry accepts a plain artifact as a version source (v1 of this model is
        # one), so go through the client and register the artifact itself.
        from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository

        try:
            client.create_registered_model(register)
            print(f"[export] created registered model {register}")
        except MlflowException as exc:
            if exc.error_code not in ("RESOURCE_ALREADY_EXISTS", "ALREADY_EXISTS"):
                raise

        try:
            version = client.create_model_version(
                name=register,
                source=RunsArtifactRepository.get_underlying_uri(uri),
                run_id=run_id,
                tags={k: tags[k] for k in ("sha256", "onnx_nc", "onnx_imgsz", "onnx_nms",
                                           "onnx_opset") if k in tags},
            )
        except MlflowException as exc:
            # The artifact and its tags are already on the run, which is what the
            # consumer reads — registration is a separate privilege on this server, and
            # losing it must not read as a failed publish.
            print(f"\n[export] REGISTRATION FAILED: {exc}", file=sys.stderr)
            print("[export] the ONNX and its tags ARE published on the run; only the "
                  "registry write was refused.", file=sys.stderr)
            print("[export] 'Permission denied' here means the service account can read "
                  "the registry but not write it."
                  " Ask for CreateModelVersion on "
                  f"'{register}', then re-run with --register.", file=sys.stderr)
            return False
        print(f"[export] registered {register} version {version.version} -> {uri}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained YOLO checkpoint to ONNX.")
    ap.add_argument("--weights", type=Path,
                    help="local checkpoint; omit to download it from --run-id")
    ap.add_argument("--run-id", help="MLflow run that produced the weights and receives the ONNX")
    ap.add_argument("--run-artifact", default="weights/best.pt",
                    help="artifact path of the checkpoint inside the run (default: weights/best.pt)")
    ap.add_argument("--imgsz", default=DEFAULT_IMGSZ,
                    help=f"N or HxW, multiples of 32 (default: {DEFAULT_IMGSZ})")
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument("--batch", type=int, default=1, help="static batch size (default: 1)")
    ap.add_argument("--nms", action="store_true",
                    help="embed NMS in the graph. Default off: keeping it out leaves conf/iou "
                         "tunable at runtime instead of frozen at export time")
    ap.add_argument("--no-simplify", action="store_true", help="skip onnxslim simplification")
    ap.add_argument("--tile-size", default=DEFAULT_TILE,
                    help=f"tile geometry the model was trained on, recorded for the consumer "
                         f"(default: {DEFAULT_TILE})")
    ap.add_argument("--name", default=None,
                    help="artifact filename (default: road_objects_<H>x<W>.onnx)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--verify-image", type=Path,
                    help="real frame to run both graphs on; without it the export ships unverified")
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="max allowed |torch - onnx| on the verify frame (default: 1e-3)")
    ap.add_argument("--register", default=None, metavar="MODEL_NAME",
                    help="also register a new version of this MLflow registered model")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing, different sha256 tag on the run")
    ap.add_argument("--dry-run", action="store_true", help="export locally, log nothing to MLflow")
    args = ap.parse_args()

    if not args.weights and not args.run_id:
        ap.error("one of --weights or --run-id is required")
    if not args.dry_run and not args.run_id:
        ap.error("--run-id is required unless --dry-run: the ONNX is logged into the run that "
                 "produced the weights, so the shipped artifact and the metrics it was "
                 "measured against stay in one place")

    hw = _parse_imgsz(args.imgsz)
    _require("onnx")
    if not args.no_simplify:
        _require("onnxslim")
    if args.verify_image:
        _require("onnxruntime")

    # MLflow is needed to DOWNLOAD the checkpoint as well as to log the result, so
    # --dry-run does not excuse a missing tracking URI when the weights live on a run.
    # Left unset, the client falls back to a local ./mlruns and the download fails with
    # "run not found", which reads as a bad run id rather than a missing credential.
    if args.weights is None or not args.dry_run:
        tracking_uri = _load_dotenv()
        if not tracking_uri:
            why = ("the checkpoint is fetched from run "
                   f"{args.run_id}" if args.weights is None else
                   "an export that isn't recorded on the run is an artifact nobody can "
                   "trace back to its metrics")
            sys.exit(f"MLFLOW_TRACKING_URI is not set — {why}. Set it (or put it in .env)"
                     + ("." if args.weights is None else
                        ", or re-run with --dry-run."))
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)

    args.output.mkdir(parents=True, exist_ok=True)
    weights = args.weights
    # What to RECORD as the source. A fetched checkpoint lands in a scratch directory
    # whose path says nothing; the run it came from is the fact worth keeping.
    source_label = str(args.weights) if args.weights else \
        f"runs:/{args.run_id}/{args.run_artifact}"
    if weights is None:
        weights = _fetch_weights(args.run_id, args.run_artifact, args.output / "checkpoint")
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")

    produced, facts = _export(weights, hw, args.opset, args.batch, args.nms,
                              simplify=not args.no_simplify)

    name = args.name or f"road_objects_{hw[0]}x{hw[1]}.onnx"
    onnx_path = args.output / name
    if produced.resolve() != onnx_path.resolve():
        shutil.move(str(produced), str(onnx_path))

    io = _graph_io(onnx_path)
    digest = _sha256(onnx_path)
    print(f"[export] -> {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    for kind in ("inputs", "outputs"):
        for n, shape in io[kind].items():
            print(f"[export]   {kind[:-1]:6s} {n}: {shape}")
    print(f"[export] sha256 {digest}")

    verified = "false"
    if args.verify_image:
        diff = _verify(weights, onnx_path, args.verify_image, hw)
        if diff > args.tolerance:
            sys.exit(f"[export] VERIFY FAILED: max |torch - onnx| = {diff:.2e} "
                     f"> tolerance {args.tolerance:.0e}. The export is not faithful — "
                     f"do not ship it.")
        print(f"[export] verified against the checkpoint: max abs diff {diff:.2e}")
        verified = f"max_abs_diff={diff:.2e}"
    else:
        print("[export] WARNING: exported without --verify-image. Nothing has checked that "
              "this graph reproduces the checkpoint's outputs.")

    names = facts["names"]
    tags = {
        # Read by dgt's fetch_models.sh, by this exact name.
        "sha256": digest,
        "onnx_artifact_path": f"onnx/{onnx_path.name}",
        # Stable across re-exports of the same checkpoint; `sha256` is not.
        "onnx_graph_sha256": io["graph_sha256"],
        "onnx_class_names": ",".join(names[i] for i in sorted(names)),
        "onnx_nc": str(len(names)),
        "onnx_imgsz": f"{hw[0]}x{hw[1]}",
        "onnx_batch": str(args.batch),
        "onnx_opset": str(args.opset),
        "onnx_dynamic": "false",
        "onnx_precision": "fp32",
        "onnx_nms": "embedded" if args.nms else "none",
        "onnx_preprocess": PREPROCESS,
        "onnx_tile_size": args.tile_size,
        "onnx_inputs": ", ".join(f"{n}:{s}" for n, s in io["inputs"].items()),
        "onnx_outputs": ", ".join(f"{n}:{s}" for n, s in io["outputs"].items()),
        "onnx_source_checkpoint": source_label,
        "onnx_verified": verified,
    }

    if args.dry_run:
        print("[export] --dry-run: nothing logged. Tags that WOULD be set:")
        for k, v in tags.items():
            print(f"[export]   {k} = {v}")
        return

    registered = _log_to_mlflow(args.run_id, onnx_path, tags, args.register,
                                args.force)

    print(f"\n[next] the consumer fetches this by run id. In the edge service,\n"
          f"       dgt/config/models.yaml:\n\n"
          f"         road_objects:\n"
          f"           source:\n"
          f"             path: null\n"
          f"             mlflow_run_id: {args.run_id}\n"
          f"             mlflow_artifact_path: onnx/{onnx_path.name}\n"
          f"             mlflow_model_id: null\n")

    if not registered:
        sys.exit(1)


if __name__ == "__main__":
    main()
