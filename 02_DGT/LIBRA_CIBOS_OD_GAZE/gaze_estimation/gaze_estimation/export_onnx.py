"""Export a checkpoint to ONNX for the edge service (../edge_inference).

    gaze-export-onnx runs/example/best.pt --output gaze_face.onnx

Interface: one input `face_img` (B, 3, 112, 112), float32 RGB in [-1, 1]; one output
`gaze` (B, 2), the gaze point in the normalised training space. Opset 17, dynamic batch.
The edge service builds an FP16 TensorRT engine from it on the device and maps the raw
point onto the target plane with its calibration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_checkpoint


def export(checkpoint: Path, output: Path, opset: int = 17) -> Path:
    model, cfg, _ = load_checkpoint(checkpoint, "cpu")
    size = cfg.data.image_size
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model, dummy, str(output),
        input_names=["face_img"], output_names=["gaze"],
        dynamic_axes={"face_img": {0: "batch"}, "gaze": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--output", type=Path, default=Path("gaze_face.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    print(f"exported -> {export(args.checkpoint, args.output, args.opset)}")


if __name__ == "__main__":
    main()
