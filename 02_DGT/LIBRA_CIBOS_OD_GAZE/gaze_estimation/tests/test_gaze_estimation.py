"""Shapes, normalisation, the subject-independent split, training and ONNX export,
on a tiny randomly initialised ViT and synthetic face crops (CPU, no download)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gaze_estimation.config import ExperimentConfig
from gaze_estimation.data import (
    FaceGazeDataset,
    compute_bounds,
    denormalise,
    load_face,
    normalise,
    read_manifest,
    select_subjects,
)
from gaze_estimation.evaluate import evaluate
from gaze_estimation.model import GazeViT
from gaze_estimation.train import EarlyStopping, euclidean_loss, train

TINY = {"depth": 2, "embed_dim": 32, "num_heads": 2}


def _make_dataset(root: Path, subjects=("s1", "s2", "s3"), per_subject=8) -> Path:
    rng = np.random.default_rng(0)
    rows = ["image_path,subject,x,y"]
    for subject in subjects:
        for i in range(per_subject):
            name = f"{subject}_{i}.png"
            pixels = rng.integers(0, 255, (60, 50, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / name)
            rows.append(f"{name},{subject},{rng.uniform(0, 190):.2f},{rng.uniform(0, 95):.2f}")
    manifest = root / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")
    return manifest


def _config(manifest: Path, out: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "model": {"name": "vit_tiny_patch16_224", "pretrained": False,
                  "n_trainable_blocks": 1, "head_hidden_dim": 16, "encoder_kwargs": TINY},
        "data": {"manifest": str(manifest), "train_subjects": ["s1"],
                 "val_subjects": ["s2"], "test_subjects": ["s3"]},
        "train": {"epochs": 2, "batch_size": 4, "num_workers": 0, "device": "cpu",
                  "output_dir": str(out)},
    })


def test_model_maps_a_batch_of_faces_to_points():
    model = GazeViT("vit_tiny_patch16_224", pretrained=False, n_trainable_blocks=1, **TINY)
    model.eval()
    assert model(torch.zeros(3, 3, 112, 112)).shape == (3, 2)


def test_only_the_last_blocks_and_the_head_train():
    model = GazeViT("vit_tiny_patch16_224", pretrained=False, n_trainable_blocks=1, **TINY)
    assert not any(p.requires_grad for p in model.encoder.blocks[0].parameters())
    assert all(p.requires_grad for p in model.encoder.blocks[-1].parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    assert not model.encoder.patch_embed.proj.weight.requires_grad


def test_normalisation_round_trip_and_range():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, (200, 2))
    bounds = compute_bounds([type("S", (), {"x": a, "y": b})() for a, b in xy], q=0.1)
    norm = normalise(xy, bounds)
    assert np.allclose(denormalise(norm, bounds), xy)
    assert np.allclose(np.quantile(norm, [0.1, 0.9], axis=0), [[-1, -1], [1, 1]])


def test_the_face_crop_is_chw_in_minus_one_to_one(tmp_path: Path):
    Image.fromarray(np.full((40, 30, 3), 255, np.uint8)).save(tmp_path / "white.png")
    face = load_face(tmp_path / "white.png")
    assert face.shape == (3, 112, 112) and face.dtype == np.float32
    assert np.allclose(face, 1.0)


def test_a_subject_in_two_splits_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="both train and val"):
        ExperimentConfig.model_validate({"data": {
            "manifest": "m.csv", "train_subjects": ["a", "b"], "val_subjects": ["b"]}})


def test_the_dataset_yields_normalised_targets(tmp_path: Path):
    samples = read_manifest(_make_dataset(tmp_path))
    train_samples = select_subjects(samples, ["s1"])
    dataset = FaceGazeDataset(train_samples, compute_bounds(train_samples))
    face, target = dataset[0]
    assert face.shape == (3, 112, 112) and target.shape == (2,)


def test_euclidean_loss_is_the_mean_distance():
    pred = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert euclidean_loss(pred, torch.zeros(2, 2)).item() == pytest.approx(2.5)


def test_early_stopping_waits_for_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.0)
    assert [stopper.step(v) for v in (1.0, 0.9, 0.95, 0.96)] == [False, False, False, True]


def test_train_evaluate_and_export(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from gaze_estimation.export_onnx import export

    data = tmp_path / "data"
    data.mkdir()
    cfg = _config(_make_dataset(data), tmp_path / "run")
    best = train(cfg)
    assert best.is_file() and (tmp_path / "run" / "metrics.json").is_file()

    metrics = evaluate(best, "test")
    assert metrics["n"] == 8 and metrics["mean_error_norm"] >= 0

    path = export(best, tmp_path / "gaze_face.onnx")
    graph = onnx.load(str(path)).graph
    assert [i.name for i in graph.input] == ["face_img"]
    assert [o.name for o in graph.output] == ["gaze"]
    dims = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim[1:]]
    assert dims == [3, 112, 112]
