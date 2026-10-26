#!/usr/bin/env python3
"""
Runs QCNet inference over track histories dumped by track_and_visualize.py
(--dump-qcnet-input), producing a futures pkl consumable by
track_and_visualize.py --future-predictor qcnet --qcnet-futures PKL.

Output schema: futures_all[scene_idx][sample_idx] = {track_id: dict(
    all_modes=[num_modes, n_steps, 2], probs=[num_modes], label=int, z=float)}
-- keeps all 6 modes + probs (not just the top-1 trajectory) so
track_and_visualize.py can pick either the top-1-by-confidence mode or the
nearest-to-IMM-rollout mode-selection heuristic at render time
(--qcnet-mode-select), without a separate QCNet-env run per choice.

Must run in the QCNet conda env (torch 2.0.1+cu118 / PyG) -- NOT the
bevfusion env (torch 1.13.1+cu117). See qcnet_predictor.py's module
docstring for why these can't be merged into one env. Needs no mmcv/mmdet3d,
only the dump pkl + nuScenes map data on disk.

Usage:
    conda run -n QCNet python qcnet_infer.py \\
        --dump tracked_scenes/qcnet_input.pkl \\
        --out  tracked_scenes/qcnet_futures.pkl \\
        --dataroot data/nuscenes --future-steps 10
"""
import argparse
import json
import os
import pickle

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes_map_adapter import build_map_layers, OBJECT_CLASS_TO_AGENT_TYPE
from qcnet_data_adapter import build_agent_block, build_hetero_data
from qcnet_predictor import QCNetPredictor

# Must match track_and_visualize.py's OBJECT_CLASSES index order exactly --
# duplicated here rather than imported since that module pulls in mmcv/
# mmdet3d, which aren't installed (and would conflict) in the QCNet env.
OBJECT_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
# barrier/traffic_cone are static map furniture, not agents worth handing to
# a trajectory predictor -- skip them (no future-trajectory line for them
# either way, same as a physics predictor would trivially return ~0 motion).
_SKIP_LABELS = {"barrier", "traffic_cone"}

MAP_QUERY_RADIUS = 200.0  # meters around ego; generous vs. QCNet's own pl2pl_radius=150


class SceneLocationResolver:
    """
    Resolves a sample token -> map location string without paying for the
    full NuScenes(...) devkit load, which unconditionally parses
    sample_data.json + ego_pose.json + sample_annotation.json (~2.5GB of
    JSON on trainval) even though only scene.json/log.json/sample.json
    (a few MB total) are needed for this lookup. On a machine shared with
    other users' jobs, the full load was observed to push the box into
    heavy swapping (see changes.txt 2026-07-14) -- this avoids that.
    """

    def __init__(self, dataroot: str, version: str):
        table_dir = os.path.join(dataroot, version)
        with open(os.path.join(table_dir, "sample.json")) as f:
            samples = json.load(f)
        with open(os.path.join(table_dir, "scene.json")) as f:
            scenes = json.load(f)
        with open(os.path.join(table_dir, "log.json")) as f:
            logs = json.load(f)

        self._sample_to_scene = {s["token"]: s["scene_token"] for s in samples}
        self._scene_to_log = {s["token"]: s["log_token"] for s in scenes}
        self._log_to_location = {l["token"]: l["location"] for l in logs}

    def resolve(self, sample_token: str) -> str:
        scene_token = self._sample_to_scene[sample_token]
        log_token = self._scene_to_log[scene_token]
        return self._log_to_location[log_token]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, metavar="PKL",
                     help="Input pkl from track_and_visualize.py --dump-qcnet-input")
    ap.add_argument("--out", required=True, metavar="PKL")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--nusc-version", default="v1.0-trainval")
    ap.add_argument("--checkpoint", default="QCNet/checkpoints/QCNet_AV2.ckpt")
    ap.add_argument("--future-steps", type=int, default=10,
                     help="Must match track_and_visualize.py's --future-steps for the "
                          "eventual render (default 10, i.e. 5s at the renderer's 2Hz).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.dump, "rb") as f:
        dump = pickle.load(f)
    print(f"[qcnet_infer] Loaded dump: {len(dump)} scene(s).")

    print("[qcnet_infer] Loading scene/log/sample tables (for scene->map location lookup)...")
    location_resolver = SceneLocationResolver(args.dataroot, args.nusc_version)

    print(f"[qcnet_infer] Loading QCNet checkpoint from {args.checkpoint} ...")
    predictor = QCNetPredictor(args.checkpoint, device=args.device)
    print(f"[qcnet_infer] num_historical_steps={predictor.num_historical_steps} "
          f"num_future_steps={predictor.num_future_steps}")

    map_cache = {}  # location -> NuScenesMap
    futures_all = {}

    for scene_idx, frames in dump.items():
        if not frames:
            continue
        location = location_resolver.resolve(frames[0]["token"])
        if location not in map_cache:
            map_cache[location] = NuScenesMap(dataroot=args.dataroot, map_name=location)
        nusc_map = map_cache[location]
        print(f"[qcnet_infer] Scene {scene_idx}: {len(frames)} frames, map={location}")

        scene_futures = {}
        for frame in frames:
            tracks = []
            track_z = {}
            for t in frame["tracks"]:
                label_name = OBJECT_CLASSES[t["label"]]
                if label_name in _SKIP_LABELS:
                    continue
                if len(t["history"]) < 2:
                    continue  # too young to give QCNet a meaningful history signal
                tracks.append(dict(id=t["id"], agent_type=OBJECT_CLASS_TO_AGENT_TYPE[label_name],
                                    history=t["history"]))
                track_z[t["id"]] = t["z"]

            if not tracks:
                scene_futures[frame["sample_idx"]] = {}
                continue

            map_layers = build_map_layers(nusc_map, frame["ego_xy"], radius=MAP_QUERY_RADIUS)
            agent_block = build_agent_block(tracks, frame["ego_history"],
                                             predictor.num_historical_steps,
                                             predictor.num_future_steps)
            data = build_hetero_data(map_layers, agent_block)
            track_ids = [t["id"] for t in tracks]
            preds = predictor.predict_futures(data, track_ids, track_z, n_steps=args.future_steps)

            # Keyed by track id (not a positional list) and keeping all 6 modes +
            # probs, not just the top-1 positions -- so track_and_visualize.py can
            # choose top-1-by-confidence OR the nearest-to-IMM-rollout heuristic
            # (--qcnet-mode-select) at render time without a separate QCNet-env
            # run. id -> original int label, recovered from frame["tracks"].
            id_to_label = {t["id"]: t["label"] for t in frame["tracks"]}
            scene_futures[frame["sample_idx"]] = {
                tid: dict(all_modes=preds[tid]["all_modes"], probs=preds[tid]["probs"],
                          label=id_to_label[tid], z=track_z[tid])
                for tid in track_ids
            }

        futures_all[scene_idx] = scene_futures

    with open(args.out, "wb") as f:
        pickle.dump(futures_all, f)
    print(f"[qcnet_infer] Saved futures for {len(futures_all)} scene(s) → {args.out}")


if __name__ == "__main__":
    main()
