#!/usr/bin/env python3
"""
QCNet inference wrapper. Loads the pretrained AV2 checkpoint and turns a
HeteroData built by qcnet_data_adapter.py into per-agent future trajectories
in the global nuScenes frame -- the same [(positions[n,3], label), ...]
shape MultiObjectTracker.get_future_trajectories() already returns, so it's
a drop-in replacement for the physics-based rollout.

Runs in the QCNet conda env only (torch 2.0.1+cu118 / PyG), which is
intentionally incompatible with the bevfusion env's torch 1.13.1+cu117 --
see changes.txt 2026-07-14 for why this had to become a separate process
rather than an in-process import (mmcv/mmdet3d's compiled extensions are
built against bevfusion's torch and would break if this env's torch were
imported into the same interpreter).
"""
import os
import sys

import numpy as np
import torch

# The `predictors` package lives inside the QCNet checkout, not on the
# default path when this module is imported from bevfusion's repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "QCNet"))
from predictors import QCNet


class QCNetPredictor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = QCNet.load_from_checkpoint(checkpoint_path=checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        self.num_historical_steps = self.model.num_historical_steps
        self.num_future_steps = self.model.num_future_steps
        self.dt_target = 0.1  # QCNet's native step spacing (10Hz, see qcnet_data_adapter.py)

    @torch.no_grad()
    def _forward(self, data):
        data = data.to(self.device)
        pred = self.model(data)
        loc = pred["loc_refine_pos"][..., :2]                 # [N, num_modes, num_future_steps, 2]
        pi = torch.softmax(pred["pi"], dim=-1)                # [N, num_modes]

        h = self.num_historical_steps
        origin = data["agent"].position[:, h - 1, :2]         # [N, 2]
        theta = data["agent"].heading[:, h - 1]                # [N]
        cos, sin = theta.cos(), theta.sin()
        n = loc.size(0)
        rot = torch.zeros(n, 2, 2, device=self.device)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        # local (agent-frame-at-last-history-step) -> global, exactly
        # mirroring QCNet.test_step's own eval transform (predictors/qcnet.py
        # L308-317), just applied to every agent instead of only category==3.
        traj_global = torch.matmul(loc, rot.unsqueeze(1)) + origin.reshape(-1, 1, 1, 2)

        return traj_global.cpu().numpy(), pi.cpu().numpy()

    def predict_futures(self, data, track_ids, track_z, n_steps: int = 10, dt_native: float = 0.5):
        """
        data: HeteroData from qcnet_data_adapter.build_hetero_data (agent 0 = ego).
        track_ids: list of agent ids in the SAME order as build_agent_block's
                   `tracks` argument (ego excluded -- this only returns
                   predictions for the real tracks, not ego itself).
        track_z: dict track_id -> z_bottom (held constant, matching the
                 physics trackers' predict_future()).
        n_steps / dt_native: caller's desired output cadence (e.g. the
                   renderer's existing 2Hz future_steps), downsampled from
                   QCNet's native 10Hz/60-step output.

        Returns dict track_id -> dict(positions=[n_steps,3] best-mode xyz,
        all_modes=[num_modes,n_steps,2], probs=[num_modes]).
        """
        traj_global, probs = self._forward(data)  # [N, K, T, 2], [N, K]

        # Downsample QCNet's 10Hz future to the caller's cadence; clip to the
        # model's 6s horizon if the caller asks for more than that.
        idx = np.clip(np.round((np.arange(1, n_steps + 1) * dt_native) / self.dt_target).astype(int) - 1,
                       0, self.num_future_steps - 1)

        out = {}
        for i, tid in enumerate(track_ids, start=1):  # agent index 0 is ego, skip it
            best_mode = int(np.argmax(probs[i]))
            xy_best = traj_global[i, best_mode][idx]           # [n_steps, 2]
            # traj_global[i, :, idx] would put the idx axis first (numpy
            # advanced-indexing quirk) -- index sub-array separately instead
            # to keep an explicit [num_modes, n_steps, 2] shape.
            xy_all = traj_global[i][:, idx, :]                  # [num_modes, n_steps, 2]
            z = track_z.get(tid, 0.0)
            positions = np.concatenate([xy_best, np.full((n_steps, 1), z, dtype=np.float32)], axis=1)
            out[tid] = dict(positions=positions, all_modes=xy_all, probs=probs[i])
        return out
