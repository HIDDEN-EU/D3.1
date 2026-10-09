"""
Per-agent deviation dump: at cells several agents contribute to, is one agent
identifiably the odd one out?

The published ``tr Sigma`` sums the agents out, so it can only say "the agents
disagree here". This records the terms before that sum -- how far each agent's
feature direction sits from the consensus -- for cells with at least
``MIN_AGENTS`` contributors, which is where "which agent is wrong" is even a
well-posed question (with two agents the consensus sits between them by
construction).

Each row is one cell at one level:
    dev_0..dev_4   per-agent squared distance to the consensus direction
    w_0..w_4       that agent's fusion weight at the cell
    mod_0..mod_4   that agent's modality (0 = absent, 1 = m1, 3 = m3, 4 = m4)
    is_vehicle     a ground-truth vehicle covers the cell
"""

import numpy as np
import torch

from opencood.tools.levy.levy_cells import _polygon_mask
from opencood.tools.levy.levy_collect import frame_identity
from opencood.utils import common_utils

MIN_AGENTS = 3
MAX_AGENTS = 5
CELLS_PER_FRAME = 200
MODALITY_CODE = {"m1": 1, "m2": 2, "m3": 3, "m4": 4}


class AgentDeviationRecorder:
    """Samples cells with several contributing agents over a run."""

    def __init__(self, pc_range, det_range, seed=0):
        self.pc_range = pc_range
        self.det_range = det_range
        self.rng = np.random.default_rng(seed)
        self.rows = []

    def add(self, frame, dataset, use_cav, batch_data, infer_result, uncertainty,
            agent_deviation):
        modalities = batch_data["ego"].get("agent_modality_list", [])
        gt = infer_result["gt_box_tensor"]
        gt_corners = (np.zeros((0, 8, 3), np.float32) if gt is None
                      else common_utils.torch_tensor_to_numpy(gt).astype(np.float32))
        scenario, _ = frame_identity(dataset, frame)

        for level, per_batch in enumerate(agent_deviation):
            maps = per_batch[0]  # batch size 1 at inference
            dev, weight = maps["deviation"], maps["weight"]
            n_agents, _, H, W = dev.shape
            n_valid = uncertainty["per_scale"][level]["n_valid"][0, 0].cpu().numpy()

            x_min, y_min, _, x_max, y_max, _ = self.pc_range
            dx, dy = (x_max - x_min) / W, (y_max - y_min) / H
            xs = x_min + (np.arange(W) + 0.5) * dx
            ys = y_min + (np.arange(H) + 0.5) * dy
            in_range = ((ys >= self.det_range[1]) & (ys <= self.det_range[4]))[:, None] & (
                (xs >= self.det_range[0]) & (xs <= self.det_range[3]))[None, :]

            pick = np.argwhere((n_valid >= MIN_AGENTS) & in_range)
            if len(pick) == 0:
                continue
            if len(pick) > CELLS_PER_FRAME:
                pick = pick[self.rng.choice(len(pick), CELLS_PER_FRAME, replace=False)]

            vehicle = _polygon_mask(gt_corners, (H, W), self.pc_range)
            dev_np = dev[:, 0].cpu().numpy()
            w_np = weight[:, 0].cpu().numpy()
            valid_np = maps["valid"][:, 0].cpu().numpy()
            for r, c in pick:
                row = {
                    "frame": frame, "scenario": scenario, "use_cav": use_cav,
                    "level": level, "n_valid": int(n_valid[r, c]),
                    "x": float(xs[c]), "y": float(ys[r]),
                    "dist": float(np.hypot(xs[c], ys[r])),
                    "is_vehicle": int(vehicle[r, c]),
                }
                for j in range(MAX_AGENTS):
                    inside = j < n_agents and valid_np[j, r, c]
                    row[f"dev_{j}"] = float(dev_np[j, r, c]) if inside else np.nan
                    row[f"w_{j}"] = float(w_np[j, r, c]) if inside else np.nan
                    row[f"mod_{j}"] = (MODALITY_CODE.get(modalities[j], 0)
                                       if j < len(modalities) and inside else 0)
                self.rows.append(row)

    def write(self, path_prefix):
        if not self.rows:
            return []
        keys = list(self.rows[0])
        out = {k: np.array([r[k] for r in self.rows]) for k in keys}
        path = f"{path_prefix}_agentdev.npz"
        np.savez_compressed(path, **out)
        return [path]


@torch.no_grad()
def record_frame(recorder, levy_fusion, **kwargs):
    if levy_fusion.agent_deviation_output is not None:
        recorder.add(uncertainty=levy_fusion.uncertainty_output,
                     agent_deviation=levy_fusion.agent_deviation_output, **kwargs)
