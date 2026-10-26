#!/usr/bin/env python3
"""
Assembles a QCNet-ready torch_geometric HeteroData from BEVFusion track
histories + nuscenes_map_adapter map layers.

QCNet's pretrained AV2 checkpoint was trained at 10Hz (num_historical_steps=
50 -> 5s of history, dt=0.1s -- see checkpoints/QCNet_AV2.ckpt hyperparameters
inspected in changes.txt 2026-07-14). BEVFusion tracking runs at nuScenes'
2Hz keyframe cadence (dt=0.5s). Feeding 2Hz samples directly would make every
agent look like it's moving 5x faster than anything QCNet trained on, so
track histories are linearly interpolated onto QCNet's native 10Hz/50-step
grid before being handed to the model. This is a real domain-gap mitigation,
not a formality -- the model has no explicit dt input, it only sees raw
positions/velocities, so cadence mismatch would silently corrupt every
predicted speed.
"""
import numpy as np
import torch
from torch_geometric.data import HeteroData

from nuscenes_map_adapter import AGENT_TYPES

_VEHICLE_TYPE_IDX = AGENT_TYPES.index("vehicle")
_SCORED_TRACK_CATEGORY = 2  # QCNet AGENT_CATEGORIES.index('SCORED_TRACK'); not read by encoder/decoder, eval-only.


def _interp_history(history, num_steps: int, dt_native: float, dt_target: float):
    """
    history: list of (x, y, vx, vy, yaw, valid) oldest->newest at dt_native
    spacing, last entry = "now". Returns (position[num_steps,2],
    heading[num_steps], velocity[num_steps,2], valid_mask[num_steps]) on a
    uniform grid of spacing dt_target ending at "now", oldest->newest.
    """
    n = len(history)
    if n == 0:
        return (np.zeros((num_steps, 2), np.float32), np.zeros(num_steps, np.float32),
                np.zeros((num_steps, 2), np.float32), np.zeros(num_steps, bool))

    hist = np.asarray(history, dtype=np.float64)  # [n, 6]
    t_native = (np.arange(n) - (n - 1)) * dt_native            # ascending, 0 = "now"
    t_target = (np.arange(num_steps) - (num_steps - 1)) * dt_target

    x = np.interp(t_target, t_native, hist[:, 0], left=hist[0, 0])
    y = np.interp(t_target, t_native, hist[:, 1], left=hist[0, 1])
    vx = np.interp(t_target, t_native, hist[:, 2], left=hist[0, 2])
    vy = np.interp(t_target, t_native, hist[:, 3], left=hist[0, 3])

    yaw_unwrapped = np.unwrap(hist[:, 4])
    yaw = np.interp(t_target, t_native, yaw_unwrapped, left=yaw_unwrapped[0])
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    valid_native = hist[:, 5] > 0.5
    nearest_idx = np.clip(np.round((t_target - t_native[0]) / dt_native).astype(int), 0, n - 1)
    valid = (t_target >= t_native[0]) & valid_native[nearest_idx]

    position = np.stack([x, y], axis=1).astype(np.float32)
    velocity = np.stack([vx, vy], axis=1).astype(np.float32)
    return position, yaw.astype(np.float32), velocity, valid


def build_agent_block(tracks, ego_history, num_historical_steps: int, num_future_steps: int,
                       dt_native: float = 0.5, dt_target: float = 0.1) -> dict:
    """
    tracks: list of dicts {id: str, agent_type: int, history: [(x,y,vx,vy,yaw,valid), ...]}
    ego_history: [(x,y,vx,vy,yaw,valid), ...] for the ego vehicle, same native cadence.

    Returns a plain dict of numpy arrays (ego is always agent index 0 / av_index).
    Historical steps are filled from interpolated history; future steps are
    left zero/invalid -- QCNet's test-time inference doesn't need GT futures.
    """
    n = len(tracks) + 1
    T = num_historical_steps + num_future_steps

    position = np.zeros((n, T, 2), dtype=np.float32)
    heading = np.zeros((n, T), dtype=np.float32)
    velocity = np.zeros((n, T, 2), dtype=np.float32)
    valid_mask = np.zeros((n, T), dtype=bool)
    agent_type = np.zeros(n, dtype=np.uint8)
    ids = [None] * n

    def _fill(i, hist, atype, tid):
        pos, head, vel, valid = _interp_history(hist, num_historical_steps, dt_native, dt_target)
        position[i, :num_historical_steps] = pos
        heading[i, :num_historical_steps] = head
        velocity[i, :num_historical_steps] = vel
        valid_mask[i, :num_historical_steps] = valid
        agent_type[i] = atype
        ids[i] = tid

    _fill(0, ego_history, _VEHICLE_TYPE_IDX, "ego")
    for i, trk in enumerate(tracks, start=1):
        _fill(i, trk["history"], trk["agent_type"], trk["id"])

    # predict_mask: False for all historical steps; True for future steps
    # only if the agent is currently valid (mirrors argoverse_v2_dataset.py
    # get_agent_features, L243-245).
    predict_mask = np.zeros((n, T), dtype=bool)
    predict_mask[:, num_historical_steps:] = valid_mask[:, num_historical_steps - 1:num_historical_steps]

    return dict(
        num_nodes=n,
        av_index=0,
        id=ids,
        type=agent_type,
        category=np.full(n, _SCORED_TRACK_CATEGORY, dtype=np.uint8),
        position=position,
        heading=heading,
        velocity=velocity,
        valid_mask=valid_mask,
        predict_mask=predict_mask,
    )


def build_hetero_data(map_layers: dict, agent_block: dict) -> HeteroData:
    data = HeteroData()

    data["agent"].num_nodes = agent_block["num_nodes"]
    data["agent"].av_index = agent_block["av_index"]
    data["agent"].id = agent_block["id"]
    data["agent"].type = torch.from_numpy(agent_block["type"]).long()
    data["agent"].category = torch.from_numpy(agent_block["category"]).long()
    data["agent"].position = torch.from_numpy(agent_block["position"]).float()
    data["agent"].heading = torch.from_numpy(agent_block["heading"]).float()
    data["agent"].velocity = torch.from_numpy(agent_block["velocity"]).float()
    data["agent"].valid_mask = torch.from_numpy(agent_block["valid_mask"]).bool()
    data["agent"].predict_mask = torch.from_numpy(agent_block["predict_mask"]).bool()

    data["map_polygon"].num_nodes = int(map_layers["polygon_position"].shape[0])
    data["map_polygon"].position = torch.from_numpy(map_layers["polygon_position"]).float()
    data["map_polygon"].orientation = torch.from_numpy(map_layers["polygon_orientation"]).float()
    data["map_polygon"].type = torch.from_numpy(map_layers["polygon_type"]).long()
    data["map_polygon"].is_intersection = torch.from_numpy(map_layers["polygon_is_intersection"]).long()

    data["map_point"].num_nodes = int(map_layers["point_position"].shape[0])
    data["map_point"].position = torch.from_numpy(map_layers["point_position"]).float()
    data["map_point"].orientation = torch.from_numpy(map_layers["point_orientation"]).float()
    data["map_point"].magnitude = torch.from_numpy(map_layers["point_magnitude"]).float()
    data["map_point"].type = torch.from_numpy(map_layers["point_type"]).long()
    data["map_point"].side = torch.from_numpy(map_layers["point_side"]).long()

    data["map_point", "to", "map_polygon"].edge_index = torch.from_numpy(
        map_layers["point_to_polygon_edge_index"]).long()
    data["map_polygon", "to", "map_polygon"].edge_index = torch.from_numpy(
        map_layers["polygon_to_polygon_edge_index"]).long()
    data["map_polygon", "to", "map_polygon"].type = torch.from_numpy(
        map_layers["polygon_to_polygon_type"]).long()

    return data
