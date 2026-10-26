#!/usr/bin/env python3
"""
nuScenes -> QCNet map schema adapter.

Builds map_polygon / map_point layers matching QCNet's Argoverse-2 HeteroData
schema (see QCNet/datasets/argoverse_v2_dataset.py get_map_features, and the
research summary in changes.txt 2026-07-14) from a NuScenesMap query around a
point. Pure numpy in, numpy out -- no torch here, so this module can be
tested without a GPU/QCNet env. qcnet_data_adapter.py converts the output to
torch tensors when assembling the HeteroData passed to the model.

Categorical indices below are copied verbatim from QCNet's
ArgoverseV2Dataset.__init__ (QCNet/datasets/argoverse_v2_dataset.py:139-149)
-- they must match the order the pretrained embedding tables were trained
on, not just the string labels.

Documented simplifications vs AV2 (nuScenes map layers have no 1:1
equivalent for these):
  - map_point only carries CENTERLINE points (side='CENTER'). AV2 also
    encodes left/right lane-boundary polylines with fine-grained marking
    types (e.g. DASH_SOLID_YELLOW); nuScenes lane records expose no boundary
    polyline or line-style metadata in that form, so boundaries are omitted
    rather than faked.
  - map_polygon type is always 'VEHICLE' -- nuScenes' base 'lane' layer
    doesn't distinguish bike/bus lanes the way AV2 does.
  - is_intersection = True for 'lane_connector' records (junction-crossing
    lanes), False for plain 'lane' records -- an approximation of AV2's
    explicit intersection flag.
  - polygon_to_polygon edges only cover PRED/SUCC (from
    get_incoming_lane_ids/get_outgoing_lane_ids). LEFT/RIGHT neighbor
    topology isn't readily queryable from the nuScenes map API and is
    omitted (edge type stays absent, not filled with 'NONE' padding).
  - Pedestrian crossings are omitted -- out of scope for the vehicle
    roundabout/turn prediction this was built for.
"""
import numpy as np

AGENT_TYPES = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
               'construction', 'riderless_bicycle', 'unknown']
AGENT_CATEGORIES = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
POLYGON_TYPES = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
POLYGON_IS_INTERSECTIONS = [True, False, None]
POINT_TYPES = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
               'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
               'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
               'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
POINT_SIDES = ['LEFT', 'RIGHT', 'CENTER']
POLYGON_TO_POLYGON_TYPES = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

# BEVFusion's 10 detection classes -> QCNet agent type index. barrier/
# traffic_cone map to 'static' since they're map furniture, not agents
# meant to be handed to the trajectory predictor (callers should skip
# QCNet prediction for those two classes entirely).
OBJECT_CLASS_TO_AGENT_TYPE = {
    "car": AGENT_TYPES.index("vehicle"),
    "truck": AGENT_TYPES.index("vehicle"),
    "construction_vehicle": AGENT_TYPES.index("construction"),
    "bus": AGENT_TYPES.index("bus"),
    "trailer": AGENT_TYPES.index("vehicle"),
    "barrier": AGENT_TYPES.index("static"),
    "motorcycle": AGENT_TYPES.index("motorcyclist"),
    "bicycle": AGENT_TYPES.index("cyclist"),
    "pedestrian": AGENT_TYPES.index("pedestrian"),
    "traffic_cone": AGENT_TYPES.index("static"),
}

_CENTERLINE_TYPE_IDX = POINT_TYPES.index("CENTERLINE")
_CENTER_SIDE_IDX = POINT_SIDES.index("CENTER")
_VEHICLE_POLY_TYPE_IDX = POLYGON_TYPES.index("VEHICLE")
_INTERSECTION_TRUE_IDX = POLYGON_IS_INTERSECTIONS.index(True)
_INTERSECTION_FALSE_IDX = POLYGON_IS_INTERSECTIONS.index(False)
_PRED_EDGE_IDX = POLYGON_TO_POLYGON_TYPES.index("PRED")
_SUCC_EDGE_IDX = POLYGON_TO_POLYGON_TYPES.index("SUCC")


def build_map_layers(nusc_map, center_xy, radius: float = 200.0, resolution_m: float = 1.0) -> dict:
    """
    Query lanes/lane_connectors around center_xy and build QCNet-schema map
    layers. All coordinates are in the nuScenes global map frame (meters),
    matching the frame track histories are recorded in.

    Returns a dict of numpy arrays:
      polygon_position   [P, 2] float32   -- first centerline point of each lane
      polygon_orientation [P]   float32   -- heading (rad) at that first point
      polygon_type        [P]   uint8
      polygon_is_intersection [P] uint8
      point_position      [M, 2] float32
      point_orientation   [M]   float32
      point_magnitude     [M]   float32   -- length of the segment leaving this point
      point_type          [M]   uint8
      point_side          [M]   uint8
      point_to_polygon_edge_index [2, M] int64   -- point idx -> owning polygon idx
      polygon_to_polygon_edge_index [2, E] int64
      polygon_to_polygon_type [E] uint8
    """
    cx, cy = center_xy
    lane_ids = nusc_map.get_records_in_radius(cx, cy, radius, ['lane', 'lane_connector'])
    lane_ids = list(dict.fromkeys(lane_ids['lane'] + lane_ids['lane_connector']))

    if not lane_ids:
        return _empty_map_layers()

    connector_id_set = set(nusc_map.get_records_in_radius(cx, cy, radius, ['lane_connector'])['lane_connector'])

    discretized = nusc_map.discretize_lanes(lane_ids, resolution_meters=resolution_m)
    # discretize_lanes silently drops ids it can't resolve (e.g. zero-length
    # lanes) -- keep only ids that actually produced a polyline.
    lane_ids = [lid for lid in lane_ids if discretized.get(lid) and len(discretized[lid]) >= 2]

    polygon_position, polygon_orientation = [], []
    polygon_type, polygon_is_intersection = [], []
    point_position, point_orientation, point_magnitude = [], [], []
    point_type, point_side, point_to_polygon = [], [], []
    id_to_polygon_idx = {}

    for p_idx, lid in enumerate(lane_ids):
        poses = np.asarray(discretized[lid], dtype=np.float32)  # [K, 3] = (x, y, yaw)
        id_to_polygon_idx[lid] = p_idx
        xy = poses[:, :2]

        # Match QCNet's own AV2 point convention exactly (get_map_features,
        # argoverse_v2_dataset.py:310-322): points are the K-1 *segment
        # start* points (last point dropped -- it has no outgoing vector),
        # with orientation/magnitude taken from the vector to the next
        # point, not the discretized pose's own heading. A frozen pretrained
        # model is sensitive to this feature distribution, so it's worth
        # matching rather than approximating.
        seg_vec = xy[1:] - xy[:-1]
        seg_len = np.linalg.norm(seg_vec, axis=1)
        seg_orient = np.arctan2(seg_vec[:, 1], seg_vec[:, 0])

        polygon_position.append(xy[0])
        polygon_orientation.append(seg_orient[0] if len(seg_orient) else poses[0, 2])
        polygon_type.append(_VEHICLE_POLY_TYPE_IDX)
        polygon_is_intersection.append(_INTERSECTION_TRUE_IDX if lid in connector_id_set else _INTERSECTION_FALSE_IDX)

        n_pts = len(seg_vec)
        if n_pts == 0:
            # Degenerate (near zero-length) lane after discretization -- skip
            # its map_point contribution but keep the polygon node itself.
            continue
        point_position.append(xy[:-1])
        point_orientation.append(seg_orient)
        point_magnitude.append(seg_len.astype(np.float32))
        point_type.append(np.full(n_pts, _CENTERLINE_TYPE_IDX, dtype=np.uint8))
        point_side.append(np.full(n_pts, _CENTER_SIDE_IDX, dtype=np.uint8))
        point_to_polygon.append(np.full(n_pts, p_idx, dtype=np.int64))

    if not id_to_polygon_idx or not point_position:
        return _empty_map_layers()

    pp_src, pp_dst, pp_type = [], [], []
    for lid, p_idx in id_to_polygon_idx.items():
        for succ in nusc_map.get_outgoing_lane_ids(lid):
            if succ in id_to_polygon_idx:
                pp_src.append(p_idx); pp_dst.append(id_to_polygon_idx[succ]); pp_type.append(_SUCC_EDGE_IDX)
        for pred in nusc_map.get_incoming_lane_ids(lid):
            if pred in id_to_polygon_idx:
                pp_src.append(p_idx); pp_dst.append(id_to_polygon_idx[pred]); pp_type.append(_PRED_EDGE_IDX)

    point_pos_arr = np.concatenate(point_position, axis=0)
    m_idx = np.arange(len(point_pos_arr), dtype=np.int64)
    pt2poly = np.concatenate(point_to_polygon, axis=0)

    return dict(
        polygon_position=np.asarray(polygon_position, dtype=np.float32),
        polygon_orientation=np.asarray(polygon_orientation, dtype=np.float32),
        polygon_type=np.asarray(polygon_type, dtype=np.uint8),
        polygon_is_intersection=np.asarray(polygon_is_intersection, dtype=np.uint8),
        point_position=point_pos_arr,
        point_orientation=np.concatenate(point_orientation, axis=0).astype(np.float32),
        point_magnitude=np.concatenate(point_magnitude, axis=0).astype(np.float32),
        point_type=np.concatenate(point_type, axis=0),
        point_side=np.concatenate(point_side, axis=0),
        point_to_polygon_edge_index=np.stack([m_idx, pt2poly], axis=0),
        polygon_to_polygon_edge_index=(np.asarray([pp_src, pp_dst], dtype=np.int64)
                                        if pp_src else np.zeros((2, 0), dtype=np.int64)),
        polygon_to_polygon_type=np.asarray(pp_type, dtype=np.uint8),
    )


def _empty_map_layers() -> dict:
    return dict(
        polygon_position=np.zeros((0, 2), dtype=np.float32),
        polygon_orientation=np.zeros((0,), dtype=np.float32),
        polygon_type=np.zeros((0,), dtype=np.uint8),
        polygon_is_intersection=np.zeros((0,), dtype=np.uint8),
        point_position=np.zeros((0, 2), dtype=np.float32),
        point_orientation=np.zeros((0,), dtype=np.float32),
        point_magnitude=np.zeros((0,), dtype=np.float32),
        point_type=np.zeros((0,), dtype=np.uint8),
        point_side=np.zeros((0,), dtype=np.uint8),
        point_to_polygon_edge_index=np.zeros((2, 0), dtype=np.int64),
        polygon_to_polygon_edge_index=np.zeros((2, 0), dtype=np.int64),
        polygon_to_polygon_type=np.zeros((0,), dtype=np.uint8),
    )
