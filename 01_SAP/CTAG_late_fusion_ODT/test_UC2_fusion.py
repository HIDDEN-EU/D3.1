"""
Tests of the UC2 fusion (run from this folder:  python -m pytest -q test_UC2_fusion.py).

The synthetic scenario of uwb_sim.py is the test bed: a VRU with a UWB tag, a neighbour walking 1-1.5 m beside it,
LiDAR occlusion 8-11 s (new LiDAR id afterwards), LiDAR id swap with the neighbour at 14 s, UWB dropout 16-19 s,
UWB latency/jitter and multipath outliers. Thresholds are loose enough for the random seeds, tight enough to fail if
the fusion stops using one of the sensors.
"""

import copy
import functools
import math

import numpy as np
import pytest

from evaluate import run, summarize
from UC2_fusion import FusionConfig, MapFrame, Meas, UC2Fusion, VRUFilter
from uwb_sim import LIDAR_SENDER, ORIGIN_LLA, TAG_ID, UWB_SENDER, ScenarioConfig, generate

BASE = FusionConfig().coop_id_base


@functools.lru_cache(maxsize=None)
def scenario(hz=10.0, seed=0, **kw):
    return generate(ScenarioConfig(uwb_hz=hz, seed=seed, **dict(kw)))


@functools.lru_cache(maxsize=None)
def result(hz=10.0, seed=0, extra_delay=0.0, **kw):
    sc = scenario(hz, seed, **kw)
    r = run(sc, extra_delay=extra_delay)
    return sc, r, summarize(r)


# ---------------------------------------------------------------- building blocks

def test_map_frame_roundtrip():
    mf = MapFrame(ORIGIN_LLA)
    p = np.array([123.4, -56.7, 2.0])
    assert np.allclose(mf.lla_to_enu(*mf.enu_to_lla(p)), p, atol=1e-3)


def test_filter_out_of_sequence_equals_in_order():
    rng = np.random.default_rng(1)
    ms = []
    for k in range(12):
        t = 0.1 * k
        z = np.array([1.0 * t, 0.5 * t, 1.0, 0.5]) + rng.normal(0, 0.05, 4)
        ms.append(Meas(t, "uwb" if k % 3 == 0 else "lidar", z, np.diag([.01, .01, .04, .04])))
    mk = lambda: VRUFilter(np.array([0, 0, 1.0, 0.5]), np.eye(4), 0.0, 1.0, 5.0)
    a = mk()
    for m in ms:
        a.add(m)
    b = mk()
    order = [0, 2, 1, 4, 3, 5, 7, 6, 8, 10, 9, 11]                 # swapped neighbours = late arrivals
    for i in order:
        b.add(ms[i])
    assert np.allclose(a.x, b.x, atol=1e-9) and np.allclose(a.P, b.P, atol=1e-9)
    assert np.allclose(a.state_at(0.55)[0], b.state_at(0.55)[0], atol=1e-9)


def test_filter_rejects_measurement_older_than_history():
    f = VRUFilter(np.zeros(4), np.eye(4), 10.0, 1.0, 0.5)
    for k in range(20):
        f.add(Meas(10.0 + 0.1 * k, "uwb", np.zeros(4), np.eye(4) * 0.1))
    assert f.add(Meas(10.2, "uwb", np.zeros(4), np.eye(4) * 0.1)) is False


# ---------------------------------------------------------------- fusion quality

def test_fusion_beats_both_sensors_at_10hz():
    for seed in (0, 1, 2):
        _, _, s = result(10.0, seed)
        assert s["fused_rmse"] < 0.2
        assert s["fused_rmse"] < 0.4 * s["uwb_rmse"]
        assert s["fused_rmse"] < 0.6 * s["lidar_rmse"]
        assert s["fused_avail"] == 1.0
        assert s["id_changes"] == 0


@pytest.mark.parametrize("hz", [1.0, 2.0, 5.0])
def test_low_uwb_rates_stay_available_and_accurate(hz):
    """A stateless per-cycle fusion would drop a UWB sample older than ~0.3 s: below ~3 Hz it would lose the UWB."""
    _, r, s = result(hz, 0)
    assert s["fused_avail"] == 1.0
    assert s["fused_rmse"] < 0.3 and s["fused_rmse"] < s["uwb_rmse"]
    assert s["id_changes"] == 0


def test_lidar_occlusion_is_bridged_by_uwb():
    sc, r, s = result(10.0, 0)
    assert s["fused_avail_occl"] == 1.0 and s["lidar_avail_occl"] < 0.3
    assert s["fused_rmse_occl"] < 0.25 and s["fused_rmse_occl"] < s["uwb_rmse_occl"]
    inside = [st for st, o in zip(r.status, r.occluded) if o]
    assert inside.count("uwb_only") > 0.8 * len(inside)


def test_uwb_dropout_is_bridged_by_lidar():
    sc, r, s = result(10.0, 0)
    assert s["fused_rmse_uwbdown"] < 0.2
    down = [st for st, d in zip(r.status, r.uwb_down) if d]
    assert down.count("lidar_only") > 0.5 * len(down)
    assert "coasting" not in down


def test_lidar_relinks_after_occlusion_with_new_id():
    sc, r, _ = result(10.0, 0)
    t = r.t - sc.t_start
    after = [st for st, tt in zip(r.status, t) if 12.0 <= tt < 13.5]
    assert after.count("fused") > 0.9 * len(after)                  # the new LiDAR id 47 was linked again


def test_lidar_id_swap_with_neighbour_is_corrected():
    sc, r, s = result(10.0, 0)
    assert r.link_wrong.sum() <= 6                                  # a few cycles at most until the swap is noticed
    assert s["fused_rmse"] < 0.2
    assert r.stats.get("links_challenged", 0) + r.stats.get("links_uwb_disagree", 0) + r.stats.get("links_lost", 0) >= 1


def test_no_wrong_links_without_id_swap():
    for seed in (0, 1, 2):
        _, r, s = result(10.0, seed, swap_time=None)
        assert s["link_correct"] > 0.995


def test_uwb_multipath_outliers_are_rejected():
    _, r, s = result(10.0, 0, uwb_outlier_prob=0.10)
    assert r.stats["uwb_outliers"] > 5
    assert s["fused_rmse"] < 0.2


def test_late_and_reordered_messages_are_handled():
    _, _, s0 = result(10.0, 0)
    _, _, s1 = result(10.0, 0, extra_delay=0.4)
    assert s1["fused_rmse"] < s0["fused_rmse"] + 0.1
    assert s1["id_changes"] == 0


def test_no_duplicate_of_the_linked_vru():
    _, r, s = result(10.0, 0)
    assert s["ghost_frac"] < 0.05


# ---------------------------------------------------------------- interface

def _fuse_all(sc, until):
    f = UC2Fusion()
    for arr, m in sc.events:
        if arr <= until:
            f.push(m)
    return f, f.output(until)


def test_output_message_follows_odt_schema():
    sc = scenario(10.0, 0)
    f, out = _fuse_all(sc, sc.t_start + 5.0)
    assert out["metadata"]["message_type"] == "Ego_3D_ODT" and out["reference_frame"]["coordinate_frame"] == "ego"
    assert set(out["sender_state"]["position"]) == {"latitude", "longitude", "altitude"}
    need = {"track_id", "category", "score", "timestamp", "x", "y", "z", "length", "width", "height", "heading",
            "yaw_rate", "vx", "vy", "optional"}
    ids = [d["track_id"] for d in out["detections"]]
    assert len(ids) == len(set(ids))
    for d in out["detections"]:
        assert need <= set(d) and all(math.isfinite(d[k]) for k in need - {"optional", "category"})
        assert 0.0 <= d["score"] <= 1.0
    coop = [d for d in out["detections"] if d["track_id"] == BASE + TAG_ID]
    assert len(coop) == 1 and coop[0]["fusion_info"]["tag_id"] == TAG_ID and coop[0]["category"] == "Pedestrian"
    assert {s["sender_id"] for s in coop[0]["sources"]} == {UWB_SENDER, LIDAR_SENDER}
    # the cars of the scene are passed through with their LiDAR ids
    assert {41, 42} <= set(ids) or {40, 41, 42} & set(ids)


def test_linked_lidar_box_is_not_published_twice():
    sc = scenario(10.0, 0)
    f, out = _fuse_all(sc, sc.t_start + 5.0)
    link = f.tracks[TAG_ID].link_id
    assert link is not None and link not in [d["track_id"] for d in out["detections"]]


def test_uwb_alone_gives_a_vru_with_default_size():
    sc = scenario(10.0, 0)
    f = UC2Fusion()
    for arr, m in sc.events:
        if m["metadata"]["sender_id"] == UWB_SENDER and arr <= sc.t_start + 5.0:
            f.push(m)
    out = f.output(sc.t_start + 5.0)
    assert len(out["detections"]) == 1
    d = out["detections"][0]
    assert d["fusion_info"]["status"] == "uwb_only" and (d["length"], d["width"], d["height"]) == (0.6, 0.6, 1.75)
    assert abs(d["z"] - 0.875) < 1e-6


def test_two_tags_are_two_objects():
    sc = scenario(10.0, 0)
    f = UC2Fusion()
    for arr, m in sc.events:
        if m["metadata"]["sender_id"] != UWB_SENDER or arr > sc.t_start + 5.0:
            continue
        f.push(m)
        m2 = copy.deepcopy(m)
        m2["detections"][0].update(track_id=8, x=m["detections"][0]["x"] + 10.0)
        f.push(m2)
    out = f.output(sc.t_start + 5.0)
    xs = {d["track_id"]: d["x"] for d in out["detections"]}
    assert set(xs) == {BASE + 7, BASE + 8} and abs((xs[BASE + 8] - xs[BASE + 7]) - 10.0) < 0.5


def test_two_equally_good_boxes_are_not_linked():
    """Two LiDAR boxes at the same distance from the tag: the fusion must not guess."""
    mf = MapFrame(ORIGIN_LLA)
    ss = {"agent_type": "ego", "position": dict(zip(("latitude", "longitude", "altitude"), mf.enu_to_lla(np.zeros(3)))),
          "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}}
    hdr = lambda sid: {"sender_id": sid, "sequence_number": 1, "send_timestamp": 0.0}
    frame = {"reference_frame": {"coordinate_frame": "ego"}, "sender_state": ss}
    f = UC2Fusion()
    for k in range(5):
        t = 100.0 + 0.1 * k
        f.push({**frame, "metadata": hdr(UWB_SENDER), "detections": [
            {"track_id": 7, "score": 0.95, "timestamp": t, "x": 10.0, "y": 0.0, "vx": 0.0, "vy": 0.0}]})
        box = lambda i, y: {"track_id": i, "category": "pedestrian", "score": 0.8, "timestamp": t, "x": 10.0, "y": y,
                            "z": 0.9, "length": .6, "width": .6, "height": 1.75, "heading": 0.0, "vx": 0.0, "vy": 0.0}
        f.push({**frame, "metadata": hdr(LIDAR_SENDER), "detections": [box(1, 0.5), box(2, -0.5)]})
    tr = f.tracks[7]
    assert tr.link_id is None and f.stats.get("assoc_ambiguous", 0) >= 1
    out = f.output(100.4)
    assert {1, 2} <= {d["track_id"] for d in out["detections"]}      # both boxes still published as ordinary objects


def test_unknown_sender_and_bad_messages_do_not_crash():
    f = UC2Fusion()
    f.push({"metadata": {"sender_id": "SOMEONE"}, "detections": []})
    f.push({"metadata": {"sender_id": UWB_SENDER}, "detections": [{"track_id": 7}]})
    f.push({})
    assert f.output(1.0)["detections"] == []
