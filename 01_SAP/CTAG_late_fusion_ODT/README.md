# UC2: fusion of the CTAG LiDAR detector/tracker with the DENSO UWB

First version of the fusion module for use case UC2 (D3.1). A VRU carries a UWB tag: the tag gives its position and
velocity at 1-10 Hz (identity known, no classification); the CTAG LiDAR module gives 3D boxes with track ids at its own
rate. The module publishes one `Ego_3D_ODT` message (ODT.json schema) with

- the **cooperative VRU**: one object per tag, estimated by a Kalman filter fed by both sensors;
- every **other LiDAR object**, passed through unchanged (except the box that was fused with the VRU).

The module takes and returns parsed JSON dicts (no ROS dependency).

| File | |
|---|---|
| `UC2_fusion.py` | the module (`UC2Fusion`, `FusionConfig`) |
| `uwb_sim.py` | synthetic UWB + LiDAR messages: moving ego, VRU with tag, neighbour pedestrian, occlusion, dropout, outliers, latency |
| `evaluate.py` | replays a scenario in arrival order and scores the fused output against the ground truth |
| `demo_UC2_fusion.py` | command-line demo: single run, `--sweep`, `--plot`, `--dump-dir examples` |
| `test_UC2_fusion.py` | 21 tests |
| `ODT.json` | the message interface |
| `requirements.txt` | Python packages |
| `examples/` | one UWB message, one LiDAR message and one fused output (JSON); `sweep_expected.txt` and `fusion_uwb_2hz.png`: reference output of the commands below |

## Reproducing the results
This folder is self-contained: it does not need the LiDAR release, nuScenes, ROS or a GPU. **There are no data files**:
`uwb_sim.py` generates the test messages from a fixed seed, in memory: the same messages on every run, and on another
Python or numpy version the same up to rounding in the last digits (below 1e-8 m).

```
pip install -r requirements.txt                 # numpy, scipy; matplotlib only for --plot, pytest only for the tests
python -m pytest -q test_UC2_fusion.py         # 21 passed, ~10 s
python demo_UC2_fusion.py --sweep              # UWB 1/2/5/10 Hz x 5 seeds: compare with examples/sweep_expected.txt
python demo_UC2_fusion.py --uwb-hz 2 --plot out.png    # one run + figure: compare with examples/fusion_uwb_2hz.png
python demo_UC2_fusion.py --dump-dir out_examples      # the JSON messages of examples/
```

`--sweep` may print `UWB disagrees with the LiDAR link` on stderr: it is the fusion logging a dropped link, not an error.
Regenerated examples differ from `examples/` only in `message_id`, `send_timestamp` and that rounding.
Tested with Python 3.8 (numpy 1.23, scipy 1.10, pytest 7.4) and Python 3.12 (numpy 2.5, scipy 1.18, pytest 9.2,
matplotlib 3.11).

## Using the module
```python
from UC2_fusion import UC2Fusion, FusionConfig
fusion = UC2Fusion(FusionConfig())
for msg in incoming:                  # any order, any rate: UWB ('DENSO_uwb_ODT') and LiDAR ('CTAG_lidar_ODTH')
    fusion.push(msg)
out = fusion.output(t_out)            # every 0.1 s (or at the rate the consumer needs), t_out in Unix seconds
```

## How it works
1. **Frames.** Each message is moved to a local ENU map frame with its own `sender_state` (WGS84 pose), so ego motion
   between capture times is compensated. Only x, y are filtered; height and box size come from the LiDAR, or class
   defaults (pedestrian 0.6 x 0.6 x 1.75 m) when only the UWB sees the VRU.
2. **Filter per tag.** Constant-velocity Kalman filter, state (x, y, vx, vy). Both sensors measure the full state with
   their own covariance (LiDAR grows with range). Measurements can arrive **out of order** (UWB latency is larger than
   the LiDAR's): the last 1.5 s are stored and replayed in time order.
3. **UWB.** Gated against the track (chi-square, 2 dof) to reject multipath jumps. A new tag id creates a track.
4. **LiDAR association.** The tag is linked to a LiDAR track id and keeps that link ("sticky"). A new link needs a
   box inside the gate (position and velocity, Mahalanobis, 4 dof), compatible class, Hungarian assignment, and **no
   second candidate within `ambiguity_margin`** (two equally good boxes: the fusion does not guess and keeps the
   UWB-only estimate). Safeguards for the cases that actually happen:
   - the linked id disappears (occlusion; the tracker gives a new id on re-detection): after `link_stale_s` the tag
     may link to another box;
   - another box fits clearly better than the linked one for 2 frames (id swap with a neighbour): link dropped;
   - the UWB disagrees persistently with the linked box (median NIS of the last second > 6 *and* the last two UWB
     samples agree with each other, which multipath outliers do not): the link is false, it is dropped and the
     filter restarts on the UWB. The UWB is cooperative, so it has the last word on identity.
5. **Output** at any time `t_out`: tracks predicted to `t_out`, converted to the ego frame at `t_out` (ego pose
   interpolated, extrapolated at most 0.5 s). The cooperative object has `track_id = 1_000_000 + tag id` (stable),
   category from the tag registry, score = noisy-OR of both sensors decayed with the age of their last sample, and
   (`emit_provenance`) `sources` and `fusion_info`:
   `status` = `fused` | `uwb_only` (LiDAR lost, e.g. occluded) | `lidar_only` (UWB stale) | `coasting` (neither),
   `lidar_track_id`, `age_uwb_s`, `age_lidar_s`, `pos_std_m`, `link_ambiguous`.
   The UWB counts as stale after 3 x its observed mean interval (0.5 s at 10 Hz, 2.5 s at 1 Hz).
   A track is deleted after `uwb_drop_s` (10 s) without UWB or `track_timeout_s` (5 s) without any measurement; its
   LiDAR box then goes back to pass-through.


