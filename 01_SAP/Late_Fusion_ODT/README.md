# UPV Late Fusion ODT (HIDDEN)

Fuses the ego vehicle's three ODT streams into one `Ego_3D_ODT` message (ODT.json schema) for CA.

| Input (sender_id, configurable) | Schema | Provides |
|---|---|---|
| `CTAG_lidar_ODTH` | ODT.json | 3D boxes, tracks |
| `UPV_cam_lidar_ODTH` | ODT.json | 3D boxes, tracks |
| `DENSO_uwb_ODT` | 2D_ODT_UWB.json | x, y, vx, vy (z/size/heading optional) |

## Usage
```python
from late_fusion_odt import LateFusionODT, FusionConfig
fusion = LateFusionODT(FusionConfig())
out = fusion.fuse([ctag_msg, upv_msg, denso_msg])   # parsed JSON dicts, latest per source
```
Call `fuse()` at the output rate (e.g. 10 Hz). Missing or stale sources are skipped automatically.
`python demo_late_fusion.py` runs a synthetic test.

## Method
1. **Spatial alignment**: each message → a local ENU map frame, using its own `sender_state`. This compensates for ego motion between the capture times of the three sensors.
2. **Temporal alignment**: each detection is propagated to the fusion time (CTRV/CV), and its covariance is inflated with the time gap. Detections older than `max_latency_s` are dropped.
3. **Association**: source-by-source Hungarian assignment on Mahalanobis distance, with a χ² gate and class compatibility. Vehicles and VRUs never merge; Pedestrian/Cyclist/Scooter may merge, with a penalty.
4. **Fusion**:
   - Position and velocity: information-weighted, with each detection weighted by its confidence. Weak members don't move the state.
   - Heading: circular mean, 3D sources only, with the 180° ambiguity resolved.
   - Size: from the best 3D source, or a class default for UWB-only objects.
   - Existence: noisy-OR of score × trust.
   - Class: weighted vote.
5. **Track management**: fused `track_id` stays stable via source-track links first, then spatial matching to predicted tracks.
6. **Output**: ODT.json in the ego frame at fusion time, plus an optional `sources` / `fusion_info` block per object.

## Must calibrate
All values in `DEFAULT_SOURCES` (noise, trust, min_score) are placeholders. Fit them on recorded data with ground truth, and check consistency (NIS).

## Requests to partners / schema
- **Velocity convention**: are vx, vy over ground or relative to ego? The code assumes over ground.
- **sender_state has no timestamp**: the ego pose is assumed valid at the detection timestamp. Ideally add `sender_state.timestamp`.
- **UWB categories**: agree on the list. 'Scooter' is mapped as a VRU class.
- **Provenance**: adding `sources` to ODT.json as an official optional field is needed by CA and by UPV_ExpGeneration / CAImpactAnalysis.

