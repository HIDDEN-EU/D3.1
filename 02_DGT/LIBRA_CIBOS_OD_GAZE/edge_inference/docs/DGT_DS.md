# DGT_DS — the message dgt sends

What the `dgt-daemon` emits to the other HIDDEN components, one message per
assessed frame. The reference below is the agreed shape, **verbatim**; the
contract that produces it is `dgt/contracts/driver_status.py`, and
`tests/test_driver_status.py` parses the JSON block below and checks the code's
key layout against it — edit one and the test makes you edit the other.

Transport: ROS 2. Each message is this JSON, compact, as the `data` of a
`std_msgs/String` on `/dgt/status` (`ROS_TOPIC_PREFIX`), BEST_EFFORT, depth 1;
rosbridge relays it to `ws://<jetson-ip>:9090`.

## Reference

```json
{
  "metadata": {
    "schema_version": "string",
    "message_type": "string  (e.g. 'DGT_DS')",
    "message_id": "string  (UUID; unique per message)",
    "sender_id": "string  (globally unique agent identifier)",
    "sequence_number": "int  (monotonically increasing per sender)",
    "send_timestamp": "float  (Unix seconds; wall-clock time when this message was sent)",
    "time_reference": "string  (e.g. 'UTC')"
  },

  "driver_status": {
    "timestamp": "float  (Unix seconds; capture/inference time of the assessed frame — must not include transmission latency)",

    "driver_att_od": {
      "_comment": "Driver gaze area and attended-object detections, passed through from LIBRA_CIBOS_OD_GAZE. All coordinates are 2D road-camera pixel space.",
      "frame_id": "string  (e.g. 'road_camera')",
      "gaze_area_px": {
        "_comment": "Region of the road-camera plane the driver is attending to (FUN-DGT-01); projectable to 3D downstream by CA.",
        "center": "array[int]  ([x, y] in road-camera pixels)",
        "radius_px": "int  (radius of the attended region in pixels)"
      },
      "detections": [
        {
          "class_id": "int  (index into contracts/detections.BDD100K_CLASSES, 0-8)",
          "class_name": "string  (one of the 9 classes in contracts/detections.BDD100K_CLASSES, e.g. 'car', 'pedestrian')",
          "bbox_px": "array[int]  ([x_min, y_min, x_max, y_max] in road-camera pixels)",
          "confidence": "float  (in [0, 1]; YOLO detection confidence)"
        }
      ]
    },

    "emotion_status": {
      "_comment": "Categorical emotion and per-class probability distribution, from LIBRA_camera_EM.",
      "frame_id": "string  (e.g. 'driver_camera')",
      "label": "string  (one of: 'angry' | 'disgust' | 'fear' | 'happy' | 'neutral' | 'sad' | 'surprise')",
      "class_id": "int  (index into ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])",
      "confidence": "float  (in [0, 1]; softmax probability of the predicted class; equals probabilities[label])",
      "probabilities": {
        "_comment": "Full softmax distribution over all 7 classes; sums to 1.0.",
        "angry": "float  (in [0, 1])",
        "disgust": "float  (in [0, 1])",
        "fear": "float  (in [0, 1])",
        "happy": "float  (in [0, 1])",
        "neutral": "float  (in [0, 1])",
        "sad": "float  (in [0, 1])",
        "surprise": "float  (in [0, 1])"
      }
    },

    "level_of_attention": {
      "_comment": "Instantaneous attentiveness score from CTAG_ATTEN multi-cue fusion.",
      "frame_id": "string  (e.g. 'driver_camera')",
      "score": "int  ([0, 100]; higher = more attentive)",
      "confidence": "float  (in [0, 1]; model certainty in the predicted score)"
    },

    "distracted": "bool  (derived from level_of_attention.score threshold and emotion state)",
    "drowsy": "bool  (derived from level_of_attention.score threshold and emotion state)",
    "confidence": "float  (in [0, 1]; fused confidence of the composite assessment)"
  }
}
```

`_comment` keys are documentation, not wire fields.

## What is filled today (2026-09-25)

**A field that is not measured is JSON `null`** — never a zero, a default or a
neutral label. A consumer cannot tell a filled-in placeholder from a
measurement, so an unimplemented block is absent, whole.

| Field | Status | Source |
|---|---|---|
| `metadata.*` | filled | `dgt/publishers/dgt_ds.py`; `sender_id` from `DGT_SENDER_ID`, the rest from `config/service.yaml` |
| `driver_status.timestamp` | filled | wall-clock time the road frame was **captured** |
| `driver_att_od.frame_id` | filled | `config/service.yaml` (`road_camera`) |
| `driver_att_od.detections` | filled | `models/road_objects.py`, full tiled grid, boxes converted to pixels. `[]` means "nothing detected"; if detection did not run the whole `driver_att_od` is `null` |
| `driver_att_od.gaze_area_px` | **`null`** | needs a measured gaze → road projection and a defined radius |
| `emotion_status` | **`null`** | ONNX is registered (`emotion_classifier`); no model class or pipeline yet |
| `level_of_attention` | **`null`** | the CTAG_ATTEN fusion is not defined yet |
| `distracted`, `drowsy`, `confidence` | **`null`** | derived from the fusion above |

## Open questions against the reference

- **"Attended" detections.** The reference describes attended-object
  detections. Selecting the attended ones needs the same trusted projection as
  `gaze_area_px`, so today `detections` carries every object in the road frame.
- **`radius_px`** has no definition yet (fixed? from gaze confidence? from the
  fixation spread?).
- **Rate.** One message per assessed frame; whether consumers want that or a
  decimated stream is not specified.
- **Transport.** ROS 2 `std_msgs/String` on `/dgt/status` — not yet confirmed
  with the consumers, nor whether they want a typed message instead.
