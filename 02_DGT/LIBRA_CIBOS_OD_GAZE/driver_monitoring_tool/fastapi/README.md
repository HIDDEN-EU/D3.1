# fastapi

A FastAPI backend for the driver-monitoring-tool. It exposes an HTTP API for:

- **Calibrating** a gaze-to-screen mapping against a 3x3 grid of reference points.
- **Predicting** where the driver is looking in real time, using the saved calibration matrix.

The backend consumes raw gaze points from one of two sources, chosen by
`SIMULATION_MODE`:

- **Simulation mode** — a built-in generator emits random gaze points. No
  camera, TensorRT, or ROS setup required.
- **Real mode** — the backend subscribes over `rosbridge` (a websocket) to a
  topic published by a separate driver-gaze daemon, which is the process
  actually running the TensorRT gaze model.

In real mode, each successful calibration is also published back to the
daemon on a second ROS topic (`ROS_CALIBRATION_TOPIC`), in addition to being
written to `CALIBRATION_MATRIX_PATH`.

## Requirements

- Python 3.11–3.12
- `venv` or `virtualenv`
- `pip`
- In real mode: a running `rosbridge` server and driver-gaze daemon reachable
  at the configured `ROSBRIDGE_URL`.

## Setup & Installation

1. **Create a virtual environment**:

   ```
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Configure the environment**:

   ```
   cp .env.example .env
   ```

   Then edit `.env`:

   | Variable | Default | Description |
   |---|---|---|
   | `SIMULATION_MODE` | `false` | `true` for the built-in random-gaze simulator, `false` to subscribe to rosbridge. |
   | `SIMULATION_FAULT` | `none` | Makes the simulator emit only invalid points (`nan`, `inf` or `zero`), to exercise the `invalid_data` health status. |
   | `PUBLISH_IN_SIMULATION` | `false` | Also publish calibrations to rosbridge in simulation mode. Off by default, because a calibration computed from random gaze is noise and would overwrite the daemon's real one. |
   | `CALIBRATION_MATRIX_PATH` | `calibration.pkl` | Path to the calibration (homography) matrix. Written by the calibration flow and read back during prediction, so it must point to the same file in both modes. |
   | `ROSBRIDGE_URL` | `ws://192.168.1.203:9090` | Websocket URL of the rosbridge server. |
   | `ROS_TOPIC` / `ROS_MESSAGE_TYPE` | `/dgt/gaze/raw` / `geometry_msgs/PointStamped` | Topic and message type the raw gaze points are received on. |
   | `ROS_CALIBRATION_TOPIC` / `ROS_CALIBRATION_MESSAGE_TYPE` | `/dgt/calibration_matrix` / `std_msgs/Float64MultiArray` | Topic and message type the calibration matrix is published on (as a flattened, row-major 3x3). |
   | `ROS_RECONNECT_DELAY` | `3` | Seconds to wait before retrying a dropped rosbridge connection. |
   | `HOST` / `PORT` | `0.0.0.0` / `8000` | Address the API server binds to when started with `python -m app.main`. |
   | `LOG_LEVEL` | `INFO` | Minimum level logged (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`). Also sets uvicorn's level when started with `python -m app.main`. |

## Running the Application

```
python -m app.main
```

This binds to `HOST`/`PORT` from `.env`. For development with auto-reload,
run uvicorn directly instead. It ignores `HOST`/`PORT`, so pass them on the
command line:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```


The app will start at `http://<HOST>:<PORT>` (default `http://0.0.0.0:8000`).

## API Overview

**Calibration flow** — walk the 9 reference points on a grid, collecting
gaze samples at each, then compute and save the mapping:

1. `POST /points/reference/initialize {"calibration": true}` — starts data
   collection and returns the 9 target points to display.
2. `POST /points/reference/update {"x_value": ..., "y_value": ...}` — call
   once per target point, as the driver looks at it, so incoming gaze
   samples get attributed to that point. An idle time is set when samples
   collected after selection are discarded while the driver's eyes settle. 
   Selecting a point again clears its samples and re-collects them, so a 
   bad point can be retaken.
3. `POST /demo/pause` (optional) — stop attributing samples. The next 
   `/points/reference/update` resumes collection.
4. `POST /points/prediction/calculate` — stops collection, computes the
   homography from the median gaze sample at each point, publishes it to
   `ROS_CALIBRATION_TOPIC` (in real mode, or with `PUBLISH_IN_SIMULATION`)
   and only then saves it to `CALIBRATION_MATRIX_PATH`. The matrix is saved
   only if the gaze subscription is active and the publish succeeded.
   Responses:
   - `204` on success.
   - `422` if no calibration session is active, if any point has no gaze
     samples or if no valid homography could be computed from the collected 
     points (recalibrate).
   - `503` if the gaze subscription is not active or publishing to rosbridge
     failed. The matrix is not saved; the collected samples are kept, so the
     call can be retried.

**Prediction flow** — after calibration, get live gaze predictions:

1. `POST /points/reference/initialize {"calibration": false}` — starts data
   collection and returns a set of random reference points for a
   validation/demo pass.
2. `POST /points/reference/update {"x_value": ..., "y_value": ...}` — as
   above, mark which point is currently being looked at.
3. `GET /points/prediction` — returns the predicted gaze point, transformed
   through the saved calibration matrix. Returns `422` if no prediction is
   available yet, e.g. there is no usable calibration file or no samples
   were collected for the current point. Requesting the prediction for the
   last point ends the session.

**Demo control:**

- `POST /demo/start` / `POST /demo/stop` — manually start or stop the
  background data-processing loop (normally started by
  `/points/reference/initialize`). Starting while the loop is already
  running restarts it; stopping when it is not running does nothing. Both
  return `204`.
- `POST /demo/pause` — see the calibration flow above.
- `GET /demo/status` — `{"running": true | false}`.

**Health:**

- `GET /health/rosbridge` — status of the gaze source (rosbridge
  subscriber or simulator) and of the calibration publisher (`null` when
  publishing is disabled):

  ```json
  {
    "subscriber": {"status": "subscribed", "last_error": null, "seconds_since_last_message": 0.03, "healthy": true},
    "publisher":  {"status": "subscribed", "last_error": null, "seconds_since_last_publish": null, "healthy": true}
  }
  ```

  `status` is one of `disconnected`, `connecting`, `connected`, `subscribed`
  (for the publisher: topic advertised), `error`, or `invalid_data`. The
  source reports `invalid_data` after 10 consecutive malformed, non-finite,
  or all-zero gaze points. Only `subscribed` counts as healthy.
