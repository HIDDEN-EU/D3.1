# Driver-monitoring tool

The calibration and driver-monitoring web tool of the DGT (D3.1 Section 3.6), developed jointly by LIBRA and CIBOS. The edge device runs without a monitor, so it hosts a local web application that an operator opens from a device on the same network.

## What the tool does

- **Debug screen:** the driver-facing camera view with the facial landmarks, the face crop and the two eye crops, the raw gaze point on the camera prediction plane, and logs.
- **Gaze screen:** the calibration targets for the driver to fixate, and in the inference phase the gaze point drawn on a test screen or on the road-facing camera feed with the detected objects.
- **Two phases:** calibration (fit of the homography, `../edge_inference/dgt/calibration.py`) and inference (monitoring). Currently the backend fits the homography separately, in `fastapi/app/helpers/calibration_session.py`; the two are to be combined.

## Interfaces with the edge service

- Gaze and driver-status topics over ROS 2, relayed to the browser by rosbridge. Currently the edge publishes only `/dgt/status`, so the gaze topics the tool uses (`/dgt/gaze/raw`, `/dgt/gaze/mapped`, `/dgt/calibration_matrix`) have no counterpart.
- Video over RTP to a Janus WebRTC gateway: four in-cabin panes (driver view, face crop, two eye crops) on ports 28004–28007, and the road view on port 28008. The gateway configuration in `../edge_inference/deploy/janus/` disables recording and sets no STUN or TURN server, so no video leaves the local network.

## Code

The web user interface is developed by CIBOS.

- [`fastapi/`](fastapi/): backend. Runs the calibration flow, serves gaze predictions and publishes the calibration matrix to the edge service over rosbridge. Has a simulation mode that needs no camera or ROS.
- [`reactapp/`](reactapp/): frontend with the debug and gaze screens.

See each folder's README for setup and configuration.

## Licence

MIT. See [LICENSE](LICENSE).
