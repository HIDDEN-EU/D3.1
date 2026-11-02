# Driver-monitoring tool

The calibration and driver-monitoring web tool of the DGT (D3.1 Section 3.6), developed jointly by LIBRA and CIBOS. The edge device runs without a monitor, so it hosts a local web application that an operator opens from a device on the same network.

## What the tool does

- **Debug screen:** the driver-facing camera view with the facial landmarks, the face crop and the two eye crops, the raw gaze point on the camera prediction plane, and logs.
- **Gaze screen:** the calibration targets for the driver to fixate, and in the inference phase the gaze point drawn on a test screen or on the road-facing camera feed with the detected objects.
- **Two phases:** calibration (fit of the homography, `../edge_inference/dgt/calibration.py`) and inference (monitoring).

## Interfaces with the edge service

- Gaze and driver-status topics over ROS 2, relayed to the browser by rosbridge.
- Video over RTP to a Janus WebRTC gateway: four in-cabin panes (driver view, face crop, two eye crops) on ports 28004–28007, and the road view on port 28008. The gateway configuration in `../edge_inference/deploy/janus/` disables recording and sets no STUN or TURN server, so no video leaves the local network.

## Code

The web user interface is developed by CIBOS. CIBOS adds its code, or a link to it, in this folder.

## Licence

To be confirmed by CIBOS.
