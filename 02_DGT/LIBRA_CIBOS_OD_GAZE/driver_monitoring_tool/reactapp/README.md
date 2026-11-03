# Hidden

## 🛠️ Getting Started

### 1. Prerequisites
- **Node.js**: v22+ (Recommended)

### 2. Frontend Configuration
Update `public/configuration/config.json` with the URLs of your backend/janus/rosbridge:
```json
{
  "apiBaseUrl": "http://<jetson-ip>:8000",
  "janusUrl": "http://<jetson-ip>/janus",
  "rosbridgeUrl": "ws://<jetson-ip>:9090",
  "appName": "HIDDEN GAZE TRACKING",
  "useKeycloak": false
}
```

### 3. Installation & Launch
```bash
npm install
npm run dev
```

## 📁 Project Structure

```text
src/
 ├── components/       
 │    ├── common/         # Global UI (ErrorBoundary, ConnectionStatus)
 │    ├── gazeTracking/   # Specialized widgets (Chart, DebugScreen)
 │    └── videoPlayer/    # Widgets for video streaming
 ├── context/             # Global state (ThemeContext, JanusContext, MountpointsContext, RosContext)
 ├── helpers/             # Infrastructure (DataManager, config)
 ├── hooks/               # Useful hooks (useJanusMountpoints, useRosTopic, useHealthCheck)
 ├── i18n/                # Localization (index, locales/en.json, locales/el.json)
 ├── layout/              # UI Shell (MainLayout with Header/Sidebar)
 ├── pages/               # Feature screens (GazeTracking)
 ├── routes/              # Routing logic
 ├── App.tsx              # Global Providers & Error Protection
 └── main.tsx             # Configuration bootstrap & entry
```

## Debug Screen 
Multi-view dashboard (opened in a separate window) which includes:
  * ✅In-cabin camera feed with landmark overlays
  * ✅Cropped face and eye views
  * ✅Gaze prediction visualization on the camera prediction plane
  * Logs and metrics (PERCLOS, facial emotion estimation, etc)

Debug Screen can open only before starting any process (calibration or inference).

## Gaze Screen
  * **Default mode**: a single reference red-dot moves through a fixed grid guiding the user's gaze. During inference, a second purple-dot is overlaid showing the model's live prediction for the current point.
  * **Inference-only alternative**: when either "Video-example mode" or "Road-facing mode" is enabled, the fullscreen surface shows a video stream (pre-recorded example or live road-facing camera).
  * On-screen controls: while in fullscreen, moving the mouse briefly reveals a "stop" button for aborting the process early.
    * For calibration only: after each reference point is shown, a confirmation pop-up appears with options:
      * **Retake point**: re-run calibration for the same point,
      * **Next point**: continue calibration with next point,
      * **Save process**: available after the last point, finalize calibration and save the calculated homography matrix,
      * **Cancel process**: available after the last point, discard calibration and exit fullscreen.
    * Note: all available actions have also a dedicated shortcut key as shown in their respective button.

## Connection Status
A connection status indicator is available in the top-right corner of gaze tracking page. It provides real-time status information for:
* Janus Server – WebRTC server connection
* ROS WebSocket – Connection to the ROS bridge WebSocket
* FastAPI Backend – Connection with rosbridge
  * Publisher: connection status for publishing the calculated homography matrix, 
  * Subscriber: connection status for receiving valid data from gaze model