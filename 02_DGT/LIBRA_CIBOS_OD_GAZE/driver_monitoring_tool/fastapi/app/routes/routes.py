"""HTTP API for controlling the demo, managing calibration/mapping reference
points, fetching predictions and checking rosbridge health."""

from app.helpers.errors import NoActiveSessionError, PointNotFoundError, SessionStateError
from fastapi import APIRouter, Request, Response, status, HTTPException
from pydantic import BaseModel
from typing import List
import websockets

router = APIRouter()

class PointData(BaseModel):
    """A single point in grid coordinates."""
    x_value: float
    y_value: float

class TargetPoints(BaseModel):
    """The reference points generated for a new session."""
    ref_points: List[PointData]

class ConfigData(BaseModel):
    """Session type: True for calibration, False for gaze mapping."""
    calibration: bool

class Status(BaseModel):
    """Whether the data processor is currently running."""
    running: bool

@router.post("/demo/start")
async def start_demo(request: Request):
    """Start processing gaze data; 503 if the gaze subscription is unhealthy."""
    subscriber = request.app.state.data_subscriber

    if not subscriber.is_healthy():
        raise HTTPException(status_code=503, detail=f"Data subscription is not currently active : {subscriber.get_status_report()}")

    data_processor = request.app.state.session
    data_processor.start()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/demo/stop")
async def stop_demo(request: Request):
    """Stop processing gaze data (no-op if already stopped)."""
    data_processor = request.app.state.session
    await data_processor.stop()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/demo/pause")
async def pause_demo(request: Request):
    """Pause sample collection in the calibration session; 409 if none is active."""
    data_processor = request.app.state.session
    try:
        await data_processor.pause()
    except SessionStateError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.get("/demo/status", response_model=Status)
async def status_demo(request: Request):
    """Report whether the data processor is running."""
    return Status(running=request.app.state.session.status())

@router.post("/points/reference/initialize", response_model=TargetPoints)
async def init_data_ref(request: Request, data: ConfigData):
    """Start a new calibration or gaze-mapping session and return its reference points."""
    temp = data.model_dump()

    data_processor = request.app.state.session
    points = data_processor.initialize_ref_points(temp['calibration'])
    return TargetPoints(ref_points = [PointData(x_value = float(x), y_value = float(y)) for x, y in points])


@router.post("/points/reference/update")
async def update_data_ref(request: Request, data: PointData):
    """Set the reference point the user is currently looking at; 404 if unknown, 409 if no session."""
    data_processor = request.app.state.session
    try:
        data_processor.update_ref_point(data.model_dump())
    except PointNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NoActiveSessionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/points/prediction/calculate")
async def calc_data_pred(request: Request):
    """Compute the calibration matrix, publish it to ROS, then save it to disk.

    The matrix is only saved once the publish succeeded, so the file never holds
    a calibration ROS didn't get.
    """
    data_processor = request.app.state.session
    publisher = request.app.state.data_publisher
    try:
        calibration_matrix = await data_processor.calculate_transformation_matrix()

        if publisher is not None:
            try:
                await publisher.publish_matrix(calibration_matrix)
            except (RuntimeError, OSError, websockets.exceptions.WebSocketException) as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Calibration not saved: publishing to rosbridge failed: {e}",
                )

        data_processor.save_transformation_matrix()

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.get("/points/prediction", response_model=PointData)
async def get_data_pred(request: Request):
    """Return the calibrated gaze prediction for the current point; 422 if unavailable."""
    data_processor = request.app.state.session
    point = await data_processor.make_calculations()

    if point is not None:
        return PointData(x_value = float(point[0]), y_value = float(point[1]))

    raise HTTPException(status_code=422, detail="Error calculating prediction")

@router.get("/health/rosbridge")
async def rosbridge_health(request: Request):
    """Return status reports for the gaze subscriber and the matrix publisher."""
    subscriber = request.app.state.data_subscriber
    publisher = request.app.state.data_publisher

    return {
        "subscriber": subscriber.get_status_report(),
        "publisher": publisher.get_status_report() if publisher is not None else None,
    }