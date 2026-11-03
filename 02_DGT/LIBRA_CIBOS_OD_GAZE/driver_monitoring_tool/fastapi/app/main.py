import asyncio
import logging

from app.transport.ros_publisher import RosbridgePublisher
from fastapi import FastAPI
from app.transport.ros_subscriber import RosbridgeSubscriber
from app.config import Settings
from app.routes import routes
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.helpers.data_processor import DataProcessor
from app.helpers.bounded_queue import GAZE_QUEUE_MAXSIZE
import uvicorn
from app.models.simulation_model import GazePredictorSimulation

logging.basicConfig(
    level=Settings().log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

async def _shutdown(component, task, label):
    await component.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Error while stopping %s", label)
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    queue = asyncio.Queue(maxsize=GAZE_QUEUE_MAXSIZE)
    data_publisher = None
    publisher_task = None

    if s.simulation_mode:
        data_subscriber = GazePredictorSimulation(queue=queue)
    else:
        data_subscriber = RosbridgeSubscriber(
            queue=queue,
            rosbridge_url=s.rosbridge_url,
            topic=s.ros_topic,
            message_type=s.ros_message_type,
        )
    
    if not s.simulation_mode or s.publish_in_simulation:
        if s.simulation_mode:
            logger.warning(
                "PUBLISH_IN_SIMULATION is on: calibrations computed from simulated gaze "
                "will be published to %s on %s", s.ros_calibration_topic, s.rosbridge_url
            )
        data_publisher = RosbridgePublisher(
            rosbridge_url=s.rosbridge_url,
            topic=s.ros_calibration_topic,
            message_type=s.ros_calibration_message_type,
        )
        publisher_task = asyncio.create_task(data_publisher.start())

    subscriber_task = asyncio.create_task(data_subscriber.start())

    app.state.session = DataProcessor(queue)
    app.state.data_subscriber = data_subscriber
    app.state.data_publisher = data_publisher

    yield

    if app.state.session.status():
        await app.state.session.stop()
        
    await _shutdown(data_subscriber, subscriber_task, "data subscriber")
    if data_publisher is not None:
        await _shutdown(data_publisher, publisher_task, "publisher")

app = FastAPI(lifespan=lifespan)
app.include_router(routes.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    s = Settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, log_level=s.log_level.lower())