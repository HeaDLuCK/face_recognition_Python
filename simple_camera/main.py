from service.sync_service import  SyncService
from camera.camera_manager import CameraManager
from face_recognition import InsightFaceEngine
from database import close_mongo_connection, connect_to_mongo
import uvicorn
import logging
from fastapi import FastAPI
import logging
from contextlib import asynccontextmanager
from service.embedding_service import EmbeddingService
from service.sync_service import SyncService
from camera.camera_process_manager import CameraProcessManager
from api import attendance, cameras, sync,register
from service.attendance_service import AttendanceService
from pathlib import Path
import multiprocessing as mp
from logging_setup import (
    configure_queue_logging,
    start_log_listener,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

@asynccontextmanager
async def lifespan(app: FastAPI):
    mp_context = mp.get_context("spawn")
    log_queue = mp_context.Queue()

    log_listener = start_log_listener(
        log_queue=log_queue,
        log_directory=LOG_DIR,
    )

    configure_queue_logging(log_queue)

    database = await connect_to_mongo()
    camera_manager: CameraProcessManager | None = None
    try:
        
        embedding_service = EmbeddingService(database)
        attendance_service = AttendanceService(database)
        app.state.embedding_service = embedding_service
        face_engine = InsightFaceEngine(
                    model_name="buffalo_s",
                    providers=["CPUExecutionProvider"],
                    ctx_id=-1,
                    det_size=640,
                    min_score=0.6,
                )
        sync_service = SyncService(
                db=database,
                embedding_service=embedding_service,
                face_engine=face_engine,
            )
        camera_manager = CameraProcessManager(
                sync_service=sync_service,
                embedding_service=embedding_service,
                attendance_service=attendance_service,
                mp_context=mp_context,
                log_queue=log_queue,
            )
        
        

        app.state.attendance_service = attendance_service
        app.state.sync_service = sync_service
        app.state.camera_process_manager = camera_manager

        result  = await camera_manager.start_all()

        logger.info(
            "Camera startup result: %s",
            result,
        )

        logger.info(
            "FastAPI startup completed"
        )
    
        yield
    except Exception:
        logger.exception(
            "Application startup or runtime failed"
        )
        raise
    finally:
        logger.info(
            "FastAPI shutdown started"
        )

        try:
            if camera_manager is not None:
                await camera_manager.stop_all()

                logger.info(
                    "Camera processes stopped"
                )

            if database is not None:
                await close_mongo_connection()

                logger.info(
                    "MongoDB connection closed"
                )
        except Exception:
            logger.exception(
                "Application shutdown failed"
            )

        finally:
            logger.info(
                "FastAPI shutdown completed"
            )
            log_listener.stop()

            log_queue.close()
            log_queue.join_thread()




app = FastAPI(lifespan=lifespan)
# app = FastAPI(title=get_settings().app_name, version="0.2.0", lifespan=lifespan)


app.include_router(sync.router, prefix="/api/sync", tags=["sync"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["camera-control"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["attendance"])
app.include_router(register.router, prefix="/api/test", tags=["recognition-test"])

if __name__ == "__main__":
    mp.freeze_support()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
