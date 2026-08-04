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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
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
            )
        
        

        app.state.attendance_service = attendance_service
        app.state.sync_service = sync_service
        app.state.camera_process_manager = camera_manager

        start_result = await camera_manager.start_all()

        logger.info(
            "Camera startup result: %s",
            start_result,
        )


        logger.info("FastAPI startup completed")
    
        yield
    finally:
        logger.info("FastAPI shutdown started")

        if camera_manager is not None:
            await camera_manager.stop_all()

        await close_mongo_connection()

        logger.info("FastAPI shutdown completed")




app = FastAPI(lifespan=lifespan)
# app = FastAPI(title=get_settings().app_name, version="0.2.0", lifespan=lifespan)


app.include_router(sync.router, prefix="/api/sync", tags=["sync"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["camera-control"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["attendance"])
app.include_router(register.router, prefix="/api/test", tags=["recognition-test"])

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
