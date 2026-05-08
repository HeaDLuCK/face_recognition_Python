import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import attendance, cameras, events, health, sync, test
from app.attendance.attendance_service import AttendanceService
from app.cameras.camera_manager import CameraManager
from app.config import get_settings
from app.database import close_mongo_connection, connect_to_mongo
from app.erp.erp_client import ErpClient
from app.events.event_service import EventService
from app.face.embedding_service import EmbeddingService
from app.face.insightface_engine import InsightFaceEngine
from app.face.recognition_service import RecognitionService
from app.runtime_state import RuntimeState
from app.services.log_service import LogService
from app.services.sync_service import SyncService
from app.storage.snapshot_service import SnapshotService


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)

    db = await connect_to_mongo()
    erp_client = ErpClient(settings)
    runtime_state = RuntimeState(settings)
    face_engine = InsightFaceEngine(settings)
    embedding_service = EmbeddingService(db)
    recognition_service = RecognitionService(
        embedding_service=embedding_service,
        face_engine=face_engine,
    )
    log_service = LogService(db)
    snapshot_service = SnapshotService(settings.snapshot_dir, db)
    event_service = EventService(db, erp_client)
    attendance_service = AttendanceService(db)
    sync_service = SyncService(
        erp_client=erp_client,
        runtime_state=runtime_state,
        embedding_service=embedding_service,
        face_engine=face_engine,
        log_service=log_service,
    )
    camera_manager = CameraManager(
        runtime_state=runtime_state,
        recognition_service=recognition_service,
        snapshot_service=snapshot_service,
        event_service=event_service,
        attendance_service=attendance_service,
        log_service=log_service,
        settings=settings,
    )

    app.state.db = db
    app.state.erp_client = erp_client
    app.state.runtime_state = runtime_state
    app.state.face_engine = face_engine
    app.state.embedding_service = embedding_service
    app.state.recognition_service = recognition_service
    app.state.snapshot_service = snapshot_service
    app.state.event_service = event_service
    app.state.attendance_service = attendance_service
    app.state.log_service = log_service
    app.state.sync_service = sync_service
    app.state.camera_manager = camera_manager

    try:
        yield
    finally:
        await camera_manager.stop_all()
        await close_mongo_connection()


app = FastAPI(title=get_settings().app_name, version="0.2.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(sync.router, prefix="/api/sync", tags=["sync"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["camera-control"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["attendance"])
app.include_router(test.router, prefix="/api/test", tags=["recognition-test"])
