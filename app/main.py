import asyncio
import logging
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI

from app.api import attendance, cameras, events, health, recovery, sync, test, unknown_faces
from app.attendance.attendance_service import AttendanceService
from app.cameras.camera_manager import CameraManager
from app.config import get_settings
from app.database import close_mongo_connection, connect_to_mongo
from app.events.event_service import EventService
from app.face.embedding_service import EmbeddingService
from app.face.insightface_engine import InsightFaceEngine
from app.face.recognition_service import RecognitionService
from app.fire.fire_detection_service import FireDetectionService
from app.plates.plate_recognition_service import PlateRecognitionService
from app.recovery.attendance_recovery_service import AttendanceRecoveryService
from app.runtime_state import RuntimeState
from app.services.log_service import LogService
from app.services.sync_service import SyncService
from app.storage.snapshot_service import SnapshotService
from app.tracking.person_detection_service import PersonDetectionService

logger = logging.getLogger(__name__)


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
    cv2.setNumThreads(settings.opencv_num_threads)
    settings.snapshot_dir.mkdir(parents=True, exist_ok=True)

    db = await connect_to_mongo()
    runtime_state = RuntimeState(settings)
    face_engine = InsightFaceEngine(settings)
    embedding_service = EmbeddingService(db)
    recognition_service = RecognitionService(
        embedding_service=embedding_service,
        face_engine=face_engine,
    )
    person_detection_probe = PersonDetectionService(settings)
    if settings.person_tracking_enabled and not person_detection_probe.available:
        logger.warning(
            "Person tracking requested but unavailable; legacy face scheduling will be used: %s",
            person_detection_probe.unavailable_reason,
        )
    plate_recognition_service = PlateRecognitionService(settings)
    fire_detection_service = FireDetectionService(settings)
    log_service = LogService(db)
    snapshot_service = SnapshotService(
        settings.snapshot_dir,
        db,
        unknown_face_db_match_limit=settings.unknown_face_db_match_limit,
        purge_batch_size=settings.snapshot_purge_batch_size,
    )
    event_service = EventService(db)
    attendance_service = AttendanceService(db)
    attendance_recovery_service = AttendanceRecoveryService(
        db=db,
        runtime_state=runtime_state,
        recognition_service=recognition_service,
        attendance_service=attendance_service,
        event_service=event_service,
        settings=settings,
    )
    sync_service = SyncService(
        runtime_state=runtime_state,
        embedding_service=embedding_service,
        face_engine=face_engine,
        log_service=log_service,
        snapshot_service=snapshot_service,
        db=db,
    )
    camera_manager = CameraManager(
        runtime_state=runtime_state,
        embedding_service=embedding_service,
        attendance_recovery_service=attendance_recovery_service,
        plate_recognition_service=plate_recognition_service,
        fire_detection_service=fire_detection_service,
        snapshot_service=snapshot_service,
        event_service=event_service,
        attendance_service=attendance_service,
        log_service=log_service,
        settings=settings,
    )

    app.state.db = db
    app.state.runtime_state = runtime_state
    app.state.face_engine = face_engine
    app.state.embedding_service = embedding_service
    app.state.recognition_service = recognition_service
    app.state.plate_recognition_service = plate_recognition_service
    app.state.fire_detection_service = fire_detection_service
    app.state.snapshot_service = snapshot_service
    app.state.event_service = event_service
    app.state.attendance_service = attendance_service
    app.state.attendance_recovery_service = attendance_recovery_service
    app.state.log_service = log_service
    app.state.sync_service = sync_service
    app.state.camera_manager = camera_manager

    image_purge_task = None
    attendance_recovery_task = None
    try:
        saved_config = await sync_service.load_saved_config()
        image_purge_task = asyncio.create_task(
            _run_image_purge_loop(sync_service),
            name="image-retention-purge",
        )
        if settings.history_recovery_enabled:
            attendance_recovery_task = asyncio.create_task(
                attendance_recovery_service.run_forever(),
                name="attendance-history-recovery",
            )
        if settings.auto_start_saved_cameras and saved_config["cameras"] > 0:
            await camera_manager.start_all()
        yield
    finally:
        if attendance_recovery_task is not None:
            attendance_recovery_service.stop()
            attendance_recovery_task.cancel()
            await asyncio.gather(attendance_recovery_task, return_exceptions=True)
        if image_purge_task is not None:
            image_purge_task.cancel()
            await asyncio.gather(image_purge_task, return_exceptions=True)
        try:
            await camera_manager.shutdown()
        finally:
            await close_mongo_connection()


app = FastAPI(title=get_settings().app_name, version="0.2.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(sync.router, prefix="/api/sync", tags=["sync"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["camera-control"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["attendance"])
app.include_router(recovery.router, prefix="/api/recovery-jobs", tags=["attendance-recovery"])
app.include_router(test.router, prefix="/api/test", tags=["recognition-test"])
app.include_router(unknown_faces.router, prefix="/api/unknown-faces", tags=["unknown-faces"])


async def _run_image_purge_loop(sync_service: SyncService) -> None:
    while True:
        await asyncio.sleep(3600)
        try:
            await sync_service.purge_images_for_loaded_rules()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Image retention purge failed")
