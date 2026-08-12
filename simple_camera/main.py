from service.sync_service import  SyncService
from face_recognition import InsightFaceEngine
from database import close_mongo_connection, connect_to_mongo
import uvicorn
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from service.embedding_service import EmbeddingService
from camera.camera_process_manager import CameraProcessManager
from api import attendance, cameras, sync,register
from service.attendance_service import AttendanceService
from service.erp_client import ErpClient
from service.erp_sync_service import ErpSyncService
from pathlib import Path
import multiprocessing as mp
from logging_setup import (
    configure_queue_logging,
    start_log_listener,
)
import argparse
from service.unknown_person_service import UnknownPersonService
from config import get_settings
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
ERP_URLS: dict[str, str] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    mp_context = mp.get_context("spawn")
    log_queue = mp_context.Queue()

    log_listener = start_log_listener(
        log_queue=log_queue,
        log_directory=LOG_DIR,
    )

    configure_queue_logging(log_queue)

    database = await connect_to_mongo()
    camera_manager: CameraProcessManager | None = None
    erp_client = None
    erp_sync_service = None
    try:
        
        embedding_service = EmbeddingService(database)
        attendance_service = AttendanceService(database)
        unknown_person_service  = UnknownPersonService(database)

        if ERP_URLS:
            erp_client = ErpClient(ERP_URLS)

            erp_sync_service =  ErpSyncService(
                    unknown_person_service=unknown_person_service,
                    attendance_service=attendance_service,
                    embedding_service=embedding_service,
                    erp_client=erp_client,
                    interval_seconds= settings.erp_sync_interval_seconds,
                    assignment_interval_seconds= settings.erp_assignment_interval_seconds,
                    batch_size=settings.erp_sync_batch_size,
                )
            await erp_sync_service.start()

        face_engine = InsightFaceEngine()
        sync_service = SyncService(
                db=database,
                embedding_service=embedding_service,
                face_engine=face_engine,
            )
        camera_manager = CameraProcessManager(
                sync_service=sync_service,
                embedding_service=embedding_service,
                attendance_service=attendance_service,
                unknown_person_service=unknown_person_service,
                mp_context=mp_context,
                log_queue=log_queue,
            )
        
        app.state.unknown_person_service = unknown_person_service
        app.state.attendance_service = attendance_service
        app.state.sync_service = sync_service
        app.state.camera_process_manager = camera_manager
        app.state.embedding_service = embedding_service
        app.state.erp_client  = erp_client 
        app.state.erp_sync_service  = erp_sync_service

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
            

            if erp_sync_service is not None:
                await erp_sync_service.stop()

            if erp_client is not None:
                await erp_client.close()

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

    parser = argparse.ArgumentParser(
        description="AI Camera Service"
    )

    parser.add_argument(
        "--erp",
        action="append",
        default=[],
        help=(
            "ERP configuration in format "
            "ETS_AUTH=URL. "
            "Can be used multiple times."
        ),
    )

    args = parser.parse_args()

    ERP_URLS = {}

    for value in args.erp:

        if "=" not in value:
            raise ValueError(
                f"Invalid ERP configuration: {value}. "
                "Expected ETS_AUTH=URL"
            )

        ets_auth, url = value.split(
            "=",
            1,
        )

        ets_auth = ets_auth.strip()

        url = (
            url
            .strip()
            .rstrip("/")
        )

        if not ets_auth:
            raise ValueError(
                "etsAuth cannot be empty"
            )

        if not url:
            raise ValueError(
                f"ERP URL cannot be empty "
                f"for etsAuth={ets_auth}"
            )

        ERP_URLS[
            ets_auth
        ] = url

    if ERP_URLS:
        print(
            "Configured ERP servers:"
        )

        for (
            ets_auth,
            url,
        ) in ERP_URLS.items():

            print(
                f"  {ets_auth} -> {url}"
            )

    else:
        print(
            "No ERP URLs provided"
        )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
