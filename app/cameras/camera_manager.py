import logging
import time
from typing import AsyncGenerator

import asyncio
import cv2

from app.attendance.attendance_service import AttendanceService
from app.cameras.camera_worker import CameraWorker
from app.cameras.rtsp_reader import RtspReader
from app.config import Settings
from app.events.event_service import EventService
from app.face.recognition_scheduler import FaceRecognitionScheduler
from app.face.recognition_service import RecognitionService
from app.fire.fire_detection_service import FireDetectionService
from app.plates.plate_recognition_service import PlateRecognitionService
from app.recovery.attendance_recovery_service import AttendanceRecoveryService
from app.runtime_state import RuntimeState
from app.services.log_service import LogService
from app.storage.snapshot_service import SnapshotService
from app.tracking.person_detection_scheduler import PersonDetectionScheduler
from app.tracking.person_detection_service import PersonDetectionService

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(
        self,
        runtime_state: RuntimeState,
        recognition_service: RecognitionService,
        face_recognition_scheduler: FaceRecognitionScheduler,
        person_detection_service: PersonDetectionService,
        person_detection_scheduler: PersonDetectionScheduler,
        attendance_recovery_service: AttendanceRecoveryService,
        plate_recognition_service: PlateRecognitionService,
        fire_detection_service: FireDetectionService,
        snapshot_service: SnapshotService,
        event_service: EventService,
        attendance_service: AttendanceService,
        log_service: LogService,
        settings: Settings,
    ):
        self.runtime_state = runtime_state
        self.recognition_service = recognition_service
        self.person_detection_service = person_detection_service
        self.person_detection_scheduler = person_detection_scheduler
        self.attendance_recovery_service = attendance_recovery_service
        self.plate_recognition_service = plate_recognition_service
        self.fire_detection_service = fire_detection_service
        self.snapshot_service = snapshot_service
        self.event_service = event_service
        self.attendance_service = attendance_service
        self.log_service = log_service
        self.settings = settings
        self.face_recognition_scheduler = face_recognition_scheduler
        self.workers: dict[str, CameraWorker] = {}

    async def start_camera(self, camera_id: str) -> dict:
        camera = self.runtime_state.get_camera(camera_id)
        if camera is None:
            raise KeyError("Camera not found in synced ERP config. Run /api/sync/cameras first.")
        if not camera.enabled:
            raise ValueError("Camera is disabled in ERP")

        existing = self.workers.get(camera_id)
        if existing and existing.is_running:
            return {"cameraId": camera_id, "status": "running", "message": "Camera already running"}
        if existing and not existing.is_running:
            self.workers.pop(camera_id, None)

        worker = CameraWorker(
            camera=camera,
            rules_by_tenant={
                assignment.tenantId: self.runtime_state.get_rules(assignment.tenantId)
                for assignment in camera.activeAssignments
            },
            recognition_service=self.recognition_service,
            recognition_scheduler=self.face_recognition_scheduler,
            person_detection_service=self.person_detection_service,
            person_detection_scheduler=self.person_detection_scheduler,
            attendance_recovery_service=self.attendance_recovery_service,
            plate_recognition_service=self.plate_recognition_service,
            fire_detection_service=self.fire_detection_service,
            snapshot_service=self.snapshot_service,
            event_service=self.event_service,
            attendance_service=self.attendance_service,
            log_service=self.log_service,
            settings=self.settings,
        )
        self.workers[camera_id] = worker
        worker.start()
        logger.info("Started worker for camera %s", camera_id)
        return {"cameraId": camera_id, "status": "starting", "message": "Camera worker starting"}

    async def stop_camera(self, camera_id: str) -> dict:
        worker = self.workers.pop(camera_id, None)
        if worker is None:
            return {"cameraId": camera_id, "status": "stopped", "message": "Camera was not running"}

        await worker.stop()
        logger.info("Stopped worker for camera %s", camera_id)
        return {"cameraId": camera_id, "status": "stopped", "message": "Camera stopped"}

    async def start_all(self) -> dict:
        results = []
        for camera in self.runtime_state.list_cameras():
            if camera.enabled:
                try:
                    results.append(await self.start_camera(camera.cameraId))
                except Exception as exc:
                    results.append({"cameraId": camera.cameraId, "status": "error", "message": str(exc)})
        return {"started": results}

    async def stop_all(self) -> dict:
        camera_ids = list(self.workers.keys())
        stopped = await asyncio.gather(
            *(self.stop_camera(camera_id) for camera_id in camera_ids),
            return_exceptions=True,
        )
        results = [
            result
            if not isinstance(result, Exception)
            else {"cameraId": camera_id, "status": "error", "message": str(result)}
            for camera_id, result in zip(camera_ids, stopped)
        ]
        return {"stopped": results}

    async def shutdown(self) -> dict:
        stopped = await self.stop_all()
        await self.face_recognition_scheduler.close()
        await self.person_detection_scheduler.close()
        return stopped

    async def restart_all(self) -> dict:
        stopped = await self.stop_all()
        started = await self.start_all()
        return {
            "stopped": stopped["stopped"],
            "started": started["started"],
        }

    def status(self) -> dict:
        configured = self.runtime_state.list_cameras()
        running_ids = {camera_id for camera_id, worker in self.workers.items() if worker.is_running}
        return {
            "configuredCameras": len(configured),
            "runningCameras": len(running_ids),
            "lastSync": self.runtime_state.last_sync,
            "faceScheduler": {
                "pendingJobs": self.face_recognition_scheduler.pending_jobs,
                "idle": self.face_recognition_scheduler.is_idle,
            },
            "personDetector": self.person_detection_scheduler.status(),
            "cameras": [
                {
                    "etsAuth": camera.tenantId,
                    "etsAuths": camera.tenantIds,
                    "cameraId": camera.cameraId,
                    "name": camera.name,
                    "enabled": camera.enabled,
                    "direction": camera.direction,
                    "capabilities": [capability.value for capability in camera.capabilities],
                    "assignments": [
                        assignment.model_dump(mode="json", by_alias=True)
                        for assignment in camera.assignments
                    ],
                    "status": "running" if camera.cameraId in running_ids else "stopped",
                    "runtime": (
                        self.workers[camera.cameraId].status()
                        if camera.cameraId in self.workers
                        else None
                    ),
                }
                for camera in configured
            ],
        }

    async def mjpeg_stream(self, camera_id: str, overlay: bool = False) -> AsyncGenerator[bytes, None]:
        camera = self.runtime_state.get_camera(camera_id)
        if camera is None:
            raise KeyError("Camera not found in synced ERP config. Run /api/sync/cameras first.")

        worker = self.workers.get(camera_id)
        if worker and worker.is_running:
            async for chunk in self._worker_stream(worker, overlay=overlay):
                yield chunk
            return

        async for chunk in self._direct_stream(camera.rtspUrl, overlay=overlay):
            yield chunk

    async def _worker_stream(self, worker: CameraWorker, overlay: bool = False) -> AsyncGenerator[bytes, None]:
        delay = 1 / self.settings.stream_fps
        worker.add_stream_viewer(overlay=overlay)
        try:
            while worker.is_running:
                jpeg = worker.latest_overlay_jpeg if overlay and worker.latest_overlay_jpeg else worker.latest_jpeg
                if jpeg:
                    yield self._mjpeg_chunk(jpeg)
                await asyncio.sleep(delay)
        finally:
            worker.remove_stream_viewer(overlay=overlay)

    async def _direct_stream(self, camera_source: str, overlay: bool = False) -> AsyncGenerator[bytes, None]:
        reader = RtspReader(camera_source, self.settings)
        delay = 1 / self.settings.stream_fps
        fps_started_at = time.perf_counter()
        fps_frames = 0
        display_fps = 0.0
        try:
            await asyncio.to_thread(reader.open)
            while True:
                frame = await asyncio.to_thread(reader.read)
                fps_frames += 1
                elapsed = time.perf_counter() - fps_started_at
                if elapsed >= 1.0:
                    display_fps = fps_frames / elapsed
                    fps_frames = 0
                    fps_started_at = time.perf_counter()
                if overlay and self.settings.show_dev_fps:
                    self._draw_fps(frame, display_fps)
                jpeg = await asyncio.to_thread(self._encode_jpeg, frame)
                if jpeg:
                    yield self._mjpeg_chunk(jpeg)
                await asyncio.sleep(delay)
        finally:
            await asyncio.to_thread(reader.close)

    def _encode_jpeg(self, frame) -> bytes | None:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality],
        )
        if not ok:
            return None
        return buffer.tobytes()

    @staticmethod
    def _mjpeg_chunk(jpeg: bytes) -> bytes:
        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    @staticmethod
    def _draw_fps(frame, fps: float) -> None:
        cv2.putText(
            frame,
            f"DEV FPS: {fps:.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
