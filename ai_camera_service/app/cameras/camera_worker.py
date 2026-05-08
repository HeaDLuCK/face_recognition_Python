import asyncio
import logging
import time
from datetime import datetime

import cv2
import numpy as np

from app.attendance.attendance_service import AttendanceService
from app.cameras.rtsp_reader import RtspReader
from app.config import Settings
from app.events.event_service import EventService
from app.face.recognition_service import RecognitionService
from app.schemas.erp_schema import AiCapability, AttendanceRules, CameraConfig, ZoneConfig
from app.schemas.runtime_schema import RuntimeEvent
from app.services.log_service import LogService
from app.services.module_registry import is_enabled
from app.storage.snapshot_service import SnapshotService

logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        rules: AttendanceRules,
        recognition_service: RecognitionService,
        snapshot_service: SnapshotService,
        event_service: EventService,
        attendance_service: AttendanceService,
        log_service: LogService,
        settings: Settings,
    ):
        self.camera = camera
        self.rules = rules
        self.recognition_service = recognition_service
        self.snapshot_service = snapshot_service
        self.event_service = event_service
        self.attendance_service = attendance_service
        self.log_service = log_service
        self.settings = settings
        self.reader = RtspReader(camera.rtspUrl, settings)
        self._task: asyncio.Task | None = None
        self._recognition_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.latest_jpeg: bytes | None = None
        self.latest_detections: list[dict] = []
        self._fps_started_at = time.perf_counter()
        self._fps_frames = 0
        self._display_fps = 0.0

    @property
    def camera_id(self) -> str:
        return self.camera.cameraId

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name=f"camera-worker-{self.camera_id}")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task

    async def _run(self) -> None:
        frame_count = 0
        try:
            await asyncio.to_thread(self.reader.open)
            await self._emit_camera_event("CAMERA_STARTED", {"startedAt": datetime.utcnow().isoformat()})

            while not self._stop_event.is_set():
                frame = await asyncio.to_thread(self.reader.read)
                self._update_fps()
                self._update_latest_jpeg(frame)
                frame_count += 1
                if frame_count % self.settings.camera_frame_skip != 0:
                    await asyncio.sleep(0)
                    continue

                if is_enabled(self.camera.capabilities, AiCapability.FACE_RECOGNITION):
                    self._schedule_face_recognition(frame)

                # Future module hooks belong here, one service per capability.
                await asyncio.sleep(0)

        except Exception as exc:
            logger.exception("Camera worker failed for camera %s", self.camera_id)
            await self.log_service.write(
                "ERROR",
                "Camera worker failed",
                tenant_id=self.camera.tenantId,
                camera_id=self.camera_id,
                metadata={"error": str(exc)},
            )
            await self._emit_camera_event("CAMERA_ERROR", {"error": str(exc)})
        finally:
            if self._recognition_task and not self._recognition_task.done():
                self._recognition_task.cancel()
            await asyncio.to_thread(self.reader.close)
            if self._stop_event.is_set():
                await self._emit_camera_event("CAMERA_STOPPED", {"stoppedAt": datetime.utcnow().isoformat()})

    def _schedule_face_recognition(self, frame) -> None:
        if self._recognition_task and not self._recognition_task.done():
            return
        self._recognition_task = asyncio.create_task(
            self._run_face_recognition(frame.copy()),
            name=f"face-recognition-{self.camera_id}",
        )

    async def _run_face_recognition(self, frame) -> None:
        await asyncio.sleep(self.settings.recognition_interval_seconds)
        results = await self.recognition_service.recognize_frame(
            tenant_id=self.camera.tenantId,
            frame=frame,
            threshold=self.rules.recognitionThreshold,
        )
        self.latest_detections = results
        for result in results:
            zone = self._matching_zone(result["bbox"])
            if self.camera.zones and zone is None:
                continue

            if result["matched"]:
                await self._handle_recognized(frame, result, zone)
            elif self.rules.saveUnknownFaces:
                await self._handle_unknown(frame, result, zone)
            else:
                await self._record_unknown_detection(result, zone)

    async def _handle_recognized(self, frame, result: dict, zone: ZoneConfig | None) -> None:
        snapshot_path = await asyncio.to_thread(
            self.snapshot_service.save_frame,
            self.camera.tenantId,
            self.camera.cameraId,
            frame,
        )
        metadata = self._metadata(result, zone)
        await self.snapshot_service.save_metadata(
            self.camera.tenantId,
            self.camera.cameraId,
            snapshot_path,
            "FACE_RECOGNITION",
            metadata,
        )

        create_attendance, direction = await self.attendance_service.should_create_attendance(
            tenant_id=self.camera.tenantId,
            employee_id=result["employeeId"],
            camera_direction=self.camera.direction,
            confidence=result["confidence"],
            rules=self.rules,
        )
        event_type = f"ATTENDANCE_{direction}" if create_attendance and direction else "FACE_RECOGNIZED"

        await self.attendance_service.record_detection(
            tenant_id=self.camera.tenantId,
            camera_id=self.camera.cameraId,
            event_type=event_type,
            employee_id=result["employeeId"],
            matched=True,
            confidence=result["confidence"],
            snapshot_path=snapshot_path,
            metadata=metadata,
        )
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=self.camera.tenantId,
                cameraId=self.camera.cameraId,
                eventType=event_type,
                employeeId=result["employeeId"],
                confidence=result["confidence"],
                snapshotPath=snapshot_path,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            ),
            send_to_erp=True,
        )

    async def _handle_unknown(self, frame, result: dict, zone: ZoneConfig | None) -> None:
        snapshot_path = await asyncio.to_thread(
            self.snapshot_service.save_frame,
            self.camera.tenantId,
            self.camera.cameraId,
            frame,
        )
        metadata = self._metadata(result, zone)
        await self.snapshot_service.save_metadata(
            self.camera.tenantId,
            self.camera.cameraId,
            snapshot_path,
            "UNKNOWN_FACE",
            metadata,
        )
        await self.attendance_service.record_detection(
            tenant_id=self.camera.tenantId,
            camera_id=self.camera.cameraId,
            event_type="UNKNOWN_FACE",
            employee_id=None,
            matched=False,
            confidence=result["confidence"],
            snapshot_path=snapshot_path,
            metadata=metadata,
        )
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=self.camera.tenantId,
                cameraId=self.camera.cameraId,
                eventType="UNKNOWN_FACE",
                confidence=result["confidence"],
                snapshotPath=snapshot_path,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            ),
            send_to_erp=True,
        )
        if self.rules.sendUnknownFaceAlert:
            await self.event_service.create_alert_event(
                RuntimeEvent(
                    tenantId=self.camera.tenantId,
                    cameraId=self.camera.cameraId,
                    eventType="UNKNOWN_FACE_ALERT",
                    confidence=result["confidence"],
                    snapshotPath=snapshot_path,
                    timestamp=datetime.utcnow(),
                    metadata=metadata,
                ),
                send_to_erp=True,
            )

    async def _record_unknown_detection(self, result: dict, zone: ZoneConfig | None) -> None:
        await self.attendance_service.record_detection(
            tenant_id=self.camera.tenantId,
            camera_id=self.camera.cameraId,
            event_type="UNKNOWN_FACE",
            employee_id=None,
            matched=False,
            confidence=result["confidence"],
            snapshot_path=None,
            metadata=self._metadata(result, zone),
        )

    async def _emit_camera_event(self, event_type: str, metadata: dict) -> None:
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=self.camera.tenantId,
                cameraId=self.camera.cameraId,
                eventType=event_type,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            ),
            send_to_erp=True,
        )

    def _matching_zone(self, bbox: list[int]) -> ZoneConfig | None:
        if not self.camera.zones:
            return None
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        for zone in self.camera.zones:
            if zone.x <= center_x <= zone.x + zone.width and zone.y <= center_y <= zone.y + zone.height:
                return zone
        return None

    def _update_latest_jpeg(self, frame) -> None:
        if self.settings.environment == "development" and self.settings.show_dev_fps:
            self._draw_fps(frame)
        if self.settings.environment == "development" and self.settings.show_dev_detections:
            self._draw_detections(frame)
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality],
        )

    def _draw_detections(self, frame) -> None:
        for detection in self.latest_detections:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox]
            matched = detection.get("matched", False)
            color = (0, 255, 0) if matched else (0, 220, 255)
            label = "FACE OK" if matched else "FACE ?"
            confidence = detection.get("confidence")
            if confidence is not None:
                label = f"{label} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            triangle = [
                (max(x1, 0), max(y1 - 24, 0)),
                (max(x1 + 18, 0), max(y1 - 4, 0)),
                (max(x1 - 18, 0), max(y1 - 4, 0)),
            ]
            cv2.fillConvexPoly(frame, np.array(triangle, dtype=np.int32), color)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 32, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        if ok:
            self.latest_jpeg = buffer.tobytes()

    def _update_fps(self) -> None:
        self._fps_frames += 1
        elapsed = time.perf_counter() - self._fps_started_at
        if elapsed >= 1.0:
            self._display_fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_started_at = time.perf_counter()

    def _draw_fps(self, frame) -> None:
        cv2.putText(
            frame,
            f"DEV FPS: {self._display_fps:.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _metadata(result: dict, zone: ZoneConfig | None) -> dict:
        metadata = {
            "bbox": result["bbox"],
            "detectionScore": result["detectionScore"],
            "employeeName": result.get("employeeName"),
        }
        if zone:
            metadata["zoneId"] = zone.zoneId
            metadata["zoneName"] = zone.name
        return metadata
