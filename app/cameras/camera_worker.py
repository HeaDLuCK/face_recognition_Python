import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from app.attendance.attendance_service import AttendanceService
from app.cameras.rtsp_reader import RtspReader
from app.config import Settings
from app.events.event_service import EventService
from app.face.recognition_scheduler import FaceRecognitionScheduler
from app.face.recognition_service import RecognitionService
from app.fire.fire_detection_service import FireDetectionService
from app.plates.plate_recognition_service import PlateRecognitionService
from app.schemas.erp_schema import AiCapability, AttendanceRules, CameraAssignment, CameraConfig, ZoneConfig
from app.schemas.runtime_schema import RuntimeEvent
from app.services.log_service import LogService
from app.services.module_registry import is_enabled
from app.storage.snapshot_service import SnapshotService

logger = logging.getLogger(__name__)


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        rules_by_tenant: dict[str, AttendanceRules],
        recognition_service: RecognitionService,
        recognition_scheduler: FaceRecognitionScheduler,
        plate_recognition_service: PlateRecognitionService,
        fire_detection_service: FireDetectionService,
        snapshot_service: SnapshotService,
        event_service: EventService,
        attendance_service: AttendanceService,
        log_service: LogService,
        settings: Settings,
    ):
        self.camera = camera
        self.rules_by_tenant = rules_by_tenant
        self.recognition_service = recognition_service
        self.recognition_scheduler = recognition_scheduler
        self.plate_recognition_service = plate_recognition_service
        self.fire_detection_service = fire_detection_service
        self.snapshot_service = snapshot_service
        self.event_service = event_service
        self.attendance_service = attendance_service
        self.log_service = log_service
        self.settings = settings
        self.reader = RtspReader(camera.rtspUrl, settings)
        self._task: asyncio.Task | None = None
        self._jpeg_task: asyncio.Task | None = None
        self._plate_recognition_worker_task: asyncio.Task | None = None
        self._plate_recognition_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=settings.plate_recognition_queue_size)
        self._plate_recognition_disabled = False
        self._fire_detection_worker_task: asyncio.Task | None = None
        self._fire_detection_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=settings.fire_detection_queue_size)
        self._fire_detection_disabled = False
        self._cloud_stream_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.latest_jpeg: bytes | None = None
        self.latest_overlay_jpeg: bytes | None = None
        self.latest_detections: list[dict] = []
        self.latest_plate_detections: list[dict] = []
        self.latest_fire_detections: list[dict] = []
        self._fps_started_at = time.perf_counter()
        self._fps_frames = 0
        self._display_fps = 0.0
        self._last_jpeg_queued_at = 0.0
        self._last_recognition_queued_at = 0.0
        self._last_plate_recognition_queued_at = 0.0
        self._last_fire_detection_queued_at = 0.0
        self._face_candidates: deque[tuple[float, float, np.ndarray]] = deque(
            maxlen=settings.face_candidate_buffer_size
        )
        self._face_candidate_window_started_at: float | None = None
        self._last_face_candidate_at = 0.0
        self._event_marker_tasks: set[asyncio.Task] = set()
        self._last_event_marker_at: dict[str, float] = {}
        self._plate_event_cache: dict[str, float] = {}
        self._fire_event_cache: dict[str, float] = {}
        self._recognized_face_cache: dict[tuple[str, str], float] = {}
        self._weak_recognition_tracks: dict[tuple[str, str], dict] = {}
        self._unknown_face_cache: deque[tuple[float, str, np.ndarray]] = deque(
            maxlen=settings.unknown_face_cache_max_entries
        )
        self._unknown_face_crop_cache: deque[tuple[str, np.ndarray, str]] = deque(
            maxlen=settings.unknown_face_crop_cache_max_entries
        )
        self._motion_zones = self._parse_motion_zones(settings.motion_zones)
        self._previous_motion_gray: np.ndarray | None = None
        self._stream_viewers = 0
        self._overlay_viewers = 0

    @property
    def camera_id(self) -> str:
        return self.camera.cameraId

    @property
    def primary_assignment(self) -> CameraAssignment:
        return self.camera.activeAssignments[0]

    def _rules_for(self, tenant_id: str) -> AttendanceRules:
        return self.rules_by_tenant[tenant_id]

    def _face_snapshots_enabled(self) -> bool:
        return any(
            self._rules_for(assignment.tenantId).saveFaceSnapshots
            for assignment in self.camera.assignments_for(AiCapability.FACE_RECOGNITION)
        )

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
            self._start_cloud_stream_push()
            if is_enabled(self.camera.capabilities, AiCapability.FACE_RECOGNITION):
                self.recognition_scheduler.register(self.camera_id, self._run_face_recognition)
            if is_enabled(self.camera.capabilities, AiCapability.PLATE_RECOGNITION):
                self._start_plate_recognition_worker()
            if is_enabled(self.camera.capabilities, AiCapability.FIRE_DETECTION):
                self._start_fire_detection_worker()

            while not self._stop_event.is_set():
                frame = await asyncio.to_thread(self.reader.read)
                self._update_fps()
                if self.stream_requested:
                    self._schedule_latest_jpeg(frame)
                frame_count += 1
                if frame_count % self.settings.motion_check_frame_skip == 0:
                    self._check_motion_zones(frame)

                if frame_count % self.settings.camera_frame_skip != 0:
                    await asyncio.sleep(0)
                    continue

                if is_enabled(self.camera.capabilities, AiCapability.FACE_RECOGNITION):
                    self._collect_face_candidate(frame)
                    if self._face_candidate_window_ready() and self._should_queue_face_recognition():
                        candidate = self._take_best_face_candidate()
                        if candidate is not None:
                            await self._queue_face_recognition(candidate)

                if (
                    not self._plate_recognition_disabled
                    and is_enabled(self.camera.capabilities, AiCapability.PLATE_RECOGNITION)
                    and self._should_queue_plate_recognition()
                ):
                    await self._queue_plate_recognition(frame)

                if (
                    not self._fire_detection_disabled
                    and is_enabled(self.camera.capabilities, AiCapability.FIRE_DETECTION)
                    and self._should_queue_fire_detection()
                ):
                    await self._queue_fire_detection(frame)

                await asyncio.sleep(0)

        except Exception as exc:
            logger.exception("Camera worker failed for camera %s", self.camera_id)
            await self.log_service.write(
                "ERROR",
                "Camera worker failed",
                tenant_id=self.primary_assignment.tenantId,
                camera_id=self.camera_id,
                metadata={"error": str(exc), "etsAuths": self.camera.tenantIds},
            )
            await self._emit_camera_event("CAMERA_ERROR", {"error": str(exc)})
        finally:
            self.recognition_scheduler.unregister(self.camera_id)
            await self._cancel_background_tasks()
            self._clear_runtime_buffers()
            await asyncio.to_thread(self.reader.close)
            if self._stop_event.is_set():
                await self._emit_camera_event("CAMERA_STOPPED", {"stoppedAt": datetime.utcnow().isoformat()})

    async def _cancel_background_tasks(self) -> None:
        tasks = {
            task
            for task in (
                self._plate_recognition_worker_task,
                self._fire_detection_worker_task,
                self._jpeg_task,
                self._cloud_stream_task,
                *self._event_marker_tasks,
            )
            if task is not None
        }
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._event_marker_tasks.clear()

    def _clear_runtime_buffers(self) -> None:
        self._face_candidates.clear()
        self._face_candidate_window_started_at = None
        self._last_face_candidate_at = 0.0
        self._unknown_face_cache.clear()
        self._unknown_face_crop_cache.clear()
        self._weak_recognition_tracks.clear()
        self._recognized_face_cache.clear()
        self._plate_event_cache.clear()
        self._fire_event_cache.clear()
        self._drain_queue(self._plate_recognition_queue)
        self._drain_queue(self._fire_detection_queue)

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                return

    def _start_plate_recognition_worker(self) -> None:
        if self._plate_recognition_worker_task and not self._plate_recognition_worker_task.done():
            return
        self._plate_recognition_worker_task = asyncio.create_task(
            self._run_plate_recognition_worker(),
            name=f"plate-recognition-worker-{self.camera_id}",
        )

    def _start_fire_detection_worker(self) -> None:
        if self._fire_detection_worker_task and not self._fire_detection_worker_task.done():
            return
        self._fire_detection_worker_task = asyncio.create_task(
            self._run_fire_detection_worker(),
            name=f"fire-detection-worker-{self.camera_id}",
        )

    @property
    def stream_requested(self) -> bool:
        return self._stream_viewers > 0 or bool(self.settings.cloud_stream_ws_url)

    @property
    def overlay_requested(self) -> bool:
        return self._overlay_viewers > 0

    def add_stream_viewer(self, overlay: bool = False) -> None:
        self._stream_viewers += 1
        if overlay:
            self._overlay_viewers += 1

    def remove_stream_viewer(self, overlay: bool = False) -> None:
        self._stream_viewers = max(self._stream_viewers - 1, 0)
        if overlay:
            self._overlay_viewers = max(self._overlay_viewers - 1, 0)

    def _should_queue_face_recognition(self) -> bool:
        now = time.monotonic()
        if now - self._last_recognition_queued_at < self.settings.recognition_interval_seconds:
            return False
        self._last_recognition_queued_at = now
        return True

    def _collect_face_candidate(self, frame: np.ndarray) -> None:
        captured_at = time.monotonic()
        sampling_interval = (
            self.settings.face_candidate_window_seconds
            / self.settings.face_candidate_buffer_size
        )
        if captured_at - self._last_face_candidate_at < sampling_interval:
            return

        self._last_face_candidate_at = captured_at
        if self._face_candidate_window_started_at is None:
            self._face_candidate_window_started_at = captured_at
        self._face_candidates.append(
            (captured_at, self._face_frame_quality_score(frame), frame)
        )

    def _face_candidate_window_ready(self) -> bool:
        if not self._face_candidates or self._face_candidate_window_started_at is None:
            return False
        return (
            time.monotonic() - self._face_candidate_window_started_at
            >= self.settings.face_candidate_window_seconds
        )

    def _take_best_face_candidate(self) -> np.ndarray | None:
        if not self._face_candidates:
            return None
        _, _, best_frame = max(self._face_candidates, key=lambda candidate: candidate[1])
        self._face_candidates.clear()
        self._face_candidate_window_started_at = None
        return best_frame

    @staticmethod
    def _face_frame_quality_score(frame: np.ndarray) -> float:
        height, width = frame.shape[:2]
        longest_side = max(height, width)
        if longest_side > 320:
            scale = 320.0 / longest_side
            sample = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            sample = frame

        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        contrast = float(gray.std())

        exposure_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
        contrast_score = min(contrast / 48.0, 1.0)
        return sharpness * (0.5 + 0.35 * exposure_score + 0.15 * contrast_score)

    async def _queue_face_recognition(self, frame) -> None:
        self.recognition_scheduler.submit(self.camera_id, frame)

    def _should_queue_plate_recognition(self) -> bool:
        now = time.monotonic()
        if now - self._last_plate_recognition_queued_at < self.settings.plate_recognition_interval_seconds:
            return False
        self._last_plate_recognition_queued_at = now
        return True

    async def _queue_plate_recognition(self, frame) -> None:
        frame_copy = frame.copy()
        if not self.settings.recognition_drop_old_frames:
            await self._plate_recognition_queue.put(frame_copy)
            return

        while True:
            try:
                self._plate_recognition_queue.get_nowait()
                self._plate_recognition_queue.task_done()
            except asyncio.QueueEmpty:
                break
        try:
            self._plate_recognition_queue.put_nowait(frame_copy)
        except asyncio.QueueFull:
            pass

    def _should_queue_fire_detection(self) -> bool:
        now = time.monotonic()
        if now - self._last_fire_detection_queued_at < self.settings.fire_detection_interval_seconds:
            return False
        self._last_fire_detection_queued_at = now
        return True

    async def _queue_fire_detection(self, frame) -> None:
        frame_copy = frame.copy()
        if not self.settings.recognition_drop_old_frames:
            await self._fire_detection_queue.put(frame_copy)
            return

        while True:
            try:
                self._fire_detection_queue.get_nowait()
                self._fire_detection_queue.task_done()
            except asyncio.QueueEmpty:
                break
        try:
            self._fire_detection_queue.put_nowait(frame_copy)
        except asyncio.QueueFull:
            pass

    def _schedule_latest_jpeg(self, frame) -> None:
        now = time.monotonic()
        if now - self._last_jpeg_queued_at < 1 / self.settings.stream_fps:
            return
        if self._jpeg_task and not self._jpeg_task.done():
            return
        self._last_jpeg_queued_at = now
        self._jpeg_task = asyncio.create_task(
            self._encode_latest_jpeg(frame.copy()),
            name=f"jpeg-encoder-{self.camera_id}",
        )

    async def _encode_latest_jpeg(self, frame) -> None:
        clean_jpeg, overlay_jpeg = await asyncio.to_thread(self._encode_stream_jpegs, frame)
        if clean_jpeg:
            self.latest_jpeg = clean_jpeg
        if overlay_jpeg:
            self.latest_overlay_jpeg = overlay_jpeg

    async def _run_plate_recognition_worker(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._plate_recognition_queue.get()
            try:
                await self._run_plate_recognition(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._plate_recognition_disabled = True
                logger.exception("Plate recognition disabled for camera %s after an error", self.camera_id)
                return
            finally:
                self._plate_recognition_queue.task_done()

    async def _run_fire_detection_worker(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._fire_detection_queue.get()
            try:
                await self._run_fire_detection(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._fire_detection_disabled = True
                logger.exception("Fire detection disabled for camera %s after an error", self.camera_id)
                return
            finally:
                self._fire_detection_queue.task_done()

    async def _run_face_recognition(self, frame) -> None:
        assignments = self.camera.assignments_for(AiCapability.FACE_RECOGNITION)
        if not assignments:
            return

        results = await self.recognition_service.recognize_frame_for_tenants(
            tenant_thresholds={
                assignment.tenantId: self._rules_for(assignment.tenantId).recognitionThreshold
                for assignment in assignments
            },
            frame=frame,
        )
        self.latest_detections = self._public_detections(results)
        handled_results = []
        for result in results:
            if result["matched"]:
                assignment = self._assignment_for_tenant(result["tenantId"])
                if assignment is None:
                    continue
                zone = self._matching_zone(result["bbox"], assignment)
                if assignment.zones and zone is None:
                    continue
                if self._is_duplicate_recognized(result, assignment.tenantId):
                    continue
                self._weak_recognition_tracks.pop(
                    (assignment.tenantId, result.get("employeeId")),
                    None,
                )
                handled_results.append(("recognized", result, assignment, zone))
            else:
                for assignment in assignments:
                    tenant_result = self._result_for_tenant(result, assignment.tenantId)
                    if self._track_weak_known_candidate(frame, tenant_result):
                        continue
                    rules = self._rules_for(assignment.tenantId)
                    if not rules.saveUnknownFaces:
                        continue
                    zone = self._matching_zone(tenant_result["bbox"], assignment)
                    if assignment.zones and zone is None:
                        continue
                    if not self._is_good_unknown_candidate(frame, tenant_result, rules):
                        continue
                    if self._is_duplicate_unknown(tenant_result, assignment.tenantId):
                        continue
                    self._schedule_event_marker(
                        "FACE_DETECTION",
                        tenant_result,
                        tenant_ids=[assignment.tenantId],
                    )
                    handled_results.append(("unknown", tenant_result, assignment, zone))

        await self._flush_ready_weak_recognition_tracks()

        if not handled_results:
            return

        save_snapshots = self._face_snapshots_enabled()
        snapshot_paths: dict[str, str | None] = {}
        for kind, result, assignment, zone in handled_results:
            tenant_id = assignment.tenantId
            rules = self._rules_for(tenant_id)
            if tenant_id not in snapshot_paths:
                snapshot_paths[tenant_id] = (
                    await self._save_frame_snapshot(frame, results, tenant_id)
                    if save_snapshots
                    else None
                )
            snapshot_path = snapshot_paths[tenant_id]
            if kind == "recognized":
                await self._handle_recognized(result, assignment, zone, snapshot_path)
            else:
                face_crop_path = (
                    await self._save_unknown_face_crop(frame, result, tenant_id)
                    if rules.saveUnknownFaceCrops
                    else None
                )
                await self._handle_unknown(
                    result,
                    assignment,
                    zone,
                    snapshot_path,
                    face_crop_path,
                )

    async def _run_plate_recognition(self, frame) -> None:
        results = await asyncio.to_thread(self.plate_recognition_service.recognize_frame, frame)
        self.latest_plate_detections = self._public_detections(results)
        assignments = self.camera.assignments_for(AiCapability.PLATE_RECOGNITION)

        for result in results:
            if self._is_duplicate_plate(result):
                continue
            for assignment in assignments:
                zone = self._matching_zone(result["bbox"], assignment)
                if assignment.zones and zone is None:
                    continue
                await self._handle_plate_detection(frame, result, assignment, zone)

    async def _handle_plate_detection(
        self,
        frame,
        result: dict,
        assignment: CameraAssignment,
        zone: ZoneConfig | None,
    ) -> None:
        event_type = "PLATE_RECOGNIZED" if result.get("plateText") else "PLATE_DETECTED"
        timestamp = datetime.utcnow()
        snapshot_path = (
            await self._save_plate_snapshot(frame, [result], assignment.tenantId)
            if self.settings.plate_save_snapshots
            else None
        )
        metadata = self._plate_metadata(result, zone)
        metadata.update(
            {
                "code": event_type,
                "channel": self._camera_channel(),
                "playbackStartTime": (
                    timestamp - timedelta(seconds=self.settings.event_marker_history_before_seconds)
                ).isoformat(),
                "playbackEndTime": (
                    timestamp + timedelta(seconds=self.settings.event_marker_history_after_seconds)
                ).isoformat(),
            }
        )
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=assignment.tenantId,
                cameraId=self.camera.cameraId,
                eventType=event_type,
                confidence=result.get("confidence"),
                snapshotPath=snapshot_path,
                timestamp=timestamp,
                metadata=metadata,
            ),
        )

    async def _run_fire_detection(self, frame) -> None:
        results = await asyncio.to_thread(self.fire_detection_service.detect_frame, frame)
        self.latest_fire_detections = self._public_detections(results)
        assignments = self.camera.assignments_for(AiCapability.FIRE_DETECTION)

        for result in results:
            if self._is_duplicate_fire(result):
                continue
            for assignment in assignments:
                zone = self._matching_zone(result["bbox"], assignment)
                if assignment.zones and zone is None:
                    continue
                await self._handle_fire_detection(frame, result, assignment, zone)

    async def _handle_fire_detection(
        self,
        frame,
        result: dict,
        assignment: CameraAssignment,
        zone: ZoneConfig | None,
    ) -> None:
        event_type = self._fire_event_type(result)
        timestamp = datetime.utcnow()
        snapshot_path = (
            await self._save_fire_snapshot(frame, [result], assignment.tenantId)
            if self.settings.fire_save_snapshots
            else None
        )
        metadata = self._fire_metadata(result, zone)
        metadata.update(
            {
                "code": event_type,
                "channel": self._camera_channel(),
                "playbackStartTime": (
                    timestamp - timedelta(seconds=self.settings.event_marker_history_before_seconds)
                ).isoformat(),
                "playbackEndTime": (
                    timestamp + timedelta(seconds=self.settings.event_marker_history_after_seconds)
                ).isoformat(),
            }
        )
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=assignment.tenantId,
                cameraId=self.camera.cameraId,
                eventType=event_type,
                confidence=result.get("confidence"),
                snapshotPath=snapshot_path,
                timestamp=timestamp,
                metadata=metadata,
            ),
        )

    async def _handle_recognized(
        self,
        result: dict,
        assignment: CameraAssignment,
        zone: ZoneConfig | None,
        snapshot_path: str | None,
    ) -> None:
        metadata = self._metadata(result, zone)
        rules = self._rules_for(assignment.tenantId)

        create_attendance, direction = await self.attendance_service.should_create_attendance(
            tenant_id=assignment.tenantId,
            employee_id=result["employeeId"],
            camera_direction=assignment.direction,
            confidence=result["confidence"],
            rules=rules,
        )
        event_type = f"ATTENDANCE_{direction}" if create_attendance and direction else "FACE_RECOGNIZED"

        await self.attendance_service.record_detection(
            tenant_id=assignment.tenantId,
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
                tenantId=assignment.tenantId,
                cameraId=self.camera.cameraId,
                eventType=event_type,
                employeeId=result["employeeId"],
                confidence=result["confidence"],
                snapshotPath=snapshot_path,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            ),
        )

    def _start_cloud_stream_push(self) -> None:
        if not self.settings.cloud_stream_ws_url:
            return
        if self._cloud_stream_task and not self._cloud_stream_task.done():
            return

        self._cloud_stream_task = asyncio.create_task(
            self._push_stream_to_cloud(),
            name=f"cloud-stream-{self.camera_id}",
        )

    async def _push_stream_to_cloud(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.error("websockets is not installed. Install uvicorn[standard] or websockets to push streams.")
            return

        delay = 1 / self.settings.cloud_stream_fps
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self.settings.cloud_stream_ws_url) as websocket:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "camera_start",
                                "etsAuth": self.primary_assignment.tenantId,
                                "etsAuths": self.camera.tenantIds,
                                "cameraId": self.camera.cameraId,
                                "cameraName": self.camera.name,
                                "token": self.settings.cloud_stream_token,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )
                    logger.info("Pushing camera %s stream to cloud", self.camera_id)
                    while not self._stop_event.is_set():
                        if self.latest_jpeg:
                            await websocket.send(self.latest_jpeg)
                        await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Cloud stream push failed for %s: %s", self.camera_id, exc)
                await asyncio.sleep(self.settings.cloud_stream_reconnect_seconds)

    async def _handle_unknown(
        self,
        result: dict,
        assignment: CameraAssignment,
        zone: ZoneConfig | None,
        snapshot_path: str | None,
        face_crop_path: str | None = None,
    ) -> None:
        metadata = self._metadata(result, zone)
        if face_crop_path:
            metadata["faceCropPath"] = face_crop_path
        await self.attendance_service.record_detection(
            tenant_id=assignment.tenantId,
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
                tenantId=assignment.tenantId,
                cameraId=self.camera.cameraId,
                eventType="UNKNOWN_FACE",
                confidence=result["confidence"],
                snapshotPath=snapshot_path,
                timestamp=datetime.utcnow(),
                metadata=metadata,
            ),
        )
        if self._rules_for(assignment.tenantId).sendUnknownFaceAlert:
            await self.event_service.create_alert_event(
                RuntimeEvent(
                    tenantId=assignment.tenantId,
                    cameraId=self.camera.cameraId,
                    eventType="UNKNOWN_FACE_ALERT",
                    confidence=result["confidence"],
                    snapshotPath=snapshot_path,
                    timestamp=datetime.utcnow(),
                    metadata=metadata,
                ),
            )

    async def _emit_camera_event(self, event_type: str, metadata: dict) -> None:
        for assignment in self.camera.activeAssignments:
            await self.event_service.create_camera_event(
                RuntimeEvent(
                    tenantId=assignment.tenantId,
                    cameraId=self.camera.cameraId,
                    eventType=event_type,
                    timestamp=datetime.utcnow(),
                    metadata=metadata,
                ),
            )

    def _assignment_for_tenant(self, tenant_id: str | None) -> CameraAssignment | None:
        if not tenant_id:
            return None
        return next(
            (
                assignment
                for assignment in self.camera.activeAssignments
                if assignment.tenantId == tenant_id
            ),
            None,
        )

    @staticmethod
    def _result_for_tenant(result: dict, tenant_id: str) -> dict:
        candidate = (result.get("_tenantCandidates") or {}).get(tenant_id) or {}
        return {
            **result,
            "tenantId": tenant_id,
            "employeeId": None,
            "employeeName": None,
            "bestCandidateTenantId": tenant_id if candidate else None,
            "bestCandidateEmployeeId": candidate.get("employeeId"),
            "bestCandidateEmployeeName": candidate.get("employeeName"),
            "bestCandidateScore": candidate.get("score"),
            "matched": False,
            "confidence": candidate.get("score"),
        }

    @staticmethod
    def _matching_zone(
        bbox: list[int],
        assignment: CameraAssignment,
    ) -> ZoneConfig | None:
        if not assignment.zones:
            return None
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        for zone in assignment.zones:
            if zone.x <= center_x <= zone.x + zone.width and zone.y <= center_y <= zone.y + zone.height:
                return zone
        return None

    def _encode_stream_jpegs(self, frame) -> tuple[bytes | None, bytes | None]:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality],
        )
        clean_jpeg = buffer.tobytes() if ok else None

        if not self.overlay_requested:
            return clean_jpeg, None

        overlay_frame = frame.copy()
        if self.settings.show_motion_zones:
            self._draw_motion_zones(overlay_frame)
        if self.settings.show_dev_fps:
            self._draw_fps(overlay_frame)
        if self.settings.show_dev_detections:
            self._draw_detections(overlay_frame)
        ok, buffer = cv2.imencode(
            ".jpg",
            overlay_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality],
        )
        overlay_jpeg = buffer.tobytes() if ok else None
        return clean_jpeg, overlay_jpeg

    def _check_motion_zones(self, frame) -> None:
        if not self._motion_zones:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._previous_motion_gray is None:
            self._previous_motion_gray = gray
            return

        diff = cv2.absdiff(self._previous_motion_gray, gray)
        self._previous_motion_gray = gray

        for zone in self._motion_zones:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [zone["points"]], 255)
            changed = cv2.bitwise_and(diff, diff, mask=mask)
            _, threshold = cv2.threshold(
                changed,
                self.settings.motion_pixel_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            changed_pixels = cv2.countNonZero(threshold)
            zone_pixels = cv2.countNonZero(mask)
            if zone_pixels == 0:
                continue

            ratio = changed_pixels / zone_pixels
            if ratio >= self.settings.motion_area_ratio:
                self._schedule_event_marker(
                    "MOTION_DETECTION",
                    {"zoneId": zone["id"], "changedRatio": ratio},
                )

    def _schedule_event_marker(
        self,
        event_type: str,
        metadata: dict | None = None,
        tenant_ids: list[str] | None = None,
    ) -> None:
        target_tenant_ids = tenant_ids or self.camera.tenantIds
        cache_key = f"{event_type}:{','.join(sorted(target_tenant_ids))}"
        now = time.monotonic()
        last_event_at = self._last_event_marker_at.get(cache_key, 0)
        if now - last_event_at < self.settings.event_clip_cooldown_seconds:
            return

        self._remember_detection_cache(self._last_event_marker_at, cache_key, now)
        task = asyncio.create_task(
            self._save_event_marker(event_type, metadata or {}, target_tenant_ids),
            name=f"event-marker-{self.camera_id}-{event_type}",
        )
        self._event_marker_tasks.add(task)
        task.add_done_callback(self._event_marker_done)

    def _event_marker_done(self, task: asyncio.Task) -> None:
        self._event_marker_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Event marker failed for camera %s", self.camera_id)

    async def _save_event_marker(
        self,
        event_type: str,
        metadata: dict,
        tenant_ids: list[str],
    ) -> None:
        timestamp = datetime.utcnow()
        marker_metadata = {
            **metadata,
            "code": event_type,
            "channel": self._camera_channel(),
            "playbackStartTime": (
                timestamp - timedelta(seconds=self.settings.event_marker_history_before_seconds)
            ).isoformat(),
            "playbackEndTime": (
                timestamp + timedelta(seconds=self.settings.event_marker_history_after_seconds)
            ).isoformat(),
        }
        for tenant_id in tenant_ids:
            await self.event_service.create_camera_event(
                RuntimeEvent(
                    tenantId=tenant_id,
                    cameraId=self.camera.cameraId,
                    eventType=event_type,
                    timestamp=timestamp,
                    metadata=marker_metadata,
                ),
            )

    def _camera_channel(self) -> str | None:
        if "/Streaming/Channels/" in self.camera.rtspUrl:
            return self.camera.rtspUrl.rsplit("/", 1)[-1]
        if "/Streaming/tracks/" in self.camera.rtspUrl:
            return self.camera.rtspUrl.rsplit("/", 1)[-1]
        return None

    def _draw_motion_zones(self, frame) -> None:
        for zone in self._motion_zones:
            cv2.polylines(frame, [zone["points"]], True, (255, 180, 0), 2)
            label_x, label_y = zone["points"][0]
            cv2.putText(
                frame,
                zone["id"],
                (int(label_x), max(int(label_y) - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

    def _draw_detections(self, frame) -> None:
        for detection in self.latest_detections:
            self._draw_face_annotation(frame, detection, show_label=True)
        for detection in self.latest_plate_detections:
            self._draw_plate_annotation(frame, detection, show_label=True)
        for detection in self.latest_fire_detections:
            self._draw_fire_annotation(frame, detection, show_label=True)

    async def _save_frame_snapshot(
        self,
        frame,
        detections: list[dict],
        tenant_id: str,
    ) -> str:
        assignment = self._assignment_for_tenant(tenant_id)
        snapshot_frame = self._snapshot_frame(frame, detections)
        snapshot_path = await asyncio.to_thread(
            self.snapshot_service.save_frame,
            tenant_id,
            self.camera.cameraId,
            snapshot_frame,
        )
        await self.snapshot_service.save_metadata(
            tenant_id,
            self.camera.cameraId,
            snapshot_path,
            "FACE_DETECTIONS",
            {
                "detections": [
                    self._metadata(
                        detection,
                        self._matching_zone(detection["bbox"], assignment) if assignment else None,
                    )
                    for detection in detections
                ]
            },
        )
        return snapshot_path

    async def _save_plate_snapshot(self, frame, detections: list[dict], tenant_id: str) -> str:
        assignment = self._assignment_for_tenant(tenant_id)
        snapshot_frame = self._plate_snapshot_frame(frame, detections)
        snapshot_path = await asyncio.to_thread(
            self.snapshot_service.save_frame,
            tenant_id,
            self.camera.cameraId,
            snapshot_frame,
        )
        await self.snapshot_service.save_metadata(
            tenant_id,
            self.camera.cameraId,
            snapshot_path,
            "PLATE_DETECTIONS",
            {
                "detections": [
                    self._plate_metadata(
                        detection,
                        self._matching_zone(detection["bbox"], assignment) if assignment else None,
                    )
                    for detection in detections
                ]
            },
        )
        return snapshot_path

    async def _save_fire_snapshot(self, frame, detections: list[dict], tenant_id: str) -> str:
        assignment = self._assignment_for_tenant(tenant_id)
        snapshot_frame = self._fire_snapshot_frame(frame, detections)
        snapshot_path = await asyncio.to_thread(
            self.snapshot_service.save_frame,
            tenant_id,
            self.camera.cameraId,
            snapshot_frame,
        )
        await self.snapshot_service.save_metadata(
            tenant_id,
            self.camera.cameraId,
            snapshot_path,
            "FIRE_DETECTIONS",
            {
                "detections": [
                    self._fire_metadata(
                        detection,
                        self._matching_zone(detection["bbox"], assignment) if assignment else None,
                    )
                    for detection in detections
                ]
            },
        )
        return snapshot_path

    async def _save_unknown_face_crop(
        self,
        frame,
        result: dict,
        tenant_id: str,
    ) -> str | None:
        existing_path = self._matched_unknown_face_crop_path(result, tenant_id)
        if existing_path:
            return existing_path

        existing_path = await self._find_stored_unknown_face_crop_path(result, tenant_id)
        if existing_path:
            self._remember_unknown_face_crop(result, existing_path, tenant_id)
            return existing_path

        crop = self._face_crop(frame, result.get("bbox"))
        if crop is None:
            return None
        crop_path = await asyncio.to_thread(
            self.snapshot_service.save_face_crop,
            tenant_id,
            self.camera.cameraId,
            crop,
        )
        self._remember_unknown_face_crop(result, crop_path, tenant_id)
        await self._register_stored_unknown_face_crop(result, crop_path, tenant_id)
        return crop_path

    @staticmethod
    def _face_crop(frame, bbox: list[int] | None):
        if not bbox or len(bbox) != 4:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        face_width = max(x2 - x1, 1)
        face_height = max(y2 - y1, 1)
        margin_x = int(face_width * 0.45)
        margin_y = int(face_height * 0.60)

        left = max(x1 - margin_x, 0)
        top = max(y1 - margin_y, 0)
        right = min(x2 + margin_x, width)
        bottom = min(y2 + margin_y, height)
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right].copy()

    def _matched_unknown_face_crop_path(self, result: dict, tenant_id: str) -> str | None:
        embedding = result.get("_embedding")
        if embedding is None:
            return None

        current = np.array(embedding, dtype=np.float32)
        for cached_tenant_id, cached, crop_path in self._unknown_face_crop_cache:
            if cached_tenant_id != tenant_id:
                continue
            score = float(np.dot(current, cached))
            if score >= self.settings.unknown_face_crop_similarity_threshold:
                path = Path(crop_path)
                if not path.is_absolute():
                    path = Path.cwd() / path
                if path.is_file():
                    return crop_path
        return None

    def _remember_unknown_face_crop(self, result: dict, crop_path: str, tenant_id: str) -> None:
        embedding = result.get("_embedding")
        if embedding is None:
            return
        self._unknown_face_crop_cache.append(
            (tenant_id, np.array(embedding, dtype=np.float32), crop_path)
        )

    async def _find_stored_unknown_face_crop_path(self, result: dict, tenant_id: str) -> str | None:
        embedding = result.get("_embedding")
        if embedding is None:
            return None
        return await self.snapshot_service.find_unknown_face_crop(
            tenant_id,
            self.camera.cameraId,
            embedding,
            self.settings.unknown_face_crop_similarity_threshold,
        )

    async def _register_stored_unknown_face_crop(
        self,
        result: dict,
        crop_path: str,
        tenant_id: str,
    ) -> None:
        embedding = result.get("_embedding")
        if embedding is None:
            return
        await self.snapshot_service.register_unknown_face_crop(
            tenant_id,
            self.camera.cameraId,
            crop_path,
            embedding,
            metadata={"bbox": result.get("bbox"), "detectionScore": result.get("detectionScore")},
        )

    def _track_weak_known_candidate(self, frame, result: dict) -> bool:
        tenant_id = result.get("bestCandidateTenantId")
        employee_id = result.get("bestCandidateEmployeeId")
        score = result.get("bestCandidateScore")
        assignment = self._assignment_for_tenant(tenant_id)
        if not tenant_id or not employee_id or score is None or assignment is None:
            return False

        zone = self._matching_zone(result["bbox"], assignment)
        if assignment.zones and zone is None:
            return False

        rules = self._rules_for(tenant_id)
        min_score = rules.recognitionThreshold - self.settings.recognition_candidate_score_margin
        if score < min_score:
            return False

        now = time.monotonic()
        track_key = (tenant_id, employee_id)
        track = self._weak_recognition_tracks.get(track_key)
        track_expired = track is not None and now - track["started_at"] > self.settings.recognition_candidate_window_seconds
        if (
            track is None
            or (track_expired and track["hits"] < self.settings.recognition_candidate_min_hits)
        ):
            track = {
                "started_at": now,
                "hits": 0,
                "best_quality_score": -1.0,
                "best_frame": None,
                "best_result": None,
                "best_zone": None,
                "assignment": assignment,
            }
            self._weak_recognition_tracks[track_key] = track

        track["hits"] += 1
        quality_score = self._candidate_quality_score(frame, result)
        if quality_score > track["best_quality_score"]:
            track["best_quality_score"] = quality_score
            track["best_frame"] = frame.copy()
            track["best_result"] = self._promoted_known_result(result, tenant_id, rules)
            track["best_zone"] = zone
        return True

    def _candidate_quality_score(self, frame, result: dict) -> float:
        bbox = result.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        face_height = max(y2 - y1, 1)
        crop = self._face_crop(frame, bbox)
        blur_score = min(self._blur_score(crop), 300.0) / 300.0 if crop is not None else 0.0
        recognition_score = float(result.get("bestCandidateScore") or 0.0)
        detection_score = float(result.get("detectionScore") or 0.0)
        size_score = min(face_height / 140.0, 1.0)
        return recognition_score * 0.55 + detection_score * 0.20 + size_score * 0.15 + blur_score * 0.10

    def _promoted_known_result(
        self,
        result: dict,
        tenant_id: str,
        rules: AttendanceRules,
    ) -> dict:
        promoted = dict(result)
        raw_score = result.get("bestCandidateScore")
        promoted["matched"] = True
        promoted["employeeId"] = result.get("bestCandidateEmployeeId")
        promoted["employeeName"] = result.get("bestCandidateEmployeeName")
        promoted["tenantId"] = tenant_id
        promoted["rawCandidateScore"] = raw_score
        promoted["confidence"] = max(raw_score or 0.0, rules.recognitionThreshold)
        promoted["promotedFromWeakCandidate"] = True
        return promoted

    async def _flush_ready_weak_recognition_tracks(self) -> None:
        now = time.monotonic()
        stale_track_keys = [
            track_key
            for track_key, track in self._weak_recognition_tracks.items()
            if track["hits"] < self.settings.recognition_candidate_min_hits
            and now - track["started_at"] >= self.settings.recognition_candidate_window_seconds
        ]
        for track_key in stale_track_keys:
            self._weak_recognition_tracks.pop(track_key, None)

        ready_track_keys = [
            track_key
            for track_key, track in self._weak_recognition_tracks.items()
            if self._weak_recognition_track_is_ready(track, now)
        ]

        for track_key in ready_track_keys:
            track = self._weak_recognition_tracks.pop(track_key, None)
            if not track or not track["best_result"] or track["best_frame"] is None:
                continue

            result = track["best_result"]
            assignment = track["assignment"]
            rules = self._rules_for(assignment.tenantId)
            if self._is_duplicate_recognized(result, assignment.tenantId):
                continue

            snapshot_path = (
                await self._save_frame_snapshot(
                    track["best_frame"],
                    [result],
                    assignment.tenantId,
                )
                if self._face_snapshots_enabled()
                else None
            )
            await self._handle_recognized(
                result,
                assignment,
                track["best_zone"],
                snapshot_path,
            )

    def _weak_recognition_track_is_ready(self, track: dict, now: float) -> bool:
        if track["hits"] < self.settings.recognition_candidate_min_hits:
            return False

        result = track.get("best_result") or {}
        assignment = track["assignment"]
        rules = self._rules_for(assignment.tenantId)
        raw_score = result.get("rawCandidateScore") or result.get("bestCandidateScore") or 0.0
        near_threshold = raw_score >= rules.recognitionThreshold - self.settings.recognition_candidate_fast_margin
        window_elapsed = now - track["started_at"] >= self.settings.recognition_candidate_window_seconds
        return near_threshold or window_elapsed

    def _is_good_unknown_candidate(
        self,
        frame,
        result: dict,
        rules: AttendanceRules,
    ) -> bool:
        detection_score = result.get("detectionScore")
        if detection_score is None or detection_score < self.settings.unknown_face_min_detection_score:
            return False

        bbox = result.get("bbox")
        if not bbox or len(bbox) != 4:
            return False

        _, y1, _, y2 = [int(value) for value in bbox]
        if y2 - y1 < self.settings.unknown_face_min_height_px:
            return False

        best_candidate_score = result.get("bestCandidateScore")
        if (
            best_candidate_score is not None
            and best_candidate_score >= rules.recognitionThreshold - self.settings.unknown_face_skip_weak_known_margin
        ):
            return False

        crop = self._face_crop(frame, bbox)
        if crop is None:
            return False

        blur_score = self._blur_score(crop)
        return blur_score >= self.settings.unknown_face_min_blur_score

    @staticmethod
    def _blur_score(frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_duplicate_unknown(self, result: dict, tenant_id: str) -> bool:
        embedding = result.get("_embedding")
        if embedding is None or self.settings.unknown_duplicate_cooldown_seconds <= 0:
            return False

        now = time.monotonic()
        cooldown = self.settings.unknown_duplicate_cooldown_seconds
        while self._unknown_face_cache and now - self._unknown_face_cache[0][0] > cooldown:
            self._unknown_face_cache.popleft()

        current = np.array(embedding, dtype=np.float32)
        for _, cached_tenant_id, cached in self._unknown_face_cache:
            if cached_tenant_id != tenant_id:
                continue
            score = float(np.dot(current, cached))
            if score >= self.settings.unknown_duplicate_similarity_threshold:
                return True

        self._unknown_face_cache.append((now, tenant_id, current))
        return False

    def _is_duplicate_recognized(self, result: dict, tenant_id: str) -> bool:
        employee_id = result.get("employeeId")
        if not employee_id or self.settings.recognized_duplicate_cooldown_seconds <= 0:
            return False

        now = time.monotonic()
        cache_key = (tenant_id, employee_id)
        last_seen_at = self._recognized_face_cache.get(cache_key)
        if last_seen_at is not None and now - last_seen_at < self.settings.recognized_duplicate_cooldown_seconds:
            return True

        self._remember_detection_cache(self._recognized_face_cache, cache_key, now)
        return False

    def _is_duplicate_plate(self, result: dict) -> bool:
        if self.settings.plate_duplicate_cooldown_seconds <= 0:
            return False

        plate_text = (result.get("plateText") or "").strip().upper()
        if plate_text:
            cache_key = f"text:{plate_text}"
        else:
            bbox = result.get("bbox") or [0, 0, 0, 0]
            center_x = int((bbox[0] + bbox[2]) / 20)
            center_y = int((bbox[1] + bbox[3]) / 20)
            cache_key = f"bbox:{center_x}:{center_y}"

        now = time.monotonic()
        last_seen_at = self._plate_event_cache.get(cache_key)
        if last_seen_at is not None and now - last_seen_at < self.settings.plate_duplicate_cooldown_seconds:
            return True

        self._remember_detection_cache(self._plate_event_cache, cache_key, now)
        return False

    def _is_duplicate_fire(self, result: dict) -> bool:
        if self.settings.fire_duplicate_cooldown_seconds <= 0:
            return False

        class_name = (result.get("className") or "fire").strip().lower()
        bbox = result.get("bbox") or [0, 0, 0, 0]
        center_x = int((bbox[0] + bbox[2]) / 40)
        center_y = int((bbox[1] + bbox[3]) / 40)
        cache_key = f"{class_name}:{center_x}:{center_y}"

        now = time.monotonic()
        last_seen_at = self._fire_event_cache.get(cache_key)
        if last_seen_at is not None and now - last_seen_at < self.settings.fire_duplicate_cooldown_seconds:
            return True

        self._remember_detection_cache(self._fire_event_cache, cache_key, now)
        return False

    def _remember_detection_cache(self, cache: dict, key, timestamp: float) -> None:
        cache[key] = timestamp
        while len(cache) > self.settings.detection_event_cache_max_entries:
            cache.pop(next(iter(cache)))

    @staticmethod
    def _public_detections(detections: list[dict]) -> list[dict]:
        return [
            {key: value for key, value in detection.items() if not key.startswith("_")}
            for detection in detections
        ]

    def _snapshot_frame(self, frame, detections: list[dict]):
        if not self.settings.draw_face_boxes_on_snapshots:
            return frame

        snapshot_frame = frame.copy()
        for detection in detections:
            self._draw_face_annotation(
                snapshot_frame,
                detection,
                show_label=self.settings.draw_face_labels_on_snapshots,
            )
        return snapshot_frame

    def _plate_snapshot_frame(self, frame, detections: list[dict]):
        snapshot_frame = frame.copy()
        for detection in detections:
            self._draw_plate_annotation(snapshot_frame, detection, show_label=True)
        return snapshot_frame

    def _fire_snapshot_frame(self, frame, detections: list[dict]):
        snapshot_frame = frame.copy()
        for detection in detections:
            self._draw_fire_annotation(snapshot_frame, detection, show_label=True)
        return snapshot_frame

    @staticmethod
    def _draw_face_annotation(frame, detection: dict, show_label: bool = True) -> None:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return

        x1, y1, x2, y2 = [int(value) for value in bbox]
        matched = detection.get("matched", False)
        color = (0, 255, 0) if matched else (0, 220, 255)
        label = "KNOWN" if matched else "UNKNOWN"
        confidence = detection.get("confidence")
        employee_name = detection.get("employeeName")
        employee_id = detection.get("employeeId")

        if matched and (employee_name or employee_id):
            label = employee_name or employee_id
        if confidence is not None:
            label = f"{label} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not show_label:
            return

        label_y = max(y1 - 10, 20)
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_plate_annotation(frame, detection: dict, show_label: bool = True) -> None:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return

        x1, y1, x2, y2 = [int(value) for value in bbox]
        color = (255, 180, 0)
        plate_text = detection.get("plateText")
        confidence = detection.get("confidence")
        label = plate_text or detection.get("className") or "PLATE"
        if confidence is not None:
            label = f"{label} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not show_label:
            return

        label_y = max(y1 - 10, 20)
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_fire_annotation(frame, detection: dict, show_label: bool = True) -> None:
        bbox = detection.get("bbox")
        if not bbox or len(bbox) != 4:
            return

        x1, y1, x2, y2 = [int(value) for value in bbox]
        color = (0, 80, 255)
        confidence = detection.get("confidence")
        label = detection.get("className") or "FIRE"
        if confidence is not None:
            label = f"{label} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if not show_label:
            return

        label_y = max(y1 - 10, 20)
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

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
    def _parse_motion_zones(raw_zones: str) -> list[dict]:
        zones = []
        for raw_zone in [item.strip() for item in raw_zones.split(";") if item.strip()]:
            if ":" not in raw_zone:
                logger.warning("Invalid MOTION_ZONES entry: %s", raw_zone)
                continue

            zone_id, raw_points = raw_zone.split(":", 1)
            points = []
            for raw_point in [item.strip() for item in raw_points.split("|") if item.strip()]:
                try:
                    x, y = raw_point.split(",", 1)
                    points.append([int(x), int(y)])
                except ValueError:
                    logger.warning("Invalid point '%s' in MOTION_ZONES entry: %s", raw_point, raw_zone)
                    points = []
                    break

            if len(points) < 3:
                logger.warning("Motion zone '%s' ignored because it needs at least 3 points", zone_id)
                continue

            zones.append({"id": zone_id.strip(), "points": np.array(points, dtype=np.int32)})
        return zones

    @staticmethod
    def _metadata(result: dict, zone: ZoneConfig | None) -> dict:
        metadata = {
            "bbox": result["bbox"],
            "detectionScore": result["detectionScore"],
            "employeeName": result.get("employeeName"),
        }
        if result.get("promotedFromWeakCandidate"):
            metadata["promotedFromWeakCandidate"] = True
            metadata["rawCandidateScore"] = result.get("rawCandidateScore")
            metadata["bestCandidateScore"] = result.get("bestCandidateScore")
        if zone:
            metadata["zoneId"] = zone.zoneId
            metadata["zoneName"] = zone.name
        return metadata

    @staticmethod
    def _plate_metadata(result: dict, zone: ZoneConfig | None) -> dict:
        metadata = {
            "bbox": result["bbox"],
            "detectionScore": result.get("detectionScore"),
            "confidence": result.get("confidence"),
            "plateText": result.get("plateText"),
            "className": result.get("className"),
            "classId": result.get("classId"),
            "characters": result.get("characters"),
            "source": result.get("source"),
        }
        if zone:
            metadata["zoneId"] = zone.zoneId
            metadata["zoneName"] = zone.name
        return metadata

    @staticmethod
    def _fire_event_type(result: dict) -> str:
        class_name = str(result.get("className") or "").lower()
        if "smoke" in class_name:
            return "SMOKE_DETECTED"
        return "FIRE_DETECTED"

    @staticmethod
    def _fire_metadata(result: dict, zone: ZoneConfig | None) -> dict:
        metadata = {
            "bbox": result["bbox"],
            "detectionScore": result.get("detectionScore"),
            "confidence": result.get("confidence"),
            "className": result.get("className"),
            "classId": result.get("classId"),
            "source": result.get("source"),
        }
        if zone:
            metadata["zoneId"] = zone.zoneId
            metadata["zoneName"] = zone.name
        return metadata
