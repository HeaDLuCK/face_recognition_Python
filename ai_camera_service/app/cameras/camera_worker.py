import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path
from datetime import datetime, timedelta

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
        self._jpeg_task: asyncio.Task | None = None
        self._recognition_worker_task: asyncio.Task | None = None
        self._recognition_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=settings.recognition_queue_size)
        self._cloud_stream_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.latest_jpeg: bytes | None = None
        self.latest_overlay_jpeg: bytes | None = None
        self.latest_detections: list[dict] = []
        self._fps_started_at = time.perf_counter()
        self._fps_frames = 0
        self._display_fps = 0.0
        event_buffer_size = max(settings.event_buffer_seconds * settings.stream_fps, 1) if settings.event_buffer_enabled else 0
        self._event_buffer = deque(maxlen=event_buffer_size)
        self._event_clip_tasks: set[asyncio.Task] = set()
        self._last_event_clip_at: dict[str, float] = {}
        self._recognized_face_cache: dict[str, float] = {}
        self._unknown_face_cache: deque[tuple[float, np.ndarray]] = deque()
        self._unknown_face_crop_cache: list[tuple[np.ndarray, str]] = []
        self._motion_zones = self._parse_motion_zones(settings.motion_zones)
        self._previous_motion_gray: np.ndarray | None = None
        self.overlay_requested = False

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
            self._start_cloud_stream_push()
            self._start_recognition_worker()

            while not self._stop_event.is_set():
                frame = await asyncio.to_thread(self.reader.read)
                self._update_fps()
                self._add_to_event_buffer(frame)
                self._schedule_latest_jpeg(frame)
                frame_count += 1
                if frame_count % self.settings.motion_check_frame_skip == 0:
                    self._check_motion_zones(frame)

                if frame_count % self.settings.camera_frame_skip != 0:
                    await asyncio.sleep(0)
                    continue

                if is_enabled(self.camera.capabilities, AiCapability.FACE_RECOGNITION):
                    self._queue_face_recognition(frame)

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
            if self._recognition_worker_task and not self._recognition_worker_task.done():
                self._recognition_worker_task.cancel()
            if self._jpeg_task and not self._jpeg_task.done():
                self._jpeg_task.cancel()
            if self._cloud_stream_task and not self._cloud_stream_task.done():
                self._cloud_stream_task.cancel()
            for task in self._event_clip_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.to_thread(self.reader.close)
            if self._stop_event.is_set():
                await self._emit_camera_event("CAMERA_STOPPED", {"stoppedAt": datetime.utcnow().isoformat()})

    def _start_recognition_worker(self) -> None:
        if self._recognition_worker_task and not self._recognition_worker_task.done():
            return
        self._recognition_worker_task = asyncio.create_task(
            self._run_recognition_worker(),
            name=f"face-recognition-worker-{self.camera_id}",
        )

    def _queue_face_recognition(self, frame) -> None:
        frame_copy = frame.copy()
        if self._recognition_queue.full():
            try:
                self._recognition_queue.get_nowait()
                self._recognition_queue.task_done()
            except asyncio.QueueEmpty:
                pass
        try:
            self._recognition_queue.put_nowait(frame_copy)
        except asyncio.QueueFull:
            pass

    def _schedule_latest_jpeg(self, frame) -> None:
        if self._jpeg_task and not self._jpeg_task.done():
            return
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

    async def _run_recognition_worker(self) -> None:
        while not self._stop_event.is_set():
            frame = await self._recognition_queue.get()
            try:
                await self._run_face_recognition(frame)
            finally:
                self._recognition_queue.task_done()

    async def _run_face_recognition(self, frame) -> None:
        results = await self.recognition_service.recognize_frame(
            tenant_id=self.camera.tenantId,
            frame=frame,
            threshold=self.rules.recognitionThreshold,
        )
        self.latest_detections = self._public_detections(results)
        handled_results = []
        for result in results:
            zone = self._matching_zone(result["bbox"])
            if self.camera.zones and zone is None:
                continue

            if result["matched"]:
                if self._is_duplicate_recognized(result):
                    continue
                handled_results.append(("recognized", result, zone))
            elif self.rules.saveUnknownFaces:
                if not self._is_good_unknown_candidate(frame, result):
                    continue
                if self._is_duplicate_unknown(result):
                    continue
                self._schedule_event_marker("FACE_DETECTION", result)
                handled_results.append(("unknown", result, zone))
            else:
                if not self._is_good_unknown_candidate(frame, result):
                    continue
                if self._is_duplicate_unknown(result):
                    continue
                self._schedule_event_marker("FACE_DETECTION", result)
                await self._record_unknown_detection(result, zone)

        if not handled_results:
            return

        snapshot_path = await self._save_frame_snapshot(frame, results) if self.rules.saveFaceSnapshots else None
        for kind, result, zone in handled_results:
            if kind == "recognized":
                await self._handle_recognized(result, zone, snapshot_path)
            else:
                face_crop_path = (
                    await self._save_unknown_face_crop(frame, result)
                    if self.rules.saveUnknownFaceCrops
                    else None
                )
                await self._handle_unknown(result, zone, snapshot_path, face_crop_path)

    async def _handle_recognized(
        self,
        result: dict,
        zone: ZoneConfig | None,
        snapshot_path: str | None,
    ) -> None:
        metadata = self._metadata(result, zone)

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
                                "etsAuth": self.camera.tenantId,
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
        zone: ZoneConfig | None,
        snapshot_path: str | None,
        face_crop_path: str | None = None,
    ) -> None:
        metadata = self._metadata(result, zone)
        if face_crop_path:
            metadata["faceCropPath"] = face_crop_path
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
        self._draw_detections(overlay_frame)
        ok, buffer = cv2.imencode(
            ".jpg",
            overlay_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.stream_jpeg_quality],
        )
        overlay_jpeg = buffer.tobytes() if ok else None
        return clean_jpeg, overlay_jpeg

    def _add_to_event_buffer(self, frame) -> None:
        if not self.settings.event_buffer_enabled:
            return
        now = datetime.utcnow()
        self._event_buffer.append((now, frame.copy()))

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

    def _schedule_event_marker(self, event_type: str, metadata: dict | None = None) -> None:
        now = time.monotonic()
        last_event_at = self._last_event_clip_at.get(event_type, 0)
        if now - last_event_at < self.settings.event_clip_cooldown_seconds:
            return

        self._last_event_clip_at[event_type] = now
        task = asyncio.create_task(
            self._save_event_marker(event_type, metadata or {}),
            name=f"event-marker-{self.camera_id}-{event_type}",
        )
        self._event_clip_tasks.add(task)
        task.add_done_callback(self._event_clip_tasks.discard)

    async def _save_event_marker(self, event_type: str, metadata: dict) -> None:
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
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=self.camera.tenantId,
                cameraId=self.camera.cameraId,
                eventType=event_type,
                timestamp=timestamp,
                metadata=marker_metadata,
            ),
            send_to_erp=True,
        )

    def _camera_channel(self) -> str | None:
        if "/Streaming/Channels/" in self.camera.rtspUrl:
            return self.camera.rtspUrl.rsplit("/", 1)[-1]
        if "/Streaming/tracks/" in self.camera.rtspUrl:
            return self.camera.rtspUrl.rsplit("/", 1)[-1]
        return None

    async def _save_event_clip(self, reason: str, buffered_frames: list[tuple[datetime, np.ndarray]], metadata: dict) -> None:
        clip_path = await asyncio.to_thread(self._write_event_clip, reason, buffered_frames)
        await self.event_service.create_camera_event(
            RuntimeEvent(
                tenantId=self.camera.tenantId,
                cameraId=self.camera.cameraId,
                eventType="EVENT_CLIP_SAVED",
                snapshotPath=str(clip_path),
                timestamp=datetime.utcnow(),
                metadata={"reason": reason, **metadata},
            ),
            send_to_erp=bool(self.settings.erp_base_url),
        )

    def _write_event_clip(self, reason: str, buffered_frames: list[tuple[datetime, np.ndarray]]) -> Path:
        tenant_dir = self.settings.event_clip_dir / self.camera.tenantId / self.camera.cameraId
        tenant_dir.mkdir(parents=True, exist_ok=True)
        clean_reason = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in reason)
        filename = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')}_{clean_reason}.mp4"
        clip_path = tenant_dir / filename

        sampled_frames = self._sample_clip_frames(buffered_frames)
        first_frame = sampled_frames[0][1]
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.settings.event_clip_fps,
            (width, height),
        )
        try:
            for _, frame in sampled_frames:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

        logger.info("Saved event clip for camera %s: %s", self.camera_id, clip_path)
        return clip_path

    def _sample_clip_frames(self, buffered_frames: list[tuple[datetime, np.ndarray]]) -> list[tuple[datetime, np.ndarray]]:
        if len(buffered_frames) <= 1:
            return buffered_frames

        min_delta = 1 / self.settings.event_clip_fps
        sampled = []
        last_timestamp = None
        for timestamp, frame in buffered_frames:
            current_timestamp = timestamp.timestamp()
            if last_timestamp is None or current_timestamp - last_timestamp >= min_delta:
                sampled.append((timestamp, frame))
                last_timestamp = current_timestamp

        return sampled or [buffered_frames[-1]]

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

    async def _save_frame_snapshot(self, frame, detections: list[dict]) -> str:
        snapshot_frame = self._snapshot_frame(frame, detections)
        snapshot_path = await asyncio.to_thread(
            self.snapshot_service.save_frame,
            self.camera.tenantId,
            self.camera.cameraId,
            snapshot_frame,
        )
        await self.snapshot_service.save_metadata(
            self.camera.tenantId,
            self.camera.cameraId,
            snapshot_path,
            "FACE_DETECTIONS",
            {"detections": [self._metadata(detection, self._matching_zone(detection["bbox"])) for detection in detections]},
        )
        return snapshot_path

    async def _save_unknown_face_crop(self, frame, result: dict) -> str | None:
        existing_path = self._matched_unknown_face_crop_path(result)
        if existing_path:
            return existing_path

        existing_path = await self._find_stored_unknown_face_crop_path(result)
        if existing_path:
            self._remember_unknown_face_crop(result, existing_path)
            return existing_path

        crop = self._face_crop(frame, result.get("bbox"))
        if crop is None:
            return None
        crop_path = await asyncio.to_thread(
            self.snapshot_service.save_face_crop,
            self.camera.tenantId,
            self.camera.cameraId,
            crop,
        )
        self._remember_unknown_face_crop(result, crop_path)
        await self._register_stored_unknown_face_crop(result, crop_path)
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

    def _matched_unknown_face_crop_path(self, result: dict) -> str | None:
        embedding = result.get("_embedding")
        if embedding is None:
            return None

        current = np.array(embedding, dtype=np.float32)
        for cached, crop_path in self._unknown_face_crop_cache:
            score = float(np.dot(current, cached))
            if score >= self.settings.unknown_face_crop_similarity_threshold:
                return crop_path
        return None

    def _remember_unknown_face_crop(self, result: dict, crop_path: str) -> None:
        embedding = result.get("_embedding")
        if embedding is None:
            return
        self._unknown_face_crop_cache.append((np.array(embedding, dtype=np.float32), crop_path))

    async def _find_stored_unknown_face_crop_path(self, result: dict) -> str | None:
        embedding = result.get("_embedding")
        if embedding is None:
            return None
        return await self.snapshot_service.find_unknown_face_crop(
            self.camera.tenantId,
            self.camera.cameraId,
            embedding,
            self.settings.unknown_face_crop_similarity_threshold,
        )

    async def _register_stored_unknown_face_crop(self, result: dict, crop_path: str) -> None:
        embedding = result.get("_embedding")
        if embedding is None:
            return
        await self.snapshot_service.register_unknown_face_crop(
            self.camera.tenantId,
            self.camera.cameraId,
            crop_path,
            embedding,
            metadata={"bbox": result.get("bbox"), "detectionScore": result.get("detectionScore")},
        )

    def _is_good_unknown_candidate(self, frame, result: dict) -> bool:
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
            and best_candidate_score >= self.rules.recognitionThreshold - self.settings.unknown_face_skip_weak_known_margin
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

    def _is_duplicate_unknown(self, result: dict) -> bool:
        embedding = result.get("_embedding")
        if embedding is None or self.settings.unknown_duplicate_cooldown_seconds <= 0:
            return False

        now = time.monotonic()
        cooldown = self.settings.unknown_duplicate_cooldown_seconds
        while self._unknown_face_cache and now - self._unknown_face_cache[0][0] > cooldown:
            self._unknown_face_cache.popleft()

        current = np.array(embedding, dtype=np.float32)
        for _, cached in self._unknown_face_cache:
            score = float(np.dot(current, cached))
            if score >= self.settings.unknown_duplicate_similarity_threshold:
                return True

        self._unknown_face_cache.append((now, current))
        return False

    def _is_duplicate_recognized(self, result: dict) -> bool:
        employee_id = result.get("employeeId")
        if not employee_id or self.settings.recognized_duplicate_cooldown_seconds <= 0:
            return False

        now = time.monotonic()
        last_seen_at = self._recognized_face_cache.get(employee_id)
        if last_seen_at is not None and now - last_seen_at < self.settings.recognized_duplicate_cooldown_seconds:
            return True

        self._recognized_face_cache[employee_id] = now
        return False

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
        if zone:
            metadata["zoneId"] = zone.zoneId
            metadata["zoneName"] = zone.name
        return metadata
