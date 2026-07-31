import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import cv2
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from app.attendance.attendance_service import AttendanceService
from app.cameras.hikvision_history import (
    hikvision_playback_url,
    parse_hikvision_rtsp,
)
from app.config import Settings
from app.events.event_service import EventService
from app.face.recognition_scheduler import FaceRecognitionScheduler
from app.face.recognition_service import RecognitionService
from app.runtime_state import RuntimeState
from app.schemas.erp_schema import AiCapability, CameraAssignment, CameraConfig
from app.schemas.runtime_schema import RuntimeEvent

logger = logging.getLogger(__name__)


class LiveAiBusyError(RuntimeError):
    pass


class AttendanceRecoveryService:
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        runtime_state: RuntimeState,
        recognition_service: RecognitionService,
        recognition_scheduler: FaceRecognitionScheduler,
        attendance_service: AttendanceService,
        event_service: EventService,
        settings: Settings,
    ):
        self.db = db
        self.runtime_state = runtime_state
        self.recognition_service = recognition_service
        self.recognition_scheduler = recognition_scheduler
        self.attendance_service = attendance_service
        self.event_service = event_service
        self.settings = settings
        self._stop_event = asyncio.Event()
        self._processed_jobs = 0
        self._recovered_people = 0
        self._last_error: str | None = None

    async def enqueue_track(
        self,
        camera: CameraConfig,
        assignment: CameraAssignment,
        track_id: int,
        first_seen_at: datetime,
        last_seen_at: datetime,
    ) -> str | None:
        if not self.settings.history_recovery_enabled:
            return None

        rules = self.runtime_state.get_rules(assignment.tenantId)
        if not rules.historyRecoveryEnabled:
            return None

        window_start = first_seen_at - timedelta(
            seconds=self.settings.history_recovery_before_seconds
        )
        window_end = last_seen_at + timedelta(
            seconds=self.settings.history_recovery_after_seconds
        )
        now = datetime.utcnow()
        next_attempt_at = now + timedelta(
            seconds=self.settings.history_recovery_initial_delay_seconds
        )
        merge_margin = timedelta(
            seconds=self.settings.history_recovery_gap_merge_seconds
        )
        existing = await self.db.attendance_recovery_jobs.find_one(
            {
                "cameraId": camera.cameraId,
                "status": "PENDING",
                "windowStart": {"$lte": window_end + merge_margin},
                "windowEnd": {"$gte": window_start - merge_margin},
            },
            {"recoveryJobId": 1},
        )
        if existing:
            await self.db.attendance_recovery_jobs.update_one(
                {"_id": existing["_id"]},
                {
                    "$min": {"windowStart": window_start},
                    "$max": {
                        "windowEnd": window_end,
                        "nextAttemptAt": next_attempt_at,
                    },
                    "$addToSet": {
                        "trackIds": track_id,
                        "tenantIds": assignment.tenantId,
                        "triggerTypes": "PERSON_TRACK",
                    },
                    "$set": {"updatedAt": now},
                },
            )
            return existing["recoveryJobId"]

        recovery_job_id = str(uuid4())
        await self.db.attendance_recovery_jobs.insert_one(
            {
                "recoveryJobId": recovery_job_id,
                "etsAuth": assignment.tenantId,
                "tenantIds": [assignment.tenantId],
                "cameraId": camera.cameraId,
                "trackIds": [track_id],
                "direction": assignment.direction,
                "windowStart": window_start,
                "windowEnd": window_end,
                "triggerTypes": ["PERSON_TRACK"],
                "status": "PENDING",
                "attempts": 0,
                "nextAttemptAt": next_attempt_at,
                "createdAt": now,
                "updatedAt": now,
            }
        )
        return recovery_job_id

    async def enqueue_stream_gap(
        self,
        camera: CameraConfig,
        gap_start: datetime,
        gap_end: datetime,
    ) -> str | None:
        if not self.settings.history_recovery_enabled:
            return None
        if gap_end <= gap_start:
            return None

        assignments = self._target_assignments(camera)
        if not assignments:
            return None

        tenant_ids = sorted({assignment.tenantId for assignment in assignments})
        window_start = gap_start - timedelta(
            seconds=self.settings.history_recovery_before_seconds
        )
        window_end = gap_end + timedelta(
            seconds=self.settings.history_recovery_after_seconds
        )
        merge_margin = timedelta(
            seconds=self.settings.history_recovery_gap_merge_seconds
        )
        now = datetime.utcnow()
        next_attempt_at = now + timedelta(
            seconds=self.settings.history_recovery_initial_delay_seconds
        )
        gap_metadata = {
            "startedAt": gap_start,
            "recoveredAt": gap_end,
        }

        existing = await self.db.attendance_recovery_jobs.find_one(
            {
                "cameraId": camera.cameraId,
                "status": "PENDING",
                "windowStart": {"$lte": window_end + merge_margin},
                "windowEnd": {"$gte": window_start - merge_margin},
            },
            {"recoveryJobId": 1},
        )
        if existing:
            await self.db.attendance_recovery_jobs.update_one(
                {"_id": existing["_id"]},
                {
                    "$min": {"windowStart": window_start},
                    "$max": {
                        "windowEnd": window_end,
                        "nextAttemptAt": next_attempt_at,
                    },
                    "$addToSet": {
                        "tenantIds": {"$each": tenant_ids},
                        "triggerTypes": "RTSP_GAP",
                    },
                    "$push": {
                        "streamGaps": {
                            "$each": [gap_metadata],
                            "$slice": -100,
                        }
                    },
                    "$set": {
                        "updatedAt": now,
                        "lastError": None,
                    },
                },
            )
            logger.info(
                "Merged RTSP gap into recovery job %s for camera %s",
                existing["recoveryJobId"],
                camera.cameraId,
            )
            return existing["recoveryJobId"]

        recovery_job_id = str(uuid4())
        await self.db.attendance_recovery_jobs.insert_one(
            {
                "recoveryJobId": recovery_job_id,
                "etsAuth": tenant_ids[0],
                "tenantIds": tenant_ids,
                "cameraId": camera.cameraId,
                "trackIds": [],
                "direction": assignments[0].direction,
                "windowStart": window_start,
                "windowEnd": window_end,
                "triggerTypes": ["RTSP_GAP"],
                "streamGaps": [gap_metadata],
                "status": "PENDING",
                "attempts": 0,
                "nextAttemptAt": next_attempt_at,
                "createdAt": now,
                "updatedAt": now,
                "createdBy": "automatic_rtsp_gap",
            }
        )
        logger.info(
            "Queued RTSP gap recovery job %s for camera %s (%s to %s)",
            recovery_job_id,
            camera.cameraId,
            window_start.isoformat(),
            window_end.isoformat(),
        )
        return recovery_job_id

    async def enqueue_window(
        self,
        camera_id: str,
        window_start: datetime,
        window_end: datetime,
        tenant_id: str | None = None,
        track_id: int | None = None,
    ) -> str:
        if window_end <= window_start:
            raise ValueError("windowEnd must be after windowStart")

        camera = self.runtime_state.get_camera(camera_id)
        if camera is None:
            raise ValueError("Camera is not present in synced configuration")

        assignments = self._target_assignments(camera, tenant_id)
        if not assignments:
            raise ValueError("Camera has no active FACE_RECOGNITION assignment for this recovery")
        tenant_ids = [assignment.tenantId for assignment in assignments]

        now = datetime.utcnow()
        existing = await self.db.attendance_recovery_jobs.find_one(
            {
                "cameraId": camera_id,
                "status": "PENDING",
                "windowStart": {"$lte": window_end},
                "windowEnd": {"$gte": window_start},
            },
            {"recoveryJobId": 1},
        )
        if existing:
            update = {
                "$min": {"windowStart": window_start},
                "$max": {"windowEnd": window_end},
                "$addToSet": {"tenantIds": {"$each": tenant_ids}},
                "$set": {
                    "nextAttemptAt": now,
                    "updatedAt": now,
                    "lastError": None,
                },
            }
            if track_id is not None:
                update["$addToSet"]["trackIds"] = track_id
            await self.db.attendance_recovery_jobs.update_one(
                {"_id": existing["_id"]},
                update,
            )
            return existing["recoveryJobId"]

        recovery_job_id = str(uuid4())
        await self.db.attendance_recovery_jobs.insert_one(
            {
                "recoveryJobId": recovery_job_id,
                "etsAuth": tenant_ids[0],
                "tenantIds": tenant_ids,
                "cameraId": camera_id,
                "trackIds": [track_id] if track_id is not None else [],
                "direction": assignments[0].direction,
                "windowStart": window_start,
                "windowEnd": window_end,
                "status": "PENDING",
                "attempts": 0,
                "nextAttemptAt": now,
                "createdAt": now,
                "updatedAt": now,
                "createdBy": "manual",
            }
        )
        return recovery_job_id

    async def run_forever(self) -> None:
        self._stop_event.clear()
        now = datetime.utcnow()
        await self.db.attendance_recovery_jobs.update_many(
            {"status": "PROCESSING"},
            {
                "$set": {
                    "status": "PENDING",
                    "nextAttemptAt": now,
                    "updatedAt": now,
                    "lastError": "Recovery service restarted during processing",
                }
            },
        )
        while not self._stop_event.is_set():
            job = await self._claim_next_job()
            if job is None:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.settings.history_recovery_poll_seconds,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            await self._process_claimed_job(job)

    def stop(self) -> None:
        self._stop_event.set()

    async def list_jobs(
        self,
        tenant_id: str | None = None,
        camera_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        query = {}
        if tenant_id:
            query["$or"] = [
                {"etsAuth": tenant_id},
                {"tenantIds": tenant_id},
            ]
        if camera_id:
            query["cameraId"] = camera_id
        if status:
            query["status"] = status.upper()
        cursor = self.db.attendance_recovery_jobs.find(query).sort("createdAt", -1).limit(limit)
        jobs = await cursor.to_list(length=limit)
        for job in jobs:
            job["_id"] = str(job["_id"])
        return jobs

    async def retry_job(self, recovery_job_id: str) -> bool:
        result = await self.db.attendance_recovery_jobs.update_one(
            {"recoveryJobId": recovery_job_id},
            {
                "$set": {
                    "status": "PENDING",
                    "attempts": 0,
                    "nextAttemptAt": datetime.utcnow(),
                    "updatedAt": datetime.utcnow(),
                    "lastError": None,
                }
            },
        )
        return result.matched_count > 0

    async def delete_duplicate_jobs(self) -> dict:
        duplicate_groups = await self.db.attendance_recovery_jobs.aggregate(
            [
                {"$match": {"status": "PENDING"}},
                {
                    "$group": {
                        "_id": {
                            "cameraId": "$cameraId",
                            "windowStart": "$windowStart",
                            "windowEnd": "$windowEnd",
                        },
                        "jobs": {
                            "$push": {
                                "_id": "$_id",
                                "recoveryJobId": "$recoveryJobId",
                                "createdAt": "$createdAt",
                            }
                        },
                        "count": {"$sum": 1},
                    }
                },
                {"$match": {"count": {"$gt": 1}}},
            ]
        ).to_list(length=None)

        deleted = 0
        for group in duplicate_groups:
            jobs = sorted(group["jobs"], key=lambda item: item.get("createdAt") or datetime.min)
            duplicate_ids = [item["_id"] for item in jobs[1:]]
            if not duplicate_ids:
                continue
            result = await self.db.attendance_recovery_jobs.delete_many({"_id": {"$in": duplicate_ids}})
            deleted += result.deleted_count

        return {
            "duplicateGroups": len(duplicate_groups),
            "deleted": deleted,
        }

    def status(self) -> dict:
        return {
            "enabled": self.settings.history_recovery_enabled,
            "processedJobs": self._processed_jobs,
            "recoveredPeople": self._recovered_people,
            "lastError": self._last_error,
        }

    async def _claim_next_job(self) -> dict | None:
        now = datetime.utcnow()
        return await self.db.attendance_recovery_jobs.find_one_and_update(
            {
                "status": "PENDING",
                "nextAttemptAt": {"$lte": now},
            },
            {
                "$set": {
                    "status": "PROCESSING",
                    "startedAt": now,
                    "updatedAt": now,
                },
                "$inc": {"attempts": 1},
            },
            sort=[("nextAttemptAt", 1), ("createdAt", 1)],
            return_document=ReturnDocument.AFTER,
        )

    async def _process_claimed_job(self, job: dict) -> None:
        try:
            camera = self.runtime_state.get_camera(job["cameraId"])
            if camera is None:
                raise RuntimeError("Camera is no longer present in the synced configuration")
            assignments = self._job_assignments(camera, job)
            if not assignments:
                raise RuntimeError("No active camera assignments are available for this recovery job")

            await self._delete_duplicate_window_jobs(job)

            recognized = await self._scan_history_stream(
                self._job_playback_url(camera, job),
                job,
                assignments,
            )
            await self._record_recovered_people(job, assignments, recognized)
            await self.db.attendance_recovery_jobs.update_one(
                {"_id": job["_id"]},
                {
                    "$set": {
                        "status": "COMPLETED",
                        "result": "RECOGNIZED" if recognized else "NO_MATCH",
                        "recognizedEmployees": [
                            {
                                "etsAuth": item["tenantId"],
                                "employeeId": item["employeeId"],
                            }
                            for item in recognized.values()
                        ],
                        "completedAt": datetime.utcnow(),
                        "updatedAt": datetime.utcnow(),
                        "lastError": None,
                    }
                },
            )
            self._processed_jobs += 1
            self._recovered_people += len(recognized)
        except LiveAiBusyError as exc:
            await self._reschedule(job, str(exc), count_as_failure=False)
        except Exception as exc:
            self._last_error = str(exc)
            logger.exception(
                "Attendance history recovery failed for job %s",
                job.get("recoveryJobId"),
            )
            await self._reschedule(job, str(exc), count_as_failure=True)

    async def _delete_duplicate_window_jobs(self, job: dict) -> None:
        await self.db.attendance_recovery_jobs.delete_many(
            {
                "_id": {"$ne": job["_id"]},
                "cameraId": job["cameraId"],
                "windowStart": job["windowStart"],
                "windowEnd": job["windowEnd"],
                "status": "PENDING",
            },
        )

    def _job_playback_url(self, camera: CameraConfig, job: dict) -> str:
        window_start = job["windowStart"]
        window_end = job["windowEnd"]
        info = parse_hikvision_rtsp(camera.rtspUrl)
        return hikvision_playback_url(
            host=info["host"],
            rtsp_port=info["rtsp_port"],
            username=info["username"],
            password=info["password"],
            channel=info["channel"],
            start=window_start.replace(tzinfo=timezone.utc),
            end=window_end.replace(tzinfo=timezone.utc),
        )

    async def _scan_history_stream(
        self,
        playback_url: str,
        job: dict,
        assignments: list[CameraAssignment],
    ) -> dict[tuple[str, str], dict]:
        capture = cv2.VideoCapture(
            playback_url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                10000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                10000,
            ],
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Unable to open Hikvision history playback stream")

        window_seconds = max((job["windowEnd"] - job["windowStart"]).total_seconds(), 1)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        sample_every_seconds = window_seconds / self.settings.history_recovery_max_frames
        max_runtime_seconds = window_seconds + 30
        assignments_by_tenant = {assignment.tenantId: assignment for assignment in assignments}
        recognized: dict[tuple[str, str], dict] = {}
        frame_index = 0
        sampled_frames = 0
        last_sample_offset = -sample_every_seconds
        started_at = time.monotonic()
        tenant_thresholds = {
            assignment.tenantId: self.runtime_state.get_rules(assignment.tenantId).recognitionThreshold
            for assignment in assignments
        }
        try:
            live_idle = await self.recognition_scheduler.wait_until_idle(
                idle_seconds=self.settings.history_recovery_live_idle_seconds,
                timeout_seconds=self.settings.history_recovery_live_idle_timeout_seconds,
            )
            if not live_idle:
                raise LiveAiBusyError("Live face recognition remained busy")

            while (
                sampled_frames < self.settings.history_recovery_max_frames
                and time.monotonic() - started_at < max_runtime_seconds
            ):
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok or frame is None:
                    break
                current_index = frame_index
                frame_index += 1
                offset_seconds = (
                    current_index / source_fps
                    if source_fps > 0
                    else time.monotonic() - started_at
                )
                if offset_seconds - last_sample_offset < sample_every_seconds:
                    continue
                last_sample_offset = offset_seconds

                results = await self.recognition_service.recognize_frame_for_tenants(
                    tenant_thresholds=tenant_thresholds,
                    frame=frame,
                )
                sampled_frames += 1
                for result in results:
                    tenant_id = result.get("tenantId")
                    employee_id = result.get("employeeId")
                    if not result.get("matched") or not tenant_id or not employee_id:
                        continue
                    assignment = assignments_by_tenant.get(tenant_id)
                    if assignment is None:
                        continue
                    if not self._result_in_assignment_zone(result, assignment):
                        continue
                    key = (tenant_id, employee_id)
                    existing = recognized.get(key)
                    if existing is not None and existing["confidence"] >= result["confidence"]:
                        continue
                    recognized[key] = {
                        **result,
                        "timestamp": job["windowStart"]
                        + timedelta(seconds=offset_seconds),
                    }
        finally:
            capture.release()
        if frame_index == 0:
            raise RuntimeError("No frames were returned by Hikvision history playback stream")
        return recognized

    @staticmethod
    def _result_in_assignment_zone(
        result: dict,
        assignment: CameraAssignment,
    ) -> bool:
        if not assignment.zones:
            return True
        bbox = result.get("bbox") or []
        if len(bbox) != 4:
            return False
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return any(
            zone.x <= center_x <= zone.x + zone.width
            and zone.y <= center_y <= zone.y + zone.height
            for zone in assignment.zones
        )

    async def _record_recovered_people(
        self,
        job: dict,
        assignments: list[CameraAssignment],
        recognized: dict[tuple[str, str], dict],
    ) -> None:
        assignments_by_tenant = {assignment.tenantId: assignment for assignment in assignments}
        for (tenant_id, employee_id), result in recognized.items():
            assignment = assignments_by_tenant.get(tenant_id)
            if assignment is None:
                continue
            rules = self.runtime_state.get_rules(tenant_id)
            event_time = result["timestamp"]
            create_attendance, direction = await self.attendance_service.should_create_attendance(
                tenant_id=tenant_id,
                employee_id=employee_id,
                camera_direction=assignment.direction,
                confidence=result["confidence"],
                rules=rules,
                event_time=event_time,
            )
            if direction and not create_attendance:
                continue

            event_type = (
                f"ATTENDANCE_{direction}"
                if create_attendance and direction
                else "FACE_RECOGNIZED"
            )
            metadata = {
                "source": "HISTORY_RECOVERY",
                "recoveryJobId": job["recoveryJobId"],
                "trackIds": job.get("trackIds", []),
                "employeeName": result.get("employeeName"),
                "detectionScore": result.get("detectionScore"),
                "bestCandidateScore": result.get("bestCandidateScore"),
            }
            await self.attendance_service.record_detection(
                tenant_id=tenant_id,
                camera_id=job["cameraId"],
                event_type=event_type,
                employee_id=employee_id,
                matched=True,
                confidence=result["confidence"],
                snapshot_path=None,
                metadata=metadata,
                timestamp=event_time,
            )
            await self.event_service.create_camera_event(
                RuntimeEvent(
                    tenantId=tenant_id,
                    cameraId=job["cameraId"],
                    eventType=event_type,
                    employeeId=employee_id,
                    confidence=result["confidence"],
                    timestamp=event_time,
                    metadata=metadata,
                )
            )

    def _target_assignments(
        self,
        camera: CameraConfig,
        tenant_id: str | None = None,
    ) -> list[CameraAssignment]:
        assignments = camera.assignments_for(AiCapability.FACE_RECOGNITION)
        if tenant_id:
            assignments = [
                assignment
                for assignment in assignments
                if assignment.tenantId == tenant_id
            ]
        return [
            assignment
            for assignment in assignments
            if self.runtime_state.get_rules(assignment.tenantId).historyRecoveryEnabled
        ]

    def _job_assignments(
        self,
        camera: CameraConfig,
        job: dict,
    ) -> list[CameraAssignment]:
        tenant_ids = job.get("tenantIds")
        if not tenant_ids and job.get("etsAuth"):
            tenant_ids = [job["etsAuth"]]
        tenant_id_set = set(tenant_ids or [])
        assignments = self._target_assignments(camera)
        if tenant_id_set:
            assignments = [
                assignment
                for assignment in assignments
                if assignment.tenantId in tenant_id_set
            ]
        return assignments

    async def _reschedule(
        self,
        job: dict,
        error: str,
        count_as_failure: bool,
    ) -> None:
        attempts = int(job.get("attempts", 0))
        exhausted = count_as_failure and attempts >= self.settings.history_recovery_max_attempts
        if exhausted:
            status = "FAILED"
            next_attempt_at = None
        else:
            status = "PENDING"
            delay_seconds = min(300, max(10, 10 * (2 ** max(attempts - 1, 0))))
            next_attempt_at = datetime.utcnow() + timedelta(seconds=delay_seconds)

        update = {
            "status": status,
            "lastError": error,
            "updatedAt": datetime.utcnow(),
        }
        if next_attempt_at is not None:
            update["nextAttemptAt"] = next_attempt_at
        if exhausted:
            update["completedAt"] = datetime.utcnow()
        await self.db.attendance_recovery_jobs.update_one(
            {"_id": job["_id"]},
            {
                "$set": update,
                **({"$inc": {"attempts": -1}} if not count_as_failure else {}),
            },
        )
