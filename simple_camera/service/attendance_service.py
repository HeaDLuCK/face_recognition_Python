from datetime import datetime, timedelta, timezone
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from database import serialize_mongo_docs
from schemas.project_schema import AttendanceRules
from typing import Any
import logging
logger = logging.getLogger(__name__)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AttendanceService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db


    
    async def record_detection(
        self,
        ets_auth: str,
        camera_id: str,
        event_type: str,
        employee_id: str | None,
        matched: bool,
        confidence: float | None,
        snapshot_path: str | None = None,
        timestamp: datetime | None = None,
        unknown_id: str | None = None,
        identity_status: str | None = None,
    ) -> dict:
        event_time = timestamp or utc_now()

        doc = {
            "detectionId": str(uuid4()),
            "etsAuth": ets_auth,
            "cameraId": camera_id,
            "eventType": event_type.strip().upper(),
            "employeeId": employee_id,
            "matched": matched,
            "confidence": confidence,
            "snapshotPath": snapshot_path,
            "timestamp": event_time,
            "createdAt": utc_now(),
        }

        if unknown_id is not None:
            doc["unknownId"] = unknown_id

            doc["identityStatus"] = (
                identity_status
                or "PENDING"
            )

        await self.db.attendance_detections.insert_one(doc)

        return serialize_mongo_docs([doc])[0]

    async def should_create_attendance(
        self,
        ets_auth: str,
        employee_id: str,
        camera_id: str,
        camera_direction: str,
        confidence: float | None,
        rule: AttendanceRules,
        event_time: datetime | None = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        direction = self._attendance_direction(
            camera_direction
        )
        if direction is None:
            return (
                False,
                None,
                {
                    "reason": "invalid_direction",
                    "cameraDirection": camera_direction,
                },
            )
        minimum_confidence = float(rule.recognitionThreshold)
        if confidence is None:
            return (
                False,
                direction,
                {
                    "reason": "missing_confidence",
                    "recognitionThreshold": minimum_confidence,
                },
            )
        if confidence < minimum_confidence:
            return (
                False,
                direction,
                {
                    "reason": "below_threshold",
                    "confidence": confidence,
                    "recognitionThreshold": minimum_confidence,
                },)
            
        now = event_time or utc_now()
        cooldown_seconds = int(
            rule.duplicateCooldownSeconds
        )
        cooldown = timedelta(
            seconds=rule.duplicateCooldownSeconds
        )
        cooldown_start = now - cooldown
        event_type = self._event_type_filter(direction,None)
        cooldown_query = {
            "etsAuth": ets_auth,
            "employeeId": employee_id,
            "cameraId": camera_id,
            "eventType": event_type,
            "timestamp": {
                "$gte": cooldown_start,
                "$lte": now,
            },
        }

        logger.info(
            "Checking attendance cooldown: "
            "etsAuth=%s employee=%s camera=%s "
            "eventType=%s start=%s end=%s "
            "cooldownSeconds=%s",
            ets_auth,
            employee_id,
            camera_id,
            event_type,
            cooldown_start.isoformat(),
            now.isoformat(),
            cooldown_seconds,
        )
        last_log = await self.db.attendance_detections.find_one(
            {
                "etsAuth": ets_auth,
                "employeeId": employee_id,
                "cameraId":camera_id,
                "eventType": event_type,
                "timestamp": {
                    "$gte": now - cooldown,
                    "$lte": now,
                },
            },
            projection={"_id": 1},
            sort=[("timestamp", DESCENDING)],
        )
        
        if last_log is not None:
            return (
                False,
                direction,
                {
                    "reason": "duplicate_cooldown",
                    "cooldownSeconds": cooldown_seconds,
                    "previousDetectionId": (
                        last_log.get("detectionId")
                        or str(last_log.get("_id"))
                    ),
                    "previousEmployeeId": last_log.get(
                        "employeeId"
                    ),
                    "previousCameraId": last_log.get(
                        "cameraId"
                    ),
                    "previousEventType": last_log.get(
                        "eventType"
                    ),
                    "previousTimestamp": last_log.get(
                        "timestamp"
                    ),
                },
            )

        return (
            True,
            direction,
            {
                "reason": "allowed",
                "confidence": confidence,
                "recognitionThreshold": minimum_confidence,
                "eventType": event_type,
            },
        )


    async def record_attendance_if_allowed(
        self,
        ets_auth: str,
        camera_id: str,
        camera_direction: str,
        employee_id: str,
        confidence: float,
        rule: AttendanceRules,
        snapshot_path: str | None = None,
        event_time: datetime | None = None,
    ) -> dict:
        now = event_time or utc_now()

        # Change these names if your rule model uses different fields.
        minimum_score = float(rule.recognitionThreshold)
        cooldown_seconds = int(rule.duplicateCooldownSeconds)

        logger.info(
            "Attendance evaluation started: "
            "etsAuth=%s employee=%s camera=%s direction=%s "
            "score=%.4f minimum_score=%.4f cooldown_seconds=%s",
            ets_auth,
            employee_id,
            camera_id,
            camera_direction,
            confidence,
            minimum_score,
            cooldown_seconds,
        )
        allowed, direction, decision = (
            await self.should_create_attendance(
                ets_auth=ets_auth,
                employee_id=employee_id,
                camera_id=camera_id,
                camera_direction=camera_direction,
                confidence=confidence,
                rule=rule,
                event_time=now,
            )
        )
        if not allowed or direction is None:
            reason = decision.get(
            "reason",
            "unknown_rejection",
            )

            logger.warning(
                "Attendance rejected: "
                "reason=%s etsAuth=%s employee=%s "
                "camera=%s cameraDirection=%s "
                "resolvedDirection=%s confidence=%.4f "
                "details=%s",
                reason,
                ets_auth,
                employee_id,
                camera_id,
                camera_direction,
                direction,
                confidence,
                decision,
            )
            return {
                "created": False,
                "direction": direction,
                "reason": reason,
                "details": decision,
            }

        event_type = self._event_type_filter(direction,None)
        logger.info(
            "Attendance accepted for insertion: "
            "etsAuth=%s employee=%s camera=%s "
            "direction=%s eventType=%s confidence=%.4f",
            ets_auth,
            employee_id,
            camera_id,
            direction,
            event_type,
            confidence,
        )
        try:
            detection = await self.record_detection(
                ets_auth=ets_auth,
                camera_id=camera_id,
                event_type=event_type,
                employee_id=employee_id,
                matched=True,
                confidence=confidence,
                snapshot_path=snapshot_path,
                timestamp=now,
            )
        except Exception:
            logger.exception(
                "MongoDB attendance insertion failed: "
                "etsAuth=%s employee=%s camera=%s "
                "direction=%s eventType=%s",
                ets_auth,
                employee_id,
                camera_id,
                direction,
                event_type,
            )
            raise
        logger.info(
            "Attendance created: "
            "etsAuth=%s employee=%s camera=%s "
            "direction=%s detectionId=%s",
            ets_auth,
            employee_id,
            camera_id,
            direction,
            detection.get("detectionId"),
            )

        return {
            "created": True,
            "direction": direction,
            "reason": "created",
            "detection": detection,
        }


    async def should_create_unknown_attendance(
        self,
        *,
        ets_auth: str,
        unknown_id: str,
        camera_id: str,
        camera_direction: str,
        rule: AttendanceRules,
        event_time: datetime | None = None,
    ) -> tuple[
        bool,
        str | None,
        dict[str, Any],
    ]:
        direction = self._attendance_direction(
            camera_direction
        )

        if direction is None:
            return (
                False,
                None,
                {
                    "reason": "invalid_direction",
                    "cameraDirection": (
                        camera_direction
                    ),
                },
            )

        now = event_time or utc_now()

        cooldown_seconds = int(
            rule.duplicateCooldownSeconds
        )

        cooldown = timedelta(
            seconds=cooldown_seconds
        )

        cooldown_start = (
            now - cooldown
        )

        if direction == "IN":
            event_type = "ATTENDANCE_IN"

        elif direction == "OUT":
            event_type = "ATTENDANCE_OUT"

        else:
            # BIDIRECTIONAL/BOTH cannot tell IN vs OUT
            # without another direction mechanism.
            event_type = "UNKNOWN_FACE_DETECTED"

        logger.info(
            "Checking unknown attendance cooldown: "
            "unknownId=%s camera=%s "
            "eventType=%s start=%s end=%s "
            "cooldownSeconds=%s",
            unknown_id,
            camera_id,
            event_type,
            cooldown_start.isoformat(),
            now.isoformat(),
            cooldown_seconds,
        )

        last_log = await (
            self.db.attendance_detections
            .find_one(
                {
                    "etsAuth": ets_auth,

                    "unknownId": unknown_id,

                    "cameraId": camera_id,

                    "eventType": event_type,

                    "timestamp": {
                        "$gte": cooldown_start,
                        "$lte": now,
                    },
                },
                projection={
                    "_id": 1,
                    "detectionId": 1,
                    "unknownId": 1,
                    "cameraId": 1,
                    "eventType": 1,
                    "timestamp": 1,
                },
                sort=[
                    (
                        "timestamp",
                        DESCENDING,
                    )
                ],
            )
        )

        if last_log is not None:
            return (
                False,
                direction,
                {
                    "reason": (
                        "duplicate_cooldown"
                    ),

                    "cooldownSeconds": (
                        cooldown_seconds
                    ),

                    "previousDetectionId": (
                        last_log.get(
                            "detectionId"
                        )
                        or str(
                            last_log.get(
                                "_id"
                            )
                        )
                    ),

                    "previousUnknownId": (
                        last_log.get(
                            "unknownId"
                        )
                    ),

                    "previousCameraId": (
                        last_log.get(
                            "cameraId"
                        )
                    ),

                    "previousEventType": (
                        last_log.get(
                            "eventType"
                        )
                    ),

                    "previousTimestamp": (
                        last_log.get(
                            "timestamp"
                        )
                    ),
                },
            )

        return (
            True,
            direction,
            {
                "reason": "allowed",
                "eventType": event_type,
            },
        )

    async def record_unknown_attendance_if_allowed(
        self,
        *,
        ets_auth: str,
        unknown_id: str,
        camera_id: str,
        camera_direction: str,
        rule: AttendanceRules,
        event_time: datetime | None = None,
    ) -> dict:
        now = event_time or utc_now()

        logger.info(
            "Unknown attendance evaluation started: "
            "unknownId=%s etsAuth=%s "
            "camera=%s direction=%s",
            unknown_id,
            ets_auth,
            camera_id,
            camera_direction,
        )

        allowed, direction, decision = (
            await self.should_create_unknown_attendance(
                ets_auth=ets_auth,
                unknown_id=unknown_id,
                camera_id=camera_id,
                camera_direction=camera_direction,
                rule=rule,
                event_time=now,
            )
        )

        if (
            not allowed
            or direction is None
        ):
            reason = decision.get(
                "reason",
                "unknown_rejection",
            )

            logger.info(
                "Unknown attendance skipped: "
                "unknownId=%s camera=%s "
                "reason=%s",
                unknown_id,
                camera_id,
                reason,
            )

            return {
                "created": False,
                "direction": direction,
                "reason": reason,
                "details": decision,
            }

        event_type = decision[
            "eventType"
        ]

        detection = await self.record_detection(
            ets_auth=ets_auth,
            camera_id=camera_id,
            event_type=event_type,

            # We don't know employee yet.
            employee_id=None,

            # Not matched to employee yet.
            matched=False,

            confidence=None,

            timestamp=now,

            unknown_id=unknown_id,
            identity_status="PENDING",
        )

        logger.info(
            "Unknown attendance created: "
            "unknownId=%s etsAuth=%s "
            "camera=%s direction=%s "
            "eventType=%s detectionId=%s",
            unknown_id,
            ets_auth,
            camera_id,
            direction,
            event_type,
            detection.get(
                "detectionId"
            ),
        )

        return {
            "created": True,
            "direction": direction,
            "reason": "created",
            "detection": detection,
        }

    async def resolve_unknown_attendance(
        self,
        *,
        unknown_id: str,
        employee_id: str,
    ) -> int:
        now = utc_now()

        result = await (
            self.db.attendance_detections
            .update_many(
                {
                    "unknownId": unknown_id,
                    "identityStatus": "PENDING",
                },
                {
                    "$set": {
                        "employeeId": (
                            employee_id
                        ),

                        "identityStatus": (
                            "RESOLVED"
                        ),

                        "resolvedAt": now,
                    }
                },
            )
        )

        logger.info(
            "Unknown attendance resolved: "
            "unknownId=%s employee=%s "
            "count=%d",
            unknown_id,
            employee_id,
            result.modified_count,
        )

        return result.modified_count
    
    async def list_attendance(
        self,
        ets_auth: str,
        limit: int = 100,
        employee_id: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> str:
        query = self._attendance_query(
            ets_auth=ets_auth,
            employee_id=employee_id,
            camera_id=camera_id,
            direction=direction,
            event_type=event_type,
            since=since,
        )
        cursor = self.db.attendance_detections.find(query).sort("timestamp", 1).limit(limit)
        data = serialize_mongo_docs(await cursor.to_list(length=limit))
        return self._format_attendance_rows(data)



    @staticmethod
    def _attendance_direction(camera_direction: str) -> str | None:
        normalized = camera_direction.upper()
        if normalized in {"IN", "OUT","BIDIRECTIONAL", "BOTH"}:
            return normalized
        return None

    @staticmethod
    def _event_type_filter(direction: str | None, event_type: str | None) -> str | None:
        if event_type:
            return event_type.strip().upper()
        if not direction:
            return None

        normalized = direction.strip().upper()
        if normalized in {"IN", "OUT"}:
            return f"ATTENDANCE_{normalized}"
        if normalized in {"BIDIRECTIONAL", "BOTH"}:
            return "FACE_RECOGNIZED"
        return normalized

    def _attendance_query(
        self,
        ets_auth: str,
        employee_id: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> dict:
        query = {"etsAuth": ets_auth}
        normalized_event_type = self._event_type_filter(direction, event_type)
        if normalized_event_type:
            query["eventType"] = normalized_event_type
        else:
            query["eventType"] = {"$in": ["ATTENDANCE_IN", "ATTENDANCE_OUT", "FACE_RECOGNIZED"]}

        if employee_id:
            query["employeeId"] = employee_id
        if camera_id:
            query["cameraId"] = camera_id
        if since:
            query["timestamp"] = {"$gt": since}
        return query

    @staticmethod
    def _format_attendance_rows(data: list[dict]) -> str:
        return "".join(
            f"{(obj.get('metadata') or {}).get('employeeName') or ''};"
            f"{obj.get('employeeId') or ''};{obj.get('timestamp') or ''}|"
            for obj in data
        )

    @staticmethod
    def _sync_key(
        employee_id: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
    ) -> str:
        return "|".join(
            [
                f"employee={employee_id or '*'}",
                f"camera={camera_id or '*'}",
                f"direction={(direction or '*').upper()}",
                f"eventType={(event_type or '*').upper()}",
            ]
        )
