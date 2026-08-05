from datetime import datetime, timedelta, timezone
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from database import serialize_mongo_docs
from schemas.project_schema import AttendanceRules
from typing import Any
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
    ) -> tuple[bool, str | None]:
        direction = self._attendance_direction(
            camera_direction
        )
        if direction is None:
            return False, None

        if (
            confidence is None
            or confidence < rule.recognitionThreshold
        ):  
            return False, direction

        now = event_time or utc_now()

        cooldown = timedelta(
            seconds=rule.duplicateCooldownSeconds
        )

        event_type = self._event_type_filter(direction,None)

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
            return False, direction
        return True, direction


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

        allowed, direction = (
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
            return {
                "created": False,
                "direction": direction,
                "reason": "threshold_or_cooldown",
            }

        event_type = self._event_type_filter(direction,None)
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

        return {
            "created": True,
            "direction": direction,
            "detection": detection,
        }

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
            etsAuth=ets_auth,
            employee_id=employee_id,
            camera_id=camera_id,
            direction=direction,
            event_type=event_type,
            since=since,
        )
        cursor = self.db.attendance_detections.find(query).sort("timestamp", 1).limit(limit)
        data = serialize_mongo_docs(await cursor.to_list(length=limit))
        return self._format_attendance_rows(data)

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
        safe_limit = max(1, min(limit, 1000))

        query = self._attendance_query(
            ets_auth=ets_auth,
            employee_id=employee_id,
            camera_id=camera_id,
            direction=direction,
            event_type=event_type,
            since=since,
        )

        cursor = (
            self.db.attendance_detections
            .find(query)
            .sort(
                [
                    ("timestamp", ASCENDING),
                    ("detectionId", ASCENDING),
                ]
            )
            .limit(safe_limit)
        )

        documents = await cursor.to_list(
            length=safe_limit
        )

        data = serialize_mongo_docs(documents)

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
