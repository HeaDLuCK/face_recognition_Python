from datetime import datetime, timedelta
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import serialize_mongo_docs
from app.schemas.erp_schema import AttendanceRules


class AttendanceService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def record_detection(
        self,
        tenant_id: str,
        camera_id: str,
        event_type: str,
        employee_id: str | None,
        matched: bool,
        confidence: float | None,
        snapshot_path: str | None,
        metadata: dict,
    ) -> dict:
        doc = {
            "detectionId": str(uuid4()),
            "etsAuth": tenant_id,
            "cameraId": camera_id,
            "eventType": event_type,
            "employeeId": employee_id,
            "matched": matched,
            "confidence": confidence,
            "snapshotPath": snapshot_path,
            "timestamp": datetime.utcnow(),
            "metadata": metadata,
        }
        await self.db.attendance_detections.insert_one(doc)
        return doc

    async def should_create_attendance(
        self,
        tenant_id: str,
        employee_id: str,
        camera_direction: str,
        confidence: float | None,
        rules: AttendanceRules,
    ) -> tuple[bool, str | None]:
        direction = self._attendance_direction(camera_direction)
        if direction is None:
            return False, None
        if confidence is None or confidence < rules.recognitionThreshold:
            return False, direction

        now = datetime.utcnow()
        cooldown = timedelta(seconds=rules.duplicateCooldownSeconds)
        last_log = await self.db.attendance_detections.find_one(
            {
                "etsAuth": tenant_id,
                "employeeId": employee_id,
                "eventType": f"ATTENDANCE_{direction}",
                "timestamp": {"$gte": now - cooldown},
            },
            {"_id": 1},
            sort=[("timestamp", -1)],
        )
        if last_log:
            return False, direction
        return True, direction

    async def list_attendance(
        self,
        tenant_id: str,
        limit: int = 100,
        employee_id: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> str:
        query = self._attendance_query(
            tenant_id=tenant_id,
            employee_id=employee_id,
            camera_id=camera_id,
            direction=direction,
            event_type=event_type,
            since=since,
        )
        cursor = self.db.attendance_detections.find(query).sort("timestamp", 1).limit(limit)
        data = serialize_mongo_docs(await cursor.to_list(length=limit))
        return self._format_attendance_rows(data)

    async def sync_attendance(
        self,
        tenant_id: str,
        limit: int = 500,
        employee_id: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
        reset: bool = False,
        since: datetime | None = None,
    ) -> str:
        sync_key = self._sync_key(
            employee_id=employee_id,
            camera_id=camera_id,
            direction=direction,
            event_type=event_type,
        )
        state_query = {"etsAuth": tenant_id, "syncKey": sync_key}
        if reset:
            await self.db.attendance_sync_state.delete_one(state_query)

        state = await self.db.attendance_sync_state.find_one(state_query)
        last_synced_at = since or (state or {}).get("lastSyncedAt")
        query = self._attendance_query(
            tenant_id=tenant_id,
            employee_id=employee_id,
            camera_id=camera_id,
            direction=direction,
            event_type=event_type,
            since=last_synced_at,
        )
        cursor = self.db.attendance_detections.find(query).sort("timestamp", 1).limit(limit)
        data = serialize_mongo_docs(await cursor.to_list(length=limit))

        update_doc = {
            "etsAuth": tenant_id,
            "syncKey": sync_key,
            "employeeId": employee_id,
            "cameraId": camera_id,
            "direction": direction,
            "eventType": event_type,
            "lastCheckedAt": datetime.utcnow(),
        }
        if data:
            update_doc["lastSyncedAt"] = max(item["timestamp"] for item in data)

        await self.db.attendance_sync_state.update_one(
            state_query,
            {"$set": update_doc, "$setOnInsert": {"createdAt": datetime.utcnow()}},
            upsert=True,
        )
        return self._format_attendance_rows(data)
    

    @staticmethod
    def _attendance_direction(camera_direction: str) -> str | None:
        normalized = camera_direction.upper()
        if normalized in {"IN", "OUT"}:
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
        tenant_id: str,
        employee_id: str | None = None,
        camera_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
    ) -> dict:
        query = {"etsAuth": tenant_id}
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
