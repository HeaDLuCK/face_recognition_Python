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
    ) -> str:
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

        cursor = self.db.attendance_detections.find(query).sort("timestamp", 1).limit(limit)
        data = serialize_mongo_docs(await cursor.to_list(length=limit))
        result = ""
        for obj in data:
            result+=obj["metadata"]["employeeName"]+";"+obj["employeeId"]+";"+str(obj["timestamp"])+"|"
        
        return result
    

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
