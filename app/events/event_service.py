import logging
from datetime import datetime
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import serialize_mongo_docs
from app.schemas.erp_schema import ErpEventPayload
from app.schemas.runtime_schema import RuntimeEvent

logger = logging.getLogger(__name__)


class EventService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def create_camera_event(self, payload: RuntimeEvent) -> dict:
        doc = {
            **payload.model_dump(mode="json", by_alias=True),
            "eventId": str(uuid4()),
            "createdAt": datetime.utcnow(),
            
        }
        await self.db.camera_events.insert_one(doc)
        return doc

    async def create_alert_event(self, payload: RuntimeEvent) -> dict:
        doc = {
            **payload.model_dump(mode="json", by_alias=True),
            "alertId": str(uuid4()),
            "createdAt": datetime.utcnow(),
        }
        await self.db.alert_events.insert_one(doc)
        return doc

    async def list_events(
        self,
        tenant_id: str,
        limit: int = 100,
        camera_id: str | None = None,
        employee_id: str | None = None,
        direction: str | None = None,
        event_type: str | None = None,
    ) -> list[dict]:
        query = {"etsAuth": tenant_id}
        if camera_id:
            query["cameraId"] = camera_id
        if employee_id:
            query["employeeId"] = employee_id
        normalized_event_type = self._event_type_filter(direction, event_type)
        if normalized_event_type:
            query["eventType"] = normalized_event_type

        cursor = self.db.camera_events.find(query).sort("timestamp", -1).limit(limit)
        return serialize_mongo_docs(await cursor.to_list(length=limit))

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

