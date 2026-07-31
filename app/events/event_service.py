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

    async def summarize_person_counts(
        self,
        tenant_id: str,
        camera_id: str | None = None,
    ) -> dict:
        match: dict = {
            "etsAuth": tenant_id,
            "eventType": {"$in": ["PERSON_ENTERED", "PERSON_EXITED"]},
        }
        if camera_id:
            match["cameraId"] = camera_id

        rows = await self.db.camera_events.aggregate(
            [
                {"$match": match},
                {
                    "$group": {
                        "_id": "$cameraId",
                        "entered": {
                            "$sum": {"$cond": [{"$eq": ["$eventType", "PERSON_ENTERED"]}, 1, 0]}
                        },
                        "exited": {
                            "$sum": {"$cond": [{"$eq": ["$eventType", "PERSON_EXITED"]}, 1, 0]}
                        },
                    }
                },
                {"$sort": {"_id": 1}},
            ]
        ).to_list(length=None)

        cameras = [
            {
                "cameraId": row["_id"],
                "entered": row["entered"],
                "exited": row["exited"],
                "occupancy": max(0, row["entered"] - row["exited"]),
            }
            for row in rows
        ]
        if camera_id and not cameras:
            cameras.append(
                {"cameraId": camera_id, "entered": 0, "exited": 0, "occupancy": 0}
            )

        entered = sum(item["entered"] for item in cameras)
        exited = sum(item["exited"] for item in cameras)
        return {
            "etsAuth": tenant_id,
            "cameraId": camera_id,
            "entered": entered,
            "exited": exited,
            "occupancy": max(0, entered - exited),
            "cameras": cameras,
        }

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
