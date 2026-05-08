import logging
from datetime import datetime
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import serialize_mongo_docs
from app.erp.erp_client import ErpClient
from app.schemas.erp_schema import ErpEventPayload
from app.schemas.runtime_schema import RuntimeEvent

logger = logging.getLogger(__name__)


class EventService:
    def __init__(self, db: AsyncIOMotorDatabase, erp_client: ErpClient):
        self.db = db
        self.erp_client = erp_client

    async def create_camera_event(self, payload: RuntimeEvent, send_to_erp: bool = True) -> dict:
        doc = {
            **payload.model_dump(mode="json"),
            "eventId": str(uuid4()),
            "createdAt": datetime.utcnow(),
            "erpDeliveryStatus": "pending" if send_to_erp else "not_required",
        }
        await self.db.camera_events.insert_one(doc)
        if send_to_erp:
            await self._send_to_erp(doc, "camera_events")
        return doc

    async def create_alert_event(self, payload: RuntimeEvent, send_to_erp: bool = True) -> dict:
        doc = {
            **payload.model_dump(mode="json"),
            "alertId": str(uuid4()),
            "createdAt": datetime.utcnow(),
            "erpDeliveryStatus": "pending" if send_to_erp else "not_required",
        }
        await self.db.alert_events.insert_one(doc)
        if send_to_erp:
            await self._send_to_erp(doc, "alert_events")
        return doc

    async def list_events(
        self,
        tenant_id: str,
        limit: int = 100,
        camera_id: str | None = None,
        employee_id: str | None = None,
    ) -> list[dict]:
        query = {"tenantId": tenant_id}
        if camera_id:
            query["cameraId"] = camera_id
        if employee_id:
            query["employeeId"] = employee_id

        cursor = self.db.camera_events.find(query).sort("timestamp", -1).limit(limit)
        return serialize_mongo_docs(await cursor.to_list(length=limit))

    async def _send_to_erp(self, doc: dict, collection_name: str) -> None:
        payload = ErpEventPayload(
            tenantId=doc["tenantId"],
            cameraId=doc["cameraId"],
            eventType=doc["eventType"],
            employeeId=doc.get("employeeId"),
            confidence=doc.get("confidence"),
            snapshotPath=doc.get("snapshotPath"),
            timestamp=doc["timestamp"],
            metadata=doc.get("metadata", {}),
        )
        try:
            await self.erp_client.send_event(payload)
            await self.db[collection_name].update_one(
                {"_id": doc["_id"]},
                {"$set": {"erpDeliveryStatus": "sent", "erpDeliveredAt": datetime.utcnow()}},
            )
        except Exception as exc:
            logger.exception("Failed to send event %s to ERP", doc.get("eventId") or doc.get("alertId"))
            await self.db[collection_name].update_one(
                {"_id": doc["_id"]},
                {"$set": {"erpDeliveryStatus": "failed", "erpDeliveryError": str(exc)}},
            )
