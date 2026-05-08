from datetime import datetime
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase


class LogService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def write(
        self,
        level: str,
        message: str,
        tenant_id: str | None = None,
        camera_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        doc = {
            "logId": str(uuid4()),
            "tenantId": tenant_id,
            "cameraId": camera_id,
            "level": level,
            "message": message,
            "metadata": metadata or {},
            "createdAt": datetime.utcnow(),
        }
        await self.db.service_logs.insert_one(doc)
        return doc

