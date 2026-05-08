from datetime import datetime
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database import serialize_mongo_docs


class EmbeddingService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def upsert_employee_embeddings(
        self,
        tenant_id: str,
        employee_id: str,
        employee_name: str | None,
        embeddings: list[list[float]],
        source_id: str,
    ) -> int:
        if not embeddings:
            return 0

        now = datetime.utcnow()
        count = 0
        for index, embedding in enumerate(embeddings):
            doc = {
                "tenantId": tenant_id,
                "employeeId": employee_id,
                "employeeName": employee_name,
                "sourceId": f"{source_id}#{index}",
                "embedding": embedding,
                "updatedAt": now,
            }
            result = await self.db.cached_embeddings.update_one(
                {
                    "tenantId": tenant_id,
                    "employeeId": employee_id,
                    "sourceId": doc["sourceId"],
                },
                {
                    "$set": doc,
                    "$setOnInsert": {"embeddingId": str(uuid4()), "createdAt": now},
                },
                upsert=True,
            )
            if result.upserted_id is not None or result.modified_count > 0:
                count += 1
        return count

    async def list_tenant_embeddings(self, tenant_id: str) -> list[dict]:
        cursor = self.db.cached_embeddings.find(
            {"tenantId": tenant_id},
            {"_id": 0, "embedding": 1, "employeeId": 1, "employeeName": 1, "embeddingId": 1},
        )
        return serialize_mongo_docs(await cursor.to_list(length=None))

    async def count_tenant_embeddings(self, tenant_id: str) -> int:
        return await self.db.cached_embeddings.count_documents({"tenantId": tenant_id})
