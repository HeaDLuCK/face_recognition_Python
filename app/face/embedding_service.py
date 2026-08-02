from datetime import datetime
from time import monotonic
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

from app.face.embedding_index import TenantEmbeddingIndex, build_embedding_index


class EmbeddingService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self._tenant_cache: dict[str, tuple[float, TenantEmbeddingIndex]] = {}
        self._cache_ttl_seconds = 10.0

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
        operations = []
        for index, embedding in enumerate(embeddings):
            doc = {
                "etsAuth": tenant_id,
                "employeeId": employee_id,
                "employeeName": employee_name,
                "sourceId": f"{source_id}#{index}",
                "embedding": embedding,
                "updatedAt": now,
            }
            operations.append(
                UpdateOne(
                    {
                        "etsAuth": tenant_id,
                        "employeeId": employee_id,
                        "sourceId": doc["sourceId"],
                    },
                    {
                        "$set": doc,
                        "$setOnInsert": {"embeddingId": str(uuid4()), "createdAt": now},
                    },
                    upsert=True,
                )
            )

        result = await self.db.cached_embeddings.bulk_write(operations, ordered=False)
        self._tenant_cache.pop(tenant_id, None)
        return result.upserted_count + result.modified_count

    async def list_tenant_embeddings(self, tenant_id: str) -> list[dict]:
        return (await self.get_tenant_index(tenant_id)).items

    async def get_tenant_index(self, tenant_id: str) -> TenantEmbeddingIndex:
        cached = self._tenant_cache.get(tenant_id)
        if cached and monotonic() - cached[0] <= self._cache_ttl_seconds:
            return cached[1]

        cursor = self.db.cached_embeddings.find(
            {"etsAuth": tenant_id},
            {"_id": 0, "embedding": 1, "employeeId": 1, "employeeName": 1, "embeddingId": 1},
        )
        index = build_embedding_index(await cursor.to_list(length=None))
        self._tenant_cache[tenant_id] = (monotonic(), index)
        return index

    async def count_tenant_embeddings(self, tenant_id: str) -> int:
        return await self.db.cached_embeddings.count_documents({"etsAuth": tenant_id})
