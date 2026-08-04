from datetime import datetime, timezone
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

from service.embedding_index import (
    EmbeddingIndex,
    build_embedding_index,
)


class EmbeddingService:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self.db = db
        self._cached_index: EmbeddingIndex | None = None

    async def get_all_embeddings_index(
        self,
    ) -> EmbeddingIndex:
        if self._cached_index is not None:
            return self._cached_index

        return await self.refresh_all_embeddings_index()

    async def refresh_all_embeddings_index(
        self,
    ) -> EmbeddingIndex:
        cursor = self.db.cached_embeddings.find(
            {},
            {
                "_id": 0,
                "embedding": 1,
                "employeeId": 1,
                "employeeName": 1,
                "embeddingId": 1,
                "etsAuth": 1,
            },
        )

        documents = await cursor.to_list(length=None)

        index = build_embedding_index(documents)
        self._cached_index = index

        return index

    def clear_embeddings_index(self) -> None:
        self._cached_index = None

    async def count_all_embeddings(self) -> int:
        return await self.db.cached_embeddings.count_documents({})

    async def upsert_employee_embeddings(
            self,
            etsAuth: str,
            employee_id: str,
            employee_name: str | None,
            embeddings: list[list[float]],
            source_id: str,
        ) -> int:
            if not embeddings:
                return 0
    
            now = datetime.now(timezone.utc)
            operations = []
            for index, embedding in enumerate(embeddings):
                doc = {
                    "etsAuth": etsAuth,
                    "employeeId": employee_id,
                    "employeeName": employee_name,
                    "sourceId": f"{source_id}#{index}",
                    "embedding": embedding,
                    "updatedAt": now,
                }
                operations.append(
                    UpdateOne(
                        {
                            "etsAuth": etsAuth,
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
            await self.refresh_all_embeddings_index()
            return result.upserted_count + result.modified_count
    