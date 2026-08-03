from motor.motor_asyncio import AsyncIOMotorDatabase

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