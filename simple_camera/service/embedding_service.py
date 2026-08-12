from datetime import datetime, timezone
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

from service.embedding_index import (
    EmbeddingIndex,
    build_embedding_index,
)
import numpy as np

class EmbeddingService:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self.db = db
        self._cached_index: EmbeddingIndex | None = None

    async def find_existing_embeddings(
            self,
            ets_auth: str,
            employeeId: str,
        ) -> list[dict]:
            cursor = self.db.cached_embeddings.find(
                {"etsAuth": ets_auth,"employeeId": employeeId,},
                {"embedding": 1,},
            )
    
            return await cursor.to_list(length=None)  

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
    
    async def filter_new_employee_embeddings(
        self,
        *,
        ets_auth: str,
        employee_id: str,
        embeddings: list[list[float]],
        duplicate_threshold: float = 0.92,
    ) -> tuple[list[list[float]], str | None]:

        if not embeddings:
            return []

        rows = await self.find_existing_embeddings(ets_auth=ets_auth,employeeId=employee_id,)
        existing_embeddings = []
        employee_name = None

        for row in rows:
            if employee_name is None and row.get("employeeName"):
                employee_name = row.get("employeeName")
            embedding = row.get("embedding")
            if not embedding:
                continue
            array = np.asarray(
                embedding,
                dtype=np.float32,
            )
            norm = np.linalg.norm(array)
            if norm == 0:
                continue
            existing_embeddings.append( array / norm)
        new_embeddings = []
        for embedding in embeddings:
            candidate = np.asarray(
                embedding,
                dtype=np.float32,
            )
            norm = np.linalg.norm(candidate)

            if norm == 0:
                continue
            candidate = candidate / norm
            duplicate = False


            for existing in existing_embeddings:
                similarity = float(np.dot(candidate, existing,))
                if similarity >= duplicate_threshold:
                    duplicate = True
                    break

            if duplicate:
                continue

            for accepted in new_embeddings:

                accepted_array = np.asarray(
                    accepted,
                    dtype=np.float32,
                )

                similarity = float(np.dot(candidate, accepted_array,))
                if similarity >= duplicate_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue

            new_embeddings.append(
                candidate.tolist()
            )

        return new_embeddings, employee_name
    