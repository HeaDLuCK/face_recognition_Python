import logging
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase
from schemas.project_schema import UnknownPersonStatus



logger = logging.getLogger(__name__)


class UnknownPersonService:
    """
    Manage unknown people detected by the cameras.

    Responsibilities:

    - create a new unknown person;
    - recognize the same unknown person later;
    - update lastSeenAt;
    - keep the best face image;
    - assign an unknown person to an employee later.
    """
    REFERENCE_DUPLICATE_THRESHOLD = 0.92
    MIN_REFERENCE_QUALITY = 0.60
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
    ) -> None:
        self.db = db
        

    async def register_seen(
        self,
        *,
        ets_auth: str,
        embedding: np.ndarray,
        face_path: str,
        quality: float,
        observed_at: datetime,
        match_threshold: float,
    ) -> dict:
        """
        Register an unknown face observation.

        First try to find whether this face already belongs to an
        existing UNASSIGNED unknown person.

        If found:f
            update that unknown.

        If not found:
            create a new unknown person.
        """

        normalized_embedding = self._normalize_embedding(
            embedding
        )

        existing = await self.find_matching_unknown(
            ets_auth=ets_auth,
            embedding=normalized_embedding,
            threshold=match_threshold,
        )

        if existing is not None:
            return await self._update_existing_unknown(
                existing=existing,
                embedding=normalized_embedding,
                face_path=face_path,
                quality=quality,
                observed_at=observed_at,
            )

        return await self._create_unknown(
            ets_auth=ets_auth,
            embedding=normalized_embedding,
            face_path=face_path,
            quality=quality,
            observed_at=observed_at,
        )

    async def find_matching_unknown(
        self,
        *,
        ets_auth: str,
        embedding: np.ndarray,
        threshold: float,
    ) -> dict | None:
        """
        Compare a new face against existing unassigned unknown people.

        Returns the best matching unknown when its similarity is above
        the provided threshold.
        """

        normalized_embedding = self._normalize_embedding(
            embedding
        )

        cursor = self.db.unknown_persons.find(
            {   "etsAuth": ets_auth,
                "status": (
                    UnknownPersonStatus.UNASSIGNED.value
                ),
            },
            {
                "unknownId": 1,
                "referenceEmbeddings": 1,
                "referenceEmbedding": 1,
                "bestQuality": 1,
                "facePath": 1,
                "contextPath": 1,
                "lastSeenAt": 1,
                "seenCount": 1,
            },
        )

        best_document: dict | None = None
        best_score = -1.0

        async for document in cursor:
            stored_embeddings = document.get("referenceEmbeddings")

            # Support old Mongo documents temporarily.
            if not stored_embeddings:
                old_embedding = document.get("referenceEmbedding")

                if old_embedding:
                    stored_embeddings = [old_embedding]

            if not stored_embeddings:
                continue

            document_best_score = -1.0

            for stored_embedding in stored_embeddings:
                stored = np.asarray(stored_embedding,dtype=np.float32,)

                stored = self._normalize_embedding(stored)

                score = float(np.dot(normalized_embedding,stored,))

                if score > document_best_score:
                    document_best_score = score
            if document_best_score > best_score:
                best_score = document_best_score
                best_document = document

        if (
            best_document is None
            or best_score < threshold
        ):
            return None
        logger.info(
            "Unknown matching result: "
            "bestUnknownId=%s "
            "bestScore=%.4f threshold=%.4f",
            (
                best_document.get("unknownId")
                if best_document is not None
                else None
            ),
            best_score,
            threshold,
        )
        best_document["matchScore"] = best_score

        return best_document

    async def _create_unknown(
        self,
        *,
        ets_auth: str,
        embedding: np.ndarray,
        face_path: str,
        quality: float,
        observed_at: datetime,
    ) -> dict:
        """
        Create a new unknown person.
        """

        now = datetime.now(
            timezone.utc
        )

        unknown_id = self._generate_unknown_id()

        document = {
            "etsAuth": ets_auth,
            "unknownId": unknown_id,

            "status": (
                UnknownPersonStatus
                .UNASSIGNED
                .value
            ),

            "firstSeenAt": observed_at,
            "lastSeenAt": observed_at,

            "seenCount": 1,

            "referenceEmbeddings": [
                embedding
                .astype(float)
                .tolist()
            ],

            "facePath": face_path,

            "bestQuality": float(
                quality
            ),

            "assignedEmployeeId": None,
            "assignedAt": None,

            # Needs to be sent to ERP.
            "erpSyncStatus": "PENDING",
            "erpSyncedAt": None,

            "createdAt": now,
            "updatedAt": now,
        }

        result = await self.db.unknown_persons.insert_one(
            document
        )

        document["_id"] = result.inserted_id

        logger.info(
            "Unknown person created: "
            "unknownId=%s quality=%.4f",
            unknown_id,
            quality,
        )

        return document

    async def _update_existing_unknown(
        self,
        *,
        existing: dict,
        embedding: np.ndarray,
        face_path: str,
        quality: float,
        observed_at: datetime,
    ) -> dict:
        """
        Update an unknown person that has been seen again.
        """

        update_fields = {
            "lastSeenAt": observed_at,
            "updatedAt": datetime.now(
                timezone.utc
            ),
        }

        current_best_quality = float(
            existing.get(
                "bestQuality",
                0.0,
            )
        )

        should_add_embedding = (
            self._should_add_reference_embedding(
                existing=existing,
                embedding=embedding,
                quality=quality,
            )
        )

        if should_add_embedding:
            reference_embeddings = list(
                existing.get(
                    "referenceEmbeddings"
                )
                or []
            )

            # Convert old document format if needed.
            if (
                not reference_embeddings
                and existing.get(
                    "referenceEmbedding"
                )
            ):
                reference_embeddings.append(
                    existing[
                        "referenceEmbedding"
                    ]
                )

            reference_embeddings.append(
                embedding
                .astype(float)
                .tolist()
            )

            # KEEP THIS INSIDE THE IF
            update_fields[
                "referenceEmbeddings"
            ] = reference_embeddings

        # Keep the better face as the reference face.
        if quality > current_best_quality:
            update_fields.update(
                {
                    "facePath": face_path,

                    "bestQuality": float(
                        quality
                    ),

                    "erpSyncStatus": "PENDING",
                    "erpSyncedAt": None,
                }
            )

        await self.db.unknown_persons.update_one(
            {
                "_id": existing["_id"]
            },
            {
                "$set": update_fields,
                "$inc": {
                    "seenCount": 1,
                },
            },
        )

        existing.update(
            update_fields
        )

        existing["seenCount"] = (
            int(
                existing.get(
                    "seenCount",
                    0,
                )
            )
            + 1
        )

        logger.info(
            "Unknown person seen again: "
            "unknownId=%s matchScore=%.4f "
            "quality=%.4f",
            existing["unknownId"],
            existing.get(
                "matchScore",
                0.0,
            ),
            quality,
        )

        return existing

    async def assign_to_employee(
        self,
        *,
        ets_auth: str,
        unknown_id: str,
        employee_id: str,
    ) -> bool:
        """
        Administrator confirms that an unknown person belongs
        to a specific employee.
        """

        now = datetime.now(
            timezone.utc
        )

        result = await self.db.unknown_persons.update_one(
            {   "etsAuth": ets_auth,
                "unknownId": unknown_id,
                "status": (
                    UnknownPersonStatus
                    .UNASSIGNED
                    .value
                ),
            },
            {
                "$set": {
                    "status": (
                        UnknownPersonStatus
                        .ASSIGNED
                        .value
                    ),
                    "assignedEmployeeId": (
                        employee_id
                    ),
                    "assignedAt": now,
                    "updatedAt": now,
                }
            },
        )

        if result.modified_count == 0:
            return False

        logger.info(
            "Unknown person assigned: "
            "unknownId=%s employee=%s",
            unknown_id,
            employee_id,
        )

        return True

    async def get_by_unknown_id(
        self,
        ets_auth: str,
        unknown_id: str,
    ) -> dict | None:
        return await self.db.unknown_persons.find_one(
            {   "etsAuth": ets_auth,
                "unknownId": unknown_id
            }
        )

    async def get_unassigned(
        self,
    ) -> list[dict]:
        cursor = self.db.unknown_persons.find(
            {
                "status": (
                    UnknownPersonStatus
                    .UNASSIGNED
                    .value
                ),
            }
        ).sort(
            "lastSeenAt",
            -1,
        )

        return [
            document
            async for document in cursor
        ]

    @staticmethod
    def _generate_unknown_id() -> str:
        return (
            "UNK-"
            + uuid4()
            .hex[:12]
            .upper()
        )

    def _should_add_reference_embedding(
        self,
        *,
        existing: dict,
        embedding: np.ndarray,
        quality: float,
    ) -> bool:

        # Ignore very poor face samples.
        if quality < self.MIN_REFERENCE_QUALITY:
            return False

        reference_embeddings = (
            existing.get(
                "referenceEmbeddings"
            )
            or []
        )

        # Backward compatibility with old records.
        if (
            not reference_embeddings
            and existing.get(
                "referenceEmbedding"
            )
        ):
            reference_embeddings = [
                existing[
                    "referenceEmbedding"
                ]
            ]

        candidate = self._normalize_embedding(
            embedding
        )

        for stored_embedding in reference_embeddings:

            stored = self._normalize_embedding(
                np.asarray(
                    stored_embedding,
                    dtype=np.float32,
                )
            )

            similarity = float(
                np.dot(
                    candidate,
                    stored,
                )
            )

            # Almost the same embedding already exists.
            if (
                similarity
                >= self.REFERENCE_DUPLICATE_THRESHOLD
            ):
                return False

        return True

    @staticmethod
    def _normalize_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:
        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = float(
            np.linalg.norm(
                embedding
            )
        )

        if norm <= 0:
            raise ValueError(
                "Unknown face embedding has zero norm"
            )

        return embedding / norm

    async def get_pending_erp_sync(
        self,
        limit: int = 10,
    ) -> list[dict]:
        """
        Return unassigned unknown people that need
        to be synchronized with the ERP.
        """

        cursor = (
            self.db.unknown_persons
            .find(
                {
                    "status": (
                        UnknownPersonStatus
                        .UNASSIGNED
                        .value
                    ),
                    "erpSyncStatus": "PENDING",
                }
            )
            .sort(
                "lastSeenAt",
                1,
            )
            .limit(
                limit
            )
        )

        return [
            document
            async for document in cursor
        ]

    async def mark_erp_synced(
        self,
        unknown_ids: list[str],
    ) -> int:
        """
        Mark unknown people as successfully sent to ERP.
        """

        if not unknown_ids:
            return 0

        now = datetime.now(
            timezone.utc
        )

        result = await (
            self.db.unknown_persons
            .update_many(
                {
                    "unknownId": {
                        "$in": unknown_ids
                    },
                    "erpSyncStatus": "PENDING",
                },
                {
                    "$set": {
                        "erpSyncStatus": "SYNCED",
                        "erpSyncedAt": now,
                        "updatedAt": now,
                    }
                },
            )
        )

        logger.info(
            "Unknown persons marked ERP synced: "
            "count=%d",
            result.modified_count,
        )

        return result.modified_count