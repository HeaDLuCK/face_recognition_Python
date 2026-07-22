import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase


class SnapshotService:
    def __init__(
        self,
        snapshot_dir: Path,
        db: AsyncIOMotorDatabase,
        unknown_face_db_match_limit: int = 500,
        purge_batch_size: int = 200,
    ):
        self.snapshot_dir = snapshot_dir
        self.db = db
        self.unknown_face_db_match_limit = unknown_face_db_match_limit
        self.purge_batch_size = purge_batch_size
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(self, tenant_id: str, camera_id: str, frame) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        folder = self.snapshot_dir / tenant_id / camera_id / today
        folder.mkdir(parents=True, exist_ok=True)

        filename = f"{datetime.utcnow().strftime('%H%M%S_%f')}_{uuid4().hex}.jpg"
        path = folder / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Unable to write snapshot to {path}")
        return str(path)

    def save_face_crop(self, tenant_id: str, camera_id: str, frame) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        folder = self.snapshot_dir / tenant_id / camera_id / today / "faces"
        folder.mkdir(parents=True, exist_ok=True)

        filename = f"unknown_{datetime.utcnow().strftime('%H%M%S_%f')}_{uuid4().hex}.jpg"
        path = folder / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"Unable to write face crop to {path}")
        return str(path)

    async def save_metadata(
        self,
        tenant_id: str,
        camera_id: str,
        snapshot_path: str,
        event_type: str,
        metadata: dict | None = None,
    ) -> dict:
        doc = {
            "snapshotId": str(uuid4()),
            "etsAuth": tenant_id,
            "cameraId": camera_id,
            "path": snapshot_path,
            "eventType": event_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        }
        await self.db.snapshot_metadata.insert_one(doc)
        return doc

    async def find_unknown_face_crop(
        self,
        tenant_id: str,
        camera_id: str,
        embedding: list[float],
        threshold: float,
    ) -> str | None:
        current = np.array(embedding, dtype=np.float32)
        cursor = self.db.unknown_face_crops.find(
            {"etsAuth": tenant_id, "cameraId": camera_id},
            {"_id": 0, "path": 1, "embedding": 1},
        ).sort("createdAt", -1).limit(self.unknown_face_db_match_limit)

        async for item in cursor:
            stored_embedding = item.get("embedding")
            path = item.get("path")
            if not stored_embedding or not path:
                continue
            score = float(np.dot(current, np.array(stored_embedding, dtype=np.float32)))
            if score >= threshold:
                return path
        return None

    async def register_unknown_face_crop(
        self,
        tenant_id: str,
        camera_id: str,
        crop_path: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> dict:
        doc = {
            "unknownFaceCropId": str(uuid4()),
            "etsAuth": tenant_id,
            "cameraId": camera_id,
            "path": crop_path,
            "embedding": embedding,
            "metadata": metadata or {},
            "createdAt": datetime.utcnow(),
        }
        await self.db.unknown_face_crops.insert_one(doc)
        return doc

    async def purge_expired_images(self, tenant_id: str, retention_days: int) -> dict:
        if retention_days <= 0:
            return {
                "etsAuth": tenant_id,
                "enabled": False,
                "retentionDays": retention_days,
                "deletedFiles": 0,
                "snapshotMetadataDeleted": 0,
                "unknownFaceCropsDeleted": 0,
            }

        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        deleted_files = await self._delete_expired_files(
            [
                (
                    self.db.snapshot_metadata,
                    {"etsAuth": tenant_id, "timestamp": {"$lt": cutoff}},
                ),
                (
                    self.db.unknown_face_crops,
                    {"etsAuth": tenant_id, "createdAt": {"$lt": cutoff}},
                ),
            ]
        )

        snapshot_result = await self.db.snapshot_metadata.delete_many(
            {"etsAuth": tenant_id, "timestamp": {"$lt": cutoff}},
        )
        unknown_result = await self.db.unknown_face_crops.delete_many(
            {"etsAuth": tenant_id, "createdAt": {"$lt": cutoff}},
        )

        return {
            "etsAuth": tenant_id,
            "enabled": True,
            "retentionDays": retention_days,
            "cutoff": cutoff.isoformat(),
            "deletedFiles": deleted_files,
            "snapshotMetadataDeleted": snapshot_result.deleted_count,
            "unknownFaceCropsDeleted": unknown_result.deleted_count,
        }

    async def _delete_expired_files(self, collection_queries: list[tuple[object, dict]]) -> int:
        deleted_files = 0
        pending_paths: list[str | None] = []
        for collection, query in collection_queries:
            cursor = collection.find(query, {"_id": 0, "path": 1}).batch_size(self.purge_batch_size)
            async for doc in cursor:
                pending_paths.append(doc.get("path"))
                if len(pending_paths) >= self.purge_batch_size:
                    deleted_files += await asyncio.to_thread(self._delete_raw_paths, pending_paths)
                    pending_paths = []

        if pending_paths:
            deleted_files += await asyncio.to_thread(self._delete_raw_paths, pending_paths)
        return deleted_files

    def _delete_raw_paths(self, raw_paths: list[str | None]) -> int:
        return self._delete_files(self._safe_existing_paths(raw_paths))

    def _safe_existing_paths(self, raw_paths: list[str | None]) -> list[Path]:
        safe_paths = []
        snapshot_root = self.snapshot_dir.resolve()
        for raw_path in raw_paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            resolved = path.resolve()
            if snapshot_root not in resolved.parents or not resolved.is_file():
                continue
            safe_paths.append(resolved)
        return list(dict.fromkeys(safe_paths))

    def _delete_files(self, paths: list[Path]) -> int:
        deleted = 0
        for path in paths:
            try:
                path.unlink()
                deleted += 1
                self._remove_empty_snapshot_dirs(path.parent)
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return deleted

    def _remove_empty_snapshot_dirs(self, folder: Path) -> None:
        snapshot_root = self.snapshot_dir.resolve()
        current = folder.resolve()
        while current != snapshot_root and snapshot_root in current.parents:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent
