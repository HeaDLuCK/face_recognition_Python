from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorDatabase


class SnapshotService:
    def __init__(self, snapshot_dir: Path, db: AsyncIOMotorDatabase):
        self.snapshot_dir = snapshot_dir
        self.db = db
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
        ).sort("createdAt", -1)

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
