from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
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
            "tenantId": tenant_id,
            "cameraId": camera_id,
            "path": snapshot_path,
            "eventType": event_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        }
        await self.db.snapshot_metadata.insert_one(doc)
        return doc
