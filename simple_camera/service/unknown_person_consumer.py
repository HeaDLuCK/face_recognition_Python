import asyncio
import logging
from datetime import datetime

import cv2
import numpy as np
from config import get_settings 

from service.unknown_person_service import (
    UnknownPersonService,
)
from service.attendance_service import (
    AttendanceService,
)

logger = logging.getLogger(__name__)


class UnknownPersonConsumer:
    def __init__(
        self,
        queue,
        service: UnknownPersonService,
        attendance_service: AttendanceService,
    ) -> None:
        self.queue = queue
        self.attendance_service = attendance_service
        self.service = service
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        self._running = True

        self._task = asyncio.create_task(
            self._run()
        )

        logger.info(
            "Unknown person consumer started"
        )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._task is not None:
            await self._task
            self._task = None

        logger.info(
            "Unknown person consumer stopped"
        )

    async def _run(self) -> None:
        while self._running:
            try:
                item = await asyncio.to_thread(
                    self.queue.get,
                    True,
                    1.0,
                )

            except Exception:
                continue

            if item is None:
                break


            try:
                await self._process_item(
                    item
                )

            except Exception:
                logger.exception(
                    "Failed to process unknown person"
                )

    async def _process_item(
        self,
        item: dict,
    ) -> None:
        embedding = np.asarray(
            item["embedding"],
            dtype=np.float32,
        )

        face_crop = self._decode_image(
            item["faceJpeg"]
        )

        observed_at = datetime.fromisoformat(
            item["observedAt"]
        )

        face_path = await self._save_image(
            image=face_crop,
            unknown_type="face",
        )

        unknown = await self.service.register_seen(
            ets_auth=item["etsAuth"],
            embedding=embedding,
            face_path=face_path,
            quality=float(
                item["quality"]
            ),
            observed_at=observed_at,
            match_threshold=float(
                item.get(
                    "matchThreshold",
                    0.55,
                )
            ),
        )

        unknown_id = unknown[
            "unknownId"
        ]

        camera_direction = item.get(
            "cameraDirection"
        )

        ets_auth = item.get(
            "etsAuth"
        )

        rule = item.get(
            "rule"
        )

        attendance_result = await (
            self.attendance_service
            .record_unknown_attendance_if_allowed(
                ets_auth=ets_auth,
                unknown_id=unknown_id,
                camera_id=item["cameraId"],
                camera_direction=(
                    camera_direction
                ),
                rule=rule,
                event_time=observed_at,
            )
        )

        logger.info(
            "Unknown attendance result: "
            "unknownId=%s camera=%s "
            "created=%s reason=%s",
            unknown_id,
            item["cameraId"],
            attendance_result.get(
                "created"
            ),
            attendance_result.get(
                "reason"
            ),
        )

    @staticmethod
    def _decode_image(
        data: bytes,
    ) -> np.ndarray:
        encoded = np.frombuffer(
            data,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Unable to decode unknown person image"
            )

        return image

    async def _save_image(
        self,
        image: np.ndarray,
        unknown_type: str,
    ) -> str:
        """
        Temporary simple image saving.

        Later we will organize files under the final unknownId.
        """

        from pathlib import Path
        from uuid import uuid4
        settings = get_settings()
        folder = (settings.snapshot_dir
                / "unknown"
            )
        
        folder.mkdir(parents=True,exist_ok=True,)

        filename = (
            f"{unknown_type}_"
            f"{uuid4().hex}.jpg"
        )

        path = folder / filename

        success = await asyncio.to_thread(
            cv2.imwrite,
            str(path),
            image,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save image: {path}"
            )

        return str(path)