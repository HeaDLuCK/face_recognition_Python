import asyncio
import logging
from datetime import datetime

import cv2
import numpy as np

from service.unknown_person_service import (
    UnknownPersonService,
)


logger = logging.getLogger(__name__)


class UnknownPersonConsumer:
    def __init__(
        self,
        queue,
        service: UnknownPersonService,
    ) -> None:
        self.queue = queue
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

        context_crop = self._decode_image(
            item["contextJpeg"]
        )

        observed_at = datetime.fromisoformat(
            item["observedAt"]
        )

        face_path = await self._save_image(
            image=face_crop,
            unknown_type="face",
        )

        context_path = await self._save_image(
            image=context_crop,
            unknown_type="context",
        )

        unknown = await self.service.register_seen(
            embedding=embedding,
            face_path=face_path,
            context_path=context_path,
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

        logger.info(
            "Unknown person processed: "
            "unknownId=%s camera=%s",
            unknown["unknownId"],
            item["cameraId"],
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

        folder = Path(
            "storage/unknown/temp"
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

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