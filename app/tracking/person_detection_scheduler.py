import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

import numpy as np

from app.tracking.models import PersonDetectionBatch
from app.tracking.person_detection_service import PersonDetectionService

logger = logging.getLogger(__name__)

PersonDetectionHandler = Callable[[PersonDetectionBatch], Awaitable[None]]


class PersonDetectionScheduler:
    """Run one shared person detector fairly across active cameras."""

    def __init__(self, detection_service: PersonDetectionService):
        self.detection_service = detection_service
        self._camera_order: list[str] = []
        self._handlers: dict[str, PersonDetectionHandler] = {}
        self._pending: dict[str, tuple[np.ndarray, datetime, float]] = {}
        self._cursor = 0
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.completed_batches = 0
        self.last_completed_at: float | None = None

    def register(self, camera_id: str, handler: PersonDetectionHandler) -> None:
        if camera_id not in self._handlers:
            self._camera_order.append(camera_id)
        self._handlers[camera_id] = handler
        self._ensure_started()
        logger.info("Registered camera %s with person detector scheduler", camera_id)

    def unregister(self, camera_id: str) -> None:
        self._handlers.pop(camera_id, None)
        self._pending.pop(camera_id, None)
        if camera_id in self._camera_order:
            removed_index = self._camera_order.index(camera_id)
            self._camera_order.pop(removed_index)
            if removed_index < self._cursor:
                self._cursor -= 1
        self._normalize_cursor()

    def submit(
        self,
        camera_id: str,
        frame: np.ndarray,
        observed_at: datetime,
        observed_monotonic: float,
    ) -> bool:
        if camera_id not in self._handlers or not self.detection_service.available:
            return False
        self._pending[camera_id] = (frame.copy(), observed_at, observed_monotonic)
        self._wake_event.set()
        return True

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._pending.clear()
        self._handlers.clear()
        self._camera_order.clear()
        self._cursor = 0
        self._wake_event.clear()

    def status(self) -> dict:
        return {
            "registeredCameras": len(self._handlers),
            "pendingCameras": len(self._pending),
            "completedBatches": self.completed_batches,
            "modelAvailable": self.detection_service.available,
            "unavailableReason": self.detection_service.unavailable_reason,
        }

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run(),
                name="person-detection-round-robin",
            )

    async def _run(self) -> None:
        while True:
            await self._wake_event.wait()
            scheduled = self._take_next()
            if scheduled is None:
                self._wake_event.clear()
                continue

            camera_id, handler, frame, observed_at, observed_monotonic = scheduled
            try:
                detections = await self.detection_service.detect(frame)
                await handler(
                    PersonDetectionBatch(
                        frame=frame,
                        detections=detections,
                        observed_at=observed_at,
                        observed_monotonic=observed_monotonic,
                    )
                )
                self.completed_batches += 1
                self.last_completed_at = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Person detection failed for camera %s", camera_id)

    def _take_next(
        self,
    ) -> tuple[str, PersonDetectionHandler, np.ndarray, datetime, float] | None:
        camera_count = len(self._camera_order)
        if camera_count == 0:
            return None

        for _ in range(camera_count):
            self._normalize_cursor()
            camera_id = self._camera_order[self._cursor]
            self._cursor = (self._cursor + 1) % camera_count
            handler = self._handlers.get(camera_id)
            pending = self._pending.pop(camera_id, None)
            if handler is not None and pending is not None:
                frame, observed_at, observed_monotonic = pending
                return camera_id, handler, frame, observed_at, observed_monotonic
        return None

    def _normalize_cursor(self) -> None:
        if not self._camera_order:
            self._cursor = 0
        else:
            self._cursor %= len(self._camera_order)
