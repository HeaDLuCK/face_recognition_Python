import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any


logger = logging.getLogger(__name__)

FaceRecognitionHandler = Callable[[Any], Awaitable[None]]


class FaceRecognitionScheduler:
    """Run one face inference at a time and rotate fairly across cameras."""

    def __init__(self) -> None:
        self._camera_order: list[str] = []
        self._handlers: dict[str, FaceRecognitionHandler] = {}
        self._pending_frames: dict[str, Any] = {}
        self._cursor = 0
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    def register(self, camera_id: str, handler: FaceRecognitionHandler) -> None:
        if camera_id not in self._handlers:
            self._camera_order.append(camera_id)
        self._handlers[camera_id] = handler
        self._ensure_started()
        logger.info("Registered camera %s with face AI scheduler", camera_id)

    def unregister(self, camera_id: str) -> None:
        self._handlers.pop(camera_id, None)
        self._pending_frames.pop(camera_id, None)
        if camera_id in self._camera_order:
            removed_index = self._camera_order.index(camera_id)
            self._camera_order.pop(removed_index)
            if removed_index < self._cursor:
                self._cursor -= 1
        self._normalize_cursor()
        logger.info("Unregistered camera %s from face AI scheduler", camera_id)

    def submit(self, camera_id: str, frame: Any) -> bool:
        if camera_id not in self._handlers:
            return False

        # One slot per camera prevents stale-frame queues while preserving fairness.
        self._pending_frames[camera_id] = frame.copy()
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
        self._pending_frames.clear()
        self._handlers.clear()
        self._camera_order.clear()
        self._cursor = 0
        self._wake_event.clear()

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._run(),
                name="face-recognition-round-robin",
            )

    async def _run(self) -> None:
        while True:
            await self._wake_event.wait()

            scheduled = self._take_next()
            if scheduled is None:
                self._wake_event.clear()
                continue

            camera_id, handler, frame = scheduled
            try:
                await handler(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Face recognition failed for camera %s; scheduler will continue",
                    camera_id,
                )

    def _take_next(self) -> tuple[str, FaceRecognitionHandler, Any] | None:
        camera_count = len(self._camera_order)
        if camera_count == 0:
            return None

        for _ in range(camera_count):
            self._normalize_cursor()
            camera_id = self._camera_order[self._cursor]
            self._cursor = (self._cursor + 1) % camera_count
            handler = self._handlers.get(camera_id)
            frame = self._pending_frames.pop(camera_id, None)
            if handler is not None and frame is not None:
                return camera_id, handler, frame

        return None

    def _normalize_cursor(self) -> None:
        if not self._camera_order:
            self._cursor = 0
        else:
            self._cursor %= len(self._camera_order)
