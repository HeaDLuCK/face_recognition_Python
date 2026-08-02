import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

FaceRecognitionHandler = Callable[[Any], Awaitable[None]]


@dataclass
class _PendingRecognition:
    payload: Any
    quality: float | None


class FaceRecognitionScheduler:
    """Run one face inference at a time and rotate fairly across cameras and tracks."""

    def __init__(self, max_pending_per_camera: int = 12) -> None:
        self.max_pending_per_camera = max_pending_per_camera
        self._camera_order: list[str] = []
        self._handlers: dict[str, FaceRecognitionHandler] = {}
        self._pending_frames: dict[str, OrderedDict[str, _PendingRecognition]] = {}
        self._cursor = 0
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._active = False
        self._last_completed_at = 0.0

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

    def submit(
        self,
        camera_id: str,
        frame: Any,
        job_id: str | int | None = None,
        quality: float | None = None,
    ) -> bool:
        if camera_id not in self._handlers:
            return False

        pending = self._pending_frames.setdefault(camera_id, OrderedDict())
        key = str(job_id) if job_id is not None else "__latest_camera_frame__"
        existing = pending.get(key)
        if (
            existing is not None
            and quality is not None
            and existing.quality is not None
            and quality <= existing.quality
        ):
            return False

        if existing is None and len(pending) >= self.max_pending_per_camera:
            return False
        payload = frame.copy() if hasattr(frame, "copy") else frame
        pending[key] = _PendingRecognition(payload=payload, quality=quality)
        self._wake_event.set()
        return True

    @property
    def pending_jobs(self) -> int:
        return sum(len(pending) for pending in self._pending_frames.values())

    @property
    def is_idle(self) -> bool:
        return not self._active and self.pending_jobs == 0

    async def wait_until_idle(self, idle_seconds: float, timeout_seconds: float) -> bool:
        started_at = time.monotonic()
        while time.monotonic() - started_at < timeout_seconds:
            if self.is_idle:
                quiet_for = time.monotonic() - self._last_completed_at
                if self._last_completed_at == 0.0 or quiet_for >= idle_seconds:
                    return True
            await asyncio.sleep(0.05)
        return False

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
        self._active = False

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
                self._active = True
                await handler(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Face recognition failed for camera %s; scheduler will continue",
                    camera_id,
                )
            finally:
                self._active = False
                self._last_completed_at = time.monotonic()

    def _take_next(self) -> tuple[str, FaceRecognitionHandler, Any] | None:
        camera_count = len(self._camera_order)
        if camera_count == 0:
            return None

        for _ in range(camera_count):
            self._normalize_cursor()
            camera_id = self._camera_order[self._cursor]
            self._cursor = (self._cursor + 1) % camera_count
            handler = self._handlers.get(camera_id)
            pending = self._pending_frames.get(camera_id)
            if handler is None or not pending:
                continue
            _, item = pending.popitem(last=False)
            if not pending:
                self._pending_frames.pop(camera_id, None)
            return camera_id, handler, item.payload

        return None

    def _normalize_cursor(self) -> None:
        if not self._camera_order:
            self._cursor = 0
        else:
            self._cursor %= len(self._camera_order)
