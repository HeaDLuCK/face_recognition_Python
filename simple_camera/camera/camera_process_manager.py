import asyncio
import multiprocessing as mp
from typing import Any
from service.sync_service import SyncService
from service.embedding_service import (
    EmbeddingService,   
    EmbeddingIndex,
)
from service.attendance_service import AttendanceService
from camera.camera_worker import read_camera
from schemas.project_schema import CameraConfig
from camera.camera_grid import show_camera_grid
import logging
from queue import Empty, Full

logger = logging.getLogger(__name__)

class CameraProcessManager:
    def __init__(
        self,
        sync_service: SyncService,
        embedding_service: EmbeddingService,
        attendance_service: AttendanceService,
        mp_context: Any,
        log_queue: Any,
    ) -> None:
        self.sync_service = sync_service
        self.embedding_service = embedding_service
        self.attendance_service = attendance_service
        # Suitable for Windows multiprocessing.
        self._context = mp_context

        self._processes: dict[str, mp.Process] = {}
        self._lock = asyncio.Lock()
        self._frame_queues: dict[str, Any] = {}
        self._log_queue = log_queue
        self._attendance_queue = self._context.Queue(maxsize=500)
        self._attendance_task: asyncio.Task | None = None
        self._stop_event = self._context.Event()
        self._display_process: mp.Process | None = None

    async def start_all(self) -> dict[str, Any]:
        async with self._lock:
            if self._processes:
                return {
                    "started": [],
                    "alreadyRunning": list(
                        self._processes.keys()
                    ),
                }
            
            if (
            self._attendance_task is None
            or self._attendance_task.done()
            ):
                self._attendance_task = asyncio.create_task(
                    self._consume_attendance()
                )

            embedding_index = await (
                self.embedding_service
                .refresh_all_embeddings_index()
            )

            cameras = await (
                self.sync_service.get_all_cameras()
            )

            started = await  self._start_processes(
                cameras,
                embedding_index,
            )

            return {
                "started": started,
                "count": len(started),
                "embeddingCount": len(
                    embedding_index.items
                ),
            }

    async def restart_all(self) -> dict[str, Any]:
        async with self._lock:
            process_items = list(
                self._processes.items()
            )
            self._processes.clear()

            stopped = await asyncio.to_thread(
                self._stop_processes,
                process_items,
            )

            # Load the newest employee embeddings.
            embedding_index = await (
                self.embedding_service
                .refresh_all_embeddings_index()
            )

            # Load the newest synchronized camera configurations.
            cameras = await (
                self.sync_service.get_all_cameras()
            )

            started = await  self._start_processes(
                cameras,
                embedding_index,
            )

            return {
                "stopped": stopped,
                "started": started,
                "embeddingCount": len(
                    embedding_index.items
                ),
            }

    async def stop_all(self) -> dict[str, Any]:
        async with self._lock:
            process_items = list(
                self._processes.items()
            )
            self._processes.clear()

            stopped = await asyncio.to_thread(
                self._stop_processes,
                process_items,
            )
            self._stop_event.set()

            try:
                self._attendance_queue.put_nowait(None)
            except Full:
                pass

            if self._attendance_task is not None:
                await self._attendance_task
                self._attendance_task = None

            return {
                "stopped": stopped,
                "count": len(stopped),
            }

    def statuses(self) -> list[dict[str, Any]]:
        return [
            {
                "cameraId": camera_id,
                "pid": process.pid,
                "alive": process.is_alive(),
                "exitCode": process.exitcode,
            }
            for camera_id, process
            in self._processes.items()
        ]

    async def _start_processes(
    self,
    cameras: list[CameraConfig],
    embedding_index: EmbeddingIndex,
    ) -> list[str]:
        started: list[str] = []

        enabled_cameras = [
            camera
            for camera in cameras
            if camera.enabled
        ]

        if not enabled_cameras:
            logger.info("No enabled cameras found")
            return started

        # Reset the shared stop signal.
        self._stop_event.clear()

        # Create one queue for each enabled camera.
        self._frame_queues = {
            camera.cameraId: self._context.Queue(maxsize=1)
            for camera in enabled_cameras
        }

        # Start every camera process.
        for camera in enabled_cameras:
            camera_data = camera.model_dump(
                by_alias=True,
                mode="json",
            )
            rule = await self.sync_service.get_rule_by_etsAuth(camera.etsAuth)

            process = self._context.Process(
                target=read_camera,
                args=(
                    camera_data,
                    embedding_index,
                    rule,
                    self._frame_queues[camera.cameraId],
                    self._attendance_queue,
                    self._log_queue,
                    self._stop_event,
                ),
                name=f"camera-{camera.cameraId}",
                daemon=False,
            )

            process.start()

            self._processes[camera.cameraId] = process
            started.append(camera.cameraId)

            logger.info(
                "Started camera %s with PID %s",
                camera.cameraId,
                process.pid,
            )

        camera_ids = [
            camera.cameraId
            for camera in enabled_cameras
        ]

        self._display_process = self._context.Process(
            target=show_camera_grid,
            args=(
                camera_ids,
                self._frame_queues,
                self._stop_event,
            ),
            name="camera-grid",
            daemon=False,
        )

        self._display_process.start()

        logger.info(
            "Started camera grid with PID %s",
            self._display_process.pid,
        )

        return started
    @staticmethod
    def _stop_processes(
        process_items: list[
            tuple[str, mp.Process]
        ],
    ) -> list[str]:
        stopped: list[str] = []

        # Signal all processes first.
        for camera_id, process in process_items:
            if process.is_alive():
                logger.info(
                    "Stopping camera %s",
                    camera_id,
                )
                process.terminate()

        # Then wait for every process.
        for camera_id, process in process_items:
            process.join(timeout=5)

            if process.is_alive():
                logger.warning(
                    "Camera %s did not stop within 5 seconds",
                    camera_id,
                )
                continue

            stopped.append(camera_id)

            logger.info(
                "Stopped camera %s",
                camera_id,
            )

        return stopped
    async def _consume_attendance(self) -> None:
        logger.info("Attendance consumer started")

        while True:
            try:
                event = await asyncio.to_thread(
                    self._attendance_queue.get,
                    True,
                    1,
                )
            except Empty:
                if self._stop_event.is_set():
                    break

                continue

            if event is None:
                break

            try:
                print("result ---->" + event["etsAuth"]+ "  " +event["cameraId"])
                result = await (
                    self.attendance_service
                    .record_attendance_if_allowed(
                        ets_auth=event["etsAuth"],
                        camera_id=event["cameraId"],
                        camera_direction=event[
                            "cameraDirection"
                        ],
                        employee_id=event["employeeId"],
                        confidence=event["confidence"],
                        rule=event["rule"],
                        snapshot_path=event["snapshotPath"],
                    )
                )

                if result.get("created"):
                    logger.info(
                        "Attendance created: "
                        "employee=%s etsAuth=%s camera=%s "
                        "reason=%s",
                        event["employeeId"],
                        event["etsAuth"],
                        event["cameraId"],
                        result.get("reason"),
                    )
                else:
                    logger.warning(
                        "Attendance skipped: "
                        "employee=%s etsAuth=%s camera=%s "
                        "reason=%s details=%s",
                        event["employeeId"],
                        event["etsAuth"],
                        event["cameraId"],
                        result.get("reason"),
                        result.get("details"),
                    )
                

            except Exception:
                logger.exception(
                    "Failed to create attendance"
                )

        logger.info("Attendance consumer stopped")