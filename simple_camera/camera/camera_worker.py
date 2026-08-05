import logging
from typing import Any

import cv2
import numpy as np

from camera.camera_manager import CameraManager
from face_recognition import InsightFaceEngine
from schemas.project_schema import CameraConfig,AiCapability
from service.embedding_index import (
    EmbeddingIndex,
    best_embedding_candidate,
)
from queue import Empty, Full
from time import monotonic
from config import get_settings  # adjust import path
from pathlib import Path
from datetime import datetime, timezone
from logging_setup import (
    configure_queue_logging,
)

logger = logging.getLogger(__name__)


def read_camera(
    camera_data: dict,
    embedding_index: EmbeddingIndex,
    frame_queue: Any | None = None,
    attendance_queue: Any | None = None,
    log_queue: Any | None = None,
    stop_event: Any | None = None,
) -> None:
    if log_queue is not None:
        configure_queue_logging(
            log_queue
        )

    cv2.setNumThreads(1)

    settings = get_settings()

    known_snapshot_dir = (settings.snapshot_dir)

    known_snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera_config = CameraConfig.model_validate(camera_data)
    logger.info(
        "Camera worker starting: camera=%s",
        camera_config.cameraId,
    )
    camera = CameraManager(camera_config)
    engine = InsightFaceEngine(
        model_name="buffalo_s",
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        det_size=640,
        min_score=0.6,
    )
    has_capability = camera_config.has_capability(AiCapability.FACE_RECOGNITION)
    capture = camera.start_camera()
    frame_number = 0
    detected_faces = []
    presence_state: dict[tuple[str, str, str], dict] = {}
    PRESENCE_RESET_SECONDS = 5.0
    QUEUE_RETRY_SECONDS = 0.5
    try:
        while stop_event is None or not stop_event.is_set():
            ret, frame = capture.read()

            if not ret or frame is None:
                logger.error(
                    "Camera %s failed to read frame",
                    camera.camera_id,
                )
                break

            frame_number += 1

            if frame_number % 4 == 0 and has_capability:
                detected_faces = engine.detect_faces(frame)

            for detected_face in detected_faces:
                detected_embedding = np.asarray(
                    detected_face.embedding,
                    dtype=np.float32,
                )

                match = best_embedding_candidate(
                    embedding_index,
                    detected_embedding,
                )

                x1, y1, x2, y2 = map(
                    int,
                    detected_face.bbox,
                )

                label = "Unknown"

                if (
                    match is not None
                    and match["score"] >= 0.50
                ):
                    employee_name = (
                        match.get("employeeName")
                        or match["employeeId"]
                    )

                    label = (
                        f"{employee_name} "
                        f"{match['score']:.2f}"
                    )
                    camera_assignment = camera_config.assigned_for(match["etsAuth"],AiCapability.FACE_RECOGNITION)
                    if (
                        camera_assignment is not None
                        and camera_assignment.enabled
                    ):
                        now = monotonic()
                        person_key = (
                            camera_config.cameraId,
                            match["etsAuth"],
                            match["employeeId"],
                        )
                        state = presence_state.get(person_key)

                        if state is None:
                            state = {
                                "last_seen": now,
                                "queued": False,
                                "next_retry": 0.0,
                            }

                            presence_state[person_key] = state
                        else:
                            state["last_seen"] = now
                        if (
                            not state["queued"]
                            and now >= state["next_retry"]
                        ):
                            snapshot_path = save_known_snapshot(
                                            snapshot_root=known_snapshot_dir,
                                            frame=frame,
                                            bbox=detected_face.bbox,
                                            camera_id=camera_config.cameraId,
                                            ets_auth=match["etsAuth"],
                                            employee_id=match["employeeId"],
                                            employee_name=employee_name,
                                            confidence=float(match["score"]),
                                        ) 
                            attendance_event = {
                                "etsAuth": match["etsAuth"],
                                "cameraId": camera_config.cameraId,
                                "cameraDirection": (
                                    camera_assignment.direction
                                ),
                                "employeeId": match["employeeId"],
                                "employeeName": employee_name,
                                "confidence": float(match["score"]),
                                "snapshotPath": snapshot_path,
                            }

                            try:
                                attendance_queue.put_nowait(
                                    attendance_event
                                )
                                  
                            except Full:
                                state["next_retry"] = (
                                    now + QUEUE_RETRY_SECONDS
                                )

                                logger.warning(
                                    "Attendance queue full: "
                                    "camera=%s employee=%s",
                                    camera_config.cameraId,
                                    match["employeeId"],
                                )

                            else:
                                state["queued"] = True
                                logger.info(
                                    "Attendance event queued: "
                                    "camera=%s employee=%s",
                                    camera_config.cameraId,
                                    match["employeeId"],
                                )   
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            # Cleanup goes after the face loop.
            now = monotonic()

            expired_people = [
                person_key
                for person_key, state
                in presence_state.items()
                if (
                    now - state["last_seen"]
                    >= PRESENCE_RESET_SECONDS
                )
            ]

            for person_key in expired_people:
                del presence_state[person_key]

            if frame_queue is not None:
                display_frame = resize_preserving_ratio( frame,
                    max_width=560,
                    max_height=315,
                )

                try:
                   send_latest_frame(
                        frame_queue,
                        display_frame,
                    )
                except Exception:
                    pass

    except Exception:
        logger.exception(
            "Camera %s worker failed",
            camera.camera_id,
        )

    finally:
        capture.release()

        logger.info(
            "Camera %s stopped",
            camera.camera_id,
        )

def send_latest_frame(
    frame_queue: Any,
    frame: np.ndarray,
) -> None:
    try:
        frame_queue.put_nowait(frame)
        return
    except Full:
        pass

    # Remove the previous frame.
    try:
        frame_queue.get_nowait()
    except Empty:
        pass

    # Insert the newest frame.
    try:
        frame_queue.put_nowait(frame)
    except Full:
        pass

def resize_preserving_ratio(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    height, width = frame.shape[:2]

    scale = min(
        max_width / width,
        max_height / height,
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=interpolation,
    )



def _safe_filename_part(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in value
    )

def save_known_snapshot(
    snapshot_root: Path,
    frame: np.ndarray,
    bbox: Any,
    camera_id: str,
    ets_auth: str,
    employee_id: str,
    employee_name: str,
    confidence: float,
) -> str | None:
    now = datetime.now(timezone.utc)

    directory = (
        snapshot_root
        / now.strftime("%Y-%m-%d")
        / _safe_filename_part(ets_auth)
        / _safe_filename_part(camera_id)
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{_safe_filename_part(employee_id)}_"
        f"{now.strftime('%H%M%S_%f')}.jpg"
    )

    snapshot_path = directory / filename

    snapshot = frame.copy()

    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(
        snapshot,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        snapshot,
        f"{employee_name} {confidence:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    saved = cv2.imwrite(
        str(snapshot_path),
        snapshot,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    if not saved:
        logger.error(
            "Failed to save snapshot: camera=%s employee=%s",
            camera_id,
            employee_id,
        )
        return None

    return str(snapshot_path.resolve())