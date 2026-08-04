import logging
from typing import Any

import cv2
import numpy as np

from service.attendance_service import AttendanceService
from camera.camera_manager import CameraManager
from face_recognition import InsightFaceEngine
from schemas.project_schema import CameraConfig
from service.embedding_index import (
    EmbeddingIndex,
    best_embedding_candidate,
)
from queue import Empty, Full



logger = logging.getLogger(__name__)


def read_camera(
    camera_data: dict,
    embedding_index: EmbeddingIndex,
    frame_queue: Any | None = None,
    attendance_queue: Any | None = None,
    stop_event: Any | None = None,
) -> None:
    cv2.setNumThreads(1)

    camera_config = CameraConfig.model_validate(camera_data)
    camera = CameraManager(camera_config)

    engine = InsightFaceEngine(
        model_name="buffalo_s",
        providers=["CPUExecutionProvider"],
        ctx_id=-1,
        det_size=640,
        min_score=0.6,
    )

    capture = camera.start_camera()
    frame_number = 0
    detected_faces = []

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

            if frame_number % 4 == 0:
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
                    try:
                        attendance_queue.put_nowait(
                            {
                                "etsAuth": match["etsAuth"],
                                "cameraId": camera_config.cameraId,
                                "cameraDirection": camera_config.direction,
                                "employeeId": match["employeeId"],
                                "employeeName": employee_name,
                                "confidence": float(match["score"]),
                            }
                        )
                    except Full:
                        logger.warning(
                            "Attendance queue is full"
                        )
                    # attendance_service.record_attendance_if_allowed(
                    #     tsAuth=match.get("etsAuth"),
                    #     camera_id=camera.cameraId,
                    #     camera_direction="TEST",
                    #     employee_id=match.get("employeeName"),
                    #     confidence=match['score'],
                    #     rules="",
                    #     snapshot_path="",
                    #     metadata="",
                    # )

                    
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

            if frame_queue is not None:
                display_frame = cv2.resize(
                    frame,
                    (1280, 720),
                    interpolation=cv2.INTER_AREA,
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