import logging
from typing import Any

import cv2
import numpy as np
from time import sleep

from camera.camera_manager import CameraManager
from face_recognition import InsightFaceEngine
from service.fire_detection_service import FireDetectionService
from schemas.project_schema import CameraConfig,AiCapability,AttendanceRules
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

from tracking.person_tracking_runtime import (PersonTrackingRuntime,)
from collections import deque


logger = logging.getLogger(__name__)


def read_camera(
    camera_data: dict,
    embedding_index: EmbeddingIndex,
    rule: AttendanceRules,
    frame_queue: Any | None = None,
    attendance_queue: Any | None = None,
    log_queue: Any | None = None,
    stop_event: Any | None = None,
) -> None:
    if log_queue is not None:
        configure_queue_logging(
            log_queue
        )

    settings = get_settings()

    cv2.setNumThreads(settings.opencv_num_threads)
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

    face_reco_engine: InsightFaceEngine | None = None
    fire_module: FireDetectionService | None = None
    person_tracking: PersonTrackingRuntime | None = None
    has_face_reco_module = camera_config.has_capability(AiCapability.FACE_RECOGNITION)
    has_fire_module = camera_config.has_capability(AiCapability.FIRE_DETECTION)
    has_tracking_module = camera_config.has_capability(AiCapability.PERSON_COUNTING)
    if has_face_reco_module:
        face_reco_engine = InsightFaceEngine()
    if has_fire_module:
        fire_module = FireDetectionService(settings)
    if has_tracking_module:
        person_tracking = PersonTrackingRuntime(settings)
    
    capture: cv2.VideoCapture | None = None

    frame_number = 0
    detected_faces = []

    presence_state: dict[
        tuple[str, str, str],
        dict,
    ] = {}

    recognition_state: dict[
        tuple[str, str, str],
        dict,
    ] = {}

    PRESENCE_RESET_SECONDS = 5.0
    QUEUE_RETRY_SECONDS = 0.5

    MAX_READ_FAILURES = 10
    READ_RETRY_DELAY_SECONDS = 0.1
    RECONNECT_DELAY_SECONDS = 2.0

    consecutive_read_failures = 0

    try:
        while (
            stop_event is None
            or not stop_event.is_set()
        ):
            # Open or reopen the camera.
            if (
                capture is None
                or not capture.isOpened()
            ):
                try:
                    capture = camera.start_camera()

                    consecutive_read_failures = 0

                    logger.info(
                        "Camera connected: camera=%s",
                        camera.camera_id,
                    )

                except Exception:
                    logger.exception(
                        "Unable to connect to camera: "
                        "camera=%s",
                        camera.camera_id,
                    )
                    sleep(RECONNECT_DELAY_SECONDS)
                    continue

            ret, frame = capture.read()

            if not ret or frame is None:
                consecutive_read_failures += 1

                logger.warning(
                    "Camera frame read failed: "
                    "camera=%s failure=%d/%d",
                    camera.camera_id,
                    consecutive_read_failures,
                    MAX_READ_FAILURES,
                )

                # Allow temporary decoding or packet-loss errors.
                if (
                    consecutive_read_failures
                    < MAX_READ_FAILURES
                ):
                    sleep(
                        READ_RETRY_DELAY_SECONDS
                    )
                    continue

                logger.error(
                    "Reconnecting camera after "
                    "%d consecutive read failures: "
                    "camera=%s",
                    consecutive_read_failures,
                    camera.camera_id,
                )

                capture.release()
                capture = None

                consecutive_read_failures = 0

                sleep(RECONNECT_DELAY_SECONDS)
                continue

            # A valid frame was received.
            consecutive_read_failures = 0
            frame_number += 1
            now = monotonic()
            expired_candidates = [
                candidate_key
                for candidate_key, state
                in recognition_state.items()
                if (
                    now - float(state["last_seen"])
                    > 2.0
                )
            ]

            for candidate_key in expired_candidates:
                del recognition_state[candidate_key]

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

            if (person_tracking is not None and frame_number % settings.camera_frame_skip == 0 ):
                try:
                    updated_tracks, expired_tracks = (
                        person_tracking.process_frame(
                            frame=frame,
                            draw=True,
                        )
                    )

                    for track in updated_tracks:
                        logger.debug(
                            "Person track updated: "
                            "camera=%s trackId=%s "
                            "moving=%s movement=%.2f "
                            "confidence=%.4f",
                            camera_config.cameraId,
                            track.track_id,
                            track.is_moving,
                            track.movement_distance,
                            track.confidence,
                        )

                    for track in expired_tracks:
                        logger.info(
                            "Person track expired: "
                            "camera=%s trackId=%s",
                            camera_config.cameraId,
                            track.track_id,
                        )

                except Exception:
                    logger.exception(
                        "Person tracking failed: camera=%s",
                        camera_config.cameraId,
                    )
            if (fire_module is not None and frame_number % settings.camera_frame_skip == 0 ):
                fire_detection:Any | None = None
                if frame_number % 20 == 0: 
                    fire_detection = fire_module.detect_frame(frame)
                if fire_detection != None:
                    for info in fire_detection:
                        x1, y1, x2, y2 = info
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cv2.putText(
                            frame,
                            "Fire",
                            (x1 + 8, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
            if (face_reco_engine is not None and frame_number % settings.camera_frame_skip == 0 ):
                detected_faces = (
                    face_reco_engine.detect_faces(frame)
                )

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
                    now = monotonic()

                    confirmed = False
                    confirmed_score = 0.0

                    if match is not None:
                        confirmed, confirmed_score = (
                            update_recognition_candidate(
                                match=match,
                                recognition_state=(recognition_state),
                                now=now,
                                threshold=(rule.recognitionThreshold),
                                window_seconds=(settings.recognition_candidate_window_seconds),
                                minimum_hits=(settings.recognition_candidate_min_hits),
                                fast_margin=(settings.recognition_candidate_fast_margin),
                                floor_margin=(settings.recognition_candidate_floor_margin),
                                hold_seconds=(settings.recognition_identity_hold_seconds),
                            )
                        )

                        logger.info(
                            "Recognition candidate: "
                            "camera=%s employee=%s "
                            "currentScore=%.4f bestScore=%.4f "
                            "threshold=%.4f confirmed=%s",
                            camera_config.cameraId,
                            match.get("employeeId"),
                            float(match["score"]),
                            confirmed_score,
                            settings.default_recognition_threshold,
                            confirmed,
                        )

                    label = "Unknown"
                    color = (0, 0, 255)
                    if confirmed and match is not None:
                        employee_name = (
                            match.get("employeeName")
                            or match["employeeId"]
                        )

                        label = (
                            f"{employee_name} "
                            f"{confirmed_score:.2f}"
                        )

                        color = (0, 255, 0)

                        camera_assignment = (
                            camera_config.assigned_for(
                                match["etsAuth"],
                                AiCapability.FACE_RECOGNITION,
                            )
                        )
                        if (
                            camera_assignment is not None
                            and camera_assignment.enabled
                        ):
                            person_key = (
                                camera_config.cameraId,
                                match["etsAuth"],
                                match["employeeId"],
                            )

                            state = presence_state.get(
                                person_key
                            )

                            if state is None:
                                state = {
                                    "last_seen": now,
                                    "queued": False,
                                    "next_retry": 0.0,
                                }

                                presence_state[
                                    person_key
                                ] = state
                            else:
                                state["last_seen"] = now
                            
                            if (
                                attendance_queue is not None
                                and not state["queued"]
                                and now
                                >= state["next_retry"]
                            ):
                                snapshot_path = (
                                    save_known_snapshot(
                                        snapshot_root=(known_snapshot_dir),
                                        frame=frame,
                                        bbox=(detected_face.bbox),
                                        camera_id=(camera_config.cameraId),
                                        ets_auth=(match["etsAuth"]),
                                        employee_id=(match["employeeId"]),
                                        employee_name=(employee_name),
                                        confidence=(confirmed_score),
                                    )
                                )

                                attendance_event = {
                                    "etsAuth": (
                                        match["etsAuth"]
                                    ),
                                    "cameraId": (
                                        camera_config.cameraId
                                    ),
                                    "rule": (
                                        rule
                                    ),
                                    "cameraDirection": (
                                        camera_assignment.direction
                                    ),
                                    "employeeId": (
                                        match["employeeId"]
                                    ),
                                    "employeeName": (
                                        employee_name
                                    ),
                                    "confidence": (
                                        confirmed_score
                                    ),
                                    "snapshotPath": (
                                        snapshot_path
                                    ),
                                }

                                try:
                                    attendance_queue.put_nowait(
                                        attendance_event
                                    )

                                except Full:
                                    state["next_retry"] = (
                                        now
                                        + QUEUE_RETRY_SECONDS
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
                                        "camera=%s employee=%s "
                                        "score=%.4f",
                                        camera_config.cameraId,
                                        match["employeeId"],
                                        confirmed_score,
                                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
            if frame_queue is not None:
                display_frame = resize_preserving_ratio( frame,
                    max_width=560,
                    max_height=315,
                )

                try:
                    send_latest_frame(frame_queue,display_frame,)
                except Exception:
                    logger.exception("Failed to send display frame: camera=%s",camera_config.cameraId,)
                    pass
    except Exception:
            logger.exception(
                "Camera %s worker failed",
                camera.camera_id,
            )
    finally:
        if capture is not None:
            capture.release()
            capture = None

        logger.info(
            "Camera %s stopped",
            camera.camera_id,
        )

def update_recognition_candidate(
    match: dict | None,
    recognition_state: dict[
        tuple[str, str],
        dict[str, Any],
    ],
    now: float,
    threshold: float,
    window_seconds: float,
    minimum_hits: int,
    fast_margin: float,
    floor_margin: float,
    hold_seconds: float,
) -> tuple[bool, float]:
    if match is None:
        return False, 0.0

    ets_auth = str(match["etsAuth"])
    employee_id = str(match["employeeId"])
    score = float(match["score"])

    candidate_key = (
        ets_auth,
        employee_id,
    )

    state = recognition_state.get(candidate_key)

    if state is None:
        state = {
            "samples": deque(),
            "confirmed_until": 0.0,
            "last_seen": now,
        }

        recognition_state[candidate_key] = state

    state["last_seen"] = now

    samples: deque = state["samples"]

    samples.append(
        (now, score)
    )

    # Remove scores outside the confirmation window.
    while (
        samples
        and now - samples[0][0]
        > window_seconds
    ):
        samples.popleft()

    candidate_floor = (
        threshold - floor_margin
    )

    # Only near-threshold scores count as useful hits.
    valid_scores = [
        sample_score
        for _, sample_score in samples
        if sample_score >= candidate_floor
    ]

    best_score = max(
        valid_scores,
        default=score,
    )

    # Very strong result: confirm immediately.
    fast_confirmation = (
        score >= threshold + fast_margin
    )

    # Normal result:
    # same candidate appears repeatedly and at least
    # one result reaches the normal threshold.
    sequence_confirmation = (
        len(valid_scores) >= minimum_hits
        and best_score >= threshold
    )

    if (
        fast_confirmation
        or sequence_confirmation
    ):
        state["confirmed_until"] = (
            now + hold_seconds
        )

    confirmed = (
        now
        <= float(state["confirmed_until"])
    )

    return confirmed, best_score


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