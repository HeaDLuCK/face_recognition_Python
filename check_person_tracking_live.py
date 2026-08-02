import argparse
import sys
import time
from datetime import datetime
<<<<<<< HEAD
=======
from pathlib import Path
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054

import cv2
import numpy as np

from app.config import get_settings
from app.tracking.person_detection_service import PersonDetectionService
from app.tracking.person_tracker import MotionActivityGate, PersonTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test live person detection/tracking with the same YOLO model used by the API."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or RTSP URL. Default: 0",
    )
    parser.add_argument(
<<<<<<< HEAD
=======
        "--model",
        default=None,
        help="Override PERSON_YOLO_MODEL_PATH. Default comes from .env.",
    )
    parser.add_argument(
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
        "--conf",
        type=float,
        default=None,
        help="Override PERSON_YOLO_CONFIDENCE.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override PERSON_YOLO_DEVICE, for example cpu or 0.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between YOLO person detections. Higher is lighter. Default: 0.5",
    )
    parser.add_argument(
        "--no-motion-gate",
        action="store_true",
        help="Run YOLO every interval even if no motion is detected.",
    )
    return parser.parse_args()


def parse_source(raw_source: str):
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def draw_body_marker(frame, bbox: tuple[int, int, int, int], blink_on: bool) -> None:
    if not blink_on:
        return

    x1, y1, x2, _ = bbox
    center_x = int((x1 + x2) / 2)
    marker_y = max(int(y1), 30)
    points = np.array(
        [
            [center_x, marker_y],
            [center_x - 14, marker_y - 26],
            [center_x + 14, marker_y - 26],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, points, (0, 220, 255))
    cv2.polylines(frame, [points], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "BODY",
        (center_x - 25, max(marker_y - 32, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, active_tracks: int, detections_run: int, fps: float) -> None:
    cv2.putText(
        frame,
        f"tracks={active_tracks} yolo_runs={detections_run} fps={fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
<<<<<<< HEAD
=======
    if args.model:
        settings.person_yolo_model_path = Path(args.model)
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    if args.conf is not None:
        settings.person_yolo_confidence = args.conf
    if args.device:
        settings.person_yolo_device = args.device

    detector = PersonDetectionService(settings)
    if not detector.available:
        raise RuntimeError(detector.unavailable_reason)

    tracker = PersonTracker(
        iou_threshold=settings.person_track_iou_threshold,
        timeout_seconds=settings.person_track_timeout_seconds,
    )
    motion_gate = MotionActivityGate(
        pixel_threshold=settings.person_motion_pixel_threshold,
        area_ratio=settings.person_motion_area_ratio,
        hold_seconds=settings.person_motion_hold_seconds,
    )

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    print("Press q or ESC to quit.")
<<<<<<< HEAD
    print(f"Model: {detector.model_path}")
=======
    print(f"Model: {settings.person_yolo_model_path}")
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    print(f"Source: {args.source}")

    last_detection_at = 0.0
    detections_run = 0
    fps_started_at = time.perf_counter()
    fps_frames = 0
    display_fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("No more frames.")
                break

            now = time.monotonic()
            observed_at = datetime.utcnow()
            tracker.expire(now)

            should_detect = now - last_detection_at >= args.interval
            if should_detect and not args.no_motion_gate:
                should_detect = motion_gate.update(frame, now)

            if should_detect:
                detections = detector.detect_frame(frame)
                tracker.update(detections, observed_at, now)
                detections_run += 1
                last_detection_at = now

            fps_frames += 1
            elapsed = time.perf_counter() - fps_started_at
            if elapsed >= 1.0:
                display_fps = fps_frames / elapsed
                fps_frames = 0
                fps_started_at = time.perf_counter()

            tracks = tracker.active_tracks()
            blink_on = int(time.monotonic() * 2) % 2 != 0
            for track in tracks:
                draw_body_marker(frame, track.bbox, blink_on)

            draw_status(frame, len(tracks), detections_run, display_fps)
            cv2.imshow("Person tracking test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
