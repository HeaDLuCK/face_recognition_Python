import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2

from app.config import get_settings
from app.face.insightface_engine import InsightFaceEngine


def build_channel_url(base_url: str, channel: str | None) -> str:
    if not channel:
        return base_url

    marker = "/Streaming/Channels/"
    if marker not in base_url:
        return base_url

    parts = urlsplit(base_url)
    path = parts.path.rsplit("/", 1)[0] + f"/{channel}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def draw_faces(frame, detections, processed_at: float) -> None:
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            frame,
            f"face {index} score {detection.detectionScore:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"faces: {len(detections)}   last AI: {processed_at:.1f}s",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a debug window showing raw InsightFace detections.")
    parser.add_argument("--video", default="", help="Local video file to test instead of RTSP.")
    parser.add_argument("--url", default="", help="RTSP URL. Defaults to RTSP_URL from .env.")
    parser.add_argument("--channel", default="", help="Optional Hikvision channel, for example 101 or 102.")
    parser.add_argument("--every", type=int, default=10, help="Run AI every N frames.")
    parser.add_argument("--resize-width", type=int, default=0, help="Resize display width only. 0 keeps original.")
    parser.add_argument("--loop", action="store_true", help="Loop local video files.")
    parser.add_argument("--plain", action="store_true", help="Show video only, without face boxes or text.")
    parser.add_argument("--fast", action="store_true", help="Do not pace local video playback to its FPS.")
    parser.add_argument("--display-delay", type=int, default=0, help="Delay displayed video by N frames to sync boxes.")
    args = parser.parse_args()

    settings = get_settings()
    source = args.video or build_channel_url(args.url or settings.rtsp_url, args.channel or None)
    if not source:
        raise SystemExit("No RTSP URL found. Send --url or set RTSP_URL in .env.")
    if args.video and not Path(args.video).exists():
        raise SystemExit(f"Video file does not exist: {args.video}")

    engine = InsightFaceEngine(settings)
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise SystemExit(f"Unable to open source: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) if args.video else 0
    frame_delay = 1.0 / source_fps if source_fps and source_fps > 1 else 0
    frame_count = 0
    ai_future = None
    display_buffer = deque(maxlen=max(args.display_delay + 1, 1))
    detections_by_frame = {}
    executor = ThreadPoolExecutor(max_workers=1)
    window_name = "Face AI Debug - press q to quit"
    print(f"Opened {source}")
    print(f"AI runs every {args.every} frame(s). Press q in the video window to quit.")

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                if args.video and args.loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if args.video:
                    print("End of video.")
                    break
                print("Frame read failed. Reconnecting may be needed.")
                time.sleep(0.2)
                continue

            frame_count += 1
            current_frame_id = frame_count
            display_buffer.append((current_frame_id, frame.copy()))
            display_frame_id, display_frame = display_buffer[0] if len(display_buffer) > args.display_delay else (current_frame_id, frame.copy())

            if not args.plain and ai_future is not None and ai_future.done():
                result_frame_id, result_detections, result_processed_at = ai_future.result()
                detections_by_frame[result_frame_id] = (result_detections, result_processed_at)
                oldest_frame_id = display_buffer[0][0] if display_buffer else result_frame_id
                detections_by_frame = {
                    frame_id: value
                    for frame_id, value in detections_by_frame.items()
                    if frame_id >= oldest_frame_id
                }
                print(
                    f"ai_frame={result_frame_id} display_frame={display_frame_id} "
                    f"faces={len(result_detections)} scores={[round(face.detectionScore, 3) for face in result_detections]}"
                )
                ai_future = None

            if not args.plain and frame_count % max(args.every, 1) == 0 and ai_future is None:
                ai_frame_id = current_frame_id
                ai_frame = frame.copy()

                def process_frame():
                    started = time.perf_counter()
                    return ai_frame_id, engine.detect_faces(ai_frame), time.perf_counter() - started

                ai_future = executor.submit(process_frame)

            if not args.plain:
                detections, processed_at = detections_by_frame.get(display_frame_id, ([], 0.0))
                draw_faces(display_frame, detections, processed_at)

            if args.resize_width and display_frame.shape[1] > args.resize_width:
                ratio = args.resize_width / display_frame.shape[1]
                display_frame = cv2.resize(display_frame, (args.resize_width, int(display_frame.shape[0] * ratio)))

            cv2.imshow(window_name, display_frame)
            if args.video and frame_delay and not args.fast:
                elapsed = time.perf_counter() - frame_started
                remaining = frame_delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
