from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote, unquote, urlsplit
import uuid

from app.services.url_utils import redact_url_credentials

logger = logging.getLogger(__name__)

def export_hikvision_history_clip(
    camera_rtsp_url: str,
    timestamp: str,
    before_seconds: int,
    after_seconds: int,
    channel: str | None,
    output_dir: Path,
) -> str:
    info = parse_hikvision_rtsp(camera_rtsp_url)
    track_id = hikvision_main_stream_channel(channel or info["channel"])
    event_time = parse_event_time(timestamp)
    start = event_time - timedelta(seconds=before_seconds)
    end = event_time + timedelta(seconds=after_seconds)
    playback_url = hikvision_playback_url(
        host=info["host"],
        rtsp_port=info["rtsp_port"],
        username=info["username"],
        password=info["password"],
        channel=track_id,
        start=start,
        end=end,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / (
        f"{track_id}_{event_time.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}.mp4"
    )
    try:
        write_rtsp_clip(playback_url, clip_path, max((end - start).total_seconds(), 1))
    except Exception:
        delete_file_safely(clip_path)
        raise
    return str(clip_path)


def parse_hikvision_rtsp(rtsp_url: str) -> dict:
    parsed = urlsplit(rtsp_url)
    if parsed.scheme != "rtsp":
        raise ValueError("Camera rtspUrl must start with rtsp://")
    if not parsed.hostname:
        raise ValueError("Could not parse camera host from rtspUrl")
    if not parsed.username or not parsed.password:
        raise ValueError("Camera rtspUrl must include username and password")

    return {
        "username": unquote(parsed.username),
        "password": unquote(parsed.password),
        "host": parsed.hostname,
        "rtsp_port": parsed.port or 554,
        "channel": extract_hikvision_channel(parsed.path),
    }


def extract_hikvision_channel(path: str) -> str:
    if "/Streaming/Channels/" in path:
        return path.rsplit("/", 1)[-1]
    if "/Streaming/tracks/" in path:
        return path.rsplit("/", 1)[-1]
    return "101"


def hikvision_main_stream_channel(channel: str) -> str:
    """Return the recorded main-stream track for a Hikvision channel ID.

    Hikvision channel IDs use the final two digits as the stream selector:
    ``01`` is the main stream, while ``02`` and ``03`` are secondary streams.
    NVRs commonly record only ``01``, even when live recognition uses ``02``.
    """
    normalized = str(channel).strip()
    if normalized.isdigit() and len(normalized) >= 3:
        stream_id = normalized[-2:]
        if stream_id in {"02", "03"}:
            return f"{normalized[:-2]}01"
    return normalized


def parse_event_time(raw_timestamp: str) -> datetime:
    value = raw_timestamp.strip()
    if not value:
        raise ValueError("timestamp is required")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "timestamp must be ISO format, for example 2026-06-11T16:29:10Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.astimezone(timezone.utc)


def hikvision_playback_url(
    host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: str,
    start: datetime,
    end: datetime,
) -> str:
    safe_user = quote(username, safe="")
    safe_pass = quote(password, safe="")
    return (
        f"rtsp://{safe_user}:{safe_pass}@{host}:{rtsp_port}/Streaming/tracks/{channel}"
        f"?starttime={_rtsp_time(start)}&endtime={_rtsp_time(end)}"
    )


def write_rtsp_clip(
    playback_url: str,
    clip_path: Path,
    expected_seconds: float,
) -> None:
    ffmpeg_path = _find_ffmpeg()
    if ffmpeg_path:
        _write_rtsp_clip_ffmpeg(
            ffmpeg_path,
            playback_url,
            clip_path,
            expected_seconds,
        )
        return

    _write_rtsp_clip_opencv(playback_url, clip_path, expected_seconds)


def _find_ffmpeg() -> str | None:
    bundled = Path("ffmpeg") / "bin" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    return shutil.which("ffmpeg")


def _write_rtsp_clip_ffmpeg(
    ffmpeg_path: str,
    playback_url: str,
    clip_path: Path,
    expected_seconds: float,
) -> None:
    timeout_seconds = max(30, int(expected_seconds) + 45)
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        "10000000",
        "-i",
        playback_url,
        "-t",
        str(max(expected_seconds, 1)),
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(clip_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        delete_file_safely(clip_path)
        raise RuntimeError(
            f"Hikvision history export timed out after {timeout_seconds} seconds"
        ) from exc

    if result.returncode != 0:
        delete_file_safely(clip_path)
        error = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Hikvision history export failed"
            + (f": {error[-500:]}" if error else "")
        )

    if not clip_path.is_file() or clip_path.stat().st_size == 0:
        delete_file_safely(clip_path)
        raise RuntimeError("Hikvision history export returned an empty clip")


def _write_rtsp_clip_opencv(
    playback_url: str,
    clip_path: Path,
    expected_seconds: float,
) -> None:
    import cv2

    open_started_at = time.monotonic()
    capture = cv2.VideoCapture(
        playback_url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            10000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            10000,
        ],
    )
    open_elapsed = time.monotonic() - open_started_at
    if not capture.isOpened():
        capture.release()
        logger.warning(
            "RTSP_SOURCE=history_clip_opencv phase=open_failed stream=%s elapsed=%.2fs",
            redact_url_credentials(playback_url),
            open_elapsed,
        )
        raise RuntimeError(
            "No Hikvision recording/playback stream is available for that time window"
        )

    logger.info(
        "RTSP_SOURCE=history_clip_opencv phase=opened stream=%s elapsed=%.2fs",
        redact_url_credentials(playback_url),
        open_elapsed,
    )

    writer = None
    frames_written = 0
    started_at = time.monotonic()
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        output_fps = source_fps if source_fps and 1 <= source_fps <= 60 else 15
        max_runtime = expected_seconds + 15
        max_frames = int(output_fps * expected_seconds * 1.5) or 1

        while time.monotonic() - started_at < max_runtime and frames_written < max_frames:
            read_started_at = time.monotonic()
            ok, frame = capture.read()
            read_elapsed = time.monotonic() - read_started_at
            if not ok or frame is None:
                if read_elapsed >= 8.0:
                    logger.warning(
                        "RTSP_SOURCE=history_clip_opencv phase=read_failed stream=%s elapsed=%.2fs frames=%s",
                        redact_url_credentials(playback_url),
                        read_elapsed,
                        frames_written,
                    )
                if frames_written:
                    break
                time.sleep(0.02)
                continue

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(clip_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not create history clip file")

            writer.write(frame)
            frames_written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frames_written == 0:
        raise RuntimeError("No frames were returned by Hikvision for that time window")


def delete_file_safely(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _rtsp_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
