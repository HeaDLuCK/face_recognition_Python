import logging
import os
import time
from collections import deque
from datetime import datetime

import cv2

from app.config import Settings
from app.services.url_utils import redact_url_credentials

logger = logging.getLogger(__name__)


class RtspReader:
    def __init__(self, camera_source: str, settings: Settings):
        self.camera_source = camera_source
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None
        self._gap_started_at: datetime | None = None
        self._gap_requires_recovery = False
        self._recovered_gaps: deque[tuple[datetime, datetime]] = deque()
        self.read_failure_count = 0
        self.reconnect_count = 0
        self.last_read_failure_at: datetime | None = None
        self.last_reconnected_at: datetime | None = None

    def open(self) -> None:
        resolved_source = self._resolve_source()
        if isinstance(resolved_source, str) and resolved_source.startswith("rtsp://"):
            capture_options = (
                "rtsp_transport;tcp|fflags;nobuffer|max_delay;0"
                if self.settings.rtsp_low_latency
                else "rtsp_transport;tcp|fflags;+genpts|max_delay;500000"
            )
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = capture_options
            self.capture = cv2.VideoCapture(
                resolved_source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5000,
                ],
            )
        else:
            self.capture = cv2.VideoCapture(resolved_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self._safe_source()}")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1 if self.settings.rtsp_low_latency else 30)
        logger.info("Opened camera source: %s", self._safe_source())

    def read(self):
        if self.capture is None:
            self.open()

        reconnect_attempts = self.settings.rtsp_read_retries + 1
        for reconnect_attempt in range(reconnect_attempts):
            if self.capture is None:
                try:
                    self.open()
                except Exception as exc:
                    self._record_read_failure()
                    logger.warning(
                        "RTSP reconnect failed for %s (%s/%s): %s",
                        self._safe_source(),
                        reconnect_attempt + 1,
                        reconnect_attempts,
                        type(exc).__name__,
                    )
                    if reconnect_attempt + 1 < reconnect_attempts:
                        self._wait_before_reconnect()
                    continue

            frame = self._read_with_failure_tolerance()
            if frame is not None:
                self._finish_recovered_gap()
                return frame

            self._gap_requires_recovery = True
            self.reconnect_count += 1
            logger.warning(
                "RTSP read failed %s consecutive times for %s; reconnecting (%s/%s)",
                self.settings.rtsp_failed_reads_before_reconnect,
                self._safe_source(),
                reconnect_attempt + 1,
                reconnect_attempts,
            )
            self.close()
            if reconnect_attempt + 1 < reconnect_attempts:
                self._wait_before_reconnect()

        raise RuntimeError(f"Unable to read frame from camera source: {self._safe_source()}")

    def pop_recovered_gaps(self) -> list[tuple[datetime, datetime]]:
        gaps = list(self._recovered_gaps)
        self._recovered_gaps.clear()
        return gaps

    def status(self) -> dict:
        return {
            "readFailures": self.read_failure_count,
            "reconnects": self.reconnect_count,
            "lastReadFailureAt": (
                self.last_read_failure_at.isoformat()
                if self.last_read_failure_at
                else None
            ),
            "lastReconnectedAt": (
                self.last_reconnected_at.isoformat()
                if self.last_reconnected_at
                else None
            ),
            "currentGapStartedAt": (
                self._gap_started_at.isoformat()
                if self._gap_started_at
                else None
            ),
        }

    def _read_with_failure_tolerance(self):
        failure_limit = self.settings.rtsp_failed_reads_before_reconnect
        for failed_read in range(1, failure_limit + 1):
            read_started_at = time.monotonic()
            try:
                for _ in range(self.settings.rtsp_drop_stale_frames):
                    self.capture.grab()
                ok, frame = self.capture.read()
                if ok and frame is not None:
                    return frame
            except (cv2.error, AttributeError) as exc:
                logger.debug(
                    "OpenCV read error for %s: %s",
                    self._safe_source(),
                    type(exc).__name__,
                )

            logger.warning(
                "RTSP_SOURCE=live_reader phase=read_failed stream=%s attempt=%s/%s elapsed=%.2fs",
                self._safe_source(),
                failed_read,
                failure_limit,
                time.monotonic() - read_started_at,
            )

            self._record_read_failure()
            if failed_read < failure_limit:
                logger.debug(
                    "Transient RTSP read failure for %s (%s/%s)",
                    self._safe_source(),
                    failed_read,
                    failure_limit,
                )
                if self.settings.rtsp_failed_read_retry_delay_seconds:
                    time.sleep(
                        self.settings.rtsp_failed_read_retry_delay_seconds
                    )
        return None

    def _record_read_failure(self) -> None:
        now = datetime.utcnow()
        self.read_failure_count += 1
        self.last_read_failure_at = now
        if self._gap_started_at is None:
            self._gap_started_at = now

    def _finish_recovered_gap(self) -> None:
        if self._gap_started_at is None:
            return
        recovered_at = datetime.utcnow()
        if self._gap_requires_recovery:
            self._recovered_gaps.append((self._gap_started_at, recovered_at))
            self.last_reconnected_at = recovered_at
            logger.info(
                "RTSP stream recovered for %s after %.2f seconds",
                self._safe_source(),
                max((recovered_at - self._gap_started_at).total_seconds(), 0.0),
            )
        self._gap_started_at = None
        self._gap_requires_recovery = False

    def _wait_before_reconnect(self) -> None:
        if self.settings.rtsp_reconnect_delay_seconds:
            time.sleep(self.settings.rtsp_reconnect_delay_seconds)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            logger.info("Closed camera source: %s", self._safe_source())

    def _resolve_source(self) -> str | int:
        if self.settings.camera_source_mode == "usb":
            return self.settings.usb_camera_index
        if self.settings.camera_source_mode == "rtsp":
            return self.camera_source

        source = self.camera_source.strip()
        if source.startswith("usb://"):
            return int(source.replace("usb://", "", 1))
        if source.isdigit():
            return int(source)
        return source

    def _safe_source(self) -> str:
        return redact_url_credentials(self.camera_source)
