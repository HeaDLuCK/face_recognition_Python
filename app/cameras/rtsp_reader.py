import logging
import os
import time

import cv2

from app.config import Settings
from app.services.url_utils import redact_url_credentials

logger = logging.getLogger(__name__)


class RtspReader:
    def __init__(self, camera_source: str, settings: Settings):
        self.camera_source = camera_source
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None

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

        attempts = self.settings.rtsp_read_retries + 1
        for attempt in range(attempts):
            try:
                for _ in range(self.settings.rtsp_drop_stale_frames):
                    self.capture.grab()
                ok, frame = self.capture.read()
                if ok and frame is not None:
                    return frame
            except cv2.error as exc:
                logger.debug("OpenCV read error for %s: %s", self._safe_source(), type(exc).__name__)

            logger.warning(
                "RTSP read failed for %s (%s/%s)",
                self._safe_source(),
                attempt + 1,
                attempts,
            )
            self.close()
            if attempt + 1 >= attempts:
                break
            if self.settings.rtsp_reconnect_delay_seconds:
                time.sleep(self.settings.rtsp_reconnect_delay_seconds)
            self.open()

        raise RuntimeError(f"Unable to read frame from camera source: {self._safe_source()}")

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
