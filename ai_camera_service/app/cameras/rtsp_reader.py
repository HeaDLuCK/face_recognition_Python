import logging

import cv2

from app.config import Settings

logger = logging.getLogger(__name__)


class RtspReader:
    def __init__(self, camera_source: str, settings: Settings):
        self.camera_source = camera_source
        self.settings = settings
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        resolved_source = self._resolve_source()
        self.capture = cv2.VideoCapture(resolved_source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.camera_source}")
        logger.info("Opened camera source: %s", self.camera_source)

    def read(self):
        if self.capture is None:
            self.open()
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Unable to read frame from camera source")
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            logger.info("Closed camera source")

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
