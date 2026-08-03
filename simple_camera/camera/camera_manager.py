import cv2

from schemas.project_schema import CameraConfig


class CameraManager:
    def __init__(
        self,
        config: CameraConfig,
    ) -> None:
        self.config = config

    @property
    def camera_id(self) -> str:
        return self.config.cameraId

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def rtsp_url(self) -> str:
        return self.config.rtspUrl

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start_camera(
        self,
        open_timeout_ms: int = 10_000,
        read_timeout_ms: int = 3_000,
    ) -> cv2.VideoCapture:
        if not self.enabled:
            raise RuntimeError(
                f"Camera {self.camera_id} is disabled"
            )

        capture = cv2.VideoCapture(
            self.rtsp_url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                open_timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                read_timeout_ms,
            ],
        )

        if not capture.isOpened():
            capture.release()

            raise RuntimeError(
                f"Unable to open camera: "
                f"{self.camera_id}"
            )

        return capture