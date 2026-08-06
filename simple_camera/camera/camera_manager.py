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


    def getVideoSource(self,raw_source) -> str:
        if isinstance(raw_source, str):
            raw_source = raw_source.strip()

        # Accept "usb://0"
        if (
            isinstance(raw_source, str)
            and raw_source.lower().startswith("usb://")
        ):
            index_text = raw_source.split("://", 1)[1]

            if not index_text.isdigit():
                raise ValueError(
                    f"Invalid USB source: {raw_source}"
                )

            source: int | str = int(index_text)

        elif (
            isinstance(raw_source, str)
            and raw_source.isdigit()
        ):
            source = int(raw_source)

        else:
            source = raw_source
        return source
    
    def start_camera(
        self,
        open_timeout_ms: int = 10_000,
        read_timeout_ms: int = 3_000,
    ) -> cv2.VideoCapture:
        if not self.enabled:
            raise RuntimeError(
                f"Camera {self.camera_id} is disabled"
            )
        print(f"Starting camera {self.camera_id} ({self.rtsp_url})")
        capture = cv2.VideoCapture(self.getVideoSource(self.rtsp_url),)

        if not capture.isOpened():
            capture.release()

            raise RuntimeError(
                f"Unable to open camera: "
                f"{self.camera_id}"
            )

        return capture