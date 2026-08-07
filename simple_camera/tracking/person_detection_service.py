import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.tracking.models import PersonDetection


logger = logging.getLogger(__name__)


class PersonDetectionService:
    """
    Detect bodies in individual OpenCV frames.

    This service only detects people. It does not assign track IDs
    and it does not calculate movement.
    """

    MODEL_PATH = (
        Path(__file__).resolve().parent
        / "model"
        / "person_yolo.pt"
    )

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def model_path(self) -> Path:
        return self.MODEL_PATH

    @property
    def available(self) -> bool:
        return (
            self.model_path.is_file()
            and self._load_error is None
        )

    @property
    def unavailable_reason(self) -> str | None:
        if self._load_error:
            return self._load_error

        if not self.model_path.is_file():
            return (
                "Person detection model does not exist: "
                f"{self.model_path}"
            )

        return None

    async def detect(
        self,
        frame: np.ndarray,
    ) -> list[PersonDetection]:
        """
        Async wrapper for FastAPI or other asyncio code.
        """

        return await asyncio.to_thread(
            self.detect_frame,
            frame,
        )

    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> list[PersonDetection]:
        """
        Detect people in one frame.
        """

        if not isinstance(frame, np.ndarray):
            logger.warning(
                "Invalid frame type for person detection: %s",
                type(frame).__name__,
            )
            return []

        if frame.size == 0:
            logger.warning(
                "Empty frame received for person detection"
            )
            return []

        if frame.ndim != 3:
            logger.warning(
                "Invalid frame shape for person detection: %s",
                frame.shape,
            )
            return []

        frame_height, frame_width = frame.shape[:2]

        with self._model_lock:
            model = self._load_model()

            predictions = model.predict(
                source=frame,
                conf=self.settings.person_yolo_confidence,
                imgsz=self.settings.person_yolo_imgsz,
                device=self.settings.person_yolo_device,
                classes=[0],
                max_det=(
                    self.settings
                    .person_yolo_max_detections
                ),
                verbose=False,
            )

        detections: list[PersonDetection] = []

        for prediction in predictions:
            boxes = getattr(
                prediction,
                "boxes",
                None,
            )

            if boxes is None:
                continue

            if boxes.xyxy is None:
                continue

            if boxes.conf is None:
                continue

            xyxy_values = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            confidence_values = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for xyxy, confidence in zip(
                xyxy_values,
                confidence_values,
            ):
                if len(xyxy) < 4:
                    continue

                raw_x1, raw_y1, raw_x2, raw_y2 = (
                    xyxy.tolist()[:4]
                )

                x1 = max(
                    0,
                    min(
                        int(raw_x1),
                        frame_width - 1,
                    ),
                )

                y1 = max(
                    0,
                    min(
                        int(raw_y1),
                        frame_height - 1,
                    ),
                )

                x2 = max(
                    0,
                    min(
                        int(raw_x2),
                        frame_width,
                    ),
                )

                y2 = max(
                    0,
                    min(
                        int(raw_y2),
                        frame_height,
                    ),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    PersonDetection(
                        bbox=(
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                        confidence=float(
                            confidence
                        ),
                    )
                )

        return detections

    def warm_up(self) -> None:
        """
        Load the model before the first camera frame.
        """

        with self._model_lock:
            self._load_model()

    def _load_model(self) -> Any:
        """
        Load the model once and reuse it.
        """

        if self._model is not None:
            return self._model

        if self._load_error is not None:
            raise RuntimeError(
                self._load_error
            )

        if not self.model_path.is_file():
            self._load_error = (
                "Person detection model does not exist: "
                f"{self.model_path}"
            )

            raise RuntimeError(
                self._load_error
            )

        try:
            from ultralytics import YOLO

            self._model = YOLO(
                str(self.model_path)
            )

        except Exception as exc:
            self._load_error = (
                "Unable to load person detection model: "
                f"{exc}"
            )

            raise RuntimeError(
                self._load_error
            ) from exc

        logger.info(
            "Loaded person detection model: "
            "path=%s classes=%s",
            self.model_path,
            getattr(
                self._model,
                "names",
                None,
            ),
        )

        return self._model