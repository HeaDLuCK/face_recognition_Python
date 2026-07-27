import asyncio
import logging
import threading

import numpy as np

from app.config import Settings
from app.tracking.models import PersonDetection

logger = logging.getLogger(__name__)


class PersonDetectionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def model_path(self):
        return self.settings.person_yolo_model_path

    @property
    def available(self) -> bool:
        return self.model_path.is_file() and self._load_error is None

    @property
    def unavailable_reason(self) -> str | None:
        if self._load_error:
            return self._load_error
        if not self.model_path.is_file():
            return f"Person YOLO model does not exist: {self.model_path}"
        return None

    async def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        return await asyncio.to_thread(self.detect_frame, frame)

    def detect_frame(self, frame: np.ndarray) -> list[PersonDetection]:
        with self._model_lock:
            model = self._load_model()
            predictions = model.predict(
                source=frame,
                conf=self.settings.person_yolo_confidence,
                imgsz=self.settings.person_yolo_imgsz,
                device=self.settings.person_yolo_device,
                classes=[0],
                max_det=self.settings.person_yolo_max_detections,
                verbose=False,
            )

        detections: list[PersonDetection] = []
        for prediction in predictions:
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            xyxy_values = boxes.xyxy.cpu().numpy()
            confidence_values = boxes.conf.cpu().numpy()
            for xyxy, confidence in zip(xyxy_values, confidence_values):
                x1, y1, x2, y2 = (int(value) for value in xyxy.tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append(
                    PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(confidence),
                    )
                )
        return detections

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise RuntimeError(self._load_error)
        if not self.model_path.is_file():
            self._load_error = f"Person YOLO model does not exist: {self.model_path}"
            raise RuntimeError(self._load_error)

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        except Exception as exc:
            self._load_error = f"Unable to load person YOLO model: {exc}"
            raise RuntimeError(self._load_error) from exc

        logger.info("Loaded person YOLO model: %s", self.model_path)
        return self._model
