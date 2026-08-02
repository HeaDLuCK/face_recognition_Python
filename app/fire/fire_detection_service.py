import logging
from pathlib import Path
from threading import RLock

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class FireDetectionService:
    MODEL_PATH = Path("app/fire/model/fire_model.pt")

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_lock = RLock()

    @property
    def model_path(self) -> Path:
        return self.MODEL_PATH

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        with self._model_lock:
            model = self._load_model()
            predictions = model.predict(
                source=frame,
                conf=self.settings.fire_yolo_confidence,
                imgsz=self.settings.fire_yolo_imgsz,
                device=self.settings.fire_yolo_device,
                verbose=False,
            )
            return self._dedupe_detections(self._raw_detections(predictions))[
                : self.settings.fire_max_detections
            ]

    def _load_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = self.model_path
            if not model_path.exists():
                raise RuntimeError(f"Fire YOLO model does not exist: {model_path}")

            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

            self._model = YOLO(str(model_path))
            logger.info("Loaded fire YOLO model: %s", model_path)
            return self._model

    def _raw_detections(self, predictions) -> list[dict]:
        model = self._load_model()
        detections = []

        for prediction in predictions:
            for box in prediction.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = str(model.names.get(class_id, class_id))
                if confidence < self.settings.fire_yolo_confidence:
                    continue
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "detectionScore": confidence,
                        "confidence": confidence,
                        "classId": class_id,
                        "className": class_name,
                        "source": "yolo",
                    }
                )

        return detections

    @staticmethod
    def _dedupe_detections(detections: list[dict]) -> list[dict]:
        selected = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            if any(FireDetectionService._iou(detection["bbox"], existing["bbox"]) >= 0.45 for existing in selected):
                continue
            selected.append(detection)
        return selected

    @staticmethod
    def _iou(first: list[int], second: list[int]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
        first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
        second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union
