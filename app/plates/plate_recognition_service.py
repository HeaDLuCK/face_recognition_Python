import logging
from pathlib import Path
from threading import RLock

import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


class PlateRecognitionService:
    MODEL_PATH = Path("app/plates/model/moroccan_plate.pt")

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._model_lock = RLock()

    @property
    def model_path(self) -> Path:
        return self.MODEL_PATH

    def recognize_frame(self, frame: np.ndarray) -> list[dict]:
        with self._model_lock:
            model = self._load_model()
            predictions = model.predict(
                source=frame,
                conf=self.settings.plate_yolo_confidence,
                imgsz=self.settings.plate_yolo_imgsz,
                device=self.settings.plate_yolo_device,
                verbose=False,
            )

            raw_detections = self._raw_detections(predictions)
        if self.settings.plate_yolo_mode == "characters":
            return self._character_detections_to_plate(raw_detections)
        return self._plate_box_detections(raw_detections)

    def _load_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = self.model_path
            if not model_path.exists():
                raise RuntimeError(f"Plate YOLO model does not exist: {model_path}")

            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from exc

            self._model = YOLO(str(model_path))
            logger.info("Loaded plate YOLO model: %s", model_path)
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
                if confidence < self.settings.plate_yolo_confidence:
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

        return self._dedupe_detections(detections)

    def _plate_box_detections(self, detections: list[dict]) -> list[dict]:
        results = []
        for detection in detections[: self.settings.plate_max_detections]:
            result = dict(detection)
            result["plateText"] = self._plate_text_from_class_name(result["className"])
            results.append(result)
        return results

    def _character_detections_to_plate(self, detections: list[dict]) -> list[dict]:
        characters = [
            detection
            for detection in detections
            if self._character_from_class_name(detection["className"]) is not None
        ]
        if len(characters) < self.settings.plate_min_characters:
            return []

        characters.sort(key=lambda item: ((item["bbox"][1] + item["bbox"][3]) // 2, item["bbox"][0]))
        plate_text = "".join(self._character_from_class_name(item["className"]) or "" for item in characters)
        plate_text = self._normalize_plate_text(plate_text)
        if len(plate_text) < self.settings.plate_min_characters:
            return []

        x1 = min(item["bbox"][0] for item in characters)
        y1 = min(item["bbox"][1] for item in characters)
        x2 = max(item["bbox"][2] for item in characters)
        y2 = max(item["bbox"][3] for item in characters)
        confidence = min(item["confidence"] for item in characters)

        return [
            {
                "bbox": [x1, y1, x2, y2],
                "detectionScore": confidence,
                "confidence": confidence,
                "plateText": plate_text,
                "characters": [
                    {
                        "bbox": item["bbox"],
                        "className": item["className"],
                        "text": self._character_from_class_name(item["className"]),
                        "confidence": item["confidence"],
                    }
                    for item in characters
                ],
                "source": "yolo-characters",
            }
        ]

    def _plate_text_from_class_name(self, class_name: str) -> str | None:
        if not self.settings.plate_yolo_class_is_text:
            return None
        text = self._normalize_plate_text(class_name)
        return text if len(text) >= self.settings.plate_min_characters else None

    @staticmethod
    def _character_from_class_name(class_name: str) -> str | None:
        text = PlateRecognitionService._normalize_plate_text(class_name)
        if len(text) == 1:
            return text
        if text.startswith("CHAR") and len(text) == 5:
            return text[-1]
        if text.startswith("DIGIT") and len(text) == 6:
            return text[-1]
        return None

    @staticmethod
    def _normalize_plate_text(text: str) -> str:
        return "".join(char for char in str(text).upper() if char.isalnum())

    @staticmethod
    def _dedupe_detections(detections: list[dict]) -> list[dict]:
        selected = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            if any(PlateRecognitionService._iou(detection["bbox"], existing["bbox"]) >= 0.45 for existing in selected):
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
