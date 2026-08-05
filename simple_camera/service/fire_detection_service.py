import logging
from pathlib import Path
from threading import RLock
from typing import Any
import numpy as np
import math
from config import Settings

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

    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> list[Any]:
        list_fire = []
        model = self._load_model()
        self.settings.fire_yolo_confidence,
        with self._model_lock:
            result  = model(frame, stream=True,verbose=False,)
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    if confidence > self.settings.fire_yolo_confidence:
                        list_fire.append(box.xyxy[0])
        return list_fire

    def _load_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = self.model_path

            if not model_path.exists():
                raise RuntimeError(
                    "Fire YOLO model does not exist: "
                    f"{model_path}"
                )

            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. "
                    "Run: pip install ultralytics"
                ) from exc

            self._model = YOLO(
                str(model_path)
            )

            logger.info(
                "Loaded fire YOLO model: %s",
                model_path,
            )

            return self._model

    