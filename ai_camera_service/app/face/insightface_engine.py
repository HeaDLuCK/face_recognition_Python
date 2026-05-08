import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: list[int]
    embedding: list[float]
    detectionScore: float


class InsightFaceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is not installed. Install requirements.txt before using recognition."
            ) from exc

        model = FaceAnalysis(
            name=self.settings.insightface_model_name,
            providers=self.settings.insightface_provider_list,
        )
        model.prepare(ctx_id=self.settings.insightface_ctx_id, det_size=(640, 640))
        self._model = model
        logger.info("Loaded InsightFace model '%s'", self.settings.insightface_model_name)
        return model

    def detect_faces(self, frame: np.ndarray) -> list[DetectedFace]:
        model = self._load_model()
        faces = model.get(frame)
        detections: list[DetectedFace] = []

        for face in faces:
            embedding = self._normalize(face.embedding)
            bbox = [int(value) for value in face.bbox.tolist()]
            detections.append(
                DetectedFace(
                    bbox=bbox,
                    embedding=embedding.tolist(),
                    detectionScore=float(face.det_score),
                )
            )

        return detections

    def extract_embeddings_from_image_bytes(self, image_bytes: bytes) -> list[DetectedFace]:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode image")
        return self.detect_faces(frame)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
