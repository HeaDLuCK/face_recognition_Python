from dataclasses import dataclass
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from config import get_settings

@dataclass
class DetectedFace:
    bbox: list[int]
    embedding: list[float]
    detectionScore: float


class InsightFaceEngine:
    def __init__(self,) -> None:
            settings = get_settings()
            self.min_score = settings.face_detection_min_score
            self.model = FaceAnalysis(
                name=settings.insightface_model_name,
                providers=settings.insightface_provider_list or ["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
    
            self.model.prepare(
                ctx_id=settings.insightface_ctx_id,
                det_size=(settings.insightface_det_size, settings.insightface_det_size),
                det_thresh=settings.face_detection_min_score,
            )
    
    
    def detect_faces(
        self,
        frame: np.ndarray,
    ) -> list[DetectedFace]:
        detected_faces = []

        for face in self.model.get(frame):
            if face.det_score < self.min_score:
                continue

            if face.normed_embedding is None:
                continue

            detected_faces.append(
                DetectedFace(
                    bbox=face.bbox.astype(int).tolist(),
                    embedding=face.normed_embedding.astype(float).tolist(),
                    detectionScore=float(face.det_score),
                )
            )

        return detected_faces
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