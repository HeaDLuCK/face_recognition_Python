import numpy as np

from app.face.embedding_service import EmbeddingService
from app.face.insightface_engine import DetectedFace, InsightFaceEngine


class RecognitionService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        face_engine: InsightFaceEngine,
    ):
        self.embedding_service = embedding_service
        self.face_engine = face_engine

    def detect_frame_faces(self, frame) -> list[DetectedFace]:
        return self.face_engine.detect_faces(frame)

    async def recognize_frame(self, tenant_id: str, frame, threshold: float) -> list[dict]:
        detections = self.face_engine.detect_faces(frame)
        stored_embeddings = await self.embedding_service.list_tenant_embeddings(tenant_id)
        return self._match_detections(detections, stored_embeddings, threshold)

    async def recognize_image_bytes(self, tenant_id: str, image_bytes: bytes, threshold: float) -> list[dict]:
        detections = self.face_engine.extract_embeddings_from_image_bytes(image_bytes)
        stored_embeddings = await self.embedding_service.list_tenant_embeddings(tenant_id)
        return self._match_detections(detections, stored_embeddings, threshold)

    def _match_detections(
        self,
        detections: list[DetectedFace],
        stored_embeddings: list[dict],
        threshold: float,
    ) -> list[dict]:
        known = [
            {
                "employeeId": item["employeeId"],
                "employeeName": item.get("employeeName"),
                "embedding": np.array(item["embedding"], dtype=np.float32),
            }
            for item in stored_embeddings
        ]

        results = []
        for detection in detections:
            best_employee_id = None
            best_employee_name = None
            best_score = -1.0
            detected_embedding = np.array(detection.embedding, dtype=np.float32)

            for item in known:
                score = float(np.dot(detected_embedding, item["embedding"]))
                if score > best_score:
                    best_score = score
                    best_employee_id = item["employeeId"]
                    best_employee_name = item.get("employeeName")

            is_match = best_employee_id is not None and best_score >= threshold
            results.append(
                {
                    "employeeId": best_employee_id if is_match else None,
                    "employeeName": best_employee_name if is_match else None,
                    "matched": is_match,
                    "confidence": best_score if best_score >= 0 else None,
                    "bbox": detection.bbox,
                    "detectionScore": detection.detectionScore,
                }
            )

        return results
