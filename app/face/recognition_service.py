import asyncio

import numpy as np

from app.face.embedding_index import TenantEmbeddingIndex, best_embedding_candidate
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
        detections = await asyncio.to_thread(self.face_engine.detect_faces, frame)
        index = await self.embedding_service.get_tenant_index(tenant_id)
        return self._match_detections(detections, index, threshold)

    async def recognize_frame_for_tenants(
        self,
        tenant_thresholds: dict[str, float],
        frame,
    ) -> list[dict]:
        detections = await asyncio.to_thread(self.face_engine.detect_faces, frame)
        tenant_ids = list(tenant_thresholds)
        indexes = await asyncio.gather(
            *[
                self.embedding_service.get_tenant_index(tenant_id)
                for tenant_id in tenant_ids
            ]
        )
        return self._match_detections_for_tenants(
            detections,
            dict(zip(tenant_ids, indexes)),
            tenant_thresholds,
        )

    async def recognize_image_bytes(self, tenant_id: str, image_bytes: bytes, threshold: float) -> list[dict]:
        detections = await asyncio.to_thread(self.face_engine.extract_embeddings_from_image_bytes, image_bytes)
        index = await self.embedding_service.get_tenant_index(tenant_id)
        return self._match_detections(detections, index, threshold)

    def _match_detections(
        self,
        detections: list[DetectedFace],
        index: TenantEmbeddingIndex,
        threshold: float,
    ) -> list[dict]:
        results = []
        for detection in detections:
            detected_embedding = np.array(detection.embedding, dtype=np.float32)
            candidate = best_embedding_candidate(index, detected_embedding)
            best_employee_id = candidate.get("employeeId") if candidate else None
            best_employee_name = candidate.get("employeeName") if candidate else None
            best_score = candidate.get("score") if candidate else None
            is_match = best_employee_id is not None and best_score is not None and best_score >= threshold
            reported_score = best_score if best_score is not None and best_score >= 0 else None
            results.append(
                {
                    "employeeId": best_employee_id if is_match else None,
                    "employeeName": best_employee_name if is_match else None,
                    "bestCandidateEmployeeId": best_employee_id,
                    "bestCandidateEmployeeName": best_employee_name,
                    "bestCandidateScore": reported_score,
                    "matched": is_match,
                    "confidence": reported_score,
                    "bbox": detection.bbox,
                    "detectionScore": detection.detectionScore,
                    "_embedding": detected_embedding.tolist(),
                }
            )

        return results

    def _match_detections_for_tenants(
        self,
        detections: list[DetectedFace],
        indexes_by_tenant: dict[str, TenantEmbeddingIndex],
        tenant_thresholds: dict[str, float],
    ) -> list[dict]:
        results = []
        for detection in detections:
            detected_embedding = np.array(detection.embedding, dtype=np.float32)
            best_candidate = None
            best_match = None
            tenant_candidates: dict[str, dict] = {}

            for tenant_id, index in indexes_by_tenant.items():
                tenant_candidate = best_embedding_candidate(index, detected_embedding)
                if tenant_candidate is None:
                    continue
                candidate = {**tenant_candidate, "tenantId": tenant_id}
                score = candidate["score"]
                if best_candidate is None or score > best_candidate["score"]:
                    best_candidate = candidate
                tenant_candidates[tenant_id] = candidate
                threshold = tenant_thresholds[tenant_id]
                if score >= threshold and (best_match is None or score > best_match["score"]):
                    best_match = candidate

            selected = best_match or best_candidate
            matched = best_match is not None
            results.append(
                {
                    "tenantId": best_match["tenantId"] if matched else None,
                    "employeeId": best_match["employeeId"] if matched else None,
                    "employeeName": best_match.get("employeeName") if matched else None,
                    "bestCandidateTenantId": selected["tenantId"] if selected else None,
                    "bestCandidateEmployeeId": selected["employeeId"] if selected else None,
                    "bestCandidateEmployeeName": selected.get("employeeName") if selected else None,
                    "bestCandidateScore": selected["score"] if selected else None,
                    "matched": matched,
                    "confidence": selected["score"] if selected else None,
                    "bbox": detection.bbox,
                    "detectionScore": detection.detectionScore,
                    "_embedding": detected_embedding.tolist(),
                    "_tenantCandidates": {
                        tenant_id: {
                            "employeeId": candidate["employeeId"],
                            "employeeName": candidate.get("employeeName"),
                            "score": candidate["score"],
                        }
                        for tenant_id, candidate in tenant_candidates.items()
                    },
                }
            )

        return results
