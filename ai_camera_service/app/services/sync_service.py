import hashlib
import logging
from datetime import datetime

from app.erp.erp_client import ErpClient
from app.face.embedding_service import EmbeddingService
from app.face.insightface_engine import InsightFaceEngine
from app.runtime_state import RuntimeState
from app.schemas.erp_schema import CameraConfig, EmployeeConfig
from app.services.log_service import LogService

logger = logging.getLogger(__name__)


class SyncService:
    def __init__(
        self,
        erp_client: ErpClient,
        runtime_state: RuntimeState,
        embedding_service: EmbeddingService,
        face_engine: InsightFaceEngine,
        log_service: LogService,
    ):
        self.erp_client = erp_client
        self.runtime_state = runtime_state
        self.embedding_service = embedding_service
        self.face_engine = face_engine
        self.log_service = log_service

    async def sync_all(self) -> dict:
        camera_result = await self.sync_cameras()
        tenant_ids = sorted({camera.tenantId for camera in self.runtime_state.list_cameras()})
        employee_results = [await self.sync_employees(tenant_id) for tenant_id in tenant_ids]
        rule_results = [await self.sync_rules(tenant_id) for tenant_id in tenant_ids]
        return {"cameras": camera_result, "employees": employee_results, "rules": rule_results}

    async def sync_cameras(self) -> dict:
        if self._use_local_dev_camera():
            cameras = self._local_dev_cameras()
        else:
            cameras = await self.erp_client.fetch_cameras()
        self.runtime_state.set_cameras(cameras)
        self.runtime_state.last_sync["cameras"] = datetime.utcnow().isoformat()
        source = "local USB development config" if self._use_local_dev_camera() else "ERP"
        await self.log_service.write("INFO", f"Synced cameras from {source}", metadata={"count": len(cameras)})
        return {"count": len(cameras), "cameraIds": [camera.cameraId for camera in cameras]}

    async def sync_employees(self, tenant_id: str | None = None) -> dict:
        if not self.erp_client.settings.erp_base_url:
            return {
                "tenants": [],
                "message": "ERP_BASE_URL is empty. Employee sync skipped in local development.",
            }
        tenant_ids = [tenant_id] if tenant_id else sorted({camera.tenantId for camera in self.runtime_state.list_cameras()})
        results = []
        for current_tenant_id in tenant_ids:
            employees = await self.erp_client.fetch_employees(current_tenant_id)
            embeddings = await self._sync_employee_embeddings(current_tenant_id, employees)
            results.append(
                {
                    "tenantId": current_tenant_id,
                    "employees": len(employees),
                    "embeddingsProcessed": embeddings,
                }
            )
            await self.log_service.write(
                "INFO",
                "Synced employee face embeddings from ERP",
                tenant_id=current_tenant_id,
                metadata=results[-1],
            )
        self.runtime_state.last_sync["employees"] = datetime.utcnow().isoformat()
        return {"tenants": results}

    async def sync_rules(self, tenant_id: str | None = None) -> dict:
        tenant_ids = [tenant_id] if tenant_id else sorted({camera.tenantId for camera in self.runtime_state.list_cameras()})
        rules = []
        for current_tenant_id in tenant_ids:
            if self.erp_client.settings.erp_base_url:
                rule = await self.erp_client.fetch_attendance_rules(current_tenant_id)
            else:
                from app.schemas.erp_schema import AttendanceRules

                rule = AttendanceRules(
                    tenantId=current_tenant_id,
                    recognitionThreshold=self.erp_client.settings.default_recognition_threshold,
                    duplicateCooldownSeconds=self.erp_client.settings.default_duplicate_cooldown_seconds,
                )
            self.runtime_state.set_rule(rule)
            rules.append(rule.model_dump())
            await self.log_service.write(
                "INFO",
                "Synced attendance rules from ERP",
                tenant_id=current_tenant_id,
                metadata=rule.model_dump(),
            )
        self.runtime_state.last_sync["rules"] = datetime.utcnow().isoformat()
        return {"rules": rules}

    async def _sync_employee_embeddings(self, tenant_id: str, employees: list[EmployeeConfig]) -> int:
        processed = 0
        for employee in employees:
            if not employee.active:
                continue
            for index, ref in enumerate(employee.faceImages):
                try:
                    image_bytes = await self.erp_client.decode_or_download_face_image(ref)
                    if not image_bytes:
                        continue
                    faces = self.face_engine.extract_embeddings_from_image_bytes(image_bytes)
                    source_id = ref.sourceId or self._source_id(employee.employeeId, ref, index)
                    processed += await self.embedding_service.upsert_employee_embeddings(
                        tenant_id=tenant_id,
                        employee_id=employee.employeeId,
                        employee_name=employee.fullName,
                        embeddings=[face.embedding for face in faces],
                        source_id=source_id,
                    )
                except Exception as exc:
                    logger.exception("Failed to process face image for employee %s", employee.employeeId)
                    await self.log_service.write(
                        "ERROR",
                        "Failed to process employee face image",
                        tenant_id=tenant_id,
                        metadata={"employeeId": employee.employeeId, "error": str(exc)},
                    )
        return processed

    @staticmethod
    def _source_id(employee_id: str, ref, index: int) -> str:
        raw = ref.imageUrl or ref.url or ref.base64 or ref.content or f"{employee_id}:{index}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"{employee_id}:{digest}"

    def _use_local_dev_camera(self) -> bool:
        settings = self.erp_client.settings
        return settings.environment == "development" and not settings.erp_base_url

    def _local_dev_cameras(self) -> list[CameraConfig]:
        settings = self.erp_client.settings
        rtsp_urls = self._local_rtsp_urls()

        if settings.camera_source_mode == "rtsp" and rtsp_urls:
            return [
                CameraConfig(
                    tenantId=settings.dev_tenant_id,
                    cameraId=f"RTSP_CAM_{index:02d}",
                    name=f"Local RTSP Camera {index:02d}",
                    rtspUrl=rtsp_url,
                    enabled=True,
                    direction="IN",
                    capabilities=["FACE_RECOGNITION"],
                    zones=[],
                )
                for index, rtsp_url in enumerate(rtsp_urls, start=1)
            ]

        return [
            CameraConfig(
                tenantId=settings.dev_tenant_id,
                cameraId=settings.dev_camera_id,
                name="Local USB Camera",
                rtspUrl=f"usb://{settings.usb_camera_index}",
                enabled=True,
                direction="IN",
                capabilities=["FACE_RECOGNITION"],
                zones=[],
            )
        ]

    def _local_rtsp_urls(self) -> list[str]:
        settings = self.erp_client.settings
        if settings.rtsp_urls:
            return [url.strip() for url in settings.rtsp_urls.split(",") if url.strip()]

        if not settings.rtsp_url:
            return []

        channels = [channel.strip() for channel in settings.rtsp_channels.split(",") if channel.strip()]
        if not channels:
            return [settings.rtsp_url]

        base_url = settings.rtsp_url.rsplit("/", 1)[0]
        return [f"{base_url}/{channel}" for channel in channels]
