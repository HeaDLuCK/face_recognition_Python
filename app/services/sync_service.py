import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any

from pymongo import UpdateOne

from app.erp.erp_client import ErpClient
from app.face.embedding_service import EmbeddingService
from app.face.insightface_engine import InsightFaceEngine
from app.runtime_state import RuntimeState
from app.schemas.erp_schema import AttendanceRules, CameraAssignment, CameraConfig, EmployeeConfig
from app.services.log_service import LogService
from app.storage.snapshot_service import SnapshotService

logger = logging.getLogger(__name__)


class SyncService:
    def __init__(
        self,
        erp_client: ErpClient,
        runtime_state: RuntimeState,
        embedding_service: EmbeddingService,
        face_engine: InsightFaceEngine,
        log_service: LogService,
        snapshot_service: SnapshotService,
        db,
    ):
        self.erp_client = erp_client
        self.runtime_state = runtime_state
        self.embedding_service = embedding_service
        self.face_engine = face_engine
        self.log_service = log_service
        self.snapshot_service = snapshot_service
        self.db = db
        self._camera_sync_lock = asyncio.Lock()
        self._employee_sync_lock = asyncio.Lock()

    async def sync_all(self) -> dict:
        camera_result = await self.sync_cameras()
        tenant_ids = self.runtime_state.list_tenant_ids()
        employee_results = [await self.sync_employees(tenant_id) for tenant_id in tenant_ids]
        rule_results = [await self.sync_rules(tenant_id) for tenant_id in tenant_ids]
        return {"cameras": camera_result, "employees": employee_results, "rules": rule_results}

    async def sync_cameras(self) -> dict:
        async with self._camera_sync_lock:
            if self._use_local_dev_camera():
                cameras = self._local_dev_cameras()
            else:
                cameras = await self.erp_client.fetch_cameras()
            return await self.set_cameras(self._group_physical_cameras(cameras), source="ERP")

    async def sync_cameras_from_payload(self, payload: Any) -> dict:
        async with self._camera_sync_lock:
            incoming = [CameraConfig.model_validate(item) for item in self._items(payload)]
            cameras = self._merge_tenant_camera_sync(
                existing=self.runtime_state.list_cameras(),
                incoming=incoming,
            )
            return await self.set_cameras(cameras, source="ERP push")

    async def set_cameras(self, cameras: list[CameraConfig], source: str) -> dict:
        self.runtime_state.set_cameras(cameras)
        await self._persist_cameras(cameras)
        self.runtime_state.last_sync["cameras"] = datetime.utcnow().isoformat()
        await self.log_service.write("INFO", f"Synced cameras from {source}", metadata={"count": len(cameras)})
        return {
            "count": len(cameras),
            "cameraIds": [camera.cameraId for camera in cameras],
            "cameras": [
                {
                    "cameraId": camera.cameraId,
                    "etsAuths": camera.tenantIds,
                    "capabilities": [capability.value for capability in camera.capabilities],
                }
                for camera in cameras
            ],
        }

    async def sync_employees(self, tenant_id: str | None = None) -> dict:
        if not self.erp_client.settings.erp_base_url:
            return {
                "tenants": [],
                "message": "ERP_BASE_URL is empty. Employee sync skipped in local development.",
            }
        tenant_ids = [tenant_id] if tenant_id else self.runtime_state.list_tenant_ids()
        results = []
        for current_tenant_id in tenant_ids:
            employees = await self.erp_client.fetch_employees(current_tenant_id)
            embeddings = await self._sync_employee_embeddings(current_tenant_id, employees)
            results.append(
                {
                    "etsAuth": current_tenant_id,
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

    async def sync_employees_from_payload(self, payload: Any) -> dict:
        employees = [EmployeeConfig.model_validate(item) for item in self._items(payload)]
        tenant_ids = sorted({employee.tenantId for employee in employees})
        results = []
        for current_tenant_id in tenant_ids:
            tenant_employees = [employee for employee in employees if employee.tenantId == current_tenant_id]
            embeddings = await self._sync_employee_embeddings(current_tenant_id, tenant_employees)
            results.append(
                {
                    "etsAuth": current_tenant_id,
                    "employees": len(tenant_employees),
                    "embeddingsProcessed": embeddings,
                }
            )
        self.runtime_state.last_sync["employees"] = datetime.utcnow().isoformat()
        return {"tenants": results}

    async def sync_rules(self, tenant_id: str | None = None) -> dict:
        tenant_ids = [tenant_id] if tenant_id else self.runtime_state.list_tenant_ids()
        rules = []
        purge_results = []
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
            await self._persist_rule(rule)
            rules.append(rule.model_dump(by_alias=True))
            purge_results.append(await self._purge_images_for_rule(rule))
            await self.log_service.write(
                "INFO",
                "Synced attendance rules from ERP",
                tenant_id=current_tenant_id,
                metadata=rule.model_dump(by_alias=True),
            )
        self.runtime_state.last_sync["rules"] = datetime.utcnow().isoformat()
        return {"rules": rules, "imagePurge": purge_results}

    async def sync_rules_from_payload(self, payload: Any) -> dict:
        rules = [AttendanceRules.model_validate(item) for item in self._items(payload)]
        purge_results = []
        for rule in rules:
            self.runtime_state.set_rule(rule)
            await self._persist_rule(rule)
            purge_results.append(await self._purge_images_for_rule(rule))
            await self.log_service.write(
                "INFO",
                "Synced attendance rules from ERP push",
                tenant_id=rule.tenantId,
                metadata=rule.model_dump(by_alias=True),
            )
        self.runtime_state.last_sync["rules"] = datetime.utcnow().isoformat()
        return {"rules": [rule.model_dump(by_alias=True) for rule in rules], "imagePurge": purge_results}

    async def load_saved_config(self) -> dict:
        camera_docs = await self.db.camera_configs.find({}).to_list(length=None)
        rule_docs = await self.db.attendance_rules.find({}).to_list(length=None)

        cameras = [CameraConfig.model_validate(doc) for doc in camera_docs]
        rules = [AttendanceRules.model_validate(doc) for doc in rule_docs]

        if cameras:
            self.runtime_state.set_cameras(cameras)
            self.runtime_state.last_sync["cameras"] = "loaded_from_mongo"
        if rules:
            for rule in rules:
                self.runtime_state.set_rule(rule)
            self.runtime_state.last_sync["rules"] = "loaded_from_mongo"

        purge_results = await self.purge_images_for_loaded_rules()

        await self.log_service.write(
            "INFO",
            "Loaded saved camera configuration",
            metadata={"cameras": len(cameras), "rules": len(rules), "imagePurge": purge_results},
        )
        return {"cameras": len(cameras), "rules": len(rules), "imagePurge": purge_results}

    async def purge_images_for_loaded_rules(self) -> list[dict]:
        return [
            await self._purge_images_for_rule(rule)
            for rule in self.runtime_state.rules.values()
        ]

    async def _persist_cameras(self, cameras: list[CameraConfig]) -> None:
        synced_camera_ids = []
        operations = []
        now = datetime.utcnow()
        for camera in cameras:
            synced_camera_ids.append(camera.cameraId)
            doc = {
                **camera.model_dump(by_alias=True),
                "updatedAt": now,
            }
            operations.append(
                UpdateOne(
                    {"cameraId": camera.cameraId},
                    {"$set": doc, "$setOnInsert": {"createdAt": now}},
                    upsert=True,
                )
            )

        if operations:
            await self.db.camera_configs.bulk_write(operations, ordered=False)

        if synced_camera_ids:
            await self.db.camera_configs.delete_many({"cameraId": {"$nin": synced_camera_ids}})
        else:
            await self.db.camera_configs.delete_many({})

    def _merge_tenant_camera_sync(
        self,
        existing: list[CameraConfig],
        incoming: list[CameraConfig],
    ) -> list[CameraConfig]:
        if not incoming:
            return existing

        synced_tenant_ids = {
            assignment.tenantId
            for camera in incoming
            for assignment in camera.assignments
        }
        merged_by_id: dict[str, CameraConfig] = {}

        for camera in existing:
            remaining_assignments = [
                assignment
                for assignment in camera.assignments
                if assignment.tenantId not in synced_tenant_ids
            ]
            if remaining_assignments:
                merged_by_id[camera.cameraId] = self._camera_with_assignments(
                    camera,
                    remaining_assignments,
                )

        for camera in incoming:
            current = merged_by_id.get(camera.cameraId)
            if current is not None and not self._same_camera_source(current.rtspUrl, camera.rtspUrl):
                raise ValueError(
                    f"Camera {camera.cameraId} was already synced with a different RTSP source"
                )

            assignments_by_tenant = {
                assignment.tenantId: assignment
                for assignment in (current.assignments if current else [])
            }
            assignments_by_tenant.update(
                {assignment.tenantId: assignment for assignment in camera.assignments}
            )
            merged_by_id[camera.cameraId] = self._camera_with_assignments(
                current or camera,
                list(assignments_by_tenant.values()),
                latest=camera,
            )

        return list(merged_by_id.values())

    def _group_physical_cameras(self, cameras: list[CameraConfig]) -> list[CameraConfig]:
        grouped: dict[str, CameraConfig] = {}
        for camera in cameras:
            current = grouped.get(camera.cameraId)
            if current is not None and not self._same_camera_source(current.rtspUrl, camera.rtspUrl):
                raise ValueError(
                    f"Camera {camera.cameraId} was received with multiple RTSP sources"
                )
            assignments_by_tenant = {
                assignment.tenantId: assignment
                for assignment in (current.assignments if current else [])
            }
            assignments_by_tenant.update(
                {assignment.tenantId: assignment for assignment in camera.assignments}
            )
            grouped[camera.cameraId] = self._camera_with_assignments(
                current or camera,
                list(assignments_by_tenant.values()),
                latest=camera,
            )
        return list(grouped.values())

    @staticmethod
    def _camera_with_assignments(
        base: CameraConfig,
        assignments: list[CameraAssignment],
        latest: CameraConfig | None = None,
    ) -> CameraConfig:
        source = latest or base
        return CameraConfig(
            cameraId=base.cameraId,
            name=source.name,
            rtspUrl=base.rtspUrl,
            assignments=assignments,
        )

    @staticmethod
    def _same_camera_source(first: str, second: str) -> bool:
        return first.rstrip("/") == second.rstrip("/")

    async def _persist_rule(self, rule: AttendanceRules) -> None:
        doc = {
            **rule.model_dump(by_alias=True),
            "updatedAt": datetime.utcnow(),
        }
        await self.db.attendance_rules.update_one(
            {"etsAuth": rule.tenantId},
            {"$set": doc, "$setOnInsert": {"createdAt": datetime.utcnow()}},
            upsert=True,
        )

    async def _purge_images_for_rule(self, rule: AttendanceRules) -> dict:
        result = await self.snapshot_service.purge_expired_images(
            tenant_id=rule.tenantId,
            retention_days=rule.imageRetentionDays,
        )
        if result["enabled"]:
            await self.log_service.write(
                "INFO",
                "Purged expired snapshot images",
                tenant_id=rule.tenantId,
                metadata=result,
            )
        return result

    async def _sync_employee_embeddings(self, tenant_id: str, employees: list[EmployeeConfig]) -> int:
        async with self._employee_sync_lock:
            processed = 0
            for employee in employees:
                if not employee.active:
                    continue
                for index, ref in enumerate(employee.faceImages):
                    try:
                        image_bytes = await self.erp_client.decode_or_download_face_image(ref)
                        if not image_bytes:
                            continue
                        faces = await asyncio.to_thread(
                            self.face_engine.extract_embeddings_from_image_bytes,
                            image_bytes,
                        )
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

    @staticmethod
    def _items(payload: Any) -> list[dict]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                return payload["items"]
            if isinstance(payload.get("cameras"), list):
                return payload["cameras"]
            if isinstance(payload.get("employees"), list):
                return payload["employees"]
            if isinstance(payload.get("rules"), list):
                return payload["rules"]
            return [payload]
        raise ValueError("Payload must be an object, an array, or an object with items/cameras/employees/rules.")

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
                    cameraId=self._channel_from_rtsp_url(rtsp_url) or f"RTSP_CAM_{index:02d}",
                    name=f"Camera {self._channel_from_rtsp_url(rtsp_url) or index}",
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

    @staticmethod
    def _channel_from_rtsp_url(rtsp_url: str) -> str | None:
        if "/Streaming/Channels/" not in rtsp_url:
            return None
        return rtsp_url.rsplit("/", 1)[-1] or None
