import asyncio
import base64
import binascii
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

from service.embedding_service import EmbeddingService
from face_recognition import InsightFaceEngine
from schemas.project_schema import (
    AttendanceRules,
    CameraAssignment,
    CameraConfig,
    EmployeeConfig,
)



logger = logging.getLogger(__name__)


class SyncService:
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        embedding_service: EmbeddingService,
        face_engine: InsightFaceEngine,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.face_engine = face_engine

        # Replaces RuntimeState.last_sync.
        self.last_sync: dict[str, str] = {}

        # Prevent simultaneous synchronization operations.
        self._camera_sync_lock = asyncio.Lock()
        self._employee_sync_lock = asyncio.Lock()
        self._rules_sync_lock = asyncio.Lock()

    # =========================================================
    # Camera synchronization
    # =========================================================

    async def sync_cameras_from_payload(
        self,
        payload: Any,
    ) -> dict:
        async with self._camera_sync_lock:
           
            incoming = [
                CameraConfig.model_validate(item)
                for item in self._items(payload)
            ]
            # Combine duplicate physical cameras from the payload.
            incoming = self._group_physical_cameras(incoming)
            # MongoDB replaces RuntimeState as the source of truth.
            existing = await self.get_all_cameras()
            cameras = self._merge_etablissement_camera_sync(
                existing=existing,
                incoming=incoming,
            )

            return await self.set_cameras(
                cameras=cameras,
                source="ERP push",
            )

    async def set_cameras(
        self,
        cameras: list[CameraConfig],
        source: str,
    ) -> dict:
        await self._persist_cameras(cameras)

        self.last_sync["cameras"] = (
            datetime.now(timezone.utc).isoformat()
        )

        # await self.log_service.write(
        #     "INFO",
        #     f"Synced cameras from {source}",
        #     metadata={
        #         "count": len(cameras),
        #     },
        # )

        return {
            "count": len(cameras),
            "cameraIds": [
                camera.cameraId
                for camera in cameras
            ],
            "cameras": [
                {
                    "cameraId": camera.cameraId,
                    "etsAuths": camera.listOfEtsAuths,
                    "enabled": camera.enabled,
                    "capabilities": [
                        capability.value
                        for capability in camera.capabilities
                    ],
                    "assignments": [
                        assignment.model_dump(
                            by_alias=True,
                            mode="json",
                        )
                        for assignment in camera.assignments
                    ],
                }
                for camera in cameras
            ],
        }

    async def get_all_cameras(
        self,
    ) -> list[CameraConfig]:
        documents = await self.db.camera_configs.find(
            {},
            {"_id": 0},
        ).to_list(length=None)

        return [
            CameraConfig.model_validate(document)
            for document in documents
        ]

    async def get_camera(
        self,
        camera_id: str,
    ) -> CameraConfig | None:
        document = await self.db.camera_configs.find_one(
            {
                "cameraId": camera_id,
            },
            {
                "_id": 0,
            },
        )

        if document is None:
            return None

        return CameraConfig.model_validate(document)

    async def _persist_cameras(
        self,
        cameras: list[CameraConfig],
    ) -> None:
        now = datetime.now(timezone.utc)
        camera_ids: list[str] = []
        operations: list[UpdateOne] = []

        for camera in cameras:
            camera_ids.append(camera.cameraId)

            document = {
                **camera.model_dump(
                    by_alias=True,
                    mode="json",
                ),
                "updatedAt": now,
            }

            operations.append(
                UpdateOne(
                    {
                        "cameraId": camera.cameraId,
                    },
                    {
                        "$set": document,
                        "$setOnInsert": {
                            "createdAt": now,
                        },
                    },
                    upsert=True,
                )
            )

        if operations:
            await self.db.camera_configs.bulk_write(
                operations,
                ordered=False,
            )

        # The synchronized list becomes the saved camera list.
        if camera_ids:
            await self.db.camera_configs.delete_many(
                {
                    "cameraId": {
                        "$nin": camera_ids,
                    }
                }
            )
        else:
            await self.db.camera_configs.delete_many({})

    def _merge_etablissement_camera_sync(
        self,
        existing: list[CameraConfig],
        incoming: list[CameraConfig],
    ) -> list[CameraConfig]:
        if not incoming:
            return existing

       
        synchronized_etablissements = {
            assignment.etsAuth
            for camera in incoming
            for assignment in camera.assignments
        }

        merged_by_id: dict[str, CameraConfig] = {}

        for camera in existing:
            remaining_assignments = [
                assignment
                for assignment in camera.assignments
                if assignment.etsAuth
                not in synchronized_etablissements
            ]

            if remaining_assignments:
                merged_by_id[camera.cameraId] = (
                    self._camera_with_assignments(
                        base=camera,
                        assignments=remaining_assignments,
                    )
                )

        # Insert or update incoming assignments.
        for camera in incoming:
            current = merged_by_id.get(camera.cameraId)

            if (
                current is not None
                and not self._same_camera_source(
                    current.rtspUrl,
                    camera.rtspUrl,
                )
            ):
                raise ValueError(
                    f"Camera {camera.cameraId} already exists "
                    "with a different RTSP source"
                )

            assignments_by_etablissement = {
                assignment.etsAuth: assignment
                for assignment in (
                    current.assignments
                    if current is not None
                    else []
                )
            }

            assignments_by_etablissement.update(
                {
                    assignment.etsAuth: assignment
                    for assignment in camera.assignments
                }
            )

            merged_by_id[camera.cameraId] = (
                self._camera_with_assignments(
                    base=current or camera,
                    assignments=list(
                        assignments_by_etablissement.values()
                    ),
                    latest=camera,
                )
            )

        return list(merged_by_id.values())

    def _group_physical_cameras(
    self,
    cameras: list[CameraConfig],
) -> list[CameraConfig]:
        grouped: dict[str, CameraConfig] = {}

        print(
            "1. INPUT CAMERAS:",
            [camera.cameraId for camera in cameras],
            flush=True,
        )

        for camera in cameras:
            try:
                print(
                    "2. LOOP START:",
                    camera.cameraId,
                    flush=True,
                )

                current = grouped.get(camera.cameraId)

                print(
                    "3. CURRENT:",
                    current,
                    flush=True,
                )

                if current is not None:
                    print(
                        "4. CHECKING RTSP:",
                        current.rtspUrl,
                        camera.rtspUrl,
                        flush=True,
                    )

                    if not self._same_camera_source(
                        current.rtspUrl,
                        camera.rtspUrl,
                    ):
                        raise ValueError(
                            f"Camera {camera.cameraId} was received "
                            "with multiple RTSP sources"
                        )

                print(
                    "5. CAMERA ASSIGNMENTS:",
                    camera.assignments,
                    flush=True,
                )

                assignments_by_ets_auth = {
                    assignment.etsAuth: assignment
                    for assignment in (
                        current.assignments
                        if current is not None
                        else []
                    )
                }

                print(
                    "6. EXISTING ASSIGNMENTS:",
                    assignments_by_ets_auth,
                    flush=True,
                )

                assignments_by_ets_auth.update(
                    {
                        assignment.etsAuth: assignment
                        for assignment in camera.assignments
                    }
                )

                print(
                    "7. MERGED ASSIGNMENTS:",
                    assignments_by_ets_auth,
                    flush=True,
                )

                print(
                    "8. CALLING _camera_with_assignments",
                    flush=True,
                )

                result_camera = self._camera_with_assignments(
                    base=current or camera,
                    assignments=list(
                        assignments_by_ets_auth.values()
                    ),
                    latest=camera,
                )

                print(
                    "9. RESULT CAMERA:",
                    result_camera,
                    flush=True,
                )

                grouped[camera.cameraId] = result_camera

                print(
                    "10. GROUPED IDS:",
                    list(grouped.keys()),
                    flush=True,
                )

            except Exception as exc:
                print(
                    "GROUPING FAILED:",
                    camera.cameraId,
                    type(exc).__name__,
                    repr(exc),
                    flush=True,
                )
                raise

        result = list(grouped.values())

        print(
            "11. FINAL RESULT:",
            [camera.cameraId for camera in result],
            flush=True,
        )

        return result

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
    def _same_camera_source(
        first: str,
        second: str,
    ) -> bool:
        return first.rstrip("/") == second.rstrip("/")

    # =========================================================
    # Employee and embedding synchronization
    # =========================================================

    async def sync_employees_from_payload(
        self,
        payload: Any,
    ) -> dict:
        employees = [
            EmployeeConfig.model_validate(item)
            for item in self._items(payload)
        ]

        listOfetsAuth = sorted(
            {
                employee.etsAuth
                for employee in employees
            }
        )

        results = []

        for etsAuth in listOfetsAuth:
            etablissement_employees = [
                employee
                for employee in employees
                if employee.etsAuth == etsAuth
            ]

            processed = await self._sync_employee_embeddings(
                etsAuth=etsAuth,
                employees=etablissement_employees,
            )

            results.append(
                {
                    "etsAuth": etsAuth,
                    "employees": len(etablissement_employees),
                    "embeddingsProcessed": processed,
                }
            )

        self.last_sync["employees"] = (
            datetime.now(timezone.utc).isoformat()
        )

        return {
            "etablissements": results,
        }

    async def _sync_employee_embeddings(
        self,
        etsAuth: str,
        employees: list[EmployeeConfig],
    ) -> int:
        async with self._employee_sync_lock:
            processed = 0

            for employee in employees:
                if not employee.active:
                    continue

                for image_index, reference in enumerate(
                    employee.faceImages
                ):
                    try:
                        image_bytes = await self._decode_face_image(
                            reference
                        )

                        if not image_bytes:
                            continue

                        # InsightFace is synchronous, so execute it
                        # outside the FastAPI event loop.
                        faces = await asyncio.to_thread(
                            self.face_engine
                            .extract_embeddings_from_image_bytes,
                            image_bytes,
                        )

                        if not faces:
                            raise ValueError(
                                "No face detected in employee image"
                            )

                        source_id = (
                            reference.sourceId
                            or self._source_id(
                                employee_id=employee.employeeId,
                                reference=reference,
                                index=image_index,
                            )
                        )

                        processed += await (
                            self.embedding_service
                            .upsert_employee_embeddings(
                                etsAuth=etsAuth,
                                employee_id=employee.employeeId,
                                employee_name=employee.fullName,
                                embeddings=[
                                    face.embedding
                                    for face in faces
                                ],
                                source_id=source_id,
                            )
                        )

                    except Exception as exc:
                        logger.exception(
                            "Failed to process face image "
                            "for employee %s",
                            employee.employeeId,
                        )

                        # await self.log_service.write(
                        #     "ERROR",
                        #     "Failed to process employee face image",
                        #     etsAuth=etsAuth,
                        #     metadata={
                        #         "employeeId": employee.employeeId,
                        #         "error": str(exc),
                        #     },
                        # )

            return processed

    async def _decode_face_image(
        self,
        reference,
    ) -> bytes | None:
        encoded = (
            reference.base64
            or reference.content
        )

        if not encoded:
            if reference.imageUrl or reference.url:
                raise ValueError(
                    "Face image URLs are not supported. "
                    "Send the image as Base64 or content."
                )

            return None

        encoded = encoded.strip()

        # Handle:
        # data:image/jpeg;base64,/9j/4AAQ...
        if encoded.lower().startswith("data:"):
            if "," not in encoded:
                raise ValueError(
                    "Employee face image has an invalid data URL"
                )

            header, encoded = encoded.split(",", 1)

            if ";base64" not in header.lower():
                raise ValueError(
                    "Employee face image data URL "
                    "is not Base64 encoded"
                )

        encoded = "".join(encoded.split())

        if not encoded:
            raise ValueError(
                "Employee face image Base64 data is empty"
            )

        # Restore missing Base64 padding.
        encoded += "=" * (-len(encoded) % 4)

        try:
            image_bytes = base64.b64decode(
                encoded,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError(
                "Employee face image contains invalid Base64 data"
            ) from exc

        if not image_bytes:
            raise ValueError(
                "Employee face image decoded to empty data"
            )

        return image_bytes

    @staticmethod
    def _source_id(
        employee_id: str,
        reference,
        index: int,
    ) -> str:
        raw = (
            reference.imageUrl
            or reference.url
            or reference.base64
            or reference.content
            or f"{employee_id}:{index}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:24]

        return f"{employee_id}:{digest}"

    # =========================================================
    # Attendance-rule synchronization
    # =========================================================

    async def sync_rules_from_payload(
        self,
        payload: Any,
    ) -> dict:
        async with self._rules_sync_lock:
            rules = [
                AttendanceRules.model_validate(item)
                for item in self._items(payload)
            ]

            purge_results = []

            for rule in rules:
                await self._persist_rule(rule)

                purge_results.append(
                    await self._purge_images_for_rule(rule)
                )

                # await self.log_service.write(
                #     "INFO",
                #     "Synced attendance rules from ERP push",
                #     etsAuth=rule.etsAuth,
                #     metadata=rule.model_dump(
                #         by_alias=True,
                #         mode="json",
                #     ),
                # )

            self.last_sync["rules"] = (
                datetime.now(timezone.utc).isoformat()
            )

            return {
                "rules": [
                    rule.model_dump(
                        by_alias=True,
                        mode="json",
                    )
                    for rule in rules
                ],
                "imagePurge": purge_results,
            }

    async def get_all_rules(
        self,
    ) -> list[AttendanceRules]:
        documents = await self.db.attendance_rules.find(
            {},
            {"_id": 0},
        ).to_list(length=None)

        return [
            AttendanceRules.model_validate(document)
            for document in documents
        ]

    async def get_rule_by_etsAuth(self,ets_auth: str,) -> AttendanceRules | None:
        document = await self.db.attendance_rules.find_one(
            {"etsAuth": ets_auth,},
            {"_id": 0,},
            sort=[("updatedAt", -1),],
        )

        if document is None:
            return None

        return AttendanceRules.model_validate(document)

    async def _persist_rule(
        self,
        rule: AttendanceRules,
    ) -> None:
        now = datetime.now(timezone.utc)

        document = {
            **rule.model_dump(
                by_alias=True,
                mode="json",
            ),
            "updatedAt": now,
        }

        await self.db.attendance_rules.update_one(
            {
                "etsAuth": rule.etsAuth,
            },
            {
                "$set": document,
                "$setOnInsert": {
                    "createdAt": now,
                },
            },
            upsert=True,
        )

    # =========================================================
    # Saved configuration and snapshot cleanup
    # =========================================================

    async def load_saved_config(self) -> dict:
        cameras = await self.get_all_cameras()
        rules = await self.get_all_rules()

        purge_results = await self.purge_images_for_loaded_rules(
            rules
        )

        self.last_sync["cameras"] = "loaded_from_mongo"
        self.last_sync["rules"] = "loaded_from_mongo"

        # await self.log_service.write(
        #     "INFO",
        #     "Loaded saved configuration",
        #     metadata={
        #         "cameras": len(cameras),
        #         "rules": len(rules),
        #         "imagePurge": purge_results,
        #     },
        # )

        return {
            "cameras": len(cameras),
            "rules": len(rules),
            "imagePurge": purge_results,
        }

    async def purge_images_for_loaded_rules(
        self,
        rules: list[AttendanceRules] | None = None,
    ) -> list[dict]:
        if rules is None:
            rules = await self.get_all_rules()

        results = []

        for rule in rules:
            result = await self._purge_images_for_rule(rule)
            results.append(result)

        return results


    # =========================================================
    # Payload helper
    # =========================================================

    @staticmethod
    def _items(
        payload: Any,
    ) -> list[dict]:
        if payload is None:
            return []

        if isinstance(payload, list):
            return payload

        if isinstance(payload, dict):
            for key in (
                "items",
                "cameras",
                "employees",
                "rules",
            ):
                value = payload.get(key)

                if isinstance(value, list):
                    return value

            return [payload]

        raise ValueError(
            "Payload must be an object, an array, or an "
            "object containing items/cameras/employees/rules"
        )