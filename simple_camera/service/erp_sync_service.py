import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

from collections import defaultdict
from service.attendance_service import AttendanceService
from service.embedding_service import EmbeddingService
from service.erp_client import ErpClient
from service.unknown_person_service import (
    UnknownPersonService,
)
from datetime import datetime, timezone
import httpx
logger = logging.getLogger(__name__)


class ErpSyncService:
    def __init__(
        self,
        unknown_person_service: UnknownPersonService,
        attendance_service:AttendanceService,
        embedding_service:EmbeddingService,
        erp_client: ErpClient,
        interval_seconds: float = 30.0,
        assignment_interval_seconds: float = 300.0,
        batch_size: int = 10,
    ) -> None:
        self.unknown_person_service = unknown_person_service
        self.attendance_service = attendance_service
        self.embedding_service = embedding_service
        self.erp_client = erp_client

        self.interval_seconds = interval_seconds
        self.assignment_interval_seconds = assignment_interval_seconds

        self.batch_size = batch_size

        self._unknown_sync_task: asyncio.Task | None = None
        self._assignment_sync_task: asyncio.Task | None = None
        
        self._running = False
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()

        self._unknown_sync_task = asyncio.create_task(
            self._run_unknown_sync()
        )

        self._assignment_sync_task = asyncio.create_task(
            self._run_assignment_sync()
        )

        logger.info(
            "ERP sync service started: "
            "interval=%.1fs batchSize=%d",
            self.interval_seconds,
            self.batch_size,
        )

        logger.info(
                "ERP assignment service started: "
                "interval=%.1fs ",
                self.assignment_interval_seconds,
            )

    async def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._unknown_sync_task is not None:
            await self._task
            self._task = None

        if self._assignment_sync_task is not None:
            await self._tak
            self._task = None

        logger.info(
            "ERP sync service stopped"
        )

    async def _run_unknown_sync(self) -> None:
        while self._running:
            try:
                await self._sync_unknown_people()

            except httpx.ConnectError:
                logger.warning(
                    "ERP server unavailable. "
                    "Unknown people will remain pending."
                )

            except Exception:
                logger.exception(
                    "ERP unknown person sync failed"
                )

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )

            except asyncio.TimeoutError:
                pass
            
    async def _run_assignment_sync(self) -> None:
        print(f"self.assignment_interval_seconds: {self.assignment_interval_seconds}")
        while self._running:
            try:
                await self._sync_unknown_assignments()

            except httpx.ConnectError:
                logger.warning(
                    "ERP server unavailable. "
                    "Unknown people will remain pending."
                )

            except Exception:
                logger.exception(
                    "ERP unknown assignment sync failed"
                )

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.assignment_interval_seconds,
                )

            except asyncio.TimeoutError:
                pass
    
    async def _sync_unknown_people(
        self,
    ) -> None:
        unknowns = await (
            self.unknown_person_service
            .get_pending_erp_sync(
                limit=self.batch_size
            )
        )

        if not unknowns:
            return

        # ---------------------------------
        # GROUP UNKNOWN PEOPLE BY etsAuth
        # ---------------------------------
        grouped_unknowns = defaultdict(list)

        for unknown in unknowns:
            ets_auth = unknown.get(
                "etsAuth"
            )

            if not ets_auth:
                logger.warning(
                    "Unknown person has no etsAuth: "
                    "unknownId=%s",
                    unknown.get("unknownId"),
                )
                continue

            grouped_unknowns[
                ets_auth
            ].append(
                unknown
            )

        # ---------------------------------
        # ONE ERP REQUEST PER etsAuth
        # ---------------------------------
        for (
            ets_auth,
            establishment_unknowns,
        ) in grouped_unknowns.items():

            items: list[
                dict[str, Any]
            ] = []

            for unknown in establishment_unknowns:
                attendance_rows = await (
                    self.attendance_service
                    .list_unknown_attendance(
                        ets_auth=ets_auth,
                        unknown_id=unknown["unknownId"],
                    )
                )
                item = await (
                    self._build_unknown_item(
                        unknown
                    )
                )

                if item is not None:
                    item["attendanceRows"] = (
                        attendance_rows
                    )
                    items.append(
                        item
                    )

            if not items:
                continue

            # Example:
            # /api/ai/SEA_FOOD/unknown-persons/batch
            await (
                self.erp_client
                .send_unknown_batch(
                    ets_auth=ets_auth,
                    unknown_persons=items,
                )
            )

            # IMPORTANT:
            # Mark ONLY this establishment's
            # successfully sent records as SYNCED.
            unknown_ids = [
                item["unknownId"]
                for item in items
            ]

            await (
                self.unknown_person_service
                .mark_erp_synced(
                    unknown_ids
                )
            )

            logger.info(
                "ERP unknown batch synchronized: "
                "etsAuth=%s count=%d",
                ets_auth,
                len(items),
            )
    
    async def _build_unknown_item(
        self,
        unknown: dict,
    ) -> dict[str, Any] | None:

        face_path_value = unknown.get("facePath")

        if not face_path_value:
            logger.warning(
                "Unknown has no facePath: "
                "unknownId=%s",
                unknown.get("unknownId"),
            )
            return None

        face_path = Path(face_path_value)

        if not face_path.is_file():
            logger.warning(
                "Unknown face file not found: "
                "unknownId=%s path=%s",
                unknown.get("unknownId"),
                face_path,
            )
            return None

        try:
            face_bytes = await asyncio.to_thread(face_path.read_bytes)
        except OSError:
            logger.exception(
                "Unable to read unknown face: "
                "unknownId=%s path=%s",
                unknown.get("unknownId"),
                face_path,
            )
            return None

        image_base64 = (
            base64.b64encode(
                face_bytes
            ).decode("utf-8")
        )

        return {
            "unknownId": unknown["unknownId"],
            "firstSeenAt": self._serialize_date(unknown.get("firstSeenAt")),
            "lastSeenAt": self._serialize_date(unknown.get("lastSeenAt")),
            "seenCount": int(unknown.get("seenCount", 1)),
            "quality": float(unknown.get("bestQuality",0.0,)),
            "imageContentType":  "image/jpeg",
            "imageBase64": image_base64,
        }

    async def _sync_unknown_assignments(
        self,
    ) -> None:

        ets_auths = self.erp_client.get_configured_ets_auths()
        print(f"ets_auths: {ets_auths}")
        logger.info( f"ets_auths: {ets_auths}" )
        for ets_auth in ets_auths:

            try:

                assignments = await self.erp_client.get_unknown_assignments(ets_auth=ets_auth)
                
                if not assignments:
                    continue

                logger.info(
                    "ERP unknown assignments received "
                    "etsAuth=%s count=%s",
                    ets_auth,
                    len(assignments),
                )

                for assignment in assignments:
                    print(f"assignment: {assignment}")
                    await self._process_unknown_assignment(ets_auth=ets_auth,assignment=assignment,)

            except Exception:
                logger.exception(
                    "Failed syncing unknown assignments "
                    "etsAuth=%s",
                    ets_auth,
                )

    async def _add_unknown_embeddings_to_employee(
        self,
        *,
        ets_auth: str,
        employee_id: str,
        unknown_id: str,
        reference_embeddings: list,
    ) -> int:

        if not reference_embeddings:
            return 0

        new_embeddings, employee_name = await self.embedding_service.filter_new_employee_embeddings(
                            ets_auth=ets_auth,
                            employee_id=employee_id,
                            embeddings=reference_embeddings,
                            duplicate_threshold=0.92,
                        )
        
        if new_embeddings:
            await self.embedding_service.upsert_employee_embeddings(
                etsAuth=ets_auth,
                employee_id=employee_id,
                employee_name=employee_name,
                embeddings=new_embeddings,
                source_id=f"unknown:{unknown_id}",
            )

        return "11"

                
    async def _process_unknown_assignment(
        self,
        *,
        ets_auth: str,
        assignment: dict,
    ) -> None:

        unknown_id = assignment.get("unknownId")

        employee_id = assignment.get("employeeId")

        if not unknown_id or not employee_id:
            return
        
        unknown = await self.unknown_person_service.get_by_unknown_id(ets_auth=ets_auth,unknown_id=unknown_id,)

        if unknown is None:
            return

        reference_embeddings = unknown.get("referenceEmbeddings",[])

        if not reference_embeddings:

            legacy_embedding = unknown.get("referenceEmbedding")
            if legacy_embedding:
                reference_embeddings = [legacy_embedding]

        await self._add_unknown_embeddings_to_employee(
            ets_auth=ets_auth,
            employee_id=employee_id,
            unknown_id=unknown_id,
            reference_embeddings=reference_embeddings,
        )

        await self.unknown_person_service.assign_to_employee(
                ets_auth=ets_auth,
                unknown_id=unknown_id,
                employee_id=employee_id,
        )
        
        await self.attendance_service.resolve_unknown_attendance(
                unknown_id=unknown_id,
                employee_id=employee_id,
        )
        

        logger.info(
            "Unknown assignment synchronized "
            "etsAuth=%s unknownId=%s employeeId=%s",
            ets_auth,
            unknown_id,
            employee_id,
        )

    @staticmethod
    def _serialize_date(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return value.strftime(
                "%b %d, %Y, %I:%M:%S %p"
            )

        return str(value)