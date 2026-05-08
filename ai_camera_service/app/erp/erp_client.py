import base64
import logging
from urllib.parse import urljoin

import httpx

from app.config import Settings
from app.schemas.erp_schema import AttendanceRules, CameraConfig, EmployeeConfig, ErpEventPayload

logger = logging.getLogger(__name__)


class ErpClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.erp_api_token:
            headers["Authorization"] = f"Bearer {self.settings.erp_api_token}"
        return headers

    def _url(self, path: str) -> str:
        if not self.settings.erp_base_url:
            raise RuntimeError("ERP_BASE_URL is not configured")
        return urljoin(self.settings.erp_base_url.rstrip("/") + "/", path.lstrip("/"))

    async def fetch_cameras(self) -> list[CameraConfig]:
        data = await self._get_json("/api/ai/cameras")
        items = data if isinstance(data, list) else data.get("items", [])
        return [CameraConfig.model_validate(item) for item in items]

    async def fetch_employees(self, tenant_id: str) -> list[EmployeeConfig]:
        data = await self._get_json("/api/ai/employees", params={"tenantId": tenant_id})
        items = data if isinstance(data, list) else data.get("items", [])
        return [EmployeeConfig.model_validate(item) for item in items]

    async def fetch_attendance_rules(self, tenant_id: str) -> AttendanceRules:
        data = await self._get_json("/api/ai/attendance-rules", params={"tenantId": tenant_id})
        return AttendanceRules.model_validate(data)

    async def send_event(self, payload: ErpEventPayload) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self._url("/api/ai/events"),
                json=payload.model_dump(exclude_none=True),
                headers=self._headers(),
            )
            response.raise_for_status()
            if response.content:
                return response.json()
            return {"status": "sent"}

    async def download_face_image(self, image_url: str) -> bytes:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self._resolve_asset_url(image_url), headers=self._headers())
            response.raise_for_status()
            return response.content

    async def decode_or_download_face_image(self, ref) -> bytes | None:
        inline = ref.base64 or ref.content
        if inline:
            if "," in inline and inline.lower().startswith("data:"):
                inline = inline.split(",", 1)[1]
            return base64.b64decode(inline)

        image_url = ref.imageUrl or ref.url
        if image_url:
            return await self.download_face_image(image_url)
        return None

    async def _get_json(self, path: str, params: dict | None = None):
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self._url(path), params=params, headers=self._headers())
            response.raise_for_status()
            return response.json()

    def _resolve_asset_url(self, image_url: str) -> str:
        if image_url.startswith(("http://", "https://")):
            return image_url
        return self._url(image_url)

