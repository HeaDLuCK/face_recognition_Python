import base64
import binascii
from urllib.parse import urljoin

import httpx

from app.config import Settings
from app.schemas.erp_schema import AttendanceRules, CameraConfig, EmployeeConfig, ErpEventPayload

class ErpClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

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
        data = await self._get_json("/api/ai/employees", params={"etsAuth": tenant_id})
        items = data if isinstance(data, list) else data.get("items", [])
        return [EmployeeConfig.model_validate(item) for item in items]

    async def fetch_attendance_rules(self, tenant_id: str) -> AttendanceRules:
        data = await self._get_json("/api/ai/attendance-rules", params={"etsAuth": tenant_id})
        return AttendanceRules.model_validate(data)

    async def send_event(self, payload: ErpEventPayload) -> dict:
        response = await self._client.post(
            self._url("/api/ai/events"),
            json=payload.model_dump(exclude_none=True, by_alias=True),
            headers=self._headers(),
        )
        response.raise_for_status()
        if response.content:
            return response.json()
        return {"status": "sent"}

    async def download_face_image(self, image_url: str) -> bytes:
        response = await self._client.get(self._resolve_asset_url(image_url), headers=self._headers())
        response.raise_for_status()
        return response.content

    async def decode_or_download_face_image(self, ref) -> bytes | None:
        inline = ref.base64 or ref.content
        if inline:
            if "," in inline and inline.lower().startswith("data:"):
                inline = inline.split(",", 1)[1]
            try:
                return base64.b64decode(inline, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Employee face image contains invalid Base64 data") from exc

        image_url = ref.imageUrl or ref.url
        if not image_url:
            return None
        return await self.download_face_image(self._decode_asset_reference(image_url))

    async def _get_json(self, path: str, params: dict | None = None):
        response = await self._client.get(self._url(path), params=params, headers=self._headers())
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _decode_asset_reference(value: str) -> str:
        candidate = value.strip()
        if candidate.startswith(("http://", "https://", "/")):
            return candidate
        try:
            decoded = base64.b64decode(candidate, validate=True).decode("utf-8").strip()
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return candidate
        return decoded or candidate

    def _resolve_asset_url(self, image_url: str) -> str:
        if image_url.startswith(("http://", "https://")):
            return image_url
        return self._url(image_url)

