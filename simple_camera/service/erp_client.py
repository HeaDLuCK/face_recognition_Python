import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class ErpClient:
    def __init__(
        self,
        base_url: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }

        return headers

    async def send_unknown_batch(
        self,
        ets_auth: str,
        unknown_persons: list[dict[str, Any]],
    ) -> None:

        if not unknown_persons:
            return

        payload = unknown_persons

        response = await self._client.post(
           f"/printCtrl?tp=unknown&auth={ets_auth}",
            json=payload,
            headers=self._headers(),
        )

        response.raise_for_status()

        logger.info(
            "Unknown persons sent to ERP: count=%d",
            len(unknown_persons),
        )

    async def close(self) -> None:
        await self._client.aclose()