import logging
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class ErpClient:
    def __init__(
        self,
        urls_by_ets_auth: dict[str, str],
    ) -> None:
        self.urls_by_ets_auth = urls_by_ets_auth

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                10.0,
                connect=3.0,
            )
        )

    def _get_base_url(
        self,
        ets_auth: str,
    ) -> str:

        base_url = self.urls_by_ets_auth.get(
            ets_auth
        )

        if not base_url:
            raise ValueError(
                f"No ERP URL configured "
                f"for etsAuth={ets_auth}"
            )

        return base_url.rstrip("/")    

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
        base_url = self._get_base_url(ets_auth)
        url =  f"{base_url}/printCtrl?tp=unknown&auth={ets_auth}"
        
        response = await self._client.post(
            url,
            json=payload,
            headers=self._headers(),
        )

        response.raise_for_status()

        logger.info(
            "Unknown persons sent to ERP: count=%d",
            len(unknown_persons),
        )

    async def get_unknown_assignments(
        self,
        *,
        ets_auth: str,
    ) -> list[dict]:

        base_url = self._get_base_url(
            ets_auth
        )

        url =  f"{base_url}/printCtrl?tp=unknown&auth={ets_auth}&sync=1"

        response = await self._client.get(url,)

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list):
            return []

        return data

    def get_configured_ets_auths(
        self,
    ) -> list[str]:

        return list(
            self.urls_by_ets_auth.keys()
        )

    async def close(self) -> None:
        await self._client.aclose() 