from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

router = APIRouter()


@router.post("/all")
async def sync_all(request: Request, payload: Any = Body(default=None)) -> dict:
    if payload is not None:
        result = {}
        if isinstance(payload, dict) and "cameras" in payload:
            result["cameras"] = await request.app.state.sync_service.sync_cameras_from_payload(payload["cameras"])
        if isinstance(payload, dict) and "employees" in payload:
            result["employees"] = await request.app.state.sync_service.sync_employees_from_payload(payload["employees"])
        if isinstance(payload, dict) and "rules" in payload:
            result["rules"] = await request.app.state.sync_service.sync_rules_from_payload(payload["rules"])
        return result
    return await request.app.state.sync_service.sync_all()


@router.post("/cameras")
async def sync_cameras(request: Request, payload: Any = Body(default=None)) -> dict:
    if payload is not None:
        return await request.app.state.sync_service.sync_cameras_from_payload(payload)
    return await request.app.state.sync_service.sync_cameras()


@router.post("/employees")
async def sync_employees(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    payload: Any = Body(default=None),
) -> dict:
    if payload is not None:
        return await request.app.state.sync_service.sync_employees_from_payload(payload)
    return await request.app.state.sync_service.sync_employees(_tenant_id(tenantId, etsAuth))


@router.post("/rules")
async def sync_rules(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    payload: Any = Body(default=None),
) -> dict:
    if payload is not None:
        return await request.app.state.sync_service.sync_rules_from_payload(payload)
    return await request.app.state.sync_service.sync_rules(_tenant_id(tenantId, etsAuth))


def _tenant_id(tenant_id: str | None, ets_auth: str | None) -> str | None:
    if tenant_id and ets_auth and tenant_id != ets_auth:
        raise HTTPException(status_code=400, detail="tenantId and etsAuth must match when both are provided")
    return tenant_id or ets_auth

