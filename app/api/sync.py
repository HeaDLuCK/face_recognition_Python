from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

router = APIRouter()


@router.post("/all")
async def sync_all(
    request: Request,
    payload: Any = Body(default=None),
    restart: bool = Query(default=True),
) -> dict:
    if payload is  None:
        raise HTTPException(status_code=400, detail="Payload is Empty")
    result = {}
    if isinstance(payload, dict) and "cameras" in payload:
        try:
            result["cameras"] = await request.app.state.sync_service.sync_cameras_from_payload(payload["cameras"])
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(payload, dict) and "employees" in payload:
        result["employees"] = await request.app.state.sync_service.sync_employees_from_payload(payload["employees"])
    if isinstance(payload, dict) and "rules" in payload:
        result["rules"] = await request.app.state.sync_service.sync_rules_from_payload(payload["rules"])
    return await _restart_if_requested(request, result, restart)


@router.post("/cameras")
async def sync_cameras(
    request: Request,
    payload: Any = Body(default=None),
    restart: bool = Query(default=True),
) -> dict:
    try:
        if payload is not None:
            result = await request.app.state.sync_service.sync_cameras_from_payload(payload)
        else:
            raise HTTPException(status_code=400, detail="Payload is Empty")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return await _restart_if_requested(request, result, restart)


@router.post("/employees")
async def sync_employees(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    restart: bool = Query(default=True),
    payload: Any = Body(default=None),
) -> dict:
    if payload is not None:
        result = await request.app.state.sync_service.sync_employees_from_payload(payload)
    else:
        raise HTTPException(status_code=400, detail="Payload is Empty")
    return await _restart_if_requested(request, result, restart)


@router.post("/rules")
async def sync_rules(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    restart: bool = Query(default=True),
    payload: Any = Body(default=None),
) -> dict:
    if payload is not None:
        result = await request.app.state.sync_service.sync_rules_from_payload(payload)
    else:
        raise HTTPException(status_code=400, detail="Payload is Empty")
    return await _restart_if_requested(request, result, restart)


async def _restart_if_requested(request: Request, result: dict, restart: bool) -> dict:
    if restart:
        result["restart"] = await request.app.state.camera_manager.restart_all()
    return result


def _tenant_id(tenant_id: str | None, ets_auth: str | None) -> str | None:
    if tenant_id and ets_auth and tenant_id != ets_auth:
        raise HTTPException(status_code=400, detail="tenantId and etsAuth must match when both are provided")
    return tenant_id or ets_auth

