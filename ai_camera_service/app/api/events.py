from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


@router.get("")
async def list_events(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    cameraId: str | None = None,
    employeeId: str | None = None,
    direction: str | None = None,
    eventType: str | None = None,
) -> list[dict]:
    return await request.app.state.event_service.list_events(
        tenant_id=_tenant_id(tenantId, etsAuth),
        limit=limit,
        camera_id=cameraId,
        employee_id=employeeId,
        direction=direction,
        event_type=eventType,
    )


def _tenant_id(tenant_id: str | None, ets_auth: str | None) -> str:
    value = tenant_id or ets_auth
    if not value:
        raise HTTPException(status_code=422, detail="tenantId or etsAuth is required")
    return value
