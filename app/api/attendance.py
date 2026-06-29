from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("", response_class=PlainTextResponse)
async def list_attendance(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    employeeId: str | None = None,
    cameraId: str | None = None,
    direction: str | None = None,
    eventType: str | None = None,
) -> str:
    return await request.app.state.attendance_service.list_attendance(
        tenant_id=_tenant_id(tenantId, etsAuth),
        limit=limit,
        employee_id=employeeId,
        camera_id=cameraId,
        direction=direction,
        event_type=eventType,
    )


def _tenant_id(tenant_id: str | None, ets_auth: str | None) -> str:
    value = tenant_id or ets_auth
    if not value:
        raise HTTPException(status_code=422, detail="tenantId or etsAuth is required")
    return value
