from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("")
async def list_events(
    request: Request,
    tenantId: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
    cameraId: str | None = None,
    employeeId: str | None = None,
) -> list[dict]:
    return await request.app.state.event_service.list_events(
        tenant_id=tenantId,
        limit=limit,
        camera_id=cameraId,
        employee_id=employeeId,
    )
