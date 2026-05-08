from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("")
async def list_attendance(
    request: Request,
    tenantId: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
    employeeId: str | None = None,
) -> list[dict]:
    return await request.app.state.attendance_service.list_attendance(
        tenant_id=tenantId,
        limit=limit,
        employee_id=employeeId,
    )
