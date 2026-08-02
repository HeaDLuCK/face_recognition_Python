from fastapi import APIRouter, HTTPException, Query, Request


router = APIRouter()


@router.get("")
async def person_counts(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    cameraId: str | None = None,
) -> dict:
    tenant_id = tenantId or etsAuth
    if not tenant_id:
        raise HTTPException(status_code=422, detail="tenantId or etsAuth is required")
    return await request.app.state.event_service.summarize_person_counts(
        tenant_id=tenant_id,
        camera_id=cameraId,
    )
