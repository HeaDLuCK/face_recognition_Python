from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.post("/all")
async def sync_all(request: Request) -> dict:
    return await request.app.state.sync_service.sync_all()


@router.post("/cameras")
async def sync_cameras(request: Request) -> dict:
    return await request.app.state.sync_service.sync_cameras()


@router.post("/employees")
async def sync_employees(request: Request, tenantId: str | None = Query(default=None)) -> dict:
    return await request.app.state.sync_service.sync_employees(tenantId)


@router.post("/rules")
async def sync_rules(request: Request, tenantId: str | None = Query(default=None)) -> dict:
    return await request.app.state.sync_service.sync_rules(tenantId)

