from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    await request.app.state.db.command("ping")
    return {"status": "ok", "service": "ai_camera_service"}


@router.get("/api/status")
async def status(request: Request) -> dict:
    return {
        "status": "ok",
        "service": "ai_camera_service",
        "attendanceRecovery": request.app.state.attendance_recovery_service.status(),
        **request.app.state.camera_manager.status(),
    }
