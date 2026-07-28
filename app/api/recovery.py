from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


class ManualRecoveryJobRequest(BaseModel):
    etsAuth: str | None = None
    cameraId: str
    timestamp: str | None = None
    windowStart: str | None = None
    windowEnd: str | None = None
    beforeSeconds: int = Field(default=10, ge=0, le=300)
    afterSeconds: int = Field(default=10, ge=1, le=300)


@router.get("")
async def list_recovery_jobs(
    request: Request,
    etsAuth: str | None = None,
    cameraId: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    return await request.app.state.attendance_recovery_service.list_jobs(
        tenant_id=etsAuth,
        camera_id=cameraId,
        status=status,
        limit=limit,
    )


@router.post("/manual")
async def create_manual_recovery_job(
    payload: ManualRecoveryJobRequest,
    request: Request,
) -> dict:
    try:
        if payload.windowStart and payload.windowEnd:
            window_start = _parse_utc_datetime(payload.windowStart)
            window_end = _parse_utc_datetime(payload.windowEnd)
        elif payload.timestamp:
            timestamp = _parse_utc_datetime(payload.timestamp)
            window_start = timestamp - _seconds(payload.beforeSeconds)
            window_end = timestamp + _seconds(payload.afterSeconds)
        else:
            raise ValueError("Send timestamp or windowStart + windowEnd")

        recovery_job_id = await request.app.state.attendance_recovery_service.enqueue_window(
            tenant_id=payload.etsAuth,
            camera_id=payload.cameraId,
            window_start=window_start,
            window_end=window_end,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "recoveryJobId": recovery_job_id,
        "status": "PENDING",
        "etsAuth": payload.etsAuth,
        "cameraId": payload.cameraId,
        "windowStart": window_start.isoformat(),
        "windowEnd": window_end.isoformat(),
    }


@router.post("/{recoveryJobId}/retry")
async def retry_recovery_job(recoveryJobId: str, request: Request) -> dict:
    retried = await request.app.state.attendance_recovery_service.retry_job(
        recoveryJobId
    )
    if not retried:
        raise HTTPException(status_code=404, detail="Recovery job not found")
    return {"recoveryJobId": recoveryJobId, "status": "PENDING"}


@router.post("/delete-duplicates")
async def delete_duplicate_recovery_jobs(request: Request) -> dict:
    return await request.app.state.attendance_recovery_service.delete_duplicate_jobs()


@router.post("/cancel-duplicates")
async def cancel_duplicate_recovery_jobs(request: Request) -> dict:
    return await request.app.state.attendance_recovery_service.delete_duplicate_jobs()


def _parse_utc_datetime(raw_value: str) -> datetime:
    value = raw_value.strip()
    if not value:
        raise ValueError("Datetime value is required")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Datetime must be ISO format, for example 2026-07-27T17:30:00Z") from exc
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _seconds(value: int):
    return timedelta(seconds=value)
