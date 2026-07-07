import base64
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.database import serialize_mongo_docs

router = APIRouter()

UNKNOWN_FACE_STATUSES = {"NEW", "UPLOADED", "ASSIGNED", "IGNORED", "EXPIRED", "FAILED"}


class UnknownFaceStatusUpdate(BaseModel):
    status: str
    cloudFaceId: str | None = None
    cloudImageUrl: str | None = None
    uploadError: str | None = None


@router.get("")
async def list_unknown_faces(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    cameraId: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    query = _unknown_face_query(_tenant_id(tenantId, etsAuth), cameraId, status)
    cursor = (
        request.app.state.db.unknown_face_crops.find(query, {"embedding": 0})
        .sort("createdAt", -1)
        .limit(limit)
    )
    items = serialize_mongo_docs(await cursor.to_list(length=limit))
    return [_with_image_url(request, item) for item in items]


@router.get("/cloud-payload")
async def list_unknown_faces_cloud_payload(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    cameraId: str | None = None,
    status: str | None = Query(default="NEW"),
    limit: int = Query(default=50, ge=1, le=100),
    ttlHours: int = Query(default=24, ge=1, le=720),
) -> dict:
    tenant_id = _tenant_id(tenantId, etsAuth)
    query = _unknown_face_query(tenant_id, cameraId, status)
    cursor = (
        request.app.state.db.unknown_face_crops.find(query, {"embedding": 0})
        .sort("createdAt", -1)
        .limit(limit)
    )
    items = serialize_mongo_docs(await cursor.to_list(length=limit))
    faces = []
    skipped = []

    for item in items:
        unknown_face_id = item.get("unknownFaceCropId")
        if not unknown_face_id:
            skipped.append({"reason": "missing_unknownFaceCropId", "path": item.get("path")})
            continue

        try:
            path = _safe_snapshot_path(item.get("path"), request.app.state.snapshot_service.snapshot_dir)
        except HTTPException as exc:
            skipped.append({"unknownFaceCropId": unknown_face_id, "reason": exc.detail})
            continue

        if not path.exists() or not path.is_file():
            skipped.append({"unknownFaceCropId": unknown_face_id, "reason": "image_file_not_found"})
            continue

        image_bytes = path.read_bytes()
        faces.append(
            {
                "unknownFaceCropId": unknown_face_id,
                "etsAuth": item.get("etsAuth"),
                "cameraId": item.get("cameraId"),
                "createdAt": item.get("createdAt"),
                "status": item.get("status", "NEW"),
                "localPath": item.get("path"),
                "fileName": path.name,
                "contentType": "image/jpeg",
                "imageBase64": base64.b64encode(image_bytes).decode("ascii"),
                "metadata": item.get("metadata", {}),
                "ttlHours": ttlHours,
            }
        )

    return {
        "etsAuth": tenant_id,
        "count": len(faces),
        "faces": faces,
        "skipped": skipped,
    }


@router.get("/faces")
async def list_unknown_faces_base64(
    request: Request,
    tenantId: str | None = Query(default=None),
    etsAuth: str | None = Query(default=None),
    cameraId: str | None = None,
    status: str | None = Query(default="NEW"),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[dict]:
    tenant_id = _tenant_id(tenantId, etsAuth)
    query = _unknown_face_query(tenant_id, cameraId, status)
    cursor = (
        request.app.state.db.unknown_face_crops.find(query, {"embedding": 0})
        .sort("createdAt", -1)
        .limit(limit)
    )
    items = serialize_mongo_docs(await cursor.to_list(length=limit))
    faces = []

    for item in items:
        unknown_face_id = item.get("unknownFaceCropId")
        if not unknown_face_id:
            continue

        try:
            path = _safe_snapshot_path(item.get("path"), request.app.state.snapshot_service.snapshot_dir)
        except HTTPException:
            continue

        if not path.exists() or not path.is_file():
            continue

        faces.append(
            {
                "unknownFaceCropId": unknown_face_id,
                "etsAuth": item.get("etsAuth"),
                "cameraId": item.get("cameraId"),
                "createdAt": item.get("createdAt"),
                "status": item.get("status", "NEW"),
                "fileName": path.name,
                "contentType": "image/jpeg",
                "imageBase64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )

    return faces


@router.post("/{unknownFaceCropId}/status")
async def update_unknown_face_status(
    unknownFaceCropId: str,
    request: Request,
    payload: UnknownFaceStatusUpdate = Body(...),
) -> dict:
    status = payload.status.strip().upper()
    if status not in UNKNOWN_FACE_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status. Use one of: {sorted(UNKNOWN_FACE_STATUSES)}")

    update = {
        "status": status,
        "updatedAt": datetime.utcnow(),
    }
    if payload.cloudFaceId:
        update["cloudFaceId"] = payload.cloudFaceId
    if payload.cloudImageUrl:
        update["cloudImageUrl"] = payload.cloudImageUrl
    if payload.uploadError:
        update["uploadError"] = payload.uploadError
    if status == "UPLOADED":
        update["uploadedAt"] = datetime.utcnow()

    result = await request.app.state.db.unknown_face_crops.update_one(
        {"unknownFaceCropId": unknownFaceCropId},
        {"$set": update},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Unknown face crop not found")

    return {"unknownFaceCropId": unknownFaceCropId, "status": status}


def _unknown_face_query(tenant_id: str, camera_id: str | None, status: str | None) -> dict:
    query = {"etsAuth": tenant_id}
    if camera_id:
        query["cameraId"] = camera_id
    if status:
        normalized_status = status.strip().upper()
        if normalized_status == "NEW":
            query["$or"] = [{"status": "NEW"}, {"status": {"$exists": False}}]
        else:
            query["status"] = normalized_status
    return query


def _with_image_url(request: Request, item: dict) -> dict:
    unknown_face_id = item.get("unknownFaceCropId")
    item.setdefault("status", "NEW")
    if unknown_face_id:
        item["imageUrl"] = str(
            request.url_for(
                "get_unknown_face_image",
                unknownFaceCropId=unknown_face_id,
            )
        )
    return item


@router.get("/{unknownFaceCropId}/image")
async def get_unknown_face_image(unknownFaceCropId: str, request: Request) -> FileResponse:
    item = await request.app.state.db.unknown_face_crops.find_one(
        {"unknownFaceCropId": unknownFaceCropId},
        {"_id": 0, "path": 1},
    )
    if not item:
        raise HTTPException(status_code=404, detail="Unknown face crop not found")

    path = _safe_snapshot_path(item.get("path"), request.app.state.snapshot_service.snapshot_dir)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown face image file not found")

    return FileResponse(path, media_type="image/jpeg", filename=path.name)


def _tenant_id(tenant_id: str | None, ets_auth: str | None) -> str:
    value = tenant_id or ets_auth
    if not value:
        raise HTTPException(status_code=422, detail="tenantId or etsAuth is required")
    return value


def _safe_snapshot_path(raw_path: str | None, snapshot_dir: Path) -> Path:
    if not raw_path:
        raise HTTPException(status_code=404, detail="Unknown face image path is missing")

    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    resolved_path = path.resolve()
    resolved_snapshot_dir = snapshot_dir.resolve()
    if resolved_snapshot_dir not in resolved_path.parents and resolved_path != resolved_snapshot_dir:
        raise HTTPException(status_code=403, detail="Image path is outside snapshot directory")
    return resolved_path
