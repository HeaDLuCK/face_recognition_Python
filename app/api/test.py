from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter()


@router.post("/recognize-image")
async def recognize_image(
    request: Request,
    tenantId: str = Form(...),
    threshold: float | None = Form(default=None),
    file: UploadFile = File(...),
) -> dict:
    image_bytes = await file.read()
    rules = request.app.state.runtime_state.get_rules(tenantId)
    recognition_threshold = threshold if threshold is not None else rules.recognitionThreshold
    try:
        results = await request.app.state.recognition_service.recognize_image_bytes(
            tenant_id=tenantId,
            image_bytes=image_bytes,
            threshold=recognition_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "tenantId": tenantId,
        "threshold": recognition_threshold,
        "facesDetected": len(results),
        "results": results,
    }


@router.get("/recognition-debug")
async def recognition_debug(request: Request, tenantId: str, cameraId: str | None = None) -> dict:
    embeddings_count = await request.app.state.embedding_service.count_tenant_embeddings(tenantId)
    camera_status = request.app.state.camera_manager.status()
    cameras = camera_status["cameras"]
    if cameraId:
        cameras = [camera for camera in cameras if camera["cameraId"] == cameraId]

    latest_detections = {}
    for current_camera_id, worker in request.app.state.camera_manager.workers.items():
        if cameraId and current_camera_id != cameraId:
            continue
        latest_detections[current_camera_id] = worker.latest_detections

    return {
        "tenantId": tenantId,
        "embeddingsCount": embeddings_count,
        "cameras": cameras,
        "latestDetections": latest_detections,
        "lastSync": request.app.state.runtime_state.last_sync,
    }

