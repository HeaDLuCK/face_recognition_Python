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

