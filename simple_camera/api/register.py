import asyncio

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile


router = APIRouter()


@router.post("/register-face")
async def register_face(
    request: Request,
    image: UploadFile = File(...),
    etsAuth: str = Form(default="TEST"),
    employeeId: str = Form(default="ME"),
    employeeName: str = Form(default="Test User"),
) -> dict:
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Image is empty",
        )

    sync_service = request.app.state.sync_service

    faces = await asyncio.to_thread(
        sync_service.face_engine.extract_embeddings_from_image_bytes,
        image_bytes,
    )

    if not faces:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image",
        )

    if len(faces) > 1:
        raise HTTPException(
            status_code=400,
            detail="The image must contain only one face",
        )

    embeddings = [
        face.embedding
        for face in faces
    ]

    await sync_service.embedding_service.upsert_employee_embeddings(
        etsAuth=etsAuth,
        employee_id=employeeId,
        employee_name=employeeName,
        embeddings=embeddings,
        source_id=f"manual-upload:{employeeId}",
    )

    # Reload embeddings and restart the camera processes.
    restart_result = (
        await request.app.state.camera_process_manager.restart_all()
    )

    return {
        "registered": True,
        "etsAuth": etsAuth,
        "employeeId": employeeId,
        "employeeName": employeeName,
        "facesDetected": len(faces),
        "restart": restart_result,
    }