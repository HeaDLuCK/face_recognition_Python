from camera.camera_manager import CameraManager
from face_recognition import InsightFaceEngine
from database import close_mongo_connection, connect_to_mongo
import cv2
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import multiprocessing as mp
from service.embedding_service import EmbeddingService
from service.embedding_index import EmbeddingIndex,best_embedding_candidate
import numpy as np
from service.sync_service import SyncService
from schemas.project_schema import CameraConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def read_camera(camera_data: dict,
                embedding_index: EmbeddingIndex) -> None:


    camera_config = CameraConfig.model_validate(camera_data)
    camera = CameraManager(camera_config)
    engine = InsightFaceEngine(
            model_name="buffalo_s",
            providers=["CPUExecutionProvider"],
            ctx_id=-1,
            det_size=640,
            min_score=0.6,
        )
    capture = camera.start_camera()
    window_name = f"Camera {camera.camera_id}"
    frame_number = 0
    detected_faces = []
    try:
        while True:
            ret, frame = capture.read()

            if not ret:
                print(f": failed to read frame", flush=True)
                break

            frame_number += 1
            if frame_number % 4 == 0:
                detected_faces = engine.detect_faces(frame)

                for detected_face in detected_faces:
                    detected_embedding = np.asarray(
                        detected_face.embedding,
                        dtype=np.float32,
                    )

                    match = best_embedding_candidate(
                        embedding_index,
                        detected_embedding,
                    )

                    x1, y1, x2, y2 = detected_face.bbox
                    label = "Unknown"
                    if match is None or match["score"] < 0.50:
                        label = "Unknown"
                    else:
                        print(match['etsAuth'])
                        label = (
                            f"{match['employeeName'] or match['employeeId']} "
                            f"{match['score']:.2f}"
                        )
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            display_frame = cv2.resize(
                frame,
                (1280, 720),
                interpolation=cv2.INTER_AREA,
            )    
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    finally:
        capture.release()
        cv2.destroyWindow(window_name)
        print(f": camera stopped", flush=True)



@asynccontextmanager
async def lifespan(app: FastAPI):
    database = await connect_to_mongo()
    embedding_service = EmbeddingService(database)
    app.state.embedding_service = embedding_service


    face_engine = InsightFaceEngine(
                model_name="buffalo_s",
                providers=["CPUExecutionProvider"],
                ctx_id=-1,
                det_size=640,
                min_score=0.6,
            )
    embedding_index = await embedding_service.refresh_all_embeddings_index()

    sync_service = SyncService(
        db=database,
        embedding_service=embedding_service,
        face_engine=face_engine,
    )
    app.state.sync_service = sync_service
    cameras = await sync_service.get_all_cameras()
    processes = []

    for camera in cameras:
        process = mp.Process(
            target=read_camera,
            args=(
                camera.model_dump(mode="json"),
                embedding_index,
            ),
        )
        process.start()
        processes.append(process)

    app.state.sync_service = sync_service
    app.state.camera_processes = processes


    logging.info("Camera process started with PID %s", process.pid)
    try:
        yield
    finally:
        if process.is_alive():
            process.terminate()

        process.join(timeout=5)

        await close_mongo_connection()
        logging.info("FastAPI shutdown completed")




app = FastAPI(lifespan=lifespan)



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

# capture = camera.start_camera()
# while True:
#     ret,frame = capture.read()
#     if not ret:
#         print("Error: Can't receive frame. Exiting...")
#         break

#     cv2.imshow('Webcam Video', frame)


#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break