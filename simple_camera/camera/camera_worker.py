import cv2

from camera.camera_manager import CameraManager
from schemas.project_schema import (
    AiCapability,
    CameraConfig,
)


def read_camera(
    camera_data: dict,
) -> None:
    # Recreate the Pydantic model inside the child process.
    config = CameraConfig.model_validate(
        camera_data
    )

    camera = CameraManager(config)

    face_assignments = config.assignments_for(
        AiCapability.FACE_RECOGNITION
    )

    print(
        f"{camera.camera_id}: face recognition tenants: "
        f"{[item.tenantId for item in face_assignments]}",
        flush=True,
    )

    capture = camera.start_camera()

    window_name = (
        f"{camera.camera_id} - {camera.name}"
    )

    try:
        while True:
            ret, frame = capture.read()

            if not ret or frame is None:
                print(
                    f"{camera.camera_id}: "
                    "failed to read frame",
                    flush=True,
                )
                break

            height, width = frame.shape[:2]

            display_width = 1280
            display_height = int(
                height * display_width / width
            )

            display_frame = cv2.resize(
                frame,
                (
                    display_width,
                    display_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

            # Put InsightFace processing here.

            cv2.imshow(
                window_name,
                display_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as exc:
        print(
            f"{camera.camera_id}: {exc}",
            flush=True,
        )

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print(
            f"{camera.camera_id}: camera stopped",
            flush=True,
        )