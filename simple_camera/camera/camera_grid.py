import math
import time
from queue import Empty
from typing import Any

import cv2
import numpy as np


def fit_camera_inside_tile(
    frame: np.ndarray,
    tile_width: int,
    tile_height: int,
) -> np.ndarray:

    frame_height, frame_width = frame.shape[:2]

    scale = min(
        tile_width / frame_width,
        tile_height / frame_height,
    )

    resized_width = max(
        1,
        int(frame_width * scale),
    )
    resized_height = max(
        1,
        int(frame_height * scale),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    tile = np.zeros(
        (tile_height, tile_width, 3),
        dtype=np.uint8,
    )

    x = (tile_width - resized_width) // 2
    y = (tile_height - resized_height) // 2

    tile[
        y:y + resized_height,
        x:x + resized_width,
    ] = resized

    return tile


def show_camera_grid(
    camera_ids: list[str],
    frame_queues: dict[str, Any],
    stop_event: Any,
) -> None:
    if not camera_ids:
        return

    window_name = "All Cameras"

    # Entire OpenCV image/window resolution.
    grid_width = 1280
    grid_height = 720

    # Each camera will be smaller than the complete grid.
    tile_width = 560
    tile_height = 315

    gap = 30
    title_height = 35

    camera_count = len(camera_ids)

    if camera_count == 1:
        columns = 1
    elif camera_count <= 4:
        columns = 2
    else:
        columns = 3

    rows = math.ceil(camera_count / columns)

    latest_frames: dict[str, np.ndarray] = {
        camera_id: np.zeros(
            (tile_height, tile_width, 3),
            dtype=np.uint8,
        )
        for camera_id in camera_ids
    }

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        grid_width,
        grid_height,
    )

    try:
        while not stop_event.is_set():
            # Receive the latest frame from every camera.
            for camera_id in camera_ids:
                camera_queue = frame_queues[camera_id]

                while True:
                    try:
                        latest_frames[camera_id] = (
                            camera_queue.get_nowait()
                        )
                    except Empty:
                        break

            # Always create a fixed 1280 × 720 grid.
            grid = np.zeros(
                (grid_height, grid_width, 3),
                dtype=np.uint8,
            )

            complete_width = (
                columns * tile_width
                + (columns - 1) * gap
            )

            complete_height = (
                rows * (tile_height + title_height)
                + (rows - 1) * gap
            )

            # Center all camera tiles inside the 1280 × 720 grid.
            start_x = max(
                0,
                (grid_width - complete_width) // 2,
            )

            start_y = max(
                0,
                (grid_height - complete_height) // 2,
            )

            for index, camera_id in enumerate(camera_ids):
                row = index // columns
                column = index % columns

                x = start_x + column * (
                    tile_width + gap
                )

                y = start_y + row * (
                    tile_height
                    + title_height
                    + gap
                )

                frame = latest_frames[camera_id]

                tile = fit_camera_inside_tile(
                    frame,
                    tile_width=tile_width,
                    tile_height=tile_height,
                )

                # Camera name above its tile.
                cv2.putText(
                    grid,
                    camera_id,
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                end_x = x + tile_width
                end_y = y + tile_height

                # Protect against an invalid layout.
                if (
                    end_x <= grid_width
                    and end_y <= grid_height
                ):
                    grid[
                        y:end_y,
                        x:end_x,
                    ] = tile

                    cv2.rectangle(
                        grid,
                        (x, y),
                        (end_x - 1, end_y - 1),
                        (80, 80, 80),
                        2,
                    )

            cv2.imshow(
                window_name,
                grid,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            time.sleep(0.01)

    finally:
        stop_event.set()
        cv2.destroyAllWindows()