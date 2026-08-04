import math
import time
from queue import Empty
from typing import Any

import cv2
import numpy as np


def show_camera_grid(
    camera_ids: list[str],
    frame_queues: dict[str, Any],
    stop_event: Any,
) -> None:
    if not camera_ids:
        return

    window_name = "All Cameras"

    tile_width = 480
    tile_height = 270
    columns = min(4, len(camera_ids))
    rows = math.ceil(len(camera_ids) / columns)

    latest_frames: dict[str, np.ndarray] = {
        camera_id: np.zeros(
            (tile_height, tile_width, 3),
            dtype=np.uint8,
        )
        for camera_id in camera_ids
    }

    try:
        while not stop_event.is_set():
            for camera_id in camera_ids:
                frame_queue = frame_queues[camera_id]

                # Read every queued item and retain the newest one.
                while True:
                    try:
                        latest_frames[camera_id] = (
                            frame_queue.get_nowait()
                        )
                    except Empty:
                        break

            tiles: list[np.ndarray] = []
            gap = 10
            for camera_id in camera_ids:
                tile = latest_frames[camera_id].copy()

                cv2.putText(
                    tile,
                    camera_id,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                tile_with_gap = cv2.copyMakeBorder(
                    tile,
                    gap,
                    gap,
                    gap,
                    gap,
                    cv2.BORDER_CONSTANT,
                    value=(40, 40, 40),
                )
                tiles.append(tile_with_gap )

            number_of_tiles = rows * columns

            while len(tiles) < number_of_tiles:
                tiles.append(
                    np.zeros(
                        (tile_height, tile_width, 3),
                        dtype=np.uint8,
                    )
                )

            grid_rows: list[np.ndarray] = []

            for row_index in range(rows):
                start = row_index * columns
                end = start + columns

                grid_rows.append(
                    np.hstack(tiles[start:end])
                )

            grid = np.vstack(grid_rows)

            cv2.imshow(window_name, grid)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break

            time.sleep(0.01)

    finally:
        stop_event.set()
        cv2.destroyAllWindows()