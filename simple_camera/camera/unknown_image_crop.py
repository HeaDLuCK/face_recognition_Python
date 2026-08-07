import numpy as np


BBox = tuple[int, int, int, int]


def crop_unknown_face(
    frame: np.ndarray,
    bbox: BBox,
    padding_ratio: float = 0.25,
) -> np.ndarray | None:
    """
    Create a padded face crop.

    This is the image that can later be used for:
    - displaying the unknown face;
    - administrator verification;
    - generating an employee embedding after assignment.

    bbox:
        (x1, y1, x2, y2)
    """

    if frame is None or frame.size == 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 0 or face_height <= 0:
        return None

    pad_x = int(
        face_width * padding_ratio
    )

    pad_y = int(
        face_height * padding_ratio
    )

    crop_x1 = max(
        0,
        x1 - pad_x,
    )

    crop_y1 = max(
        0,
        y1 - pad_y,
    )

    crop_x2 = min(
        frame_width,
        x2 + pad_x,
    )

    crop_y2 = min(
        frame_height,
        y2 + pad_y,
    )

    if (
        crop_x2 <= crop_x1
        or crop_y2 <= crop_y1
    ):
        return None

    return frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()


def crop_unknown_context(
    frame: np.ndarray,
    bbox: BBox,
) -> np.ndarray | None:
    """
    Create a larger context image around the detected face.

    This is mainly for the administrator.

    It includes:
    - face;
    - head;
    - shoulders;
    - some upper body;
    """

    if frame is None or frame.size == 0:
        return None

    frame_height, frame_width = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 0 or face_height <= 0:
        return None

    # Wider area around the face.
    extra_left = int(
        face_width * 0.8
    )

    extra_right = int(
        face_width * 0.8
    )

    # Keep some area above the head.
    extra_top = int(
        face_height * 0.5
    )

    # Include shoulders and upper body.
    extra_bottom = int(
        face_height * 2.5
    )

    crop_x1 = max(
        0,
        x1 - extra_left,
    )

    crop_y1 = max(
        0,
        y1 - extra_top,
    )

    crop_x2 = min(
        frame_width,
        x2 + extra_right,
    )

    crop_y2 = min(
        frame_height,
        y2 + extra_bottom,
    )

    if (
        crop_x2 <= crop_x1
        or crop_y2 <= crop_y1
    ):
        return None

    return frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()


def face_is_large_enough(
    bbox: BBox,
    minimum_width: int = 80,
    minimum_height: int = 80,
) -> bool:
    """
    Reject faces that are too small to be useful.

    We can tune these values later using your real CCTV footage.
    """

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    return (
        face_width >= minimum_width
        and face_height >= minimum_height
    )