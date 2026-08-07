import logging
from datetime import datetime, timezone
from time import monotonic

import cv2
import numpy as np

from config import Settings
from tracking.models import PersonTrack
from tracking.person_detection_service import (
    PersonDetectionService,
)
from tracking.person_tracker import PersonTracker


logger = logging.getLogger(__name__)


class PersonTrackingRuntime:
    """
    Connect person detection and person tracking.

    process_frame() modifies the provided frame by drawing:
    - body bounding boxes;
    - track IDs;
    - MOVING or STILL;
    - recent movement trajectories.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.detector = PersonDetectionService(
            settings=settings
        )

        self.tracker = PersonTracker(
            iou_threshold=(
                settings.person_tracker_iou_threshold
            ),
            timeout_seconds=(
                settings.person_tracker_timeout_seconds
            ),
            movement_threshold_ratio=(
                settings
                .person_movement_threshold_ratio
            ),
            minimum_movement_pixels=(
                settings
                .person_minimum_movement_pixels
            ),
            trajectory_size=(
                settings.person_trajectory_size
            ),
        )

    def warm_up(self) -> None:
        self.detector.warm_up()

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> tuple[
        list[PersonTrack],
        list[PersonTrack],
    ]:
        observed_at = datetime.now(
            timezone.utc
        )

        observed_monotonic = monotonic()

        detections = self.detector.detect_frame(
            frame
        )

        updated_tracks, expired_tracks = (
            self.tracker.update(
                detections=detections,
                observed_at=observed_at,
                observed_monotonic=(
                    observed_monotonic
                ),
            )
        )

        if draw:
            self.draw_tracks(
                frame=frame,
                tracks=self.tracker.active_tracks(),
            )

        return updated_tracks, expired_tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[PersonTrack],
    ) -> None:
        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            status = (
                "MOVING"
                if track.is_moving
                else "STILL"
            )

            box_color = (
                (0, 255, 0)
                if track.is_moving
                else (255, 0, 0)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            label = (
                f"ID {track.track_id} "
                f"{status} "
                f"{track.movement_distance:.1f}px "
                f"{track.confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )

            trajectory = list(
                track.trajectory
            )

            for index in range(
                1,
                len(trajectory),
            ):
                previous_point = (
                    int(
                        trajectory[index - 1][0]
                    ),
                    int(
                        trajectory[index - 1][1]
                    ),
                )

                current_point = (
                    int(
                        trajectory[index][0]
                    ),
                    int(
                        trajectory[index][1]
                    ),
                )

                cv2.line(
                    frame,
                    previous_point,
                    current_point,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if trajectory:
                latest_point = trajectory[-1]

                cv2.circle(
                    frame,
                    (
                        int(latest_point[0]),
                        int(latest_point[1]),
                    ),
                    5,
                    (0, 0, 255),
                    -1,
                )

    def reset(self) -> None:
        """
        Call after RTSP reconnection.
        """

        self.tracker.clear()