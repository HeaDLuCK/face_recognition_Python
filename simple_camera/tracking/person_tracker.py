import math
from collections import deque
from datetime import datetime

from tracking.models import (
    BBox,
    Point,
    PersonDetection,
    PersonTrack,
)


class PersonTracker:
    """
    Basic body tracker.

    Matching uses:

    1. Bounding-box overlap.
    2. Distance between bounding-box centers.

    Body movement uses the bottom-center point of the body box.
    """

    def __init__(
        self,
        iou_threshold: float,
        timeout_seconds: float,
        movement_threshold_ratio: float = 0.02,
        minimum_movement_pixels: float = 3.0,
        trajectory_size: int = 30,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be greater than 0 "
                "and less than or equal to 1"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0"
            )

        if movement_threshold_ratio < 0:
            raise ValueError(
                "movement_threshold_ratio cannot be negative"
            )

        if minimum_movement_pixels < 0:
            raise ValueError(
                "minimum_movement_pixels cannot be negative"
            )

        if trajectory_size <= 0:
            raise ValueError(
                "trajectory_size must be greater than 0"
            )

        self.iou_threshold = iou_threshold
        self.timeout_seconds = timeout_seconds

        self.movement_threshold_ratio = (
            movement_threshold_ratio
        )

        self.minimum_movement_pixels = (
            minimum_movement_pixels
        )

        self.trajectory_size = trajectory_size

        self._tracks: dict[int, PersonTrack] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[PersonDetection],
        observed_at: datetime,
        observed_monotonic: float,
    ) -> tuple[list[PersonTrack], list[PersonTrack]]:
        """
        Update tracks with current-frame detections.

        Returns:
            updated_tracks, expired_tracks
        """

        expired_tracks = self.expire(
            observed_monotonic
        )

        matches = self._match(
            detections
        )

        matched_track_ids = set(
            matches.keys()
        )

        matched_detection_indexes: set[int] = set()
        updated_tracks: list[PersonTrack] = []

        # Tracks not matched in this frame are not moving.
        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue

            track.movement_distance = 0.0
            track.is_moving = False

        # Update existing tracks.
        for track_id, detection_index in matches.items():
            detection = detections[detection_index]
            track = self._tracks[track_id]

            self._update_track(
                track=track,
                detection=detection,
                observed_at=observed_at,
                observed_monotonic=(
                    observed_monotonic
                ),
            )

            matched_detection_indexes.add(
                detection_index
            )

            updated_tracks.append(
                track
            )

        # Create tracks for new people.
        for detection_index, detection in enumerate(
            detections
        ):
            if (
                detection_index
                in matched_detection_indexes
            ):
                continue

            track = self._create_track(
                detection=detection,
                observed_at=observed_at,
                observed_monotonic=(
                    observed_monotonic
                ),
            )

            self._tracks[track.track_id] = track

            updated_tracks.append(
                track
            )

        return updated_tracks, expired_tracks

    def _update_track(
        self,
        track: PersonTrack,
        detection: PersonDetection,
        observed_at: datetime,
        observed_monotonic: float,
    ) -> None:
        previous_bbox = track.bbox
        current_bbox = detection.bbox

        previous_point = bottom_center(
            previous_bbox
        )

        current_point = bottom_center(
            current_bbox
        )

        movement_distance = math.dist(
            previous_point,
            current_point,
        )

        body_height = max(
            1,
            current_bbox[3] - current_bbox[1],
        )

        movement_threshold = max(
            self.minimum_movement_pixels,
            (
                body_height
                * self.movement_threshold_ratio
            ),
        )

        track.previous_bbox = previous_bbox
        track.bbox = current_bbox
        track.confidence = detection.confidence

        track.movement_distance = (
            movement_distance
        )

        track.is_moving = (
            movement_distance
            >= movement_threshold
        )

        track.last_seen_at = observed_at
        track.last_seen_monotonic = (
            observed_monotonic
        )

        track.trajectory.append(
            current_point
        )

    def _create_track(
        self,
        detection: PersonDetection,
        observed_at: datetime,
        observed_monotonic: float,
    ) -> PersonTrack:
        initial_point = bottom_center(
            detection.bbox
        )

        track = PersonTrack(
            track_id=self._next_track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
            first_seen_monotonic=(
                observed_monotonic
            ),
            last_seen_monotonic=(
                observed_monotonic
            ),
            previous_bbox=None,
            movement_distance=0.0,
            is_moving=False,
            trajectory=deque(
                [initial_point],
                maxlen=self.trajectory_size,
            ),
        )

        self._next_track_id += 1

        return track

    def expire(
        self,
        now: float,
    ) -> list[PersonTrack]:
        """
        Remove tracks that have not been detected recently.
        """

        expired_ids = [
            track_id
            for track_id, track
            in self._tracks.items()
            if (
                now
                - track.last_seen_monotonic
                >= self.timeout_seconds
            )
        ]

        expired_tracks: list[PersonTrack] = []

        for track_id in expired_ids:
            expired_track = self._tracks.pop(
                track_id
            )

            expired_tracks.append(
                expired_track
            )

        return expired_tracks

    def active_tracks(
        self,
    ) -> list[PersonTrack]:
        return list(
            self._tracks.values()
        )

    def moving_tracks(
        self,
    ) -> list[PersonTrack]:
        return [
            track
            for track in self._tracks.values()
            if track.is_moving
        ]

    def get_track(
        self,
        track_id: int,
    ) -> PersonTrack | None:
        return self._tracks.get(
            track_id
        )

    def clear(self) -> None:
        """
        Clear tracking state after camera reconnection.
        """

        self._tracks.clear()
        self._next_track_id = 1

    def _match(
        self,
        detections: list[PersonDetection],
    ) -> dict[int, int]:
        candidates: list[
            tuple[float, int, int]
        ] = []

        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(
                detections
            ):
                score = self._match_score(
                    track.bbox,
                    detection.bbox,
                )

                if score is None:
                    continue

                candidates.append(
                    (
                        score,
                        track_id,
                        detection_index,
                    )
                )

        candidates.sort(
            key=lambda candidate: candidate[0],
            reverse=True,
        )

        matches: dict[int, int] = {}
        used_detections: set[int] = set()

        for (
            _,
            track_id,
            detection_index,
        ) in candidates:
            if track_id in matches:
                continue

            if detection_index in used_detections:
                continue

            matches[track_id] = detection_index

            used_detections.add(
                detection_index
            )

        return matches

    def _match_score(
        self,
        first: BBox,
        second: BBox,
    ) -> float | None:
        overlap = intersection_over_union(
            first,
            second,
        )

        if overlap >= self.iou_threshold:
            return 1.0 + overlap

        first_center = box_center(
            first
        )

        second_center = box_center(
            second
        )

        distance = math.dist(
            first_center,
            second_center,
        )

        largest_side = max(
            first[2] - first[0],
            first[3] - first[1],
            second[2] - second[0],
            second[3] - second[1],
            1,
        )

        normalized_distance = (
            distance / largest_side
        )

        if normalized_distance > 0.65:
            return None

        return max(
            0.0,
            1.0 - normalized_distance,
        )


def box_center(
    bbox: BBox,
) -> Point:
    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def bottom_center(
    bbox: BBox,
) -> Point:
    """
    Approximate the person's floor position.
    """

    x1, _, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        float(y2),
    )


def intersection_over_union(
    first: BBox,
    second: BBox,
) -> float:
    left = max(
        first[0],
        second[0],
    )

    top = max(
        first[1],
        second[1],
    )

    right = min(
        first[2],
        second[2],
    )

    bottom = min(
        first[3],
        second[3],
    )

    intersection_width = max(
        0,
        right - left,
    )

    intersection_height = max(
        0,
        bottom - top,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    if intersection <= 0:
        return 0.0

    first_area = (
        max(
            0,
            first[2] - first[0],
        )
        * max(
            0,
            first[3] - first[1],
        )
    )

    second_area = (
        max(
            0,
            second[2] - second[0],
        )
        * max(
            0,
            second[3] - second[1],
        )
    )

    union = (
        first_area
        + second_area
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union