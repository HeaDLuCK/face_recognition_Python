import math
from datetime import datetime

import numpy as np

from app.tracking.models import PersonDetection, PersonTrack


class MotionActivityGate:
    def __init__(
        self,
        pixel_threshold: int,
        area_ratio: float,
        hold_seconds: float,
    ):
        self.pixel_threshold = pixel_threshold
        self.area_ratio = area_ratio
        self.hold_seconds = hold_seconds
        self._previous_gray: np.ndarray | None = None
        self._active_until = 0.0

    def update(self, frame: np.ndarray, now: float) -> bool:
        import cv2

        height, width = frame.shape[:2]
        scale = min(1.0, 320.0 / max(width, 1))
        sample = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if self._previous_gray is None:
            self._previous_gray = gray
            self._active_until = now + self.hold_seconds
            return True

        diff = cv2.absdiff(self._previous_gray, gray)
        self._previous_gray = gray
        _, changed = cv2.threshold(
            diff,
            self.pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        ratio = cv2.countNonZero(changed) / max(changed.size, 1)
        if ratio >= self.area_ratio:
            self._active_until = now + self.hold_seconds
        return now <= self._active_until

    def reset(self) -> None:
        self._previous_gray = None
        self._active_until = 0.0


class PersonTracker:
    def __init__(self, iou_threshold: float, timeout_seconds: float):
        self.iou_threshold = iou_threshold
        self.timeout_seconds = timeout_seconds
        self._tracks: dict[int, PersonTrack] = {}
        self._next_track_id = 1

    def update(
        self,
        detections: list[PersonDetection],
        observed_at: datetime,
        observed_monotonic: float,
    ) -> tuple[list[PersonTrack], list[PersonTrack]]:
        matches = self._match(detections)
        matched_detection_indexes = set()
        updated_tracks: list[PersonTrack] = []

        for track_id, detection_index in matches.items():
            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.last_seen_at = observed_at
            track.last_seen_monotonic = observed_monotonic
            matched_detection_indexes.add(detection_index)
            updated_tracks.append(track)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indexes:
                continue
            track = PersonTrack(
                track_id=self._next_track_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                first_seen_at=observed_at,
                last_seen_at=observed_at,
                first_seen_monotonic=observed_monotonic,
                last_seen_monotonic=observed_monotonic,
            )
            self._next_track_id += 1
            self._tracks[track.track_id] = track
            updated_tracks.append(track)

        return updated_tracks, self.expire(observed_monotonic)

    def expire(self, now: float) -> list[PersonTrack]:
        expired_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen_monotonic >= self.timeout_seconds
        ]
        return [self._tracks.pop(track_id) for track_id in expired_ids]

    def active_tracks(self) -> list[PersonTrack]:
        return list(self._tracks.values())

    def clear(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def _match(self, detections: list[PersonDetection]) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                score = self._match_score(track.bbox, detection.bbox)
                if score is not None:
                    candidates.append((score, track_id, detection_index))

        candidates.sort(reverse=True)
        matches: dict[int, int] = {}
        used_detections = set()
        for _, track_id, detection_index in candidates:
            if track_id in matches or detection_index in used_detections:
                continue
            matches[track_id] = detection_index
            used_detections.add(detection_index)
        return matches

    def _match_score(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float | None:
        overlap = _iou(first, second)
        if overlap >= self.iou_threshold:
            return 1.0 + overlap

        first_center = _center(first)
        second_center = _center(second)
        distance = math.dist(first_center, second_center)
        largest_side = max(
            first[2] - first[0],
            first[3] - first[1],
            second[2] - second[0],
            second[3] - second[1],
            1,
        )
        normalized_distance = distance / largest_side
        if normalized_distance > 0.65:
            return None
        return max(0.0, 1.0 - normalized_distance)


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0

    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
