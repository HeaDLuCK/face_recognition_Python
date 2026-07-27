from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class PersonDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class PersonTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime
    first_seen_monotonic: float
    last_seen_monotonic: float


@dataclass(frozen=True)
class PersonDetectionBatch:
    frame: np.ndarray
    detections: list[PersonDetection]
    observed_at: datetime
    observed_monotonic: float


@dataclass(frozen=True)
class TrackedRecognitionJob:
    frame: np.ndarray
    track_id: int
    person_bbox: tuple[int, int, int, int]
    observed_at: datetime
    quality: float
