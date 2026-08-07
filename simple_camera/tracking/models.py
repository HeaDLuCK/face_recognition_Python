from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class PersonDetection:

    bbox: BBox
    confidence: float


@dataclass(slots=True)
class PersonTrack:

    track_id: int

    bbox: BBox
    confidence: float

    first_seen_at: datetime
    last_seen_at: datetime

    first_seen_monotonic: float
    last_seen_monotonic: float

    previous_bbox: BBox | None = None

    movement_distance: float = 0.0
    is_moving: bool = False

    trajectory: deque[Point] = field(
        default_factory=lambda: deque(maxlen=30)
    )