import math
from dataclasses import dataclass
from typing import Literal

from app.tracking.models import PersonTrack


CrossingDirection = Literal["IN", "OUT"]


@dataclass(frozen=True)
class CountingLine:
    x1: float = 0.1
    y1: float = 0.5
    x2: float = 0.9
    y2: float = 0.5
    in_side: Literal["POSITIVE", "NEGATIVE"] = "POSITIVE"
    hysteresis: float = 0.015


class PersonLineCounter:
    """Detect stable crossings using the bottom-center (feet) of each track."""

    def __init__(self, line: CountingLine):
        self.line = line
        self._stable_side_by_track: dict[int, int] = {}

    def update(
        self,
        track: PersonTrack,
        frame_width: int,
        frame_height: int,
    ) -> CrossingDirection | None:
        if frame_width <= 0 or frame_height <= 0:
            return None

        foot_x = ((track.bbox[0] + track.bbox[2]) / 2.0) / frame_width
        foot_y = track.bbox[3] / frame_height
        side = self._stable_side(foot_x, foot_y)
        if side == 0:
            return None

        previous = self._stable_side_by_track.get(track.track_id)
        self._stable_side_by_track[track.track_id] = side
        if previous is None or previous == side:
            return None

        entered_positive_side = side > 0
        if self.line.in_side == "NEGATIVE":
            entered_positive_side = not entered_positive_side
        return "IN" if entered_positive_side else "OUT"

    def forget(self, track_id: int) -> None:
        self._stable_side_by_track.pop(track_id, None)

    def clear(self) -> None:
        self._stable_side_by_track.clear()

    def _stable_side(self, x: float, y: float) -> int:
        dx = self.line.x2 - self.line.x1
        dy = self.line.y2 - self.line.y1
        length_squared = dx * dx + dy * dy
        projection = (
            (x - self.line.x1) * dx + (y - self.line.y1) * dy
        ) / max(length_squared, 1e-9)
        if projection < 0.0 or projection > 1.0:
            return 0
        signed_distance = (dx * (y - self.line.y1) - dy * (x - self.line.x1)) / max(
            math.hypot(dx, dy),
            1e-9,
        )
        if signed_distance > self.line.hysteresis:
            return 1
        if signed_distance < -self.line.hysteresis:
            return -1
        return 0
