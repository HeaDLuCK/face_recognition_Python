import unittest
from datetime import datetime, timezone

from app.tracking.models import PersonTrack
from app.tracking.person_line_counter import CountingLine, PersonLineCounter


def track(track_id: int, foot_y: int) -> PersonTrack:
    return PersonTrack(
        track_id=track_id,
        bbox=(40, foot_y - 40, 60, foot_y),
        confidence=0.9,
        first_seen_at=datetime.now(timezone.utc),
        last_seen_at=datetime.now(timezone.utc),
        first_seen_monotonic=1.0,
        last_seen_monotonic=1.0,
    )


class PersonLineCounterTests(unittest.TestCase):
    def test_crossing_into_positive_side_is_in(self) -> None:
        counter = PersonLineCounter(CountingLine(hysteresis=0.02))

        self.assertIsNone(counter.update(track(1, 40), 100, 100))
        self.assertEqual(counter.update(track(1, 70), 100, 100), "IN")

    def test_reverse_crossing_is_out(self) -> None:
        counter = PersonLineCounter(CountingLine(hysteresis=0.02))

        self.assertIsNone(counter.update(track(1, 70), 100, 100))
        self.assertEqual(counter.update(track(1, 40), 100, 100), "OUT")

    def test_hysteresis_ignores_jitter_on_line(self) -> None:
        counter = PersonLineCounter(CountingLine(hysteresis=0.05))

        self.assertIsNone(counter.update(track(1, 40), 100, 100))
        self.assertIsNone(counter.update(track(1, 48), 100, 100))
        self.assertIsNone(counter.update(track(1, 52), 100, 100))
        self.assertEqual(counter.update(track(1, 60), 100, 100), "IN")

    def test_negative_in_side_inverts_direction(self) -> None:
        counter = PersonLineCounter(
            CountingLine(in_side="NEGATIVE", hysteresis=0.02)
        )

        self.assertIsNone(counter.update(track(1, 40), 100, 100))
        self.assertEqual(counter.update(track(1, 70), 100, 100), "OUT")


if __name__ == "__main__":
    unittest.main()
