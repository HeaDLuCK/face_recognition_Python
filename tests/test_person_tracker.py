import unittest
from datetime import datetime, timedelta, timezone

from app.tracking.models import PersonDetection
from app.tracking.person_tracker import PersonTracker


class PersonTrackerTests(unittest.TestCase):
    def test_moving_person_keeps_track_id_and_second_person_is_separate(self) -> None:
        tracker = PersonTracker(iou_threshold=0.2, timeout_seconds=2.0)
        observed_at = datetime.now(timezone.utc).replace(tzinfo=None)

        first, ended = tracker.update(
            [PersonDetection((10, 10, 110, 210), 0.9)],
            observed_at,
            10.0,
        )
        second, ended_again = tracker.update(
            [
                PersonDetection((20, 12, 120, 212), 0.92),
                PersonDetection((250, 20, 350, 220), 0.88),
            ],
            observed_at + timedelta(milliseconds=200),
            10.2,
        )

        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertNotEqual(second[0].track_id, second[1].track_id)
        self.assertEqual(ended, [])
        self.assertEqual(ended_again, [])

    def test_track_expires_after_timeout(self) -> None:
        tracker = PersonTracker(iou_threshold=0.2, timeout_seconds=2.0)
        observed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        active, _ = tracker.update(
            [PersonDetection((10, 10, 110, 210), 0.9)],
            observed_at,
            20.0,
        )

        expired = tracker.expire(22.1)

        self.assertEqual([track.track_id for track in expired], [active[0].track_id])
        self.assertEqual(tracker.active_tracks(), [])


if __name__ == "__main__":
    unittest.main()
