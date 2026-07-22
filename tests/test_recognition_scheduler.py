import asyncio
import unittest

from app.face.recognition_scheduler import FaceRecognitionScheduler


class FaceRecognitionSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_robin_uses_latest_pending_frame_per_camera(self) -> None:
        scheduler = FaceRecognitionScheduler()
        seen = []
        first_round_done = asyncio.Event()
        second_round_done = asyncio.Event()

        async def handle(camera_id: str, frame: dict) -> None:
            seen.append((camera_id, frame["value"]))
            if len(seen) == 3:
                first_round_done.set()
            if len(seen) == 5:
                second_round_done.set()

        try:
            for camera_id in ("CAM1", "CAM2", "CAM3"):
                scheduler.register(
                    camera_id,
                    lambda frame, camera_id=camera_id: handle(camera_id, frame),
                )

            scheduler.submit("CAM1", {"value": "old"})
            scheduler.submit("CAM1", {"value": "new"})
            scheduler.submit("CAM2", {"value": "two"})
            scheduler.submit("CAM3", {"value": "three"})
            await asyncio.wait_for(first_round_done.wait(), timeout=1.0)

            scheduler.submit("CAM1", {"value": "next-one"})
            scheduler.submit("CAM2", {"value": "next-two"})
            await asyncio.wait_for(second_round_done.wait(), timeout=1.0)

            self.assertEqual(
                seen,
                [
                    ("CAM1", "new"),
                    ("CAM2", "two"),
                    ("CAM3", "three"),
                    ("CAM1", "next-one"),
                    ("CAM2", "next-two"),
                ],
            )
        finally:
            await scheduler.close()


if __name__ == "__main__":
    unittest.main()
