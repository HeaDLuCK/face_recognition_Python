import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.recovery.attendance_recovery_service import AttendanceRecoveryService


class FakeHistoryCapture:
    def __init__(self, frames: list[object]) -> None:
        self.frames = iter(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _property: int) -> float:
        return 25.0

    def read(self) -> tuple[bool, object | None]:
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


class RecoveryFrameProcessingTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_decoded_history_frame_is_recognized(self) -> None:
        frames = [object(), object(), object(), object()]
        capture = FakeHistoryCapture(frames)
        recognize = AsyncMock(return_value=[])
        service = AttendanceRecoveryService.__new__(AttendanceRecoveryService)
        service.runtime_state = SimpleNamespace(
            get_rules=lambda _tenant_id: SimpleNamespace(recognitionThreshold=0.55)
        )
        service.recognition_service = SimpleNamespace(
            recognize_frame_for_tenants=recognize
        )
        started_at = datetime.utcnow()
        job = {
            "recoveryJobId": "TEST-RECOVERY",
            "windowStart": started_at,
            "windowEnd": started_at + timedelta(seconds=30),
        }
        assignments = [SimpleNamespace(tenantId="TENANT-1", zones=[])]

        with patch(
            "app.recovery.attendance_recovery_service.cv2.VideoCapture",
            return_value=capture,
        ):
            await service._scan_history_stream("rtsp://history", job, assignments)

        processed_frames = [call.kwargs["frame"] for call in recognize.await_args_list]
        self.assertEqual(processed_frames, frames)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
