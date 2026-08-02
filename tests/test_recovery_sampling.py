import unittest

from app.recovery.sampling import history_sample_interval_seconds


class RecoverySamplingTests(unittest.TestCase):
    def test_short_windows_use_fixed_target_interval(self) -> None:
        self.assertEqual(history_sample_interval_seconds(30, 0.5, 1000), 0.5)
        self.assertEqual(history_sample_interval_seconds(300, 0.5, 1000), 0.5)

    def test_long_windows_expand_interval_to_respect_frame_cap(self) -> None:
        self.assertEqual(history_sample_interval_seconds(600, 0.5, 1000), 0.6)
        self.assertEqual(history_sample_interval_seconds(1500, 0.5, 1000), 1.5)

    def test_invalid_runtime_values_remain_safe(self) -> None:
        self.assertEqual(history_sample_interval_seconds(0, 0.5, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
