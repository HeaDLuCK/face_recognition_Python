import unittest
from datetime import datetime, timezone

from app.cameras.hikvision_history import (
    hikvision_main_stream_channel,
    hikvision_playback_url,
    parse_hikvision_rtsp,
)


class HikvisionHistoryTests(unittest.TestCase):
    def test_encoded_credentials_round_trip_into_playback_url(self) -> None:
        info = parse_hikvision_rtsp(
            "rtsp://admin:CalAbc998%3F%21@192.168.100.5:554/Streaming/Channels/301"
        )
        playback_url = hikvision_playback_url(
            host=info["host"],
            rtsp_port=info["rtsp_port"],
            username=info["username"],
            password=info["password"],
            channel=info["channel"],
            start=datetime(2026, 7, 27, 7, 0, tzinfo=timezone.utc),
            end=datetime(2026, 7, 27, 7, 1, tzinfo=timezone.utc),
        )

        self.assertIn("CalAbc998%3F%21@", playback_url)
        self.assertIn("/Streaming/tracks/301?", playback_url)
        self.assertIn("starttime=20260727T070000Z", playback_url)

    def test_substream_channel_maps_to_recorded_main_stream(self) -> None:
        self.assertEqual(hikvision_main_stream_channel("102"), "101")
        self.assertEqual(hikvision_main_stream_channel("702"), "701")
        self.assertEqual(hikvision_main_stream_channel("1003"), "1001")

    def test_main_stream_and_unknown_channel_formats_are_unchanged(self) -> None:
        self.assertEqual(hikvision_main_stream_channel("801"), "801")
        self.assertEqual(hikvision_main_stream_channel("1"), "1")
        self.assertEqual(hikvision_main_stream_channel("custom"), "custom")


if __name__ == "__main__":
    unittest.main()
