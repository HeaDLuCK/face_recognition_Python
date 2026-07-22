import unittest

from app.services.url_utils import redact_url_credentials


class UrlRedactionTests(unittest.TestCase):
    def test_rtsp_credentials_are_hidden(self) -> None:
        value = "rtsp://admin:p%40ss@192.168.1.5:554/Streaming/Channels/101"
        redacted = redact_url_credentials(value)

        self.assertEqual(
            redacted,
            "rtsp://***:***@192.168.1.5:554/Streaming/Channels/101",
        )
        self.assertNotIn("admin", redacted)
        self.assertNotIn("p%40ss", redacted)

    def test_url_without_credentials_is_unchanged(self) -> None:
        value = "http://localhost:8000/health"
        self.assertEqual(redact_url_credentials(value), value)


if __name__ == "__main__":
    unittest.main()
