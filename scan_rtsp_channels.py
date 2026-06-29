import os
import re
import time
from urllib.parse import urlsplit, urlunsplit

import cv2


def load_env(path=".env"):
    if not os.path.exists(path):
        print("im here")
        print(os.path.exists(".env"))
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_url(base_url, channel):
    parts = urlsplit(base_url)
    path = re.sub(r"/Streaming/Channels/\d+$", f"/Streaming/Channels/{channel}", parts.path)
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def test_channel(url, timeout_seconds=5):
    started_at = time.time()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    while time.time() - started_at < timeout_seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            cap.release()
            return True, width, height
        time.sleep(0.1)

    cap.release()
    return False, None, None


def main():
    load_env()
    base_url = os.getenv("RTSP_URL")
    if not base_url:
        raise SystemExit("RTSP_URL is missing in .env")

    max_camera = int(os.getenv("SCAN_MAX_CAMERA", "16"))
    timeout_seconds = int(os.getenv("SCAN_TIMEOUT_SECONDS", "5"))
    channels = []
    for camera_number in range(1, max_camera + 1):
        channels.append(f"{camera_number}01")
        channels.append(f"{camera_number}02")

    print(f"Testing {len(channels)} channels using RTSP_URL from .env")
    print("This can take a few minutes if many channels are offline.\n")

    working = []
    for channel in channels:
        url = build_url(base_url, channel)
        ok, width, height = test_channel(url, timeout_seconds)
        if ok:
            working.append(channel)
            print(f"OK   channel {channel}  {width}x{height}")
        else:
            print(f"FAIL channel {channel}")

    print("\nWorking channels:")
    if working:
        print(",".join(working))
        print(f"\nPut this in .env:\nRTSP_CHANNELS={','.join(working)}")
    else:
        print("None found. Check IP, username/password, RTSP enabled, or camera network access.")


if __name__ == "__main__":
    main()
