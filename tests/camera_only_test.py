import argparse
import logging
import time
from getpass import getpass

import cv2

from app.cameras.rtsp_reader import RtspReader
from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open and continuously read one camera without running any AI models."
    )
    parser.add_argument("rtsp_url", nargs="?", help="Camera RTSP URL")
    parser.add_argument("--show", action="store_true", help="Show the camera window")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    cv2.setNumThreads(settings.opencv_num_threads)

    rtsp_url = args.rtsp_url or getpass("RTSP URL: ")
    reader = RtspReader(rtsp_url, settings)
    frame_count = 0
    report_started_at = time.monotonic()

    try:
        reader.open()
        print("Camera is running. Press Ctrl+C to stop.")

        while True:
            frame = reader.read()
            frame_count += 1

            now = time.monotonic()
            elapsed = now - report_started_at
            if elapsed >= 5:
                print(f"Camera decode rate: {frame_count / elapsed:.1f} FPS")
                frame_count = 0
                report_started_at = now

            if args.show:
                cv2.imshow("Camera only test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
