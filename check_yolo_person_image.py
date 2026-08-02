import argparse
import sys
from pathlib import Path

import cv2

from app.config import get_settings
from app.tracking.person_detection_service import PersonDetectionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the attendance person detector on one image."
    )
    parser.add_argument("image", help="Path to image")
<<<<<<< HEAD
=======
    parser.add_argument("--model", default=None, help="Override PERSON_YOLO_MODEL_PATH")
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override PERSON_YOLO_CONFIDENCE",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override PERSON_YOLO_DEVICE, for example cpu or 0",
    )
    parser.add_argument(
        "--save-debug",
        default=None,
        help="Optional path to save an annotated debug image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Unable to read image: {image_path}")

    settings = get_settings()
<<<<<<< HEAD
=======
    if args.model:
        settings.person_yolo_model_path = Path(args.model)
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    if args.conf is not None:
        settings.person_yolo_confidence = args.conf
    if args.device:
        settings.person_yolo_device = args.device

    service = PersonDetectionService(settings)
    detections = service.detect_frame(frame)
<<<<<<< HEAD
    print(f"Model: {service.model_path}")
=======
    print(f"Model: {settings.person_yolo_model_path}")
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    print(f"Detections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(
            f"#{index} bbox={list(detection.bbox)} "
            f"confidence={detection.confidence:.4f}"
        )
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            frame,
            f"PERSON {detection.confidence:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

    if args.save_debug:
        output_path = Path(args.save_debug)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        print(f"Saved debug image: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
