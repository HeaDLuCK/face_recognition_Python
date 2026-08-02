import argparse
import sys
from pathlib import Path

import cv2

from app.config import get_settings
from app.plates.plate_recognition_service import PlateRecognitionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the YOLO Moroccan plate model on one image.")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--mode", choices=["plate", "characters"], default=None, help="Override PLATE_YOLO_MODE")
    parser.add_argument("--conf", type=float, default=None, help="Override PLATE_YOLO_CONFIDENCE")
    parser.add_argument("--device", default=None, help="Override PLATE_YOLO_DEVICE, for example cpu or 0")
    parser.add_argument("--save-debug", default=None, help="Optional path to save an annotated debug image")
    return parser.parse_args()


def read_image(path: str):
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return frame


def draw_detection(frame, detection: dict) -> None:
    x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
    label = detection.get("plateText") or detection.get("className") or "PLATE"
    confidence = detection.get("confidence")
    if confidence is not None:
        label = f"{label} {confidence:.2f}"

    color = (255, 180, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.mode:
        settings.plate_yolo_mode = args.mode
    if args.conf is not None:
        settings.plate_yolo_confidence = args.conf
    if args.device:
        settings.plate_yolo_device = args.device

    frame = read_image(args.image)
    service = PlateRecognitionService(settings)
    detections = service.recognize_frame(frame)

    print(f"Model: {service.model_path}")
    print(f"Mode: {settings.plate_yolo_mode}")
    print(f"Confidence: {settings.plate_yolo_confidence}")
    print(f"Device: {settings.plate_yolo_device}")
    print(f"Detections: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        print(
            f"#{index} bbox={detection.get('bbox')} "
            f"text={detection.get('plateText')} "
            f"class={detection.get('className')} "
            f"score={detection.get('confidence')} "
            f"source={detection.get('source')}"
        )
        draw_detection(frame, detection)

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
