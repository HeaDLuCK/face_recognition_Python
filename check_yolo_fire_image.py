import argparse
import sys
from pathlib import Path

import cv2

from app.config import get_settings
from app.fire.fire_detection_service import FireDetectionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the YOLO fire model on one image.")
    parser.add_argument("image", help="Path to image")
<<<<<<< HEAD
=======
    parser.add_argument("--model", default=None, help="Override FIRE_YOLO_MODEL_PATH")
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    parser.add_argument("--conf", type=float, default=None, help="Override FIRE_YOLO_CONFIDENCE")
    parser.add_argument("--device", default=None, help="Override FIRE_YOLO_DEVICE, for example cpu or 0")
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
    bbox = detection.get("bbox")
    if not bbox or len(bbox) != 4:
        return

    x1, y1, x2, y2 = [int(value) for value in bbox]
    label = detection.get("className") or "FIRE"
    confidence = detection.get("confidence")
    if confidence is not None:
        label = f"{label} {confidence:.2f}"

    color = (0, 80, 255)
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
<<<<<<< HEAD
=======
    if args.model:
        settings.fire_yolo_model_path = Path(args.model)
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    if args.conf is not None:
        settings.fire_yolo_confidence = args.conf
    if args.device:
        settings.fire_yolo_device = args.device

    frame = read_image(args.image)
    service = FireDetectionService(settings)
    detections = service.detect_frame(frame)

<<<<<<< HEAD
    print(f"Model: {service.model_path}")
=======
    print(f"Model: {settings.fire_yolo_model_path}")
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
    print(f"Confidence: {settings.fire_yolo_confidence}")
    print(f"Device: {settings.fire_yolo_device}")
    print(f"Detections: {len(detections)}")

    for index, detection in enumerate(detections, start=1):
        print(
            f"#{index} bbox={detection.get('bbox')} "
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
