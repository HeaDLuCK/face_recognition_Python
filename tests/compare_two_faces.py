import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.face.insightface_engine import DetectedFace, InsightFaceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two images with the same InsightFace model used by the service."
    )
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for SAME. Defaults to DEFAULT_RECOGNITION_THRESHOLD from .env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override INSIGHTFACE_MODEL_NAME for this test, for example buffalo_s or buffalo_l.",
    )
    parser.add_argument(
        "--min-face-score",
        type=float,
        default=None,
        help="Override FACE_DETECTION_MIN_SCORE for this test.",
    )
    return parser.parse_args()


def read_image(path: str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return frame


def face_area(face: DetectedFace) -> int:
    x1, y1, x2, y2 = face.bbox
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def best_pair(faces1: list[DetectedFace], faces2: list[DetectedFace]) -> tuple[DetectedFace, DetectedFace, float]:
    best_face1 = faces1[0]
    best_face2 = faces2[0]
    best_score = -1.0

    for face1 in faces1:
        embedding1 = np.array(face1.embedding, dtype=np.float32)
        for face2 in faces2:
            embedding2 = np.array(face2.embedding, dtype=np.float32)
            score = float(np.dot(embedding1, embedding2))
            if score > best_score:
                best_score = score
                best_face1 = face1
                best_face2 = face2

    return best_face1, best_face2, best_score


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.model:
        settings.insightface_model_name = args.model
    if args.min_face_score is not None:
        settings.face_detection_min_score = args.min_face_score

    threshold = args.threshold if args.threshold is not None else settings.default_recognition_threshold
    engine = InsightFaceEngine(settings)

    frame1 = read_image(args.image1)
    frame2 = read_image(args.image2)

    faces1 = sorted(engine.detect_faces(frame1), key=face_area, reverse=True)
    faces2 = sorted(engine.detect_faces(frame2), key=face_area, reverse=True)

    if not faces1:
        raise RuntimeError(f"No face found in first image: {args.image1}")
    if not faces2:
        raise RuntimeError(f"No face found in second image: {args.image2}")

    face1, face2, score = best_pair(faces1, faces2)
    same_person = score >= threshold

    print(f"Model: {settings.insightface_model_name}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Image 1 faces: {len(faces1)}")
    print(f"Image 2 faces: {len(faces2)}")
    print(f"Best similarity: {score:.4f}")
    print(f"Result: {'SAME' if same_person else 'NOT SAME'}")
    print(f"Image 1 best bbox: {face1.bbox}, detectionScore={face1.detectionScore:.4f}")
    print(f"Image 2 best bbox: {face2.bbox}, detectionScore={face2.detectionScore:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
