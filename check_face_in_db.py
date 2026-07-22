import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import get_settings
from app.face.insightface_engine import DetectedFace, InsightFaceEngine
from app.services.url_utils import redact_url_credentials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether face(s) in one image exist in Mongo cached employee embeddings."
    )
    parser.add_argument("image", help="Path to the image to check")
    parser.add_argument(
        "--etsAuth",
        "--tenantId",
        dest="ets_auth",
        default=None,
        help="Tenant/company code. If omitted, auto-used only when Mongo has one tenant.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for MATCH. Defaults to DEFAULT_RECOGNITION_THRESHOLD from .env.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many closest employees to print for each detected face.",
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


def normalized_embedding(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    embedding = np.array(value, dtype=np.float32)
    if embedding.ndim != 1 or embedding.size == 0:
        return None
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


async def resolve_ets_auth(db, requested_ets_auth: str | None) -> str:
    if requested_ets_auth:
        return requested_ets_auth

    values = set(await db.cached_embeddings.distinct("etsAuth"))
    values.update(await db.cached_embeddings.distinct("tenantId"))
    values = {str(value) for value in values if value}

    if len(values) == 1:
        return next(iter(values))
    if not values:
        raise RuntimeError("No employee face embeddings found in Mongo cached_embeddings.")

    options = ", ".join(sorted(values))
    raise RuntimeError(f"More than one etsAuth found. Pass one with --etsAuth. Available: {options}")


async def load_employee_embeddings(db, ets_auth: str) -> list[dict]:
    cursor = db.cached_embeddings.find(
        {"$or": [{"etsAuth": ets_auth}, {"tenantId": ets_auth}]},
        {
            "_id": 0,
            "embedding": 1,
            "employeeId": 1,
            "employeeName": 1,
            "sourceId": 1,
            "embeddingId": 1,
        },
    )
    docs = await cursor.to_list(length=None)
    usable = []
    for doc in docs:
        embedding = normalized_embedding(doc.get("embedding"))
        if embedding is None:
            continue
        usable.append({**doc, "_embedding_array": embedding})
    return usable


def rank_employee_matches(face: DetectedFace, embeddings: list[dict], top: int) -> list[dict]:
    face_embedding = normalized_embedding(face.embedding)
    if face_embedding is None:
        return []

    best_by_employee: dict[str, dict] = {}
    for item in embeddings:
        employee_id = item.get("employeeId") or "UNKNOWN_EMPLOYEE"
        score = float(np.dot(face_embedding, item["_embedding_array"]))
        current = best_by_employee.get(employee_id)
        if current is None or score > current["score"]:
            best_by_employee[employee_id] = {
                "employeeId": employee_id,
                "employeeName": item.get("employeeName"),
                "score": score,
                "sourceId": item.get("sourceId") or item.get("embeddingId"),
            }

    return sorted(best_by_employee.values(), key=lambda item: item["score"], reverse=True)[:top]


async def main_async() -> int:
    args = parse_args()
    settings = get_settings()
    if args.model:
        settings.insightface_model_name = args.model
    if args.min_face_score is not None:
        settings.face_detection_min_score = args.min_face_score

    threshold = args.threshold if args.threshold is not None else settings.default_recognition_threshold

    client = AsyncIOMotorClient(settings.mongo_url)
    try:
        await client.admin.command("ping")
        db = client[settings.mongo_db_name]
        ets_auth = await resolve_ets_auth(db, args.ets_auth)
        stored_embeddings = await load_employee_embeddings(db, ets_auth)
    finally:
        client.close()

    if not stored_embeddings:
        raise RuntimeError(f"No usable employee embeddings found for etsAuth={ets_auth}. Sync employees first.")

    engine = InsightFaceEngine(settings)
    frame = read_image(args.image)
    faces = sorted(engine.detect_faces(frame), key=face_area, reverse=True)
    if not faces:
        raise RuntimeError(f"No face found in image: {args.image}")

    print(f"Mongo: {redact_url_credentials(settings.mongo_url)}")
    print(f"Database: {settings.mongo_db_name}")
    print(f"etsAuth: {ets_auth}")
    print(f"Model: {settings.insightface_model_name}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Stored embeddings: {len(stored_embeddings)}")
    print(f"Image faces: {len(faces)}")

    for index, face in enumerate(faces, start=1):
        ranked = rank_employee_matches(face, stored_embeddings, max(args.top, 1))
        best = ranked[0] if ranked else None
        matched = bool(best and best["score"] >= threshold)

        print("")
        print(f"Face #{index}")
        print(f"bbox: {face.bbox}, detectionScore={face.detectionScore:.4f}")
        if best:
            print(f"Result: {'MATCH' if matched else 'NO MATCH'}")
            print(
                "Best: "
                f"{best['employeeId']} | {best.get('employeeName') or ''} | "
                f"score={best['score']:.4f} | source={best.get('sourceId') or ''}"
            )
        else:
            print("Result: NO MATCH")

        if len(ranked) > 1:
            print("Top candidates:")
            for item in ranked:
                print(
                    f"- {item['employeeId']} | {item.get('employeeName') or ''} | "
                    f"score={item['score']:.4f} | source={item.get('sourceId') or ''}"
                )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
