import argparse
import asyncio
import sys
from pathlib import Path

import cv2
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import get_settings
from app.face.insightface_engine import InsightFaceEngine
from app.services.url_utils import redact_url_credentials
from check_face_in_db import face_area, load_employee_embeddings, rank_employee_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a video and check whether visible faces exist in Mongo cached employee embeddings."
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--etsAuth",
        "--tenantId",
        dest="ets_auth",
        default=None,
        help="Optional tenant/company code, for example SEA_FOOD. If omitted, checks all tenants.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for MATCH. Defaults to DEFAULT_RECOGNITION_THRESHOLD from .env.",
    )
    parser.add_argument(
        "--sample-every",
        type=float,
        default=2.0,
        help="Seconds between sampled frames. Default: 2.0",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum sampled frames to check. Default: 120",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
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
    parser.add_argument(
        "--save-debug-dir",
        default=None,
        help="Optional folder to save annotated frames that contain faces.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a window and show the video while checking faces.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=960,
        help="Resize the preview window to this width. Default: 960",
    )
    return parser.parse_args()


def draw_face(frame, face, label: str, matched: bool, blink_on: bool = True) -> None:
    if matched and not blink_on:
        return

    x1, y1, x2, y2 = face.bbox
    color = (0, 255, 0) if matched else (0, 220, 255)
    thickness = 4 if matched else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if matched:
        corner = 18
        cv2.line(frame, (x1, y1), (x1 + corner, y1), (255, 255, 255), 2)
        cv2.line(frame, (x1, y1), (x1, y1 + corner), (255, 255, 255), 2)
        cv2.line(frame, (x2, y1), (x2 - corner, y1), (255, 255, 255), 2)
        cv2.line(frame, (x2, y1), (x2, y1 + corner), (255, 255, 255), 2)
        cv2.line(frame, (x1, y2), (x1 + corner, y2), (255, 255, 255), 2)
        cv2.line(frame, (x1, y2), (x1, y2 - corner), (255, 255, 255), 2)
        cv2.line(frame, (x2, y2), (x2 - corner, y2), (255, 255, 255), 2)
        cv2.line(frame, (x2, y2), (x2, y2 - corner), (255, 255, 255), 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def resize_for_display(frame, target_width: int):
    if target_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / width
    return cv2.resize(
        frame,
        (target_width, max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


async def main_async() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

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
        embeddings_by_tenant = await load_embeddings_by_tenant(db, args.ets_auth)
    finally:
        client.close()

    if not embeddings_by_tenant:
        raise RuntimeError("No usable employee embeddings found in Mongo cached_embeddings. Sync employees first.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_step = max(1, int(fps * args.sample_every)) if fps > 0 else 30
    engine = InsightFaceEngine(settings)
    debug_dir = Path(args.save_debug_dir) if args.save_debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mongo: {redact_url_credentials(settings.mongo_url)}")
    print(f"Database: {settings.mongo_db_name}")
    print(f"etsAuth: {args.ets_auth or 'ALL'}")
    print(f"Video: {video_path}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Tenants: {', '.join(sorted(embeddings_by_tenant))}")
    print(f"Stored embeddings: {sum(len(items) for items in embeddings_by_tenant.values())}")
    print(f"Sample every: {args.sample_every}s")

    frame_index = 0
    sampled = 0
    faces_seen = 0
    best_by_employee: dict[tuple[str, str], dict] = {}

    try:
        while sampled < args.max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            current_index = frame_index
            frame_index += 1
            if current_index % frame_step != 0:
                continue

            sampled += 1
            timestamp_seconds = current_index / fps if fps > 0 else sampled * args.sample_every
            faces = sorted(engine.detect_faces(frame), key=face_area, reverse=True)
            annotated = frame.copy() if debug_dir or args.show else None
            if not faces:
                if args.show:
                    cv2.imshow(
                        "Video face database check",
                        resize_for_display(frame, args.display_width),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                continue

            faces_seen += len(faces)
            print("")
            print(f"Frame {current_index} at {timestamp_seconds:.1f}s: {len(faces)} face(s)")

            for face_number, face in enumerate(faces, start=1):
                ranked = rank_matches_for_tenants(face, embeddings_by_tenant, max(args.top, 1))
                best = ranked[0] if ranked else None
                matched = bool(best and best["score"] >= threshold)

                if best:
                    label = (
                        f"{'MATCH' if matched else 'NO MATCH'} "
                        f"{best['etsAuth']} {best['employeeId']} {best.get('employeeName') or ''} "
                        f"{best['score']:.3f}"
                    )
                    print(f"  Face #{face_number}: {label}")
                    if matched:
                        key = (best["etsAuth"], best["employeeId"])
                        current = best_by_employee.get(key)
                        if current is None or best["score"] > current["score"]:
                            best_by_employee[key] = {
                                **best,
                                "timestampSeconds": timestamp_seconds,
                                "frameIndex": current_index,
                            }
                else:
                    label = "NO MATCH"
                    print(f"  Face #{face_number}: NO MATCH")

                if annotated is not None:
                    draw_face(
                        annotated,
                        face,
                        label[:80],
                        matched,
                        blink_on=sampled % 2 == 1,
                    )

                if len(ranked) > 1:
                    for item in ranked[1:]:
                        print(
                            f"    candidate: {item['etsAuth']} | {item['employeeId']} | "
                            f"{item.get('employeeName') or ''} | score={item['score']:.4f}"
                        )

            if debug_dir is not None and annotated is not None:
                output = debug_dir / f"frame_{current_index:06d}.jpg"
                cv2.imwrite(str(output), annotated)

            if args.show:
                display = annotated if annotated is not None else frame
                cv2.imshow(
                    "Video face database check",
                    resize_for_display(display, args.display_width),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()

    print("")
    print("Summary")
    print(f"Sampled frames: {sampled}")
    print(f"Detected faces: {faces_seen}")
    print(f"Matched employees: {len(best_by_employee)}")
    for employee in sorted(best_by_employee.values(), key=lambda item: (item["etsAuth"], item["employeeId"])):
        print(
            f"- {employee['etsAuth']} | {employee['employeeId']} | {employee.get('employeeName') or ''} | "
            f"bestScore={employee['score']:.4f} | "
            f"at={employee['timestampSeconds']:.1f}s | frame={employee['frameIndex']}"
        )

    return 0


async def load_embeddings_by_tenant(db, requested_ets_auth: str | None) -> dict[str, list[dict]]:
    if requested_ets_auth:
        embeddings = await load_employee_embeddings(db, requested_ets_auth)
        return {requested_ets_auth: embeddings} if embeddings else {}

    values = set(await db.cached_embeddings.distinct("etsAuth"))
    values.update(await db.cached_embeddings.distinct("tenantId"))
    tenant_ids = sorted(str(value) for value in values if value)
    result = {}
    for tenant_id in tenant_ids:
        embeddings = await load_employee_embeddings(db, tenant_id)
        if embeddings:
            result[tenant_id] = embeddings
    return result


def rank_matches_for_tenants(face, embeddings_by_tenant: dict[str, list[dict]], top: int) -> list[dict]:
    ranked = []
    for tenant_id, embeddings in embeddings_by_tenant.items():
        for item in rank_employee_matches(face, embeddings, top):
            ranked.append({"etsAuth": tenant_id, **item})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top]


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
