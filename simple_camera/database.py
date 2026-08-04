import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import get_settings

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient | None = None
database: AsyncIOMotorDatabase | None = None

ETABLISSEMENT_INDEXED_COLLECTIONS = (
    "cached_embeddings",
    "attendance_detections",
    "camera_events",
    "alert_events",
    "snapshot_metadata",
    "unknown_face_crops",
    "camera_configs",
    "attendance_rules",
    "attendance_sync_state",
    "service_logs",
    "attendance_recovery_jobs",
)


async def connect_to_mongo() -> AsyncIOMotorDatabase:
    global client, database

    if client is not None and database is not None:
        return database

    settings = get_settings()
    new_client = AsyncIOMotorClient(settings.mongo_url)
    new_database = new_client[settings.mongo_db_name]
    try:
        await new_client.admin.command("ping")
        await ensure_indexes(new_database)
    except Exception:
        new_client.close()
        raise

    client = new_client
    database = new_database
    logger.info("Connected to MongoDB database '%s'", settings.mongo_db_name)
    return database


async def close_mongo_connection() -> None:
    global client, database

    if client is not None:
        client.close()
        logger.info("Closed MongoDB connection")
    client = None
    database = None


def get_database() -> AsyncIOMotorDatabase:
    if database is None:
        raise RuntimeError("MongoDB is not connected")
    return database


async def ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    for collection_name in ETABLISSEMENT_INDEXED_COLLECTIONS:
        if collection_name == "attendance_rules":
            continue
        await db[collection_name].create_index([("etsAuth", 1)])

    await db.cached_embeddings.create_index(
        [("etsAuth", 1), ("employeeId", 1), ("sourceId", 1)],
        unique=True,
    )
    await db.attendance_detections.create_index([("etsAuth", 1), ("employeeId", 1), ("timestamp", -1)])
    await db.attendance_detections.create_index([("etsAuth", 1), ("eventType", 1), ("timestamp", 1)])
    await db.camera_events.create_index([("etsAuth", 1), ("cameraId", 1), ("timestamp", -1)])
    await db.alert_events.create_index([("etsAuth", 1), ("cameraId", 1), ("timestamp", -1)])
    await db.snapshot_metadata.create_index([("etsAuth", 1), ("cameraId", 1), ("timestamp", -1)])
    await db.unknown_face_crops.create_index([("etsAuth", 1), ("cameraId", 1), ("createdAt", -1)])
    await db.unknown_face_crops.create_index([("etsAuth", 1), ("status", 1), ("createdAt", -1)])
    await db.unknown_face_crops.create_index(
        [("unknownFaceCropId", 1)],
        unique=True,
        partialFilterExpression={"unknownFaceCropId": {"$exists": True}},
    )
    await db.camera_configs.create_index([("cameraId", 1)], unique=True)
    await db.camera_configs.create_index([("etsAuth", 1), ("enabled", 1)])
    await db.camera_configs.create_index([("assignments.etsAuth", 1), ("enabled", 1)])
    await db.attendance_rules.create_index(
        [("etsAuth", 1)],
        unique=True,
        name="attendance_rules_etsAuth_unique",
    )
    await db.attendance_sync_state.create_index(
        [("etsAuth", 1), ("syncKey", 1)],
        unique=True,
    )
    await db.service_logs.create_index([("etsAuth", 1), ("createdAt", -1)])
    await db.attendance_recovery_jobs.create_index(
        [("status", 1), ("nextAttemptAt", 1), ("createdAt", 1)]
    )
    await db.attendance_recovery_jobs.create_index(
        [("etsAuth", 1), ("cameraId", 1), ("windowStart", 1), ("windowEnd", 1)]
    )
    await db.attendance_recovery_jobs.create_index(
        [("recoveryJobId", 1)],
        unique=True,
    )

def serialize_mongo_doc(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if doc is None:
        return None
    serialized = dict(doc)
    if "_id" in serialized:
        serialized["_id"] = str(serialized["_id"])
    return serialized

def serialize_mongo_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_mongo_doc(doc) for doc in docs if doc is not None]
