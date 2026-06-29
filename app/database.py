import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient | None = None
database: AsyncIOMotorDatabase | None = None

TENANT_INDEXED_COLLECTIONS = (
    "cached_embeddings",
    "attendance_detections",
    "camera_events",
    "alert_events",
    "snapshot_metadata",
    "unknown_face_crops",
    "camera_configs",
    "attendance_rules",
    "service_logs",
)


async def connect_to_mongo() -> AsyncIOMotorDatabase:
    global client, database

    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongo_url)
    database = client[settings.mongo_db_name]
    await client.admin.command("ping")
    await migrate_tenant_field(database)
    await ensure_indexes(database)
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
    for collection_name in TENANT_INDEXED_COLLECTIONS:
        if collection_name == "attendance_rules":
            continue
        await db[collection_name].create_index([("etsAuth", 1)])

    await db.cached_embeddings.create_index(
        [("etsAuth", 1), ("employeeId", 1), ("sourceId", 1)],
        unique=True,
    )
    await db.attendance_detections.create_index([("etsAuth", 1), ("employeeId", 1), ("timestamp", -1)])
    await db.camera_events.create_index([("etsAuth", 1), ("cameraId", 1), ("timestamp", -1)])
    await db.alert_events.create_index([("etsAuth", 1), ("cameraId", 1), ("timestamp", -1)])
    await db.snapshot_metadata.create_index([("etsAuth", 1), ("cameraId", 1), ("timestamp", -1)])
    await db.unknown_face_crops.create_index([("etsAuth", 1), ("cameraId", 1), ("createdAt", -1)])
    await db.camera_configs.create_index([("cameraId", 1)], unique=True)
    await db.camera_configs.create_index([("etsAuth", 1), ("enabled", 1)])
    await drop_index_by_name(db.attendance_rules, "etsAuth_1")
    await db.attendance_rules.create_index(
        [("etsAuth", 1)],
        unique=True,
        name="attendance_rules_etsAuth_unique",
    )
    await db.service_logs.create_index([("etsAuth", 1), ("createdAt", -1)])


async def drop_index_by_name(collection, index_name: str) -> None:
    async for index in collection.list_indexes():
        if index.get("name") == index_name:
            await collection.drop_index(index_name)
            return


async def migrate_tenant_field(db: AsyncIOMotorDatabase) -> None:
    for collection_name in TENANT_INDEXED_COLLECTIONS:
        collection = db[collection_name]
        async for index in collection.list_indexes():
            keys = dict(index.get("key", {}))
            if "tenantId" in keys:
                await collection.drop_index(index["name"])

        await collection.update_many(
            {"tenantId": {"$exists": True}, "etsAuth": {"$exists": False}},
            {"$rename": {"tenantId": "etsAuth"}},
        )
        await collection.update_many(
            {"tenantId": {"$exists": True}, "etsAuth": {"$exists": True}},
            {"$unset": {"tenantId": ""}},
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
