from app.schemas.erp_schema import AiCapability


IMPLEMENTED_CAPABILITIES = {AiCapability.FACE_RECOGNITION}

FUTURE_CAPABILITIES = {
    AiCapability.PLATE_RECOGNITION,
    AiCapability.OBJECT_COUNTING,
    AiCapability.PERSON_COUNTING,
    AiCapability.SMOKE_DETECTION,
    AiCapability.FIRE_DETECTION,
    AiCapability.SUSPICIOUS_BEHAVIOR,
    AiCapability.POSTURE_DETECTION,
}


def is_enabled(camera_capabilities: list[AiCapability], capability: AiCapability) -> bool:
    return capability in camera_capabilities and capability in IMPLEMENTED_CAPABILITIES

