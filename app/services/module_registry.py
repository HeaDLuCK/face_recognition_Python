from app.schemas.erp_schema import AiCapability


IMPLEMENTED_CAPABILITIES = {
    AiCapability.FACE_RECOGNITION,
    AiCapability.PLATE_RECOGNITION,
    AiCapability.FIRE_DETECTION,
    AiCapability.PERSON_COUNTING,
}

FUTURE_CAPABILITIES = {
    AiCapability.OBJECT_COUNTING,
    AiCapability.SMOKE_DETECTION,
    AiCapability.SUSPICIOUS_BEHAVIOR,
    AiCapability.POSTURE_DETECTION,
}


def is_enabled(camera_capabilities: list[AiCapability], capability: AiCapability) -> bool:
    return capability in camera_capabilities and capability in IMPLEMENTED_CAPABILITIES

