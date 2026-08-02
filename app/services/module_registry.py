from app.schemas.erp_schema import AiCapability


IMPLEMENTED_CAPABILITIES = {
    AiCapability.FACE_RECOGNITION,
    AiCapability.PLATE_RECOGNITION,
    AiCapability.FIRE_DETECTION,
<<<<<<< HEAD
=======
    AiCapability.PERSON_COUNTING,
>>>>>>> f1937361af33f961bcbefd1ebc6425add24b3054
}

FUTURE_CAPABILITIES = {
    AiCapability.OBJECT_COUNTING,
    AiCapability.SMOKE_DETECTION,
    AiCapability.SUSPICIOUS_BEHAVIOR,
    AiCapability.POSTURE_DETECTION,
}


def is_enabled(camera_capabilities: list[AiCapability], capability: AiCapability) -> bool:
    return capability in camera_capabilities and capability in IMPLEMENTED_CAPABILITIES

