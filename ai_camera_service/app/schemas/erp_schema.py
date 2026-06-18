from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ErpBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class AiCapability(str, Enum):
    FACE_RECOGNITION = "FACE_RECOGNITION"
    PLATE_RECOGNITION = "PLATE_RECOGNITION"
    OBJECT_COUNTING = "OBJECT_COUNTING"
    PERSON_COUNTING = "PERSON_COUNTING"
    SMOKE_DETECTION = "SMOKE_DETECTION"
    FIRE_DETECTION = "FIRE_DETECTION"
    SUSPICIOUS_BEHAVIOR = "SUSPICIOUS_BEHAVIOR"
    POSTURE_DETECTION = "POSTURE_DETECTION"


CameraDirection = Literal["IN", "OUT", "BIDIRECTIONAL"]


class ZoneConfig(ErpBaseModel):
    zoneId: str
    name: str | None = None
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class CameraConfig(ErpBaseModel):
    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    cameraId: str
    name: str
    rtspUrl: str
    enabled: bool = True
    direction: CameraDirection = "BIDIRECTIONAL"
    capabilities: list[AiCapability] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)


class FaceImageRef(ErpBaseModel):
    sourceId: str | None = None
    url: str | None = None
    imageUrl: str | None = None
    base64: str | None = None
    content: str | None = None


class EmployeeConfig(ErpBaseModel):
    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    employeeId: str
    fullName: str | None = None
    active: bool = True
    faceImages: list[FaceImageRef] = Field(default_factory=list)


class AttendanceRules(ErpBaseModel):
    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    recognitionThreshold: float = Field(default=0.55, ge=0.0, le=1.0)
    duplicateCooldownSeconds: int = Field(default=60, ge=0)
    saveFaceSnapshots: bool = True
    saveUnknownFaces: bool = True
    saveUnknownFaceCrops: bool = True
    sendUnknownFaceAlert: bool = False


class ErpEventPayload(ErpBaseModel):
    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    cameraId: str
    eventType: str
    employeeId: str | None = None
    confidence: float | None = None
    snapshotPath: str | None = None
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)
