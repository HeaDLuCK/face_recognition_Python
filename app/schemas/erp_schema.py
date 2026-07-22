from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


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


class CameraAssignment(ErpBaseModel):
    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    enabled: bool = True
    direction: CameraDirection = "BIDIRECTIONAL"
    capabilities: list[AiCapability] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)


class CameraConfig(ErpBaseModel):
    tenantId: str | None = Field(
        default=None,
        validation_alias=AliasChoices("tenantId", "etsAuth"),
        serialization_alias="etsAuth",
    )
    cameraId: str
    name: str
    rtspUrl: str
    enabled: bool = True
    direction: CameraDirection = "BIDIRECTIONAL"
    capabilities: list[AiCapability] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    assignments: list[CameraAssignment] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_assignments(self):
        if not self.assignments:
            if not self.tenantId:
                raise ValueError("Camera requires etsAuth/tenantId or assignments")
            self.assignments = [
                CameraAssignment(
                    tenantId=self.tenantId,
                    enabled=self.enabled,
                    direction=self.direction,
                    capabilities=self.capabilities,
                    zones=self.zones,
                )
            ]

        assignments_by_tenant = {
            assignment.tenantId: assignment
            for assignment in self.assignments
        }
        self.assignments = list(assignments_by_tenant.values())
        self.enabled = any(assignment.enabled for assignment in self.assignments)
        primary_assignment = next(
            (assignment for assignment in self.assignments if assignment.enabled),
            self.assignments[0],
        )
        self.tenantId = primary_assignment.tenantId

        effective_capabilities = []
        for assignment in self.assignments:
            if not assignment.enabled:
                continue
            for capability in assignment.capabilities:
                if capability not in effective_capabilities:
                    effective_capabilities.append(capability)
        self.capabilities = effective_capabilities
        self.direction = primary_assignment.direction
        self.zones = primary_assignment.zones
        return self

    @property
    def tenantIds(self) -> list[str]:
        return [assignment.tenantId for assignment in self.assignments]

    @property
    def activeAssignments(self) -> list[CameraAssignment]:
        return [assignment for assignment in self.assignments if assignment.enabled]

    def assignments_for(self, capability: AiCapability) -> list[CameraAssignment]:
        return [
            assignment
            for assignment in self.activeAssignments
            if capability in assignment.capabilities
        ]


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
    imageRetentionDays: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices("imageRetentionDays", "snapshotRetentionDays", "purgeImagesAfterDays"),
    )


class ErpEventPayload(ErpBaseModel):
    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    cameraId: str
    eventType: str
    employeeId: str | None = None
    confidence: float | None = None
    snapshotPath: str | None = None
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)
