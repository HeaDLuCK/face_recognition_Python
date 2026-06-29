from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tenantId: str = Field(validation_alias=AliasChoices("tenantId", "etsAuth"), serialization_alias="etsAuth")
    cameraId: str
    eventType: str
    employeeId: str | None = None
    confidence: float | None = None
    snapshotPath: str | None = None
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

