from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RuntimeEvent(BaseModel):
    tenantId: str
    cameraId: str
    eventType: str
    employeeId: str | None = None
    confidence: float | None = None
    snapshotPath: str | None = None
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

