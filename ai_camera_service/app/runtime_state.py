from app.config import Settings
from app.schemas.erp_schema import AttendanceRules, CameraConfig


class RuntimeState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cameras: dict[str, CameraConfig] = {}
        self.rules: dict[str, AttendanceRules] = {}
        self.last_sync: dict[str, str] = {}

    def set_cameras(self, cameras: list[CameraConfig]) -> None:
        self.cameras = {camera.cameraId: camera for camera in cameras}

    def get_camera(self, camera_id: str) -> CameraConfig | None:
        return self.cameras.get(camera_id)

    def list_cameras(self) -> list[CameraConfig]:
        return list(self.cameras.values())

    def set_rule(self, rule: AttendanceRules) -> None:
        self.rules[rule.tenantId] = rule

    def get_rules(self, tenant_id: str) -> AttendanceRules:
        return self.rules.get(
            tenant_id,
            AttendanceRules(
                tenantId=tenant_id,
                recognitionThreshold=self.settings.default_recognition_threshold,
                duplicateCooldownSeconds=self.settings.default_duplicate_cooldown_seconds,
            ),
        )

