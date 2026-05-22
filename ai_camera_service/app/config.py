from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Camera Service"
    environment: str = "development"
    log_level: str = "INFO"

    erp_base_url: str = ""
    erp_api_token: str = ""

    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "ai_camera_service"

    snapshot_dir: Path = Path("snapshots")

    insightface_model_name: str = "buffalo_l"
    insightface_providers: str = "CPUExecutionProvider"
    insightface_ctx_id: int = -1
    default_recognition_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    default_duplicate_cooldown_seconds: int = Field(default=60, ge=0)

    camera_frame_skip: int = Field(default=5, ge=1)
    recognition_interval_seconds: float = Field(default=1.0, ge=0.1)
    camera_source_mode: Literal["auto", "usb", "rtsp"] = "auto"
    usb_camera_index: int = Field(default=0, ge=0)
    rtsp_url: str = ""
    rtsp_urls: str = ""
    rtsp_channels: str = ""
    dev_tenant_id: str = "DEV_COMPANY"
    dev_camera_id: str = "USB_CAM_01"
    stream_fps: int = Field(default=20, ge=1, le=30)
    stream_jpeg_quality: int = Field(default=80, ge=1, le=100)
    show_dev_fps: bool = True
    show_dev_detections: bool = True
    draw_face_boxes_on_snapshots: bool = True
    draw_face_labels_on_snapshots: bool = True
    cloud_stream_ws_url: str = ""
    cloud_stream_token: str = ""
    cloud_stream_fps: int = Field(default=10, ge=1, le=30)
    cloud_stream_reconnect_seconds: int = Field(default=5, ge=1)
    event_clip_dir: Path = Path("event_clips")
    event_buffer_seconds: int = Field(default=30, ge=1)
    event_clip_fps: int = Field(default=15, ge=1, le=30)
    event_clip_cooldown_seconds: int = Field(default=30, ge=0)
    motion_zones: str = ""
    motion_check_frame_skip: int = Field(default=5, ge=1)
    motion_pixel_threshold: int = Field(default=35, ge=1, le=255)
    motion_area_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    show_motion_zones: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def insightface_provider_list(self) -> list[str]:
        return [provider.strip() for provider in self.insightface_providers.split(",") if provider.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
