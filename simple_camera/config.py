from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "AI Camera Service"

    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "ai_camera_service"
    opencv_num_threads: int = Field(default=1, ge=1, le=16)

    snapshot_dir: Path = Path("snapshots")

    insightface_model_name: str = "buffalo_s"
    insightface_providers: str = "CPUExecutionProvider"
    insightface_ctx_id: int = -1
    insightface_det_size: int = Field(default=640, ge=160, le=1280)
    face_detection_min_score: float = Field(default=0.5, ge=0.0, le=1.0)

    fire_yolo_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    fire_yolo_imgsz: int = Field(default=640, ge=320, le=1920)
    fire_yolo_device: str = "cpu"
    fire_max_detections: int = Field(default=5, ge=1, le=50)
    camera_frame_skip: int = Field(default=1, ge=1)
    default_recognition_threshold: float = Field(default=0.50,ge=0.0,le=1.0)
    recognition_candidate_window_seconds: float = Field(default=0.8,ge=0.1,le=5.0,)
    recognition_candidate_min_hits: int = Field(default=2,ge=1,le=20,)
    recognition_candidate_fast_margin: float = Field(default=0.08,ge=0.0,le=0.5,)
    recognition_candidate_floor_margin: float = Field(default=0.05,ge=0.0,le=0.2,)
    recognition_identity_hold_seconds: float = Field(default=1.2,ge=0.0,le=5.0,)


    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def insightface_provider_list(self) -> list[str]:
        return [provider.strip() for provider in self.insightface_providers.split(",") if provider.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
