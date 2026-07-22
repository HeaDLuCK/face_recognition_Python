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
    auto_start_saved_cameras: bool = True
    opencv_num_threads: int = Field(default=1, ge=1, le=16)

    snapshot_dir: Path = Path("snapshots")

    insightface_model_name: str = "buffalo_l"
    insightface_providers: str = "CPUExecutionProvider"
    insightface_ctx_id: int = -1
    insightface_det_size: int = Field(default=640, ge=160, le=1280)
    face_detection_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    default_recognition_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    default_duplicate_cooldown_seconds: int = Field(default=60, ge=0)
    recognized_duplicate_cooldown_seconds: int = Field(default=180, ge=0)
    unknown_duplicate_cooldown_seconds: int = Field(default=3600, ge=0)
    unknown_duplicate_similarity_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    unknown_face_crop_similarity_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    unknown_face_cache_max_entries: int = Field(default=1000, ge=1, le=10000)
    unknown_face_crop_cache_max_entries: int = Field(default=500, ge=1, le=5000)
    unknown_face_db_match_limit: int = Field(default=500, ge=1, le=5000)
    detection_event_cache_max_entries: int = Field(default=2000, ge=10, le=50000)
    unknown_face_min_detection_score: float = Field(default=0.75, ge=0.0, le=1.0)
    unknown_face_min_height_px: int = Field(default=70, ge=1)
    unknown_face_min_blur_score: float = Field(default=80.0, ge=0.0)
    unknown_face_skip_weak_known_margin: float = Field(default=0.10, ge=0.0, le=1.0)

    camera_frame_skip: int = Field(default=1, ge=1)
    recognition_interval_seconds: float = Field(default=0.2, ge=0.1)
    recognition_drop_old_frames: bool = True
    face_candidate_buffer_size: int = Field(default=4, ge=1, le=30)
    face_candidate_window_seconds: float = Field(default=0.5, ge=0.1, le=5.0)
    recognition_candidate_window_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    recognition_candidate_min_hits: int = Field(default=2, ge=1, le=30)
    recognition_candidate_score_margin: float = Field(default=0.25, ge=0.0, le=0.5)
    recognition_candidate_fast_margin: float = Field(default=0.05, ge=0.0, le=0.2)
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
    plate_yolo_model_path: Path = Path("app/plates/model/moroccan_plate.pt")
    plate_yolo_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    plate_yolo_imgsz: int = Field(default=640, ge=320, le=1920)
    plate_yolo_device: str = "cpu"
    plate_yolo_mode: Literal["plate", "characters"] = "plate"
    plate_yolo_class_is_text: bool = False
    plate_recognition_interval_seconds: float = Field(default=1.0, ge=0.1)
    plate_recognition_queue_size: int = Field(default=2, ge=1, le=300)
    plate_duplicate_cooldown_seconds: int = Field(default=60, ge=0)
    plate_save_snapshots: bool = True
    plate_min_characters: int = Field(default=4, ge=1, le=20)
    plate_max_detections: int = Field(default=5, ge=1, le=50)
    fire_yolo_model_path: Path = Path("app/fire/model/fire_model.pt")
    fire_yolo_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    fire_yolo_imgsz: int = Field(default=640, ge=320, le=1920)
    fire_yolo_device: str = "cpu"
    fire_detection_interval_seconds: float = Field(default=1.0, ge=0.1)
    fire_detection_queue_size: int = Field(default=2, ge=1, le=300)
    fire_duplicate_cooldown_seconds: int = Field(default=60, ge=0)
    fire_save_snapshots: bool = True
    fire_max_detections: int = Field(default=5, ge=1, le=50)
    cloud_stream_ws_url: str = ""
    cloud_stream_token: str = ""
    cloud_stream_fps: int = Field(default=10, ge=1, le=30)
    cloud_stream_reconnect_seconds: int = Field(default=5, ge=1)
    event_clip_dir: Path = Path("event_clips")
    history_clip_dir: Path = Path("history_clips")
    event_buffer_enabled: bool = False
    event_buffer_seconds: int = Field(default=30, ge=1)
    event_clip_fps: int = Field(default=15, ge=1, le=30)
    event_clip_cooldown_seconds: int = Field(default=30, ge=0)
    event_marker_history_before_seconds: int = Field(default=30, ge=0)
    event_marker_history_after_seconds: int = Field(default=30, ge=0)
    motion_zones: str = ""
    motion_check_frame_skip: int = Field(default=5, ge=1)
    motion_pixel_threshold: int = Field(default=35, ge=1, le=255)
    motion_area_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    show_motion_zones: bool = True
    rtsp_drop_stale_frames: int = Field(default=1, ge=0, le=10)
    rtsp_low_latency: bool = False
    rtsp_read_retries: int = Field(default=3, ge=0, le=20)
    rtsp_reconnect_delay_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    snapshot_purge_batch_size: int = Field(default=200, ge=10, le=2000)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def insightface_provider_list(self) -> list[str]:
        return [provider.strip() for provider in self.insightface_providers.split(",") if provider.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
