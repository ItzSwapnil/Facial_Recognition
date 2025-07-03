"""
Enhanced Configuration Module for Ultra-Modern Face Recognition System
====================================================================

Professional configuration management with type safety, validation,
and environment-based settings for the 2025 SOTA face recognition system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import json
import yaml
from rich.console import Console

console = Console()


class ModelType(str, Enum):
    """Available model types"""
    YUNET = "yunet"
    SFACE = "sface"
    OPENCV_DNN = "opencv_dnn"
    HOG = "hog"
    CNN = "cnn"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CameraConfig(BaseModel):
    """Camera configuration settings with validation."""
    
    camera_index: int = Field(default=0, ge=0, le=10, description="Default camera index")
    width: int = Field(default=1280, ge=320, le=3840, description="Camera width resolution")
    height: int = Field(default=720, ge=240, le=2160, description="Camera height resolution")
    fps: int = Field(default=30, ge=1, le=120, description="Frames per second")
    buffer_size: int = Field(default=1, ge=1, le=10, description="Camera buffer size")
    auto_detect: bool = Field(default=True, description="Auto-detect available cameras")
    preferred_backend: str = Field(default="DirectShow", description="Preferred camera backend")
    
    @validator('width', 'height')
    def validate_resolution(cls, v):
        """Validate resolution values"""
        if v % 2 != 0:
            raise ValueError("Resolution must be even number")
        return v


class DetectionConfig(BaseModel):
    """Face detection configuration with 2025 SOTA models."""
    
    # Primary detection model (2025 SOTA)
    detection_model: ModelType = Field(default=ModelType.YUNET, description="Primary detection model")
    recognition_model: ModelType = Field(default=ModelType.SFACE, description="Recognition model")
    
    # Detection parameters
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Detection confidence threshold")
    recognition_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Recognition confidence threshold")
    nms_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Non-maximum suppression threshold")
    
    # Face size constraints
    min_face_size: Tuple[int, int] = Field(default=(40, 40), description="Minimum face size (width, height)")
    max_face_size: Tuple[int, int] = Field(default=(300, 300), description="Maximum face size (width, height)")
    
    # Processing optimization
    detection_frequency: int = Field(default=1, ge=1, le=10, description="Detect every N frames")
    max_faces_per_frame: int = Field(default=10, ge=1, le=50, description="Maximum faces to process per frame")
    
    # 3D modeling parameters
    enable_3d_modeling: bool = Field(default=True, description="Enable 3D face modeling")
    angle_threshold: float = Field(default=15.0, ge=0.0, le=45.0, description="Angle threshold for pose detection")
    
    @validator('min_face_size', 'max_face_size')
    def validate_face_size(cls, v):
        """Validate face size tuples"""
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Face size must be tuple of two positive integers")
        return v


class AlertConfig(BaseModel):
    """Advanced alert system configuration."""
    
    # Basic alert settings
    enable_notifications: bool = Field(default=True, description="Enable desktop notifications")
    enable_sound: bool = Field(default=True, description="Enable sound alerts")
    enable_logging: bool = Field(default=True, description="Enable alert logging")
    
    # Timing settings
    notification_timeout: int = Field(default=5, ge=1, le=60, description="Notification timeout in seconds")
    alert_cooldown: int = Field(default=30, ge=5, le=300, description="Cooldown between alerts (seconds)")
    
    # Advanced features
    save_recognition_images: bool = Field(default=True, description="Save images on recognition")
    image_save_path: Path = Field(default=Path("data/recognition_images"), description="Path to save recognition images")
    
    # Sound settings
    sound_file: Optional[Path] = Field(default=None, description="Custom alert sound file")
    sound_volume: float = Field(default=0.5, ge=0.0, le=1.0, description="Sound volume (0.0-1.0)")
    
    # Webhook/API notifications
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL for notifications")
    enable_webhook: bool = Field(default=False, description="Enable webhook notifications")


class PerformanceConfig(BaseModel):
    """Performance optimization settings for production use."""
    
    # Threading and processing
    enable_multithreading: bool = Field(default=True, description="Enable multithreaded processing")
    max_threads: int = Field(default=4, ge=1, le=16, description="Maximum number of processing threads")
    frame_skip: int = Field(default=0, ge=0, le=5, description="Skip every N frames for performance")
    
    # Memory management
    max_memory_usage_mb: int = Field(default=1024, ge=256, le=8192, description="Maximum memory usage in MB")
    cleanup_interval: int = Field(default=300, ge=60, le=3600, description="Memory cleanup interval (seconds)")
    
    # Processing optimization
    enable_gpu_acceleration: bool = Field(default=True, description="Enable GPU acceleration if available")
    batch_processing: bool = Field(default=False, description="Enable batch processing")
    batch_size: int = Field(default=4, ge=1, le=32, description="Batch size for processing")
    
    # Caching
    enable_face_cache: bool = Field(default=True, description="Enable face encoding cache")
    cache_size: int = Field(default=1000, ge=100, le=10000, description="Maximum cache size")
    cache_ttl: int = Field(default=3600, ge=300, le=86400, description="Cache TTL in seconds")


class DatabaseConfig(BaseModel):
    """Database and storage configuration."""
    
    # File paths
    database_path: Path = Field(default=Path("data/face_database.pkl"), description="Face database file path")
    backup_path: Path = Field(default=Path("data/backups"), description="Backup directory path")
    log_path: Path = Field(default=Path("logs"), description="Log files directory")
    model_path: Path = Field(default=Path("data/models"), description="Model files directory")
    
    # Backup settings
    enable_auto_backup: bool = Field(default=True, description="Enable automatic backups")
    backup_interval: int = Field(default=86400, ge=3600, le=604800, description="Backup interval in seconds")
    max_backups: int = Field(default=7, ge=1, le=30, description="Maximum number of backups to keep")
    
    # Data validation
    validate_on_load: bool = Field(default=True, description="Validate data integrity on load")
    compress_data: bool = Field(default=True, description="Compress stored data")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    # Basic settings
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    enable_rich_formatting: bool = Field(default=True, description="Enable rich console formatting")
    
    # File settings
    log_file: Path = Field(default=Path("logs/face_recognition.log"), description="Log file path")
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of log backup files")
    
    # Content settings
    include_timestamps: bool = Field(default=True, description="Include timestamps in logs")
    include_thread_info: bool = Field(default=False, description="Include thread information")
    log_performance_metrics: bool = Field(default=True, description="Log performance metrics")


class SecurityConfig(BaseModel):
    """Security and privacy configuration."""
    
    # Data protection
    encrypt_database: bool = Field(default=False, description="Encrypt face database")
    encryption_key_file: Optional[Path] = Field(default=None, description="Encryption key file path")
    
    # Privacy settings
    blur_faces_in_logs: bool = Field(default=True, description="Blur faces in log images")
    anonymize_unknown_faces: bool = Field(default=True, description="Anonymize unknown faces")
    
    # Access control
    require_authentication: bool = Field(default=False, description="Require authentication")
    admin_password: Optional[str] = Field(default=None, description="Admin password")
    
    # Data retention
    max_data_retention_days: int = Field(default=90, ge=1, le=365, description="Maximum data retention in days")
    auto_cleanup_old_data: bool = Field(default=True, description="Automatically cleanup old data")


class Config(BaseModel):
    """Main configuration class."""
    
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    known_faces_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "known_faces")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    # Known faces database
    known_faces: Dict[str, List[str]] = Field(default_factory=dict, description="Known faces database")
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        return cls()
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        for directory in [self.data_dir, self.known_faces_dir, self.models_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_known_faces_list(self) -> List[str]:
        """Get list of known face names."""
        return list(self.known_faces.keys())
    
    def add_known_face(self, name: str, image_paths: List[str]) -> None:
        """Add a known face to the database."""
        self.known_faces[name] = image_paths
    
    def remove_known_face(self, name: str) -> None:
        """Remove a known face from the database."""
        if name in self.known_faces:
            del self.known_faces[name]


# Global configuration instance
config = Config()

# Default configuration file path
DEFAULT_CONFIG_PATH = config.project_root / "config.json"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or create default."""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    
    global config
    config = Config.load_from_file(config_path)
    config.create_directories()
    return config


def save_config(config_path: Optional[str] = None) -> None:
    """Save current configuration to file."""
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)
    
    config.save_to_file(config_path)


def get_config() -> Config:
    """Get the current configuration."""
    return config
