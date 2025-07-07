"""
Models for Ultra-Modern Face Recognition System (2025 SOTA)
Contains data structures for face recognition and management
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import hashlib

@dataclass
class ModernFaceEncoding:
    """Modern face encoding with comprehensive metadata and 3D modeling support"""
    encoding: np.ndarray
    person_name: str
    confidence: float
    timestamp: datetime
    model_used: str  # 'sface', 'arcface', 'facenet'
    embedding_size: int
    detection_score: float
    angle_type: str = "frontal"  # 'frontal', 'left_profile', 'right_profile', 'up_angle', 'down_angle'
    pose_data: Optional[Dict] = None  # Head pose estimation data
    unique_id: str = ""  # Unique identifier for this encoding

    def __post_init__(self):
        if not self.unique_id:
            # Generate unique ID based on timestamp and person name
            data = f"{self.person_name}_{self.timestamp.isoformat()}_{self.angle_type}"
            self.unique_id = hashlib.md5(data.encode()).hexdigest()[:12]
