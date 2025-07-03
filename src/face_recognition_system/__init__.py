"""
Ultra-Modern Face Recognition System Package
==========================================

A state-of-the-art facial recognition system using 2025 SOTA technology:
- YuNet face detection (2024 SOTA)
- SFace face recognition (2024 SOTA)
- 3D face modeling and multi-angle capture
- Advanced person management
- Real-time processing with optimization
"""

__version__ = "2.0.0"
__author__ = "Face Recognition Team"
__email__ = "team@facialrecognition.ai"

# Legacy system imports (for backward compatibility)
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .alert_system import AlertSystem
from .camera_handler import CameraHandler
from .config import Config
from .main_system import FacialRecognitionSystem

# Enhanced system imports
from .enhanced_main_system import EnhancedFacialRecognitionSystem

__all__ = [
    'FaceDetector',
    'FaceRecognizer', 
    'AlertSystem',
    'CameraHandler',
    'Config',
    'FacialRecognitionSystem',
    'EnhancedFacialRecognitionSystem'
]
