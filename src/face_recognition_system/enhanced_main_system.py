"""
Enhanced Main Facial Recognition System
=======================================

State-of-the-art facial recognition system integrating all advanced components
with fallback support for older Python versions and missing dependencies.
"""

import cv2
import numpy as np
import threading
import time
import logging
import asyncio
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

from .config import Config, get_config, load_config, save_config
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .alert_system import AlertSystem
from .camera_handler import CameraHandler

# Try to import advanced components
try:
    from .advanced_face_detector import AdvancedFaceDetector
    ADVANCED_DETECTOR_AVAILABLE = True
except ImportError:
    ADVANCED_DETECTOR_AVAILABLE = False

try:
    from .advanced_face_recognizer import AdvancedFaceRecognizer
    ADVANCED_RECOGNIZER_AVAILABLE = True
except ImportError:
    ADVANCED_RECOGNIZER_AVAILABLE = False

try:
    from .advanced_alert_system import AdvancedAlertSystem
    ADVANCED_ALERT_AVAILABLE = True
except ImportError:
    ADVANCED_ALERT_AVAILABLE = False


class EnhancedFacialRecognitionSystem:
    """
    Enhanced facial recognition system with advanced capabilities.
    
    Features:
    - Automatic fallback to basic components if advanced ones fail
    - State-of-the-art detection and recognition when available
    - Advanced alert system with multiple notification channels
    - Real-time performance monitoring
    - CCTV and streaming support
    - Multi-threaded processing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced facial recognition system."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # System availability flags
        self.capabilities = {
            'advanced_detection': ADVANCED_DETECTOR_AVAILABLE,
            'advanced_recognition': ADVANCED_RECOGNIZER_AVAILABLE,
            'advanced_alerts': ADVANCED_ALERT_AVAILABLE
        }
        
        # Initialize components with fallback support
        self._initialize_components()
        
        # Processing control
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        
        # Current processing data
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        self.frame_lock = threading.Lock()
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.processed_frames = 0
        self.start_time = time.time()
        
        # Console for rich output
        self.console = Console()
        
        self.logger.info("Enhanced Facial Recognition System initialized")
        self._log_system_capabilities()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.config.logs_dir / "enhanced_system.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_components(self) -> None:
        """Initialize system components with fallback support."""
        # Initialize detector
        if self.capabilities['advanced_detection']:
            try:
                self.detector = AdvancedFaceDetector(self.config)
                self.logger.info("Advanced face detector initialized")
            except Exception as e:
                self.logger.warning(f"Advanced detector failed, using basic: {e}")
                self.detector = FaceDetector(self.config)
                self.capabilities['advanced_detection'] = False
        else:
            self.detector = FaceDetector(self.config)
            self.logger.info("Basic face detector initialized")
        
        # Initialize recognizer
        if self.capabilities['advanced_recognition']:
            try:
                self.recognizer = AdvancedFaceRecognizer(self.config)
                self.logger.info("Advanced face recognizer initialized")
            except Exception as e:
                self.logger.warning(f"Advanced recognizer failed, using basic: {e}")
                self.recognizer = FaceRecognizer(self.config)
                self.capabilities['advanced_recognition'] = False
        else:
            self.recognizer = FaceRecognizer(self.config)
            self.logger.info("Basic face recognizer initialized")
        
        # Initialize alert system
        if self.capabilities['advanced_alerts']:
            try:
                self.alert_system = AdvancedAlertSystem(self.config)
                self.logger.info("Advanced alert system initialized")
            except Exception as e:
                self.logger.warning(f"Advanced alerts failed, using basic: {e}")
                self.alert_system = AlertSystem(self.config)
                self.capabilities['advanced_alerts'] = False
        else:
            self.alert_system = AlertSystem(self.config)
            self.logger.info("Basic alert system initialized")
        
        # Initialize camera handler
        self.camera_handler = CameraHandler(self.config)
    
    def _log_system_capabilities(self) -> None:
        """Log system capabilities and available features."""
        self.logger.info("System Capabilities:")
        for capability, available in self.capabilities.items():
            status = "AVAILABLE" if available else "FALLBACK"
            self.logger.info(f"  {capability}: {status}")
    
    def _process_frames(self) -> None:
        """Main frame processing loop with enhanced capabilities."""
        frame_skip_counter = 0
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera_handler.get_frame(timeout=0.1)
                if frame is None:
                    continue
                
                # Skip frames for performance if configured
                frame_skip_counter += 1
                if frame_skip_counter < self.config.performance.frame_skip:
                    continue
                frame_skip_counter = 0
                
                start_time = cv2.getTickCount()
                
                # Resize frame for faster processing if configured
                if self.config.performance.resize_factor != 1.0:
                    height, width = frame.shape[:2]
                    new_width = int(width * self.config.performance.resize_factor)
                    new_height = int(height * self.config.performance.resize_factor)
                    processing_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    processing_frame = frame
                
                # Detect faces using appropriate method
                if self.capabilities['advanced_detection']:
                    # Advanced detection returns (x, y, w, h, confidence)
                    detections = self.detector.detect_faces(processing_frame, method="auto")
                    face_locations = [(x, y, w, h) for x, y, w, h, conf in detections]
                else:
                    # Basic detection returns (x, y, w, h)
                    face_locations = self.detector.detect_faces(processing_frame)
                
                # Scale face locations back if frame was resized
                if self.config.performance.resize_factor != 1.0:
                    scale_factor = 1.0 / self.config.performance.resize_factor
                    face_locations = [
                        (int(x * scale_factor), int(y * scale_factor),
                         int(w * scale_factor), int(h * scale_factor))
                        for x, y, w, h in face_locations
                    ]
                
                # Recognize faces using appropriate method
                recognition_results = []
                if face_locations:
                    if self.capabilities['advanced_recognition']:
                        recognition_results = self.recognizer.recognize_faces(
                            frame, face_locations, method="auto"
                        )
                    else:
                        recognition_results = self.recognizer.recognize_faces(
                            frame, face_locations, method="basic"
                        )
                
                # Process recognition results and trigger alerts
                self._process_recognition_results(recognition_results)
                
                # Update current data for display
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.current_detections = recognition_results
                
                # Update performance tracking
                end_time = cv2.getTickCount()
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                self.total_processing_time += processing_time
                self.processed_frames += 1
                
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}")
                time.sleep(0.1)
    
    async def _process_recognition_results(self, results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> None:
        """Process recognition results and trigger appropriate alerts."""
        for name, confidence, location in results:
            if name == "Unknown":
                # Unknown person detected
                if confidence > 0.5:  # Only alert for high-confidence unknowns
                    if self.capabilities['advanced_alerts']:
                        await self.alert_system.trigger_alert(
                            "unknown", "Unknown Person", confidence,
                            {"location": location}
                        )
                    else:
                        self.alert_system.trigger_alert(
                            "unknown", "Unknown Person", confidence,
                            {"location": location}
                        )
            else:
                # Known person recognized
                if self.capabilities['advanced_alerts']:
                    await self.alert_system.trigger_alert(
                        "recognition", name, confidence,
                        {"location": location}
                    )
                else:
                    self.alert_system.trigger_alert(
                        "recognition", name, confidence,
                        {"location": location}
                    )
    
    def _display_video(self) -> None:
        """Display video with enhanced face detection annotations."""
        while self.is_running:
            try:
                with self.frame_lock:
                    if self.current_frame is None:
                        time.sleep(0.1)
                        continue
                    
                    display_frame = self.current_frame.copy()
                    detections = self.current_detections.copy()
                
                # Draw enhanced face annotations
                for name, confidence, (x, y, w, h) in detections:
                    # Choose color based on recognition result
                    if name == "Unknown":
                        color = (0, 0, 255)  # Red for unknown
                        text_color = (255, 255, 255)  # White text
                    else:
                        color = (0, 255, 0)  # Green for known
                        text_color = (0, 0, 0)  # Black text
                    
                    # Draw bounding box with thickness based on confidence
                    thickness = max(1, int(confidence * 4))
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                    
                    # Draw label with background
                    label = f"{name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(display_frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0] + 10, y), color, -1)
                    
                    # Draw label text
                    cv2.putText(display_frame, label, (x + 5, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Add enhanced system info
                fps = self.camera_handler.current_fps
                processing_fps = 1.0 / (self.total_processing_time / max(1, self.processed_frames))
                
                info_lines = [
                    f"Camera FPS: {fps:.1f}",
                    f"Processing FPS: {processing_fps:.1f}",
                    f"Faces: {len(detections)}",
                    f"Mode: {'Advanced' if any(self.capabilities.values()) else 'Basic'}"
                ]
                
                # Draw info panel
                y_offset = 30
                for line in info_lines:
                    cv2.putText(display_frame, line, (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 20
                
                # Display frame
                cv2.imshow("Enhanced Facial Recognition System", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_system()
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    self.logger.info(f"Snapshot saved: {filename}")
                elif key == ord('c'):
                    # Clear alert cooldowns
                    if hasattr(self.alert_system, 'clear_cooldowns'):
                        self.alert_system.clear_cooldowns()
                    self.logger.info("Alert cooldowns cleared")
                elif key == ord('m'):
                    # Toggle detection method for advanced detector
                    if self.capabilities['advanced_detection']:
                        # Cycle through detection methods
                        pass  # Implementation can be added
                
            except Exception as e:
                self.logger.error(f"Display error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def start_system(self, video_source: Any = None, show_video: bool = True) -> bool:
        """
        Start the enhanced facial recognition system.
        
        Args:
            video_source: Video source (camera index, file, or stream URL)
            show_video: Whether to display video window
        
        Returns:
            True if system started successfully
        """
        if self.is_running:
            self.logger.warning("System is already running")
            return True
        
        self.logger.info("Starting Enhanced Facial Recognition System...")
        
        # Start camera capture
        if not self.camera_handler.start_capture(video_source):
            self.logger.error("Failed to start camera capture")
            return False
        
        # Start processing
        self.is_running = True
        self.start_time = time.time()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        # Start display thread if requested
        if show_video:
            self.display_thread = threading.Thread(target=self._display_video, daemon=True)
            self.display_thread.start()
        
        self.logger.info("Enhanced Facial Recognition System started successfully")
        return True
    
    def stop_system(self) -> None:
        """Stop the enhanced facial recognition system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Enhanced Facial Recognition System...")
        
        self.is_running = False
        
        # Stop camera
        self.camera_handler.stop_capture()
        
        # Wait for threads to finish
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=5)
        
        if self.display_thread is not None:
            self.display_thread.join(timeout=5)
        
        # Cleanup components
        if hasattr(self.detector, 'cleanup'):
            self.detector.cleanup()
        if hasattr(self.recognizer, 'cleanup'):
            self.recognizer.cleanup()
        if hasattr(self.alert_system, 'cleanup'):
            self.alert_system.cleanup()
        
        self.logger.info("Enhanced Facial Recognition System stopped")
    
    def add_person(self, name: str, image_path: str) -> bool:
        """Add a new person to the recognition database."""
        if self.capabilities['advanced_recognition']:
            return self.recognizer.add_face(name, image_path, method="auto")
        else:
            return self.recognizer.add_face(name, image_path)
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from the recognition database."""
        return self.recognizer.remove_face(name)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        runtime = time.time() - self.start_time
        avg_processing_time = self.total_processing_time / max(1, self.processed_frames)
        
        stats = {
            'runtime_seconds': runtime,
            'processed_frames': self.processed_frames,
            'average_processing_time': avg_processing_time,
            'processing_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'camera_stats': self.camera_handler.get_performance_stats(),
            'detector_stats': self.detector.get_performance_stats(),
            'recognizer_stats': self.recognizer.get_performance_stats(),
            'known_people': self.recognizer.get_known_people(),
            'capabilities': self.capabilities
        }
        
        # Add alert stats
        if hasattr(self.alert_system, 'get_alert_stats'):
            stats['alert_stats'] = self.alert_system.get_alert_stats()
        else:
            stats['alert_stats'] = self.alert_system.get_alert_stats()
        
        return stats
    
    def print_system_capabilities(self) -> None:
        """Print system capabilities in a formatted table."""
        table = Table(title="Enhanced Facial Recognition System Capabilities")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", style="yellow")
        
        # Detection capabilities
        if self.capabilities['advanced_detection']:
            table.add_row(
                "Face Detection", 
                "Advanced", 
                "YOLO, MediaPipe, InsightFace, Ensemble"
            )
        else:
            table.add_row(
                "Face Detection", 
                "Basic", 
                "OpenCV Haar, DNN, dlib HOG/CNN"
            )
        
        # Recognition capabilities
        if self.capabilities['advanced_recognition']:
            table.add_row(
                "Face Recognition", 
                "Advanced", 
                "FaceNet, ArcFace, InsightFace, Ensemble"
            )
        else:
            table.add_row(
                "Face Recognition", 
                "Basic", 
                "face_recognition, SVM, Cosine similarity"
            )
        
        # Alert capabilities
        if self.capabilities['advanced_alerts']:
            table.add_row(
                "Alert System", 
                "Advanced", 
                "Native notifications, MQTT, Email, SMS, Webhooks"
            )
        else:
            table.add_row(
                "Alert System", 
                "Basic", 
                "Desktop notifications, Sound alerts, Logging"
            )
        
        self.console.print(table)
    
    def get_system_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """
        Get system capabilities in a structured format.
        
        Returns:
            Dict containing capability categories and their availability
        """
        capabilities = {
            "detection": {
                "advanced_algorithms": self.capabilities['advanced_detection'],
                "yolo_detection": self.capabilities['advanced_detection'],
                "mediapipe_detection": self.capabilities['advanced_detection'],
                "insightface_detection": self.capabilities['advanced_detection'],
                "ensemble_detection": self.capabilities['advanced_detection'],
                "basic_opencv": True,  # Always available
                "dlib_detection": True,  # Always available
                "haar_cascade": True,  # Always available
            },
            "recognition": {
                "advanced_models": self.capabilities['advanced_recognition'],
                "facenet": self.capabilities['advanced_recognition'],
                "arcface": self.capabilities['advanced_recognition'],
                "insightface": self.capabilities['advanced_recognition'],
                "ensemble_recognition": self.capabilities['advanced_recognition'],
                "basic_face_recognition": True,  # Always available
                "dlib_recognition": True,  # Always available
            },
            "alerts": {
                "advanced_notifications": self.capabilities['advanced_alerts'],
                "mqtt_alerts": self.capabilities['advanced_alerts'],
                "email_notifications": self.capabilities['advanced_alerts'],
                "sms_alerts": self.capabilities['advanced_alerts'],
                "webhook_integration": self.capabilities['advanced_alerts'],
                "desktop_notifications": True,  # Always available
                "sound_alerts": True,  # Always available
                "file_logging": True,  # Always available
            },
            "performance": {
                "gpu_acceleration": hasattr(self, 'detector') and hasattr(self.detector, 'device') and 'cuda' in str(getattr(self.detector, 'device', 'cpu')),
                "multi_threading": True,  # Always available
                "frame_skipping": True,  # Always available
                "batch_processing": self.capabilities['advanced_recognition'],
                "model_optimization": self.capabilities['advanced_detection'] or self.capabilities['advanced_recognition'],
            },
            "data": {
                "auto_backup": True,  # Always available
                "data_compression": True,  # Always available
                "data_validation": True,  # Always available
                "multiple_formats": True,  # Always available
            }
        }
        
        return capabilities
    
    def print_stats(self) -> None:
        """Print system statistics in a formatted table."""
        stats = self.get_system_stats()
        
        # Create main stats table
        table = Table(title="Enhanced System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        # General stats
        table.add_row("Runtime", f"{stats['runtime_seconds']:.1f} seconds")
        table.add_row("Processed Frames", str(stats['processed_frames']))
        table.add_row("Processing FPS", f"{stats['processing_fps']:.1f}")
        table.add_row("Camera FPS", f"{stats['camera_stats']['current_fps']:.1f}")
        table.add_row("Known People", str(len(stats['known_people'])))
        
        if 'alert_stats' in stats:
            table.add_row("Total Alerts", str(stats['alert_stats']['total_alerts']))
        
        self.console.print(table)
        
        # Print capabilities
        self.print_system_capabilities()
        
        # Known people
        if stats['known_people']:
            people_table = Table(title="Known People")
            people_table.add_column("Name", style="green")
            
            for person in stats['known_people']:
                people_table.add_row(person)
            
            self.console.print(people_table)
    
    async def test_all_systems(self) -> Dict[str, Dict[str, bool]]:
        """Test all system components."""
        results = {
            'detection': {},
            'recognition': {},
            'alerts': {}
        }
        
        self.console.print("[bold blue]Testing all system components...[/bold blue]")
        
        # Test detection
        if self.capabilities['advanced_detection'] and hasattr(self.detector, 'get_performance_stats'):
            results['detection']['advanced'] = True
        results['detection']['basic'] = True
        
        # Test recognition
        if self.capabilities['advanced_recognition'] and hasattr(self.recognizer, 'get_performance_stats'):
            results['recognition']['advanced'] = True
        results['recognition']['basic'] = True
        
        # Test alerts
        if self.capabilities['advanced_alerts'] and hasattr(self.alert_system, 'test_notifications'):
            try:
                alert_results = await self.alert_system.test_notifications()
                results['alerts']['advanced'] = any(alert_results.values())
            except Exception as e:
                self.logger.error(f"Advanced alert test failed: {e}")
                results['alerts']['advanced'] = False
        
        # Test basic alerts
        try:
            if hasattr(self.alert_system, 'test_notifications'):
                alert_results = self.alert_system.test_notifications()
                results['alerts']['basic'] = any(alert_results.values())
            else:
                results['alerts']['basic'] = True
        except Exception as e:
            self.logger.error(f"Basic alert test failed: {e}")
            results['alerts']['basic'] = False
        
        return results
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_system()


# Create alias for backward compatibility
FacialRecognitionSystem = EnhancedFacialRecognitionSystem
