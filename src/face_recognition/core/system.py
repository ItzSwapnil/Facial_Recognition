"""
Main system module for Ultra-Modern Face Recognition
Integrates all components into a cohesive system
"""

import cv2
import time
import logging
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from rich.console import Console

from src.face_recognition.core.face_detector import FaceDetector
from src.face_recognition.core.face_recognizer import FaceRecognizer
from src.face_recognition.core.face_storage import FaceStorage
from src.face_recognition.models.face_models import ModernFaceEncoding
from src.face_recognition.utils.managers import DatabaseManager, NotificationManager, CameraManager
from src.face_recognition.utils.onnx_helper import onnx_helper
from src.face_recognition.ui.ui_components import FaceRecognitionUI

# Get project root directory - handle both direct execution and import scenarios
current_file = Path(__file__)
if current_file.is_absolute():
    project_root = current_file.parent.parent.parent.parent
else:
    # If running from a different directory, use relative path from current working dir
    project_root = Path.cwd()

# Create logs directory regardless of where we're running from
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
log_file = log_dir / "face_recognition.log"

# Set up logging with absolute path to log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class UltraModernFaceRecognitionSystem:
    """
    2025 State-of-the-Art Face Recognition System

    Integrates:
    - YuNet face detection (2025 SOTA)
    - SFace face recognition (2025 SOTA)
    - ONNX runtime optimization
    - GPU acceleration
    - Real-time processing
    - 3D face modeling
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the integrated face recognition system

        Args:
            data_dir: Directory for data storage
        """
        self.console = Console()
        self.ui = FaceRecognitionUI(self.console)

        # Initialize paths
        self.data_dir = Path(data_dir)
        self.known_faces_dir = self.data_dir / "known_faces"
        self.models_dir = self.data_dir / "models"

        # Create directories
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize managers
        self.camera_manager = CameraManager(self.console)
        self.db_manager = DatabaseManager(self.known_faces_dir, self.console)
        self.notification_manager = NotificationManager(self.console)

        # Initialize components
        self.face_detector = FaceDetector(self.models_dir)
        self.face_recognizer = FaceRecognizer(self.models_dir)

        # Initialize face storage
        self.embedding_recorder = FaceStorage(
            face_detector=self.face_detector,
            face_recognizer=self.face_recognizer,
            database_path=self.known_faces_dir / "modern_face_database.pkl"
        )

        # Recognition settings
        self.recognition_threshold = 0.6  # Cosine similarity threshold
        self.input_size = (320, 240)  # Optimized for real-time processing

        # Load known faces
        self.face_encodings = self.db_manager.load_known_faces()

        # Initialize camera detection
        self.camera_manager.detect_available_cameras()

        # Setup ONNX Runtime status using the helper
        self.onnx_available = onnx_helper.onnx_available
        self.onnx_providers = onnx_helper.onnx_providers

        # Log ONNX status
        logger.info(onnx_helper.get_status_message())

    def add_known_face(self, image: np.ndarray, person_name: str, angle_type: str = "frontal") -> bool:
        """
        Add a known face using the latest technology with angle support

        Args:
            image: Input image containing a face
            person_name: Name of the person
            angle_type: Type of angle ('frontal', 'left_profile', 'right_profile', 'up_angle', 'down_angle')

        Returns:
            Success status as boolean
        """
        # Detect faces using YuNet
        faces = self.face_detector.detect_faces(image)

        if len(faces) == 0:
            self.console.print(f"❌ No face detected for {person_name}", style="red")
            return False

        if len(faces) > 1:
            self.console.print(f"⚠️ Multiple faces detected, using the one with highest confidence", style="yellow")
            # Use the face with highest confidence
            faces = [max(faces, key=lambda f: f[14] if len(f) > 14 else 1.0)]

        face_data = faces[0]

        # Extract encoding using SFace (SOTA 2025)
        encoding = self.face_recognizer.extract_face_encoding(image, face_data)

        if encoding is None or len(encoding) == 0:
            self.console.print(f"❌ Failed to extract encoding for {person_name}", style="red")
            return False

        # Auto-detect pose if angle_type is "auto"
        if angle_type == "auto":
            x, y, w, h = face_data[:4].astype(int)
            face_region = image[max(0, y):min(image.shape[0], y+h),
                               max(0, x):min(image.shape[1], x+w)]
            if face_region.size > 0:
                angle_type = self.face_detector.detect_head_pose(face_region)
            else:
                angle_type = "frontal"

        # Create modern face encoding with angle information
        face_encoding = ModernFaceEncoding(
            encoding=encoding,
            person_name=person_name,
            confidence=1.0,
            timestamp=datetime.now(),
            model_used='sface' if self.face_recognizer.face_recognizer else 'manual_advanced',
            embedding_size=len(encoding),
            detection_score=float(face_data[14]) if len(face_data) > 14 else 1.0,
            angle_type=angle_type
        )

        # Add to database
        self.face_encodings.append(face_encoding)

        # Save to disk
        self.db_manager.save_known_faces(self.face_encodings)

        self.console.print(f"✅ Added {person_name} to face database using {face_encoding.model_used}", style="green")
        self.console.print(f"   Embedding size: {face_encoding.embedding_size}, Detection score: {face_encoding.detection_score:.3f}, Angle: {angle_type}", style="cyan")

        return True

    def recognize_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize faces using ultra-modern algorithms

        Args:
            image: Input image for face recognition

        Returns:
            List of recognition results
        """
        # Detect faces
        faces = self.face_detector.detect_faces(image)

        results = []

        for face_data in faces:
            # Extract encoding
            encoding = self.face_recognizer.extract_face_encoding(image, face_data)

            if encoding is None or len(encoding) == 0:
                continue

            # Find best match using the face recognizer
            best_match_idx, best_similarity = self.face_recognizer.find_best_match(encoding, self.face_encodings)

            # Prepare result
            if best_match_idx >= 0:
                best_match = self.face_encodings[best_match_idx]
                result = {
                    'name': best_match.person_name,
                    'confidence': best_similarity,
                    'box': face_data[:4].astype(int).tolist(),
                    'recognized': True,
                    'model_used': best_match.model_used,
                    'detection_score': float(face_data[14]) if len(face_data) > 14 else 1.0
                }

                # Send notification
                self.notification_manager.send_notification(best_match.person_name, best_similarity)

            else:
                result = {
                    'name': 'Unknown',
                    'confidence': 0.0,
                    'box': face_data[:4].astype(int).tolist(),
                    'recognized': False,
                    'model_used': 'none',
                    'detection_score': float(face_data[14]) if len(face_data) > 14 else 1.0
                }

            results.append(result)

        return results

    def run_live_recognition(self, camera_index: Optional[int] = None):
        """
        Run live face recognition with advanced UI

        Args:
            camera_index: Camera index to use (if None, uses current selected camera)
        """
        if camera_index is None:
            camera_index = self.camera_manager.current_camera_index

        self.console.print("🎥 Starting ultra-modern live recognition...", style="bold cyan")
        self.console.print("📡 Using 2025 SOTA algorithms", style="cyan")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            self.console.print("❌ Failed to open camera", style="red")
            return

        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Performance tracking
        fps_counter = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Recognize faces
                frame_start = time.time()
                results = self.recognize_faces(frame)
                processing_time = time.time() - frame_start

                # Update FPS calculation
                fps_counter += 1
                elapsed = time.time() - start_time
                fps = fps_counter / elapsed if elapsed > 0 else 0

                if fps_counter % 30 == 0:  # Update every 30 frames
                    # Recalculate FPS
                    elapsed = time.time() - start_time
                    fps = fps_counter / elapsed

                # Draw results
                frame = self.ui.draw_recognition_results(frame, results, processing_time, fps)

                # Show frame
                cv2.imshow('Ultra-Modern Face Recognition System', frame)

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.console.print("\n🛑 Stopping face recognition...", style="yellow")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Final performance report
            if fps_counter > 0:
                total_time = time.time() - start_time
                avg_fps = fps_counter / total_time
                self.console.print(f"📊 Session completed: {fps_counter} frames, {avg_fps:.1f} avg FPS", style="cyan")

    def capture_simple_face(self, person_name: str):
        """
        Capture a single face for recognition

        Args:
            person_name: Name of the person
        """
        self.console.print(f"📸 Setting up camera to capture {person_name}'s face...", style="cyan")
        self.console.print(f"🎥 Using Camera {self.camera_manager.current_camera_index}", style="blue")
        self.console.print("💡 Position yourself clearly in front of the camera", style="yellow")
        self.console.print("⌨️ Press 's' to save, 'q' to quit", style="yellow")

        # Capture face
        cap = cv2.VideoCapture(self.camera_manager.current_camera_index)
        if not cap.isOpened():
            self.console.print("❌ Failed to open camera", style="red")
            return False

        success = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Show preview with face detection
                faces = self.face_detector.detect_faces(frame)

                if len(faces) == 0:
                    cv2.putText(frame, "No face detected - position yourself clearly",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    for face_data in faces:
                        x, y, w, h = face_data[:4].astype(int)
                        confidence = face_data[14] if len(face_data) > 14 else 1.0
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Face detected ({confidence:.2f})", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Add instructions
                cv2.putText(frame, "Press 's' to save face, 'q' to quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Adding: {person_name}",
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow('Capture Face - Ultra-Modern System', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    success = self.add_known_face(frame, person_name)
                    if success:
                        self.console.print(f"✅ Successfully added {person_name} using SOTA technology!", style="green")
                    break
                elif key == ord('q'):
                    break

        except KeyboardInterrupt:
            pass

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return success

    def get_person_statistics(self):
        """
        Get statistics about persons in the database

        Returns:
            Dictionary of person statistics
        """
        person_stats = {}
        for face in self.face_encodings:
            name = face.person_name
            if name not in person_stats:
                person_stats[name] = {
                    'count': 0,
                    'angles': set(),
                    'encodings': [],
                    'latest_timestamp': face.timestamp,
                    'avg_confidence': 0,
                    'avg_detection_score': 0
                }

            stats = person_stats[name]
            stats['count'] += 1
            stats['angles'].add(getattr(face, 'angle_type', 'frontal'))
            stats['encodings'].append(face)
            if face.timestamp > stats['latest_timestamp']:
                stats['latest_timestamp'] = face.timestamp

        # Calculate averages
        for name, stats in person_stats.items():
            stats['avg_confidence'] = sum(f.confidence for f in stats['encodings']) / len(stats['encodings'])
            stats['avg_detection_score'] = sum(f.detection_score for f in stats['encodings']) / len(stats['encodings'])
            stats['angles'] = list(stats['angles'])

        return person_stats

    def delete_person(self, person_name: str):
        """
        Delete all encodings for a specific person

        Args:
            person_name: Name of the person to delete

        Returns:
            Success status as boolean
        """
        initial_count = len(self.face_encodings)
        self.face_encodings = [face for face in self.face_encodings if face.person_name != person_name]
        deleted_count = initial_count - len(self.face_encodings)

        if deleted_count > 0:
            self.db_manager.save_known_faces(self.face_encodings)
            self.console.print(f"✅ Deleted {deleted_count} encoding(s) for {person_name}", style="green")
            return True
        else:
            self.console.print(f"❌ No encodings found for {person_name}", style="red")
            return False

    def merge_persons(self, source_name: str, target_name: str):
        """
        Merge all encodings from source_name to target_name

        Args:
            source_name: Source person name (from)
            target_name: Target person name (to)

        Returns:
            Success status as boolean
        """
        merged_count = 0
        for face in self.face_encodings:
            if face.person_name == source_name:
                face.person_name = target_name
                merged_count += 1

        if merged_count > 0:
            self.db_manager.save_known_faces(self.face_encodings)
            self.console.print(f"✅ Merged {merged_count} encoding(s) from '{source_name}' to '{target_name}'", style="green")
            return True
        else:
            self.console.print(f"❌ No encodings found for '{source_name}'", style="red")
            return False

    def manage_persons_menu(self):
        """Interactive person management menu"""
        while True:
            choice = self.ui.display_person_management_menu()

            # Show person statistics
            person_stats = self.get_person_statistics()
            self.ui.display_persons_table(person_stats)

            if choice == '1':
                # Delete person
                name = input("👤 Enter person name to delete: ").strip()
                if name and name in person_stats:
                    confirm = input(f"⚠️ Are you sure you want to delete all data for '{name}'? (y/N): ").strip().lower()
                    if confirm == 'y':
                        self.delete_person(name)
                else:
                    self.console.print("❌ Person not found", style="red")

            elif choice == '2':
                # Merge persons
                source = input("👤 Enter source person name (to merge from): ").strip()
                target = input("👤 Enter target person name (to merge to): ").strip()
                if source and target and source in person_stats:
                    if target not in person_stats:
                        confirm = input(f"Target '{target}' doesn't exist. Create new? (y/N): ").strip().lower()
                        if confirm != 'y':
                            continue
                    self.merge_persons(source, target)
                else:
                    self.console.print("❌ Source person not found", style="red")

            elif choice == '3':
                # View detailed statistics
                name = input("👤 Enter person name for detailed stats: ").strip()
                if name and name in person_stats:
                    self.ui.display_person_details(name, person_stats[name])
                    input("\nPress Enter to continue...")
                else:
                    self.console.print("❌ Person not found", style="red")

            elif choice == '4':
                # Back to main menu
                break

            else:
                self.console.print("❌ Invalid choice", style="red")

    def get_system_info(self):
        """
        Get system information for display

        Returns:
            Dictionary of system information
        """
        from src.face_recognition.utils.onnx_helper import onnx_helper

        # Get more detailed ONNX information
        onnx_status = "Available" if self.onnx_available else "Not Available"

        if self.onnx_available:
            if 'CUDAExecutionProvider' in self.onnx_providers:
                onnx_details = f"GPU Optimized (CUDA) - v{onnx_helper.onnx_version}"
            elif 'TensorrtExecutionProvider' in self.onnx_providers:
                onnx_details = f"GPU Optimized (TensorRT) - v{onnx_helper.onnx_version}"
            else:
                onnx_details = f"CPU Only - v{onnx_helper.onnx_version}"
        else:
            error_msg = onnx_helper.error_message or "Unknown error"
            if "DLL load failed" in error_msg:
                onnx_details = "DLL initialization failed - CUDA/driver mismatch"
            else:
                onnx_details = f"Not installed or error: {error_msg[:50]}..."

        # OpenCV info with proper encoding
        opencv_status = "Available"

        info = {
            "OpenCV": {
                "status": opencv_status,
                "details": cv2.__version__
            },
            "YuNet Face Detection": {
                "status": "Available" if hasattr(self.face_detector, 'yunet_available') and self.face_detector.yunet_available else "Not Available",
                "details": "2025 SOTA"
            },
            "SFace Recognition": {
                "status": "Available" if self.face_recognizer.face_recognizer else "Not Available",
                "details": "2025 SOTA"
            },
            "ONNX Runtime": {
                "status": onnx_status,
                "details": onnx_details
            },
            "Known Faces": {
                "status": "Loaded",
                "details": str(len(self.face_encodings))
            },
            "Recognition Threshold": {
                "status": "Configured",
                "details": str(self.recognition_threshold)
            },
            "Current Camera": {
                "status": "Selected",
                "details": f"Camera {self.camera_manager.current_camera_index}"
            },
            "Available Cameras": {
                "status": "Detected",
                "details": str(len(self.camera_manager.available_cameras))
            }
        }

        return info


    def recover_camera(self, cap=None):
        """
        Attempt to recover a failed camera connection, particularly useful for MSMF errors

        Args:
            cap: Optional existing VideoCapture object to close

        Returns:
            A new VideoCapture object
        """
        logger.info("Attempting to recover camera connection")
        self._camera_recovery_count = getattr(self, '_camera_recovery_count', 0) + 1

        try:
            # Close existing capture if provided
            if cap is not None:
                cap.release()

            # Small delay to allow camera to reset
            time.sleep(0.5)

            # Create new capture
            new_cap = self._setup_video_capture()

            if new_cap is not None and new_cap.isOpened():
                logger.info(f"Camera recovery successful (attempt {self._camera_recovery_count})")
                return new_cap
            else:
                logger.error(f"Camera recovery failed (attempt {self._camera_recovery_count})")
                # If we've tried too many times, fall back to any available camera
                if self._camera_recovery_count >= 3:
                    # Try different camera indexes
                    for idx in range(3):  # Try first 3 camera indexes
                        alt_cap = cv2.VideoCapture(idx)
                        if alt_cap.isOpened():
                            logger.info(f"Found alternative working camera at index {idx}")
                            return alt_cap

                # Last resort: return a new capture of the current camera
                return cv2.VideoCapture(self.camera_manager.current_camera_index)
        except Exception as e:
            logger.error(f"Error during camera recovery: {e}")
            return cv2.VideoCapture(self.camera_manager.current_camera_index)

    def _setup_video_capture(self):
        """
        Set up video capture with optimal settings

        Returns:
            VideoCapture object or None if failed
        """
        try:
            cap = cv2.VideoCapture(self.camera_manager.current_camera_index)
            if not cap.isOpened():
                logger.error("Failed to open camera")
                return None

            # Set camera properties for optimal performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            # Disable auto focus if available (to reduce fluctuations)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            return cap
        except Exception as e:
            logger.error(f"Error setting up camera: {e}")
            return None
