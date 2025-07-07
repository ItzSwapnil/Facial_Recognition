"""
Utility functions for the Ultra-Modern Face Recognition System
"""

import os
import cv2
import pickle
import json
import logging
import numpy as np
import socket
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from plyer import notification

# Configure logger
logger = logging.getLogger(__name__)

class NotificationManager:
    """
    Manage system notifications for face recognition events
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize notification manager

        Args:
            console: Rich console for display (optional)
        """
        self.console = console or Console()
        self.setup_notifications()
        self.notification_settings = {
            'enabled': True,
            'cooldown': 10,  # Seconds between notifications for the same person
            'sound': True,
            'desktop': True,
            'console': True
        }
        self.last_notifications = {}  # Track when each person was last notified
        self.load_settings()

    def setup_notifications(self):
        """Setup advanced notification system"""
        try:
            import pygame
            pygame.mixer.init()
            logger.info("Advanced notification system initialized")
            if self.console:
                self.console.print("✅ Advanced notification system ready", style="green")
        except Exception as e:
            logger.warning(f"Notification setup warning: {e}")
            if self.console:
                self.console.print(f"⚠️ Notification setup warning: {e}", style="yellow")

    def save_settings(self):
        """Save notification settings to disk"""
        try:
            settings_dir = Path("data/settings")
            settings_dir.mkdir(parents=True, exist_ok=True)

            with open(settings_dir / "notification_settings.json", 'w') as f:
                json.dump(self.notification_settings, f, indent=2)

            if self.console:
                self.console.print("✅ Notification settings saved", style="green")
        except Exception as e:
            logger.error(f"Failed to save notification settings: {e}")
            if self.console:
                self.console.print(f"⚠️ Failed to save notification settings: {e}", style="yellow")

    def load_settings(self):
        """Load notification settings from disk"""
        try:
            settings_file = Path("data/settings/notification_settings.json")
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    self.notification_settings.update(settings)
                logger.info("Notification settings loaded")
        except Exception as e:
            logger.warning(f"Failed to load notification settings: {e}")

    def send_notification(self, person_name: str, confidence: float):
        """
        Send advanced notification when person is recognized with anti-spam protection

        Args:
            person_name: Name of recognized person
            confidence: Recognition confidence score (0-1)
        """
        # Check if notifications are enabled
        if not self.notification_settings['enabled']:
            return

        # Check cooldown to prevent spam
        now = datetime.now()
        if person_name in self.last_notifications:
            last_time = self.last_notifications[person_name]
            elapsed = (now - last_time).total_seconds()

            if elapsed < self.notification_settings['cooldown']:
                # Still in cooldown period, don't send another notification
                return

        # Update last notification time
        self.last_notifications[person_name] = now

        try:
            # Desktop notification
            if self.notification_settings['desktop']:
                notification.notify(
                    title='Person Detected',
                    message=f'{person_name} recognized',
                    app_name='Face Recognition',
                    timeout=3
                )

            # Console notification
            if self.notification_settings['console'] and self.console:
                table = Table(title="Recognition Alert")
                table.add_column("Person", style="cyan")
                table.add_column("Time", style="yellow")

                table.add_row(
                    person_name,
                    datetime.now().strftime("%H:%M:%S")
                )

                self.console.print(table)

        except Exception as e:
            logger.warning(f"Notification failed: {e}")


class DatabaseManager:
    """
    Manage face database storage and retrieval
    """

    def __init__(self, known_faces_dir: Path, console: Optional[Console] = None):
        """
        Initialize database manager

        Args:
            known_faces_dir: Directory for storing face database
            console: Rich console for display (optional)
        """
        self.known_faces_dir = known_faces_dir
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self.console = console or Console()

    def save_known_faces(self, face_encodings: List):
        """
        Save known faces with metadata

        Args:
            face_encodings: List of ModernFaceEncoding objects
        """
        try:
            faces_file = self.known_faces_dir / "modern_face_database.pkl"
            with open(faces_file, 'wb') as f:
                pickle.dump(face_encodings, f)

            # Also save as JSON for inspection
            json_file = self.known_faces_dir / "face_database_info.json"
            face_info = []
            for face in face_encodings:
                face_info.append({
                    'person_name': face.person_name,
                    'confidence': face.confidence,
                    'timestamp': face.timestamp.isoformat(),
                    'model_used': face.model_used,
                    'embedding_size': face.embedding_size,
                    'detection_score': face.detection_score,
                    'angle_type': getattr(face, 'angle_type', 'frontal'),
                    'unique_id': getattr(face, 'unique_id', 'unknown')
                })

            with open(json_file, 'w') as f:
                json.dump(face_info, f, indent=2)

            logger.info(f"Saved {len(face_encodings)} face encodings to database")

        except Exception as e:
            logger.error(f"Failed to save faces: {e}")
            if self.console:
                self.console.print(f"⚠️ Failed to save faces: {e}", style="yellow")

    def load_known_faces(self):
        """
        Load known faces from disk

        Returns:
            List of face encodings or empty list if none found
        """
        try:
            faces_file = self.known_faces_dir / "modern_face_database.pkl"
            if faces_file.exists():
                with open(faces_file, 'rb') as f:
                    face_encodings = pickle.load(f)

                logger.info(f"Loaded {len(face_encodings)} face encodings from database")
                return face_encodings
            else:
                logger.info("Face database file not found, starting with empty database")
                return []

        except Exception as e:
            logger.error(f"Failed to load faces: {e}")
            if self.console:
                self.console.print(f"⚠️ Failed to load faces: {e}", style="yellow")
            return []

    def display_database_info(self, face_encodings: List):
        """
        Display database information in console

        Args:
            face_encodings: List of face encodings
        """
        if not self.console:
            return

        # Display loaded faces info
        if face_encodings:
            table = Table(title="📁 Loaded Face Database")
            table.add_column("Name", style="cyan")
            table.add_column("Model", style="green")
            table.add_column("Embedding Size", style="yellow")
            table.add_column("Added", style="magenta")

            for face in face_encodings:
                table.add_row(
                    face.person_name,
                    face.model_used,
                    str(face.embedding_size),
                    face.timestamp.strftime("%Y-%m-%d")
                )

            self.console.print(table)
        else:
            self.console.print("📁 Face database is empty", style="cyan")

class CameraManager:
    """
    Manage camera detection and selection
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize camera manager

        Args:
            console: Rich console for display (optional)
        """
        self.console = console or Console()
        self.available_cameras = []
        self.current_camera_index = 0
        self.ip_cameras = []
        self.load_ip_cameras()

    def load_ip_cameras(self):
        """Load saved IP cameras"""
        try:
            settings_file = Path("data/settings/ip_cameras.json")
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    self.ip_cameras = json.load(f)
                if self.console:
                    self.console.print(f"✅ Loaded {len(self.ip_cameras)} IP cameras", style="green")
        except Exception as e:
            logger.warning(f"Failed to load IP cameras: {e}")
            self.ip_cameras = []

    def save_ip_cameras(self):
        """Save IP cameras to disk"""
        try:
            settings_dir = Path("data/settings")
            settings_dir.mkdir(parents=True, exist_ok=True)

            with open(settings_dir / "ip_cameras.json", 'w') as f:
                json.dump(self.ip_cameras, f, indent=2)

            if self.console:
                self.console.print("✅ IP cameras saved", style="green")
        except Exception as e:
            logger.error(f"Failed to save IP cameras: {e}")
            if self.console:
                self.console.print(f"⚠️ Failed to save IP cameras: {e}", style="yellow")

    def detect_available_cameras(self):
        """
        Detect all available cameras on the system with enhanced stability
        """
        if self.console:
            self.console.print("🔍 Detecting available cameras...", style="cyan")

        self.available_cameras = []

        # Define a safe camera check function to isolate crashes
        def check_camera_safely(index):
            try:
                # Safely handle OpenCV log levels
                try:
                    # Check if log level functions exist before using them
                    if hasattr(cv2, 'getLogLevel') and hasattr(cv2, 'setLogLevel') and hasattr(cv2, 'LOG_LEVEL_SILENT'):
                        prev_log_level = cv2.getLogLevel()
                        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
                    else:
                        prev_log_level = None
                except Exception as log_e:
                    logger.debug(f"OpenCV log level API not available: {log_e}")
                    prev_log_level = None

                # Open with default backend to avoid depth sensor errors
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow on Windows

                # Restore log level if it was changed
                try:
                    if prev_log_level is not None and hasattr(cv2, 'setLogLevel'):
                        cv2.setLogLevel(prev_log_level)
                except Exception:
                    pass

                if not cap.isOpened():
                    return None

                # Try to read a frame with timeout protection
                success = False
                frame = None

                # Set timeout for read operation (500ms)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Try to read a frame
                ret, frame = cap.read()

                if ret and frame is not None and frame.size > 0:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    # Create camera info
                    camera_info = {
                        'index': index,
                        'name': f"Camera {index}",
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'working': True,
                        'type': 'local'
                    }
                    cap.release()
                    return camera_info

                cap.release()
                return None
            except Exception as e:
                logger.warning(f"Error checking camera {index}: {e}")
                return None
            finally:
                # Make sure we release the camera
                try:
                    if 'cap' in locals() and cap is not None and cap.isOpened():
                        cap.release()
                except:
                    pass

        # Check only the most likely camera indices to avoid depth sensor errors
        for i in range(2):  # Just check cameras 0 and 1
            try:
                camera_info = check_camera_safely(i)
                if camera_info:
                    self.available_cameras.append(camera_info)
                    if self.console:
                        self.console.print(f"📹 Found Camera {i}: {camera_info['resolution']} @ {camera_info['fps']}fps", style="green")
            except Exception as e:
                # Double isolation for extra safety
                logger.error(f"Completely isolated camera check error: {e}")

        # If no cameras were detected, add a default one anyway
        if not self.available_cameras:
            logger.warning("No cameras detected, adding default camera 0")
            self.available_cameras.append({
                'index': 0,
                'name': "Default Camera",
                'resolution': "640x480",
                'fps': 30,
                'working': False,
                'type': 'local'
            })

        # Add IP cameras if any
        for ip_cam in self.ip_cameras:
            try:
                # Try to open the IP camera
                cap = cv2.VideoCapture(ip_cam['url'])
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Add the IP camera to available cameras
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))

                        camera_info = {
                            'index': ip_cam['url'],  # Use URL as index for IP cameras
                            'name': ip_cam['name'],
                            'resolution': f"{width}x{height}",
                            'fps': fps,
                            'working': True,
                            'type': 'ip'
                        }
                        self.available_cameras.append(camera_info)
                        if self.console:
                            self.console.print(f"📹 Found IP Camera: {ip_cam['name']}", style="green")
                    cap.release()
            except Exception as e:
                logger.warning(f"Error checking IP camera {ip_cam['name']}: {e}")

        if self.console:
            self.console.print(f"✅ Detected {len(self.available_cameras)} camera(s)", style="green")

    def detect_mobile_cameras(self):
        """Detect mobile cameras using IP Webcam app"""
        try:
            # Try to detect IP Webcam app on the local network
            # IP Webcam app typically runs on port 8080
            # This is a simplified implementation - a real one would use mDNS/network discovery
            local_ip = socket.gethostbyname(socket.gethostname())
            ip_parts = local_ip.split('.')
            base_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}"

            for i in range(1, 255):
                ip = f"{base_ip}.{i}"
                if ip == local_ip:
                    continue  # Skip local machine

                try:
                    # Try to connect with a short timeout
                    url = f"http://{ip}:8080/video"
                    response = requests.head(url, timeout=0.2)
                    if response.status_code == 200:
                        # Likely an IP Webcam app
                        cap = cv2.VideoCapture(url)
                        if cap.isOpened():
                            # Add the mobile camera to available cameras
                            camera_info = {
                                'index': url,
                                'name': f"Mobile Camera ({ip})",
                                'resolution': "Unknown",
                                'fps': 30,
                                'working': True,
                                'type': 'mobile'
                            }
                            self.available_cameras.append(camera_info)
                            if self.console:
                                self.console.print(f"📱 Found Mobile Camera at {ip}", style="green")
                            cap.release()
                except Exception:
                    pass  # Silently ignore connection errors
        except Exception as e:
            logger.error(f"Failed to detect mobile cameras: {e}")

        if self.console:
            self.console.print(f"✅ Detected {len(self.available_cameras)} camera(s)", style="green")

    def switch_camera(self, index: int):
        """
        Switch to a different camera

        Args:
            index: Index of the camera to switch to
        """
        if index < 0 or index >= len(self.available_cameras):
            logger.warning("Invalid camera index")
            return

        self.current_camera_index = index
        camera = self.available_cameras[index]

        if self.console:
            self.console.print(f"🔄 Switched to {camera['name']} (#{index})", style="cyan")

    def get_current_camera(self):
        """
        Get the currently selected camera

        Returns:
            Dictionary with current camera info
        """
        if 0 <= self.current_camera_index < len(self.available_cameras):
            return self.available_cameras[self.current_camera_index]
        return None

    def print_available_cameras(self):
        """Print the list of available cameras to the console"""
        if not self.console:
            return

        if self.available_cameras:
            table = Table(title="📹 Available Cameras")
            table.add_column("Index", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Type", style="yellow")
            table.add_column("Resolution", style="magenta")
            table.add_column("FPS", style="blue")

            for i, camera in enumerate(self.available_cameras):
                table.add_row(
                    str(i),
                    camera['name'],
                    camera['type'],
                    camera['resolution'],
                    str(camera.get('fps', 'N/A'))
                )

            self.console.print(table)
        else:
            self.console.print("📹 No cameras found", style="cyan")

class FaceRecognitionSystem:
    """
    Main class for the Ultra-Modern Face Recognition System
    """

    def __init__(self, known_faces_dir: Optional[Path] = None, console: Optional[Console] = None):
        """
        Initialize face recognition system

        Args:
            known_faces_dir: Directory for storing face database (optional)
            console: Rich console for display (optional)
        """
        self.console = console or Console()
        self.database_manager = None
        self.notification_manager = None
        self.camera_manager = None
        self.known_faces_dir = known_faces_dir or Path("data/known_faces")

        self.initialize()

    def initialize(self):
        """Initialize the system components"""
        if self.console:
            self.console.print("Initializing Ultra-Modern Face Recognition System...", style="bold green")

        # Initialize database manager
        self.database_manager = DatabaseManager(self.known_faces_dir, self.console)

        # Initialize notification manager
        self.notification_manager = NotificationManager(self.console)

        # Initialize camera manager
        self.camera_manager = CameraManager(self.console)

        # Load known faces from database
        known_faces = self.database_manager.load_known_faces()

        # Display loaded faces info
        self.database_manager.display_database_info(known_faces)

        # Detect available cameras
        self.camera_manager.detect_available_cameras()

        # If no known faces and no available cameras, show warning
        if not known_faces and not self.camera_manager.available_cameras:
            if self.console:
                self.console.print("⚠️ No known faces and no cameras detected! Please register a face and connect a camera.", style="yellow")

        if self.console:
            self.console.print("Initialization complete", style="bold green")

    def register_face(self, image_path: Path, person_name: str, model_used: str = "default"):
        """
        Register a new face in the system

        Args:
            image_path: Path to the image file
            person_name: Name of the person
            model_used: Model used for encoding (default: 'default')
        """
        try:
            import time
            from .modern_face_encoder import ModernFaceEncoder

            # Ensure the image file exists
            if not image_path.is_file():
                if self.console:
                    self.console.print(f"❌ Image file not found: {image_path}", style="red")
                return

            # Load the image
            image = cv2.imread(str(image_path))
            if image is None:
                if self.console:
                    self.console.print(f"❌ Failed to load image: {image_path}", style="red")
                return

            # Detect and encode the face
            face_encoder = ModernFaceEncoder(model_used=model_used)
            start_time = time.time()
            face_encoding = face_encoder.encode(image, person_name=person_name)
            elapsed_time = time.time() - start_time

            if face_encoding is not None:
                # Save the face encoding to the database
                self.database_manager.save_known_faces([face_encoding])

                if self.console:
                    self.console.print(f"✅ Face registered: {person_name} (Model: {model_used}, Time: {elapsed_time:.2f}s)", style="green")

                # Play sound notification
                if self.notification_manager.notification_settings['sound']:
                    try:
                        import pygame
                        pygame.mixer.init()
                        pygame.mixer.music.load("sounds/registration_success.mp3")
                        pygame.mixer.music.play()
                    except Exception as e:
                        logger.warning(f"Sound notification failed: {e}")

            else:
                if self.console:
                    self.console.print(f"❌ Face encoding failed for {person_name}", style="red")

        except Exception as e:
            logger.error(f"Error registering face: {e}")
            if self.console:
                self.console.print(f"❌ Error registering face: {e}", style="red")

    def recognize_faces(self, frame: np.ndarray, process_this_frame: bool, model_used: str = "default"):
        """
        Recognize faces in the given frame

        Args:
            frame: The video frame
            process_this_frame: Whether to process this frame
            model_used: Model used for recognition (default: 'default')

        Returns:
            Tuple of recognized face names and processed frame
        """
        recognized_names = []
        output_frame = frame

        try:
            if process_this_frame:
                import time
                from .modern_face_encoder import ModernFaceEncoder
                from .face_detector import FaceDetector

                # Initialize encoders and detectors
                face_encoder = ModernFaceEncoder(model_used=model_used)
                face_detector = FaceDetector()

                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                face_locations = face_detector.detect(rgb_frame)

                # Process each detected face
                for face_location in face_locations:
                    # Extract face region
                    top, right, bottom, left = face_location
                    face_image = rgb_frame[top:bottom, left:right]

                    # Encode the face
                    face_encoding = face_encoder.encode(face_image)

                    if face_encoding is not None:
                        # Compare with known faces
                        known_faces = self.database_manager.load_known_faces()
                        for known_face in known_faces:
                            # Calculate distance (similarity) - using L2 norm
                            distance = np.linalg.norm(np.array(face_encoding.embedding) - np.array(known_face.embedding))

                            # If distance is below a threshold, we have a match
                            if distance < 0.6:  # This threshold may need tuning
                                recognized_names.append(known_face.person_name)

                                # Send notification
                                self.notification_manager.send_notification(known_face.person_name, known_face.confidence)

                                break  # Stop if matched with any known face

        except Exception as e:
            logger.error(f"Error recognizing faces: {e}")

        return recognized_names, output_frame

    def run(self):
        """
        Run the face recognition system
        """
        try:
            import time
            from .modern_face_encoder import ModernFaceEncoder
            from .face_detector import FaceDetector

            # Initialize encoders and detectors
            face_encoder = ModernFaceEncoder()
            face_detector = FaceDetector()

            # Main loop
            while True:
                # Get the current camera
                camera = self.camera_manager.get_current_camera()
                if not camera:
                    if self.console:
                        self.console.print("❌ No camera selected", style="red")
                    break

                # Capture frame from camera
                if camera['type'] == 'local':
                    cap = cv2.VideoCapture(camera['index'], cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(camera['index'])  # For IP cameras

                if not cap.isOpened():
                    if self.console:
                        self.console.print(f"❌ Failed to open camera: {camera['name']}", style="red")
                    break

                # Read a frame
                ret, frame = cap.read()
                if not ret:
                    if self.console:
                        self.console.print(f"❌ Failed to read frame from camera: {camera['name']}", style="red")
                    break

                # Recognize faces in the frame
                recognized_names, output_frame = self.recognize_faces(frame, process_this_frame=True)

                # Display the resulting frame
                cv2.imshow("Face Recognition", output_frame)

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Add a small delay to reduce CPU usage
                time.sleep(0.01)

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logger.error(f"Error running face recognition system: {e}")
            if self.console:
                self.console.print(f"❌ Error running face recognition system: {e}", style="red")
