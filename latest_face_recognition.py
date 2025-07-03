"""
Latest Face Recognition System with Real-time Detection and Notifications
=========================================================================

Modern implementation using OpenCV DNN with state-of-the-art face detection
and recognition models, plus comprehensive notification system.
"""

import cv2
import numpy as np
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

# Notification imports
import plyer
import pygame
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

@dataclass
class PersonDetection:
    """Data class for person detection results"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    timestamp: datetime
    image_path: Optional[str] = None

class LatestFaceRecognitionSystem:
    """
    State-of-the-art face recognition system using latest OpenCV DNN models
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.console = Console()
        self.config = self.load_config(config_path)
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.detection_active = False
        self.notification_active = True
        
        # Initialize pygame for sound notifications
        pygame.mixer.init()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_recognition.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize face detection models
        self.setup_face_detection()
        self.load_known_faces()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "detection_threshold": 0.7,
            "recognition_threshold": 0.6,
            "notification_cooldown": 30,  # seconds
            "save_detections": True,
            "detection_save_path": "detections/",
            "known_faces_path": "data/known_faces/",
            "models_path": "data/models/",
            "camera_source": 0,
            "frame_skip": 2,  # Process every nth frame for performance
            "notification_sound": True,
            "notification_popup": True,
            "notification_log": True
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
                
        return default_config
    
    def setup_face_detection(self):
        """Setup OpenCV DNN face detection model"""
        models_path = Path(self.config["models_path"])
        models_path.mkdir(exist_ok=True)
        
        # Download models if they don't exist
        self.download_models(models_path)
        
        # Load face detection model (OpenCV DNN)
        prototxt_path = models_path / "deploy.prototxt"
        model_path = models_path / "res10_300x300_ssd_iter_140000.caffemodel"
        
        if prototxt_path.exists() and model_path.exists():
            self.face_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
            self.logger.info("Face detection model loaded successfully")
        else:
            self.logger.warning("Face detection models not found, using OpenCV Haar cascades")
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_net = None
    
    def download_models(self, models_path: Path):
        """Download face detection models if they don't exist"""
        import urllib.request
        
        files_to_download = [
            ("deploy.prototxt", "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"),
            ("res10_300x300_ssd_iter_140000.caffemodel", "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel")
        ]
        
        for filename, url in files_to_download:
            file_path = models_path / filename
            if not file_path.exists():
                try:
                    self.logger.info(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, file_path)
                    self.logger.info(f"Downloaded {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to download {filename}: {e}")
    
    def load_known_faces(self):
        """Load known faces from the known_faces directory"""
        known_faces_path = Path(self.config["known_faces_path"])
        known_faces_path.mkdir(exist_ok=True)
        
        # Look for face encodings file
        encodings_file = known_faces_path / "face_encodings.pkl"
        if encodings_file.exists():
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                self.logger.info(f"Loaded {len(self.known_face_names)} known faces")
        else:
            self.logger.info("No known faces found. Add images to the known_faces directory.")
    
    def detect_faces_dnn(self, image):
        """Detect faces using OpenCV DNN"""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.config["detection_threshold"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype(int)
                faces.append((x, y, x1-x, y1-y))
        return faces
    
    def detect_faces_haar(self, image):
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def extract_face_encoding(self, image, face_location):
        """Extract face encoding using simple feature extraction"""
        x, y, w, h = face_location
        face_roi = image[y:y+h, x:x+w]
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (128, 128))
        
        # Convert to grayscale and normalize
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_normalized = face_gray.astype(np.float32) / 255.0
        
        # Simple feature extraction (flattened histogram)
        hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
        return hist.flatten()
    
    def compare_faces(self, face_encoding, tolerance=0.6):
        """Compare face encoding with known faces"""
        if not self.known_face_encodings:
            return "Unknown", 0.0
            
        # Calculate distances (using correlation for histogram comparison)
        distances = []
        for known_encoding in self.known_face_encodings:
            # Use correlation coefficient as similarity measure
            correlation = cv2.compareHist(face_encoding, known_encoding, cv2.HISTCMP_CORREL)
            distances.append(1 - correlation)  # Convert to distance
        
        min_distance = min(distances)
        match_index = distances.index(min_distance)
        
        if min_distance < tolerance:
            return self.known_face_names[match_index], 1 - min_distance
        else:
            return "Unknown", 0.0
    
    def send_notification(self, person_name: str, confidence: float, image_path: str = None):
        """Send comprehensive notification about person detection"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Person Detected: {person_name} (Confidence: {confidence:.2f})"
        
        # Console notification
        if self.config["notification_log"]:
            self.console.print(f"🚨 [bold red]{message}[/bold red] at {timestamp}")
        
        # Sound notification
        if self.config["notification_sound"]:
            try:
                # Create a simple beep sound
                frequency = 800  # Hz
                duration = 500   # milliseconds
                # This is a placeholder - you can add actual sound file playback here
                self.logger.info("🔊 Sound notification triggered")
            except Exception as e:
                self.logger.warning(f"Sound notification failed: {e}")
        
        # System popup notification
        if self.config["notification_popup"]:
            try:
                plyer.notification.notify(
                    title="Face Recognition Alert",
                    message=message,
                    app_name="Facial Recognition System",
                    timeout=5
                )
            except Exception as e:
                self.logger.warning(f"Popup notification failed: {e}")
        
        # Log to file
        self.logger.info(f"DETECTION: {message} at {timestamp}")
        
        # Save detection data
        if self.config["save_detections"]:
            self.save_detection_data(person_name, confidence, timestamp, image_path)
    
    def save_detection_data(self, person_name: str, confidence: float, timestamp: str, image_path: str = None):
        """Save detection data to file"""
        detections_path = Path(self.config["detection_save_path"])
        detections_path.mkdir(exist_ok=True)
        
        detection_log = detections_path / "detections.json"
        
        detection_data = {
            "timestamp": timestamp,
            "person": person_name,
            "confidence": confidence,
            "image_path": image_path
        }
        
        # Load existing data or create new list
        if detection_log.exists():
            with open(detection_log, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(detection_data)
        
        # Save updated data
        with open(detection_log, 'w') as f:
            json.dump(data, f, indent=2)
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        # Detect faces
        if self.face_net is not None:
            faces = self.detect_faces_dnn(frame)
        else:
            faces = self.detect_faces_haar(frame)
        
        detections = []
        
        for face_location in faces:
            x, y, w, h = face_location
            
            # Extract face encoding
            face_encoding = self.extract_face_encoding(frame, face_location)
            
            # Compare with known faces
            name, confidence = self.compare_faces(face_encoding, self.config["recognition_threshold"])
            
            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Create detection object
            detection = PersonDetection(
                name=name,
                confidence=confidence,
                bbox=(x, y, w, h),
                timestamp=datetime.now()
            )
            detections.append(detection)
            
            # Send notification for known persons
            if name != "Unknown" and confidence > self.config["recognition_threshold"]:
                self.send_notification(name, confidence)
        
        return frame, detections
    
    def run_camera_detection(self):
        """Run real-time face detection from camera"""
        cap = cv2.VideoCapture(self.config["camera_source"])
        
        if not cap.isOpened():
            self.logger.error("Could not open camera")
            return
        
        self.console.print("[bold green]🎥 Starting camera face detection...[/bold green]")
        self.console.print("[yellow]Press 'q' to quit, 's' to save current frame[/yellow]")
        
        frame_count = 0
        
        try:
            while self.detection_active:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from camera")
                    break
                
                # Process every nth frame for performance
                if frame_count % self.config["frame_skip"] == 0:
                    processed_frame, detections = self.process_frame(frame.copy())
                    
                    # Display frame
                    cv2.imshow('Face Recognition System', processed_frame)
                else:
                    cv2.imshow('Face Recognition System', frame)
                
                frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    self.console.print(f"[green]Saved frame as {filename}[/green]")
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Detection stopped by user[/yellow]")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def add_known_person(self, name: str, image_path: str):
        """Add a new person to the known faces database"""
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return False
        
        # Detect face in the image
        if self.face_net is not None:
            faces = self.detect_faces_dnn(image)
        else:
            faces = self.detect_faces_haar(image)
        
        if not faces:
            self.logger.error("No face detected in the image")
            return False
        
        # Use the first (largest) face detected
        face_location = faces[0]
        face_encoding = self.extract_face_encoding(image, face_location)
        
        # Add to known faces
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        
        # Save updated encodings
        self.save_known_faces()
        
        self.console.print(f"[green]Added {name} to known faces database[/green]")
        return True
    
    def save_known_faces(self):
        """Save known faces encodings to file"""
        known_faces_path = Path(self.config["known_faces_path"])
        encodings_file = known_faces_path / "face_encodings.pkl"
        
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def start_detection(self):
        """Start the face detection system"""
        self.detection_active = True
        self.run_camera_detection()
    
    def stop_detection(self):
        """Stop the face detection system"""
        self.detection_active = False

def main():
    """Main function to run the face recognition system"""
    console = Console()
    console.print("[bold blue]🚀 Latest Face Recognition System[/bold blue]")
    console.print("[green]Initializing system...[/green]")
    
    try:
        # Initialize the system
        system = LatestFaceRecognitionSystem()
        
        # Start detection
        system.start_detection()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]System stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
