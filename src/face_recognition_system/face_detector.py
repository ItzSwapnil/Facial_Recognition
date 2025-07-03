"""
Advanced Face Detection Module
==============================

Supports multiple detection backends including OpenCV Haar cascades,
OpenCV DNN, and dlib HOG/CNN detectors for optimal performance.
"""

import cv2
import dlib
import numpy as np
from typing import List, Tuple, Optional, Union
import face_recognition
import logging
from pathlib import Path

from .config import Config, get_config


class FaceDetector:
    """
    Advanced face detector supporting multiple detection algorithms.
    
    Supported models:
    - OpenCV Haar Cascades (fast, moderate accuracy)
    - OpenCV DNN (balanced speed/accuracy)
    - dlib HOG (good accuracy, CPU optimized)
    - dlib CNN (best accuracy, GPU recommended)
    - face_recognition library (high accuracy)
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the face detector with specified configuration."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection models
        self._init_opencv_detector()
        self._init_dlib_detector()
        
        # Performance tracking
        self.detection_count = 0
        self.total_detection_time = 0.0
        
    def _init_opencv_detector(self) -> None:
        """Initialize OpenCV face detector."""
        try:
            # Load Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Load DNN model for face detection
            try:
                # Download OpenCV DNN face detection model if not exists
                model_dir = self.config.models_dir
                model_dir.mkdir(parents=True, exist_ok=True)
                
                prototxt_path = model_dir / "deploy.prototxt"
                weights_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
                
                if not prototxt_path.exists() or not weights_path.exists():
                    self._download_opencv_dnn_model(model_dir)
                
                self.dnn_net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(weights_path))
                self.dnn_available = True
                
            except Exception as e:
                self.logger.warning(f"Failed to load OpenCV DNN model: {e}")
                self.dnn_available = False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenCV detector: {e}")
            raise
    
    def _init_dlib_detector(self) -> None:
        """Initialize dlib face detectors."""
        try:
            # HOG detector (CPU optimized)
            self.hog_detector = dlib.get_frontal_face_detector()
            
            # CNN detector (more accurate, GPU recommended)
            try:
                # Download dlib CNN model if not exists
                model_path = self.config.models_dir / "mmod_human_face_detector.dat"
                if not model_path.exists():
                    self._download_dlib_cnn_model(model_path)
                
                self.cnn_detector = dlib.cnn_face_detection_model_v1(str(model_path))
                self.cnn_available = True
                
            except Exception as e:
                self.logger.warning(f"Failed to load dlib CNN model: {e}")
                self.cnn_available = False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize dlib detector: {e}")
            raise
    
    def _download_opencv_dnn_model(self, model_dir: Path) -> None:
        """Download OpenCV DNN face detection model."""
        import urllib.request
        
        base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
        
        # Download prototxt
        prototxt_url = base_url + "deploy.prototxt"
        prototxt_path = model_dir / "deploy.prototxt"
        
        # Download weights
        weights_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        weights_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        try:
            self.logger.info("Downloading OpenCV DNN face detection model...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            urllib.request.urlretrieve(weights_url, weights_path)
            self.logger.info("OpenCV DNN model downloaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to download OpenCV DNN model: {e}")
            raise
    
    def _download_dlib_cnn_model(self, model_path: Path) -> None:
        """Download dlib CNN face detection model."""
        import urllib.request
        import bz2
        
        url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
        compressed_path = model_path.with_suffix('.dat.bz2')
        
        try:
            self.logger.info("Downloading dlib CNN face detection model...")
            urllib.request.urlretrieve(url, compressed_path)
            
            # Decompress the file
            with bz2.BZ2File(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            # Remove compressed file
            compressed_path.unlink()
            self.logger.info("dlib CNN model downloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to download dlib CNN model: {e}")
            raise
    
    def detect_faces_opencv_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.detection.scale_factor,
            minNeighbors=self.config.detection.min_neighbors,
            minSize=self.config.detection.min_face_size,
            maxSize=self.config.detection.max_face_size
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_faces_opencv_dnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN."""
        if not self.dnn_available:
            return []
        
        h, w = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.config.detection.confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def detect_faces_dlib_hog(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib HOG detector."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.hog_detector(gray)
        return [(face.left(), face.top(), face.width(), face.height()) for face in faces]
    
    def detect_faces_dlib_cnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib CNN detector."""
        if not self.cnn_available:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        faces = self.cnn_detector(gray)
        return [(face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()) for face in faces]
    
    def detect_faces_face_recognition(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_image, 
            model=self.config.detection.detection_model
        )
        
        # Convert from (top, right, bottom, left) to (x, y, w, h)
        faces = []
        for (top, right, bottom, left) in face_locations:
            faces.append((left, top, right - left, bottom - top))
        
        return faces
    
    def detect_faces(self, image: np.ndarray, method: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using the specified or configured method.
        
        Args:
            image: Input image as numpy array
            method: Detection method ('haar', 'dnn', 'hog', 'cnn', 'face_recognition')
                   If None, uses config.detection.detection_model
        
        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        if method is None:
            method = self.config.detection.detection_model
        
        start_time = cv2.getTickCount()
        
        try:
            if method == 'haar':
                faces = self.detect_faces_opencv_haar(image)
            elif method == 'dnn':
                faces = self.detect_faces_opencv_dnn(image)
            elif method == 'hog':
                faces = self.detect_faces_dlib_hog(image)
            elif method == 'cnn':
                faces = self.detect_faces_dlib_cnn(image)
            elif method == 'face_recognition':
                faces = self.detect_faces_face_recognition(image)
            else:
                self.logger.warning(f"Unknown detection method: {method}, using 'hog'")
                faces = self.detect_faces_dlib_hog(image)
            
            # Update performance tracking
            end_time = cv2.getTickCount()
            detection_time = (end_time - start_time) / cv2.getTickFrequency()
            self.detection_count += 1
            self.total_detection_time += detection_time
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed with method {method}: {e}")
            return []
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        avg_time = self.total_detection_time / max(1, self.detection_count)
        return {
            'total_detections': self.detection_count,
            'total_time': self.total_detection_time,
            'average_time_per_detection': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.detection_count = 0
        self.total_detection_time = 0.0
