"""
Advanced Face Recognition Module
================================

High-performance face recognition using multiple algorithms including
face_recognition library, OpenCV, and custom neural network models.
"""

import cv2
import numpy as np
import face_recognition
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import os
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from .config import Config, get_config


class FaceRecognizer:
    """
    Advanced face recognition system supporting multiple recognition algorithms.
    
    Features:
    - face_recognition library (dlib-based)
    - Custom SVM classifier
    - Cosine similarity matching
    - Multiple face encoding methods
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the face recognizer."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Face encodings database
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.face_encoding_model = self.config.detection.recognition_model
        
        # SVM classifier for recognition
        self.svm_classifier: Optional[SVC] = None
        self.use_svm = False
        
        # Performance tracking
        self.recognition_count = 0
        self.total_recognition_time = 0.0
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self) -> None:
        """Load known faces from the database directory."""
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        
        known_faces_dir = self.config.known_faces_dir
        if not known_faces_dir.exists():
            known_faces_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning("Known faces directory created. Please add face images.")
            return
        
        # Load encodings from pickle file if exists
        encodings_file = self.config.models_dir / "face_encodings.pkl"
        if encodings_file.exists():
            try:
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    self.logger.info(f"Loaded {len(self.known_face_names)} known faces from cache")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to load cached encodings: {e}")
        
        # Process face images from directories
        self._process_face_images()
        
        # Save encodings to cache
        self._save_face_encodings()
        
        # Train SVM classifier if we have enough faces
        if len(set(self.known_face_names)) >= 2:
            self._train_svm_classifier()
    
    def _process_face_images(self) -> None:
        """Process face images from the known faces directory."""
        known_faces_dir = self.config.known_faces_dir
        
        for person_dir in known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            self.logger.info(f"Processing images for {person_name}")
            
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(person_dir.glob(ext))
                image_files.extend(person_dir.glob(ext.upper()))
            
            for image_path in image_files:
                try:
                    encodings = self._extract_face_encodings(str(image_path))
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(person_name)
                    
                    if encodings:
                        self.logger.debug(f"Processed {image_path.name}: {len(encodings)} faces")
                    else:
                        self.logger.warning(f"No faces found in {image_path.name}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
        
        self.logger.info(f"Total loaded: {len(self.known_face_encodings)} face encodings for {len(set(self.known_face_names))} people")
    
    def _extract_face_encodings(self, image_path: str) -> List[np.ndarray]:
        """Extract face encodings from an image file."""
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(
                image, 
                model=self.config.detection.detection_model
            )
            
            if not face_locations:
                return []
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                image, 
                face_locations, 
                model=self.face_encoding_model
            )
            
            return face_encodings
            
        except Exception as e:
            self.logger.error(f"Failed to extract encodings from {image_path}: {e}")
            return []
    
    def _save_face_encodings(self) -> None:
        """Save face encodings to cache file."""
        try:
            encodings_file = self.config.models_dir / "face_encodings.pkl"
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            
            with open(encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Saved face encodings to {encodings_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save face encodings: {e}")
    
    def _train_svm_classifier(self) -> None:
        """Train SVM classifier for face recognition."""
        try:
            if len(self.known_face_encodings) < 2:
                return
            
            # Prepare training data
            X = np.array(self.known_face_encodings)
            y = np.array(self.known_face_names)
            
            # Train SVM classifier
            self.svm_classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            
            self.svm_classifier.fit(X, y)
            self.use_svm = True
            
            # Save the classifier
            classifier_file = self.config.models_dir / "svm_classifier.pkl"
            joblib.dump(self.svm_classifier, classifier_file)
            
            self.logger.info("SVM classifier trained and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to train SVM classifier: {e}")
            self.use_svm = False
    
    def recognize_faces_basic(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Basic face recognition using face_recognition library."""
        if not self.known_face_encodings:
            return [("Unknown", 0.0) for _ in face_encodings]
        
        results = []
        
        for encoding in face_encodings:
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]
                confidence = 1.0 - min_distance
                
                if confidence >= self.config.detection.confidence_threshold:
                    name = self.known_face_names[best_match_index]
                    results.append((name, confidence))
                else:
                    results.append(("Unknown", confidence))
            else:
                results.append(("Unknown", 0.0))
        
        return results
    
    def recognize_faces_svm(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Face recognition using SVM classifier."""
        if not self.use_svm or self.svm_classifier is None:
            return self.recognize_faces_basic(face_encodings)
        
        results = []
        
        for encoding in face_encodings:
            try:
                # Predict using SVM
                encoding_2d = encoding.reshape(1, -1)
                prediction = self.svm_classifier.predict(encoding_2d)[0]
                probabilities = self.svm_classifier.predict_proba(encoding_2d)[0]
                
                # Get confidence (max probability)
                max_prob_index = np.argmax(probabilities)
                confidence = probabilities[max_prob_index]
                
                if confidence >= self.config.detection.confidence_threshold:
                    results.append((prediction, confidence))
                else:
                    results.append(("Unknown", confidence))
                    
            except Exception as e:
                self.logger.error(f"SVM recognition failed: {e}")
                results.append(("Unknown", 0.0))
        
        return results
    
    def recognize_faces_cosine(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Face recognition using cosine similarity."""
        if not self.known_face_encodings:
            return [("Unknown", 0.0) for _ in face_encodings]
        
        results = []
        known_encodings_array = np.array(self.known_face_encodings)
        
        for encoding in face_encodings:
            # Calculate cosine similarities
            similarities = cosine_similarity([encoding], known_encodings_array)[0]
            
            if len(similarities) > 0:
                best_match_index = np.argmax(similarities)
                max_similarity = similarities[best_match_index]
                
                # Convert similarity to confidence (cosine similarity is between -1 and 1)
                confidence = (max_similarity + 1) / 2
                
                if confidence >= self.config.detection.confidence_threshold:
                    name = self.known_face_names[best_match_index]
                    results.append((name, confidence))
                else:
                    results.append(("Unknown", confidence))
            else:
                results.append(("Unknown", 0.0))
        
        return results
    
    def recognize_faces(self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                       method: str = "basic") -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the given image.
        
        Args:
            image: Input image as numpy array
            face_locations: List of face bounding boxes as (x, y, width, height)
            method: Recognition method ('basic', 'svm', 'cosine')
        
        Returns:
            List of (name, confidence, (x, y, w, h)) tuples
        """
        start_time = cv2.getTickCount()
        
        try:
            # Convert face locations to face_recognition format (top, right, bottom, left)
            face_locations_rgb = []
            for (x, y, w, h) in face_locations:
                face_locations_rgb.append((y, x + w, y + h, x))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations_rgb, 
                model=self.face_encoding_model
            )
            
            # Recognize faces using specified method
            if method == "svm":
                recognition_results = self.recognize_faces_svm(face_encodings)
            elif method == "cosine":
                recognition_results = self.recognize_faces_cosine(face_encodings)
            else:  # basic
                recognition_results = self.recognize_faces_basic(face_encodings)
            
            # Combine results with face locations
            results = []
            for i, (name, confidence) in enumerate(recognition_results):
                if i < len(face_locations):
                    results.append((name, confidence, face_locations[i]))
            
            # Update performance tracking
            end_time = cv2.getTickCount()
            recognition_time = (end_time - start_time) / cv2.getTickFrequency()
            self.recognition_count += 1
            self.total_recognition_time += recognition_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return [("Unknown", 0.0, loc) for loc in face_locations]
    
    def add_face(self, name: str, image_path: str) -> bool:
        """Add a new face to the known faces database."""
        try:
            # Extract encodings from the image
            encodings = self._extract_face_encodings(image_path)
            
            if not encodings:
                self.logger.error(f"No faces found in {image_path}")
                return False
            
            # Add to known faces
            for encoding in encodings:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)
            
            # Save updated encodings
            self._save_face_encodings()
            
            # Retrain SVM if needed
            if len(set(self.known_face_names)) >= 2:
                self._train_svm_classifier()
            
            self.logger.info(f"Added {len(encodings)} face encodings for {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add face for {name}: {e}")
            return False
    
    def remove_face(self, name: str) -> bool:
        """Remove all faces for a person from the database."""
        try:
            # Remove all encodings for the person
            indices_to_remove = [i for i, face_name in enumerate(self.known_face_names) if face_name == name]
            
            if not indices_to_remove:
                self.logger.warning(f"No faces found for {name}")
                return False
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                self.known_face_encodings.pop(i)
                self.known_face_names.pop(i)
            
            # Save updated encodings
            self._save_face_encodings()
            
            # Retrain SVM if needed
            if len(set(self.known_face_names)) >= 2:
                self._train_svm_classifier()
            
            self.logger.info(f"Removed {len(indices_to_remove)} face encodings for {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove face for {name}: {e}")
            return False
    
    def get_known_people(self) -> List[str]:
        """Get list of known people."""
        return list(set(self.known_face_names))
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        avg_time = self.total_recognition_time / max(1, self.recognition_count)
        return {
            'total_recognitions': self.recognition_count,
            'total_time': self.total_recognition_time,
            'average_time_per_recognition': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0,
            'known_people': len(set(self.known_face_names)),
            'total_face_encodings': len(self.known_face_encodings),
            'svm_enabled': self.use_svm
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.recognition_count = 0
        self.total_recognition_time = 0.0
