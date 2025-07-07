"""
Module for storing and managing face embeddings
"""

import cv2
import numpy as np
import logging
import pickle
from pathlib import Path
from datetime import datetime

from src.face_recognition.core.face_detector import FaceDetector
from src.face_recognition.core.face_recognizer import FaceRecognizer
from src.face_recognition.models.face_models import ModernFaceEncoding

logger = logging.getLogger(__name__)

class FaceStorage:
    """
    Handles storage and management of face embeddings
    """

    def __init__(self,
                face_detector: FaceDetector,
                face_recognizer: FaceRecognizer,
                database_path: Path):
        """
        Initialize face storage

        Args:
            face_detector: Initialized face detector
            face_recognizer: Initialized face recognizer
            database_path: Path to save face database
        """
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.database_path = database_path

        # Internal variables
        self._is_active = False
        self._current_embedding = None
        self._current_person_name = ""
        self._status_message = ""

    def start_recording(self, person_name: str) -> bool:
        """
        Start recording process for a person (compatibility method)

        Args:
            person_name: Name of the person

        Returns:
            True if started successfully
        """
        if self._is_active:
            self._status_message = "Already processing a face"
            return False

        self._is_active = True
        self._current_embedding = None
        self._current_person_name = person_name
        self._status_message = f"Processing face for {person_name}"
        logger.info(f"Processing face for {person_name}")

        return True

    def stop_recording(self) -> bool:
        """
        Stop recording process (compatibility method)

        Returns:
            True if stopped successfully
        """
        if not self._is_active:
            return False

        self._is_active = False
        self._status_message = "Processing stopped"
        logger.info("Face processing stopped")

        return True

    def is_recording(self) -> bool:
        """Check if processing is active (compatibility method)"""
        return self._is_active

    def get_status(self) -> str:
        """Get current status message"""
        return self._status_message

    def get_embedding_count(self) -> int:
        """Get number of embeddings collected (compatibility method)"""
        return 1 if self._current_embedding is not None else 0

    def add_face(self, face_image: np.ndarray) -> bool:
        """
        Add a face from an image

        Args:
            face_image: Image containing a face

        Returns:
            True if face was added successfully
        """
        if not self._is_active:
            return False

        try:
            # Align face
            aligned_face = self.face_recognizer.align_face(face_image)
            if aligned_face is None:
                return False

            # Extract embedding
            embedding = self.face_recognizer.extract_face_embedding(aligned_face)
            if embedding is None:
                return False

            # Store embedding
            self._current_embedding = embedding
            return True

        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False

    def process_frame(self, frame: np.ndarray):
        """
        Process a video frame (compatibility method)

        Args:
            frame: Video frame to process

        Returns:
            Tuple of (processed_frame, is_complete)
        """
        if not self._is_active or frame is None:
            return frame, False

        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)

            # Draw on frame for visualization
            if len(faces) == 1:
                face = faces[0]

                # Extract coordinates
                if isinstance(face, np.ndarray) and len(face) >= 4:
                    x, y, w, h = map(int, face[:4])
                elif isinstance(face, dict) and 'box' in face:
                    x, y, w, h = face['box']
                else:
                    return frame, False

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Extract face region and get embedding
                face_img = frame[y:y+h, x:x+w]

                # Process if we don't have an embedding yet
                if self._current_embedding is None:
                    # Add the face
                    if self.add_face(face_img):
                        # Draw text to show success
                        cv2.putText(frame, "Face captured!", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        # Return complete
                        return frame, True

            # Draw text if no single face found
            if len(faces) != 1:
                cv2.putText(frame, "Position one face in frame", (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Not complete yet
            return frame, False

        except Exception as e:
            logger.error(f"Error in processing frame: {e}")
            return frame, False

    def save_embeddings(self) -> bool:
        """
        Save current face embedding to database

        Returns:
            True if saved successfully
        """
        if not self._current_embedding or not self._current_person_name:
            self._status_message = "No face to save"
            return False

        try:
            # Create database directory if it doesn't exist
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing database if exists
            database = {}
            if self.database_path.exists():
                with open(self.database_path, 'rb') as f:
                    database = pickle.load(f)

            # Create face encoding object
            database[self._current_person_name] = ModernFaceEncoding(
                encoding=self._current_embedding,
                person_name=self._current_person_name,
                confidence=1.0,
                timestamp=datetime.now(),
                model_used="standard",
                embedding_size=len(self._current_embedding),
                detection_score=0.9,
                angle_type="frontal"
            )

            # Save database
            with open(self.database_path, 'wb') as f:
                pickle.dump(database, f)

            self._status_message = f"Saved face for {self._current_person_name}"
            logger.info(f"Saved face embedding for {self._current_person_name}")
            return True

        except Exception as e:
            self._status_message = f"Error saving face: {str(e)}"
            logger.error(f"Error saving face: {e}")
            return False

    # Compatibility methods for system.py
    def set_progress_callback(self, callback):
        """Compatibility method - not used"""
        pass
