"""
Core face recognition module for Ultra-Modern Face Recognition System
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, List
from pathlib import Path
from src.face_recognition.utils.onnx_helper import onnx_helper

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """
    Face recognizer using 2025 SOTA SFace model

    Features:
    - SFace face recognition (OpenCV 2025 SOTA)
    - Advanced manual feature extraction fallback
    - Optimized for real-time processing with ONNX support
    """

    def __init__(self, models_dir, recognition_threshold=0.6, recognition_size=(112, 112)):
        """
        Initialize face recognizer with SFace or manual fallback methods

        Args:
            models_dir: Directory containing face recognition models
            recognition_threshold: Threshold for face matching (cosine similarity)
            recognition_size: Standard size for face recognition input
        """
        self.models_dir = models_dir
        self.recognition_threshold = recognition_threshold
        self.recognition_size = recognition_size
        self.face_recognizer = None
        self.using_cuda = False

        # Initialize recognizer
        self._initialize_recognizer()

    def _initialize_recognizer(self):
        """Initialize SFace recognizer based on availability"""
        sface_path = self.models_dir / "sface_recognition_2021dec.onnx"

        # Try to use the optimized model if it exists
        optimized_path = Path(self.models_dir) / "sface_recognition_2021dec_optimized.onnx"
        if optimized_path.exists():
            sface_path = optimized_path
            logger.info(f"Using optimized SFace model: {optimized_path}")
        elif onnx_helper.onnx_available:
            # Try to optimize the model
            optimized_model = onnx_helper.optimize_onnx_model(str(sface_path))
            if optimized_model:
                sface_path = Path(optimized_model)
                logger.info(f"Created and using optimized SFace model: {sface_path}")

        if sface_path.exists():
            try:
                # Check for OpenCV CUDA support
                self.using_cuda = onnx_helper.opencv_dnn_cuda

                # Create face recognizer with CUDA settings
                if self.using_cuda:
                    # Configure global OpenCV DNN settings for CUDA
                    cv2.setUseOptimized(True)
                    if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
                        cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                # Create the face recognizer
                self.face_recognizer = cv2.FaceRecognizerSF.create(
                    str(sface_path), ""
                )

                acceleration = "CUDA" if self.using_cuda else "CPU"
                logger.info(f"SFace recognizer initialized (2025 SOTA) using {acceleration} acceleration")
            except Exception as e:
                logger.error(f"Failed to initialize SFace: {e}")
                self.face_recognizer = None
        else:
            logger.warning(f"SFace model not found at {sface_path}")
            self.face_recognizer = None

    def extract_face_encoding(self, image: np.ndarray, face_data) -> Optional[np.ndarray]:
        """
        Extract face encoding using SFace or fallback to manual methods

        Args:
            image: Input image (BGR format)
            face_data: Face detection data

        Returns:
            Face encoding as numpy array
        """
        if self.face_recognizer is not None:
            return self.extract_face_encoding_sface(image, face_data)
        else:
            return self.extract_face_encoding_manual(image, face_data)

    def extract_face_encoding_sface(self, image: np.ndarray, face_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract face encoding using SFace (2025 SOTA)"""
        if self.face_recognizer is None:
            return self.extract_face_encoding_manual(image, face_data)

        try:
            # Extract face coordinates
            x, y, w, h = face_data[:4].astype(int)

            # Ensure coordinates are within image bounds
            h_img, w_img = image.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))

            # Extract and align face
            face_roi = image[y:y+h, x:x+w]
            if face_roi.size == 0:
                return None

            # Resize to standard recognition size
            face_aligned = cv2.resize(face_roi, self.recognition_size)

            # Extract feature using SFace
            feature = self.face_recognizer.feature(face_aligned)

            return feature.flatten()

        except Exception as e:
            logger.error(f"SFace encoding error: {e}")
            return self.extract_face_encoding_manual(image, face_data)

    def align_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Align face for consistent recognition using SFace alignment or basic resize

        Args:
            face_image: Input face image region (BGR format)

        Returns:
            Aligned face image ready for embedding extraction
        """
        try:
            # Validate input
            if face_image is None or face_image.size == 0:
                logger.warning("Empty face image provided for alignment")
                return None

            # Check face image dimensions
            if face_image.shape[0] < 10 or face_image.shape[1] < 10:
                logger.warning(f"Face image too small for alignment: {face_image.shape}")
                return None

            # Use SFace's built-in alignment if available
            if self.face_recognizer is not None:
                try:
                    # If SFace has alignment capability, use it (depends on OpenCV version)
                    if hasattr(self.face_recognizer, 'alignCrop'):
                        # Create a face_box parameter for alignCrop (required)
                        h, w = face_image.shape[:2]
                        face_box = np.array([0, 0, w, h])  # Use the entire image as face box
                        aligned_face = self.face_recognizer.alignCrop(face_image, face_box)
                        return aligned_face
                except Exception as e:
                    logger.warning(f"SFace alignment failed, falling back to basic resize: {e}")

            # Basic alignment fallback - resize to standard size
            aligned_face = cv2.resize(face_image, self.recognition_size,
                                     interpolation=cv2.INTER_AREA)

            return aligned_face

        except Exception as e:
            logger.error(f"Face alignment error: {e}")
            return None

    def extract_face_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding from an already aligned face image

        Args:
            aligned_face: Aligned face image (from align_face method)

        Returns:
            Face embedding as numpy array
        """
        if aligned_face is None or aligned_face.size == 0:
            return None

        try:
            if self.face_recognizer is not None:
                # Use SFace feature extraction
                feature = self.face_recognizer.feature(aligned_face)
                return feature.flatten()
            else:
                # Fallback to manual feature extraction
                return self._manual_feature_extraction(aligned_face)
        except Exception as e:
            logger.error(f"Face embedding extraction error: {e}")
            return None

    def extract_face_encoding_manual(self, image: np.ndarray, face_data) -> Optional[np.ndarray]:
        """Manual feature extraction using advanced computer vision"""
        if isinstance(face_data, np.ndarray) and len(face_data) >= 4:
            x, y, w, h = face_data[:4].astype(int)
        else:
            x, y, w, h = face_data

        # Extract face with padding
        padding = 20
        h_img, w_img = image.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        # Resize to standard size
        face_img = cv2.resize(face_img, self.recognition_size)

        # Convert to grayscale for feature extraction
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Advanced feature extraction using multiple descriptors
        features = []

        # 1. Local Binary Pattern (modern texture descriptor)
        lbp = self.calculate_advanced_lbp(gray_face)
        features.extend(lbp.flatten())

        # 2. Histogram of Oriented Gradients
        hog = self.calculate_hog_features(gray_face)
        features.extend(hog)

        # 3. SIFT keypoints (if available)
        try:
            sift = cv2.SIFT_create(nfeatures=50)
            kp, desc = sift.detectAndCompute(gray_face, None)
            if desc is not None:
                # Use mean descriptor as feature
                sift_feature = np.mean(desc, axis=0)
                features.extend(sift_feature)
        except:
            pass

        # Convert to numpy array and normalize
        encoding = np.array(features, dtype=np.float32)
        if len(encoding) > 0:
            encoding = encoding / (np.linalg.norm(encoding) + 1e-8)  # L2 normalization

        return encoding

    def calculate_advanced_lbp(self, image: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
        """Calculate uniform Local Binary Pattern"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        # Create uniform LBP lookup table
        uniform_patterns = self.get_uniform_patterns(n_points)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                code = 0

                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(j + radius * np.cos(angle))
                    y = int(i + radius * np.sin(angle))

                    if 0 <= x < w and 0 <= y < h:
                        if image[y, x] >= center:
                            code |= (1 << p)

                # Map to uniform pattern
                lbp[i, j] = uniform_patterns.get(code, n_points + 1)

        # Calculate histogram
        hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 1))
        return hist

    def get_uniform_patterns(self, n_points: int) -> Dict[int, int]:
        """Get uniform patterns for LBP"""
        patterns = {}
        pattern_id = 0

        for i in range(2 ** n_points):
            # Count transitions
            binary = format(i, f'0{n_points}b')
            transitions = 0
            for j in range(n_points):
                if binary[j] != binary[(j + 1) % n_points]:
                    transitions += 1

            # Uniform patterns have at most 2 transitions
            if transitions <= 2:
                patterns[i] = pattern_id
                pattern_id += 1

        return patterns

    def calculate_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Calculate Histogram of Oriented Gradients"""
        # Parameters for HOG
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9

        # Resize image to fit HOG window
        resized = cv2.resize(image, win_size)

        # Create HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

        # Compute HOG features
        features = hog.compute(resized)

        return features.flatten()

    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Compare two face encodings using cosine similarity

        Args:
            encoding1: First face encoding
            encoding2: Second face encoding

        Returns:
            Similarity score (0-1, higher means more similar)
        """
        if len(encoding1) != len(encoding2):
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(encoding1, encoding2) / (
            np.linalg.norm(encoding1) * np.linalg.norm(encoding2) + 1e-8
        )

        return max(0.0, min(1.0, similarity))  # Bound between 0 and 1

    def find_best_match(self, encoding: np.ndarray, known_encodings: List) -> tuple:
        """
        Find best matching face from known encodings

        Args:
            encoding: Query face encoding
            known_encodings: List of known face encodings

        Returns:
            Tuple of (best_match_index, similarity_score)
            If no match found, returns (-1, 0.0)
        """
        best_match = -1
        best_similarity = -1

        for i, known_face in enumerate(known_encodings):
            if len(known_face.encoding) != len(encoding):
                continue  # Skip if encoding sizes don't match

            similarity = self.compare_faces(encoding, known_face.encoding)

            if similarity > best_similarity and similarity > self.recognition_threshold:
                best_similarity = similarity
                best_match = i

        return best_match, best_similarity

    def calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings

        Args:
            encoding1: First face embedding
            encoding2: Second face embedding

        Returns:
            Similarity score between 0 and 1 (higher is more similar)
        """
        try:
            if encoding1 is None or encoding2 is None:
                return 0.0

            if len(encoding1) != len(encoding2):
                logger.warning(f"Embedding dimensions don't match: {len(encoding1)} vs {len(encoding2)}")
                return 0.0

            # Normalize embeddings if they aren't already
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(encoding1, encoding2) / (norm1 * norm2)

            # Ensure result is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))

            return similarity
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
