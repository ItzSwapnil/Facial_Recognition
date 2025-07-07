"""
Core face detection module for Ultra-Modern Face Recognition System
"""

import cv2
import numpy as np
from typing import List, Tuple
import logging
from pathlib import Path
from src.face_recognition.utils.onnx_helper import onnx_helper

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detector using 2025 SOTA YuNet model

    Features:
    - YuNet face detection (OpenCV 2025 SOTA)
    - Fallback to traditional detection methods
    - Optimized for real-time processing
    """

    def __init__(self, models_dir, input_size=(320, 240), detection_confidence=0.8, nms_threshold=0.3):
        """
        Initialize face detector with YuNet or fallback detector

        Args:
            models_dir: Directory containing face detection models
            input_size: Default size for detection input
            detection_confidence: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.models_dir = models_dir
        self.input_size = input_size
        self.detection_confidence = detection_confidence
        self.nms_threshold = nms_threshold
        self.yunet_available = False
        self.using_cuda = False

        # Initialize detector
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize YuNet or fallback detector based on availability"""
        yunet_path = self.models_dir / "yunet_face_detection_2023mar.onnx"

        # Try to use the optimized model if it exists
        optimized_path = Path(self.models_dir) / "yunet_face_detection_2023mar_optimized.onnx"
        if optimized_path.exists():
            yunet_path = optimized_path
            logger.info(f"Using optimized YuNet model: {optimized_path}")
        elif onnx_helper.onnx_available:
            # Try to optimize the model
            optimized_model = onnx_helper.optimize_onnx_model(str(yunet_path))
            if optimized_model:
                yunet_path = Path(optimized_model)
                logger.info(f"Created and using optimized YuNet model: {yunet_path}")

        if yunet_path.exists():
            try:
                # Check for OpenCV CUDA support
                self.using_cuda = onnx_helper.opencv_dnn_cuda

                # Create face detector with CUDA backend if available
                self.face_detector = cv2.FaceDetectorYN.create(
                    str(yunet_path),
                    "",
                    self.input_size,
                    score_threshold=self.detection_confidence,
                    nms_threshold=self.nms_threshold,
                    backend_id=cv2.dnn.DNN_BACKEND_CUDA if self.using_cuda else cv2.dnn.DNN_BACKEND_DEFAULT,
                    target_id=cv2.dnn.DNN_TARGET_CUDA if self.using_cuda else cv2.dnn.DNN_TARGET_CPU
                )

                self.yunet_available = True
                acceleration = "CUDA" if self.using_cuda else "CPU"
                logger.info(f"YuNet face detector initialized (2025 SOTA) using {acceleration} acceleration")
            except Exception as e:
                logger.error(f"Failed to initialize YuNet detector: {e}")
                self._initialize_fallback_detector()
        else:
            logger.warning(f"YuNet model not found at {yunet_path}")
            self._initialize_fallback_detector()

    def _initialize_fallback_detector(self):
        """Initialize traditional cascade classifier as fallback"""
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.yunet_available = False
        logger.info("Using fallback face detector (Haar Cascade)")

    def detect_faces(self, image: np.ndarray) -> List:
        """
        Detect faces in an image, using YuNet if available or fallback to traditional methods

        Args:
            image: Input image (BGR format)

        Returns:
            List of face data (format depends on detector type)
        """
        if self.yunet_available:
            return self.detect_faces_yunet(image)
        else:
            # Convert traditional detector results to match YuNet format
            faces = self.detect_faces_fallback(image)
            return [np.array([x, y, w, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0])
                   for x, y, w, h in faces]

    def detect_faces_yunet(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces using YuNet (2025 SOTA)"""
        h, w = image.shape[:2]

        # Set input size for this frame
        self.face_detector.setInputSize((w, h))

        # Detect faces
        _, faces = self.face_detector.detect(image)

        if faces is None:
            return []

        # Filter by confidence and return face regions
        valid_faces = []
        for face in faces:
            confidence = face[14]  # Detection confidence
            if confidence >= self.detection_confidence:
                valid_faces.append(face)

        return valid_faces

    def detect_faces_fallback(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback face detection using Haar Cascade classifier"""
        # Use traditional Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces if len(faces) > 0 else []

    def detect_head_pose(self, face_region: np.ndarray) -> str:
        """
        Estimate head pose for 3D face modeling

        Args:
            face_region: Extracted face region from image

        Returns:
            Angle classification: frontal, left_profile, right_profile, up_angle, down_angle
        """
        try:
            # Validate input to prevent error on corrupted frames
            if face_region is None or not isinstance(face_region, np.ndarray):
                logger.warning("Invalid face region provided to pose detector")
                return "frontal"

            # Check for empty or corrupted image
            if face_region.size == 0 or face_region.shape[0] == 0 or face_region.shape[1] == 0:
                logger.warning("Empty or corrupted face region detected")
                return "frontal"

            # Convert to grayscale for facial landmark detection
            try:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                height, width = gray.shape
            except cv2.error as e:
                logger.warning(f"OpenCV error during grayscale conversion: {e}")
                return "frontal"

            # Detect face to get landmarks
            try:
                faces = self.detect_faces(face_region)
            except Exception as e:
                logger.warning(f"Face detection failed during pose estimation: {e}")
                return "frontal"

            if len(faces) > 0:
                face_data = faces[0]

                # Extra validation to prevent array access errors
                if not isinstance(face_data, np.ndarray) or face_data.size < 15:
                    logger.warning("Invalid face data format detected")
                    return "frontal"

                # Extract landmarks if available (YuNet provides some facial points)
                if len(face_data) >= 15 and self.yunet_available:
                    # Safe extraction of landmarks with error handling
                    try:
                        # Extract landmark points
                        landmarks = face_data[4:14].reshape(-1, 2)

                        if len(landmarks) >= 2:
                            left_eye = landmarks[0]
                            right_eye = landmarks[1]

                            # Calculate eye distance and center
                            eye_center_x = (left_eye[0] + right_eye[0]) / 2
                            face_center_x = width / 2

                            # Determine horizontal pose
                            horizontal_offset = (eye_center_x - face_center_x) / width

                            if horizontal_offset > 0.15:
                                return "left_profile"
                            elif horizontal_offset < -0.15:
                                return "right_profile"
                            else:
                                # Check vertical pose based on face position
                                if len(landmarks) >= 3:
                                    nose_y = landmarks[2][1] if len(landmarks) > 2 else height/2
                                    face_center_y = height / 2
                                    vertical_offset = (nose_y - face_center_y) / height

                                    if vertical_offset > 0.1:
                                        return "down_angle"
                                    elif vertical_offset < -0.1:
                                        return "up_angle"
                    except IndexError as e:
                        logger.warning(f"Landmark extraction error: {e}")
                    except ValueError as e:
                        logger.warning(f"Landmark processing error: {e}")

            # If all else fails or no clear orientation detected, return default pose
            return "frontal"
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return "frontal"
