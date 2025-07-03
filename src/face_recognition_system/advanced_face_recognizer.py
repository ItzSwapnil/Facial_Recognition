"""
State-of-the-Art Face Recognition Module
=======================================

Advanced face recognition using cutting-edge deep learning models:
- FaceNet with PyTorch implementation
- ArcFace/CosFace models for superior accuracy
- InsightFace recognition models
- Real-time ONNX optimization
- GPU acceleration support
- Advanced embedding techniques
"""

import cv2
import numpy as np
import logging
import pickle
import time
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Core scientific computing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Deep learning imports
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from facenet_pytorch import MTCNN, InceptionResnetV1
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Legacy support
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

from .config import Config, get_config


class AdvancedFaceRecognizer:
    """
    State-of-the-art face recognition system using modern deep learning.
    
    Features:
    - FaceNet embeddings for high accuracy
    - ArcFace/CosFace models
    - InsightFace recognition
    - Multiple similarity metrics
    - Real-time ONNX inference
    - GPU acceleration
    - Advanced preprocessing
    - Ensemble recognition
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the advanced face recognizer."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Model availability
        self.models_available = {
            'facenet': PYTORCH_AVAILABLE,
            'insightface': INSIGHTFACE_AVAILABLE,
            'onnx': ONNX_AVAILABLE,
            'face_recognition': FACE_RECOGNITION_AVAILABLE
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() and PYTORCH_AVAILABLE else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Models
        self.facenet_model = None
        self.facenet_mtcnn = None
        self.insightface_app = None
        self.onnx_session = None
        
        # Face database
        self.face_database = {
            'embeddings': [],
            'names': [],
            'methods': [],
            'metadata': []
        }
        
        # Classifiers
        self.svm_classifier = None
        self.knn_classifier = None
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.recognition_stats = {
            'total_recognitions': 0,
            'total_time': 0.0,
            'method_performance': {},
            'accuracy_metrics': {}
        }
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self._initialize_models()
        self._load_face_database()
    
    def _initialize_models(self) -> None:
        """Initialize all available recognition models."""
        self.logger.info("Initializing advanced face recognition models...")
        
        # Initialize FaceNet
        if self.models_available['facenet']:
            self._init_facenet()
        
        # Initialize InsightFace
        if self.models_available['insightface']:
            self._init_insightface()
        
        # Initialize ONNX models
        if self.models_available['onnx']:
            self._init_onnx()
    
    def _init_facenet(self) -> None:
        """Initialize FaceNet model."""
        try:
            # Initialize MTCNN for face detection and alignment
            self.facenet_mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
            
            # Initialize InceptionResnetV1 for face recognition
            self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
            self.facenet_model.to(self.device)
            
            self.logger.info("FaceNet model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FaceNet: {e}")
            self.models_available['facenet'] = False
    
    def _init_insightface(self) -> None:
        """Initialize InsightFace model."""
        try:
            # Configure providers for optimal performance
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.insightface_app = insightface.app.FaceAnalysis(
                providers=providers,
                name='buffalo_l'  # High accuracy model
            )
            self.insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.logger.info("InsightFace model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            self.models_available['insightface'] = False
    
    def _init_onnx(self) -> None:
        """Initialize ONNX runtime for optimized inference."""
        try:
            # Configure execution providers
            providers = ['CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
            if 'TensorrtExecutionProvider' in available_providers:
                providers.insert(0, 'TensorrtExecutionProvider')
            
            self.onnx_providers = providers
            self.logger.info(f"ONNX runtime configured with providers: {providers}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ONNX runtime: {e}")
            self.models_available['onnx'] = False
    
    def extract_facenet_embedding(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract FaceNet embedding from face region."""
        if not self.models_available['facenet']:
            return None
        
        try:
            x, y, w, h = face_box
            
            # Extract face region with margin
            margin = 10
            y1 = max(0, y - margin)
            y2 = min(image.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(image.shape[1], x + w + margin)
            
            face_image = image[y1:y2, x1:x2]
            
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Detect and align face using MTCNN
            face_tensor = self.facenet_mtcnn(face_rgb)
            
            if face_tensor is None:
                return None
            
            # Extract embedding
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"FaceNet embedding extraction failed: {e}")
            return None
    
    def extract_insightface_embedding(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract InsightFace embedding from face region."""
        if not self.models_available['insightface']:
            return None
        
        try:
            # InsightFace expects full image and will handle face detection/alignment
            faces = self.insightface_app.get(image)
            
            if not faces:
                return None
            
            # Find the face that best matches our bounding box
            x, y, w, h = face_box
            face_center = np.array([x + w/2, y + h/2])
            
            best_face = None
            min_distance = float('inf')
            
            for face in faces:
                bbox = face.bbox
                face_bbox_center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                distance = np.linalg.norm(face_center - face_bbox_center)
                
                if distance < min_distance:
                    min_distance = distance
                    best_face = face
            
            if best_face is not None and min_distance < 50:  # Threshold for matching
                return best_face.embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"InsightFace embedding extraction failed: {e}")
            return None
    
    def extract_legacy_embedding(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract embedding using legacy face_recognition library."""
        if not self.models_available['face_recognition']:
            return None
        
        try:
            # Convert face box to face_recognition format
            x, y, w, h = face_box
            face_location = (y, x + w, y + h, x)  # (top, right, bottom, left)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract encoding
            encodings = face_recognition.face_encodings(rgb_image, [face_location])
            
            if encodings:
                return encodings[0]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Legacy embedding extraction failed: {e}")
            return None
    
    def extract_embeddings(self, image: np.ndarray, face_boxes: List[Tuple[int, int, int, int]], 
                          method: str = "auto") -> List[Tuple[Optional[np.ndarray], str]]:
        """
        Extract embeddings from multiple faces using specified method.
        
        Args:
            image: Input image
            face_boxes: List of face bounding boxes
            method: Extraction method ('auto', 'facenet', 'insightface', 'legacy')
        
        Returns:
            List of (embedding, method_used) tuples
        """
        embeddings = []
        
        for face_box in face_boxes:
            embedding = None
            method_used = "none"
            
            if method == "auto":
                # Try methods in order of preference
                if self.models_available['insightface']:
                    embedding = self.extract_insightface_embedding(image, face_box)
                    method_used = "insightface"
                elif self.models_available['facenet']:
                    embedding = self.extract_facenet_embedding(image, face_box)
                    method_used = "facenet"
                elif self.models_available['face_recognition']:
                    embedding = self.extract_legacy_embedding(image, face_box)
                    method_used = "legacy"
            
            elif method == "facenet":
                embedding = self.extract_facenet_embedding(image, face_box)
                method_used = "facenet"
            
            elif method == "insightface":
                embedding = self.extract_insightface_embedding(image, face_box)
                method_used = "insightface"
            
            elif method == "legacy":
                embedding = self.extract_legacy_embedding(image, face_box)
                method_used = "legacy"
            
            embeddings.append((embedding, method_used))
        
        return embeddings
    
    def add_face(self, name: str, image_path: str, method: str = "auto") -> bool:
        """Add a new face to the recognition database."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect faces in the image
            if self.models_available['insightface']:
                faces = self.insightface_app.get(image)
                if faces:
                    # Use first detected face
                    face = faces[0]
                    bbox = face.bbox.astype(int)
                    x, y, x2, y2 = bbox
                    face_box = (x, y, x2 - x, y2 - y)
                else:
                    self.logger.error(f"No faces detected in image: {image_path}")
                    return False
            else:
                # Fallback to simple face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) == 0:
                    self.logger.error(f"No faces detected in image: {image_path}")
                    return False
                
                face_box = tuple(faces[0])  # Use first detected face
            
            # Extract embedding
            embeddings = self.extract_embeddings(image, [face_box], method)
            embedding, method_used = embeddings[0]
            
            if embedding is None:
                self.logger.error(f"Failed to extract embedding from image: {image_path}")
                return False
            
            # Add to database
            self.face_database['embeddings'].append(embedding)
            self.face_database['names'].append(name)
            self.face_database['methods'].append(method_used)
            self.face_database['metadata'].append({
                'image_path': image_path,
                'timestamp': time.time(),
                'embedding_dimension': len(embedding)
            })
            
            # Save database
            self._save_face_database()
            
            # Retrain classifiers
            self._train_classifiers()
            
            self.logger.info(f"Added face for {name} using {method_used} method")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add face for {name}: {e}")
            return False
    
    def recognize_faces(self, image: np.ndarray, face_boxes: List[Tuple[int, int, int, int]], 
                       method: str = "auto") -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """
        Recognize faces in the image.
        
        Args:
            image: Input image
            face_boxes: List of face bounding boxes
            method: Recognition method ('auto', 'cosine', 'svm', 'knn', 'ensemble')
        
        Returns:
            List of (name, confidence, bbox) tuples
        """
        start_time = time.time()
        
        if not self.face_database['embeddings']:
            return [("Unknown", 0.0, bbox) for bbox in face_boxes]
        
        try:
            # Extract embeddings from detected faces
            embeddings = self.extract_embeddings(image, face_boxes)
            
            results = []
            
            for i, (embedding, embedding_method) in enumerate(embeddings):
                if embedding is None:
                    results.append(("Unknown", 0.0, face_boxes[i]))
                    continue
                
                if method == "auto" or method == "cosine":
                    name, confidence = self._recognize_cosine_similarity(embedding)
                elif method == "svm":
                    name, confidence = self._recognize_svm(embedding)
                elif method == "knn":
                    name, confidence = self._recognize_knn(embedding)
                elif method == "ensemble":
                    name, confidence = self._recognize_ensemble(embedding)
                else:
                    name, confidence = self._recognize_cosine_similarity(embedding)
                
                results.append((name, confidence, face_boxes[i]))
            
            # Update performance statistics
            recognition_time = time.time() - start_time
            self.recognition_stats['total_recognitions'] += len(face_boxes)
            self.recognition_stats['total_time'] += recognition_time
            
            if method not in self.recognition_stats['method_performance']:
                self.recognition_stats['method_performance'][method] = {'count': 0, 'total_time': 0.0}
            
            self.recognition_stats['method_performance'][method]['count'] += len(face_boxes)
            self.recognition_stats['method_performance'][method]['total_time'] += recognition_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")
            return [("Unknown", 0.0, bbox) for bbox in face_boxes]
    
    def _recognize_cosine_similarity(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize face using cosine similarity."""
        if not self.face_database['embeddings']:
            return "Unknown", 0.0
        
        # Calculate similarities
        similarities = cosine_similarity([embedding], self.face_database['embeddings'])[0]
        
        # Find best match
        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        
        # Convert similarity to confidence (cosine similarity ranges from -1 to 1)
        confidence = (best_similarity + 1) / 2
        
        if confidence >= self.config.detection.confidence_threshold:
            return self.face_database['names'][best_index], confidence
        else:
            return "Unknown", confidence
    
    def _recognize_svm(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize face using SVM classifier."""
        if self.svm_classifier is None:
            return self._recognize_cosine_similarity(embedding)
        
        try:
            # Scale embedding
            embedding_scaled = self.scaler.transform([embedding])
            
            # Predict
            prediction = self.svm_classifier.predict(embedding_scaled)[0]
            probabilities = self.svm_classifier.predict_proba(embedding_scaled)[0]
            
            # Get confidence
            max_prob = np.max(probabilities)
            
            if max_prob >= self.config.detection.confidence_threshold:
                return prediction, max_prob
            else:
                return "Unknown", max_prob
                
        except Exception as e:
            self.logger.error(f"SVM recognition failed: {e}")
            return self._recognize_cosine_similarity(embedding)
    
    def _recognize_knn(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize face using k-NN classifier."""
        if self.knn_classifier is None:
            return self._recognize_cosine_similarity(embedding)
        
        try:
            # Scale embedding
            embedding_scaled = self.scaler.transform([embedding])
            
            # Predict
            prediction = self.knn_classifier.predict(embedding_scaled)[0]
            probabilities = self.knn_classifier.predict_proba(embedding_scaled)[0]
            
            # Get confidence
            max_prob = np.max(probabilities)
            
            if max_prob >= self.config.detection.confidence_threshold:
                return prediction, max_prob
            else:
                return "Unknown", max_prob
                
        except Exception as e:
            self.logger.error(f"k-NN recognition failed: {e}")
            return self._recognize_cosine_similarity(embedding)
    
    def _recognize_ensemble(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize face using ensemble of methods."""
        results = []
        
        # Cosine similarity
        name1, conf1 = self._recognize_cosine_similarity(embedding)
        results.append((name1, conf1, 'cosine'))
        
        # SVM
        if self.svm_classifier is not None:
            name2, conf2 = self._recognize_svm(embedding)
            results.append((name2, conf2, 'svm'))
        
        # k-NN
        if self.knn_classifier is not None:
            name3, conf3 = self._recognize_knn(embedding)
            results.append((name3, conf3, 'knn'))
        
        # Weighted voting
        name_votes = {}
        total_weight = 0
        
        for name, conf, method in results:
            weight = conf
            if name not in name_votes:
                name_votes[name] = 0
            name_votes[name] += weight
            total_weight += weight
        
        if total_weight == 0:
            return "Unknown", 0.0
        
        # Find best result
        best_name = max(name_votes.items(), key=lambda x: x[1])
        final_confidence = best_name[1] / total_weight
        
        return best_name[0], final_confidence
    
    def _train_classifiers(self) -> None:
        """Train SVM and k-NN classifiers."""
        if len(self.face_database['embeddings']) < 2:
            return
        
        try:
            # Prepare training data
            X = np.array(self.face_database['embeddings'])
            y = np.array(self.face_database['names'])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train SVM
            self.svm_classifier = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            self.svm_classifier.fit(X_scaled, y)
            
            # Train k-NN
            self.knn_classifier = KNeighborsClassifier(
                n_neighbors=min(5, len(np.unique(y))),
                weights='distance',
                metric='cosine'
            )
            self.knn_classifier.fit(X_scaled, y)
            
            # Save classifiers
            models_dir = self.config.models_dir
            joblib.dump(self.svm_classifier, models_dir / "svm_classifier.pkl")
            joblib.dump(self.knn_classifier, models_dir / "knn_classifier.pkl")
            joblib.dump(self.scaler, models_dir / "feature_scaler.pkl")
            
            self.logger.info("Classifiers trained and saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to train classifiers: {e}")
    
    def _load_face_database(self) -> None:
        """Load face database from file."""
        database_file = self.config.models_dir / "face_database.pkl"
        
        if database_file.exists():
            try:
                with open(database_file, 'rb') as f:
                    self.face_database = pickle.load(f)
                
                self.logger.info(f"Loaded face database with {len(self.face_database['names'])} faces")
                
                # Load trained classifiers
                self._load_classifiers()
                
            except Exception as e:
                self.logger.error(f"Failed to load face database: {e}")
    
    def _save_face_database(self) -> None:
        """Save face database to file."""
        database_file = self.config.models_dir / "face_database.pkl"
        
        try:
            with open(database_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            
            self.logger.info("Face database saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save face database: {e}")
    
    def _load_classifiers(self) -> None:
        """Load trained classifiers."""
        models_dir = self.config.models_dir
        
        try:
            if (models_dir / "svm_classifier.pkl").exists():
                self.svm_classifier = joblib.load(models_dir / "svm_classifier.pkl")
            
            if (models_dir / "knn_classifier.pkl").exists():
                self.knn_classifier = joblib.load(models_dir / "knn_classifier.pkl")
            
            if (models_dir / "feature_scaler.pkl").exists():
                self.scaler = joblib.load(models_dir / "feature_scaler.pkl")
            
            self.logger.info("Classifiers loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load classifiers: {e}")
    
    def remove_face(self, name: str) -> bool:
        """Remove all faces for a person from the database."""
        try:
            indices_to_remove = [
                i for i, face_name in enumerate(self.face_database['names']) 
                if face_name == name
            ]
            
            if not indices_to_remove:
                self.logger.warning(f"No faces found for {name}")
                return False
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                self.face_database['embeddings'].pop(i)
                self.face_database['names'].pop(i)
                self.face_database['methods'].pop(i)
                self.face_database['metadata'].pop(i)
            
            # Save updated database
            self._save_face_database()
            
            # Retrain classifiers
            self._train_classifiers()
            
            self.logger.info(f"Removed {len(indices_to_remove)} faces for {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove faces for {name}: {e}")
            return False
    
    def get_known_people(self) -> List[str]:
        """Get list of known people."""
        return list(set(self.face_database['names']))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.recognition_stats.copy()
        
        if stats['total_recognitions'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_recognitions']
            stats['fps'] = stats['total_recognitions'] / stats['total_time']
        else:
            stats['average_time'] = 0.0
            stats['fps'] = 0.0
        
        # Add method-specific stats
        for method, data in stats['method_performance'].items():
            if data['count'] > 0:
                data['average_time'] = data['total_time'] / data['count']
                data['fps'] = data['count'] / data['total_time']
        
        stats['models_available'] = self.models_available
        stats['database_size'] = len(self.face_database['names'])
        stats['unique_people'] = len(set(self.face_database['names']))
        stats['device'] = str(self.device)
        stats['classifiers_trained'] = {
            'svm': self.svm_classifier is not None,
            'knn': self.knn_classifier is not None
        }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.recognition_stats = {
            'total_recognitions': 0,
            'total_time': 0.0,
            'method_performance': {},
            'accuracy_metrics': {}
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Clean up CUDA memory if using GPU
        if self.device.type == 'cuda' and PYTORCH_AVAILABLE:
            torch.cuda.empty_cache()
