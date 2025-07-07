"""
ONNX Runtime helper module for facial recognition system
Provides robust ONNX handling and CUDA detection
"""

import logging
import os
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)

class OnnxRuntimeHelper:
    """
    Helper class for ONNX Runtime integration that handles
    various edge cases and CUDA compatibility issues
    """

    def __init__(self):
        self.onnx_available = False
        self.onnx_providers = []
        self.onnx_version = None
        self.cuda_available = False
        self.tensorrt_available = False
        self.error_message = None
        self.opencv_onnx_available = self._check_opencv_onnx()
        self.opencv_dnn_cuda = self._check_opencv_cuda()

        # Initialize
        self.check_onnx_availability()

    def _check_opencv_onnx(self) -> bool:
        """Check if OpenCV has ONNX support built in"""
        try:
            # Check if OpenCV can use ONNX models
            has_dnn = hasattr(cv2, 'dnn') and hasattr(cv2.dnn, 'readNetFromONNX')
            has_face_modules = hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2, 'FaceRecognizerSF')

            if has_dnn and has_face_modules:
                logger.info("OpenCV has native ONNX model support")
                return True
            else:
                logger.warning("OpenCV doesn't have complete ONNX support")
                return False
        except Exception as e:
            logger.error(f"Error checking OpenCV ONNX support: {str(e)}")
            return False

    def _check_opencv_cuda(self) -> bool:
        """Check if OpenCV has CUDA support"""
        try:
            # Check if OpenCV was built with CUDA support
            cv_info = cv2.getBuildInformation()
            has_cuda = "NVIDIA CUDA" in cv_info and "YES" in cv_info.split("NVIDIA CUDA")[1].split("\n")[0]

            if has_cuda:
                logger.info("OpenCV has CUDA support built in")
                cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
                if cuda_device_count > 0:
                    logger.info(f"Found {cuda_device_count} CUDA-capable device(s)")
                    return True
                else:
                    logger.warning("OpenCV has CUDA support but no devices found")
                    return False
            else:
                logger.info("OpenCV was built without CUDA support")
                return False
        except Exception as e:
            logger.error(f"Error checking OpenCV CUDA support: {str(e)}")
            return False

    def check_onnx_availability(self) -> bool:
        """
        Check if ONNX Runtime is available and which providers are supported
        Returns True if ONNX Runtime is available
        """
        try:
            # First, try to import ONNX Runtime
            import onnxruntime as ort
            self.onnx_version = ort.__version__

            # Get available providers
            available_providers = ort.get_available_providers()

            # Check for CUDA
            self.cuda_available = 'CUDAExecutionProvider' in available_providers
            self.tensorrt_available = 'TensorrtExecutionProvider' in available_providers

            # Set up providers list in preferred order
            self.onnx_providers = []
            if self.tensorrt_available:
                self.onnx_providers.append('TensorrtExecutionProvider')
            if self.cuda_available:
                self.onnx_providers.append('CUDAExecutionProvider')
            self.onnx_providers.append('CPUExecutionProvider')

            # Verify ONNX Runtime is actually usable with a simple test
            try:
                # Create test data
                test_data = np.random.rand(1, 3, 10, 10).astype(np.float32)

                # Create session options
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3  # Warning

                # Try to create a simple session with the CPU provider only
                # This should work regardless of CUDA availability
                cpu_session = ort.InferenceSession(
                    None,
                    session_options=session_options,
                    providers=['CPUExecutionProvider']
                )

                # ONNX Runtime is confirmed working
                self.onnx_available = True
                logger.info(f"ONNX Runtime {self.onnx_version} is available")

                # Log provider information
                providers_info = []
                if self.cuda_available:
                    providers_info.append("CUDA")
                if self.tensorrt_available:
                    providers_info.append("TensorRT")
                providers_info.append("CPU")

                logger.info(f"Available ONNX providers: {', '.join(providers_info)}")

                # Now test if CUDA is actually working if it's reported as available
                if self.cuda_available:
                    try:
                        # Try to get CUDA device info
                        cuda_device = ort.get_device("CUDA")
                        if cuda_device and cuda_device.is_available():
                            logger.info("CUDA device is available and working")
                        else:
                            logger.warning("CUDA device reported as available but not working correctly")
                            # Keep CUDA in providers list but warn about it
                    except Exception as cuda_err:
                        logger.warning(f"CUDA device initialization error: {str(cuda_err)}")
                        # Remove CUDA from providers if it fails initialization
                        if 'CUDAExecutionProvider' in self.onnx_providers:
                            self.onnx_providers.remove('CUDAExecutionProvider')
                            self.cuda_available = False

                return True

            except Exception as session_err:
                # ONNX Runtime imported but session creation failed
                self.error_message = f"ONNX Runtime initialization error: {str(session_err)}"
                logger.warning(self.error_message)
                self.onnx_available = False
                return False

        except ImportError as import_err:
            # ONNX Runtime not installed or import failed
            self.error_message = f"ONNX Runtime import error: {str(import_err)}"
            logger.warning(self.error_message)
            self.onnx_available = False
            return False

        except Exception as e:
            # Other unexpected errors
            self.error_message = f"Unexpected error checking ONNX availability: {str(e)}"
            logger.error(self.error_message)
            self.onnx_available = False
            return False

    def get_status_message(self) -> str:
        """Returns a human-readable status message about ONNX Runtime"""
        # First, check OpenCV's ONNX support
        opencv_status = "OpenCV ONNX support: Available"
        if self.opencv_dnn_cuda:
            opencv_status += " with CUDA acceleration"

        # Then, check standalone ONNX Runtime
        if not self.onnx_available:
            return f"[NOT AVAILABLE] ONNX Runtime not available: {self.error_message}\n{opencv_status}"

        providers = []
        if self.tensorrt_available:
            providers.append("TensorRT")
        if self.cuda_available:
            providers.append("CUDA")
        providers.append("CPU")

        return f"[AVAILABLE] ONNX Runtime {self.onnx_version} available with {', '.join(providers)}\n{opencv_status}"

    def optimize_onnx_model(self, model_path: str) -> Optional[str]:
        """
        Try to optimize an ONNX model if ONNX Runtime is available

        Args:
            model_path: Path to the ONNX model file

        Returns:
            Path to the optimized model if successful, None otherwise
        """
        if not self.onnx_available:
            logger.info("Cannot optimize model: ONNX Runtime not available")
            return None

        try:
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType

            # Create optimized model path
            model_path = Path(model_path)
            optimized_path = model_path.parent / f"{model_path.stem}_optimized{model_path.suffix}"

            # Quantize the model for better performance
            quantize_dynamic(
                model_input=str(model_path),
                model_output=str(optimized_path),
                weight_type=QuantType.QUInt8
            )

            logger.info(f"Successfully optimized model: {optimized_path}")
            return str(optimized_path)

        except Exception as e:
            logger.error(f"Failed to optimize model: {str(e)}")
            return None

    def configure_opencv_dnn(self, net=None):
        """
        Configure OpenCV DNN module to use CUDA if available

        Args:
            net: Optional DNN network to configure

        Returns:
            Configured network or None
        """
        if not net and not hasattr(cv2, 'dnn'):
            return None

        try:
            # Configure OpenCV DNN to use CUDA if available
            if self.opencv_dnn_cuda:
                logger.info("Setting OpenCV DNN to use CUDA")
                cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                # Configure specific network if provided
                if net:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                logger.info("Using OpenCV DNN with CPU")
                if net:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            return net
        except Exception as e:
            logger.error(f"Error configuring OpenCV DNN: {str(e)}")
            return net

    def create_inference_session(self, model_path: str, **kwargs) -> Optional[object]:
        """
        Create an ONNX Runtime inference session with appropriate error handling

        Args:
            model_path: Path to the ONNX model file
            **kwargs: Additional arguments to pass to ort.InferenceSession

        Returns:
            InferenceSession object or None if ONNX Runtime is not available
        """
        if not self.onnx_available:
            logger.warning("Cannot create inference session: ONNX Runtime not available")
            return None

        try:
            import onnxruntime as ort

            # Create session options
            session_options = kwargs.pop('session_options', ort.SessionOptions())

            # Use the providers determined during initialization
            providers = kwargs.pop('providers', self.onnx_providers)

            # Create the session
            session = ort.InferenceSession(
                model_path,
                session_options=session_options,
                providers=providers
            )

            # Log which provider was actually selected
            logger.info(f"Created ONNX session using provider: {session.get_providers()[0]}")

            return session

        except Exception as e:
            logger.error(f"Failed to create ONNX inference session: {str(e)}")
            return None

# Singleton instance for app-wide use
onnx_helper = OnnxRuntimeHelper()
