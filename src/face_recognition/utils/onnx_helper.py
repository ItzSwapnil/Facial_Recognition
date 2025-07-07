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
import importlib.util

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
        self.fallback_to_opencv = False

        # Initialize
        self.check_onnx_availability()

        # If ONNX Runtime is not available but OpenCV ONNX is, set fallback mode
        if not self.onnx_available and self.opencv_onnx_available:
            logger.info("Setting up OpenCV as fallback for ONNX models")
            self.fallback_to_opencv = True

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
            # Safer check for CUDA support
            has_cuda = "CUDA:" in cv_info and "YES" in cv_info.split("CUDA:")[1].split("\n")[0]

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

    def _check_dll_exists(self) -> bool:
        """Check if the required ONNX Runtime DLL files exist"""
        try:
            # Check if onnxruntime module spec exists
            ort_spec = importlib.util.find_spec("onnxruntime")
            if not ort_spec:
                return False

            # Get the package directory
            ort_dir = Path(ort_spec.origin).parent

            # Check for the core DLL
            core_dll = ort_dir / "onnxruntime_pybind11_state.dll"
            return core_dll.exists()
        except Exception as e:
            logger.error(f"Error checking ONNX Runtime DLLs: {str(e)}")
            return False

    def fix_missing_dlls(self) -> bool:
        """
        Attempt to fix missing DLL issues by installing the CPU version of ONNX Runtime
        Returns True if successful
        """
        logger.info("Attempting to fix missing ONNX Runtime DLLs")
        try:
            # Try to install the CPU version which has better compatibility
            import subprocess
            import sys

            logger.info("Installing onnxruntime CPU version")
            result = subprocess.run(
                [sys.executable, "-m", "uv", "add", "onnxruntime"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("Successfully installed onnxruntime")
                # Re-check if DLLs now exist
                if self._check_dll_exists():
                    logger.info("ONNX Runtime DLLs are now available")
                    # Re-initialize since we've fixed the issue
                    self.check_onnx_availability()
                    return True
                else:
                    logger.error("DLLs still missing after installation")
                    return False
            else:
                logger.error(f"Failed to install onnxruntime: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error fixing missing DLLs: {str(e)}")
            return False

    def check_onnx_availability(self) -> bool:
        """
        Check if ONNX Runtime is available and which providers are supported
        Returns True if ONNX Runtime is available
        """
        # First, check if the DLL files exist for Windows
        if sys.platform == 'win32' and not self._check_dll_exists():
            self.error_message = "Required ONNX Runtime DLL files are missing"
            logger.warning(self.error_message)
            self.onnx_available = False
            return False

        try:
            # Try to import ONNX Runtime
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

        if self.fallback_to_opencv:
            return f"[FALLBACK] Using OpenCV's native ONNX support. {self.error_message}\n{opencv_status}"

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
            # Check if CUDA is available in OpenCV
            has_cuda_backend = hasattr(cv2.dnn, 'DNN_BACKEND_CUDA')
            has_cuda_target = hasattr(cv2.dnn, 'DNN_TARGET_CUDA')

            # Check CUDA device availability
            cuda_device_count = 0
            if hasattr(cv2, 'cuda'):
                try:
                    cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
                except Exception:
                    pass

            # Determine if CUDA can be used
            cuda_available = has_cuda_backend and has_cuda_target and cuda_device_count > 0

            if cuda_available:
                logger.info("Setting OpenCV DNN to use CUDA backend")
                cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.opencv_dnn_cuda = True

                # Configure specific network if provided
                if net:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    logger.info("Network configured to use CUDA")
            else:
                # Check for OpenCL as a fallback
                has_opencl = cv2.ocl.haveOpenCL()
                if has_opencl:
                    cv2.ocl.setUseOpenCL(True)
                    if cv2.ocl.useOpenCL():
                        logger.info("Setting OpenCV DNN to use OpenCL")
                        cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                        cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

                        if net:
                            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                            logger.info("Network configured to use OpenCL")
                    else:
                        logger.info("OpenCL available but not enabled, using CPU")
                        if net:
                            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
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

        If ONNX Runtime is not available, falls back to OpenCV for model loading
        if possible.

        Args:
            model_path: Path to the ONNX model file
            **kwargs: Additional arguments to pass to ort.InferenceSession

        Returns:
            InferenceSession object or OpenCV DNN model if fallback, or None if neither is available
        """
        # If fallback mode is enabled, use OpenCV
        if self.fallback_to_opencv:
            try:
                logger.info(f"Using OpenCV fallback for ONNX model: {model_path}")
                net = cv2.dnn.readNetFromONNX(model_path)
                return self.configure_opencv_dnn(net)
            except Exception as e:
                logger.error(f"OpenCV fallback failed: {str(e)}")
                return None

        # Attempt to use ONNX Runtime if available
        if not self.onnx_available:
            logger.warning("Cannot create inference session: ONNX Runtime not available")

            # Try OpenCV fallback if not already in fallback mode
            if self.opencv_onnx_available:
                try:
                    logger.info(f"Attempting OpenCV fallback for ONNX model: {model_path}")
                    net = cv2.dnn.readNetFromONNX(model_path)
                    self.fallback_to_opencv = True
                    return self.configure_opencv_dnn(net)
                except Exception as e:
                    logger.error(f"OpenCV fallback failed: {str(e)}")
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

            # Try OpenCV fallback if ONNX Runtime fails
            if self.opencv_onnx_available:
                try:
                    logger.info(f"ONNX Runtime failed, trying OpenCV fallback for: {model_path}")
                    net = cv2.dnn.readNetFromONNX(model_path)
                    self.fallback_to_opencv = True
                    return self.configure_opencv_dnn(net)
                except Exception as cv_err:
                    logger.error(f"OpenCV fallback failed: {str(cv_err)}")
            return None

    def is_opencv_model(self, model):
        """Check if a model is an OpenCV DNN model"""
        return isinstance(model, cv2.dnn.Net)

# Create a global instance of the OnnxRuntimeHelper for other modules to import
onnx_helper = OnnxRuntimeHelper()
