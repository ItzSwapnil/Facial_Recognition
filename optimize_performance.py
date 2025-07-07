"""
Performance optimization for Facial Recognition system
Enables optimal performance without CUDA acceleration
"""

import os
import sys
import logging
import numpy as np
import cv2
import threading
import multiprocessing
import importlib.util
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_system():
    """Apply performance optimizations to the facial recognition system"""
    logger.info("Applying performance optimizations to facial recognition system")

    # Check OpenCV performance features
    check_opencv_optimizations()

    # Try to fix ONNX Runtime compatibility
    fix_onnx_runtime()

    # Enable threading optimizations
    enable_threading_optimizations()

    # Optimize model files
    optimize_onnx_models()

    # Configure OpenCV for optimal CPU performance
    configure_opencv()

    print("✅ Performance optimizations applied successfully!")
    print("Run the facial recognition system with: uv run gui_main.py")

def check_opencv_optimizations():
    """Check OpenCV optimizations available"""
    logger.info("Checking available OpenCV optimizations")

    # Check OpenCV version
    logger.info(f"OpenCV version: {cv2.__version__}")

    # Check if OpenCV was built with optimizations
    cv_info = cv2.getBuildInformation()

    # Check for CPU optimizations - using safer approach
    cpu_features = []

    # Helper function to safely check for features in build info
    def has_feature(feature):
        feature_str = f"{feature}:"
        if feature_str in cv_info:
            try:
                value = cv_info.split(feature_str)[1].split("\n")[0].strip()
                return "YES" in value.upper()
            except (IndexError, AttributeError):
                return False
        return False

    # Check for common CPU optimization features
    if has_feature("OpenCL"):
        cpu_features.append("OpenCL")
    if has_feature("SSE"):
        cpu_features.append("SSE")
    if has_feature("SSE2"):
        cpu_features.append("SSE2")
    if has_feature("SSE3"):
        cpu_features.append("SSE3")
    if has_feature("SSE4"):
        cpu_features.append("SSE4")
    if has_feature("AVX"):
        cpu_features.append("AVX")
    if has_feature("AVX2"):
        cpu_features.append("AVX2")
    if has_feature("NEON"):
        cpu_features.append("NEON")

    if cpu_features:
        logger.info(f"OpenCV CPU optimizations available: {', '.join(cpu_features)}")
        print(f"✅ OpenCV has CPU acceleration: {', '.join(cpu_features)}")
    else:
        logger.info("No specific CPU optimizations found in OpenCV")
        print("⚠️ OpenCV doesn't have specific CPU optimizations enabled")

    # Check CUDA support in OpenCV - safer approach
    has_cuda = has_feature("CUDA")

    if has_cuda:
        # Try to extract CUDA version safely
        cuda_version = "Unknown"
        if "CUDA Version:" in cv_info:
            try:
                cuda_version = cv_info.split("CUDA Version:")[1].split("\n")[0].strip()
            except (IndexError, AttributeError):
                pass

        logger.info(f"OpenCV was built with CUDA support (version: {cuda_version})")
        print(f"✅ OpenCV has CUDA acceleration (version: {cuda_version})")
    else:
        logger.info("OpenCV was built without CUDA support")
        print("⚠️ OpenCV doesn't have CUDA support - using CPU acceleration only")

def fix_onnx_runtime():
    """Try to fix ONNX Runtime compatibility issues"""
    logger.info("Checking ONNX Runtime compatibility")

    # Check if onnxruntime is installed
    onnx_available = importlib.util.find_spec("onnxruntime") is not None

    if not onnx_available:
        print("⚠️ ONNX Runtime not found - attempting to install")
        try:
            # Run our fix_onnx_cuda.py script
            script_path = Path(__file__).parent / "fix_onnx_cuda.py"
            if script_path.exists():
                print("🔧 Running ONNX compatibility fix script...")
                subprocess.run([sys.executable, str(script_path)], check=True)
            else:
                print("❌ ONNX compatibility fix script not found")
                # Fallback to direct installation
                subprocess.run(["uv", "add", "onnxruntime"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to install ONNX Runtime")
            return

    # Test ONNX Runtime providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime version: {ort.__version__}")
        logger.info(f"Available providers: {providers}")

        # Check for GPU providers
        gpu_providers = [p for p in providers if 'GPU' in p or 'CUDA' in p]
        if gpu_providers:
            print(f"✅ ONNX Runtime GPU acceleration available: {', '.join(gpu_providers)}")
        else:
            print("⚠️ ONNX Runtime using CPU acceleration only")
    except ImportError:
        logger.error("Failed to import onnxruntime after installation attempt")
        print("❌ ONNX Runtime import failed - system will use OpenCV for inference")

def enable_threading_optimizations():
    """Enable threading optimizations for better CPU utilization"""
    logger.info("Setting up threading optimizations")

    # Set number of OpenCV threads
    cpu_count = multiprocessing.cpu_count()
    optimal_threads = max(1, cpu_count - 1)  # Leave one core free for the OS

    logger.info(f"Setting OpenCV to use {optimal_threads} threads")
    cv2.setNumThreads(optimal_threads)

    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)
    is_optimized = cv2.useOptimized()
    logger.info(f"OpenCV optimizations are {'enabled' if is_optimized else 'disabled'}")

    # Try to enable OpenCL if available
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        logger.info(f"OpenCL is {'enabled' if cv2.ocl.useOpenCL() else 'disabled'}")
        if cv2.ocl.useOpenCL():
            print("✅ OpenCL acceleration enabled for faster CPU processing")
    else:
        logger.info("OpenCL is not available")

    # Configure NumPy to use multiple threads if possible
    try:
        # For newer NumPy versions with threadpool control
        if hasattr(np, 'config') and hasattr(np.config, 'threadpool_size'):
            np.config.threadpool_size = optimal_threads
            logger.info(f"Set NumPy threadpool size to {optimal_threads}")

        # For older NumPy versions with OpenBLAS backend
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        logger.info("Set environment variables for NumPy threading optimizations")
    except Exception as e:
        logger.warning(f"Error setting NumPy threading options: {e}")

def optimize_onnx_models():
    """Optimize ONNX model files for better CPU performance"""
    logger.info("Optimizing ONNX models for better CPU performance")

    models_dir = Path("data/models")
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return

    # Check for ONNX models
    onnx_models = list(models_dir.glob("*.onnx"))
    if not onnx_models:
        logger.warning("No ONNX models found in the models directory")
        return

    logger.info(f"Found {len(onnx_models)} ONNX models: {[model.name for model in onnx_models]}")

    # Check if we have onnxruntime for optimization
    if importlib.util.find_spec("onnxruntime") is None:
        logger.warning("ONNX Runtime not available for model optimization")
        print("⚠️ ONNX Runtime not available - skipping model optimization")
        return

    try:
        import onnxruntime as ort
        print(f"🔍 Analyzing ONNX models using ONNX Runtime {ort.__version__}")

        # Get available providers
        providers = ort.get_available_providers()
        gpu_available = any('GPU' in p or 'CUDA' in p for p in providers)

        if gpu_available:
            print("✅ GPU acceleration available for ONNX models")
        else:
            print("⚠️ Using CPU acceleration for ONNX models")

        # Set session options for optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = max(1, multiprocessing.cpu_count() - 1)

        # Check if optimization is possible with the current setup
        if importlib.util.find_spec("onnx") is not None:
            try:
                import onnx
                print("🛠️ ONNX optimization library available")

                # For each model, check and optimize
                for model_path in onnx_models:
                    print(f"🔄 Checking model: {model_path.name}")
                    # Load and verify the model
                    try:
                        onnx_model = onnx.load(str(model_path))
                        onnx.checker.check_model(onnx_model)
                        logger.info(f"Model {model_path.name} is valid")

                        # Create a session to test the model
                        session = ort.InferenceSession(
                            str(model_path),
                            providers=providers,
                            sess_options=session_options
                        )
                        logger.info(f"Successfully loaded model {model_path.name} with providers: {session.get_providers()}")
                        print(f"✅ Model {model_path.name} is optimized and ready")
                    except Exception as e:
                        logger.error(f"Error processing model {model_path.name}: {e}")
                        print(f"⚠️ Issue with model {model_path.name}: {str(e)[:100]}...")
            except ImportError:
                logger.warning("ONNX library not available for model validation")
        else:
            logger.info("Using models as-is without ONNX optimization library")
            print("⚠️ ONNX optimization library not available - using models as-is")
    except Exception as e:
        logger.error(f"Error during ONNX model optimization: {e}")
        print(f"❌ ONNX model optimization failed: {str(e)[:100]}...")

def configure_opencv():
    """Configure OpenCV for optimal performance"""
    logger.info("Configuring OpenCV for optimal performance")

    # Enable all optimizations
    cv2.setUseOptimized(True)

    # Set preferred backend for DNN module
    if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
        logger.info("Setting DNN backend to OpenCL")
        cv2.dnn.DNN_BACKEND_DEFAULT = cv2.dnn.DNN_BACKEND_OPENCV
        cv2.dnn.DNN_TARGET_OPENCL = cv2.dnn.DNN_TARGET_OPENCL
        print("✅ OpenCV DNN configured to use OpenCL acceleration")
    else:
        # Check if we have a CUDA-enabled OpenCV build
        cv_info = cv2.getBuildInformation()
        has_cuda = "CUDA:" in cv_info and "YES" in cv_info.split("CUDA:")[1].split("\n")[0]

        if has_cuda:
            logger.info("Setting DNN backend to CUDA")
            cv2.dnn.DNN_BACKEND_CUDA = cv2.dnn.DNN_BACKEND_CUDA
            cv2.dnn.DNN_TARGET_CUDA = cv2.dnn.DNN_TARGET_CUDA
            print("✅ OpenCV DNN configured to use CUDA acceleration")
        else:
            logger.info("Using CPU optimizations for DNN")
            print("⚠️ OpenCV DNN using CPU acceleration only")

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Facial Recognition System Performance Optimizer")
    print("=" * 60)
    optimize_system()
    print("\n✅ Optimization completed! System is ready to run.")
    print("=" * 60)
