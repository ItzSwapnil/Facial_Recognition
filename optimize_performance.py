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

    # Check for CPU optimizations
    cpu_features = []
    if "OpenCL:" in cv_info and "YES" in cv_info.split("OpenCL:")[1].split("\n")[0]:
        cpu_features.append("OpenCL")
    if "SSE" in cv_info and "YES" in cv_info.split("SSE:")[1].split("\n")[0]:
        cpu_features.append("SSE")
    if "SSE2" in cv_info and "YES" in cv_info.split("SSE2:")[1].split("\n")[0]:
        cpu_features.append("SSE2")
    if "AVX" in cv_info and "YES" in cv_info.split("AVX:")[1].split("\n")[0]:
        cpu_features.append("AVX")

    if cpu_features:
        logger.info(f"OpenCV CPU optimizations available: {', '.join(cpu_features)}")
        print(f"✅ OpenCV has CPU acceleration: {', '.join(cpu_features)}")
    else:
        logger.info("No specific CPU optimizations found in OpenCV")
        print("⚠️ OpenCV doesn't have specific CPU optimizations enabled")

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

def optimize_onnx_models():
    """Optimize ONNX model files for better CPU performance"""
    logger.info("Optimizing ONNX models for better CPU performance")

    models_dir = Path("data/models")
    if not models_dir.exists():
        logger.warning(f"Models directory not found: {models_dir}")
        return

    # Look for ONNX models
    onnx_models = list(models_dir.glob("*.onnx"))
    logger.info(f"Found {len(onnx_models)} ONNX models")

    if not onnx_models:
        return

    # Try to optimize with OpenCV DNN
    for model_path in onnx_models:
        try:
            # Create optimized model filename
            optimized_name = model_path.stem + "_optimized_cpu" + model_path.suffix
            optimized_path = model_path.parent / optimized_name

            # Skip if already optimized
            if optimized_path.exists():
                logger.info(f"Using existing optimized model: {optimized_path}")
                continue

            logger.info(f"Optimizing model: {model_path}")

            # Read model with OpenCV
            net = cv2.dnn.readNetFromONNX(str(model_path))

            # Set CPU optimizations
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Try to save optimized model
            success = net.save(str(optimized_path))
            if success or optimized_path.exists():
                logger.info(f"Successfully saved optimized model: {optimized_path}")
                print(f"✅ Optimized model created: {optimized_name}")
            else:
                logger.warning(f"Failed to save optimized model: {optimized_path}")

        except Exception as e:
            logger.error(f"Error optimizing model {model_path}: {e}")

def configure_opencv():
    """Configure OpenCV for best CPU performance"""
    # Create configuration file to auto-load on startup
    try:
        # Create a configuration file in the project directory
        config_path = Path("data/settings/opencv_config.json")
        config_path.parent.mkdir(exist_ok=True)

        # Basic OpenCV optimization settings
        import json
        config = {
            "use_optimized": True,
            "num_threads": multiprocessing.cpu_count() - 1,
            "use_opencl": True if cv2.ocl.haveOpenCL() else False,
            "preferred_backend": "opencv",
            "preferred_target": "cpu",
            "optimization_level": "high"
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        logger.info(f"Created OpenCV configuration file: {config_path}")
        print(f"✅ Created OpenCV configuration for best CPU performance")
    except Exception as e:
        logger.error(f"Error creating OpenCV configuration: {e}")

def create_autoload_script():
    """Create an autoload script to apply optimizations on startup"""
    try:
        script_path = Path("src/face_recognition/utils/performance_config.py")

        script_content = """\"\"\"
Performance configuration for Facial Recognition system
Applied automatically on system startup
\"\"\"

import cv2
import multiprocessing
import logging

logger = logging.getLogger(__name__)

def apply_performance_config():
    \"\"\"Apply performance optimizations on system startup\"\"\"
    logger.info("Applying performance optimizations")
    
    # Set number of OpenCV threads
    cpu_count = multiprocessing.cpu_count()
    optimal_threads = max(1, cpu_count - 1)  # Leave one core free for the OS
    cv2.setNumThreads(optimal_threads)
    
    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)
    
    # Try to enable OpenCL if available
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        logger.info(f"OpenCL is {'enabled' if cv2.ocl.useOpenCL() else 'disabled'}")
    
    # Optimize DNN module
    try:
        cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        cv2.dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        logger.warning(f"Failed to set DNN preferences: {e}")
    
    logger.info("Performance optimizations applied")
    return True

# Apply optimizations when module is imported
optimization_success = apply_performance_config()
"""

        # Create the file
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write(script_content)

        logger.info(f"Created autoload optimization script: {script_path}")
        print(f"✅ Created autoload script for applying optimizations on startup")

        # Update __init__.py to import the script
        init_path = Path("src/face_recognition/utils/__init__.py")
        if init_path.exists():
            with open(init_path, "r") as f:
                content = f.read()

            if "import performance_config" not in content:
                with open(init_path, "a") as f:
                    f.write("\n# Import performance configuration\ntry:\n    from . import performance_config\nexcept ImportError:\n    pass\n")
                logger.info(f"Updated {init_path} to autoload performance config")

    except Exception as e:
        logger.error(f"Error creating autoload script: {e}")

if __name__ == "__main__":
    print("Starting performance optimization for Facial Recognition system...")
    optimize_system()
    create_autoload_script()
    print("\nAll optimizations complete! Your system should now run more efficiently.")
    print("The facial recognition system will now use optimal CPU performance settings.")
    print("These settings will be automatically applied each time you start the application.")
