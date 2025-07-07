"""
Script to fix ONNX Runtime CUDA compatibility issue
Adapted for UV package manager and Python 3.13
"""

import sys
import subprocess
import os
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_onnx_compatibility():
    """
    Fix ONNX Runtime compatibility with CUDA by trying compatible versions
    Uses UV package manager instead of pip
    """
    logger.info("Checking CUDA compatibility for ONNX Runtime")
    logger.info(f"Python version: {platform.python_version()}")

    # Check if using Python 3.13
    python_version = tuple(map(int, platform.python_version().split('.')))
    is_python_313 = python_version >= (3, 13)

    if is_python_313:
        logger.warning("Python 3.13 detected - most ONNX Runtime GPU wheels are not compatible")
        logger.info("Will try CPU-only version which may be compatible")

    # First uninstall current versions
    logger.info("Uninstalling current ONNX Runtime installations")
    try:
        # UV doesn't support -y flag
        subprocess.check_call(["uv", "pip", "uninstall", "onnxruntime"])
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error during uninstallation of onnxruntime: {e}")

    try:
        subprocess.check_call(["uv", "pip", "uninstall", "onnxruntime-gpu"])
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error during uninstallation of onnxruntime-gpu: {e}")

    # If Python 3.13, we can only try CPU version
    if is_python_313:
        logger.info("For Python 3.13, only trying CPU version as GPU version is not compatible")
        try:
            subprocess.check_call(["uv", "pip", "install", "onnxruntime"])
            logger.info("Successfully installed CPU-only version of ONNX Runtime")

            # Verify installation
            try:
                import onnxruntime as ort
                logger.info(f"Verified ONNX Runtime CPU installation: {ort.__version__}")
                return False  # No GPU, but CPU is working
            except ImportError:
                logger.error("Failed to import ONNX Runtime after installation")
                return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CPU version: {e}")
            return False

    # For Python < 3.13, try GPU versions
    versions_to_try = [
        "onnxruntime-gpu==1.16.3",  # Compatible with CUDA 11.8
        "onnxruntime-gpu==1.15.1",  # Compatible with CUDA 11.8
        "onnxruntime-gpu==1.14.1",  # Compatible with CUDA 11.6
        "onnxruntime-gpu==1.13.1",  # Compatible with CUDA 11.6
    ]

    # Try installing each version until one works
    for version in versions_to_try:
        try:
            logger.info(f"Attempting to install {version}")
            subprocess.check_call(["uv", "pip", "install", version])

            # Test if the installation works
            test_result = test_onnx_installation()
            if test_result:
                logger.info(f"✅ Successfully installed compatible version: {version}")
                return True
            else:
                logger.warning(f"❌ Installation of {version} failed CUDA test")
                # Uninstall the failed version
                subprocess.check_call(["uv", "pip", "uninstall", "onnxruntime-gpu"])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Installation error for {version}: {e}")

    # If all attempts fail, install CPU version as fallback
    logger.info("All GPU versions failed, installing CPU version as fallback")
    try:
        subprocess.check_call(["uv", "pip", "install", "onnxruntime"])
        logger.info("✅ Successfully installed CPU version as fallback")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install CPU fallback: {e}")
        return False

def test_onnx_installation():
    """Test if ONNX Runtime with CUDA works correctly"""
    test_script = """
import onnxruntime as ort
import numpy as np

print(f"ONNX Runtime version: {ort.__version__}")
providers = ort.get_available_providers()
print(f"Available providers: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("CUDA is available!")
    # Try to initialize CUDA provider
    try:
        # Test with a simple operation
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        session_options = ort.SessionOptions()
        session = ort.InferenceSession(None, session_options, providers=['CUDAExecutionProvider'])
        print("CUDA initialization successful!")
        exit(0)
    except Exception as e:
        print(f"CUDA initialization failed: {e}")
        exit(1)
else:
    print("CUDA provider not available")
    exit(1)
"""

    # Write test script to temporary file
    with open("test_onnx.py", "w") as f:
        f.write(test_script)

    try:
        # Run the test script
        subprocess.check_call([sys.executable, "test_onnx.py"])
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        # Clean up
        if os.path.exists("test_onnx.py"):
            os.remove("test_onnx.py")

if __name__ == "__main__":
    print("Starting ONNX Runtime compatibility fix...")
    result = fix_onnx_compatibility()
    if result:
        print("Successfully installed ONNX Runtime with CUDA support!")
    else:
        print("Reverted to CPU-only ONNX Runtime as fallback")
