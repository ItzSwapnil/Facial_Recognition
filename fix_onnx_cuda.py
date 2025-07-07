"""
Script to fix ONNX Runtime CUDA compatibility issue
Adapted for UV package manager and Python 3.13
"""

import sys
import subprocess
import os
import logging
import platform
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_onnx_installation():
    """Test if ONNX Runtime can be imported and GPU providers are available"""
    try:
        # Check if onnxruntime can be imported
        if importlib.util.find_spec("onnxruntime") is None:
            logger.error("ONNX Runtime is not installed")
            return False

        import onnxruntime as ort
        logger.info(f"ONNX Runtime version: {ort.__version__}")

        # Check available providers
        providers = ort.get_available_providers()
        logger.info(f"Available providers: {providers}")

        # Check if GPU providers are available
        gpu_providers = [p for p in providers if 'GPU' in p or 'CUDA' in p]
        if gpu_providers:
            logger.info(f"GPU acceleration is available: {gpu_providers}")
            return True
        else:
            logger.warning("GPU acceleration is not available")
            return False

    except Exception as e:
        logger.error(f"Error testing ONNX Runtime: {str(e)}")
        return False

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
        logger.warning("Python 3.13 detected - special handling required for ONNX Runtime GPU wheels")

    # First uninstall current versions
    logger.info("Uninstalling current ONNX Runtime installations")
    try:
        subprocess.check_call(["uv", "remove", "onnxruntime"], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logger.info("onnxruntime was not installed")

    try:
        subprocess.check_call(["uv", "remove", "onnxruntime-gpu"], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logger.info("onnxruntime-gpu was not installed")

    # Check for CUDA installation
    cuda_available = False
    try:
        # Look for nvcc or other CUDA indicators
        result = subprocess.run(["where", "nvcc"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            cuda_available = True
            logger.info(f"CUDA found: {result.stdout.strip()}")
        else:
            # Check for CUDA_PATH environment variable
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path and os.path.exists(cuda_path):
                cuda_available = True
                logger.info(f"CUDA found via environment variable: {cuda_path}")
    except Exception as e:
        logger.warning(f"Error checking for CUDA: {e}")

    if not cuda_available:
        logger.warning("CUDA does not appear to be installed. Will use CPU-only version.")
        try:
            subprocess.check_call(["uv", "add", "onnxruntime"])
            logger.info("Successfully installed CPU-only version of ONNX Runtime")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CPU version: {e}")
            return False

    # If Python 3.13 and CUDA is available, try specific compatible versions
    if is_python_313:
        logger.info("For Python 3.13, trying specially built ONNX Runtime versions")

        # For Python 3.13, newer versions might work better
        versions_to_try = [
            "onnxruntime-gpu==1.17.0",  # Try a slightly older version first
            "onnxruntime-gpu",  # Latest version (as of script update)
        ]

        for version in versions_to_try:
            try:
                logger.info(f"Attempting to install {version}")
                subprocess.check_call(["uv", "add", version])

                # Test if the installation works
                if test_onnx_installation():
                    logger.info(f"✅ Successfully installed compatible version: {version}")
                    return True
                else:
                    logger.warning(f"❌ Installation of {version} failed GPU test")
                    # Uninstall the failed version
                    subprocess.check_call(["uv", "remove", "onnxruntime-gpu"], stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Installation error for {version}: {e}")

        # If GPU versions fail, try CPU version
        logger.info("GPU versions failed, trying CPU version for Python 3.13")
        try:
            subprocess.check_call(["uv", "add", "onnxruntime"])
            logger.info("✅ Installed CPU-only version for Python 3.13")
            return test_onnx_installation()
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install CPU version: {e}")
            return False

    # For Python < 3.13, try GPU versions
    versions_to_try = [
        "onnxruntime-gpu==1.17.0",  # Compatible with many CUDA versions
        "onnxruntime-gpu==1.16.3",  # Compatible with CUDA 11.8
        "onnxruntime-gpu==1.15.1",  # Compatible with CUDA 11.8
        "onnxruntime-gpu==1.14.1",  # Compatible with CUDA 11.6
    ]

    # Try installing each version until one works
    for version in versions_to_try:
        try:
            logger.info(f"Attempting to install {version}")
            subprocess.check_call(["uv", "add", version])

            # Test if the installation works
            if test_onnx_installation():
                logger.info(f"✅ Successfully installed compatible version: {version}")
                return True
            else:
                logger.warning(f"❌ Installation of {version} failed GPU test")
                # Uninstall the failed version
                subprocess.check_call(["uv", "remove", "onnxruntime-gpu"], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Installation error for {version}: {e}")

    # If all attempts fail, install CPU version as fallback
    logger.info("All GPU versions failed, installing CPU version as fallback")
    try:
        subprocess.check_call(["uv", "add", "onnxruntime"])
        logger.info("✅ Successfully installed CPU version as fallback")
        return test_onnx_installation()
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install CPU fallback: {e}")
        return False

if __name__ == "__main__":
    logger.info("Running ONNX Runtime CUDA compatibility fix script")
    success = fix_onnx_compatibility()
    if success:
        logger.info("Successfully fixed ONNX Runtime compatibility issues")
        sys.exit(0)
    else:
        logger.warning("Could not enable GPU acceleration, but CPU version should work")
        sys.exit(1)
