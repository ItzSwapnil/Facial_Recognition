"""
GPU Acceleration Setup Guide for Facial Recognition System
This script will guide you through setting up GPU acceleration
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_nvidia_gpu():
    """Check if NVIDIA GPU is present in the system"""
    print_header("CHECKING NVIDIA GPU")

    try:
        # Try using nvidia-smi to detect NVIDIA GPU
        result = subprocess.run(["nvidia-smi"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)

        if result.returncode == 0:
            # Extract GPU info
            output = result.stdout
            print("✅ NVIDIA GPU detected:")

            # Display the GPU model line
            for line in output.split('\n'):
                if '|' in line and any(x in line.lower() for x in ['geforce', 'quadro', 'tesla', 'rtx', 'gtx']):
                    print(f"   {line.strip()}")
                    break
            return True
        else:
            print("❌ No NVIDIA GPU detected with nvidia-smi")
            print("   Make sure NVIDIA drivers are installed if you have an NVIDIA GPU")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False

def check_cuda_installation():
    """Check if CUDA toolkit is installed"""
    print_header("CHECKING CUDA INSTALLATION")

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        print(f"✅ CUDA_PATH environment variable found: {cuda_path}")

        # Check if the directory exists
        if os.path.exists(cuda_path):
            print(f"✅ CUDA installation directory exists: {cuda_path}")

            # Check if nvcc exists
            nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
            if os.path.exists(nvcc_path):
                try:
                    result = subprocess.run([nvcc_path, "--version"],
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE,
                                          text=True)
                    if result.returncode == 0:
                        cuda_version = result.stdout.strip().split("release ")[-1].split(",")[0]
                        print(f"✅ CUDA Toolkit version: {cuda_version}")
                        return True, cuda_version
                    else:
                        print("❌ nvcc found but failed to get version")
                except Exception as e:
                    print(f"❌ Error checking nvcc: {e}")
            else:
                print(f"❌ nvcc not found at {nvcc_path}")
        else:
            print(f"❌ CUDA installation directory does not exist: {cuda_path}")
    else:
        print("❌ CUDA_PATH environment variable not found")

    # Try to find CUDA through nvcc in PATH
    try:
        result = subprocess.run(["nvcc", "--version"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode == 0:
            cuda_version = result.stdout.strip().split("release ")[-1].split(",")[0]
            print(f"✅ CUDA Toolkit found in PATH, version: {cuda_version}")
            return True, cuda_version
        else:
            print("❌ nvcc found in PATH but failed to get version")
    except FileNotFoundError:
        print("❌ nvcc not found in PATH")

    return False, None

def install_cuda_compatible_opencv():
    """Install CUDA-compatible OpenCV for Python"""
    print_header("INSTALLING CUDA-COMPATIBLE OPENCV")

    # First check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")

    # Check if Python version is 3.13
    is_python_313 = python_version.startswith("3.13")
    if is_python_313:
        print("⚠️ Python 3.13 detected - limited CUDA-compatible packages available")

        # Check if any compatible OpenCV versions are installed
        try:
            print("\nAttempting to install CUDA-compatible OpenCV...")
            print("This may take a few minutes...")

            # Try installing opencv-contrib-python with CUDA support
            subprocess.run([sys.executable, "-m", "uv", "pip", "install",
                            "--force-reinstall",
                            "opencv-contrib-python"],
                          check=True)
            print("\n✅ OpenCV installation completed")
            return True
        except Exception as e:
            print(f"❌ Failed to install CUDA-compatible OpenCV: {e}")
            return False
    else:
        # For older Python versions, more options are available
        print("Attempting to install CUDA-compatible OpenCV...")

        try:
            # Try installing opencv-contrib-python with CUDA support
            subprocess.run([sys.executable, "-m", "uv", "pip", "install",
                            "--force-reinstall",
                            "opencv-contrib-python"],
                          check=True)
            print("\n✅ OpenCV installation completed")
            return True
        except Exception as e:
            print(f"❌ Failed to install CUDA-compatible OpenCV: {e}")
            return False

def verify_opencv_cuda():
    """Verify OpenCV CUDA support after installation"""
    print_header("VERIFYING OPENCV CUDA SUPPORT")

    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")

        # Check build information for CUDA
        build_info = cv2.getBuildInformation()

        # Check if CUDA is in the build info
        if "CUDA:" in build_info and "YES" in build_info.split("CUDA:")[1].split("\n")[0]:
            print("✅ OpenCV built with CUDA support")

            # Extract CUDA version
            if "CUDA Version:" in build_info:
                cuda_version = build_info.split("CUDA Version:")[1].split("\n")[0].strip()
                print(f"✅ OpenCV CUDA version: {cuda_version}")

            # Check CUDA modules
            if hasattr(cv2, 'cuda'):
                print("✅ OpenCV CUDA module available")

                # Check CUDA device count
                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                print(f"CUDA-enabled devices: {device_count}")

                if device_count > 0:
                    print("✅ CUDA devices detected by OpenCV")
                    return True
                else:
                    print("❌ No CUDA devices detected by OpenCV")
            else:
                print("❌ OpenCV CUDA module not available")
        else:
            print("❌ OpenCV not built with CUDA support")
    except ImportError:
        print("❌ Failed to import OpenCV")
    except Exception as e:
        print(f"❌ Error verifying OpenCV CUDA: {e}")

    return False

def setup_fallback_acceleration():
    """Configure fallback acceleration options"""
    print_header("CONFIGURING FALLBACK ACCELERATION")

    # Enable OpenCL if available
    try:
        import cv2
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            if cv2.ocl.useOpenCL():
                print("✅ OpenCL acceleration enabled as fallback")
                return True
            else:
                print("❌ Failed to enable OpenCL acceleration")
        else:
            print("❌ OpenCL not available")
    except Exception as e:
        print(f"❌ Error setting up OpenCL: {e}")

    print("⚠️ Using CPU-only processing")
    return False

def generate_recommendations(has_nvidia_gpu, cuda_installed, opencv_cuda_working):
    """Generate recommendations based on test results"""
    print_header("RECOMMENDATIONS")

    if not has_nvidia_gpu:
        print("Your system does not have an NVIDIA GPU or has driver issues:")
        print("1. If you have an NVIDIA GPU, download and install the latest drivers")
        print("   from https://www.nvidia.com/Download/index.aspx")
        print("2. If you don't have an NVIDIA GPU, configure the system for CPU-only")
        print("   operation by running: uv run optimize_performance.py")
        return

    if not cuda_installed:
        print("CUDA Toolkit is not installed or not properly configured:")
        print("1. Download and install CUDA Toolkit from")
        print("   https://developer.nvidia.com/cuda-downloads")
        print("2. Add CUDA bin directory to your PATH")
        print("3. Set CUDA_PATH environment variable to point to your CUDA installation")
        return

    if not opencv_cuda_working:
        print("OpenCV with CUDA support is not working:")
        print("1. Try reinstalling OpenCV with CUDA support:")
        print("   uv pip install --force-reinstall opencv-contrib-python")
        print("2. Configure your application to use CPU acceleration:")
        print("   uv run optimize_performance.py")
    else:
        print("✅ Your system is properly configured for CUDA acceleration.")
        print("Make sure your application is using the GPU:")
        print("1. Set cv2.dnn.DNN_BACKEND_CUDA and cv2.dnn.DNN_TARGET_CUDA")
        print("2. Use cv2.cuda modules for image processing operations")
        print("3. Update the ONNX helper to use CUDA-enabled OpenCV")

def main():
    print("\n📊 GPU ACCELERATION SETUP GUIDE 📊\n")

    # Check for NVIDIA GPU
    has_nvidia_gpu = check_nvidia_gpu()

    # Check CUDA installation
    cuda_installed, cuda_version = check_cuda_installation()

    # Set up CUDA-compatible OpenCV if needed
    if has_nvidia_gpu and cuda_installed:
        install_cuda_compatible_opencv()

    # Verify OpenCV CUDA support
    opencv_cuda_working = verify_opencv_cuda()

    # Configure fallback acceleration if needed
    if not opencv_cuda_working:
        setup_fallback_acceleration()

    # Generate recommendations
    generate_recommendations(has_nvidia_gpu, cuda_installed, opencv_cuda_working)

    print("\nSetup process completed.")

if __name__ == "__main__":
    main()
