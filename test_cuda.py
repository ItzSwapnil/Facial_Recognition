"""
Test CUDA availability and performance for facial recognition system
"""

import cv2
import numpy as np
import time
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator():
    print("=" * 70)

def test_opencv_cuda():
    """Test if OpenCV can use CUDA"""
    print_separator()
    print("TESTING OPENCV CUDA SUPPORT")
    print_separator()

    # Check OpenCV version
    print(f"OpenCV version: {cv2.__version__}")

    # Get build information
    build_info = cv2.getBuildInformation()

    # Check for CUDA in build info
    cuda_section = None
    if "CUDA:" in build_info:
        cuda_section = build_info.split("CUDA:")[1].split("\n")[0].strip()

    has_cuda_build = cuda_section is not None and "YES" in cuda_section
    print(f"OpenCV built with CUDA support: {has_cuda_build}")

    if has_cuda_build:
        cuda_version = "Unknown"
        if "CUDA Version:" in build_info:
            cuda_version = build_info.split("CUDA Version:")[1].split("\n")[0].strip()
        print(f"CUDA Version in OpenCV: {cuda_version}")

    # Check if CUDA module is available
    has_cuda_module = hasattr(cv2, 'cuda')
    print(f"OpenCV CUDA module available: {has_cuda_module}")

    if has_cuda_module:
        # Check CUDA device count
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA-enabled devices: {device_count}")

            if device_count > 0:
                # Get device info
                for i in range(device_count):
                    props = cv2.cuda.getDevice()
                    name = cv2.cuda.getDeviceName(i)
                    print(f"CUDA Device {i}: {name}")

                # Try to create CUDA objects
                try:
                    stream = cv2.cuda.Stream()
                    print("✅ Successfully created CUDA Stream")

                    # Try a simple operation with GPU
                    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

                    print("Testing GPU vs CPU performance...")

                    # CPU processing time
                    start = time.time()
                    gray_cpu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur_cpu = cv2.GaussianBlur(gray_cpu, (5, 5), 0)
                    cpu_time = time.time() - start
                    print(f"CPU processing time: {cpu_time:.4f} seconds")

                    # GPU processing time
                    start = time.time()
                    gpu_img = cv2.cuda_GpuMat(img)
                    gray_gpu = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                    blur_gpu = cv2.cuda.GaussianBlur(gray_gpu, (5, 5), 0)
                    result = blur_gpu.download()
                    gpu_time = time.time() - start
                    print(f"GPU processing time: {gpu_time:.4f} seconds")

                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                    print(f"GPU speedup: {speedup:.2f}x")

                    if speedup > 1.2:
                        print("✅ GPU acceleration is working and providing significant speedup")
                    else:
                        print("⚠️ GPU is available but not providing significant speedup")

                except Exception as e:
                    print(f"❌ CUDA operations failed: {e}")
            else:
                print("❌ No CUDA-enabled devices found")
        except Exception as e:
            print(f"❌ Error checking CUDA devices: {e}")

    return has_cuda_build and has_cuda_module and cv2.cuda.getCudaEnabledDeviceCount() > 0

def test_dnn_cuda():
    """Test if OpenCV DNN module can use CUDA"""
    print_separator()
    print("TESTING DNN CUDA ACCELERATION")
    print_separator()

    # Check if the DNN module has CUDA backend
    has_cuda_backend = hasattr(cv2.dnn, 'DNN_BACKEND_CUDA')
    print(f"DNN CUDA backend available: {has_cuda_backend}")

    has_cuda_target = hasattr(cv2.dnn, 'DNN_TARGET_CUDA')
    print(f"DNN CUDA target available: {has_cuda_target}")

    if has_cuda_backend and has_cuda_target:
        # Try to load a model with CUDA
        try:
            # Find an ONNX model file
            models_dir = Path("data/models")
            model_files = list(models_dir.glob("*.onnx"))

            if model_files:
                model_path = model_files[0]
                print(f"Testing with model: {model_path}")

                # Try with CPU backend first as baseline
                print("Testing CPU inference...")
                net_cpu = cv2.dnn.readNetFromONNX(str(model_path))
                net_cpu.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net_cpu.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

                # Create test input
                input_size = (640, 480)
                dummy_input = np.random.random((1, 3, input_size[1], input_size[0])).astype(np.float32)

                # Time CPU inference
                start = time.time()
                net_cpu.setInput(dummy_input)
                cpu_out = net_cpu.forward()
                cpu_time = time.time() - start
                print(f"CPU inference time: {cpu_time:.4f} seconds")

                # Try with CUDA backend
                print("Testing CUDA inference...")
                try:
                    net_cuda = cv2.dnn.readNetFromONNX(str(model_path))
                    net_cuda.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net_cuda.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                    # Time CUDA inference
                    start = time.time()
                    net_cuda.setInput(dummy_input)
                    cuda_out = net_cuda.forward()
                    cuda_time = time.time() - start
                    print(f"CUDA inference time: {cuda_time:.4f} seconds")

                    speedup = cpu_time / cuda_time if cuda_time > 0 else 0
                    print(f"CUDA speedup: {speedup:.2f}x")

                    if speedup > 1.2:
                        print("✅ DNN CUDA acceleration is working and providing significant speedup")
                        return True
                    else:
                        print("⚠️ DNN CUDA is available but not providing significant speedup")
                        return False

                except Exception as e:
                    print(f"❌ DNN CUDA inference failed: {e}")
                    return False
            else:
                print("❌ No model files found for testing")
                return False
        except Exception as e:
            print(f"❌ Error in DNN CUDA test: {e}")
            return False
    else:
        print("❌ DNN CUDA backend or target not available")
        return False

def suggest_cuda_optimizations(opencv_cuda_working, dnn_cuda_working):
    """Suggest optimizations based on CUDA test results"""
    print_separator()
    print("CUDA OPTIMIZATION RECOMMENDATIONS")
    print_separator()

    if opencv_cuda_working:
        print("✅ OpenCV CUDA support is working")

        if dnn_cuda_working:
            print("✅ DNN CUDA acceleration is working")
            print("\nRecommendations:")
            print("1. Update onnx_helper.py to prioritize OpenCV's CUDA backend")
            print("2. Configure face detector and recognizer to use CUDA acceleration")
            print("3. Consider batch processing for even better GPU utilization")
        else:
            print("⚠️ DNN CUDA acceleration is not working effectively")
            print("\nRecommendations:")
            print("1. Check CUDA compatibility with your models")
            print("2. Try different CUDA versions or OpenCV builds")
            print("3. Fall back to OpenCV's general CUDA acceleration for image processing")
    else:
        print("❌ OpenCV CUDA support is not working")
        print("\nRecommendations:")
        print("1. Install CUDA toolkit compatible with your system")
        print("2. Reinstall OpenCV with CUDA support (cv2 wheel with CUDA)")
        print("3. Configure environment variables properly:")
        print("   - CUDA_PATH should point to your CUDA installation")
        print("   - Add CUDA bin directory to PATH")

if __name__ == "__main__":
    print("\n🚀 CUDA ACCELERATION TEST 🚀\n")

    opencv_cuda = test_opencv_cuda()
    print()
    dnn_cuda = test_dnn_cuda()

    suggest_cuda_optimizations(opencv_cuda, dnn_cuda)
