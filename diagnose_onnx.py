"""
Diagnostics tool for ONNX Runtime loading issues on Windows with Python 3.13
"""

import os
import sys
import platform
import importlib.util
import logging
import ctypes
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator():
    print("=" * 70)

def check_environment():
    """Print environment information"""
    print_separator()
    print("ENVIRONMENT INFORMATION")
    print_separator()
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")

    # Python path
    print(f"\nPython executable: {sys.executable}")
    print(f"Python path:")
    for path in sys.path:
        print(f"  {path}")

def find_onnx_files():
    """Find ONNX Runtime files in the environment"""
    print_separator()
    print("ONNX RUNTIME FILES")
    print_separator()

    # Check if we can find the module spec
    ort_spec = importlib.util.find_spec("onnxruntime")
    if ort_spec:
        print(f"ONNX Runtime package found at: {ort_spec.origin}")

        # Get the package directory
        ort_dir = Path(ort_spec.origin).parent
        print(f"ONNX Runtime directory: {ort_dir}")

        # List DLL files
        print("\nDLL files in ONNX Runtime directory:")
        dll_files = list(ort_dir.glob("*.dll"))
        if dll_files:
            for dll in dll_files:
                print(f"  {dll.name}")
        else:
            print("  No DLL files found")

        # Check specifically for the problematic DLL
        problem_dll = ort_dir / "onnxruntime_pybind11_state.dll"
        if problem_dll.exists():
            print(f"\nProblem DLL exists: {problem_dll}")
            # Try to load the DLL directly to see what happens
            try:
                dll = ctypes.CDLL(str(problem_dll))
                print("✅ Successfully loaded DLL directly")
            except Exception as e:
                print(f"❌ Failed to load DLL: {e}")

                # Check if dll dependencies can be loaded
                print("\nChecking DLL dependencies...")
                try:
                    # On Windows, we can use the GetModuleFileName function to check DLL loading
                    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                    kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
                    kernel32.LoadLibraryExW.restype = ctypes.c_void_p

                    handle = kernel32.LoadLibraryExW(str(problem_dll), None, 0)
                    if handle:
                        print("✅ LoadLibraryExW succeeded")
                        kernel32.FreeLibrary(handle)
                    else:
                        error_code = ctypes.get_last_error()
                        print(f"❌ LoadLibraryExW failed with error code: {error_code}")

                        # Error 126 means "The specified module could not be found"
                        if error_code == 126:
                            print("This indicates missing dependencies.")
                except Exception as e:
                    print(f"Error checking dependencies: {e}")
        else:
            print(f"\n❌ Problem DLL does not exist: {problem_dll}")
    else:
        print("❌ ONNX Runtime package not found")

def try_import_onnx():
    """Try to import onnxruntime and report details"""
    print_separator()
    print("ONNX RUNTIME IMPORT TEST")
    print_separator()

    try:
        import onnxruntime as ort
        print(f"✅ Successfully imported onnxruntime {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        print(f"Device: {ort.get_device()}")
    except ImportError as e:
        print(f"❌ Failed to import onnxruntime: {e}")
        print("\nDetailed exception info:")
        import traceback
        traceback.print_exc()

def check_opencv_onnx():
    """Check if OpenCV can use ONNX models directly"""
    print_separator()
    print("OPENCV ONNX SUPPORT")
    print_separator()

    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")

        # Check if dnn module is available
        if hasattr(cv2, 'dnn'):
            print("OpenCV DNN module is available")

            # Try to read a model
            model_path = Path("data/models/yunet_face_detection_2023mar.onnx")
            if model_path.exists():
                print(f"Testing with model: {model_path}")
                try:
                    net = cv2.dnn.readNetFromONNX(str(model_path))
                    print("✅ OpenCV successfully loaded the ONNX model")

                    # Check available backends
                    backends = []
                    targets = []

                    # Check for OpenCV DNN backends
                    if hasattr(cv2.dnn, 'DNN_BACKEND_DEFAULT'):
                        backends.append("DEFAULT")
                    if hasattr(cv2.dnn, 'DNN_BACKEND_OPENCV'):
                        backends.append("OPENCV")
                    if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
                        backends.append("CUDA")

                    # Check for targets
                    if hasattr(cv2.dnn, 'DNN_TARGET_CPU'):
                        targets.append("CPU")
                    if hasattr(cv2.dnn, 'DNN_TARGET_CUDA'):
                        targets.append("CUDA")
                    if hasattr(cv2.dnn, 'DNN_TARGET_OPENCL'):
                        targets.append("OPENCL")

                    print(f"Available DNN backends: {backends}")
                    print(f"Available DNN targets: {targets}")

                except Exception as e:
                    print(f"❌ Failed to load ONNX model with OpenCV: {e}")
            else:
                print(f"❌ Model not found: {model_path}")
        else:
            print("❌ OpenCV DNN module is not available")
    except ImportError as e:
        print(f"❌ Failed to import OpenCV: {e}")

def suggest_fix():
    """Suggest fixes based on diagnostics"""
    print_separator()
    print("SUGGESTED FIXES")
    print_separator()

    print("Based on diagnostics, here are potential fixes:")
    print("1. Use OpenCV's native ONNX support as a fallback")
    print("   - Your logs show OpenCV has native ONNX model support")
    print("2. Try downgrading ONNX Runtime to version 1.15.1 or 1.16.3")
    print("   - These versions may have better compatibility with Python 3.13")
    print("3. For GPU support on Python 3.13, CUDA 11.8 or 12.x is usually required")
    print("4. Modify code to gracefully fall back to CPU inference when GPU fails")
    print("\nRecommended action: Update the src/face_recognition/utils/onnx_helper.py file to")
    print("handle the DLL error gracefully and use OpenCV's ONNX support as fallback.")

if __name__ == "__main__":
    print("\n🔍 ONNX RUNTIME DIAGNOSTICS TOOL 🔍\n")
    check_environment()
    find_onnx_files()
    try_import_onnx()
    check_opencv_onnx()
    suggest_fix()
    print("\nDiagnostics complete.")
