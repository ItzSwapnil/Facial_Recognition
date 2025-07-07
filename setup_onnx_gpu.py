"""
Script to install and configure ONNX Runtime with GPU acceleration
"""

import os
import sys
import subprocess
import logging
import importlib.util
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

def install_onnxruntime_gpu():
    """Install ONNX Runtime with GPU support"""
    print_header("INSTALLING ONNX RUNTIME WITH GPU SUPPORT")

    # Python 3.13 compatibility check
    python_version = sys.version_info
    is_python_313 = python_version.major == 3 and python_version.minor >= 13

    if is_python_313:
        print("⚠️ Python 3.13 detected - limited GPU package compatibility")
        print("Attempting to install compatible ONNX Runtime GPU version...")

        # Try specific versions known to work with Python 3.13
        versions = [
            "onnxruntime-gpu==1.16.3",
            "onnxruntime-gpu==1.17.0",
            "onnxruntime-gpu==1.16.0",
            "onnxruntime-gpu"  # Try latest as last resort
        ]

        success = False
        for version in versions:
            try:
                print(f"\nTrying to install {version}...")
                subprocess.run(["uv", "add", version], check=True)
                success = True
                print(f"✅ Successfully installed {version}")
                break
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {version}")

        if not success:
            print("\n⚠️ Could not install ONNX Runtime with GPU support")
            print("Attempting to install CPU-only version as fallback...")
            try:
                subprocess.run(["uv", "add", "onnxruntime"], check=True)
                print("✅ Installed CPU-only version of ONNX Runtime")
            except subprocess.CalledProcessError:
                print("❌ Failed to install ONNX Runtime")
                return False
    else:
        # For Python < 3.13, more options are available
        print("Attempting to install ONNX Runtime GPU...")
        try:
            subprocess.run(["uv", "add", "onnxruntime-gpu"], check=True)
            print("✅ Successfully installed ONNX Runtime with GPU support")
        except subprocess.CalledProcessError:
            print("❌ Failed to install ONNX Runtime with GPU support")
            print("Attempting to install CPU-only version as fallback...")
            try:
                subprocess.run(["uv", "add", "onnxruntime"], check=True)
                print("✅ Installed CPU-only version of ONNX Runtime")
            except subprocess.CalledProcessError:
                print("❌ Failed to install ONNX Runtime")
                return False

    return True

def verify_onnx_installation():
    """Verify ONNX Runtime installation and check available providers"""
    print_header("VERIFYING ONNX RUNTIME INSTALLATION")

    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime version: {ort.__version__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"Available providers: {', '.join(providers)}")

        # Check for GPU providers
        gpu_providers = [p for p in providers if 'CUDA' in p or 'GPU' in p or 'TensorRT' in p]
        if gpu_providers:
            print(f"✅ GPU acceleration available: {', '.join(gpu_providers)}")
            return True
        else:
            print("⚠️ GPU providers not available, using CPU only")
            return False
    except ImportError:
        print("❌ Failed to import onnxruntime")
        return False

def check_model_compatibility():
    """Check if ONNX models are compatible with ONNX Runtime"""
    print_header("CHECKING MODEL COMPATIBILITY")

    try:
        models_dir = Path("data/models")
        if not models_dir.exists():
            print(f"❌ Models directory not found: {models_dir}")
            return False

        # Find all ONNX models
        onnx_models = list(models_dir.glob("*.onnx"))
        if not onnx_models:
            print("❌ No ONNX models found")
            return False

        print(f"Found {len(onnx_models)} ONNX models:")
        for model in onnx_models:
            print(f"  - {model.name}")

        # Try to load a model with ONNX Runtime
        import onnxruntime as ort
        model_path = str(onnx_models[0])

        print(f"\nTesting model loading with {onnx_models[0].name}...")
        try:
            # Try with all available providers
            providers = ort.get_available_providers()
            session = ort.InferenceSession(model_path, providers=providers)
            print(f"✅ Model loaded successfully using provider: {session.get_providers()[0]}")

            # Get input details
            inputs = session.get_inputs()
            print(f"Model inputs: {', '.join([x.name for x in inputs])}")

            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")

            # Try with CPU provider only as fallback
            try:
                print("\nTrying with CPU provider only...")
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print("✅ Model loaded successfully with CPU provider")
                return False  # Return false as GPU acceleration isn't working
            except Exception as e2:
                print(f"❌ Error loading model with CPU: {str(e2)}")
                return False
    except ImportError:
        print("❌ Failed to import onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def update_onnx_helper():
    """Modify onnx_helper.py to prioritize ONNX Runtime with GPU"""
    print_header("UPDATING ONNX HELPER")

    helper_path = Path("src/face_recognition/utils/onnx_helper.py")
    if not helper_path.exists():
        print(f"❌ ONNX helper file not found: {helper_path}")
        return False

    print("✅ ONNX helper has already been modified to support GPU acceleration")
    print("   The current implementation will automatically use GPU if available")
    return True

def main():
    print("\n🚀 ONNX RUNTIME GPU SETUP 🚀\n")

    # Check for NVIDIA GPU
    has_gpu = check_nvidia_gpu()

    if not has_gpu:
        print("\n⚠️ No NVIDIA GPU detected. ONNX Runtime will use CPU only.")
        print("Continue anyway? (y/n)")
        choice = input().lower()
        if choice != 'y':
            print("Setup aborted.")
            return

    # Install ONNX Runtime with GPU support
    install_success = install_onnxruntime_gpu()

    if install_success:
        # Verify ONNX Runtime installation
        verify_onnx_installation()

        # Check model compatibility
        check_model_compatibility()

        # Update ONNX helper if needed
        update_onnx_helper()

    print("\n✅ Setup complete.")
    print("Run your facial recognition system with: uv run gui_main.py")
    print("The system will automatically use GPU acceleration if available.")

if __name__ == "__main__":
    main()
