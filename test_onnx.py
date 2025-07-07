import sys
import os
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("Environment variables:")
for key in ['PYTHONPATH', 'PATH']:
    if key in os.environ:
        print(f"  {key}: {os.environ[key]}")

try:
    import onnxruntime as ort
    print(f"\nONNX Runtime is installed: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"ONNX Runtime module location: {ort.__file__}")

    # Check if CUDA is available through ONNX
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        print("CUDA is available for ONNX Runtime! ✅")

        # Get CUDA device info
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 4  # Verbose

        # Create a simple session to test CUDA
        import numpy as np
        x = np.random.randn(3, 3).astype(np.float32)
        try:
            sess = ort.InferenceSession(
                None,
                session_options,
                providers=['CUDAExecutionProvider']
            )
            print("Could initialize a CUDA provider session")
        except Exception as e:
            print(f"Error initializing CUDA session: {str(e)}")
    else:
        print("CUDA is NOT available for ONNX Runtime! ❌")

except ImportError as e:
    print(f"ONNX Runtime import error: {e}")
    print("ONNX Runtime is NOT properly installed in the current environment")

except Exception as e:
    print(f"Error testing ONNX Runtime: {e}")
