#!/usr/bin/env python3
"""
Launcher script for the Facial Recognition System
Handles compatibility issues and provides graceful error handling
"""

import sys
import os
import warnings

def setup_environment():
    """Setup environment to handle Python 3.13 compatibility issues"""
    # Suppress NumPy warnings that are known issues with Python 3.13
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
    warnings.filterwarnings("ignore", message=".*Numpy built with MINGW.*")
    warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
    
    # Set environment variables for better NumPy behavior
    os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning:numpy'
    os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX512F'  # Disable problematic CPU features

def check_dependencies():
    """Check if critical dependencies can be imported"""
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__} loaded (with compatibility warnings suppressed)")
    except Exception as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
        
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} loaded")
    except Exception as e:
        print(f"✗ Failed to import OpenCV: {e}")
        return False
        
    try:
        import face_recognition
        print(f"✓ face_recognition library loaded")
    except Exception as e:
        print(f"✗ Failed to import face_recognition: {e}")
        return False
        
    return True

def main():
    """Main launcher function"""
    print("🚀 Facial Recognition System Launcher")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Some critical dependencies failed to load.")
        print("This is likely due to Python 3.13 compatibility issues.")
        print("The system may still work with limited functionality.")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    print("\n✅ Dependencies loaded successfully!")
    print("\n🎯 Starting Facial Recognition System...")
    
    # Now import and run the main system
    try:
        from main import main as run_main
        run_main()
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")
        print("\nTrying fallback mode...")
        
        try:
            # Try with basic components only
            from src.face_recognition_system.main_system import FacialRecognitionSystem
            from src.face_recognition_system.config import get_config
            
            config = get_config()
            system = FacialRecognitionSystem(config)
            system.run()
        except Exception as e2:
            print(f"❌ Fallback mode also failed: {e2}")
            print("\n🔧 Please check the installation and try again.")
            sys.exit(1)

if __name__ == "__main__":
    main()
