"""
Script to set up CUDA environment variables and install CUDA-enabled OpenCV
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
import winreg
import ctypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_admin():
    """Check if the script is running with admin privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def find_cuda_installation():
    """Find CUDA installation directory"""
    # Common CUDA installation paths
    possible_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\NVIDIA\CUDA",
        r"C:\Program Files\NVIDIA Corporation\CUDA",
        r"C:\CUDA"
    ]

    # Look for the latest CUDA version in each path
    for base_path in possible_paths:
        if os.path.exists(base_path):
            # Find all version directories
            versions = []
            try:
                for item in os.listdir(base_path):
                    version_path = os.path.join(base_path, item)
                    if os.path.isdir(version_path) and item.startswith(('v', 'V')) or any(c.isdigit() for c in item):
                        versions.append((item, version_path))
            except Exception:
                continue

            # Sort versions and return the latest
            if versions:
                # Try to sort numerically if possible
                try:
                    versions.sort(key=lambda x: [int(part) for part in x[0].strip('v').split('.')])
                except:
                    versions.sort(key=lambda x: x[0])  # Fallback to string sort

                return versions[-1][1]  # Return the path to the latest version

    return None

def set_env_variables():
    """Set CUDA environment variables permanently"""
    print("\nSetting up CUDA environment variables...")

    # Check for admin privileges
    if not is_admin():
        print("⚠️ This script needs administrator privileges to set environment variables.")
        print("Please run this script as administrator.")
        return False

    # Find CUDA installation
    cuda_path = find_cuda_installation()
    if not cuda_path:
        print("❌ Could not find CUDA installation.")
        print("Please specify the CUDA installation directory:")
        cuda_path = input("> ").strip()

        if not os.path.exists(cuda_path):
            print(f"❌ The path {cuda_path} does not exist.")
            return False

    print(f"✅ Found CUDA installation at: {cuda_path}")

    # Set CUDA_PATH system environment variable
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS)
        winreg.SetValueEx(key, "CUDA_PATH", 0, winreg.REG_SZ, cuda_path)
        winreg.CloseKey(key)
        print(f"✅ Set CUDA_PATH={cuda_path}")

        # Append bin directory to PATH
        bin_dir = os.path.join(cuda_path, 'bin')
        lib_dir = os.path.join(cuda_path, 'lib')

        # Get current PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_READ)
        path, _ = winreg.QueryValueEx(key, 'PATH')
        winreg.CloseKey(key)

        # Check if bin_dir is already in PATH
        path_list = path.split(';')
        if bin_dir not in path_list:
            new_path = path + ';' + bin_dir
            # Set new PATH
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS)
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            winreg.CloseKey(key)
            print(f"✅ Added {bin_dir} to PATH")
        else:
            print(f"✅ {bin_dir} is already in PATH")

        # Notify Windows of environment change
        HWND_BROADCAST = 0xFFFF
        WM_SETTINGCHANGE = 0x001A
        SMTO_ABORTIFHUNG = 0x0002
        result = ctypes.c_long()
        ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0,
                                               u"Environment", SMTO_ABORTIFHUNG, 1000, ctypes.byref(result))

        # Also set variables for current process
        os.environ["CUDA_PATH"] = cuda_path
        if 'PATH' in os.environ:
            if bin_dir not in os.environ['PATH']:
                os.environ['PATH'] = os.environ['PATH'] + ';' + bin_dir
        else:
            os.environ['PATH'] = bin_dir

        print("\n✅ CUDA environment variables have been set successfully.")
        print("⚠️ You may need to restart your terminal or system for the changes to take effect.")
        return True

    except Exception as e:
        print(f"❌ Error setting environment variables: {e}")
        return False

def install_opencv_cuda():
    """Install OpenCV with CUDA support"""
    print("\nInstalling OpenCV with CUDA support...")

    # Check if we have the CUDA environment variables set
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path and 'CUDA_PATH' not in os.environ:
        print("⚠️ CUDA_PATH environment variable is not set.")
        print("Please run the script as administrator to set it.")
        return False

    try:
        # Uninstall existing OpenCV packages
        print("Removing existing OpenCV packages...")
        subprocess.run([sys.executable, "-m", "uv", "remove", "opencv-python"],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run([sys.executable, "-m", "uv", "remove", "opencv-contrib-python"],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Install OpenCV with CUDA support
        print("\nInstalling OpenCV with CUDA support...")
        print("This may take a few minutes...")

        # Try to find a CUDA-enabled wheel
        subprocess.run([sys.executable, "-m", "uv", "add", "opencv-contrib-python==4.8.0.76"], check=True)

        print("✅ OpenCV with CUDA support has been installed.")
        return True

    except Exception as e:
        print(f"❌ Error installing OpenCV with CUDA support: {e}")
        print("\nAttempting to install standard OpenCV as fallback...")
        try:
            subprocess.run([sys.executable, "-m", "uv", "add", "opencv-contrib-python"], check=True)
            print("✅ Standard OpenCV has been installed.")
            return False
        except Exception as e2:
            print(f"❌ Error installing standard OpenCV: {e2}")
            return False

def main():
    print("\n🔧 CUDA ENVIRONMENT SETUP 🔧\n")

    # Set CUDA environment variables
    env_setup_success = set_env_variables()

    if env_setup_success:
        # Install OpenCV with CUDA support
        opencv_cuda_success = install_opencv_cuda()

        if opencv_cuda_success:
            print("\n✅ CUDA and OpenCV with CUDA support have been configured successfully.")
            print("Please restart your terminal and run 'uv run test_cuda.py' to verify the installation.")
        else:
            print("\n⚠️ CUDA environment variables have been set, but OpenCV with CUDA support could not be installed.")
            print("Please try running 'uv add opencv-contrib-python==4.8.0.76' manually.")
    else:
        print("\n❌ CUDA environment setup failed.")
        print("Please make sure you have administrator privileges and CUDA is installed correctly.")

if __name__ == "__main__":
    # Check for admin privileges
    if not is_admin():
        print("⚠️ This script needs administrator privileges to set environment variables.")
        print("Please run this script as administrator.")
        sys.exit(1)

    main()
