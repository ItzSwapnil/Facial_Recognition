"""
Cleanup script to remove unnecessary files after removing multi-angle feature
"""

import os
import shutil
from pathlib import Path

def cleanup():
    print("Starting cleanup of unnecessary files...")

    # Define project root
    project_root = Path(__file__).parent

    # 1. Clean up temp_faces folder
    temp_faces_dir = project_root / "temp_faces"
    if temp_faces_dir.exists():
        print(f"Cleaning temp_faces directory: {temp_faces_dir}")
        try:
            # Remove all jpg files
            jpg_count = 0
            for file in temp_faces_dir.glob("*.jpg"):
                file.unlink()
                jpg_count += 1
            print(f"Removed {jpg_count} temporary face images")

            # Keep the directory itself for compatibility
            print("Kept temp_faces directory for compatibility")
        except Exception as e:
            print(f"Error cleaning temp_faces directory: {e}")
    else:
        print("temp_faces directory not found, nothing to clean")

    print("Cleanup completed!")

if __name__ == "__main__":
    cleanup()
