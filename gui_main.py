"""
Main entry point for Ultra-Modern Face Recognition System with GUI
===================================================================

Launches the GUI for the face recognition system
Supports PyQt6 if available, falls back to Tkinter if not
"""

import sys
import logging
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent
sys.path.append(str(src_path))

# Ensure logs directory exists
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'gui_application.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Launch the face recognition system with GUI"""
    # Try to import and use PyQt6 GUI
    try:
        import PyQt6
        # Try importing the actual GUI module
        try:
            from src.face_recognition.ui.qt_gui import main as gui_main
            logger.info("Using PyQt6 GUI interface")
            gui_main()
        except ImportError as e:
            logger.error(f"Failed to import Qt GUI module: {e}")
            raise ImportError("PyQt6 is installed but Qt GUI module could not be loaded")
    except ImportError as e:
        # Fall back to Tkinter GUI which is always available
        logger.info("PyQt6 not available. Using Tkinter GUI instead.")
        try:
            from src.face_recognition.ui.tk_gui import main as tk_main
            tk_main()
        except ImportError as inner_e:
            logger.error(f"Failed to import Tkinter GUI module: {inner_e}")
            print("Error: Could not load any GUI interface. Please check installation.")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}")
