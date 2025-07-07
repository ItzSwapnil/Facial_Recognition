"""
Main application for Ultra-Modern Face Recognition System (2025 SOTA)
===================================================================

Entry point that integrates all modular components
"""

import sys
import logging
from pathlib import Path
from rich.console import Console

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.face_recognition.core.system import UltraModernFaceRecognitionSystem


def main():
    """Main function for the ultra-modern face recognition system"""
    console = Console()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/face_recognition.log'),
            logging.StreamHandler()
        ]
    )

    # Display system banner
    console.print("🚀 Ultra-Modern Face Recognition System", style="bold blue")
    console.print("🔬 2025 State-of-the-Art Technology", style="cyan")
    console.print("📡 YuNet + SFace + ONNX Runtime", style="green")
    console.print("=" * 60, style="white")

    # Initialize system
    system = UltraModernFaceRecognitionSystem()

    while True:
        # Display main menu
        system.ui.display_banner()
        system.ui.display_main_menu()

        try:
            choice = input(f"\n🎯 Enter your choice (1-8) [Current camera: {system.camera_manager.current_camera_index}]: ").strip()

            if choice == '1':
                # Add simple face
                name = input("👤 Enter your name: ").strip()
                if not name:
                    console.print("❌ Name cannot be empty", style="red")
                    continue

                system.capture_simple_face(name)

            elif choice == '2':
                # Add face with 3D model (multiple angles)
                name = input("👤 Enter your name for 3D face model: ").strip()
                if not name:
                    console.print("❌ Name cannot be empty", style="red")
                    continue

                # Get 3D model type
                model_choice = system.ui.display_3d_model_menu()

                if model_choice == '1':
                    # Quick 3D (3 angles)
                    system.capture_3d_face_model(name, quick_mode=True)
                elif model_choice == '2':
                    # Full 3D (5 angles)
                    system.capture_3d_face_model(name, quick_mode=False)
                else:
                    console.print("❌ Invalid choice", style="red")

            elif choice == '3':
                # Start live recognition
                system.run_live_recognition()

            elif choice == '4':
                # View face database
                system.db_manager.display_database_info(system.face_encodings)
                system.ui.display_face_database(system.face_encodings)
                input("\nPress Enter to continue...")

            elif choice == '5':
                # Person management
                system.manage_persons_menu()

            elif choice == '6':
                # Select camera
                system.camera_manager.select_camera()

            elif choice == '7':
                # System information
                system_info = system.get_system_info()
                system.ui.display_system_info(system_info)
                input("\nPress Enter to continue...")

            elif choice == '8':
                # Exit system
                console.print("👋 Exiting Ultra-Modern Face Recognition System", style="blue")
                console.print("Thank you for using our technology!", style="green")
                break

            else:
                console.print("❌ Invalid choice, please try again", style="red")

        except Exception as e:
            console.print(f"❌ Error: {e}", style="bold red")
            logging.error(f"Error in main loop: {e}", exc_info=True)
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logging.error(f"Application crashed: {e}", exc_info=True)
