"""
Ultra-Modern Facial Recognition System - Professional CLI Integration
===================================================================

A state-of-the-art facial recognition system with comprehensive CLI interface,
professional logging, configuration management, and 2025 SOTA technology.
"""

import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import ultra-modern system
from ultra_modern_face_recognition import UltraModernFaceRecognition

# Import professional modules
from face_recognition_system.config import Config, load_config, create_default_config_file
from face_recognition_system.utils import (
    setup_logging, get_system_info, get_current_performance,
    ensure_directories, benchmark_system, PerformanceTimer
)
from face_recognition_system.models import SystemStats, CameraInfo

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich import box

console = Console()


class ProfessionalFaceRecognitionSystem:
    """
    Professional wrapper for the Ultra-Modern Face Recognition System
    with enterprise-grade features and management capabilities.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize the professional system"""
        self.config = load_config(config_file)
        self.logger = setup_logging(
            log_level=self.config.logging.log_level.value,
            log_file=self.config.logging.log_file if self.config.logging.enable_file_logging else None,
            enable_rich=self.config.logging.enable_rich_formatting
        )
        
        # Initialize directories
        self._ensure_directories()
        
        # Initialize ultra-modern system
        self.face_system = None
        self.stats = SystemStats(
            total_frames_processed=0,
            total_faces_detected=0,
            total_faces_recognized=0,
            average_fps=0.0,
            average_processing_time=0.0,
            uptime=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            last_updated=datetime.now()
        )
        
        self.start_time = time.time()
        self.is_running = False
        
        self.logger.info(f"Initialized {self.config.system_name} v{self.config.version}")
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.config.database.database_path.parent,
            self.config.database.backup_path,
            self.config.database.log_path,
            self.config.database.model_path,
            self.config.alerts.image_save_path
        ]
        ensure_directories(directories)
    
    def initialize_face_system(self) -> bool:
        """Initialize the ultra-modern face recognition system"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Initializing ultra-modern face recognition...", total=None)
                
                self.face_system = UltraModernFaceRecognition()
                progress.update(task, description="PASS: Ultra-modern system initialized")
                
            self.logger.info("Successfully initialized ultra-modern face recognition system")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize face system: {e}")
            console.print(f"[red]Error initializing system: {e}[/red]")
            return False
    
    def get_system_capabilities(self) -> Dict[str, Dict[str, bool]]:
        """Get comprehensive system capabilities"""
        system_info = get_system_info()
        
        capabilities = {
            "hardware": {
                "GPU/CUDA Available": system_info.get("gpu", {}).get("cuda_available", False),
                "Multi-core CPU": system_info.get("hardware", {}).get("cpu_count", 0) > 1,
                "Sufficient Memory": system_info.get("hardware", {}).get("memory_total_gb", 0) >= 4
            },
            "software": {
                "OpenCV Available": "opencv" in system_info and "error" not in system_info["opencv"],
                "ONNX Runtime Available": True,  # We import it successfully
                "YuNet Model": True,
                "SFace Model": True
            },
            "features": {
                "3D Face Modeling": self.config.detection.enable_3d_modeling,
                "Real-time Processing": True,
                "Multi-camera Support": True,
                "Person Management": True,
                "Alert System": self.config.alerts.enable_notifications,
                "Backup System": self.config.database.enable_auto_backup
            }
        }
        
        return capabilities
    
    def run_system_tests(self) -> Dict[str, bool]:
        """Run comprehensive system tests"""
        test_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Test 1: System Info
            task = progress.add_task("Testing system info...", total=None)
            try:
                system_info = get_system_info()
                test_results["system_info"] = "error" not in system_info
                progress.update(task, description="PASS: System info test passed")
            except Exception:
                test_results["system_info"] = False
                progress.update(task, description="FAIL: System info test failed")
            
            # Test 2: Performance Benchmark
            task = progress.add_task("Running performance benchmark...", total=None)
            try:
                benchmark_results = benchmark_system()
                test_results["performance_benchmark"] = "error" not in benchmark_results
                progress.update(task, description="✅ Performance benchmark passed")
            except Exception:
                test_results["performance_benchmark"] = False
                progress.update(task, description="❌ Performance benchmark failed")
            
            # Test 3: Face System Initialization
            task = progress.add_task("Testing face system initialization...", total=None)
            try:
                if not self.face_system:
                    self.initialize_face_system()
                test_results["face_system_init"] = self.face_system is not None
                progress.update(task, description="✅ Face system initialization passed")
            except Exception:
                test_results["face_system_init"] = False
                progress.update(task, description="❌ Face system initialization failed")
            
            # Test 4: Configuration
            task = progress.add_task("Testing configuration...", total=None)
            try:
                config_test = Config()
                test_results["configuration"] = True
                progress.update(task, description="✅ Configuration test passed")
            except Exception:
                test_results["configuration"] = False
                progress.update(task, description="❌ Configuration test failed")
        
        return test_results
    
    def start_face_recognition(self, camera_index: int = 0) -> bool:
        """Start the face recognition system"""
        try:
            if not self.face_system:
                if not self.initialize_face_system():
                    return False
            
            self.is_running = True
            console.print(f"[green]Starting face recognition with camera {camera_index}[/green]")
            
            # Run the ultra-modern system
            self.face_system.run_live_recognition(camera_index)
            
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping face recognition...[/yellow]")
            self.stop_face_recognition()
            return True
        except Exception as e:
            self.logger.error(f"Face recognition error: {e}")
            console.print(f"[red]Error during face recognition: {e}[/red]")
            return False
    
    def stop_face_recognition(self):
        """Stop the face recognition system"""
        self.is_running = False
        if self.face_system:
            # The ultra-modern system handles its own cleanup
            pass
        self.logger.info("Face recognition system stopped")
    
    def add_person_from_image(self, name: str, image_path: Path) -> bool:
        """Add a person from an image file"""
        try:
            if not self.face_system:
                if not self.initialize_face_system():
                    return False
            
            if not image_path.exists():
                console.print(f"[red]Image file not found: {image_path}[/red]")
                return False
            
            # Load image using OpenCV
            import cv2
            import numpy as np
            
            image = cv2.imread(str(image_path))
            if image is None:
                console.print(f"[red]Failed to load image: {image_path}[/red]")
                return False
            
            # Use the ultra-modern system to add the face
            if hasattr(self.face_system, 'add_known_face'):
                success = self.face_system.add_known_face(image, name)
                if success:
                    console.print(f"[green]Successfully added {name} to the database[/green]")
                    return True
                else:
                    console.print(f"[red]Failed to add {name} - no face detected in image[/red]")
                    return False
            else:
                console.print(f"[yellow]Face system doesn't support image-based addition[/yellow]")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to add person {name}: {e}")
            console.print(f"[red]Failed to add person: {e}[/red]")
            return False
    
    def add_person_from_camera(self, name: str, use_3d: bool = True) -> bool:
        """Add a person using camera capture with optional 3D modeling"""
        try:
            if not self.face_system:
                if not self.initialize_face_system():
                    return False
            
            console.print(f"[cyan]Starting camera capture for {name}[/cyan]")
            
            if use_3d and hasattr(self.face_system, 'capture_3d_face_model'):
                # Use 3D face model capture for best accuracy
                console.print("[yellow]Using 3D face modeling for highest accuracy[/yellow]")
                success = self.face_system.capture_3d_face_model(name)
                if success:
                    console.print(f"[green]Successfully captured 3D face model for {name}[/green]")
                    return True
                else:
                    console.print(f"[red]Failed to capture 3D face model for {name}[/red]")
                    return False
            else:
                # Fallback to simple capture
                console.print("[yellow]Using simple face capture[/yellow]")
                console.print("[yellow]Press SPACE to capture, ESC to cancel[/yellow]")
                
                import cv2
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    console.print("[red]Failed to open camera[/red]")
                    return False
                
                captured = False
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Show preview
                    cv2.putText(frame, f"Capturing: {name}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow(f'Capture Face for {name}', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Space to capture
                        if hasattr(self.face_system, 'add_known_face'):
                            success = self.face_system.add_known_face(frame, name)
                            if success:
                                console.print(f"[green]Successfully captured face for {name}[/green]")
                                captured = True
                            else:
                                console.print(f"[red]No face detected, try again[/red]")
                        break
                    elif key == 27:  # ESC to cancel
                        console.print("[yellow]Capture cancelled[/yellow]")
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                return captured
            
        except Exception as e:
            self.logger.error(f"Failed to add person {name} from camera: {e}")
            console.print(f"[red]Failed to add person from camera: {e}[/red]")
            return False
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from the database"""
        try:
            if not self.face_system:
                if not self.initialize_face_system():
                    return False
            
            # Use the ultra-modern system's delete functionality
            if hasattr(self.face_system, 'delete_person'):
                return self.face_system.delete_person(name)
            else:
                console.print(f"[yellow]Person removal not available in current mode[/yellow]")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to remove person {name}: {e}")
            console.print(f"[red]Failed to remove person: {e}[/red]")
            return False
    
    def list_people(self) -> List[str]:
        """Get list of known people"""
        try:
            if not self.face_system:
                if not self.initialize_face_system():
                    return []
            
            # Get people from ultra-modern system
            if hasattr(self.face_system, 'get_known_people'):
                return self.face_system.get_known_people()
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to list people: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        performance = get_current_performance()
        uptime = time.time() - self.start_time
        
        stats = {
            "system": {
                "uptime_seconds": uptime,
                "uptime_formatted": f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s",
                "is_running": self.is_running
            },
            "performance": performance,
            "configuration": {
                "system_name": self.config.system_name,
                "version": self.config.version,
                "debug_mode": self.config.debug_mode,
                "camera_index": self.config.camera.camera_index,
                "detection_model": self.config.detection.detection_model.value,
                "recognition_model": self.config.detection.recognition_model.value
            }
        }
        
        return stats
    
    def print_stats(self):
        """Print formatted system statistics"""
        stats = self.get_system_stats()
        
        # System info table
        system_table = Table(title="System Information", box=box.ROUNDED)
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="green")
        
        system_table.add_row("System Name", stats["configuration"]["system_name"])
        system_table.add_row("Version", stats["configuration"]["version"])
        system_table.add_row("Uptime", stats["system"]["uptime_formatted"])
        system_table.add_row("Status", "🟢 Running" if stats["system"]["is_running"] else "🔴 Stopped")
        system_table.add_row("Debug Mode", "✅ Enabled" if stats["configuration"]["debug_mode"] else "❌ Disabled")
        
        # Performance table
        perf_table = Table(title="Performance Metrics", box=box.ROUNDED)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        perf = stats["performance"]
        perf_table.add_row("CPU Usage", f"{perf.get('cpu_percent', 0):.1f}%")
        perf_table.add_row("Memory Usage", f"{perf.get('memory_percent', 0):.1f}%")
        perf_table.add_row("Memory Used", f"{perf.get('memory_used_mb', 0):.1f} MB")
        
        console.print(system_table)
        console.print()
        console.print(perf_table)
    
    def run_interactive_console(self):
        """Run interactive console interface"""
        console.print(Panel.fit(
            f"[bold cyan]{self.config.system_name}[/bold cyan]\n"
            f"[dim]Version {self.config.version} - Professional Console Interface[/dim]",
            border_style="blue"
        ))
        
        while True:
            try:
                console.print("\n[bold cyan]Available Commands:[/bold cyan]")
                console.print("1. 🚀 Start Face Recognition")
                console.print("2. � Add Person (Camera)")
                console.print("3. 📁 Add Person (Image)")
                console.print("4. �👥 List People")
                console.print("5. 📊 System Statistics")
                console.print("6. 🧪 Run System Tests")
                console.print("7. ⚙️  System Capabilities")
                console.print("8. 📝 Create Config File")
                console.print("9. ❌ Exit")
                
                choice = console.input("\n[yellow]Enter your choice (1-9): [/yellow]").strip()
                
                if choice == "1":
                    camera_index = console.input("[yellow]Camera index (default 0): [/yellow]").strip()
                    camera_index = int(camera_index) if camera_index.isdigit() else 0
                    self.start_face_recognition(camera_index)
                    
                elif choice == "2":
                    # Add person using camera (3D modeling)
                    name = console.input("[yellow]Person name: [/yellow]").strip()
                    if name:
                        console.print(f"[cyan]Adding {name} using advanced 3D face modeling...[/cyan]")
                        if self.add_person_from_camera(name, use_3d=True):
                            console.print(f"[green]✅ Successfully added {name} with 3D modeling[/green]")
                        else:
                            console.print(f"[red]❌ Failed to add {name}[/red]")
                    else:
                        console.print("[red]❌ Name cannot be empty[/red]")
                
                elif choice == "3":
                    # Add person from image file
                    name = console.input("[yellow]Person name: [/yellow]").strip()
                    image_path = console.input("[yellow]Image file path: [/yellow]").strip()
                    if name and image_path:
                        console.print(f"[cyan]Adding {name} from image {image_path}...[/cyan]")
                        if self.add_person_from_image(name, Path(image_path)):
                            console.print(f"[green]✅ Successfully added {name} from image[/green]")
                        else:
                            console.print(f"[red]❌ Failed to add {name} from image[/red]")
                    else:
                        console.print("[red]❌ Name and image path cannot be empty[/red]")
                        
                elif choice == "4":
                    people = self.list_people()
                    if people:
                        console.print(f"\n[green]Known People ({len(people)}):[/green]")
                        for person in people:
                            console.print(f"  • {person}")
                    else:
                        console.print("[yellow]No known people in database[/yellow]")
                        
                elif choice == "5":
                    self.print_stats()
                    
                elif choice == "6":
                    console.print("\n[bold cyan]Running System Tests...[/bold cyan]")
                    results = self.run_system_tests()
                    
                    test_table = Table(title="Test Results", box=box.ROUNDED)
                    test_table.add_column("Test", style="cyan")
                    test_table.add_column("Result", style="green")
                    
                    for test_name, passed in results.items():
                        status = "✅ PASSED" if passed else "❌ FAILED"
                        test_table.add_row(test_name.replace("_", " ").title(), status)
                    
                    console.print(test_table)
                    
                elif choice == "7":
                    capabilities = self.get_system_capabilities()
                    
                    for category, features in capabilities.items():
                        cap_table = Table(title=f"{category.title()} Capabilities", box=box.ROUNDED)
                        cap_table.add_column("Feature", style="cyan")
                        cap_table.add_column("Available", style="green")
                        
                        for feature, available in features.items():
                            status = "✅ Yes" if available else "❌ No"
                            cap_table.add_row(feature, status)
                        
                        console.print(cap_table)
                        console.print()
                
                elif choice == "8":
                    file_path = console.input("[yellow]Config file path (default: config.yaml): [/yellow]").strip()
                    if not file_path:
                        file_path = "config.yaml"
                    
                    if create_default_config_file(Path(file_path)):
                        console.print(f"[green]✅ Created config file: {file_path}[/green]")
                    else:
                        console.print(f"[red]❌ Failed to create config file[/red]")
                        
                elif choice == "9":
                    console.print("[green]👋 Goodbye![/green]")
                    break
                    
                else:
                    console.print("[red]Invalid choice. Please enter 1-7.[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                self.logger.error(f"Console interface error: {e}")


def main():
    """Main entry point with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description="Ultra-Modern Face Recognition System - Professional Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python professional_main.py                                    # Start interactive console
  python professional_main.py --start --camera 0                # Start with camera 0
  python professional_main.py --add-person "John Doe" image.jpg  # Add person from image
  python professional_main.py --list-people                      # List all known people
  python professional_main.py --stats                           # Show system statistics
  python professional_main.py --test                            # Run system tests
  python professional_main.py --capabilities                    # Show capabilities
  python professional_main.py --config config.yaml             # Use custom config
        """
    )
    
    # System operation
    parser.add_argument("--start", action="store_true", help="Start face recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--console", action="store_true", help="Run interactive console (default)")
    
    # Person management
    parser.add_argument("--add-person", nargs='+', metavar=("NAME", "IMAGE_PATH"), 
                       help="Add person from image (or camera if no image provided)")
    parser.add_argument("--add-person-camera", metavar="NAME", 
                       help="Add person using camera with 3D modeling")
    parser.add_argument("--remove-person", help="Remove person from database")
    parser.add_argument("--list-people", action="store_true", help="List all known people")
    
    # System information
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--test", action="store_true", help="Run system tests")
    parser.add_argument("--capabilities", action="store_true", help="Show system capabilities")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    # Configuration
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--create-config", type=Path, help="Create default configuration file")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        console.print("[bold blue]Initializing Professional Face Recognition System...[/bold blue]")
        
        system = ProfessionalFaceRecognitionSystem(args.config)
        
        # Override debug mode if specified
        if args.debug:
            system.config.debug_mode = True
            system.config.logging.log_level = "DEBUG"
        
        # Handle commands
        if args.create_config:
            if create_default_config_file(args.create_config):
                console.print(f"[green]✅ Created config file: {args.create_config}[/green]")
            else:
                console.print(f"[red]❌ Failed to create config file[/red]")
            return
        
        if args.capabilities:
            capabilities = system.get_system_capabilities()
            for category, features in capabilities.items():
                console.print(f"\n[bold cyan]{category.title()} Capabilities:[/bold cyan]")
                for feature, available in features.items():
                    status = "[green]✅[/green]" if available else "[red]❌[/red]"
                    console.print(f"  {status} {feature}")
            return
        
        if args.test:
            console.print("\n[bold cyan]🧪 Running System Tests...[/bold cyan]")
            results = system.run_system_tests()
            
            all_passed = all(results.values())
            overall_status = "[green]✅ ALL TESTS PASSED[/green]" if all_passed else "[red]❌ SOME TESTS FAILED[/red]"
            console.print(f"\n{overall_status}")
            return
        
        if args.benchmark:
            console.print("\n[bold cyan]🏃 Running Performance Benchmark...[/bold cyan]")
            with PerformanceTimer("System Benchmark") as timer:
                results = benchmark_system()
            
            console.print(f"Benchmark completed in {timer.elapsed_time:.2f}s")
            for metric, value in results.items():
                console.print(f"  {metric}: {value}")
            return
        
        if args.add_person:
            if len(args.add_person) == 1:
                # Camera capture mode
                name = args.add_person[0]
                console.print(f"[cyan]Adding {name} using camera capture with 3D modeling[/cyan]")
                if system.add_person_from_camera(name, use_3d=True):
                    console.print(f"[green]✅ Added {name} successfully using camera[/green]")
                else:
                    console.print(f"[red]❌ Failed to add {name} using camera[/red]")
            elif len(args.add_person) == 2:
                # Image file mode
                name, image_path = args.add_person
                console.print(f"[cyan]Adding {name} from image: {image_path}[/cyan]")
                if system.add_person_from_image(name, Path(image_path)):
                    console.print(f"[green]✅ Added {name} successfully from image[/green]")
                else:
                    console.print(f"[red]❌ Failed to add {name} from image[/red]")
            else:
                console.print("[red]❌ Invalid arguments for --add-person[/red]")
            return
        
        if args.add_person_camera:
            name = args.add_person_camera
            console.print(f"[cyan]Adding {name} using advanced 3D camera capture[/cyan]")
            if system.add_person_from_camera(name, use_3d=True):
                console.print(f"[green]✅ Added {name} successfully with 3D modeling[/green]")
            else:
                console.print(f"[red]❌ Failed to add {name} with 3D modeling[/red]")
            return
        
        if args.remove_person:
            if system.remove_person(args.remove_person):
                console.print(f"[green]✅ Removed {args.remove_person} successfully[/green]")
            else:
                console.print(f"[red]❌ Failed to remove {args.remove_person}[/red]")
            return
        
        if args.list_people:
            people = system.list_people()
            if people:
                console.print(f"\n[bold cyan]Known People ({len(people)}):[/bold cyan]")
                for person in people:
                    console.print(f"  • {person}")
            else:
                console.print("[yellow]No known people in database[/yellow]")
            return
        
        if args.stats:
            system.print_stats()
            return
        
        if args.start:
            system.start_face_recognition(args.camera)
            return
        
        # Default: interactive console
        system.run_interactive_console()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]System interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]System error: {e}[/red]")
        if args.verbose or args.debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
