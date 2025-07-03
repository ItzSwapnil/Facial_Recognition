"""
Main Facial Recognition System
==============================

Integrates all components for real-time facial recognition with alerts.
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from .config import Config, get_config, load_config, save_config
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer
from .alert_system import AlertSystem
from .camera_handler import CameraHandler


class FacialRecognitionSystem:
    """
    Main facial recognition system integrating all components.
    
    Features:
    - Real-time face detection and recognition
    - Multi-threaded processing
    - Live video display with annotations
    - Alert system integration
    - Performance monitoring
    - Configuration management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the facial recognition system."""
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector = FaceDetector(self.config)
        self.recognizer = FaceRecognizer(self.config)
        self.alert_system = AlertSystem(self.config)
        self.camera_handler = CameraHandler(self.config)
        
        # Processing control
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.display_thread: Optional[threading.Thread] = None
        
        # Current processing data
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        self.frame_lock = threading.Lock()
        
        # Performance tracking
        self.total_processing_time = 0.0
        self.processed_frames = 0
        self.start_time = time.time()
        
        # Console for rich output
        self.console = Console()
        
        self.logger.info("Facial Recognition System initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.config.logs_dir / "system.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _process_frames(self) -> None:
        """Main frame processing loop."""
        frame_skip_counter = 0
        
        while self.is_running:
            try:
                # Get frame from camera
                frame = self.camera_handler.get_frame(timeout=0.1)
                if frame is None:
                    continue
                
                # Skip frames for performance if configured
                frame_skip_counter += 1
                if frame_skip_counter < self.config.performance.frame_skip:
                    continue
                frame_skip_counter = 0
                
                start_time = cv2.getTickCount()
                
                # Resize frame for faster processing if configured
                if self.config.performance.resize_factor != 1.0:
                    height, width = frame.shape[:2]
                    new_width = int(width * self.config.performance.resize_factor)
                    new_height = int(height * self.config.performance.resize_factor)
                    processing_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    processing_frame = frame
                
                # Detect faces
                face_locations = self.detector.detect_faces(processing_frame)
                
                # Scale face locations back if frame was resized
                if self.config.performance.resize_factor != 1.0:
                    scale_factor = 1.0 / self.config.performance.resize_factor
                    face_locations = [
                        (int(x * scale_factor), int(y * scale_factor),
                         int(w * scale_factor), int(h * scale_factor))
                        for x, y, w, h in face_locations
                    ]
                
                # Recognize faces
                recognition_results = []
                if face_locations:
                    recognition_results = self.recognizer.recognize_faces(
                        frame, face_locations, method="basic"
                    )
                
                # Process recognition results and trigger alerts
                self._process_recognition_results(recognition_results)
                
                # Update current data for display
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.current_detections = recognition_results
                
                # Update performance tracking
                end_time = cv2.getTickCount()
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                self.total_processing_time += processing_time
                self.processed_frames += 1
                
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}")
                time.sleep(0.1)
    
    def _process_recognition_results(self, results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> None:
        """Process recognition results and trigger appropriate alerts."""
        for name, confidence, location in results:
            if name == "Unknown":
                # Unknown person detected
                if confidence > 0.5:  # Only alert for high-confidence unknowns
                    self.alert_system.trigger_alert(
                        "unknown", "Unknown Person", confidence,
                        {"location": location}
                    )
            else:
                # Known person recognized
                self.alert_system.trigger_alert(
                    "recognition", name, confidence,
                    {"location": location}
                )
    
    def _display_video(self) -> None:
        """Display video with face detection annotations."""
        while self.is_running:
            try:
                with self.frame_lock:
                    if self.current_frame is None:
                        time.sleep(0.1)
                        continue
                    
                    display_frame = self.current_frame.copy()
                    detections = self.current_detections.copy()
                
                # Draw face annotations
                for name, confidence, (x, y, w, h) in detections:
                    # Draw bounding box
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"{name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(display_frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(display_frame, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add system info
                info_text = f"FPS: {self.camera_handler.current_fps:.1f} | Faces: {len(detections)}"
                cv2.putText(display_frame, info_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Facial Recognition System", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_system()
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    self.logger.info(f"Snapshot saved: {filename}")
                elif key == ord('c'):
                    # Clear alert cooldowns
                    self.alert_system.clear_cooldowns()
                    self.logger.info("Alert cooldowns cleared")
                
            except Exception as e:
                self.logger.error(f"Display error: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def start_system(self, video_source: Any = None, show_video: bool = True) -> bool:
        """
        Start the facial recognition system.
        
        Args:
            video_source: Video source (camera index, file, or stream URL)
            show_video: Whether to display video window
        
        Returns:
            True if system started successfully
        """
        if self.is_running:
            self.logger.warning("System is already running")
            return True
        
        self.logger.info("Starting Facial Recognition System...")
        
        # Start camera capture
        if not self.camera_handler.start_capture(video_source):
            self.logger.error("Failed to start camera capture")
            return False
        
        # Start processing
        self.is_running = True
        self.start_time = time.time()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        # Start display thread if requested
        if show_video:
            self.display_thread = threading.Thread(target=self._display_video, daemon=True)
            self.display_thread.start()
        
        self.logger.info("Facial Recognition System started successfully")
        return True
    
    def stop_system(self) -> None:
        """Stop the facial recognition system."""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Facial Recognition System...")
        
        self.is_running = False
        
        # Stop camera
        self.camera_handler.stop_capture()
        
        # Wait for threads to finish
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=5)
        
        if self.display_thread is not None:
            self.display_thread.join(timeout=5)
        
        self.logger.info("Facial Recognition System stopped")
    
    def add_person(self, name: str, image_path: str) -> bool:
        """Add a new person to the recognition database."""
        return self.recognizer.add_face(name, image_path)
    
    def remove_person(self, name: str) -> bool:
        """Remove a person from the recognition database."""
        return self.recognizer.remove_face(name)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        runtime = time.time() - self.start_time
        avg_processing_time = self.total_processing_time / max(1, self.processed_frames)
        
        return {
            'runtime_seconds': runtime,
            'processed_frames': self.processed_frames,
            'average_processing_time': avg_processing_time,
            'processing_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
            'camera_stats': self.camera_handler.get_performance_stats(),
            'detector_stats': self.detector.get_performance_stats(),
            'recognizer_stats': self.recognizer.get_performance_stats(),
            'alert_stats': self.alert_system.get_alert_stats(),
            'known_people': self.recognizer.get_known_people()
        }
    
    def print_stats(self) -> None:
        """Print system statistics in a formatted table."""
        stats = self.get_system_stats()
        
        # Create main stats table
        table = Table(title="Facial Recognition System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        
        # General stats
        table.add_row("Runtime", f"{stats['runtime_seconds']:.1f} seconds")
        table.add_row("Processed Frames", str(stats['processed_frames']))
        table.add_row("Processing FPS", f"{stats['processing_fps']:.1f}")
        table.add_row("Camera FPS", f"{stats['camera_stats']['current_fps']:.1f}")
        table.add_row("Known People", str(len(stats['known_people'])))
        table.add_row("Total Alerts", str(stats['alert_stats']['total_alerts']))
        
        self.console.print(table)
        
        # Known people
        if stats['known_people']:
            people_table = Table(title="Known People")
            people_table.add_column("Name", style="green")
            
            for person in stats['known_people']:
                people_table.add_row(person)
            
            self.console.print(people_table)
    
    def run_console_interface(self) -> None:
        """Run interactive console interface."""
        self.console.print("[bold green]Facial Recognition System Console[/bold green]")
        self.console.print("Commands: start, stop, stats, add, remove, quit")
        
        while True:
            try:
                command = typer.prompt("Enter command").strip().lower()
                
                if command == "start":
                    video_source = typer.prompt("Video source (press Enter for default camera)", default="0")
                    if video_source.isdigit():
                        video_source = int(video_source)
                    self.start_system(video_source)
                
                elif command == "stop":
                    self.stop_system()
                
                elif command == "stats":
                    self.print_stats()
                
                elif command == "add":
                    name = typer.prompt("Person name")
                    image_path = typer.prompt("Image path")
                    if self.add_person(name, image_path):
                        self.console.print(f"[green]Added {name} successfully[/green]")
                    else:
                        self.console.print(f"[red]Failed to add {name}[/red]")
                
                elif command == "remove":
                    name = typer.prompt("Person name")
                    if self.remove_person(name):
                        self.console.print(f"[green]Removed {name} successfully[/green]")
                    else:
                        self.console.print(f"[red]Failed to remove {name}[/red]")
                
                elif command in ["quit", "exit"]:
                    self.stop_system()
                    break
                
                else:
                    self.console.print("[red]Unknown command[/red]")
                    
            except KeyboardInterrupt:
                self.stop_system()
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_system()
