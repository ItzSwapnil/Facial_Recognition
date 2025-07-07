"""
User Interface components for the Ultra-Modern Face Recognition System
"""

import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

class FaceRecognitionUI:
    """
    User interface handler for face recognition system
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the UI handler

        Args:
            console: Rich console for display
        """
        self.console = console or Console()

    def display_banner(self):
        """Display system banner"""
        self.console.print("🚀 Ultra-Modern Face Recognition System", style="bold blue")
        self.console.print("🔬 2025 State-of-the-Art Technology", style="cyan")
        self.console.print("📡 YuNet + SFace + ONNX Runtime", style="green")
        self.console.print("=" * 60, style="white")

    def display_main_menu(self):
        """Display main menu options"""
        self.console.print("\n📋 Available Options:")
        self.console.print("1. 📸 Add face (simple)")
        self.console.print("2. 🧊 Add face (3D model - multiple angles)")
        self.console.print("3. 🎥 Start live recognition")
        self.console.print("4. 👥 View face database")
        self.console.print("5. 🔧 Person management")
        self.console.print("6. 📹 Select camera")
        self.console.print("7. ⚙️ System information")
        self.console.print("8. ❌ Exit")

    def display_3d_model_menu(self):
        """Display 3D model capture options menu"""
        self.console.print("\n🎯 3D Face Model Options:")
        self.console.print("1. Quick 3D (3 angles: frontal, left, right)")
        self.console.print("2. Full 3D (5 angles: frontal, left, right, up, down)")
        return input("Choose option (1-2): ").strip()

    def display_person_management_menu(self):
        """Display person management menu"""
        self.console.print("\n👥 Person Management", style="bold cyan")
        self.console.print("\n📋 Management Options:")
        self.console.print("1. 🗑️ Delete person")
        self.console.print("2. 🔗 Merge persons")
        self.console.print("3. 📊 View detailed statistics")
        self.console.print("4. ↩️ Back to main menu")
        return input("\n🎯 Enter your choice (1-4): ").strip()

    def display_persons_table(self, person_stats: Dict):
        """Display table of persons in database"""
        if not person_stats:
            self.console.print("📭 No persons in database", style="yellow")
            return

        # Display person table
        table = Table(title="👤 Persons in Database")
        table.add_column("Name", style="cyan")
        table.add_column("Encodings", style="green")
        table.add_column("Angles", style="yellow")
        table.add_column("Avg Confidence", style="magenta")
        table.add_column("Last Updated", style="blue")

        for name, stats in person_stats.items():
            table.add_row(
                name,
                str(stats['count']),
                ", ".join(stats['angles']),
                f"{stats['avg_confidence']:.1%}",
                stats['latest_timestamp'].strftime("%Y-%m-%d")
            )

        self.console.print(table)

    def display_person_details(self, name: str, stats: Dict):
        """Display detailed statistics for a person"""
        self.console.print(f"\n📊 Detailed Statistics for {name}", style="bold cyan")
        self.console.print(f"📸 Total encodings: {stats['count']}")
        self.console.print(f"📐 Captured angles: {', '.join(stats['angles'])}")
        self.console.print(f"🎯 Average confidence: {stats['avg_confidence']:.1%}")
        self.console.print(f"🔍 Average detection score: {stats['avg_detection_score']:.3f}")
        self.console.print(f"📅 Last updated: {stats['latest_timestamp']}")

        # Show individual encodings
        enc_table = Table(title=f"Individual Encodings for {name}")
        enc_table.add_column("ID", style="cyan")
        enc_table.add_column("Angle", style="green")
        enc_table.add_column("Confidence", style="yellow")
        enc_table.add_column("Date", style="magenta")

        for i, face in enumerate(stats['encodings'], 1):
            enc_table.add_row(
                getattr(face, 'unique_id', f'enc_{i}')[:8],
                getattr(face, 'angle_type', 'frontal'),
                f"{face.confidence:.1%}",
                face.timestamp.strftime("%Y-%m-%d %H:%M")
            )

        self.console.print(enc_table)

    def display_system_info(self, system_info: Dict):
        """Display system information table"""
        table = Table(title="🔧 System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Version/Details", style="yellow")

        for component, info in system_info.items():
            status = info.get("status", "Unknown")
            details = info.get("details", "")
            table.add_row(component, status, details)

        self.console.print(table)

    def display_face_database(self, face_encodings: List):
        """Display face database contents"""
        if not face_encodings:
            self.console.print("📭 Face database is empty", style="yellow")
            return

        table = Table(title="👥 Face Database")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Model Used", style="yellow")
        table.add_column("Embedding Size", style="magenta")
        table.add_column("Detection Score", style="blue")
        table.add_column("Added", style="white")

        for i, face in enumerate(face_encodings, 1):
            table.add_row(
                str(i),
                face.person_name,
                face.model_used,
                str(face.embedding_size),
                f"{face.detection_score:.3f}",
                face.timestamp.strftime("%Y-%m-%d %H:%M")
            )

        self.console.print(table)

    def create_progress(self):
        """Create a progress bar for operations"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def draw_recognition_results(self, frame: np.ndarray, results: List[Dict],
                                processing_time: float, fps: float):
        """
        Draw face recognition results on a frame

        Args:
            frame: Input frame to draw on
            results: List of recognition results
            processing_time: Time taken to process the frame
            fps: Current frames per second

        Returns:
            Frame with results drawn on it
        """
        # Draw results with modern styling
        for result in results:
            x, y, w, h = result['box']
            name = result['name']
            confidence = result['confidence']
            model_used = result.get('model_used', 'unknown')
            detection_score = result.get('detection_score', 0.0)

            # Choose color based on recognition and confidence
            if result['recognized']:
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                else:
                    color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for unknown

            # Draw modern bounding box with rounded corners effect
            thickness = 3
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

            # Draw corner markers for modern look
            corner_length = 20
            cv2.line(frame, (x, y), (x + corner_length, y), color, thickness + 1)
            cv2.line(frame, (x, y), (x, y + corner_length), color, thickness + 1)
            cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness + 1)
            cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness + 1)
            cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness + 1)
            cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness + 1)
            cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness + 1)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness + 1)

            # Draw comprehensive label
            if result['recognized']:
                label = f"{name} ({confidence:.1%})"
                sub_label = f"Model: {model_used} | Det: {detection_score:.2f}"
            else:
                label = "Unknown Person"
                sub_label = f"Det: {detection_score:.2f}"

            # Main label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 35),
                        (x + max(label_size[0], 200), y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Sub label
            cv2.putText(frame, sub_label, (x + 5, y - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Add performance overlay
        fps_text = f"FPS: {fps:.1f} | Processing: {processing_time*1000:.1f}ms"
        cv2.putText(frame, fps_text, (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add title overlay
        title = "Ultra-Modern Face Recognition (2025 SOTA)"
        cv2.putText(frame, title, (10, frame.shape[0] - 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
