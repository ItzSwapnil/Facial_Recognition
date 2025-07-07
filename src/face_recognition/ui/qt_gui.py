"""
PyQt6-based GUI for the Ultra-Modern Face Recognition System
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for PyQt6 dependency
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QComboBox, QTabWidget, QLineEdit,
        QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar,
        QSplitter, QFrame, QGridLayout, QSpacerItem, QSizePolicy,
        QDialog, QDialogButtonBox, QScrollArea, QTextEdit, QGroupBox,
        QSlider, QCheckBox, QInputDialog, QSpinBox  # Added missing widgets
    )
    from PyQt6.QtGui import QPixmap, QImage, QColor, QPalette, QIcon, QFont
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize, QThread, QRunnable, QThreadPool, QObject

    PYQT6_AVAILABLE = True
except ImportError as e:
    PYQT6_AVAILABLE = False
    if __name__ == '__main__':
        print("ERROR: PyQt6 is not installed. Please install it with: pip install PyQt6")
        print(f"Import error details: {e}")
        print("\nYou should run the application using gui_main.py instead of directly running qt_gui.py")
        sys.exit(1)

from datetime import datetime
import threading
import time
import queue

# Import face recognition system
from src.face_recognition.core.system import UltraModernFaceRecognitionSystem
from src.face_recognition.models.face_models import ModernFaceEncoding
# Import settings dialogs
from src.face_recognition.ui.settings_dialogs import NotificationSettingsDialog, CameraSettingsDialog, RecordingProgressDialog
# Import notification settings manager
from src.face_recognition.utils.notification_settings import NotificationSettings

if not PYQT6_AVAILABLE:
    # Create placeholder class for type checking when PyQt6 is not available
    class VideoThread:
        pass
else:
    class VideoThread(QThread):
        """Thread for video capture and processing"""
        frame_ready = pyqtSignal(np.ndarray)
        recognition_results = pyqtSignal(list, float)
        recording_progress = pyqtSignal(int, int)  # current_embeddings, min_embeddings

        def __init__(self, system: UltraModernFaceRecognitionSystem, camera_index: int = 0, parent=None):
            super().__init__(parent)
            self.system = system
            self.camera_index = camera_index
            self.running = False
            self.mode = "preview"  # "preview", "recognition", "capture"
            self.person_name = ""
            self.angle_type = "frontal"
            self.captured_frame = None
            self.capture_result = False
            self.recording_duration = 20
            self.recording_start_time = 0

        def set_mode(self, mode: str, person_name: str = "", angle_type: str = "frontal"):
            """Set the operating mode of the video thread"""
            self.mode = mode
            self.person_name = person_name
            self.angle_type = angle_type

        def capture_frame(self):
            """Set flag to capture the next frame"""
            self.captured_frame = None
            self.mode = "capture"

        def get_captured_frame(self) -> Optional[np.ndarray]:
            """Get the most recently captured frame"""
            return self.captured_frame

        def run(self):
            """Main thread execution"""
            self.running = True
            cap = cv2.VideoCapture(self.camera_index)

            # Try to set higher resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # FPS calculation variables
            fps_counter = 0
            fps = 0
            start_time = time.time()

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame based on mode
                if self.mode == "recognition":
                    # Perform face recognition
                    process_start = time.time()
                    results = self.system.recognize_faces(frame)
                    process_time = time.time() - process_start

                    # Draw recognition results
                    for result in results:
                        x, y, w, h = result['box']
                        name = result['name']
                        confidence = result['confidence']
                        recognized = result['recognized']

                        # Choose color based on recognition result
                        if recognized:
                            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)  # Green or Yellow
                        else:
                            color = (0, 0, 255)  # Red for unknown

                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        # Draw name and confidence
                        label = f"{name} ({confidence:.1%})" if recognized else "Unknown"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Update FPS
                    fps_counter += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 1.0:  # Update FPS every second
                        fps = fps_counter / elapsed_time
                        fps_counter = 0
                        start_time = time.time()

                    # Draw FPS and processing time
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Processing: {process_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Emit recognition results
                    self.recognition_results.emit(results, fps)

                elif self.mode == "capture":
                    # Store the captured frame
                    self.captured_frame = frame.copy()
                    self.mode = "preview"  # Return to preview mode

                    # Detect faces in the captured frame
                    faces = self.system.face_detector.detect_faces(frame)
                    if len(faces) > 0:
                        # Draw boxes around detected faces
                        for face_data in faces:
                            x, y, w, h = face_data[:4].astype(int)
                            confidence = face_data[14] if len(face_data) > 14 else 1.0
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, f"Face detected ({confidence:.2f})", (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Add face to database if name is provided
                        if self.person_name:
                            self.capture_result = self.system.add_known_face(
                                self.captured_frame,
                                self.person_name,
                                self.angle_type
                            )
                    else:
                        # No face detected
                        self.capture_result = False

                elif self.mode == "preview":
                    # Just show preview with face detection
                    faces = self.system.face_detector.detect_faces(frame)

                    if len(faces) > 0:
                        for face_data in faces:
                            x, y, w, h = face_data[:4].astype(int)
                            confidence = face_data[14] if len(face_data) > 14 else 1.0

                            # Determine head pose if applicable
                            face_region = frame[max(0, y):min(frame.shape[0], y+h),
                                              max(0, x):min(frame.shape[1], x+w)]
                            if face_region.size > 0:
                                pose = self.system.face_detector.detect_head_pose(face_region)

                                # Color based on matching target angle
                                color = (0, 255, 0) if pose == self.angle_type else (0, 255, 255)

                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(frame, f"Detected: {pose}", (x, y-30),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv2.putText(frame, f"Target: {self.angle_type}", (x, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, f"Face detected ({confidence:.2f})", (x, y-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Convert frame to format for UI
                self.frame_ready.emit(frame)

                # Sleep to reduce CPU usage
                time.sleep(0.01)

            cap.release()

        def stop(self):
            """Stop the thread"""
            self.running = False
            self.wait()


class FrameWidget(QLabel):
    """Widget to display video frames"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; color: white;")
        self.setText("Initializing camera...")

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        """Update the widget with a new frame"""
        # Convert the frame from BGR to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a QImage from the frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale the image to fit the widget while maintaining aspect ratio
        pixmap = QPixmap.fromImage(img)

        # Get the widget's size
        widget_width = self.width()
        widget_height = self.height()

        # Scale the pixmap to fill the available space
        scaled_pixmap = pixmap.scaled(widget_width, widget_height,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)

        self.setPixmap(scaled_pixmap)


class PersonManagementDialog(QDialog):
    """Dialog for managing persons in the database"""

    def __init__(self, system: UltraModernFaceRecognitionSystem, parent=None):
        super().__init__(parent)
        self.system = system
        self.setWindowTitle("Person Management")
        self.setMinimumSize(800, 600)

        # Create layout
        layout = QVBoxLayout(self)

        # Create person table
        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Name", "Encodings", "Angles", "Confidence", "Last Updated"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Add table to layout
        layout.addWidget(self.table)

        # Create buttons
        button_layout = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Person")
        self.merge_btn = QPushButton("Merge Persons")
        self.details_btn = QPushButton("View Details")

        button_layout.addWidget(self.delete_btn)
        button_layout.addWidget(self.merge_btn)
        button_layout.addWidget(self.details_btn)
        layout.addLayout(button_layout)

        # Add close button
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Connect signals
        self.delete_btn.clicked.connect(self.delete_person)
        self.merge_btn.clicked.connect(self.merge_persons)
        self.details_btn.clicked.connect(self.view_details)

        # Load data
        self.load_persons()

    def load_persons(self):
        """Load persons from the database"""
        person_stats = self.system.get_person_statistics()

        # Clear table
        self.table.setRowCount(0)

        # Add rows to table
        row = 0
        for name, stats in person_stats.items():
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(str(stats['count'])))
            self.table.setItem(row, 2, QTableWidgetItem(", ".join(stats['angles'])))
            self.table.setItem(row, 3, QTableWidgetItem(f"{stats['avg_confidence']:.1%}"))
            self.table.setItem(row, 4, QTableWidgetItem(stats['latest_timestamp'].strftime("%Y-%m-%d")))
            row += 1

        # Resize columns to content
        self.table.resizeColumnsToContents()

    def delete_person(self):
        """Delete a person from the database"""
        # Get selected row
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a person to delete")
            return

        # Get person name
        person_name = self.table.item(selected_rows[0].row(), 0).text()

        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Deletion",
                                    f"Are you sure you want to delete {person_name}?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            success = self.system.delete_person(person_name)
            if success:
                QMessageBox.information(self, "Success", f"Deleted {person_name}")
                self.load_persons()  # Reload the table
            else:
                QMessageBox.warning(self, "Error", f"Failed to delete {person_name}")

    def merge_persons(self):
        """Merge two persons in the database"""
        # Create merge dialog
        merge_dialog = QDialog(self)
        merge_dialog.setWindowTitle("Merge Persons")

        # Create layout
        layout = QVBoxLayout(merge_dialog)

        # Create source/target selection
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Merge from:"), 0, 0)
        form_layout.addWidget(QLabel("Merge to:"), 1, 0)

        # Get all person names
        person_stats = self.system.get_person_statistics()
        person_names = list(person_stats.keys())

        # Create comboboxes
        source_combo = QComboBox()
        target_combo = QComboBox()

        source_combo.addItems(person_names)
        target_combo.addItems(person_names)

        form_layout.addWidget(source_combo, 0, 1)
        form_layout.addWidget(target_combo, 1, 1)

        layout.addLayout(form_layout)

        # Create buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(merge_dialog.accept)
        button_box.rejected.connect(merge_dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        if merge_dialog.exec() == QDialog.DialogCode.Accepted:
            source = source_combo.currentText()
            target = target_combo.currentText()

            if source == target:
                QMessageBox.warning(self, "Error", "Source and target cannot be the same")
                return

            success = self.system.merge_persons(source, target)
            if success:
                QMessageBox.information(self, "Success", f"Merged {source} into {target}")
                self.load_persons()  # Reload the table
            else:
                QMessageBox.warning(self, "Error", f"Failed to merge persons")

    def view_details(self):
        """View details of a person"""
        # Get selected row
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a person to view")
            return

        # Get person name
        person_name = self.table.item(selected_rows[0].row(), 0).text()

        # Get person statistics
        person_stats = self.system.get_person_statistics()
        if person_name not in person_stats:
            QMessageBox.warning(self, "Error", f"Person {person_name} not found")
            return

        stats = person_stats[person_name]

        # Create details dialog
        details_dialog = QDialog(self)
        details_dialog.setWindowTitle(f"Details for {person_name}")
        details_dialog.setMinimumSize(600, 400)

        # Create layout
        layout = QVBoxLayout(details_dialog)

        # Create info section
        info_group = QGroupBox("Person Information")
        info_layout = QGridLayout(info_group)

        info_layout.addWidget(QLabel("Name:"), 0, 0)
        info_layout.addWidget(QLabel(person_name), 0, 1)

        info_layout.addWidget(QLabel("Total Encodings:"), 1, 0)
        info_layout.addWidget(QLabel(str(stats['count'])), 1, 1)

        info_layout.addWidget(QLabel("Captured Angles:"), 2, 0)
        info_layout.addWidget(QLabel(", ".join(stats['angles'])), 2, 1)

        info_layout.addWidget(QLabel("Average Confidence:"), 3, 0)
        info_layout.addWidget(QLabel(f"{stats['avg_confidence']:.1%}"), 3, 1)

        info_layout.addWidget(QLabel("Average Detection Score:"), 4, 0)
        info_layout.addWidget(QLabel(f"{stats['avg_detection_score']:.3f}"), 4, 1)

        info_layout.addWidget(QLabel("Last Updated:"), 5, 0)
        info_layout.addWidget(QLabel(stats['latest_timestamp'].strftime("%Y-%m-%d %H:%M")), 5, 1)

        layout.addWidget(info_group)

        # Create encodings table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["ID", "Angle", "Confidence", "Date"])

        # Add rows to table
        for i, face in enumerate(stats['encodings']):
            table.insertRow(i)
            unique_id = getattr(face, 'unique_id', f'enc_{i}')[:8]
            table.setItem(i, 0, QTableWidgetItem(unique_id))
            table.setItem(i, 1, QTableWidgetItem(getattr(face, 'angle_type', 'frontal')))
            table.setItem(i, 2, QTableWidgetItem(f"{face.confidence:.1%}"))
            table.setItem(i, 3, QTableWidgetItem(face.timestamp.strftime("%Y-%m-%d %H:%M")))

        # Resize columns to content
        table.resizeColumnsToContents()

        layout.addWidget(table)

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(details_dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        details_dialog.exec()


class SystemInfoDialog(QDialog):
    """Dialog for showing system information"""

    def __init__(self, system: UltraModernFaceRecognitionSystem, parent=None):
        super().__init__(parent)
        self.system = system
        self.setWindowTitle("System Information")
        self.setMinimumSize(600, 400)

        # Create layout
        layout = QVBoxLayout(self)

        # Create info table
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Component", "Status", "Details"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Add table to layout
        layout.addWidget(self.table)

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Load system info
        self.load_system_info()

    def load_system_info(self):
        """Load system information"""
        system_info = self.system.get_system_info()

        # Clear table
        self.table.setRowCount(0)

        # Add rows to table
        row = 0
        for component, info in system_info.items():
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(component))
            self.table.setItem(row, 1, QTableWidgetItem(info['status']))
            self.table.setItem(row, 2, QTableWidgetItem(info['details']))
            row += 1

        # Resize columns to content
        self.table.resizeColumnsToContents()


class ModelCaptureDialog(QDialog):
    """Dialog for capturing a 3D face model"""

    angles_completed = pyqtSignal(list)

    def __init__(self, system: UltraModernFaceRecognitionSystem, video_thread: VideoThread, parent=None):
        super().__init__(parent)
        self.system = system
        self.video_thread = video_thread
        self.current_angle_index = 0
        self.captured_angles = []
        self.person_name = ""
        self.full_3d = True

        # Define angles to capture
        self.angles_full = [
            ("frontal", "Look straight at the camera"),
            ("left_profile", "Turn your head slightly to the left"),
            ("right_profile", "Turn your head slightly to the right"),
            ("up_angle", "Tilt your head slightly up"),
            ("down_angle", "Tilt your head slightly down")
        ]

        self.angles_quick = [
            ("frontal", "Look straight at the camera"),
            ("left_profile", "Turn your head slightly to the left"),
            ("right_profile", "Turn your head slightly to the right")
        ]

        self.angles = self.angles_full

        # Setup UI
        self.setWindowTitle("3D Face Model Capture")
        self.setMinimumSize(800, 600)

        # Create layout
        layout = QVBoxLayout(self)

        # Title label
        self.title_label = QLabel("3D Face Model Capture")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin: 10px;")
        layout.addWidget(self.title_label)

        # Person name and model type
        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("Person Name:"), 0, 0)
        self.name_input = QLineEdit()
        form_layout.addWidget(self.name_input, 0, 1)

        form_layout.addWidget(QLabel("Model Type:"), 1, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Full 3D (5 angles)", "Quick 3D (3 angles)"])
        self.model_combo.currentIndexChanged.connect(self.update_model_type)
        form_layout.addWidget(self.model_combo, 1, 1)

        layout.addLayout(form_layout)

        # Camera preview
        self.preview = FrameWidget()
        layout.addWidget(self.preview)

        # Instruction label
        self.instruction_label = QLabel("Please enter a name and choose model type")
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("font-size: 14pt; margin: 10px;")
        layout.addWidget(self.instruction_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, len(self.angles))
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Status label
        self.status_label = QLabel("Ready to begin capture")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Capture")
        self.start_btn.clicked.connect(self.start_capture)

        self.capture_btn = QPushButton("Capture Angle")
        self.capture_btn.clicked.connect(self.capture_angle)
        self.capture_btn.setEnabled(False)

        self.skip_btn = QPushButton("Skip Angle")
        self.skip_btn.clicked.connect(self.skip_angle)
        self.skip_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.skip_btn)
        layout.addLayout(button_layout)

        # Add close button
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.close_dialog)
        layout.addWidget(self.button_box)

        # Connect video thread signals
        self.video_thread.frame_ready.connect(self.preview.update_frame)

    def update_model_type(self, index: int):
        """Update the model type based on combo box selection"""
        self.full_3d = (index == 0)  # Full 3D is index 0
        self.angles = self.angles_full if self.full_3d else self.angles_quick
        self.progress.setRange(0, len(self.angles))

    def start_capture(self):
        """Start the capture process"""
        self.person_name = self.name_input.text().strip()
        if not self.person_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a person name")
            return

        # Update UI
        self.start_btn.setEnabled(False)
        self.capture_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)
        self.name_input.setEnabled(False)
        self.model_combo.setEnabled(False)

        # Reset capture state
        self.current_angle_index = 0
        self.captured_angles = []

        # Update progress
        self.progress.setValue(0)

        # Start with the first angle
        self.prepare_for_angle(0)

    def prepare_for_angle(self, index: int):
        """Prepare UI for capturing a specific angle"""
        if index >= len(self.angles):
            self.finish_capture()
            return

        angle_type, instruction = self.angles[index]

        # Update video thread with current angle
        self.video_thread.set_mode("preview", self.person_name, angle_type)

        # Update UI
        self.status_label.setText(f"Capturing angle {index+1} of {len(self.angles)}: {angle_type}")
        self.instruction_label.setText(f"Please {instruction}")

    def capture_angle(self):
        """Capture the current angle"""
        if self.current_angle_index >= len(self.angles):
            return

        angle_type, _ = self.angles[self.current_angle_index]

        # Update video thread to capture frame
        self.video_thread.set_mode("capture", self.person_name, angle_type)
        self.video_thread.capture_frame()

        # Wait a moment for capture to complete
        QTimer.singleShot(500, self.process_captured_frame)

    def process_captured_frame(self):
        """Process the captured frame"""
        frame = self.video_thread.get_captured_frame()
        if frame is not None and self.video_thread.capture_result:
            # Capture successful
            angle_type, _ = self.angles[self.current_angle_index]
            self.captured_angles.append(angle_type)
            self.status_label.setText(f"Successfully captured {angle_type}")

            # Move to next angle
            self.current_angle_index += 1
            self.progress.setValue(len(self.captured_angles))

            if self.current_angle_index < len(self.angles):
                # Prepare for next angle
                QTimer.singleShot(1000, lambda: self.prepare_for_angle(self.current_angle_index))
            else:
                # Finished all angles
                self.finish_capture()
        else:
            # Capture failed
            QMessageBox.warning(self, "Capture Failed",
                               "Failed to detect a face. Please position yourself correctly and try again.")
            self.status_label.setText("Capture failed. Please try again.")

    def skip_angle(self):
        """Skip the current angle"""
        if self.current_angle_index >= len(self.angles):
            return

        angle_type, _ = self.angles[self.current_angle_index]
        self.status_label.setText(f"Skipped {angle_type}")

        # Move to next angle
        self.current_angle_index += 1

        if self.current_angle_index < len(self.angles):
            # Prepare for next angle
            QTimer.singleShot(500, lambda: self.prepare_for_angle(self.current_angle_index))
        else:
            # Finished all angles
            self.finish_capture()

    def finish_capture(self):
        """Finish the capture process"""
        if len(self.captured_angles) > 0:
            self.status_label.setText(f"Completed! Captured {len(self.captured_angles)} angles")
            self.instruction_label.setText("3D face model capture complete")

            # Emit signal with captured angles
            self.angles_completed.emit(self.captured_angles)

            # Show completion message
            QMessageBox.information(self, "Capture Complete",
                                   f"Successfully created 3D face model with {len(self.captured_angles)} angles")
        else:
            self.status_label.setText("No angles captured")
            QMessageBox.warning(self, "No Captures", "No angles were captured")

        # Reset UI
        self.start_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.name_input.setEnabled(True)
        self.model_combo.setEnabled(True)

        # Reset video thread mode
        self.video_thread.set_mode("preview")

    def close_dialog(self):
        """Close the dialog"""
        # Make sure to reset video thread mode
        self.video_thread.set_mode("preview")
        self.reject()


class NotificationSettingsDialog(QDialog):
    """Dialog for configuring notification settings"""

    settings_changed = pyqtSignal(dict)

    def __init__(self, settings_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Notification Settings")
        self.setMinimumSize(400, 300)

        # Store settings dictionary
        self.settings = settings_dict.copy() if isinstance(settings_dict, dict) else {
            'enabled': True,
            'cooldown': 10,
            'min_confidence': 0.7,
            'notification_type': 0,
            'notify_unknown': False
        }

        # Create layout
        layout = QVBoxLayout(self)

        # Enable notifications checkbox
        self.enable_notifications_cb = QCheckBox("Enable Notifications")
        self.enable_notifications_cb.setChecked(self.settings.get('enabled', True))
        layout.addWidget(self.enable_notifications_cb)

        # Notification settings group
        settings_group = QGroupBox("Notification Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Confidence threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Minimum Confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(self.settings.get('min_confidence', 0.7) * 100))
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_value_label = QLabel(f"{self.settings.get('min_confidence', 0.7):.0%}")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_value_label.setText(f"{v/100:.0%}"))
        threshold_layout.addWidget(self.confidence_slider)
        threshold_layout.addWidget(self.confidence_value_label)
        settings_layout.addLayout(threshold_layout)

        # Cooldown setting
        cooldown_layout = QHBoxLayout()
        cooldown_layout.addWidget(QLabel("Cooldown Period (seconds):"))
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(1, 60)
        self.cooldown_spinbox.setValue(self.settings.get('cooldown', 10))
        cooldown_layout.addWidget(self.cooldown_spinbox)
        settings_layout.addLayout(cooldown_layout)

        # Notification type
        notification_type_layout = QHBoxLayout()
        notification_type_layout.addWidget(QLabel("Notification Type:"))
        self.notification_type_combo = QComboBox()
        self.notification_type_combo.addItems(["Pop-up", "Status Bar", "Both"])
        self.notification_type_combo.setCurrentIndex(self.settings.get('notification_type', 0))
        notification_type_layout.addWidget(self.notification_type_combo)
        settings_layout.addLayout(notification_type_layout)

        # Include unknown faces
        self.unknown_faces_cb = QCheckBox("Notify for Unknown Faces")
        self.unknown_faces_cb.setChecked(self.settings.get('notify_unknown', False))
        settings_layout.addWidget(self.unknown_faces_cb)

        # Add settings group to main layout
        layout.addWidget(settings_group)

        # Test notification button
        self.test_btn = QPushButton("Test Notification")
        self.test_btn.clicked.connect(self.test_notification)
        layout.addWidget(self.test_btn)

        # Add spacer
        layout.addStretch()

        # Add button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.save_and_close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def test_notification(self):
        """Send a test notification"""
        if self.parent():
            try:
                self.parent().show_notification("Test Notification", "This is a test notification", force=True)
            except AttributeError:
                QMessageBox.information(self, "Test Notification", "This is a test notification")
        else:
            QMessageBox.information(self, "Test Notification", "This is a test notification")

    def save_and_close(self):
        """Save settings and close the dialog"""
        # Get current settings from UI
        settings = {
            "enabled": self.enable_notifications_cb.isChecked(),
            "min_confidence": self.confidence_slider.value() / 100.0,
            "cooldown": self.cooldown_spinbox.value(),
            "notification_type": self.notification_type_combo.currentIndex(),
            "notify_unknown": self.unknown_faces_cb.isChecked()
        }

        # Emit signal with the updated settings
        self.settings_changed.emit(settings)

        # Accept the dialog (close it)
        self.accept()


class MainWindow(QMainWindow):
    """Main window for the Ultra-Modern Face Recognition GUI"""

    def __init__(self):
        super().__init__()

        # Setup window properties
        self.setWindowTitle("Ultra-Modern Face Recognition System")
        self.setMinimumSize(1024, 768)

        # Initialize face recognition system
        self.system = UltraModernFaceRecognitionSystem()

        # Initialize notification settings manager
        self.notification_settings_manager = NotificationSettings()

        # Configure notification settings from loaded settings
        self.enable_notifications = self.notification_settings_manager.get_setting('enabled', True)
        self.notification_min_confidence = self.notification_settings_manager.get_setting('min_confidence', 0.7)
        self.notification_cooldown = self.notification_settings_manager.get_setting('cooldown', 5)
        self.notification_type = self.notification_settings_manager.get_setting('notification_type', 0)
        self.notify_unknown = self.notification_settings_manager.get_setting('notify_unknown', False)

        # Initialize video thread
        self.video_thread = VideoThread(self.system)

        # Setup UI
        self.setup_ui()

        # Start video thread
        self.video_thread.frame_ready.connect(self.video_frame.update_frame)
        self.video_thread.recognition_results.connect(self.update_recognition_results)
        self.video_thread.start()

        # Additional state variables
        self.is_recognition_active = False
        self.recognition_results = []

        # Setup notification timer to prevent notification spam
        self.last_notification_time = {}  # Store last notification time for each person

    def setup_ui(self):
        """Set up the main UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)

        # Create splitter for video and controls
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Video display
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)

        # Video frame display
        self.video_frame = FrameWidget()
        video_layout.addWidget(self.video_frame)

        # Video display size controls
        size_control_layout = QHBoxLayout()
        size_control_layout.addWidget(QLabel("Display Size:"))

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(50, 150)  # 50% to 150% of default size
        self.size_slider.setValue(100)  # Default 100%
        self.size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.size_slider.setTickInterval(25)
        self.size_slider.valueChanged.connect(self.update_display_size)

        self.size_label = QLabel("100%")
        size_control_layout.addWidget(self.size_slider)
        size_control_layout.addWidget(self.size_label)

        video_layout.addLayout(size_control_layout)

        # Video controls
        video_controls = QHBoxLayout()

        self.camera_combo = QComboBox()
        self.populate_camera_list()

        self.camera_btn = QPushButton("Change Camera")
        self.camera_btn.clicked.connect(self.change_camera)

        video_controls.addWidget(QLabel("Camera:"))
        video_controls.addWidget(self.camera_combo)
        video_controls.addWidget(self.camera_btn)

        video_layout.addLayout(video_controls)

        # Add video container to splitter
        splitter.addWidget(video_container)

        # Right side - Controls and info
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)

        # Title and info
        title_label = QLabel("Ultra-Modern Face Recognition")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; margin: 10px;")

        subtitle_label = QLabel("2025 State-of-the-Art Technology")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 12pt; margin: 5px;")

        control_layout.addWidget(title_label)
        control_layout.addWidget(subtitle_label)

        # Operation buttons
        self.start_recognition_btn = QPushButton("Start Live Recognition")
        self.start_recognition_btn.clicked.connect(self.toggle_recognition)
        self.start_recognition_btn.setStyleSheet("font-size: 14pt; padding: 10px;")

        self.add_face_btn = QPushButton("Add Face (Simple)")
        self.add_face_btn.clicked.connect(self.add_face_simple)

        self.add_face_3d_btn = QPushButton("Add Face (3D Model)")
        self.add_face_3d_btn.clicked.connect(self.add_face_3d)

        self.view_db_btn = QPushButton("View Face Database")
        self.view_db_btn.clicked.connect(self.view_database)

        self.manage_persons_btn = QPushButton("Person Management")
        self.manage_persons_btn.clicked.connect(self.manage_persons)

        self.system_info_btn = QPushButton("System Information")
        self.system_info_btn.clicked.connect(self.show_system_info)

        # Add buttons to layout
        control_layout.addWidget(self.start_recognition_btn)

        # Group other buttons
        button_grid = QGridLayout()
        button_grid.addWidget(self.add_face_btn, 0, 0)
        button_grid.addWidget(self.add_face_3d_btn, 0, 1)
        button_grid.addWidget(self.view_db_btn, 1, 0)
        button_grid.addWidget(self.manage_persons_btn, 1, 1)
        button_grid.addWidget(self.system_info_btn, 2, 0, 1, 2)

        control_layout.addLayout(button_grid)

        # Recognition results area
        results_group = QGroupBox("Recognition Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        results_layout.addWidget(self.results_text)
        control_layout.addWidget(results_group)

        # Parameter controls group
        params_group = QGroupBox("Recognition Parameters")
        params_layout = QVBoxLayout(params_group)

        # Confidence threshold slider
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(70)  # Default to 70%
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)

        self.confidence_value = QLabel("70%")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_value.setText(f"{v}%"))

        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value)
        params_layout.addLayout(confidence_layout)

        # Detection threshold slider
        detection_layout = QHBoxLayout()
        detection_layout.addWidget(QLabel("Detection:"))
        self.detection_slider = QSlider(Qt.Orientation.Horizontal)
        self.detection_slider.setRange(1, 100)
        self.detection_slider.setValue(50)  # Default to 50%
        self.detection_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.detection_slider.setTickInterval(10)
        self.detection_slider.valueChanged.connect(self.update_detection_threshold)

        self.detection_value = QLabel("50%")
        self.detection_slider.valueChanged.connect(
            lambda v: self.detection_value.setText(f"{v}%"))

        detection_layout.addWidget(self.detection_slider)
        detection_layout.addWidget(self.detection_value)
        params_layout.addLayout(detection_layout)

        # Recognition quality slider
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(75)  # Default to 75%
        self.quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.quality_slider.setTickInterval(10)
        self.quality_slider.valueChanged.connect(self.update_quality_threshold)

        self.quality_value = QLabel("75%")
        self.quality_slider.valueChanged.connect(
            lambda v: self.quality_value.setText(f"{v}%"))

        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_value)
        params_layout.addLayout(quality_layout)

        control_layout.addWidget(params_group)

        # Notification settings button
        self.notification_settings_btn = QPushButton("Notification Settings")
        self.notification_settings_btn.clicked.connect(self.open_notification_settings)
        control_layout.addWidget(self.notification_settings_btn)

        # Status bar
        self.status_label = QLabel("System ready")
        control_layout.addWidget(self.status_label)

        # Add control container to splitter
        splitter.addWidget(control_container)

        # Set initial splitter sizes
        splitter.setSizes([600, 400])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Status bar
        self.statusBar().showMessage("System ready")

    def populate_camera_list(self):
        """Populate the camera dropdown list"""
        self.camera_combo.clear()

        for cam in self.system.camera_manager.available_cameras:
            self.camera_combo.addItem(f"Camera {cam['index']}: {cam['resolution']}", cam['index'])

        # Set current index
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == self.system.camera_manager.current_camera_index:
                self.camera_combo.setCurrentIndex(i)
                break

    def change_camera(self):
        """Change the camera being used"""
        camera_index = self.camera_combo.currentData()

        # Stop current video thread
        self.video_thread.stop()

        # Update system camera
        self.system.camera_manager.current_camera_index = camera_index

        # Create new video thread
        self.video_thread = VideoThread(self.system, camera_index)
        self.video_thread.frame_ready.connect(self.video_frame.update_frame)
        self.video_thread.recognition_results.connect(self.update_recognition_results)

        # Restore recognition mode if active
        if self.is_recognition_active:
            self.video_thread.set_mode("recognition")

        # Start thread
        self.video_thread.start()

        self.statusBar().showMessage(f"Changed to camera {camera_index}")

    def toggle_recognition(self):
        """Toggle live face recognition"""
        if self.is_recognition_active:
            # Stop recognition
            self.video_thread.set_mode("preview")
            self.is_recognition_active = False
            self.start_recognition_btn.setText("Start Live Recognition")
            self.results_text.clear()
            self.statusBar().showMessage("Recognition stopped")
        else:
            # Start recognition
            self.video_thread.set_mode("recognition")
            self.is_recognition_active = True
            self.start_recognition_btn.setText("Stop Recognition")
            self.statusBar().showMessage("Running live recognition...")

    def update_recognition_results(self, results, fps):
        """Update recognition results display"""
        self.recognition_results = results

        # Clear previous results
        self.results_text.clear()

        # Add FPS info
        self.results_text.append(f"FPS: {fps:.1f}\n")

        # Add recognition results
        for result in results:
            if result['recognized']:
                name = result['name']
                confidence = result['confidence']
                self.results_text.append(f"✅ {name} ({confidence:.1%})")
            else:
                self.results_text.append("❓ Unknown Person")

            # Add details
            self.results_text.append(f"   Model: {result.get('model_used', 'unknown')}")
            self.results_text.append(f"   Score: {result.get('detection_score', 0.0):.3f}\n")

            # Check if notification is enabled and confidence is above threshold
            if self.enable_notifications and result['recognized'] and confidence >= self.notification_min_confidence:
                self.show_notification(f"Recognized: {name}", f"Confidence: {confidence:.1%}")

    def show_notification(self, title, message, force=False):
        """Show a notification (stub function, implement as needed)"""
        current_time = time.time()

        # Check cooldown period
        if not force and title in self.last_notification_time:
            elapsed_time = current_time - self.last_notification_time[title]
            if elapsed_time < self.notification_cooldown:
                return  # Skip notification, still in cooldown

        # Update last notification time
        self.last_notification_time[title] = current_time

        # Show notification based on selected type
        if self.notification_type == 0:  # Pop-up
            QMessageBox.information(self, title, message)
        elif self.notification_type == 1:  # Status Bar
            self.statusBar().showMessage(f"{title}: {message}")
        elif self.notification_type == 2:  # Both
            QMessageBox.information(self, title, message)
            self.statusBar().showMessage(f"{title}: {message}")

    def add_face_simple(self):
        """Add a simple face to the database"""
        # Get person name
        person_name, ok = QInputDialog.getText(self, "Add Face", "Enter person name:")

        if ok and person_name:
            # Create dialog for capture
            capture_dialog = QDialog(self)
            capture_dialog.setWindowTitle("Capture Face")
            capture_dialog.setMinimumSize(800, 600)

            # Create layout
            layout = QVBoxLayout(capture_dialog)

            # Camera preview
            preview = FrameWidget()
            layout.addWidget(preview)

            # Instructions
            instructions = QLabel("Position your face in the camera and press 'Capture'")
            instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
            instructions.setStyleSheet("font-size: 14pt; margin: 10px;")
            layout.addWidget(instructions)

            # Buttons
            button_layout = QHBoxLayout()
            capture_btn = QPushButton("Capture")
            cancel_btn = QPushButton("Cancel")

            button_layout.addWidget(capture_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)

            # Set up video connection
            self.video_thread.frame_ready.connect(preview.update_frame)

            # Define capture action
            def do_capture():
                self.video_thread.set_mode("capture", person_name)
                self.video_thread.capture_frame()

                # Wait a moment for capture
                QTimer.singleShot(1000, check_capture)

            def check_capture():
                if self.video_thread.capture_result:
                    QMessageBox.information(capture_dialog, "Success",
                                          f"Successfully added {person_name} to the database")
                    capture_dialog.accept()
                else:
                    QMessageBox.warning(capture_dialog, "Capture Failed",
                                      "Failed to detect a face. Please try again.")

            # Connect signals
            capture_btn.clicked.connect(do_capture)
            cancel_btn.clicked.connect(capture_dialog.reject)

            # Show dialog
            self.video_thread.set_mode("preview")
            capture_dialog.exec()

            # Disconnect preview
            self.video_thread.frame_ready.disconnect(preview.update_frame)
            self.video_thread.set_mode("preview")

    def add_face_3d(self):
        """Add a 3D face model to the database"""
        # Create and show the 3D model capture dialog
        capture_dialog = ModelCaptureDialog(self.system, self.video_thread, self)
        capture_dialog.exec()

        # Reset video mode
        self.video_thread.set_mode("preview")

    def view_database(self):
        """View the face database"""
        # Create view dialog
        view_dialog = QDialog(self)
        view_dialog.setWindowTitle("Face Database")
        view_dialog.setMinimumSize(800, 600)

        # Create layout
        layout = QVBoxLayout(view_dialog)

        # Create table
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["ID", "Name", "Model", "Angle", "Confidence", "Added"])

        # Add data to table
        for i, face in enumerate(self.system.face_encodings):
            table.insertRow(i)
            unique_id = getattr(face, 'unique_id', f'face_{i}')[:8]

            table.setItem(i, 0, QTableWidgetItem(unique_id))
            table.setItem(i, 1, QTableWidgetItem(face.person_name))
            table.setItem(i, 2, QTableWidgetItem(face.model_used))
            table.setItem(i, 3, QTableWidgetItem(getattr(face, 'angle_type', 'frontal')))
            table.setItem(i, 4, QTableWidgetItem(f"{face.confidence:.1%}"))
            table.setItem(i, 5, QTableWidgetItem(face.timestamp.strftime("%Y-%m-%d %H:%M")))

        # Resize columns to content
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)

        # Add table to layout
        layout.addWidget(table)

        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(view_dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        view_dialog.exec()

    def manage_persons(self):
        """Manage persons in the database"""
        # Create and show person management dialog
        dialog = PersonManagementDialog(self.system, self)
        dialog.exec()

    def show_system_info(self):
        """Show system information"""
        # Create and show system info dialog
        dialog = SystemInfoDialog(self.system, self)
        dialog.exec()

    def open_notification_settings(self):
        """Open the notification settings dialog"""
        try:
            # Get current notification settings from manager
            settings_dict = self.notification_settings_manager.get_all_settings()

            # Create the dialog with notification settings
            dialog = NotificationSettingsDialog(settings_dict, self)

            # Connect settings changed signal to handler
            dialog.settings_changed.connect(self.update_notification_settings)

            # Show dialog
            dialog.exec()
        except Exception as e:
            # Log error and show a message if dialog fails to open
            logger.error(f"Error opening notification settings: {e}")
            QMessageBox.warning(self, "Settings Error",
                              f"Could not open notification settings: {str(e)}")

    def update_notification_settings(self, settings):
        """Update notification settings from dialog"""
        try:
            # Update main window settings
            self.enable_notifications = settings.get('enabled', True)
            self.notification_min_confidence = settings.get('min_confidence', 0.7)
            self.notification_cooldown = settings.get('cooldown', 10)
            self.notification_type = settings.get('notification_type', 0)
            self.notify_unknown = settings.get('notify_unknown', False)

            # Update notification manager settings
            self.notification_settings_manager.update_settings(settings)

            # Save settings to disk
            self.notification_settings_manager.save_settings()

            self.statusBar().showMessage("Notification settings updated")
        except Exception as e:
            logger.error(f"Error updating notification settings: {e}")
            QMessageBox.warning(self, "Settings Error",
                              f"Could not update notification settings: {str(e)}")

    def update_confidence_threshold(self, value: int):
        """Update the confidence threshold based on slider value"""
        threshold = value / 100.0  # Convert to fraction
        self.video_thread.system.confidence_threshold = threshold

        # Update status bar message
        self.statusBar().showMessage(f"Confidence threshold set to {threshold:.2%}")

    def update_detection_threshold(self, value: int):
        """Update the detection threshold based on slider value"""
        threshold = value / 100.0  # Convert to fraction
        self.video_thread.system.detection_threshold = threshold

        # Update status bar message
        self.statusBar().showMessage(f"Detection threshold set to {threshold:.2%}")

    def update_quality_threshold(self, value: int):
        """Update the recognition quality threshold based on slider value"""
        threshold = value / 100.0  # Convert to fraction
        self.video_thread.system.recognition_quality_threshold = threshold

        # Update status bar message
        self.statusBar().showMessage(f"Recognition quality set to {threshold:.2%}")

    def update_display_size(self, value: int):
        """Update the display size percentage"""
        self.size_label.setText(f"{value}%")
        # Here you would add code to actually resize the video display if needed

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop video thread
        if self.video_thread.isRunning():
            self.video_thread.stop()

        # Accept the close event
        event.accept()


def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Set dark theme
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # Create main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
