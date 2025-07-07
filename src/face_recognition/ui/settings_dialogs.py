"""
Dialog settings components for PyQt6-based GUI
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QComboBox, QProgressBar, QMessageBox,
    QFormLayout, QGroupBox, QCheckBox, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QFont

class RecordingProgressDialog(QDialog):
    """Dialog that shows recording progress"""

    cancel_recording = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording in Progress")
        self.setMinimumWidth(400)
        self.setup_ui()

        # Timer for updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_countdown)
        self.remaining_seconds = 0

    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Status message
        self.status_label = QLabel("Recording in progress...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        # Countdown
        self.countdown_label = QLabel("Time remaining: 0 seconds")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.countdown_label)

        # Progress bars
        self.time_progress = QProgressBar()
        self.time_progress.setRange(0, 100)
        self.time_progress.setValue(0)
        layout.addWidget(QLabel("Time progress:"))
        layout.addWidget(self.time_progress)

        self.embedding_progress = QProgressBar()
        self.embedding_progress.setRange(0, 100)
        self.embedding_progress.setValue(0)
        layout.addWidget(QLabel("Embeddings collected:"))
        layout.addWidget(self.embedding_progress)

        # Cancel button
        self.cancel_button = QPushButton("Cancel Recording")
        self.cancel_button.clicked.connect(self.cancel_recording.emit)
        layout.addWidget(self.cancel_button)

    def start_countdown(self, seconds):
        """Start the countdown timer"""
        self.remaining_seconds = seconds
        self.update_countdown()
        self.timer.start(1000)  # Update every second

    def update_countdown(self):
        """Update the countdown display"""
        if self.remaining_seconds > 0:
            self.countdown_label.setText(f"Time remaining: {self.remaining_seconds} seconds")
            self.time_progress.setValue(100 - int(self.remaining_seconds / self.time_progress.maximum() * 100))
            self.remaining_seconds -= 1
        else:
            self.timer.stop()
            self.countdown_label.setText("Recording complete!")
            self.time_progress.setValue(100)

    def update_status(self, status_text):
        """Update the status text"""
        self.status_label.setText(status_text)

    def update_embedding_progress(self, current, maximum):
        """Update the embedding progress bar"""
        self.embedding_progress.setValue(int(min(100, (current / maximum) * 100)))

    def closeEvent(self, event):
        """Handle close event"""
        self.timer.stop()
        self.cancel_recording.emit()
        super().closeEvent(event)


class NotificationSettingsDialog(QDialog):
    """Dialog for configuring notification settings"""

    settings_changed = pyqtSignal(dict)

    def __init__(self, notification_settings, parent=None):
        """Initialize dialog with current notification settings

        Args:
            notification_settings: Dictionary of current notification settings
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Notification Settings")
        self.setMinimumWidth(400)

        # Make a deep copy of the notification settings to avoid direct modification
        self.notification_settings = notification_settings.copy() if notification_settings else {
            'enabled': True,
            'cooldown': 10,
            'sound': True,
            'desktop': True,
            'console': True,
            'min_confidence': 0.7,
            'notification_type': 0,
            'notify_unknown': False
        }

        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create main form layout
        form_layout = QFormLayout()
        layout.addLayout(form_layout)

        # Enable/disable notifications
        self.enable_checkbox = QCheckBox("Enable Notifications")
        self.enable_checkbox.setChecked(self.notification_settings.get('enabled', True))
        form_layout.addRow("Notifications:", self.enable_checkbox)

        # Notification cooldown
        self.cooldown_spinbox = QSpinBox()
        self.cooldown_spinbox.setRange(1, 60)
        self.cooldown_spinbox.setValue(self.notification_settings.get('cooldown', 10))
        self.cooldown_spinbox.setSuffix(" seconds")
        form_layout.addRow("Cooldown Period:", self.cooldown_spinbox)

        # Minimum confidence threshold
        self.confidence_layout = QHBoxLayout()
        self.confidence_label = QLabel("Minimum Confidence:")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        min_confidence = int(self.notification_settings.get('min_confidence', 0.7) * 100)
        self.confidence_slider.setValue(min_confidence)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)

        self.confidence_value_label = QLabel(f"{min_confidence}%")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_value_label.setText(f"{v}%")
        )

        self.confidence_layout.addWidget(self.confidence_slider)
        self.confidence_layout.addWidget(self.confidence_value_label)
        form_layout.addRow("Minimum Confidence:", self.confidence_layout)

        # Create checkboxes group
        group_box = QGroupBox("Notification Types")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)

        # Sound notifications
        self.sound_checkbox = QCheckBox("Sound Notifications")
        self.sound_checkbox.setChecked(self.notification_settings.get('sound', True))
        group_layout.addWidget(self.sound_checkbox)

        # Desktop notifications
        self.desktop_checkbox = QCheckBox("Desktop Notifications")
        self.desktop_checkbox.setChecked(self.notification_settings.get('desktop', True))
        group_layout.addWidget(self.desktop_checkbox)

        # Console notifications
        self.console_checkbox = QCheckBox("Console Notifications")
        self.console_checkbox.setChecked(self.notification_settings.get('console', True))
        group_layout.addWidget(self.console_checkbox)

        # Include unknown faces
        self.unknown_faces_cb = QCheckBox("Notify for Unknown Faces")
        self.unknown_faces_cb.setChecked(self.notification_settings.get('notify_unknown', False))
        group_layout.addWidget(self.unknown_faces_cb)

        layout.addWidget(group_box)

        # Notification type selection
        notify_type_layout = QHBoxLayout()
        notify_type_layout.addWidget(QLabel("Notification Style:"))
        self.notification_type_combo = QComboBox()
        self.notification_type_combo.addItems(["Pop-up", "Status Bar", "Both"])
        self.notification_type_combo.setCurrentIndex(self.notification_settings.get('notification_type', 0))
        notify_type_layout.addWidget(self.notification_type_combo)
        layout.addLayout(notify_type_layout)

        # Test button
        self.test_btn = QPushButton("Test Notification")
        self.test_btn.clicked.connect(self.test_notification)
        layout.addWidget(self.test_btn)

        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

    def save_settings(self):
        """Save the notification settings"""
        # Update settings from UI
        self.notification_settings['enabled'] = self.enable_checkbox.isChecked()
        self.notification_settings['cooldown'] = self.cooldown_spinbox.value()
        self.notification_settings['sound'] = self.sound_checkbox.isChecked()
        self.notification_settings['desktop'] = self.desktop_checkbox.isChecked()
        self.notification_settings['console'] = self.console_checkbox.isChecked()
        self.notification_settings['min_confidence'] = self.confidence_slider.value() / 100.0
        self.notification_settings['notification_type'] = self.notification_type_combo.currentIndex()
        self.notification_settings['notify_unknown'] = self.unknown_faces_cb.isChecked()

        # Emit signal with updated settings
        self.settings_changed.emit(self.notification_settings)
        self.accept()

    def test_notification(self):
        """Send a test notification"""
        # This is just a placeholder - the actual notification is handled by the parent window
        if self.parent():
            try:
                # Try to call parent's show_notification method
                self.parent().show_notification("Test Notification", "This is a test notification from settings dialog", force=True)
            except AttributeError:
                # If parent doesn't have this method, show a message box
                QMessageBox.information(self, "Test Notification", "This is a test notification")


class CameraSettingsDialog(QDialog):
    """Dialog for camera settings"""

    settings_changed = pyqtSignal(dict)

    def __init__(self, camera_settings, available_cameras, parent=None):
        """Initialize dialog with current camera settings

        Args:
            camera_settings: Dictionary of current camera settings
            available_cameras: List of available cameras
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Camera Settings")
        self.setMinimumWidth(450)
        self.camera_settings = camera_settings.copy()
        self.available_cameras = available_cameras
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Camera selection
        camera_group = QGroupBox("Camera Selection")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)

        self.camera_combo = QComboBox()
        for i, camera in enumerate(self.available_cameras):
            self.camera_combo.addItem(f"{camera['name']} ({camera['resolution']})", camera['index'])

        current_index = self.camera_settings.get('camera_index', 0)
        # Find the combo box index that corresponds to the camera index
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == current_index:
                self.camera_combo.setCurrentIndex(i)
                break

        camera_layout.addWidget(self.camera_combo)
        layout.addWidget(camera_group)

        # Resolution settings
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QFormLayout()
        resolution_group.setLayout(resolution_layout)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("320x240 (Low)")
        self.resolution_combo.addItem("640x480 (Medium)")
        self.resolution_combo.addItem("1280x720 (High)")
        self.resolution_combo.addItem("1920x1080 (Full HD)")

        # Set current resolution
        resolution = self.camera_settings.get('resolution', '640x480')
        if resolution == "320x240":
            self.resolution_combo.setCurrentIndex(0)
        elif resolution == "640x480":
            self.resolution_combo.setCurrentIndex(1)
        elif resolution == "1280x720":
            self.resolution_combo.setCurrentIndex(2)
        elif resolution == "1920x1080":
            self.resolution_combo.setCurrentIndex(3)

        resolution_layout.addRow("Resolution:", self.resolution_combo)
        layout.addWidget(resolution_group)

        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

    def save_settings(self):
        """Save the camera settings"""
        # Update settings from UI
        camera_index = self.camera_combo.currentData()

        resolution_map = {
            0: "320x240",
            1: "640x480",
            2: "1280x720",
            3: "1920x1080"
        }
        resolution = resolution_map.get(self.resolution_combo.currentIndex(), "640x480")

        self.camera_settings['camera_index'] = camera_index
        self.camera_settings['resolution'] = resolution

        # Emit signal with updated settings
        self.settings_changed.emit(self.camera_settings)
        self.accept()
