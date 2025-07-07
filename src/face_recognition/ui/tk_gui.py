"""
Tkinter-based GUI for the Ultra-Modern Face Recognition System
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import threading
import time
import queue
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from typing import Optional, List, Dict
from datetime import datetime

# Import face recognition system
from src.face_recognition.core.system import UltraModernFaceRecognitionSystem


class VideoStream:
    """Thread for video capture and processing"""

    def __init__(self, system, camera_index=0):
        self.system = system
        self.camera_index = camera_index
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.mode = "preview"  # "preview", "recognition", "capture"
        self.person_name = ""
        self.angle_type = "frontal"
        self.captured_frame = None
        self.capture_result = False
        self.recognition_results = []
        self.fps = 0
        self.recording_progress_callback = None  # Callback for updating recording progress

        # Add visual effects settings (parity with Qt version)
        self.apply_effects = False
        self.effect_level = 0.5
        self.effect_type = "none"  # "none", "enhance", "vintage", "cool", "warm"

    def start(self):
        """Start the video stream thread"""
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the video stream thread"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    def set_effect_level(self, level: float):
        """Set effect intensity level"""
        self.effect_level = max(0.0, min(1.0, level))

    def apply_visual_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply visual effect to frame based on current settings"""
        if not self.apply_effects or self.effect_level <= 0:
            return frame

        # Create a copy of the frame to modify
        result = frame.copy()

        # Apply effect based on type and level
        if self.effect_type == "enhance":
            # Enhance contrast and brightness
            alpha = 1.0 + (0.5 * self.effect_level)  # Contrast control
            beta = 10.0 * self.effect_level  # Brightness control
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        elif self.effect_type == "vintage":
            # Sepia effect for vintage look
            kernel = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            sepia = cv2.transform(result, kernel)
            result = cv2.addWeighted(result, 1.0 - self.effect_level, sepia, self.effect_level, 0)

        elif self.effect_type == "cool":
            # Cool blue tint
            b, g, r = cv2.split(result)
            b = cv2.addWeighted(b, 1 + self.effect_level*0.5, b, 0, 0)
            result = cv2.merge([b, g, r])

        elif self.effect_type == "warm":
            # Warm orange/red tint
            b, g, r = cv2.split(result)
            r = cv2.addWeighted(r, 1 + self.effect_level*0.5, r, 0, 0)
            result = cv2.merge([b, g, r])

        return result

    def _update(self):
        """Main thread loop to update frames"""
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
                    self.fps = fps

                # Draw FPS and processing time
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Processing: {process_time*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Store results for UI
                self.recognition_results = results

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

            # Convert to RGB for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Put in queue for UI thread to pick up
            if not self.frame_queue.full():
                self.frame_queue.put(frame_rgb)

            # Sleep to reduce CPU usage
            time.sleep(0.01)

        cap.release()

    def set_mode(self, mode, person_name="", angle_type="frontal"):
        """Set the operating mode of the video thread"""
        self.mode = mode
        self.person_name = person_name
        self.angle_type = angle_type

    def capture_frame(self):
        """Set flag to capture the next frame"""
        self.captured_frame = None
        self.mode = "capture"

    def get_captured_frame(self):
        """Get the most recently captured frame"""
        return self.captured_frame


class FrameViewer(ttk.Frame):
    """Widget to display video frames"""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.photo = None
        self.display_scale = 1.0  # Default display scale

    def update_frame(self, frame):
        """Update the widget with a new frame"""
        h, w = frame.shape[:2]

        # Get canvas dimensions
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            # Canvas not fully initialized yet, use default size
            canvas_w = 640
            canvas_h = 480

        # Calculate scaling factor to fit frame into canvas
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        scale = min(scale_w, scale_h)

        # Resize frame to fit canvas
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Apply display scale
        if self.display_scale != 1.0:
            new_w = int(w * self.display_scale)
            new_h = int(h * self.display_scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Convert to PhotoImage for Tkinter
        self.image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=self.image)

        # Update canvas
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def set_display_scale(self, scale):
        """Set the display scale for the frame viewer"""
        self.display_scale = scale

        # Update the current frame with the new scale
        if self.photo is not None:
            self.update_frame(np.array(self.photo))


class PersonManagementDialog(tk.Toplevel):
    """Dialog for managing persons in the database"""

    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("Person Management")
        self.geometry("800x600")
        self.minsize(800, 600)

        # Make the dialog modal
        self.transient(parent)
        self.grab_set()

        # Create widgets
        self.create_widgets()

        # Load data
        self.load_persons()

        # Wait for window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create the dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create table frame
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create person table
        columns = ('name', 'encodings', 'angles', 'confidence', 'last_updated')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings')
        self.table.heading('name', text='Name')
        self.table.heading('encodings', text='Encodings')
        self.table.heading('angles', text='Angles')
        self.table.heading('confidence', text='Confidence')
        self.table.heading('last_updated', text='Last Updated')

        # Set column widths
        self.table.column('name', width=150)
        self.table.column('encodings', width=70)
        self.table.column('angles', width=200)
        self.table.column('confidence', width=100)
        self.table.column('last_updated', width=150)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)

        # Pack table and scrollbar
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Action buttons
        self.delete_btn = ttk.Button(button_frame, text="Delete Person", command=self.delete_person)
        self.merge_btn = ttk.Button(button_frame, text="Merge Persons", command=self.merge_persons)
        self.details_btn = ttk.Button(button_frame, text="View Details", command=self.view_details)
        self.close_btn = ttk.Button(button_frame, text="Close", command=self.on_close)

        # Pack buttons
        self.delete_btn.pack(side=tk.LEFT, padx=5)
        self.merge_btn.pack(side=tk.LEFT, padx=5)
        self.details_btn.pack(side=tk.LEFT, padx=5)
        self.close_btn.pack(side=tk.RIGHT, padx=5)

    def load_persons(self):
        """Load persons from the database"""
        # Clear table
        for item in self.table.get_children():
            self.table.delete(item)

        # Get person statistics
        person_stats = self.system.get_person_statistics()

        # Add rows to table
        for name, stats in person_stats.items():
            self.table.insert('', tk.END, values=(
                name,
                stats['count'],
                ', '.join(stats['angles']),
                f"{stats['avg_confidence']:.1%}",
                stats['latest_timestamp'].strftime("%Y-%m-%d")
            ))

    def delete_person(self):
        """Delete a person from the database"""
        # Get selected row
        selected_item = self.table.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a person to delete")
            return

        # Get person name
        person_name = self.table.item(selected_item[0])['values'][0]

        # Confirm deletion
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {person_name}?"):
            success = self.system.delete_person(person_name)
            if success:
                messagebox.showinfo("Success", f"Deleted {person_name}")
                self.load_persons()  # Reload the table
            else:
                messagebox.showwarning("Error", f"Failed to delete {person_name}")

    def merge_persons(self):
        """Merge two persons in the database"""
        # Get person statistics
        person_stats = self.system.get_person_statistics()
        person_names = list(person_stats.keys())

        if len(person_names) < 2:
            messagebox.showinfo("Not Enough Persons", "Need at least 2 persons to merge")
            return

        # Create merge dialog
        dialog = tk.Toplevel(self)
        dialog.title("Merge Persons")
        dialog.geometry("400x200")
        dialog.transient(self)
        dialog.grab_set()

        # Frame for form
        form_frame = ttk.Frame(dialog, padding=20)
        form_frame.pack(fill=tk.BOTH, expand=True)

        # Source and target selection
        ttk.Label(form_frame, text="Merge from:").grid(row=0, column=0, sticky=tk.W, pady=5)
        source_var = tk.StringVar()
        source_combo = ttk.Combobox(form_frame, textvariable=source_var, values=person_names)
        source_combo.grid(row=0, column=1, sticky=tk.W+tk.E, pady=5)
        source_combo.current(0)

        ttk.Label(form_frame, text="Merge to:").grid(row=1, column=0, sticky=tk.W, pady=5)
        target_var = tk.StringVar()
        target_combo = ttk.Combobox(form_frame, textvariable=target_var, values=person_names)
        target_combo.grid(row=1, column=1, sticky=tk.W+tk.E, pady=5)
        if len(person_names) > 1:
            target_combo.current(1)
        else:
            target_combo.current(0)

        # Button frame
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)

        def do_merge():
            source = source_var.get()
            target = target_var.get()

            if source == target:
                messagebox.showwarning("Error", "Source and target cannot be the same")
                return

            success = self.system.merge_persons(source, target)
            dialog.destroy()

            if success:
                messagebox.showinfo("Success", f"Merged {source} into {target}")
                self.load_persons()  # Reload the table
            else:
                messagebox.showwarning("Error", "Failed to merge persons")

        ttk.Button(button_frame, text="Merge", command=do_merge).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def view_details(self):
        """View details of a person"""
        # Get selected row
        selected_item = self.table.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a person to view")
            return

        # Get person name
        person_name = self.table.item(selected_item[0])['values'][0]

        # Get person statistics
        person_stats = self.system.get_person_statistics()
        if person_name not in person_stats:
            messagebox.showwarning("Error", f"Person {person_name} not found")
            return

        stats = person_stats[person_name]

        # Create details dialog
        dialog = tk.Toplevel(self)
        dialog.title(f"Details for {person_name}")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.grab_set()

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Person info frame
        info_frame = ttk.LabelFrame(main_frame, text="Person Information", padding=10)
        info_frame.pack(fill=tk.X)

        # Info grid
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill=tk.X)

        # Add info rows
        ttk.Label(info_grid, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_grid, text=person_name).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Total Encodings:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_grid, text=str(stats['count'])).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Captured Angles:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_grid, text=", ".join(stats['angles'])).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Average Confidence:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_grid, text=f"{stats['avg_confidence']:.1%}").grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Average Detection Score:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_grid, text=f"{stats['avg_detection_score']:.3f}").grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(info_grid, text="Last Updated:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(info_grid, text=stats['latest_timestamp'].strftime("%Y-%m-%d %H:%M")).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        # Encodings table
        encodings_frame = ttk.LabelFrame(main_frame, text="Face Encodings", padding=10)
        encodings_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create table frame
        table_frame = ttk.Frame(encodings_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create encodings table
        columns = ('id', 'angle', 'confidence', 'date')
        table = ttk.Treeview(table_frame, columns=columns, show='headings')
        table.heading('id', text='ID')
        table.heading('angle', text='Angle')
        table.heading('confidence', text='Confidence')
        table.heading('date', text='Date')

        # Set column widths
        table.column('id', width=100)
        table.column('angle', width=150)
        table.column('confidence', width=100)
        table.column('date', width=150)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)

        # Pack table and scrollbar
        table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add encodings to table
        for i, face in enumerate(stats['encodings'], 1):
            unique_id = getattr(face, 'unique_id', f'enc_{i}')[:8]
            table.insert('', tk.END, values=(
                unique_id,
                getattr(face, 'angle_type', 'frontal'),
                f"{face.confidence:.1%}",
                face.timestamp.strftime("%Y-%m-%d %H:%M")
            ))

        # Close button
        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=10)

    def on_close(self):
        """Handle dialog close"""
        self.grab_release()
        self.destroy()


class SystemInfoDialog(tk.Toplevel):
    """Dialog for showing system information"""

    def __init__(self, parent, system):
        super().__init__(parent)
        self.system = system
        self.title("System Information")
        self.geometry("700x400")
        self.minsize(700, 400)

        # Make the dialog modal
        self.transient(parent)
        self.grab_set()

        # Create widgets
        self.create_widgets()

        # Load system info
        self.load_system_info()

        # Wait for window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create the dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create table frame
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create info table
        columns = ('component', 'status', 'details')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings')
        self.table.heading('component', text='Component')
        self.table.heading('status', text='Status')
        self.table.heading('details', text='Details')

        # Set column widths
        self.table.column('component', width=150)
        self.table.column('status', width=150)
        self.table.column('details', width=350)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)

        # Pack table and scrollbar
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Close button
        ttk.Button(main_frame, text="Close", command=self.on_close).pack(pady=10)

    def load_system_info(self):
        """Load system information"""
        # Get system info
        system_info = self.system.get_system_info()

        # Clear table
        for item in self.table.get_children():
            self.table.delete(item)

        # Add rows to table
        for component, info in system_info.items():
            self.table.insert('', tk.END, values=(
                component,
                info['status'],
                info['details']
            ))

    def on_close(self):
        """Handle dialog close"""
        self.grab_release()
        self.destroy()


class ModelCaptureDialog(tk.Toplevel):
    """Dialog for capturing a 3D face model"""

    def __init__(self, parent, system, video_stream):
        super().__init__(parent)
        self.system = system
        self.video_stream = video_stream
        self.parent = parent
        self.title("3D Face Model Capture")
        self.geometry("800x700")
        self.minsize(800, 700)

        # Make the dialog modal
        self.transient(parent)
        self.grab_set()

        # Initialize variables
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

        # Create widgets
        self.create_widgets()

        # Setup frame update
        self.update_frame()

        # Wait for window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create the dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        title_label = ttk.Label(main_frame, text="3D Face Model Capture", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Form frame
        form_frame = ttk.Frame(main_frame)
        form_frame.pack(fill=tk.X)

        # Person name input
        name_frame = ttk.Frame(form_frame)
        name_frame.pack(fill=tk.X, pady=5)

        ttk.Label(name_frame, text="Person Name:").pack(side=tk.LEFT, padx=5)
        self.name_entry = ttk.Entry(name_frame, width=30)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Model type selection
        model_frame = ttk.Frame(form_frame)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="Model Type:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="Full 3D (5 angles)")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                  values=["Full 3D (5 angles)", "Quick 3D (3 angles)"])
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        model_combo.bind("<<ComboboxSelected>>", self.update_model_type)

        # Video frame
        self.frame_viewer = FrameViewer(main_frame)
        self.frame_viewer.pack(fill=tk.BOTH, expand=True, pady=10)

        # Instruction label
        self.instruction_label = ttk.Label(main_frame, text="Please enter a name and choose model type",
                                        font=("Arial", 12))
        self.instruction_label.pack(pady=10)

        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to begin capture")
        self.status_label.pack(pady=5)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Buttons
        self.start_btn = ttk.Button(button_frame, text="Start Capture", command=self.start_capture)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.capture_btn = ttk.Button(button_frame, text="Capture Angle", command=self.capture_angle)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        self.capture_btn.state(['disabled'])

        self.skip_btn = ttk.Button(button_frame, text="Skip Angle", command=self.skip_angle)
        self.skip_btn.pack(side=tk.LEFT, padx=5)
        self.skip_btn.state(['disabled'])

        self.close_btn = ttk.Button(button_frame, text="Close", command=self.on_close)
        self.close_btn.pack(side=tk.RIGHT, padx=5)

    def update_frame(self):
        """Update the video frame"""
        try:
            # Get frame from video stream
            if not self.video_stream.frame_queue.empty():
                frame = self.video_stream.frame_queue.get_nowait()
                self.frame_viewer.update_frame(frame)
        except queue.Empty:
            pass

        # Schedule next update
        self.after(30, self.update_frame)

    def update_model_type(self, event=None):
        """Update the model type based on combobox selection"""
        self.full_3d = "Full 3D" in self.model_var.get()
        self.angles = self.angles_full if self.full_3d else self.angles_quick
        self.progress_bar.configure(maximum=len(self.angles))

    def start_capture(self):
        """Start the capture process"""
        self.person_name = self.name_entry.get().strip()
        if not self.person_name:
            messagebox.showwarning("Missing Name", "Please enter a person name")
            return

        # Update UI
        self.start_btn.state(['disabled'])
        self.capture_btn.state(['!disabled'])
        self.skip_btn.state(['!disabled'])
        self.name_entry.state(['disabled'])

        # Reset capture state
        self.current_angle_index = 0
        self.captured_angles = []
        self.progress_var.set(0)

        # Start with the first angle
        self.prepare_for_angle(0)

    def prepare_for_angle(self, index):
        """Prepare UI for capturing a specific angle"""
        if index >= len(self.angles):
            self.finish_capture()
            return

        angle_type, instruction = self.angles[index]

        # Update video thread with current angle
        self.video_stream.set_mode("preview", self.person_name, angle_type)

        # Update UI
        self.status_label.config(text=f"Capturing angle {index+1} of {len(self.angles)}: {angle_type}")
        self.instruction_label.config(text=f"Please {instruction}")

    def capture_angle(self):
        """Capture the current angle"""
        if self.current_angle_index >= len(self.angles):
            return

        angle_type, _ = self.angles[self.current_angle_index]

        # Update video thread to capture frame
        self.video_stream.set_mode("capture", self.person_name, angle_type)
        self.video_stream.capture_frame()

        # Wait a moment for capture to complete
        self.after(500, self.process_captured_frame)

    def process_captured_frame(self):
        """Process the captured frame"""
        if self.video_stream.captured_frame is not None and self.video_stream.capture_result:
            # Capture successful
            angle_type, _ = self.angles[self.current_angle_index]
            self.captured_angles.append(angle_type)
            self.status_label.config(text=f"Successfully captured {angle_type}")

            # Move to next angle
            self.current_angle_index += 1
            self.progress_var.set(len(self.captured_angles))

            if self.current_angle_index < len(self.angles):
                # Prepare for next angle
                self.after(1000, lambda: self.prepare_for_angle(self.current_angle_index))
            else:
                # Finished all angles
                self.finish_capture()
        else:
            # Capture failed
            messagebox.showwarning("Capture Failed",
                                 "Failed to detect a face. Please position yourself correctly and try again.")
            self.status_label.config(text="Capture failed. Please try again.")

    def skip_angle(self):
        """Skip the current angle"""
        if self.current_angle_index >= len(self.angles):
            return

        angle_type, _ = self.angles[self.current_angle_index]
        self.status_label.config(text=f"Skipped {angle_type}")

        # Move to next angle
        self.current_angle_index += 1

        if self.current_angle_index < len(self.angles):
            # Prepare for next angle
            self.after(500, lambda: self.prepare_for_angle(self.current_angle_index))
        else:
            # Finished all angles
            self.finish_capture()

    def finish_capture(self):
        """Finish the capture process"""
        if len(self.captured_angles) > 0:
            self.status_label.config(text=f"Completed! Captured {len(self.captured_angles)} angles")
            self.instruction_label.config(text="3D face model capture complete")

            # Show completion message
            messagebox.showinfo("Capture Complete",
                               f"Successfully created 3D face model with {len(self.captured_angles)} angles")
        else:
            self.status_label.config(text="No angles captured")
            messagebox.showwarning("No Captures", "No angles were captured")

        # Reset UI
        self.start_btn.state(['!disabled'])
        self.capture_btn.state(['disabled'])
        self.skip_btn.state(['disabled'])
        self.name_entry.state(['!disabled'])

        # Reset video thread mode
        self.video_stream.set_mode("preview")

    def on_close(self):
        """Handle dialog close"""
        # Make sure to reset video thread mode
        self.video_stream.set_mode("preview")
        self.grab_release()
        self.destroy()


class NotificationSettingsDialog(tk.Toplevel):
    """Dialog for configuring notification settings"""

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.title("Notification Settings")
        self.geometry("400x350")
        self.minsize(400, 350)
        self.transient(parent)
        self.grab_set()

        # Create main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Enable notifications checkbox
        self.enable_var = tk.BooleanVar(value=self.main_app.enable_notifications)
        enable_cb = ttk.Checkbutton(main_frame, text="Enable Notifications",
                                   variable=self.enable_var)
        enable_cb.pack(anchor=tk.W, pady=5)

        # Settings group
        settings_frame = ttk.LabelFrame(main_frame, text="Notification Settings", padding=10)
        settings_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Minimum confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conf_frame, text="Minimum Confidence:").pack(side=tk.LEFT)

        self.conf_value = tk.StringVar(value=f"{self.main_app.notification_min_confidence:.0%}")
        conf_label = ttk.Label(conf_frame, textvariable=self.conf_value, width=5)
        conf_label.pack(side=tk.RIGHT)

        self.conf_scale = ttk.Scale(settings_frame, from_=0, to=100,
                                   value=self.main_app.notification_min_confidence * 100,
                                   orient=tk.HORIZONTAL)
        self.conf_scale.pack(fill=tk.X, pady=5)

        # Update confidence value label when scale changes
        self.conf_scale.configure(command=self.update_conf_label)

        # Cooldown period
        cool_frame = ttk.Frame(settings_frame)
        cool_frame.pack(fill=tk.X, pady=10)

        ttk.Label(cool_frame, text="Cooldown Period (seconds):").pack(side=tk.LEFT)

        self.cooldown_var = tk.IntVar(value=self.main_app.notification_cooldown)
        cooldown_spin = ttk.Spinbox(cool_frame, from_=1, to=60,
                                   textvariable=self.cooldown_var, width=5)
        cooldown_spin.pack(side=tk.RIGHT)

        # Notification type
        type_frame = ttk.Frame(settings_frame)
        type_frame.pack(fill=tk.X, pady=10)

        ttk.Label(type_frame, text="Notification Type:").pack(side=tk.LEFT)

        self.type_var = tk.StringVar()
        type_combo = ttk.Combobox(type_frame, textvariable=self.type_var,
                                 values=["Pop-up", "Status Bar", "Both"], width=10)
        type_combo.current(self.main_app.notification_type)
        type_combo.pack(side=tk.RIGHT)
        type_combo.state(["readonly"])

        # Notify for unknown faces
        self.unknown_var = tk.BooleanVar(value=self.main_app.notify_unknown)
        unknown_cb = ttk.Checkbutton(settings_frame, text="Notify for Unknown Faces",
                                    variable=self.unknown_var)
        unknown_cb.pack(anchor=tk.W, pady=10)

        # Test notification button
        test_btn = ttk.Button(main_frame, text="Test Notification",
                             command=self.test_notification)
        test_btn.pack(pady=10)

        # Button frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        save_btn = ttk.Button(btn_frame, text="Save", command=self.save_settings)
        cancel_btn = ttk.Button(btn_frame, text="Cancel", command=self.destroy)

        save_btn.pack(side=tk.RIGHT, padx=5)
        cancel_btn.pack(side=tk.RIGHT, padx=5)

    def update_conf_label(self, value):
        """Update the confidence value label"""
        value = float(value)
        self.conf_value.set(f"{value/100:.0%}")

    def test_notification(self):
        """Send a test notification"""
        self.main_app.show_notification("Test Notification",
                                      "This is a test notification", force=True)

    def save_settings(self):
        """Save the notification settings"""
        # Get settings from UI components
        self.main_app.enable_notifications = self.enable_var.get()
        self.main_app.notification_min_confidence = float(self.conf_scale.get()) / 100
        self.main_app.notification_cooldown = self.cooldown_var.get()
        self.main_app.notification_type = ["Pop-up", "Status Bar", "Both"].index(self.type_var.get())
        self.main_app.notify_unknown = self.unknown_var.get()

        # Show confirmation and close
        self.main_app.status_var.set("Notification settings updated")
        self.destroy()


class VisualEffectsDialog(tk.Toplevel):
    """Dialog for configuring visual effects"""

    def __init__(self, parent, video_stream):
        super().__init__(parent)
        self.video_stream = video_stream
        self.title("Visual Effects Settings")
        self.geometry("500x450")
        self.minsize(500, 450)
        self.transient(parent)
        self.grab_set()

        # Create main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Visual Effects Settings",
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Enable effects checkbox
        self.enabled_var = tk.BooleanVar(value=self.video_stream.apply_effects)
        ttk.Checkbutton(main_frame, text="Enable Visual Effects",
                      variable=self.enabled_var,
                      command=self.toggle_effects).pack(anchor=tk.W, pady=10)

        # Effects type selector
        type_frame = ttk.LabelFrame(main_frame, text="Effect Type", padding=10)
        type_frame.pack(fill=tk.X, pady=10)

        self.effect_type_var = tk.StringVar(value=self.video_stream.effect_type)

        # Create effect type radio buttons
        effects = [
            ("None", "none"),
            ("Enhanced", "enhance"),
            ("Vintage", "vintage"),
            ("Cool Tone", "cool"),
            ("Warm Tone", "warm")
        ]

        for text, value in effects:
            ttk.Radiobutton(type_frame, text=text, value=value,
                          variable=self.effect_type_var,
                          command=self.update_effect_type).pack(anchor=tk.W, pady=5)

        # Effect level slider
        level_frame = ttk.LabelFrame(main_frame, text="Effect Intensity", padding=10)
        level_frame.pack(fill=tk.X, pady=10)

        slider_frame = ttk.Frame(level_frame)
        slider_frame.pack(fill=tk.X, pady=5)

        ttk.Label(slider_frame, text="Subtle").pack(side=tk.LEFT)
        self.level_var = tk.DoubleVar(value=self.video_stream.effect_level)
        level_slider = ttk.Scale(slider_frame, from_=0, to=1.0, orient=tk.HORIZONTAL,
                               variable=self.level_var, command=self.update_effect_level)
        level_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Label(slider_frame, text="Strong").pack(side=tk.LEFT)

        # Effect level display
        self.level_text = tk.StringVar(value=f"{int(self.video_stream.effect_level * 100)}%")
        ttk.Label(level_frame, textvariable=self.level_text).pack(pady=5)

        # Advanced settings
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Settings", padding=10)
        adv_frame.pack(fill=tk.X, pady=10)

        # Contrast and brightness adjustments (more advanced than Qt version)
        contrast_frame = ttk.Frame(adv_frame)
        contrast_frame.pack(fill=tk.X, pady=5)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_slider = ttk.Scale(contrast_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL,
                                  variable=self.contrast_var)
        contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        brightness_frame = ttk.Frame(adv_frame)
        brightness_frame.pack(fill=tk.X, pady=5)
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT)
        self.brightness_var = tk.DoubleVar(value=0.0)
        brightness_slider = ttk.Scale(brightness_frame, from_=-50, to=50, orient=tk.HORIZONTAL,
                                    variable=self.brightness_var)
        brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Preview button
        ttk.Button(main_frame, text="Apply Changes",
                 command=self.apply_changes).pack(pady=10)

        # Button frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(btn_frame, text="OK", command=self.save_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def toggle_effects(self):
        """Toggle visual effects on/off"""
        self.video_stream.apply_effects = self.enabled_var.get()

    def update_effect_type(self):
        """Update effect type"""
        self.video_stream.effect_type = self.effect_type_var.get()

    def update_effect_level(self, value):
        """Update effect level"""
        try:
            level = float(value)
            self.video_stream.set_effect_level(level)
            self.level_text.set(f"{int(level * 100)}%")
        except ValueError:
            pass

    def apply_changes(self):
        """Apply the current settings to the video stream"""
        self.video_stream.apply_effects = self.enabled_var.get()
        self.video_stream.effect_type = self.effect_type_var.get()
        self.video_stream.set_effect_level(self.level_var.get())

    def save_settings(self):
        """Save settings and close dialog"""
        self.apply_changes()
        self.destroy()


class MainApplication(tk.Tk):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Ultra-Modern Face Recognition System")
        self.geometry("1200x700")
        self.minsize(1000, 600)

        # Initialize style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))

        # State variables and settings - move these before create_widgets()
        self.display_size_factor = 100
        self.is_recognition_active = False
        self.update_results_id = None

        # Initialize face recognition system
        self.system = UltraModernFaceRecognitionSystem()

        # Configure notification settings
        self.enable_notifications = True
        self.notification_min_confidence = 0.7
        self.notification_cooldown = 5  # seconds
        self.notification_type = 0  # 0: Pop-up, 1: Status Bar, 2: Both
        self.notify_unknown = False

        # Last notification time for cooldown tracking
        self.last_notification_time = {}

        # Initialize video stream
        self.video_stream = VideoStream(self.system)

        # Initialize UI components
        self.create_widgets()

        # Start video stream
        self.video_stream.start()

        # Setup frame update
        self.update_frame()


        # Configure window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create the application widgets"""
        # Main container
        main_container = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side - Video display
        video_frame = ttk.Frame(main_container)
        main_container.add(video_frame, weight=3)

        # Video display
        video_display_frame = ttk.LabelFrame(video_frame, text="Camera Feed")
        video_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_viewer = FrameViewer(video_display_frame)
        self.frame_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Camera controls
        camera_control_frame = ttk.Frame(video_frame)
        camera_control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(camera_control_frame, text="Camera:").pack(side=tk.LEFT, padx=5)

        # Camera selection
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_control_frame, textvariable=self.camera_var)
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.populate_camera_list()

        ttk.Button(camera_control_frame, text="Change Camera",
                 command=self.change_camera).pack(side=tk.LEFT, padx=5)

        # Right side - Controls and info
        control_frame = ttk.Frame(main_container)
        main_container.add(control_frame, weight=1)

        # Title and info
        title_frame = ttk.Frame(control_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=10)

        title_label = ttk.Label(title_frame, text="Ultra-Modern Face Recognition",
                              font=("Arial", 16, "bold"))
        title_label.pack()

        subtitle_label = ttk.Label(title_frame, text="2025 State-of-the-Art Technology",
                                 font=("Arial", 12))
        subtitle_label.pack()

        # Main controls
        main_controls_frame = ttk.LabelFrame(control_frame, text="Controls")
        main_controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Recognition button
        self.recognition_btn = ttk.Button(main_controls_frame, text="Start Live Recognition",
                                       command=self.toggle_recognition)
        self.recognition_btn.pack(fill=tk.X, padx=5, pady=5)

        # Other controls
        control_buttons_frame = ttk.Frame(main_controls_frame)
        control_buttons_frame.pack(fill=tk.X, padx=5, pady=5)

        # Button grid
        self.add_face_btn = ttk.Button(control_buttons_frame, text="Add Face (Simple)",
                                    command=self.add_face_simple)
        self.add_face_btn.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

        self.add_face_3d_btn = ttk.Button(control_buttons_frame, text="Add Face (3D Model)",
                                       command=self.add_face_3d)
        self.add_face_3d_btn.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)

        self.view_db_btn = ttk.Button(control_buttons_frame, text="View Face Database",
                                   command=self.view_database)
        self.view_db_btn.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)

        self.manage_persons_btn = ttk.Button(control_buttons_frame, text="Person Management",
                                         command=self.manage_persons)
        self.manage_persons_btn.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

        self.system_info_btn = ttk.Button(control_buttons_frame, text="System Information",
                                       command=self.show_system_info)
        self.system_info_btn.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)

        # Configure grid columns
        control_buttons_frame.columnconfigure(0, weight=1)
        control_buttons_frame.columnconfigure(1, weight=1)

        # Recognition results
        results_frame = ttk.LabelFrame(control_frame, text="Recognition Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results text
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="System ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Control frame for sliders and settings
        settings_frame = ttk.LabelFrame(control_frame, text="Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=10)

        # Display size slider
        ttk.Label(settings_frame, text="Display Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.size_var = tk.IntVar(value=self.display_size_factor)
        size_scale = ttk.Scale(settings_frame, from_=10, to=200, variable=self.size_var,
                             orient=tk.HORIZONTAL, command=self.update_display_size)
        size_scale.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Recognition sensitivity slider
        ttk.Label(settings_frame, text="Recognition Sensitivity:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.sensitivity_var = tk.DoubleVar(value=self.system.recognition_threshold * 100)
        sensitivity_scale = ttk.Scale(settings_frame, from_=0, to=100, variable=self.sensitivity_var,
                                     orient=tk.HORIZONTAL, command=self.update_recognition_sensitivity)
        sensitivity_scale.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)

        # Close button
        ttk.Button(control_frame, text="Close", command=self.on_close).pack(pady=10)

        # Initialize camera list and set initial pane sizes
        self.populate_camera_list()
        main_container.sashpos(0, 800)

    def populate_camera_list(self):
        """Populate the camera dropdown list"""
        self.camera_combo['values'] = [f"Camera {cam['index']}: {cam['resolution']}"
                                     for cam in self.system.camera_manager.available_cameras]

        # Set current index
        current_cam = self.system.camera_manager.current_camera_index
        for i, cam in enumerate(self.system.camera_manager.available_cameras):
            if cam['index'] == current_cam:
                self.camera_combo.current(i)
                break

    def update_frame(self):
        """Update the video frame"""
        try:
            # Get frame from video stream
            if not self.video_stream.frame_queue.empty():
                frame = self.video_stream.frame_queue.get_nowait()
                self.frame_viewer.update_frame(frame)
        except queue.Empty:
            pass

        # Schedule next update
        self.after(30, self.update_frame)

    def update_recognition_results(self):
        """Update recognition results display"""
        # Get results from video stream
        results = self.video_stream.recognition_results
        fps = self.video_stream.fps

        # Clear previous results
        self.results_text.delete(1.0, tk.END)

        # Add FPS info
        self.results_text.insert(tk.END, f"FPS: {fps:.1f}\n\n")

        # Add recognition results
        for result in results:
            if result['recognized']:
                name = result['name']
                confidence = result['confidence']
                self.results_text.insert(tk.END, f"✅ {name} ({confidence:.1%})\n")
            else:
                self.results_text.insert(tk.END, "❓ Unknown Person\n")

            # Add details
            self.results_text.insert(tk.END, f"   Model: {result.get('model_used', 'unknown')}\n")
            self.results_text.insert(tk.END, f"   Score: {result.get('detection_score', 0.0):.3f}\n\n")

        # Schedule next update if recognition is active
        if self.is_recognition_active:
            self.update_results_id = self.after(500, self.update_recognition_results)

    def toggle_recognition(self):
        """Toggle live face recognition"""
        if self.is_recognition_active:
            # Stop recognition
            self.video_stream.set_mode("preview")
            self.is_recognition_active = False
            self.recognition_btn.config(text="Start Live Recognition")
            self.results_text.delete(1.0, tk.END)
            self.status_var.set("Recognition stopped")

            # Cancel scheduled updates
            if self.update_results_id:
                self.after_cancel(self.update_results_id)
                self.update_results_id = None
        else:
            # Start recognition
            self.video_stream.set_mode("recognition")
            self.is_recognition_active = True
            self.recognition_btn.config(text="Stop Recognition")
            self.status_var.set("Running live recognition...")

            # Start updating results
            self.update_recognition_results()

    def change_camera(self):
        """Change the camera being used"""
        selection = self.camera_combo.get()

        if not selection:
            return

        # Extract camera index from selection
        try:
            camera_index = int(selection.split(":")[0].replace("Camera ", ""))
        except (ValueError, IndexError):
            messagebox.showwarning("Invalid Selection", "Please select a valid camera")
            return

        # Stop current video stream
        self.video_stream.stop()

        # Update system camera
        self.system.camera_manager.current_camera_index = camera_index

        # Create new video stream
        self.video_stream = VideoStream(self.system, camera_index)

        # Restore recognition mode if active
        if self.is_recognition_active:
            self.video_stream.set_mode("recognition")

        # Start thread
        self.video_stream.start()

        self.status_var.set(f"Changed to camera {camera_index}")

    def add_face_simple(self):
        """Add a simple face to the database"""
        # Get person name
        person_name = simpledialog.askstring("Add Face", "Enter person name:")

        if not person_name:
            return

        # Create dialog for capture
        dialog = tk.Toplevel(self)
        dialog.title("Capture Face")
        dialog.geometry("800x600")
        dialog.transient(self)
        dialog.grab_set()

        # Create layout
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions
        instructions = ttk.Label(main_frame, text="Position your face in the camera and press 'Capture'",
                               font=("Arial", 12))
        instructions.pack(pady=10)

        # Video frame
        frame_viewer = FrameViewer(main_frame)
        frame_viewer.pack(fill=tk.BOTH, expand=True, pady=10)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        def update_frame():
            """Update the video frame"""
            try:
                # Get frame from video stream
                if not self.video_stream.frame_queue.empty():
                    frame = self.video_stream.frame_queue.get_nowait()
                    frame_viewer.update_frame(frame)
            except queue.Empty:
                pass

            # Schedule next update if dialog still exists
            if dialog.winfo_exists():
                dialog.after(30, update_frame)

        def do_capture():
            """Capture a face"""
            self.video_stream.set_mode("capture", person_name)
            self.video_stream.capture_frame()

            # Wait a moment for capture
            dialog.after(1000, check_capture)

        def check_capture():
            """Check if capture was successful"""
            if self.video_stream.capture_result:
                messagebox.showinfo("Success", f"Successfully added {person_name} to the database")
                dialog.destroy()
            else:
                messagebox.showwarning("Capture Failed", "Failed to detect a face. Please try again.")

        # Create buttons
        ttk.Button(button_frame, text="Capture", command=do_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

        # Set video mode and start updating
        self.video_stream.set_mode("preview")
        update_frame()

    def add_face_3d(self):
        """Add a 3D face model to the database"""
        # Create and show the 3D model capture dialog
        ModelCaptureDialog(self, self.system, self.video_stream)


    def view_database(self):
        """View the face database"""
        # Create view dialog
        dialog = tk.Toplevel(self)
        dialog.title("Face Database")
        dialog.geometry("800x600")
        dialog.minsize(800, 600)

        # Make the dialog modal
        dialog.transient(self)
        dialog.grab_set()

        # Create main frame
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create table frame
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)

        # Create table
        columns = ('id', 'name', 'model', 'angle', 'confidence', 'added')
        table = ttk.Treeview(table_frame, columns=columns, show='headings')
        table.heading('id', text='ID')
        table.heading('name', text='Name')
        table.heading('model', text='Model')
        table.heading('angle', text='Angle')
        table.heading('confidence', text='Confidence')
        table.heading('added', text='Added')

        # Set column widths
        table.column('id', width=70)
        table.column('name', width=150)
        table.column('model', width=100)
        table.column('angle', width=100)
        table.column('confidence', width=100)
        table.column('added', width=150)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
        table.configure(yscrollcommand=scrollbar.set)

        # Pack table and scrollbar
        table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add data to table
        for i, face in enumerate(self.system.face_encodings):
            unique_id = getattr(face, 'unique_id', f'face_{i}')[:8]

            table.insert('', tk.END, values=(
                unique_id,
                face.person_name,
                face.model_used,
                getattr(face, 'angle_type', 'frontal'),
                f"{face.confidence:.1%}",
                face.timestamp.strftime("%Y-%m-%d %H:%M")
            ))

        # Close button
        ttk.Button(main_frame, text="Close", command=dialog.destroy).pack(pady=10)

    def manage_persons(self):
        """Manage persons in the database"""
        # Create and show person management dialog
        PersonManagementDialog(self, self.system)

    def show_system_info(self):
        """Show system information"""
        # Create and show system info dialog
        SystemInfoDialog(self, self.system)

    def on_close(self):
        """Handle window close"""
        # Stop video stream
        self.video_stream.stop()

        # Destroy window
        self.destroy()

    def update_display_size(self, value):
        """
        Update the display size based on slider value

        Args:
            value: Slider value (10-200)
        """
        try:
            # Convert string to float (tkinter passes value as string)
            self.display_size_factor = float(value)

            # Update frame viewer if it exists
            if hasattr(self, 'frame_viewer'):
                self.frame_viewer.set_display_scale(self.display_size_factor / 100.0)
        except ValueError:
            pass  # Ignore invalid values

    def update_recognition_sensitivity(self, value):
        """
        Update the recognition sensitivity based on slider value

        Args:
            value: Slider value (0-100)
        """
        try:
            # Convert string to float (tkinter passes value as string)
            sensitivity = float(value)

            # Update system recognition threshold (inverse relationship -
            # higher slider value = lower threshold = more sensitive)
            threshold = 1.0 - (sensitivity / 100.0)
            # Ensure threshold is in a reasonable range (0.3 to 0.8)
            threshold = max(0.3, min(0.8, threshold))

            # Update system
            self.system.recognition_threshold = threshold

            # Update status bar
            self.status_var.set(f"Recognition sensitivity set to {sensitivity:.0f}%")
        except ValueError:
            pass  # Ignore invalid values

    def show_notification(self, title, message, force=False):
        """Show a notification based on current settings"""
        # Check if notifications are enabled
        if not self.enable_notifications and not force:
            return

        current_time = time.time()

        # Check cooldown period if not forced
        if not force and title in self.last_notification_time:
            elapsed = current_time - self.last_notification_time[title]
            if elapsed < self.notification_cooldown:
                return

        # Update last notification time
        self.last_notification_time[title] = current_time

        # Show notification based on type
        if self.notification_type == 0 or self.notification_type == 2:  # Pop-up
            messagebox.showinfo(title, message)

        if self.notification_type == 1 or self.notification_type == 2:  # Status bar
            self.status_var.set(f"{title}: {message}")


def main():
    """Main application entry point"""
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    main()
