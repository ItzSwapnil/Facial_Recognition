"""
Advanced Camera and Video Handling System
=========================================

Comprehensive camera management supporting multiple video sources
including webcams, IP cameras, RTSP streams, and video files.
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, Any, Tuple, List
from queue import Queue, Full, Empty
from pathlib import Path
import imutils

from .config import Config, get_config


class CameraHandler:
    """
    Advanced camera handler supporting multiple video sources.
    
    Features:
    - Multiple camera/video sources
    - Threaded frame capture for performance
    - RTSP stream support
    - Frame buffering and processing
    - Auto-reconnection for network streams
    - Frame rate control
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the camera handler."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_source = None
        
        # Threading
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_queue = Queue(maxsize=10)
        self.frame_lock = threading.Lock()
        
        # Current frame
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Source management
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Frame processing callbacks
        self.frame_callbacks: List[Callable] = []
        
        # Performance tracking
        self.total_frames_processed = 0
        self.dropped_frames = 0
        
    def _setup_camera(self, source: Any) -> bool:
        """Setup camera with optimal settings."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            # Create video capture
            if isinstance(source, str) and (source.startswith('rtsp://') or source.startswith('http://')):
                # Network stream
                self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera.buffer_size)
            else:
                # Local camera or file
                self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {source}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.camera.buffer_size)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera setup: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.current_source = source
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup camera: {e}")
            return False
    
    def _capture_frames(self) -> None:
        """Continuous frame capture in separate thread."""
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(1)
                        continue
                
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to read frame")
                    if not self._reconnect():
                        time.sleep(1)
                        continue
                
                # Update frame
                with self.frame_lock:
                    self.current_frame = frame.copy()
                    self.frame_count += 1
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except Full:
                    # Queue is full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                        self.dropped_frames += 1
                    except Empty:
                        pass
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(frame.copy())
                    except Exception as e:
                        self.logger.error(f"Frame callback failed: {e}")
                
                # Update FPS
                self._update_fps()
                
                # Frame rate control
                time.sleep(1.0 / self.config.camera.fps)
                
            except Exception as e:
                self.logger.error(f"Frame capture error: {e}")
                time.sleep(1)
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to video source."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return False
        
        self.logger.info(f"Attempting to reconnect... (attempt {self.reconnect_attempts + 1})")
        self.reconnect_attempts += 1
        
        if self._setup_camera(self.current_source):
            self.logger.info("Reconnection successful")
            return True
        
        time.sleep(self.reconnect_delay)
        return False
    
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def start_capture(self, source: Any = None) -> bool:
        """
        Start video capture from specified source.
        
        Args:
            source: Video source (camera index, file path, or stream URL)
                   If None, uses config.camera.camera_index
        
        Returns:
            True if capture started successfully
        """
        if self.is_running:
            self.logger.warning("Capture is already running")
            return True
        
        if source is None:
            source = self.config.camera.camera_index
        
        if not self._setup_camera(source):
            return False
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        self.logger.info(f"Video capture started from source: {source}")
        return True
    
    def stop_capture(self) -> None:
        """Stop video capture."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5)
        
        # Release resources
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        self.logger.info("Video capture stopped")
    
    def get_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the latest frame from queue.
        
        Args:
            timeout: Timeout in seconds (None for non-blocking)
        
        Returns:
            Latest frame or None if no frame available
        """
        try:
            if timeout is None:
                return self.frame_queue.get_nowait()
            else:
                return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (latest captured)."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def add_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Add a callback function to be called for each frame."""
        self.frame_callbacks.append(callback)
    
    def remove_frame_callback(self, callback: Callable) -> None:
        """Remove a frame callback."""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
    
    def is_capture_running(self) -> bool:
        """Check if capture is currently running."""
        return self.is_running
    
    def get_camera_info(self) -> dict:
        """Get camera information and settings."""
        if self.cap is None:
            return {}
        
        return {
            'source': str(self.current_source),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'current_fps': self.current_fps,
            'frame_count': self.frame_count,
            'is_running': self.is_running,
            'buffer_size': int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
        }
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        total_frames = self.frame_count
        processing_rate = total_frames / max(1, time.time() - self.last_fps_time) if total_frames > 0 else 0
        
        return {
            'total_frames_captured': total_frames,
            'dropped_frames': self.dropped_frames,
            'current_fps': self.current_fps,
            'drop_rate': self.dropped_frames / max(1, total_frames),
            'queue_size': self.frame_queue.qsize(),
            'reconnect_attempts': self.reconnect_attempts
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.frame_count = 0
        self.dropped_frames = 0
        self.reconnect_attempts = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
    
    def capture_snapshot(self, filename: Optional[str] = None) -> bool:
        """
        Capture a snapshot from current frame.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            True if snapshot saved successfully
        """
        frame = self.get_current_frame()
        if frame is None:
            self.logger.error("No frame available for snapshot")
            return False
        
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
            
            # Ensure directory exists
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            success = cv2.imwrite(str(output_path), frame)
            
            if success:
                self.logger.info(f"Snapshot saved: {filename}")
                return True
            else:
                self.logger.error(f"Failed to save snapshot: {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"Snapshot capture failed: {e}")
            return False
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """Set camera property."""
        if self.cap is None:
            return False
        
        try:
            success = self.cap.set(property_id, value)
            if success:
                self.logger.debug(f"Camera property {property_id} set to {value}")
            else:
                self.logger.warning(f"Failed to set camera property {property_id} to {value}")
            return success
        except Exception as e:
            self.logger.error(f"Error setting camera property: {e}")
            return False
    
    def get_camera_property(self, property_id: int) -> float:
        """Get camera property value."""
        if self.cap is None:
            return 0.0
        
        try:
            return self.cap.get(property_id)
        except Exception as e:
            self.logger.error(f"Error getting camera property: {e}")
            return 0.0
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_capture()
