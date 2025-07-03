"""
Advanced Alert and Notification System
======================================

Comprehensive alert system supporting multiple notification methods
including desktop notifications, sound alerts, logging, and custom webhooks.
"""

import logging
import time
import threading
from typing import Dict, Optional, List, Callable, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

# Notification libraries
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

import requests
import numpy as np

from .config import Config, get_config


class AlertSystem:
    """
    Advanced alert system for facial recognition events.
    
    Features:
    - Desktop notifications
    - Sound alerts
    - Event logging
    - Cooldown periods
    - Custom webhooks
    - Multiple alert types
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the alert system."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Alert history and cooldowns
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Sound system
        self.sound_initialized = False
        self._init_sound_system()
        
        # Custom alert callbacks
        self.custom_callbacks: List[Callable] = []
        
        # Alert statistics
        self.alert_count = 0
        self.alerts_by_person: Dict[str, int] = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging for alerts."""
        if self.config.alerts.enable_logging:
            # Create alerts log file
            log_file = self.config.logs_dir / "alerts.log"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to logger
            alert_logger = logging.getLogger('alerts')
            alert_logger.setLevel(logging.INFO)
            alert_logger.addHandler(file_handler)
    
    def _init_sound_system(self) -> None:
        """Initialize pygame for sound alerts."""
        if not PYGAME_AVAILABLE or not self.config.alerts.enable_sound:
            return
        
        try:
            pygame.mixer.init()
            self.sound_initialized = True
            self.logger.info("Sound system initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize sound system: {e}")
            self.sound_initialized = False
    
    def _is_on_cooldown(self, person_name: str) -> bool:
        """Check if alerts for a person are on cooldown."""
        if person_name not in self.last_alert_times:
            return False
        
        cooldown_period = timedelta(seconds=self.config.alerts.alert_cooldown)
        time_since_last = datetime.now() - self.last_alert_times[person_name]
        
        return time_since_last < cooldown_period
    
    def _update_cooldown(self, person_name: str) -> None:
        """Update the last alert time for a person."""
        self.last_alert_times[person_name] = datetime.now()
    
    def _send_desktop_notification(self, title: str, message: str, timeout: int = None) -> None:
        """Send desktop notification."""
        if not PLYER_AVAILABLE or not self.config.alerts.enable_notifications:
            return
        
        timeout = timeout or self.config.alerts.notification_timeout
        
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=timeout,
                app_name="Facial Recognition System"
            )
            self.logger.debug(f"Desktop notification sent: {title}")
        except Exception as e:
            self.logger.warning(f"Failed to send desktop notification: {e}")
    
    def _play_sound_alert(self, sound_file: Optional[str] = None) -> None:
        """Play sound alert."""
        if not self.sound_initialized:
            return
        
        sound_path = sound_file or self.config.alerts.sound_file
        
        try:
            if sound_path and Path(sound_path).exists():
                # Play custom sound file
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
            else:
                # Play default beep sound
                # Generate a simple beep using pygame
                duration = 0.5  # seconds
                sample_rate = 22050
                frames = int(duration * sample_rate)
                arr = []
                
                for i in range(frames):
                    time_point = float(i) / sample_rate
                    arr.append([4096 * np.sin(2 * np.pi * 440 * time_point)])
                
                sound = pygame.sndarray.make_sound(np.array(arr))
                sound.play()
            
            self.logger.debug("Sound alert played")
            
        except Exception as e:
            self.logger.warning(f"Failed to play sound alert: {e}")
    
    def _log_alert(self, alert_type: str, person_name: str, confidence: float, 
                   timestamp: datetime, additional_info: Dict[str, Any] = None) -> None:
        """Log alert to file and memory."""
        alert_data = {
            'timestamp': timestamp.isoformat(),
            'type': alert_type,
            'person': person_name,
            'confidence': confidence,
            'additional_info': additional_info or {}
        }
        
        # Add to memory
        self.alert_history.append(alert_data)
        
        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Log to file
        if self.config.alerts.enable_logging:
            alert_logger = logging.getLogger('alerts')
            log_message = f"{alert_type} - {person_name} (confidence: {confidence:.2f})"
            if additional_info:
                log_message += f" - {json.dumps(additional_info)}"
            alert_logger.info(log_message)
    
    def _call_custom_callbacks(self, alert_type: str, person_name: str, 
                              confidence: float, additional_info: Dict[str, Any] = None) -> None:
        """Call custom alert callbacks."""
        for callback in self.custom_callbacks:
            try:
                callback(alert_type, person_name, confidence, additional_info)
            except Exception as e:
                self.logger.error(f"Custom callback failed: {e}")
    
    def send_webhook(self, url: str, alert_data: Dict[str, Any]) -> bool:
        """Send alert data to webhook URL."""
        try:
            response = requests.post(
                url, 
                json=alert_data, 
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            self.logger.debug(f"Webhook sent successfully to {url}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send webhook to {url}: {e}")
            return False
    
    def trigger_alert(self, alert_type: str, person_name: str, confidence: float,
                     additional_info: Dict[str, Any] = None, force: bool = False) -> bool:
        """
        Trigger an alert for face recognition event.
        
        Args:
            alert_type: Type of alert ('recognition', 'unknown', 'unauthorized', etc.)
            person_name: Name of the person detected
            confidence: Recognition confidence score
            additional_info: Additional information about the detection
            force: Force alert even if on cooldown
        
        Returns:
            True if alert was sent, False if on cooldown or failed
        """
        # Check cooldown
        if not force and self._is_on_cooldown(person_name):
            return False
        
        timestamp = datetime.now()
        
        # Update statistics
        self.alert_count += 1
        self.alerts_by_person[person_name] = self.alerts_by_person.get(person_name, 0) + 1
        
        # Create alert message
        if alert_type == "recognition":
            title = "Person Recognized"
            message = f"{person_name} detected with {confidence:.1%} confidence"
        elif alert_type == "unknown":
            title = "Unknown Person Detected"
            message = f"Unrecognized person detected (confidence: {confidence:.1%})"
        elif alert_type == "unauthorized":
            title = "Unauthorized Access Alert"
            message = f"Unauthorized person: {person_name} (confidence: {confidence:.1%})"
        else:
            title = "Facial Recognition Alert"
            message = f"{alert_type}: {person_name} (confidence: {confidence:.1%})"
        
        # Send desktop notification
        self._send_desktop_notification(title, message)
        
        # Play sound alert
        self._play_sound_alert()
        
        # Log alert
        self._log_alert(alert_type, person_name, confidence, timestamp, additional_info)
        
        # Call custom callbacks
        self._call_custom_callbacks(alert_type, person_name, confidence, additional_info)
        
        # Update cooldown
        self._update_cooldown(person_name)
        
        self.logger.info(f"Alert triggered: {title} - {message}")
        return True
    
    def add_custom_callback(self, callback: Callable) -> None:
        """Add a custom callback function for alerts."""
        self.custom_callbacks.append(callback)
    
    def remove_custom_callback(self, callback: Callable) -> None:
        """Remove a custom callback function."""
        if callback in self.custom_callbacks:
            self.custom_callbacks.remove(callback)
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        return self.alert_history[-limit:]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            'total_alerts': self.alert_count,
            'alerts_by_person': self.alerts_by_person.copy(),
            'active_cooldowns': len(self.last_alert_times),
            'history_size': len(self.alert_history),
            'most_detected': max(self.alerts_by_person.items(), 
                               key=lambda x: x[1]) if self.alerts_by_person else ("None", 0)
        }
    
    def clear_cooldowns(self) -> None:
        """Clear all alert cooldowns."""
        self.last_alert_times.clear()
        self.logger.info("All alert cooldowns cleared")
    
    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        self.logger.info("Alert history cleared")
    
    def reset_stats(self) -> None:
        """Reset alert statistics."""
        self.alert_count = 0
        self.alerts_by_person.clear()
        self.clear_cooldowns()
        self.clear_history()
        self.logger.info("Alert statistics reset")
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification methods."""
        results = {}
        
        # Test desktop notification
        try:
            self._send_desktop_notification(
                "Test Notification", 
                "Facial Recognition System test alert", 
                timeout=3
            )
            results['desktop_notification'] = True
        except Exception as e:
            self.logger.error(f"Desktop notification test failed: {e}")
            results['desktop_notification'] = False
        
        # Test sound alert
        try:
            self._play_sound_alert()
            results['sound_alert'] = True
        except Exception as e:
            self.logger.error(f"Sound alert test failed: {e}")
            results['sound_alert'] = False
        
        # Test logging
        try:
            self._log_alert("test", "Test User", 0.95, datetime.now(), {"test": True})
            results['logging'] = True
        except Exception as e:
            self.logger.error(f"Logging test failed: {e}")
            results['logging'] = False
        
        return results
