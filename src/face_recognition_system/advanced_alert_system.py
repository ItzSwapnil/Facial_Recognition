"""
Advanced Alert and Notification System
======================================

State-of-the-art alert system with modern notification capabilities:
- Native Windows 10/11 notifications
- Real-time webhooks and API integration
- MQTT messaging for IoT integration
- Email and SMS notifications
- System tray integration
- Advanced logging and analytics
- Multi-channel alert routing
"""

import logging
import time
import threading
import asyncio
import json
from typing import Dict, Optional, List, Callable, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import platform
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Advanced notification libraries
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

# Windows-specific notifications
if platform.system() == "Windows":
    try:
        from win10toast import ToastNotifier
        WIN10_TOAST_AVAILABLE = True
    except ImportError:
        WIN10_TOAST_AVAILABLE = False
else:
    WIN10_TOAST_AVAILABLE = False

# System tray integration
try:
    import pystray
    from PIL import Image
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False

# Networking and messaging
import requests
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import asyncio_mqtt as aiomqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

import numpy as np
from .config import Config, get_config


class AdvancedAlertSystem:
    """
    Advanced alert system with multiple notification channels.
    
    Features:
    - Native OS notifications
    - Real-time webhooks
    - MQTT messaging
    - Email notifications
    - SMS alerts (via API)
    - System tray integration
    - Alert routing and escalation
    - Analytics and reporting
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the advanced alert system."""
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Notification system availability
        self.notification_methods = {
            'plyer': PLYER_AVAILABLE,
            'win10_toast': WIN10_TOAST_AVAILABLE,
            'pystray': PYSTRAY_AVAILABLE,
            'pygame': PYGAME_AVAILABLE,
            'websockets': WEBSOCKETS_AVAILABLE,
            'mqtt': MQTT_AVAILABLE
        }
        
        # Alert management
        self.alert_history: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Notification handlers
        self.win10_toast = None
        self.system_tray = None
        self.mqtt_client = None
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_person': {},
            'alerts_by_method': {},
            'response_times': [],
            'escalation_count': 0
        }
        
        # Alert routing rules
        self.alert_rules = {
            'unknown': ['desktop', 'sound', 'log'],
            'recognition': ['desktop', 'log'],
            'unauthorized': ['desktop', 'sound', 'email', 'webhook', 'log'],
            'intrusion': ['desktop', 'sound', 'email', 'sms', 'webhook', 'log']
        }
        
        # Email configuration
        self.email_config = {
            'smtp_server': '',
            'smtp_port': 587,
            'username': '',
            'password': '',
            'from_email': '',
            'to_emails': []
        }
        
        # SMS configuration (using API services)
        self.sms_config = {
            'provider': 'twilio',  # 'twilio', 'nexmo', etc.
            'api_key': '',
            'api_secret': '',
            'from_number': '',
            'to_numbers': []
        }
        
        # Webhook configuration
        self.webhook_config = {
            'urls': [],
            'timeout': 5,
            'retry_attempts': 3
        }
        
        # MQTT configuration
        self.mqtt_config = {
            'broker': 'localhost',
            'port': 1883,
            'topic_prefix': 'facial_recognition',
            'username': '',
            'password': ''
        }
        
        self._initialize_systems()
        self._setup_logging()
    
    def _initialize_systems(self) -> None:
        """Initialize notification systems."""
        self.logger.info("Initializing advanced alert systems...")
        
        # Initialize Windows Toast notifications
        if self.notification_methods['win10_toast']:
            self._init_win10_toast()
        
        # Initialize system tray
        if self.notification_methods['pystray']:
            self._init_system_tray()
        
        # Initialize sound system
        if self.notification_methods['pygame']:
            self._init_sound_system()
        
        # Initialize MQTT client
        if self.notification_methods['mqtt']:
            self._init_mqtt_client()
    
    def _init_win10_toast(self) -> None:
        """Initialize Windows 10/11 toast notifications."""
        try:
            self.win10_toast = ToastNotifier()
            self.logger.info("Windows toast notifications initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Windows toast: {e}")
            self.notification_methods['win10_toast'] = False
    
    def _init_system_tray(self) -> None:
        """Initialize system tray integration."""
        try:
            # Create a simple icon (you can replace with your own)
            image = Image.new('RGB', (64, 64), color='red')
            
            menu = pystray.Menu(
                pystray.MenuItem("Show Stats", self._show_tray_stats),
                pystray.MenuItem("Clear Alerts", self._clear_alerts_tray),
                pystray.MenuItem("Exit", self._exit_tray)
            )
            
            self.system_tray = pystray.Icon(
                "Facial Recognition",
                image,
                menu=menu
            )
            
            # Start system tray in background thread
            threading.Thread(target=self.system_tray.run, daemon=True).start()
            self.logger.info("System tray initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system tray: {e}")
            self.notification_methods['pystray'] = False
    
    def _init_sound_system(self) -> None:
        """Initialize sound alert system."""
        try:
            pygame.mixer.init()
            self.logger.info("Sound system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize sound system: {e}")
            self.notification_methods['pygame'] = False
    
    def _init_mqtt_client(self) -> None:
        """Initialize MQTT client for IoT integration."""
        try:
            # MQTT client will be created when needed
            self.logger.info("MQTT client configuration ready")
        except Exception as e:
            self.logger.error(f"Failed to configure MQTT: {e}")
            self.notification_methods['mqtt'] = False
    
    def _setup_logging(self) -> None:
        """Setup advanced logging for alerts."""
        log_file = self.config.logs_dir / "advanced_alerts.log"
        
        # Create file handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Create alert logger
        alert_logger = logging.getLogger('advanced_alerts')
        alert_logger.setLevel(logging.INFO)
        alert_logger.addHandler(file_handler)
    
    def _is_on_cooldown(self, person_name: str, alert_type: str) -> bool:
        """Check if alert is on cooldown."""
        cooldown_key = f"{person_name}_{alert_type}"
        
        if cooldown_key not in self.alert_cooldowns:
            return False
        
        cooldown_period = timedelta(seconds=self.config.alerts.alert_cooldown)
        time_since_last = datetime.now() - self.alert_cooldowns[cooldown_key]
        
        return time_since_last < cooldown_period
    
    def _update_cooldown(self, person_name: str, alert_type: str) -> None:
        """Update alert cooldown."""
        cooldown_key = f"{person_name}_{alert_type}"
        self.alert_cooldowns[cooldown_key] = datetime.now()
    
    async def _send_desktop_notification(self, title: str, message: str, 
                                       priority: str = "normal") -> bool:
        """Send desktop notification using best available method."""
        success = False
        
        # Try Windows toast notifications first
        if self.notification_methods['win10_toast'] and self.win10_toast:
            try:
                self.win10_toast.show_toast(
                    title,
                    message,
                    duration=self.config.alerts.notification_timeout,
                    threaded=True
                )
                success = True
                self.logger.debug(f"Windows toast notification sent: {title}")
            except Exception as e:
                self.logger.warning(f"Windows toast notification failed: {e}")
        
        # Fallback to plyer
        if not success and self.notification_methods['plyer']:
            try:
                notification.notify(
                    title=title,
                    message=message,
                    timeout=self.config.alerts.notification_timeout,
                    app_name="Facial Recognition System"
                )
                success = True
                self.logger.debug(f"Plyer notification sent: {title}")
            except Exception as e:
                self.logger.warning(f"Plyer notification failed: {e}")
        
        return success
    
    async def _play_sound_alert(self, alert_type: str = "default") -> bool:
        """Play sound alert with different sounds for different alert types."""
        if not self.notification_methods['pygame']:
            return False
        
        try:
            # Define different sounds for different alert types
            sound_configs = {
                'recognition': {'frequency': 800, 'duration': 0.3},
                'unknown': {'frequency': 600, 'duration': 0.5},
                'unauthorized': {'frequency': 400, 'duration': 1.0},
                'intrusion': {'frequency': 300, 'duration': 2.0}
            }
            
            config = sound_configs.get(alert_type, sound_configs['recognition'])
            
            # Generate tone
            sample_rate = 22050
            frames = int(config['duration'] * sample_rate)
            arr = []
            
            for i in range(frames):
                time_point = float(i) / sample_rate
                wave = 4096 * np.sin(2 * np.pi * config['frequency'] * time_point)
                arr.append([wave, wave])  # Stereo
            
            sound = pygame.sndarray.make_sound(np.array(arr, dtype=np.int16))
            sound.play()
            
            self.logger.debug(f"Sound alert played for {alert_type}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Sound alert failed: {e}")
            return False
    
    async def _send_email_alert(self, title: str, message: str, 
                               alert_data: Dict[str, Any]) -> bool:
        """Send email alert."""
        if not all([self.email_config['smtp_server'], 
                   self.email_config['username'], 
                   self.email_config['to_emails']]):
            return False
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"[Facial Recognition Alert] {title}"
            
            # Create detailed email body
            body = f"""
            Alert: {title}
            Message: {message}
            
            Details:
            - Timestamp: {alert_data.get('timestamp', 'N/A')}
            - Alert Type: {alert_data.get('type', 'N/A')}
            - Person: {alert_data.get('person', 'N/A')}
            - Confidence: {alert_data.get('confidence', 'N/A')}
            - Location: {alert_data.get('location', 'N/A')}
            
            Additional Information:
            {json.dumps(alert_data.get('additional_info', {}), indent=2)}
            
            --
            Facial Recognition System
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent: {title}")
            return True
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
            return False
    
    async def _send_webhook_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send webhook alert to configured URLs."""
        if not self.webhook_config['urls']:
            return False
        
        success = False
        
        for url in self.webhook_config['urls']:
            for attempt in range(self.webhook_config['retry_attempts']):
                try:
                    response = requests.post(
                        url,
                        json=alert_data,
                        timeout=self.webhook_config['timeout'],
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()
                    
                    self.logger.info(f"Webhook alert sent to {url}")
                    success = True
                    break
                    
                except Exception as e:
                    self.logger.warning(f"Webhook attempt {attempt + 1} failed for {url}: {e}")
                    if attempt < self.webhook_config['retry_attempts'] - 1:
                        await asyncio.sleep(1)  # Wait before retry
        
        return success
    
    async def _send_mqtt_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send MQTT alert for IoT integration."""
        if not self.notification_methods['mqtt']:
            return False
        
        try:
            topic = f"{self.mqtt_config['topic_prefix']}/alerts"
            
            async with aiomqtt.Client(
                hostname=self.mqtt_config['broker'],
                port=self.mqtt_config['port'],
                username=self.mqtt_config.get('username'),
                password=self.mqtt_config.get('password')
            ) as client:
                await client.publish(topic, json.dumps(alert_data))
            
            self.logger.info("MQTT alert sent")
            return True
            
        except Exception as e:
            self.logger.error(f"MQTT alert failed: {e}")
            return False
    
    async def trigger_alert(self, alert_type: str, person_name: str, confidence: float,
                          additional_info: Dict[str, Any] = None, force: bool = False) -> bool:
        """
        Trigger a comprehensive alert using multiple notification methods.
        
        Args:
            alert_type: Type of alert
            person_name: Person detected
            confidence: Detection confidence
            additional_info: Additional alert data
            force: Force alert even if on cooldown
        
        Returns:
            True if alert was processed successfully
        """
        # Check cooldown
        if not force and self._is_on_cooldown(person_name, alert_type):
            return False
        
        timestamp = datetime.now()
        
        # Create comprehensive alert data
        alert_data = {
            'timestamp': timestamp.isoformat(),
            'type': alert_type,
            'person': person_name,
            'confidence': confidence,
            'location': additional_info.get('location') if additional_info else None,
            'additional_info': additional_info or {},
            'system_info': {
                'hostname': platform.node(),
                'platform': platform.system(),
                'version': '1.0.0'
            }
        }
        
        # Generate alert messages
        title, message = self._generate_alert_messages(alert_type, person_name, confidence)
        
        # Get notification methods for this alert type
        methods = self.alert_rules.get(alert_type, ['desktop', 'log'])
        
        # Track notification results
        notification_results = {}
        
        try:
            # Desktop notification
            if 'desktop' in methods:
                result = await self._send_desktop_notification(title, message, alert_type)
                notification_results['desktop'] = result
            
            # Sound alert
            if 'sound' in methods:
                result = await self._play_sound_alert(alert_type)
                notification_results['sound'] = result
            
            # Email alert
            if 'email' in methods:
                result = await self._send_email_alert(title, message, alert_data)
                notification_results['email'] = result
            
            # Webhook alert
            if 'webhook' in methods:
                result = await self._send_webhook_alert(alert_data)
                notification_results['webhook'] = result
            
            # MQTT alert
            if 'mqtt' in methods:
                result = await self._send_mqtt_alert(alert_data)
                notification_results['mqtt'] = result
            
            # Logging (always enabled)
            self._log_alert(alert_data)
            notification_results['log'] = True
            
            # Update statistics
            self._update_alert_stats(alert_type, person_name, notification_results)
            
            # Update cooldown
            self._update_cooldown(person_name, alert_type)
            
            # Add to alert history
            alert_data['notification_results'] = notification_results
            self.alert_history.append(alert_data)
            
            # Maintain history size
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            self.logger.info(f"Alert processed: {title} - Methods: {list(notification_results.keys())}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Alert processing failed: {e}")
            return False
    
    def _generate_alert_messages(self, alert_type: str, person_name: str, 
                               confidence: float) -> Tuple[str, str]:
        """Generate alert title and message."""
        if alert_type == "recognition":
            title = "Person Recognized"
            message = f"{person_name} detected with {confidence:.1%} confidence"
        elif alert_type == "unknown":
            title = "Unknown Person Detected"
            message = f"Unidentified person detected (confidence: {confidence:.1%})"
        elif alert_type == "unauthorized":
            title = "⚠️ Unauthorized Access Alert"
            message = f"Unauthorized person: {person_name} (confidence: {confidence:.1%})"
        elif alert_type == "intrusion":
            title = "🚨 SECURITY BREACH"
            message = f"Potential intrusion detected: {person_name} (confidence: {confidence:.1%})"
        else:
            title = "Facial Recognition Alert"
            message = f"{alert_type}: {person_name} (confidence: {confidence:.1%})"
        
        return title, message
    
    def _log_alert(self, alert_data: Dict[str, Any]) -> None:
        """Log alert with structured data."""
        alert_logger = logging.getLogger('advanced_alerts')
        
        log_message = (
            f"ALERT - {alert_data['type']} - {alert_data['person']} "
            f"(confidence: {alert_data['confidence']:.2f}) - "
            f"Location: {alert_data.get('location', 'N/A')}"
        )
        
        alert_logger.info(log_message)
        alert_logger.debug(f"Full alert data: {json.dumps(alert_data, indent=2)}")
    
    def _update_alert_stats(self, alert_type: str, person_name: str, 
                          notification_results: Dict[str, bool]) -> None:
        """Update alert statistics."""
        self.alert_stats['total_alerts'] += 1
        
        # Update by type
        if alert_type not in self.alert_stats['alerts_by_type']:
            self.alert_stats['alerts_by_type'][alert_type] = 0
        self.alert_stats['alerts_by_type'][alert_type] += 1
        
        # Update by person
        if person_name not in self.alert_stats['alerts_by_person']:
            self.alert_stats['alerts_by_person'][person_name] = 0
        self.alert_stats['alerts_by_person'][person_name] += 1
        
        # Update by method
        for method, success in notification_results.items():
            if method not in self.alert_stats['alerts_by_method']:
                self.alert_stats['alerts_by_method'][method] = {'success': 0, 'failure': 0}
            
            if success:
                self.alert_stats['alerts_by_method'][method]['success'] += 1
            else:
                self.alert_stats['alerts_by_method'][method]['failure'] += 1
    
    def configure_email(self, smtp_server: str, smtp_port: int, username: str, 
                       password: str, from_email: str, to_emails: List[str]) -> None:
        """Configure email notification settings."""
        self.email_config.update({
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_email': from_email,
            'to_emails': to_emails
        })
        self.logger.info("Email configuration updated")
    
    def configure_webhooks(self, urls: List[str], timeout: int = 5, 
                          retry_attempts: int = 3) -> None:
        """Configure webhook settings."""
        self.webhook_config.update({
            'urls': urls,
            'timeout': timeout,
            'retry_attempts': retry_attempts
        })
        self.logger.info(f"Webhook configuration updated: {len(urls)} URLs")
    
    def configure_mqtt(self, broker: str, port: int = 1883, topic_prefix: str = "facial_recognition",
                      username: str = "", password: str = "") -> None:
        """Configure MQTT settings."""
        self.mqtt_config.update({
            'broker': broker,
            'port': port,
            'topic_prefix': topic_prefix,
            'username': username,
            'password': password
        })
        self.logger.info(f"MQTT configuration updated: {broker}:{port}")
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        stats = self.alert_stats.copy()
        stats['notification_methods_available'] = self.notification_methods
        stats['active_cooldowns'] = len(self.alert_cooldowns)
        stats['history_size'] = len(self.alert_history)
        
        # Calculate success rates
        for method, data in stats['alerts_by_method'].items():
            total = data['success'] + data['failure']
            if total > 0:
                data['success_rate'] = data['success'] / total
            else:
                data['success_rate'] = 0.0
        
        return stats
    
    def clear_cooldowns(self) -> None:
        """Clear all alert cooldowns."""
        self.alert_cooldowns.clear()
        self.logger.info("All alert cooldowns cleared")
    
    def clear_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        self.logger.info("Alert history cleared")
    
    def reset_stats(self) -> None:
        """Reset all alert statistics."""
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_person': {},
            'alerts_by_method': {},
            'response_times': [],
            'escalation_count': 0
        }
        self.clear_cooldowns()
        self.clear_history()
        self.logger.info("Alert statistics reset")
    
    # System tray callbacks
    def _show_tray_stats(self, icon, item) -> None:
        """Show stats from system tray."""
        stats = self.get_alert_stats()
        message = f"Total Alerts: {stats['total_alerts']}\nActive Cooldowns: {stats['active_cooldowns']}"
        
        if self.notification_methods['win10_toast'] and self.win10_toast:
            self.win10_toast.show_toast(
                "Alert Statistics",
                message,
                duration=5
            )
    
    def _clear_alerts_tray(self, icon, item) -> None:
        """Clear alerts from system tray."""
        self.clear_cooldowns()
        
        if self.notification_methods['win10_toast'] and self.win10_toast:
            self.win10_toast.show_toast(
                "Alerts Cleared",
                "All alert cooldowns have been cleared",
                duration=3
            )
    
    def _exit_tray(self, icon, item) -> None:
        """Exit from system tray."""
        icon.stop()
    
    async def test_notifications(self) -> Dict[str, bool]:
        """Test all notification methods."""
        results = {}
        
        test_alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'test',
            'person': 'Test User',
            'confidence': 0.95,
            'additional_info': {'test': True}
        }
        
        # Test desktop notification
        results['desktop'] = await self._send_desktop_notification(
            "Test Notification",
            "Facial Recognition System test alert"
        )
        
        # Test sound alert
        results['sound'] = await self._play_sound_alert('test')
        
        # Test webhook (if configured)
        if self.webhook_config['urls']:
            results['webhook'] = await self._send_webhook_alert(test_alert_data)
        
        # Test MQTT (if configured)
        if self.notification_methods['mqtt']:
            results['mqtt'] = await self._send_mqtt_alert(test_alert_data)
        
        # Test email (if configured)
        if self.email_config['smtp_server']:
            results['email'] = await self._send_email_alert(
                "Test Email Alert",
                "This is a test email from the Facial Recognition System",
                test_alert_data
            )
        
        return results
    
    def cleanup(self) -> None:
        """Clean up alert system resources."""
        if self.system_tray:
            self.system_tray.stop()
        
        if self.notification_methods['pygame']:
            pygame.mixer.quit()
