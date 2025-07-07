"""
Notification settings manager for facial recognition system
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class NotificationSettings:
    """
    Manages notification settings for the face recognition system

    Handles loading, saving, and accessing notification configuration
    """

    def __init__(self, settings_dir: Optional[Path] = None):
        """
        Initialize notification settings manager

        Args:
            settings_dir: Directory for settings storage (defaults to data/settings)
        """
        self.settings_dir = settings_dir or Path("data/settings")
        self.settings_file = self.settings_dir / "notification_settings.json"

        # Default settings
        self.settings = {
            'enabled': True,
            'cooldown': 10,  # seconds between notifications for the same person
            'sound': True,    # enable sound notifications
            'desktop': True,  # enable desktop notifications
            'console': True,  # enable console notifications
            'min_confidence': 0.7,  # minimum confidence threshold for notifications
            'notification_type': 0,  # 0: Pop-up, 1: Status Bar, 2: Both
            'notify_unknown': False  # whether to notify for unknown faces
        }

        # Create settings directory if it doesn't exist
        self.settings_dir.mkdir(parents=True, exist_ok=True)

        # Load settings if file exists
        self.load_settings()

    def load_settings(self) -> bool:
        """
        Load settings from file

        Returns:
            True if settings were loaded successfully, False otherwise
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Update settings, preserving defaults for any missing keys
                    self.settings.update(loaded_settings)
                logger.info("Notification settings loaded")
                return True
            else:
                logger.info("Notification settings file not found, using defaults")
                return False
        except Exception as e:
            logger.warning(f"Failed to load notification settings: {e}")
            return False

    def save_settings(self) -> bool:
        """
        Save settings to file

        Returns:
            True if settings were saved successfully, False otherwise
        """
        try:
            # Ensure directory exists
            self.settings_dir.mkdir(parents=True, exist_ok=True)

            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)

            logger.info("Notification settings saved")
            return True
        except Exception as e:
            logger.error(f"Failed to save notification settings: {e}")
            return False

    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update settings with new values

        Args:
            new_settings: Dictionary with new setting values
        """
        if new_settings:
            self.settings.update(new_settings)
            logger.info("Notification settings updated")

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all notification settings

        Returns:
            Dictionary containing all notification settings
        """
        return self.settings.copy()

    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value

        Args:
            key: Setting key
            default: Default value if key doesn't exist

        Returns:
            Setting value or default
        """
        return self.settings.get(key, default)
