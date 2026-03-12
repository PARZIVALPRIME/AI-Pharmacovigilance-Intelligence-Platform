"""Config package initializer."""
from .settings import get_settings, settings
from .logging_config import configure_logging

try:
    from .logging_config import logger
except Exception:
    import logging
    logger = logging.getLogger(__name__)

__all__ = ["settings", "get_settings", "configure_logging", "logger"]
