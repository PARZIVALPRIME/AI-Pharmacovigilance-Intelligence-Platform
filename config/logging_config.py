"""
Logging configuration for AI Pharmacovigilance Intelligence Platform.
Uses loguru for structured, production-grade logging.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging(
    level: str = "INFO",
    log_dir: Path = LOG_DIR,
    rotation: str = "10 MB",
    retention: str = "30 days",
) -> None:
    """Configure loguru with console + rotating file handlers."""
    logger.remove()  # Remove default handler

    # Console handler with colours
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # General application log
    logger.add(
        log_dir / "app.log",
        level=level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )

    # Error-only log for alerting / ops
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        encoding="utf-8",
    )

    logger.info("Logging initialised at level={}", level)


# Auto-configure on import with sensible defaults
configure_logging()

__all__ = ["logger", "configure_logging"]
