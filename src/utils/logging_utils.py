"""Logging configuration and utilities for reproducible MLOps execution."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "aqi_assistant",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configures and returns a standard logger with consistent formatting.

    Args:
        name: Name of the logger, typically __name__.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should also be written.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional file handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
