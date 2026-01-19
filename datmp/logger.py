"""
Unified logging configuration for the datmp project.

This module provides a centralized logger configuration that can be used
throughout the entire project. It sets up consistent logging format and
handlers for all modules.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "datmp",
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Set up and configure a logger for the datmp project.

    Args:
        name: Logger name. Defaults to "datmp".
        level: Logging level (e.g., logging.INFO, logging.DEBUG, "INFO", "DEBUG").
               Can be an integer or string.
        log_file: Optional path to log file. If provided, logs will also be written to file.
        format_string: Optional custom format string. If None, uses default format.
        date_format: Date format string for log timestamps.

    Returns:
        Configured logger instance.

    Examples:
        >>> logger = setup_logger()
        >>> logger.info("This is an info message")

        >>> logger = setup_logger(level=logging.DEBUG, log_file="logs/app.log")
        >>> logger.debug("Debug message")
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if logger already configured
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Console handler (always add)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file is provided)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger instance. If name is provided, returns a child logger.
    Otherwise, returns the main datmp logger.

    Args:
        name: Optional logger name. If provided, creates a child logger
              (e.g., "datmp.dataset" or "datmp.model").

    Returns:
        Logger instance.

    Examples:
        >>> logger = get_logger()  # Returns main datmp logger
        >>> logger = get_logger("dataset")  # Returns datmp.dataset logger
    """
    if name is None:
        # Return main datmp logger
        return logging.getLogger("datmp")
    else:
        # Return child logger
        return logging.getLogger(f"datmp.{name}")


# Initialize the main datmp logger
logger = setup_logger("datmp", level=logging.INFO)

__all__ = ["setup_logger", "get_logger", "logger"]
