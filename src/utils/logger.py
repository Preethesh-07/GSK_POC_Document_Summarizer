"""
Structured logging setup with Rich handler.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a named logger with Rich console output.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
        )
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(name)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
