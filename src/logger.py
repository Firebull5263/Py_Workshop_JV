"""
Application logging utilities.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
