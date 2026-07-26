"""
utils/logger.py — Simple logging setup used across all modules.
"""

import logging
import sys


def get_logger(name: str = "fraud_detector") -> logging.Logger:
    """Return a configured logger. Call once per module."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
