from __future__ import annotations

import logging


def get_logger(name: str = "oviqs") -> logging.Logger:
    return logging.getLogger(name)
