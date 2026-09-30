"""Logging helpers built on the standard library.

Library modules obtain loggers with :func:`get_logger` and never add handlers
themselves. Entry points (scripts, notebooks) call :func:`configure_logging`
once to attach console and optional file output.
"""

from __future__ import annotations

import logging
from pathlib import Path

PACKAGE_LOGGER = "seismic_ml"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_HANDLER_MARKER = "_seismic_ml_handler"


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``seismic_ml`` hierarchy.

    ``get_logger("training")`` and ``get_logger("seismic_ml.training")`` return
    the same logger.
    """
    if name != PACKAGE_LOGGER and not name.startswith(PACKAGE_LOGGER + "."):
        name = f"{PACKAGE_LOGGER}.{name}"
    return logging.getLogger(name)


def configure_logging(
    level: int | str = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Attach console (and optionally file) handlers to the package logger.

    Idempotent: handlers added by a previous call are replaced, so calling it
    twice does not duplicate log lines.

    Args:
        level: Logging level name or number.
        log_file: Optional path of a log file (parent directories are created).

    Returns:
        The configured ``seismic_ml`` logger.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)

    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    formatter = logging.Formatter(_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    # Avoid duplicate lines if the root logger is also configured.
    logger.propagate = False
    return logger
