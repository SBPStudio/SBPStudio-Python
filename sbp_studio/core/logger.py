"""
logger.py — Application-wide logging (rotating file + console).

Separation of concerns: this module is pure stdlib (no GUI, no segyio) and lives
in the core. Library/core code obtains a logger via :func:`get_logger`; the
APPLICATION entry points (the GUI bootstrap ``gui.app.main`` and the headless
CLI) call :func:`configure_logging` ONCE at startup to attach the rotating
``app.log`` file handler and a console handler.

Until ``configure_logging`` runs, the ``sbp_studio`` logger carries only a
``NullHandler`` (and does not propagate), so importing the core never writes
files or prints — keeping the test suite and any library use clean and silent.
This is the standard "library logs, application configures" pattern.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

LOGGER_NAME = "sbp_studio"
LOG_FILENAME = "app.log"
_MAX_BYTES = 2_000_000          # ~2 MB per file
_BACKUPS = 5                    # app.log + app.log.1 … app.log.5
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def app_root() -> str:
    """Directory where ``app.log`` is written.

    * Frozen ``.exe`` → the folder CONTAINING the executable (next to it on the
      user's disk), so field logs sit beside the app.
    * Source run → the project root (parents[2] of this file).

    Mirrors the CLI's relative-path resolution so logs land alongside the app
    rather than in a temp folder or an unpredictable working directory."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return str(Path(__file__).resolve().parents[2])


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the shared ``sbp_studio`` logger, or a dotted child (e.g.
    ``get_logger("io_segy")`` → ``sbp_studio.io_segy``)."""
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


# Library default: stay completely silent until an application configures
# handlers (no "No handlers could be found" noise, no propagation to root).
_root_logger = get_logger()
_root_logger.addHandler(logging.NullHandler())
_root_logger.propagate = False


def configure_logging(level: int = logging.INFO,
                      log_dir: Optional[str] = None,
                      console: bool = True) -> logging.Logger:
    """Attach a rotating ``app.log`` file handler (+ optional console handler) to
    the ``sbp_studio`` logger and return it.

    Idempotent — only the first call wires handlers; later calls are no-ops, so
    it's safe to call from multiple entry points. If the target directory is not
    writable (e.g. a read-only install path), logging degrades to console-only
    rather than crashing startup."""
    logger = get_logger()
    if getattr(logger, "_topas_configured", False):
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    log_path = os.path.join(log_dir or app_root(), LOG_FILENAME)
    try:
        fh = RotatingFileHandler(log_path, maxBytes=_MAX_BYTES,
                                 backupCount=_BACKUPS, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Logging started -> %s", log_path)
    except OSError as exc:
        logger.warning("Could not open log file %s (%s); console only.",
                       log_path, exc)

    logger._topas_configured = True            # type: ignore[attr-defined]
    return logger
