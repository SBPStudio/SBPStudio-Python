"""
test_logging.py — Persistent logging infrastructure (core/logger.py).

Fast, no heavy I/O. Each test that touches the shared ``topassuite`` logger uses
the ``fresh_logger`` fixture, which snapshots and fully restores the logger's
handler state so the rest of the suite stays silent (no stray app.log).
"""
from __future__ import annotations

import logging
import os

import pytest

from topassuite.core import logger as L


@pytest.fixture
def fresh_logger():
    """Give a test a clean (unconfigured) ``topassuite`` logger, then restore."""
    lg = L.get_logger()
    saved_handlers = lg.handlers[:]
    saved = (lg.level, lg.propagate, getattr(lg, "_topas_configured", None))
    lg.handlers.clear()
    if hasattr(lg, "_topas_configured"):
        del lg._topas_configured
    try:
        yield lg
    finally:
        for h in lg.handlers[:]:                 # close file handlers on tmp dirs
            try:
                h.close()
            except Exception:
                pass
        lg.handlers[:] = saved_handlers
        lg.setLevel(saved[0])
        lg.propagate = saved[1]
        if saved[2] is not None:
            lg._topas_configured = saved[2]
        elif hasattr(lg, "_topas_configured"):
            del lg._topas_configured


def test_app_root_is_project_root():
    root = L.app_root()
    assert os.path.isdir(os.path.join(root, "topassuite"))


def test_get_logger_names():
    assert L.get_logger().name == "topassuite"
    assert L.get_logger("io_segy").name == "topassuite.io_segy"


def test_default_is_silent_until_configured():
    # Importing the core must NOT print or write — only a NullHandler, no propagate.
    lg = L.get_logger()
    assert any(isinstance(h, logging.NullHandler) for h in lg.handlers)
    assert lg.propagate is False


def test_configure_writes_rotating_app_log(fresh_logger, tmp_path):
    L.configure_logging(log_dir=str(tmp_path), console=False)
    L.get_logger("io_segy").error("boom-marker")
    for h in fresh_logger.handlers:
        h.flush()
    p = tmp_path / L.LOG_FILENAME
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "boom-marker" in text and "ERROR" in text and "topassuite.io_segy" in text


def test_configure_is_idempotent(fresh_logger, tmp_path):
    L.configure_logging(log_dir=str(tmp_path), console=False)
    n = len(fresh_logger.handlers)
    L.configure_logging(log_dir=str(tmp_path), console=False)   # second call → no-op
    assert len(fresh_logger.handlers) == n


def test_exception_traceback_is_captured(fresh_logger, tmp_path):
    L.configure_logging(log_dir=str(tmp_path), console=False)
    try:
        raise ValueError("kapow")
    except ValueError:
        L.get_logger("worker").exception("task blew up")
    for h in fresh_logger.handlers:
        h.flush()
    text = (tmp_path / L.LOG_FILENAME).read_text(encoding="utf-8")
    # The full traceback must survive — this is the windowed-.exe failure case.
    assert "task blew up" in text
    assert "ValueError: kapow" in text and "Traceback" in text


def test_unwritable_log_dir_falls_back_to_console(fresh_logger, tmp_path):
    # A bogus directory must not crash configure_logging (degrades to console).
    bogus = str(tmp_path / "does" / "not" / "exist")
    L.configure_logging(log_dir=bogus, console=True)
    assert getattr(fresh_logger, "_topas_configured", False) is True
    assert not os.path.exists(os.path.join(bogus, L.LOG_FILENAME))
