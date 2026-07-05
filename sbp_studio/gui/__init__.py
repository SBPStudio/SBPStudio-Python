"""
sbp_studio.gui — PyQt6 presentation layer for SBP Studio.

Strictly decoupled from the computational core: this package only *imports and
calls* ``sbp_studio.core`` functions; it never duplicates DSP/I-O logic and the
core never imports Qt. Heavy work runs in background workers
(:mod:`sbp_studio.gui.workers`); interactive views use PyQtGraph/QtCharts.

Entry point:
    python -m sbp_studio.gui

Lazy public API (PEP 562)
--------------------------
``create_app`` / ``main`` / ``MainWindow`` resolve on first ACCESS instead of
at package import. This keeps ``import sbp_studio.gui.<pure submodule>``
(e.g. ``gui.export_headless``, ``gui.dsp.nodes``, ``gui.tabs._render``)
completely PyQt6-free — the property that lets a spawned batch-export worker
process import the export render pipeline without paying the Qt stack. GUI
callers are unaffected: ``from sbp_studio.gui import main`` works exactly as
before, just resolved at the access point.
"""
from __future__ import annotations

__all__ = ["create_app", "main", "MainWindow"]


def __getattr__(name: str):
    if name in ("create_app", "main"):
        from .app import create_app, main
        return {"create_app": create_app, "main": main}[name]
    if name == "MainWindow":
        from .main_window import MainWindow
        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
