"""
sbp_studio.gui — PyQt6 presentation layer for SBP Studio.

Strictly decoupled from the computational core: this package only *imports and
calls* ``sbp_studio.core`` functions; it never duplicates DSP/I-O logic and the
core never imports Qt. Heavy work runs in background workers
(:mod:`sbp_studio.gui.workers`); interactive views use PyQtGraph/QtCharts.

Entry point:
    python -m sbp_studio.gui
"""
from .app import create_app, main
from .main_window import MainWindow

__all__ = ["create_app", "main", "MainWindow"]
