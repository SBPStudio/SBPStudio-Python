"""
topassuite.gui — PyQt6 presentation layer for TOPAS Suite.

Strictly decoupled from the computational core: this package only *imports and
calls* ``topassuite.core`` functions; it never duplicates DSP/I-O logic and the
core never imports Qt. Heavy work runs in background workers
(:mod:`topassuite.gui.workers`); interactive views use PyQtGraph/QtCharts.

Entry point:
    python -m topassuite.gui
"""
from .app import create_app, main
from .main_window import MainWindow

__all__ = ["create_app", "main", "MainWindow"]
