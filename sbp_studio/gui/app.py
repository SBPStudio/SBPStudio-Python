"""
app.py — Application bootstrap.

Creates the ``QApplication``, installs the dark theme and the default-language
translator, builds the :class:`MainWindow`, and starts the Qt event loop. This
is the single place where the GUI is assembled; the window is import-safe and
can also be constructed directly from tests.
"""
from __future__ import annotations

import sys
from typing import Optional

from PyQt6.QtWidgets import QApplication

from .i18n import language_manager
from .main_window import MainWindow
from .theme import apply_theme


def create_app(argv: Optional[list[str]] = None) -> QApplication:
    """Return a themed, translated ``QApplication`` (reusing an existing one)."""
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    apply_theme(app)                 # install the dark theme stylesheet
    language_manager.install(app)    # install the default-language translator
    return app


def main(argv: Optional[list[str]] = None) -> int:
    """Build the window and run the event loop. Returns the process exit code."""
    # Configure persistent logging FIRST so any startup/runtime error (incl. a
    # crash in a windowed .exe with no console) is captured in app.log. Done here
    # in the entry point — NOT in MainWindow — so tests that build the window
    # directly never write a log file.
    from sbp_studio.core import configure_logging
    configure_logging()
    app = create_app(argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
