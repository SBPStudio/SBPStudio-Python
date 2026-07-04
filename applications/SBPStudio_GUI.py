#!/usr/bin/env python3
"""
SBPStudio_GUI.py — Launcher for the SBP Studio PyQt6 desktop application.

Run directly (``python applications/SBPStudio_GUI.py``) or via the package
entry point (``python -m sbp_studio.gui``). This thin wrapper only ensures the
repository root is importable when executed as a standalone script, then hands
off to :func:`sbp_studio.gui.app.main`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly from a checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sbp_studio.gui.app import main  # noqa: E402

if __name__ == "__main__":
    # PyInstaller frozen-exe worker guard — MUST run before main(). When a
    # ProcessPoolExecutor (e.g. the parallel batch export) spawns a worker on
    # Windows, the child re-executes THIS .exe; freeze_support() detects the
    # multiprocessing bootstrap and runs the worker target instead of falling
    # through to main() — the difference between a worker rendering a file and
    # an infinite cascade of GUI windows. No-op in an unfrozen dev checkout.
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
