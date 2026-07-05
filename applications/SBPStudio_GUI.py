#!/usr/bin/env python3
"""
SBPStudio_GUI.py — Launcher for the SBP Studio PyQt6 desktop application.

Run directly (``python applications/SBPStudio_GUI.py``) or via the package
entry point (``python -m sbp_studio.gui``). This thin wrapper only ensures the
repository root is importable when executed as a standalone script, then hands
off to :func:`sbp_studio.gui.app.main`.

Import-order contract (audit-hardened)
---------------------------------------
NOTHING heavy may be imported at module level here. In the frozen exe a
multiprocessing worker (the parallel batch export) re-executes THIS script from
the top until ``freeze_support()`` intercepts it — every module-level import is
paid by EVERY worker. With the GUI import deferred until after
``freeze_support()``, a worker process never imports PyQt6 at all: it runs its
headless render target and exits, keeping per-worker RAM at the core+viz stack
only. The same deferral also keeps dev-mode spawn children (which re-import
this module as ``__mp_main__``) free of the Qt import.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly from a checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ == "__main__":
    # PyInstaller frozen-exe worker guard — MUST run before anything else. When
    # a ProcessPoolExecutor (e.g. the parallel batch export) spawns a worker on
    # Windows, the child re-executes THIS .exe; freeze_support() detects the
    # multiprocessing bootstrap and runs the worker target instead of falling
    # through to main() — the difference between a worker rendering a file and
    # an infinite cascade of GUI windows. No-op in an unfrozen dev checkout.
    import multiprocessing
    multiprocessing.freeze_support()

    # Heavy import ONLY on the real GUI path (see the module docstring).
    from sbp_studio.gui.app import main
    sys.exit(main())
