#!/usr/bin/env python3
"""
SBPStudio_CLI.py — Launcher for the SBP Studio command-line interface.

Run directly (``python applications/SBPStudio_CLI.py info file.sgy``) or via
the package entry point (``python -m sbp_studio.cli.main``). This thin wrapper
only ensures the repository root is importable when executed as a standalone
script, then hands off to :func:`sbp_studio.cli.main.main`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly from a checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sbp_studio.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
