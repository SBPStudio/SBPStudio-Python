#!/usr/bin/env python3
"""
TopasSuite_CLI.py — Launcher for the TOPAS Suite command-line interface.

Run directly (``python applications/TopasSuite_CLI.py info file.sgy``) or via
the package entry point (``python -m topassuite.cli.main``). This thin wrapper
only ensures the repository root is importable when executed as a standalone
script, then hands off to :func:`topassuite.cli.main.main`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly from a checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from topassuite.cli.main import main  # noqa: E402

if __name__ == "__main__":
    main()
