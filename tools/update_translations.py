#!/usr/bin/env python3
"""
update_translations.py — Regenerate and (optionally) compile the GUI translations.

Workflow
--------
1. ``pylupdate6`` scans the GUI source for ``tr()`` / ``translate()`` calls and
   MERGES them into each ``topassuite_<lang>.ts`` file — existing translations
   are preserved, new strings are added as ``unfinished``, removed ones marked
   ``vanished``. Edit the ``.ts`` files (Qt Linguist or by hand) to translate.

2. If ``lrelease`` is on PATH, each ``.ts`` is compiled to a ``.qm``. The app
   loads the ``.qm`` automatically when present; otherwise it parses the ``.ts``
   directly (see topassuite/gui/i18n.py), so compilation is optional.

English is the base/source language and has no ``.ts`` file.

Usage
-----
    python tools/update_translations.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GUI = REPO / "topassuite" / "gui"
TRANSLATIONS = GUI / "translations"
LANGS = ["es"]  # languages with a .ts file (English is the source language)


def _sources() -> list[str]:
    return [str(p) for p in sorted(GUI.rglob("*.py"))]


def main() -> int:
    TRANSLATIONS.mkdir(exist_ok=True)
    pylupdate = shutil.which("pylupdate6")
    if not pylupdate:
        print("ERROR: pylupdate6 not found on PATH.", file=sys.stderr)
        return 1

    for lang in LANGS:
        ts = TRANSLATIONS / f"topassuite_{lang}.ts"
        print(f"[pylupdate6] -> {ts.name}")
        subprocess.run([pylupdate, *_sources(), "-ts", str(ts)], check=True)

    lrelease = shutil.which("lrelease") or shutil.which("lrelease-qt6")
    if lrelease:
        for lang in LANGS:
            ts = TRANSLATIONS / f"topassuite_{lang}.ts"
            qm = TRANSLATIONS / f"topassuite_{lang}.qm"
            print(f"[lrelease] {ts.name} -> {qm.name}")
            subprocess.run([lrelease, str(ts), "-qm", str(qm)], check=True)
    else:
        print("NOTE: lrelease not found — skipping .qm compilation. "
              "The app will parse the .ts files directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
