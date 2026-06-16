"""
rthook_utf8.py — PyInstaller runtime hook: make the frozen app's console I/O
UTF-8 and crash-proof on legacy Windows code pages.

The CLI prints non-ASCII characters (→, ×, µ, …) in help text and progress
output. On a default Windows console (cp1252/cp850) the frozen interpreter
otherwise raises UnicodeEncodeError and aborts — even on `--help`. This hook
runs before the entry script and:

  1. Switches the Windows console to the UTF-8 code page (65001) so modern
     terminals render the characters correctly.
  2. Reconfigures stdout/stderr to UTF-8 with 'backslashreplace' so output can
     never crash, even on a terminal that ignores the code-page change.

In windowed (GUI) mode stdout/stderr may be None; those cases are skipped.
"""
import sys

if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
