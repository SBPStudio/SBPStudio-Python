"""Enables ``python -m sbp_studio.gui``."""
import sys

if __name__ == "__main__":
    # Frozen/spawn worker guard + deferred heavy import — same import-order
    # contract as applications/SBPStudio_GUI.py (see its module docstring):
    # freeze_support() first, so a multiprocessing worker never falls through
    # to the GUI, and the PyQt6 import is paid only on the real GUI path.
    import multiprocessing
    multiprocessing.freeze_support()

    from .app import main
    sys.exit(main())
