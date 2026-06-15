"""
sbp_studio — Headless computational core for SBP SEG-Y seismic profiles.

Package structure:
  sbp_studio.core  — processing, I/O, model, constants (GUI-free)
  sbp_studio.viz   — headless figure rendering (Agg/Pillow)
  sbp_studio.cli   — command-line interface

The original GUI (TopasSUITE.py) remains unchanged and continues to
work independently of this package.
"""

__version__ = "0.5.0"
