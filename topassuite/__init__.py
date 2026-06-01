"""
topassuite — Headless computational core for TOPAS SEG-Y seismic profiles.

Package structure:
  topassuite.core  — processing, I/O, model, constants (GUI-free)
  topassuite.viz   — headless figure rendering (Agg/Pillow)
  topassuite.cli   — command-line interface

The original GUI (TopasSUITE.py) remains unchanged and continues to
work independently of this package.
"""

__version__ = "0.1.0"
