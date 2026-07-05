"""Main tool tabs for the SBP Studio GUI (one encapsulated class each).

Lazy public API (PEP 562): the tab classes resolve on first access so that
importing a PURE submodule of this package (``_render``'s scale/DPI math,
used by the headless batch-export worker) never drags in PyQt6 — see
``sbp_studio.gui.__init__``'s docstring for the full rationale.
"""
from __future__ import annotations

__all__ = ["VisualizerTab", "ReprojectorTab"]


def __getattr__(name: str):
    if name == "VisualizerTab":
        from .visualizer_tab import VisualizerTab
        return VisualizerTab
    if name == "ReprojectorTab":
        from .reprojector_tab import ReprojectorTab
        return ReprojectorTab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
