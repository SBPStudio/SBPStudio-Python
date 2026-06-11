"""
reprojector_tab.py — Tab B · Reprojector.

CRS management (EPSG/WKT) and navigation-line export (SHP / GeoJSON / CSV).

Scaffold stage: placeholder panel. CRS presets/validation come from the core
(``PRESETS_CRS``, ``resolve_crs`` / ``validate_crs``) and the exporters
(``write_navline_*``); reprojection runs in a :class:`CoreWorker`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..components import PlaceholderView
from ..i18n import language_manager
from ..state import AppState


class ReprojectorTab(QWidget):
    """Tab B — CRS management and navline export (structure only for now)."""

    def __init__(self, state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._placeholder = PlaceholderView()
        lay.addWidget(self._placeholder)

        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self._placeholder.set_text(self.tr("◈  Module under construction"))
