"""
chains_tab.py — Tab C · Chains.

Visualises a merged continuous profile (a :class:`ProfileChain`) with the same
live DSP-pipeline preview and seismic view as the Visualizer, plus chain-join
boundary markers. Sourced from the active chain in :class:`AppState`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ...core import to_geographic
from ._base import HEADERS, MAP, PROFILE, SPECTRUM, SubTabbedTab
from ..state import AppState


class ChainsTab(SubTabbedTab):
    """Tab C — visualises a selected chain of merged profiles with a live preview."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(state, tasks, parent=parent)
        self.state.active_chain_changed.connect(self._on_active_chain_changed)

    def empty_message(self) -> str:
        return self.tr("Detect and select a chain")

    def _active_object(self):
        return self.state.active_chain

    def _is_chain(self) -> bool:
        return True

    def _export_basename(self) -> str:
        ch = self.state.active_chain
        profiles = getattr(ch, "profiles", None) if ch is not None else None
        return getattr(profiles[0], "stem", "cadena") if profiles else "cadena"

    # ── State → preview ──────────────────────────────────────────────────────

    def _on_active_chain_changed(self, chain: object) -> None:
        if chain is None or getattr(chain, "data", None) is None:
            self.preview.set_source(None)
            self._map.clear()
            self._headers.clear()
            for page in self.pages:
                page.show_placeholder(self.empty_message())
            return
        self.pages[PROFILE].set_view(self._seismic)
        self.pages[SPECTRUM].set_view(self._spectrum)
        self.pages[MAP].set_view(self._map)
        self.pages[HEADERS].set_view(self._headers)
        # Cleaned display track → WGS84 geographic via core (passthrough if
        # already geographic) so projected chains land on the lon/lat basemap.
        mx, my = to_geographic(chain.track_lons, chain.track_lats,
                               getattr(chain, "detected_crs", None))
        self._map.set_track(mx, my)
        self._headers.set_source(chain)
        self.preview.set_source(chain)
