"""
visualizer_tab.py — Tab A · Visualizer.

Single-profile inspection. The Profile sub-tab hosts a live PyQtGraph seismic
section driven by the :class:`PreviewController`: the DSP node pipeline is
applied to the visible ViewBox window (column-decimated, rows full-res so AGC
stays exact) every time the user pans/zooms or edits a node. Map / Spectrum /
Headers remain placeholders.

Lazy-load states
----------------
* No profile selected            → placeholder "Select a profile…".
* Profile stub (data=None)        → placeholder "Loading data…".
* Profile loaded                  → live section; add nodes to process.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ...core import to_geographic
from ._base import HEADERS, MAP, PROFILE, SPECTRUM, SubTabbedTab
from ..state import AppState


class VisualizerTab(SubTabbedTab):
    """Tab A — visualises a single selected profile with a live DSP preview."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(state, tasks, parent=parent)
        self.state.active_profile_changed.connect(self._on_active_profile_changed)

    def empty_message(self) -> str:
        return self.tr("Select a profile in the list on the left")

    def _active_object(self):
        return self.state.active_profile

    def _export_basename(self) -> str:
        sd = self.state.active_profile
        return sd.stem if sd is not None else "perfil"

    # ── State → preview ──────────────────────────────────────────────────────

    def _on_active_profile_changed(self, profile: object) -> None:
        if profile is None:
            self.preview.set_source(None)
            self._map.clear()
            self._headers.clear()
            for page in self.pages:
                page.show_placeholder(self.empty_message())
            return
        if getattr(profile, "data", None) is None:
            # Header-only stub; MainWindow._load_profile_traces is loading it.
            # The HEADERS are already available, so populate the inspector now.
            self.preview.set_source(None)
            self._map.clear()
            self.pages[HEADERS].set_view(self._headers)
            self._headers.set_source(profile)
            for i in (PROFILE, SPECTRUM, MAP):
                self.pages[i].show_placeholder(self.tr("Loading data…"))
            return
        # Loaded → show the live section + spectrum + map + headers, kick off preview.
        self.pages[PROFILE].set_view(self._seismic)
        self.pages[SPECTRUM].set_view(self._spectrum)
        self.pages[MAP].set_view(self._map)
        self.pages[HEADERS].set_view(self._headers)
        self._headers.set_source(profile)
        # Cleaned display track (median-filtered in core); raw lons/lats stay
        # export-only. Reproject to WGS84 geographic via the core (passthrough if
        # already geographic) so projected/UTM files render on the lon/lat
        # basemap. Set BEFORE the preview fit (the fit's render feeds the
        # distance axis → visible_traces_changed → the map highlights it).
        mx, my = to_geographic(profile.track_lons, profile.track_lats,
                               getattr(profile, "detected_crs", None))
        self._map.set_track(mx, my)
        self.preview.set_source(profile)
        # Lock the view to the chosen aspect ON LOAD (the fit render above set the
        # full-section geometry). Without this the section loads free-fill and
        # stretches on zoom until the user touches the Scale spinner.
        self._seismic.set_aspect(self.controls.aspect(), fit=True)
