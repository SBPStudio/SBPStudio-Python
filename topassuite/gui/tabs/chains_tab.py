"""
chains_tab.py — Tab C · Chains.

Visualises a merged continuous profile (a :class:`ProfileChain`) with the same
controls and Profile seismic view as the Visualizer, plus chain-join boundary
markers. Sourced from the active chain in :class:`AppState`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ._base import PROFILE, SubTabbedTab
from ._render import compute_section
from ..components import SeismicView
from ..state import AppState


class ChainsTab(SubTabbedTab):
    """Tab C — visualises a single selected chain of merged profiles."""

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

    # ── State ────────────────────────────────────────────────────────────────

    def _on_active_chain_changed(self, chain: object) -> None:
        msg = (self.tr("Press Render to display the section.")
               if chain is not None else self.empty_message())
        for page in self.pages:
            page.show_placeholder(msg)

    # ── Render ───────────────────────────────────────────────────────────────

    def _on_render_requested(self) -> None:
        ch = self.state.active_chain
        if ch is None:
            return
        params = self.controls.params()
        boundaries = tuple(getattr(ch, "boundaries_km", ()) or ())
        title = getattr(ch, "label", getattr(ch, "name", ""))

        def job(progress, cancel) -> dict:
            from topassuite.core import process_chain_data
            progress(float("nan"), "")
            data = process_chain_data(ch, params)
            cancel.check()
            vp = compute_section(ch, data, params, boundaries=boundaries)
            vp["title"] = title
            return vp

        self.tasks.run_task(job, self._show_chain, self.tr("Rendering chain…"))

    def _show_chain(self, vp: dict) -> None:
        if self._seismic is None:
            self._seismic = SeismicView()
        self._seismic.show_image(
            vp["arr"], vp["dist0"], vp["dist1"], vp["t0"], vp["t1"],
            cmap_name=vp["cmap_name"], vmax=vp["vmax"], title=vp["title"],
            boundaries=vp["boundaries"], fixes=vp["fixes"],
            aspect=self.controls.aspect(),
            boundaries_visible=self.controls.boundaries_visible())
        self.pages[PROFILE].set_view(self._seismic)
