"""
visualizer_tab.py — Tab A · Visualizer.

Single-profile inspection. The Profile sub-tab renders a real PyQtGraph
seismic section: the DSP controls build a params dict, a CoreWorker runs
``core.process_profile_data`` + display-buffer computation off-thread, and
the result is shown in a :class:`SeismicView`. Map / Spectrum / Headers remain
placeholders.

Lazy-load states
----------------
* Profile stub added (data=None) → placeholder shows "Loading data…"
  (``_load_profile_traces`` in MainWindow runs a background load).
* Full profile loaded → placeholder shows "Press Render".
* After Render → section shown in the Profile sub-tab.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ._base import PROFILE, SubTabbedTab
from ._render import compute_section
from ..components import SeismicView
from ..state import AppState


class VisualizerTab(SubTabbedTab):
    """Tab A — visualises a single selected profile."""

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

    # ── State ────────────────────────────────────────────────────────────────

    def _on_active_profile_changed(self, profile: object) -> None:
        if profile is None:
            msg = self.empty_message()
        elif getattr(profile, "data", None) is None:
            # Header-only stub; MainWindow._load_profile_traces is loading it.
            msg = self.tr("Loading data…")
        else:
            msg = self.tr("Press Render to display the section.")
        for page in self.pages:
            page.show_placeholder(msg)

    # ── Render ───────────────────────────────────────────────────────────────

    def _on_render_requested(self) -> None:
        sd = self.state.active_profile
        if sd is None or getattr(sd, "error", None):
            return
        params = self.controls.params()
        title = f"{sd.name}  ·  {sd.n_traces} trazas  ·  {sd.dt_us} µs"

        def job(progress, cancel) -> dict:
            from topassuite.core import process_profile_data, load_profile
            # Safety load: cover the race where Render is clicked before the
            # background lazy-load finishes (unlikely but must be safe).
            _sd = sd
            if getattr(_sd, "data", None) is None:
                _sd = load_profile(_sd.path, load_traces=True)
                if getattr(_sd, "error", None):
                    raise RuntimeError(f"Failed to load {_sd.name}: {_sd.error}")
            progress(float("nan"), "")
            data = process_profile_data(_sd, params)
            cancel.check()
            vp = compute_section(_sd, data, params)
            vp["title"] = title
            return vp

        self.tasks.run_task(job, self._show_profile, self.tr("Rendering profile…"))

    def _show_profile(self, vp: dict) -> None:
        if self._seismic is None:
            self._seismic = SeismicView()
        self._seismic.show_image(
            vp["arr"], vp["dist0"], vp["dist1"], vp["t0"], vp["t1"],
            cmap_name=vp["cmap_name"], vmax=vp["vmax"], title=vp["title"],
            boundaries=vp["boundaries"], fixes=vp["fixes"],
            aspect=self.controls.aspect(),
            boundaries_visible=self.controls.boundaries_visible())
        self.pages[PROFILE].set_view(self._seismic)
