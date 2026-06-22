"""
_handlers.py — Source strategies for the unified Visualizer (profile vs chain).

Disciplined Strategy + Adapter
------------------------------
The unified :class:`VisualizerTab` renders EITHER an individual ``SegyProfile``
OR a merged ``ProfileChain`` in one UI space. The two kinds diverge in only a
handful of operations; everything else (map, spectrum, headers, DSP preview,
scale, DPI) is already type-agnostic and driven by duck-typed attributes.

Rather than scatter ``if is_chain`` boolean dispatch across the tab and the
shared :class:`SubTabbedTab` base, each kind's divergent behaviour is
encapsulated behind a single :class:`SourceHandler` strategy:

  * ``active_object()``       — ADAPTER: hands the views the current object
                                (``state.active_profile`` / ``state.active_chain``)
                                so the base never references concrete types.
  * ``on_selected(obj)``      — the type-specific state→preview UI transition.
  * ``load_full(obj, cancel)``— ensure the trace matrix is loaded; return the
                                object that carries ``.data`` (raises on error).
  * ``render_figure(...)``    — pick the core Matplotlib renderer.
  * ``basename(obj)``         — export filename stem.
  * ``source_path(obj)``      — disk path used for batch output naming.
  * ``release_after_batch``   — free a JIT-loaded matrix between batch items.

With these in place the base class becomes 100 % type-agnostic: it calls
``self._handler.<op>(...)`` and never knows whether it is driving a profile or a
chain. Adding a new source kind later = one new handler class + one selection
slot; no edits to the base dispatch or the existing handlers.

GUI-thread vs worker safety
---------------------------
``active_object`` / ``on_selected`` run on the GUI thread (they touch Qt
widgets). ``load_full`` / ``render_figure`` / ``release_after_batch`` /
``source_path`` / ``basename`` are pure logic over the core objects and are
called from ``CoreWorker`` threads — they never touch Qt, so they are safe to
invoke off-thread (the caller captures a stable handler reference before the
job starts, exactly as the old ``is_chain`` snapshot did).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ...core import crs_produces_geographic, to_geographic
from ._base import HEADERS, MAP, PROFILE, SPECTRUM


class SourceHandler(ABC):
    """Strategy + Adapter for one source kind, bound to its host VisualizerTab.

    The handler is a lightweight, reusable strategy (one instance per kind,
    created once in the tab). It holds a back-reference to the tab so the UI
    transition can drive the shared view widgets, and reads the active object
    from the tab's :class:`AppState`."""

    def __init__(self, tab) -> None:
        self._tab = tab
        self._state = tab.state

    # ── Adapter ───────────────────────────────────────────────────────────────

    @abstractmethod
    def active_object(self):
        """The core object this handler currently represents (or None)."""

    # ── GUI-thread: type-specific state→preview transition ────────────────────

    @abstractmethod
    def on_selected(self, obj) -> None:
        """Drive the tab's view widgets for a freshly selected, non-None object."""

    # ── Worker-thread: pure logic over the core object ────────────────────────

    @abstractmethod
    def load_full(self, obj, cancel):
        """Ensure ``obj``'s trace matrix is loaded; return the object carrying
        ``.data``. Idempotent (returns ``obj`` unchanged if already loaded).
        Raises ``RuntimeError`` if the load fails."""

    @abstractmethod
    def render_figure(self, obj, data, params, *, show_raster: bool = True,
                      style: str = "density", layout_mode: str = "aspect",
                      **kwargs):
        """Render ``obj`` + processed ``data`` to a Matplotlib Figure via the
        appropriate core renderer.

        Layered-rendering controls are explicit so the kind→renderer dispatch
        carries them uniformly: ``show_raster`` gates the density base layer,
        ``style`` ∈ {``"density"``, ``"wiggle"``} selects the section style, and
        ``layout_mode`` ∈ {``"aspect"``, ``"decoupled"``} echoes the scale layout
        (applied to ``figsize`` upstream). All other ``kwargs`` (figsize, dpi,
        ticks, theme, va_fill, wiggle_gain, …) are forwarded verbatim."""

    @abstractmethod
    def basename(self, obj) -> str:
        """Filename stem for single/FIX exports of ``obj``."""

    @abstractmethod
    def source_path(self, obj) -> str:
        """Disk path used to derive the per-item batch output location."""

    def release_after_batch(self, obj) -> None:
        """Free a JIT-loaded trace matrix after a batch item. Default: no-op
        (profiles render from a transient copy that is simply GC'd)."""
        return None


class ProfileHandler(SourceHandler):
    """Strategy for an individual :class:`SegyProfile`."""

    def active_object(self):
        return self._state.active_profile

    def on_selected(self, profile) -> None:
        tab = self._tab
        if getattr(profile, "data", None) is None:
            # Header-only stub; MainWindow._load_profile_traces is loading it.
            # The HEADERS are already available, so populate the inspector now.
            tab.preview.set_source(None)
            tab._map.clear()
            tab.pages[HEADERS].set_view(tab._headers)
            tab._headers.set_source(profile)
            for i in (PROFILE, SPECTRUM, MAP):
                tab.pages[i].show_placeholder(tab.tr("Loading data…"))
            return
        # Loaded → live section + spectrum + map + headers, kick off preview.
        tab.pages[PROFILE].set_view(tab._seismic)
        tab.pages[SPECTRUM].set_view(tab._spectrum)
        tab.pages[MAP].set_view(tab._map)
        tab.pages[HEADERS].set_view(tab._headers)
        tab._headers.set_source(profile)
        # Cleaned display track (median-filtered in core); raw lons/lats stay
        # export-only. Reproject to WGS84 geographic via the core (passthrough if
        # already geographic) so projected/UTM files render on the lon/lat
        # basemap. Set BEFORE the preview fit (the fit's render feeds the
        # distance axis → visible_traces_changed → the map highlights it).
        crs = getattr(profile, "detected_crs", None)
        mx, my = to_geographic(profile.track_lons, profile.track_lats, crs)
        # Authoritative coordinate-unit flag from the ACTUAL CRS (Bug #10): a
        # known CRS means the track is now WGS84 lon/lat; absent CRS → None →
        # the map uses its magnitude heuristic.
        tab._map.set_track(mx, my, is_geographic=crs_produces_geographic(crs))
        tab.preview.set_source(profile)
        # Engage the aspect lock immediately (same path as touching a scale
        # control) so the section is NEVER left in free/unlocked aspect mode —
        # free mode lets a side-panel resize stretch the image, since pyqtgraph
        # only preserves the data RANGE (not the screen aspect) across resizes
        # when unlocked.
        tab._on_scale_changed()

    def load_full(self, profile, cancel):
        if getattr(profile, "data", None) is not None:
            return profile
        from sbp_studio.core import load_profile
        loaded = load_profile(profile.path, load_traces=True)
        if getattr(loaded, "error", None):
            raise RuntimeError(f"Cannot load {loaded.name}: {loaded.error}")
        return loaded

    def render_figure(self, profile, data, params, *, show_raster: bool = True,
                      style: str = "density", layout_mode: str = "aspect",
                      **kwargs):
        from sbp_studio.viz.render import render_profile_figure
        return render_profile_figure(
            profile, data, params, show_raster=show_raster, style=style,
            layout_mode=layout_mode, **kwargs)

    def basename(self, profile) -> str:
        return profile.stem if profile is not None else "perfil"

    def source_path(self, profile) -> str:
        return profile.path


class ChainHandler(SourceHandler):
    """Strategy for a merged :class:`ProfileChain`."""

    def active_object(self):
        return self._state.active_chain

    def on_selected(self, chain) -> None:
        tab = self._tab
        if getattr(chain, "data", None) is None:
            # Lazy stub: the trace matrix is being assembled by
            # MainWindow._load_chain_traces. The track + inspector headers are
            # already available from the lightweight metadata, so show them now
            # and mark the section/spectrum as loading.
            tab.preview.set_source(None)
            crs = getattr(chain, "detected_crs", None)
            mx, my = to_geographic(chain.track_lons, chain.track_lats, crs)
            tab.pages[MAP].set_view(tab._map)
            tab._map.set_track(mx, my, is_geographic=crs_produces_geographic(crs))
            tab.pages[HEADERS].set_view(tab._headers)
            tab._headers.set_source(chain)
            for i in (PROFILE, SPECTRUM):
                tab.pages[i].show_placeholder(tab.tr("Loading data…"))
            return
        tab.pages[PROFILE].set_view(tab._seismic)
        tab.pages[SPECTRUM].set_view(tab._spectrum)
        tab.pages[MAP].set_view(tab._map)
        tab.pages[HEADERS].set_view(tab._headers)
        # Cleaned display track → WGS84 geographic via core (passthrough if
        # already geographic) so projected chains land on the lon/lat basemap.
        mx, my = to_geographic(chain.track_lons, chain.track_lats,
                               getattr(chain, "detected_crs", None))
        tab._map.set_track(mx, my)
        tab._headers.set_source(chain)
        tab.preview.set_source(chain)
        # Engage the aspect lock immediately — see ProfileHandler.on_selected.
        tab._on_scale_changed()

    def load_full(self, chain, cancel):
        # Chains assemble their stitched matrix in place and return self.
        if getattr(chain, "data", None) is None:
            chain.load_chain_traces(cancel=cancel)
        return chain

    def render_figure(self, chain, data, params, *, show_raster: bool = True,
                      style: str = "density", layout_mode: str = "aspect",
                      **kwargs):
        from sbp_studio.viz.render import render_chain_figure
        return render_chain_figure(
            chain, data, params, show_raster=show_raster, style=style,
            layout_mode=layout_mode, **kwargs)

    def basename(self, chain) -> str:
        profiles = getattr(chain, "profiles", None) if chain is not None else None
        return getattr(profiles[0], "stem", "cadena") if profiles else "cadena"

    def source_path(self, chain) -> str:
        return chain.profiles[0].path

    def release_after_batch(self, chain) -> None:
        # Chains mutate in place, so a JIT-loaded matrix must be reverted to a
        # stub to keep the batch RAM-flat (profiles render from a transient copy).
        chain.data = None
        chain.clip_p99 = None
