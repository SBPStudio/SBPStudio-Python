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

    # Whether ``load_full`` may be run on a BACKGROUND THREAD for the batch
    # prefetch-one pipeline (roadmap #4). True only when load_full returns a
    # FRESH, self-contained object without mutating any shared state — the
    # profile case. A chain loads its stitched matrix IN PLACE (mutating the
    # shared object the main thread also reads), so it must stay synchronous.
    prefetch_safe: bool = False

    # Which headless renderer this kind dispatches to — the plain, picklable
    # stand-in for the strategy object itself (see
    # gui.export_headless.render_export_figure's ``kind`` parameter).
    render_kind: str = "profile"

    # Whether batch items of this kind may be exported in WORKER PROCESSES
    # (gui.export_headless.render_batch_item): True only when one item is
    # fully reconstructable from a single file path — the profile case. A
    # chain assembles multi-file in-place state the worker can't rebuild
    # cheaply, so it stays on the in-process (prefetch/sequential) path.
    parallel_export_safe: bool = False


class ProfileHandler(SourceHandler):
    """Strategy for an individual :class:`SegyProfile`."""

    # load_full returns a fresh loaded profile (never mutates the stub) — safe
    # to run on the batch prefetch thread. See SourceHandler.prefetch_safe.
    prefetch_safe = True
    render_kind = "profile"
    # One profile == one file path → a worker process can rebuild the item
    # from scratch. See SourceHandler.parallel_export_safe.
    parallel_export_safe = True

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
        # export-only. _refresh_map_track resolves WGS84 via safe_map_coords
        # (reprojects known CRSs incl. projected/UTM; NEVER passes through
        # unresolved projected metres — the "UTM-as-degrees blows up the
        # basemap" crash fix) — output is ALWAYS valid WGS84-or-NaN. Set
        # BEFORE the preview fit (the fit's render feeds the distance axis →
        # visible_traces_changed → the map highlights it).
        tab._refresh_map_track()
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

    render_kind = "chain"

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
            tab.pages[MAP].set_view(tab._map)
            tab._refresh_map_track()
            tab.pages[HEADERS].set_view(tab._headers)
            tab._headers.set_source(chain)
            for i in (PROFILE, SPECTRUM):
                tab.pages[i].show_placeholder(tab.tr("Loading data…"))
            return
        tab.pages[PROFILE].set_view(tab._seismic)
        tab.pages[SPECTRUM].set_view(tab._spectrum)
        tab.pages[MAP].set_view(tab._map)
        tab.pages[HEADERS].set_view(tab._headers)
        # Cleaned display track → guaranteed-safe WGS84 via _refresh_map_track
        # (reprojects known CRSs incl. projected/UTM chains; never passes
        # through unresolved projected metres).
        tab._refresh_map_track()
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
