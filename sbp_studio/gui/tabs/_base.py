"""
_base.py — Shared structure for the sub-tabbed tool tabs.

Both the Visualizer and the Chains tab present the same four analysis views
(Profile / Map / Spectrum / Headers) beside the DSP controls column. Each
sub-tab is a :class:`_Page` that can switch between a placeholder and a real
view widget. :class:`SubTabbedTab` builds the skeleton, hosts the controls, and
forwards Render requests to an overridable hook; subclasses do the actual
per-profile / per-chain rendering.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDockWidget, QFileDialog, QFrame, QHBoxLayout,
    QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy, QStackedWidget,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..components import (
    CRSSelectorDialog, ExportDialog, HeaderView, MapView,
    PickingExportImportDialog, PipelinePanel,
    PlaceholderView, ProcessingControls, SeismicView, SpectrumView,
)
from ..components.processing_controls import DockDragMixin
from ..dsp import DSPContext, PreviewController
from ..i18n import language_manager
from ..state import AppState
from ._render import (
    dpi_for_budget, effective_aspect, effective_export_dpi, figsize_for_scale,
)

if TYPE_CHECKING:  # type-only; avoids a runtime import cycle (_handlers ← _base)
    from ._handlers import SourceHandler

# Sub-tab indices.
PROFILE, MAP, SPECTRUM, HEADERS = range(4)


class _Page(QStackedWidget):
    """A sub-tab page that toggles between a placeholder and a real view."""

    def __init__(self) -> None:
        super().__init__()
        self.placeholder = PlaceholderView()
        self.addWidget(self.placeholder)
        self._view: Optional[QWidget] = None

    def set_view(self, view: QWidget) -> None:
        if self._view is not None and self._view is not view:
            self.removeWidget(self._view)
            self._view.deleteLater()
        self._view = view
        if self.indexOf(view) == -1:
            self.addWidget(view)
        self.setCurrentWidget(view)

    def show_placeholder(self, text: str) -> None:
        self.placeholder.set_text(text)
        self.setCurrentWidget(self.placeholder)


class DockTitleBar(QWidget, DockDragMixin):
    """Custom title-bar widget for the Controls dock (see
    _build_controls_dock): hosts the "Controles y Procesado" / "Marcas y
    Exportación" tab strip directly, via ``dock.setTitleBarWidget(this)`` —
    promoting the tabs to the dock's absolute top and removing the native
    title bar's now-redundant "Controles" label entirely.

    DockDragMixin handles dragging the dock by this bar (including the
    EMPTY space beside the tabs — dragging an actual tab is handled by the
    tab bar itself, see _DraggableTabBar in processing_controls.py): a
    custom title bar widget does not get Qt's built-in drag-to-float
    handling for free, since installing one replaces that entirely."""

    def __init__(self, dock, tab_bar, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._dock = dock
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(0)
        # Bigger + bold so the promoted tabs read with the same visual
        # weight the native dock title text used to have.
        font = tab_bar.font()
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        tab_bar.setFont(font)
        lay.addWidget(tab_bar)
        lay.addStretch(1)

    def mousePressEvent(self, event) -> None:
        self._dock_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._dock_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dock_release(event)
        super().mouseReleaseEvent(event)


def _batch_output_path(src_path: str, fmt: str, used: set,
                       out_dir: "Optional[str]" = None) -> "Path":
    """Folder-named export path for a source SEG-Y file: ``<dir>/<dir>.<fmt>``.

    e.g. ``Z:/data/Line_01/1.sgy`` → ``Z:/data/Line_01/Line_01.pdf``.

    ``out_dir`` — optional single target folder for the whole batch. When given
    (non-empty), the file is written THERE instead of the source folder, but the
    NAME is unchanged: still the name of the folder that contains the source SEG-Y
    (``<out_dir>/Line_01.pdf``) — the naming convention the user relies on. When
    ``None``/empty, the historical per-source-folder behaviour is used.

    If the chosen path was already produced in this batch (two lines whose source
    folders share a name — far more likely when everything lands in one custom
    folder) it is de-duped with the source file stem so nothing is silently
    overwritten. Returns a ``Path``; the caller adds ``str(path)`` to ``used``.
    """
    from pathlib import Path
    src_dir = Path(src_path).parent
    dest_dir = Path(out_dir) if out_dir else src_dir
    out = dest_dir / f"{src_dir.name}.{fmt}"
    if str(out) in used:
        out = dest_dir / f"{src_dir.name}_{Path(src_path).stem}.{fmt}"
    return out


# Per-item batch-export RAM estimate multiplier over the raw trace matrix: the
# float32 input + the DSP-processed copy + transient node working buffers. 4× is
# a conservative peak used only to decide whether TWO items (current render +
# one prefetched) fit — the prefetch is skipped otherwise (see export_batch).
_BATCH_ITEM_BYTES_MULT = 4.0


def _estimate_batch_item_bytes(items: list) -> float:
    """Peak per-item resident footprint across ``items``, from the DEEPEST ×
    WIDEST header geometry (any item could be the one resident). Header-only —
    reads ns/n_traces already on the stubs, no trace load."""
    peak = 0.0
    for o in items:
        ns = int(getattr(o, "ns", 0) or 0)
        nt = int(getattr(o, "n_traces", 0) or 0)
        peak = max(peak, ns * nt * 4.0)
    return max(1.0, peak) * _BATCH_ITEM_BYTES_MULT


def _run_batch_export_loop(items, handler, *, make_out, render_save,
                           progress, cancel):
    """GUI-free batch export engine with the prefetch-one pipeline (roadmap #4).

    Drives every item through: resolve output path (``make_out(obj)``) → ensure
    its trace matrix is loaded → ``render_save(render_obj, out)`` → release. One
    bad item is recorded and skipped; a ``Cancelled`` aborts the whole batch.
    Returns ``(saved: list[str], failed: list[(name, error)])``.

    Prefetch-one: while item i renders (GIL-bound), a single background thread
    pre-loads item i+1 (segyio releases the GIL during I/O, so the load truly
    overlaps). Enabled ONLY when ``handler.prefetch_safe`` (profiles — load_full
    returns a fresh object; chains mutate in place and stay synchronous) AND the
    RAM budget fits TWO items at once (``plan_workers(per_item) >= 2``). On a
    low-RAM box it degrades to exactly the synchronous path — at most one extra
    item is ever resident (single loader thread).

    Extracted from the Qt method so this orchestration (gating, ordering,
    failure isolation, release, RAM-flatness) is unit-testable with fakes.
    ``make_out`` / ``render_save`` are injected callables; the Qt method binds
    them to the real ``_batch_output_path`` + ``_render_export_figure`` path.
    """
    from concurrent.futures import ThreadPoolExecutor
    from sbp_studio.core.tasks import Cancelled
    from sbp_studio.core._backends import plan_workers

    n = len(items)
    saved: list = []
    failed: list = []

    per_item = _estimate_batch_item_bytes(items)
    do_prefetch = (getattr(handler, "prefetch_safe", False)
                  and plan_workers(per_item) >= 2)
    pool = ThreadPoolExecutor(max_workers=1) if do_prefetch else None
    pending: dict = {}      # index → Future(loaded_obj)

    def _submit(idx: int) -> None:
        if pool is not None and 0 <= idx < n:
            pending[idx] = pool.submit(handler.load_full, items[idx], cancel)

    try:
        _submit(0)
        for i, obj in enumerate(items):
            cancel.check()
            name = getattr(obj, "label", None) or getattr(obj, "name", "item")
            progress(i / n, name)
            out = make_out(obj)
            was_loaded = getattr(obj, "data", None) is not None
            # Kick the NEXT load off BEFORE this item's long render so disk I/O
            # overlaps the render.
            _submit(i + 1)
            try:
                fut = pending.pop(i, None)
                render_obj = (fut.result() if fut is not None
                              else handler.load_full(obj, cancel))
                render_save(render_obj, out)
                saved.append(str(out))
            except Cancelled:
                raise                                    # abort the whole batch
            except Exception as exc:                     # one bad item ≠ kill batch
                failed.append((name, str(exc)))
            finally:
                # RAM-flat: release the JIT-loaded matrix (chains revert in
                # place; profiles GC the transient copy) only when WE loaded it.
                if not was_loaded:
                    handler.release_after_batch(obj)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    progress(1.0, "")
    return saved, failed


def _process_full_array(obj, params, node_cfg, align_enabled, cancel):
    """Delegates to the extracted Qt-free engine — see
    ``gui.export_headless.process_full_array`` (moved verbatim so the single
    export, the sequential batch and the process-pool batch all run literally
    the same code)."""
    from sbp_studio.gui.export_headless import process_full_array
    return process_full_array(obj, params, node_cfg, align_enabled, cancel)


def _crop_for_viewport(obj, proc, t0_full, x_range, y_range):
    """Delegates to the Qt-free engine (moved to ``gui.export_headless``) so the
    HQ viewport overlay and the WYSIWYG export crop share one implementation."""
    from sbp_studio.gui.export_headless import _crop_for_viewport as _impl
    return _impl(obj, proc, t0_full, x_range, y_range)


def _render_export_figure(obj, cfg, params, node_cfg, scale_cfg, align_enabled,
                          handler, cancel, picks=None, view_range=None,
                          view_figsize=None):
    """Delegates to the extracted Qt-free engine — see
    ``gui.export_headless.render_export_figure`` (the former body of this
    function, moved VERBATIM: same DSP pass, same RAM-capped DPI, same WYSIWYG
    aspect-fit loop). The live SourceHandler maps to a plain ``kind`` string
    ("profile" | "chain" — see SourceHandler.render_kind); everything else is
    passed through unchanged, so single/batch/pool exports cannot drift.

    ``view_range`` / ``view_figsize`` — the live ViewBox window + its physical
    on-screen size, forwarded so the SINGLE export can crop to the on-screen
    zoom AND size its page dynamically from the viewport ('Auto' paper).
    Batch/pool pass ``None`` for both and stay full-line/formula-sized."""
    from sbp_studio.gui.export_headless import render_export_figure
    kind = getattr(handler, "render_kind", "profile")
    return render_export_figure(obj, cfg, params, node_cfg, scale_cfg,
                                align_enabled, kind, cancel, picks=picks,
                                view_range=view_range,
                                view_figsize=view_figsize)


class SubTabbedTab(QWidget):
    """Controls column + four-view sub-notebook with real Render dispatch."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.tasks = tasks  # task service (MainWindow.run_task / notify / show_error)
        # The active source strategy (ProfileHandler / ChainHandler), assigned by
        # the subclass on selection. Declared here so the base — which calls
        # _update_dpi_estimate() → _active_object() during construction — and the
        # export/HQ/batch paths can drive everything through the handler interface
        # without ever knowing the concrete source type.
        self._handler: "Optional[SourceHandler]" = None

        # Session memory: the folder of the last INDIVIDUAL image export, so the
        # next Save dialog reopens there instead of resetting to the CWD each
        # time. Session-scoped only (not persisted to disk); empty until the
        # first successful export.
        self._last_export_dir: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Dockable workspace (multi-monitor support): an inner QMainWindow
        # embedded as a plain child widget (a fully supported Qt pattern —
        # the same trick Qt Designer itself uses) gives every panel a real,
        # native QDockWidget that can be dragged out to a floating window on
        # a second monitor, independent of the outer MainWindow. No central
        # widget is set, so the Controls dock and the tabified Profile/Map/
        # Spectrum/Headers docks fill the whole area themselves.
        self._dock_host = QMainWindow()
        self._dock_host.setWindowFlags(Qt.WindowType.Widget)
        self._dock_host.setDockNestingEnabled(True)
        # GroupedDragging lets the whole tabified group move as a unit by
        # its tab strip; AllowTabbedDocks is what makes tabifyDockWidget's
        # browser-tab-style grouping (below) possible in the first place.
        self._dock_host.setDockOptions(
            self._dock_host.dockOptions()
            | QMainWindow.DockOption.GroupedDragging
            | QMainWindow.DockOption.AllowTabbedDocks)
        self._dock_host.setTabPosition(
            Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        self._dock_host.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self._build_controls_dock())
        self._build_view_docks()
        root.addWidget(self._dock_host)

        # Eager SeismicView on the Profile page, driven by the live preview
        # controller (pan/zoom + node edits → ViewBox-limited pipeline preview).
        self._seismic = SeismicView()
        self._seismic.enable_preview(True)
        self.pages[PROFILE].set_view(self._seismic)
        # Eager advanced Spectrum QC panel on the Spectrum page — run ON DEMAND
        # (its Generate button), decoupled from the live 300 ms preview loop.
        self._spectrum = SpectrumView()
        self.pages[SPECTRUM].set_view(self._spectrum)
        # Eager navigation Map on the Map page — bi-directional sync with the
        # seismic ViewBox (segment follows the profile; click jumps the profile).
        self._map = MapView()
        self.pages[MAP].set_view(self._map)
        # Eager Header Inspector on the Headers page — textual header + NumPy-
        # backed trace-header table (handles 50k+ traces without a QTableWidget).
        self._headers = HeaderView()
        self.pages[HEADERS].set_view(self._headers)
        self.preview = PreviewController(
            self._seismic, self.pipeline_panel,
            get_source=self._preview_source,
            get_display=self.controls.display_params,
            parent=self)
        self._spectrum.generate_requested.connect(self._on_generate_spectrum)
        # Profile → Map: bright segment tracks the visible window, addressed by
        # absolute trace index (lock-step with the image array; GPS-plateau-proof).
        self._seismic.visible_traces_changed.connect(self._map.set_visible_range)
        # Map → Profile: click-to-jump scrolls the seismic view to that trace.
        self._map.trace_clicked.connect(self._on_map_trace_clicked)
        # Add Layer (GIS overlay): read the file in the CORE off-thread, then hand
        # the parsed WGS84 layer back to the map for rendering.
        self._map.layer_file_requested.connect(self._on_map_layer_requested)
        # Trace selection sync → Header Inspector. A click on the Map or the
        # Seismic section scrolls+highlights that trace's header row; clicking a
        # header row recentres the profile (which drags the map segment along).
        self._map.trace_clicked.connect(self._headers.select_trace)
        self._seismic.trace_clicked.connect(self._headers.select_trace)
        self._headers.trace_selected.connect(self._on_header_trace_selected)

        # Link Views (cross-module sync, OFF by default — see _build_subtabs):
        # live navigation cursor on hover + "Add Anomaly to Map" POI from the
        # seismic view's right-click menu. Both handlers no-op while the
        # toggle is unchecked; set_link_views_enabled also greys out the
        # menu action itself (see SeismicView._update_ruler_menu_state).
        self._seismic.cursor_trace_changed.connect(self._on_seismic_cursor_moved)
        self._seismic.add_anomaly_requested.connect(self._on_seismic_anomaly_requested)

        # Render Full → re-fit the whole section (the live loop is automatic).
        self.controls.render_requested.connect(self.preview.fit)
        # Render Viewport HQ → HQ Matplotlib-quality render of the visible crop,
        # shown as an ephemeral overlay INSIDE the live viewer (no file export).
        self.controls.render_viewport_requested.connect(
            self._on_hq_preview_requested)
        self.controls.export_image_requested.connect(self._on_export_image_requested)
        self.controls.export_fix_requested.connect(self._on_export_fix_requested)
        # Interpretation & Picking (Phase 3): the toggle button drives the
        # view's double-click behaviour directly; the export/import button
        # opens the chooser dialog (mirrors the FIX-marks export flow).
        self.controls.picking_toggled.connect(self._seismic.set_pick_mode)
        self.controls.export_import_picking_requested.connect(
            self._on_picking_export_import)
        self.controls.scale_changed.connect(self._on_scale_changed)
        self.controls.boundaries_toggled.connect(self._on_boundaries_toggled)
        # Initial sync (same idiom as set_image_interpolation below): a
        # checkbox that starts unchecked never fires toggled() on
        # construction (no real transition happens), so without this the
        # view's own _boundaries_visible default could silently disagree
        # with the checkbox's displayed (unchecked) state.
        self._on_boundaries_toggled(self.controls.boundaries_visible())
        self.controls.align_toggled.connect(lambda *_: self.preview.alignment_changed())
        self.controls.display_changed.connect(self.preview.display_changed)
        # Clip slider: deliberately NOT display_changed — instant
        # [vmin, vmax] update from cached samples, no PipelineWorker
        # round-trip (see PreviewController.clip_changed).
        self.controls.clip_changed.connect(self.preview.clip_changed)
        # Live raster pixel-scaling (nearest/bilinear) — a paint-time hint on the
        # view, not a data change, so it's applied directly (no DSP refresh).
        self.controls.interp_changed.connect(self._seismic.set_image_interpolation)
        self._seismic.set_image_interpolation(self.controls.interp_mode())
        # Live export-DPI readout (Part 2): refresh when the scale settings OR the
        # active source change, so the user always sees the resolution that will be
        # generated for the current configuration.
        self.state.active_profile_changed.connect(lambda *_: self._update_dpi_estimate())
        self.state.active_chain_changed.connect(lambda *_: self._update_dpi_estimate())
        # Picking needs the active object to resolve x_coord/y_coord for any
        # NEW marker (core.resolve_pick_coords reads obj.lons/obj.lats) — see
        # SeismicView.set_picking_source's docstring for why switching to a
        # genuinely different profile/chain clears the on-screen markers
        # while a same-profile DSP/Render Full refresh never does.
        self.state.active_profile_changed.connect(
            lambda *_: self._seismic.set_picking_source(self._active_object()))
        self.state.active_chain_changed.connect(
            lambda *_: self._seismic.set_picking_source(self._active_object()))
        # Map-redraw mechanism (Task 1): ANY code path that resolves a CRS in
        # place (the Map-tab just-in-time prompt below, or the sidebar's
        # Metadata Inspector "Edit CRS…") calls state.notify_crs_updated();
        # every tab showing that object reacts here. One-way notification —
        # the refresh itself never changes the CRS — so this cannot cycle.
        self.state.crs_updated.connect(self._on_crs_updated)
        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()
        self._update_dpi_estimate()

    # ── Public surface (used by the sidebar 'Add to map' batch loader) ───────

    @property
    def map_view(self) -> "MapView":
        """The navigation MapView hosted on this tab's Map sub-tab."""
        return self._map

    def reveal_map(self) -> None:
        """Make the Map sub-tab show the live map (not a placeholder) and bring
        it to front. Used when batch-adding tracks so the result is visible even
        when no profile is active. Does NOT change the active profile or section."""
        self.pages[MAP].set_view(self._map)
        self._docks[MAP].show()
        self._docks[MAP].raise_()

    # ── Map redraw (Task 1) + just-in-time CRS prompt (Task 2) ──────────────

    def _refresh_map_track(self) -> None:
        """(Re)compute the active object's map track via
        ``core.safe_map_coords`` and push it to the MapView — the single
        shared code path for 'put the active object's track on the map',
        used by _handlers.py's on_selected AND by every CRS-change reaction
        below. Always safe to call with nothing active (no-op)."""
        obj = self._active_object()
        if obj is None:
            return
        lons = getattr(obj, "track_lons", None)
        lats = getattr(obj, "track_lats", None)
        if lons is None or lats is None or len(lons) == 0:
            return
        from sbp_studio.core import safe_map_coords
        mx, my = safe_map_coords(lons, lats, getattr(obj, "coord_unit", 0),
                                 getattr(obj, "detected_crs", None))
        self._map.set_track(mx, my, is_geographic=True,
                            crs_is_unknown=getattr(obj, "crs_is_unknown", False))

    def _on_crs_updated(self, obj) -> None:
        """state.crs_updated reaction: only redraw if the object whose CRS
        just changed is the one THIS tab is currently showing — a CRS edit
        on a profile that isn't on screen here must not touch this map."""
        if obj is self._active_object():
            self._refresh_map_track()

    def _on_map_dock_visible(self, visible: bool) -> None:
        if visible:
            self._prompt_crs_if_needed()

    def _prompt_crs_if_needed(self) -> None:
        """GIS-style CRS resolution (QGIS/Petrel workflow), triggered ONLY by
        actually landing on the Map sub-tab — never at file-load time, so
        pure signal-processing work is never interrupted. Standard SEG-Y has
        no field for the projection/zone (see io_segy._detect_crs), so a
        PROJECTED (CoordinateUnits=1) file can't be auto-resolved; prompt
        once per visit while it stays unresolved. Cancelling just leaves the
        track empty (safe_map_coords already refuses to plot raw projected
        metres as if they were WGS84 degrees) rather than forcing a choice."""
        obj = self._active_object()
        if obj is None or getattr(obj, "error", None):
            return
        if getattr(obj, "coord_unit", 0) != 1 or getattr(obj, "detected_crs", None) is not None:
            return
        from sbp_studio.core import set_crs_override
        name = getattr(obj, "name", None) or getattr(obj, "label", "") or ""
        dlg = CRSSelectorDialog(self, file_label=name)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_crs():
            set_crs_override(obj, dlg.selected_crs())
            self.state.notify_crs_updated(obj)

    # ── Overridable hooks ───────────────────────────────────────────────────

    def empty_message(self) -> str:
        """Translated message shown when nothing is selected."""
        return ""

    def _active_object(self):
        """The SegyProfile / ProfileChain currently driving this tab (or None)."""
        return None

    def _preview_source(self):
        """Source for the live preview: the active object only if traces loaded."""
        obj = self._active_object()
        if obj is None or getattr(obj, "error", None) or getattr(obj, "data", None) is None:
            return None
        return obj

    def _on_scale_changed(self) -> None:
        """Apply the selected scale mode's effective aspect to the on-screen
        section live, so the preview proportions match what the export will
        produce. For the VE / hybrid modes the aspect is derived from THIS line's
        length & TWT (so the preview re-shapes per line, exactly like the PDF)."""
        if self._seismic is not None and self._seismic.has_image():
            asp = effective_aspect(self._active_object(),
                                   self.controls.scale_config())
            self._seismic.set_aspect(asp)
        self._update_dpi_estimate()

    # Assumed export settings for the live DPI readout (the export dialog's
    # defaults): the matplotlib export starts at this DPI and budget unless the
    # user changes them in the dialog.
    _DPI_ESTIMATE_BASE = 1200
    _DPI_ESTIMATE_BUDGET_GB = 6.0

    def _update_dpi_estimate(self) -> None:
        """Compute the resolution the export will actually generate for the current
        scale configuration and show it in the controls panel (Part 2). Uses only
        header dimensions (works on lazy stubs, before any trace data is loaded)."""
        obj = self._active_object()
        if obj is None or not int(getattr(obj, "n_traces", 0) or 0) \
                or not int(getattr(obj, "ns", 0) or 0):
            self.controls.set_dpi_estimate(
                QCoreApplication.translate("SubTabbedTab", "Export DPI: — (no line)"))
            return
        try:
            scfg = self.controls.scale_config()
            figsize = figsize_for_scale(obj, scfg, self._DPI_ESTIMATE_BASE, 1500.0)
            eff = effective_export_dpi(figsize, (int(obj.ns), int(obj.n_traces)),
                                       self._DPI_ESTIMATE_BASE)
            cap = dpi_for_budget(figsize, self._DPI_ESTIMATE_BUDGET_GB)
            limited = cap is not None and eff > cap
            if limited:
                eff = max(50, cap)
            mpx = (figsize[0] * eff) * (figsize[1] * eff) / 1e6
            tail = QCoreApplication.translate("SubTabbedTab", " (RAM-limited)") if limited else ""
            self.controls.set_dpi_estimate(QCoreApplication.translate(
                "SubTabbedTab", "Export ≈ {0} DPI · {1:.0f} Mpx{2}").format(
                    int(eff), mpx, tail))
        except Exception:
            self.controls.set_dpi_estimate("")

    def _on_map_trace_clicked(self, idx: int) -> None:
        """Map → Profile: scroll the seismic view to the clicked trace."""
        self._center_on_trace(idx)

    def _on_header_trace_selected(self, idx: int) -> None:
        """Header row clicked → recentre the profile on that trace (the map
        segment follows via the seismic range change)."""
        self._center_on_trace(idx)

    def _center_on_trace(self, idx: int) -> None:
        obj = self._active_object()
        dist = getattr(obj, "dist_km", None) if obj is not None else None
        if dist is not None and 0 <= idx < len(dist):
            self._seismic.center_on_distance(float(dist[idx]))

    def _on_link_views_toggled(self, checked: bool) -> None:
        """Link Views turned off → drop the live navigation cursor immediately
        (it would otherwise sit stale at its last hover position), and grey
        out the seismic view's "Add Anomaly to Map" menu action."""
        self._seismic.set_link_views_enabled(checked)
        if not checked:
            self._map.hide_navigation_marker()

    def _on_seismic_cursor_moved(self, idx: int) -> None:
        """Live Navigation Cursor (Part 1): only active while Link Views is
        checked. The seismic view emits a bare trace index; the map already
        holds the matching lon/lat track arrays (set_track), so no coordinate
        crosses the signal boundary — same split as trace_clicked."""
        if self.chk_link_views.isChecked():
            self._map.show_navigation_marker(idx)

    def _on_seismic_anomaly_requested(self, idx: int) -> None:
        """Anomaly Waypoint (Part 1): "Add Anomaly to Map" on the seismic
        view's right-click menu drops a permanent POI marker on the map.
        The menu action is itself greyed out while Link Views is off (see
        set_link_views_enabled), so this check is a defensive no-op only."""
        if self.chk_link_views.isChecked():
            self._map.add_poi_marker(idx)

    def _on_map_layer_requested(self, path: str) -> None:
        """Read a GIS overlay file in the core (off the GUI thread) and add the
        parsed WGS84 layer to the map."""
        def job(progress, cancel):
            from sbp_studio.core import read_gis_layer
            progress(float("nan"), "")
            return read_gis_layer(path)

        self.tasks.run_task(
            job, self._map.add_layer,
            QCoreApplication.translate("SubTabbedTab", "Loading map layer…"))

    def _on_boundaries_toggled(self, visible: bool) -> None:
        """Show/hide the red file-seam lines in the live view (interactive only).

        This is INDEPENDENT of the export dialog's boundary checkbox — toggling
        here never affects what gets exported.
        """
        if self._seismic is not None:
            self._seismic.set_boundaries_visible(visible)

    def _export_basename(self) -> str:
        """Filename stem for exports."""
        return "export"

    # ── Advanced spectrum analysis (on demand, worker-backed) ────────────────

    def _on_generate_spectrum(self, scope: str) -> None:
        """Run the heavy Welch analysis for the chosen scope in a worker thread,
        then render it into the advanced Spectrum panel. Never runs on the live
        preview loop."""
        import numpy as np
        inp = self.preview.analysis_inputs(scope)
        if inp is None:
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "Select an item first."))
            return
        arr0 = inp["arr"]
        node_cfgs = inp["node_cfgs"]
        dt_us = inp["dt_us"]
        dist_km = inp["dist_km"]
        boundaries = inp["boundaries"]

        def job(progress, cancel):
            from sbp_studio.core import compute_spectrum
            from sbp_studio.gui.dsp import DSPContext, make_node
            progress(float("nan"), "")
            # Apply the dynamic per-window nodes at FULL resolution (the prepared
            # base already has static alignment + pre-crop mute baked in).
            arr = arr0
            ctx = DSPContext(dt_us=dt_us, ns=arr.shape[0], n_traces=arr.shape[1])
            for key, npar in node_cfgs:
                cancel.check()
                arr = make_node(key, npar).apply(arr, ctx)
            cancel.check()
            sp = compute_spectrum(np.nan_to_num(arr, nan=0.0), 1e6 / dt_us)
            return sp, 1e6 / dt_us, dist_km, boundaries

        self.tasks.run_task(
            job,
            lambda r: self._spectrum.show_result(r[0], r[1], r[2], r[3]),
            QCoreApplication.translate("SubTabbedTab", "Computing spectrum…"))

    # ── Export (shared by Visualizer and Chains) ─────────────────────────────
    # NOTE: these literals live in SubTabbedTab but run on subclass instances, so
    # self.tr() would resolve the wrong context. QCoreApplication.translate with
    # the literal context "SubTabbedTab" is used inline so it both resolves
    # correctly at runtime AND is extractable by pylupdate6.
    #
    # Image/PDF export uses the CORE headless matplotlib renderer
    # (sbp_studio.viz.render) on a worker thread — the same high-quality,
    # PDF-capable path as the CLI `export-image` — NOT a PyQtGraph screenshot.

    def _confirm_memory_budget(self, mem_budget_gb: float) -> bool:
        """Validate the export memory budget against the machine's free RAM.

        Returns True to proceed. If the budget exceeds free RAM, a QMessageBox
        warns the user (with the measured numbers) and lets them continue at their
        own risk or cancel — preventing a silent out-of-memory crash. If free RAM
        cannot be measured (no psutil), we proceed without blocking."""
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 1024 ** 3
        except Exception:
            return True
        if mem_budget_gb <= free_gb:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(QCoreApplication.translate(
            "SubTabbedTab", "Memory budget exceeds free RAM"))
        box.setText(QCoreApplication.translate(
            "SubTabbedTab",
            "The export memory budget ({0:.1f} GB) is larger than the free RAM "
            "on this machine ({1:.1f} GB).").format(mem_budget_gb, free_gb))
        box.setInformativeText(QCoreApplication.translate(
            "SubTabbedTab",
            "Continuing may run out of memory and crash the application. Lower the "
            "Memory budget (GB) in the export dialog, or close other programs, to "
            "be safe.\n\nContinue anyway?"))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _on_export_image_requested(self) -> None:
        obj = self._active_object()
        if obj is None or getattr(obj, "error", None):
            self.tasks.notify(QCoreApplication.translate("SubTabbedTab", "Select an item first."))
            return
        dlg = ExportDialog(self, source=obj, scale_cfg=self.controls.scale_config(),
                           velocity=1500.0,
                           view_figsize=(self._seismic.viewport_inches()
                                         if self._seismic is not None and
                                         self._seismic.has_image() else None))
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            cfg = dlg.config()
        finally:
            dlg.deleteLater()
        # RAM validation: if the requested memory budget exceeds the machine's
        # actual free RAM, alert before rendering so the user can cancel and lower
        # it rather than risk an out-of-memory crash.
        if not self._confirm_memory_budget(cfg.get("mem_budget_gb", 6.0)):
            return
        fmt = cfg["format"]
        # Seed the Save dialog with the last-used export folder (session memory,
        # Feature 3) so it doesn't reset to the CWD each time. The default file
        # NAME is still the current line's basename; only the directory is
        # remembered. Empty on the first export → Qt's own default location.
        from pathlib import Path as _Path
        default_name = f"{self._export_basename()}.{fmt}"
        start = (str(_Path(self._last_export_dir) / default_name)
                 if self._last_export_dir else default_name)
        out, _ = QFileDialog.getSaveFileName(
            self, QCoreApplication.translate("SubTabbedTab", "Export image"),
            start, f"{fmt.upper()} (*.{fmt})")
        if not out:
            return
        # Remember the directory the user actually chose for next time.
        self._last_export_dir = str(_Path(out).parent)

        handler = self._handler
        scale_cfg = self.controls.scale_config()  # aspect / VE / hybrid mode + values
        params = dict(self.controls.display_params())  # presentation only (cmap/clip/fix)
        params["clip_lo"] = 0.0
        params["fill_value"] = 0.0 if cfg["fill_zero"] else float("nan")
        # File-boundary lines in the export come from the EXPORT DIALOG's own
        # checkbox — independent of the interactive viewer's toggle. The core
        # renderer reads params["draw_file_boundaries"] to gate ax.axvline.
        params["draw_file_boundaries"] = bool(cfg["draw_file_boundaries"])

        # Snapshot the DSP node configuration (key + params copy) so the worker
        # rebuilds the pipeline thread-safely. EXPORT uses the SAME node config
        # as the preview, but runs it over the 100 % FULL array (no ViewBox /
        # decimation) — completely decoupled from the on-screen preview.
        # active_nodes() = enabled-only, so a MUTED node is bypassed on export
        # exactly as it is in the preview.
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.active_nodes()]
        # Static delay-alignment (geometry) is applied to the base BEFORE the
        # dynamic nodes — exactly as the preview does — read from the static
        # checkbox. ``params["align"]`` drives the renderer's time axis
        # (t0 = min_delay when aligned).
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled

        # WYSIWYG amplitude: inject the live viewport's locked vmin/vmax so the
        # PDF matches what's on screen exactly — the core renderer otherwise
        # recomputes its own percentile from the full array, which differs from
        # the preview's 5-block estimate and causes the "washed out PDF" effect.
        if self._seismic is not None and self._seismic.has_image():
            _live_vmin, _live_vmax = self._seismic.current_levels()
            if _live_vmax is not None and _live_vmax > 0:
                params["vmax_override"] = _live_vmax
            # WYSIWYG aspect (free mode): use the live ViewBox's pixel proportions
            # as the figure's W/H ratio so the export page has the same
            # landscape/portrait feel as the screen — without this, free mode
            # always falls back to the VE formula, which ignores the user's
            # manual zoom.
            if scale_cfg.get("mode") == "free":
                _vb = self._seismic.plot.getViewBox()
                _vb_w = max(1.0, _vb.width())
                _vb_h = max(1.0, _vb.height())
                scale_cfg = dict(scale_cfg, pixel_aspect=_vb_w / _vb_h)

        # Picks must be read from the live SeismicView HERE, on the GUI thread
        # (Qt widgets aren't thread-safe) — get_picks() already returns a
        # shallow copy of plain PickPoint dataclasses, safe to hand to the
        # background worker's closure below. Empty when the checkbox is off,
        # so _render_export_figure's picks param is a no-op (see _draw_picks).
        picks = self._seismic.get_picks() if cfg.get("overlay_picks") else []

        # WYSIWYG snapshot, taken on the GUI thread (Qt objects aren't
        # thread-safe) as ONE unit so the two halves can never diverge:
        #   view_range   — the visible ViewBox window; a zoomed sub-window crops
        #                  the export to it (full view → no-op, full line).
        #   view_figsize — the viewport's physical on-screen inches; with the
        #                  'Auto' paper size (the default) the export page takes
        #                  exactly these dimensions/proportions instead of being
        #                  squeezed onto a fixed sheet (the 'A4 trap' fix).
        _has_view = self._seismic is not None and self._seismic.has_image()
        view_range = self._seismic.current_view_range() if _has_view else None
        view_figsize = self._seismic.viewport_inches() if _has_view else None

        def job(progress, cancel) -> str:
            from sbp_studio.viz.render import save_figure
            # Lazy-load safety: profiles are header-only stubs and CHAINS assemble
            # their matrix lazily. The handler loads/assembles JIT on the worker
            # thread so the export never fails (race, or chain never viewed).
            if getattr(obj, "data", None) is None:
                progress(float("nan"), "Loading traces for export…")
            _obj = handler.load_full(obj, cancel)
            progress(float("nan"), "")
            # Shared render pipeline (identical to every batch item): full-res DSP
            # + decimation-free DPI floor + WYSIWYG aspect fit.
            fig, render_dpi = _render_export_figure(
                _obj, cfg, params, node_cfg, scale_cfg, align_enabled, handler, cancel,
                picks=picks, view_range=view_range, view_figsize=view_figsize)
            try:
                save_figure(fig, out, dpi=render_dpi, fmt=fmt, pdf_page=cfg["pdf_page"])
            except OSError as exc:
                # Most common cause: the target PDF/PNG is open in another app
                # (Acrobat, image viewer) → Windows locks it (PermissionError).
                # Surface a clear, actionable message instead of a raw traceback.
                from pathlib import Path
                from sbp_studio.core.tasks import ExportError
                name = Path(out).name
                if isinstance(exc, PermissionError):
                    msg = QCoreApplication.translate(
                        "SubTabbedTab",
                        "Cannot save “{0}”: the file is open in another program. "
                        "Close it and try again.").format(name)
                else:
                    detail = getattr(exc, "strerror", None) or str(exc)
                    msg = QCoreApplication.translate(
                        "SubTabbedTab",
                        "Cannot save “{0}”: {1}").format(name, detail)
                err = ExportError(msg)
                err.title = QCoreApplication.translate("SubTabbedTab", "Export failed")
                raise err from exc
            finally:
                fig.clear()   # releases imshow raster; Figure() is unmanaged — plt.close is a no-op
            return out

        self.tasks.run_task(
            job, self._on_image_exported,
            QCoreApplication.translate("SubTabbedTab", "Exporting image…"))

    def _on_image_exported(self, out: str) -> None:
        from pathlib import Path
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "Image saved: {0}").format(Path(out).name))

    # ── Render Viewport HQ → ephemeral in-viewer HQ overlay ───────────────────

    # Horizontal oversample for the overlay raster: ≥ this many px/trace so the
    # crop is smoothly interpolated (the live ImageItem is nearest-neighbour →
    # blocky when zoomed; the overlay restores the export's bilinear sharpness).
    _HQ_MIN_PX_PER_TRACE = 8.0
    _HQ_MAX_PX = 8000          # per-axis pixel cap for the overlay raster

    def _on_hq_preview_requested(self) -> None:
        """Render the currently visible ViewBox crop at Matplotlib-export quality
        and lay it over the live view as an ephemeral overlay (auto-removed on the
        next pan/zoom). No file is written."""
        obj = self._active_object()
        if obj is None or getattr(obj, "error", None):
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "Select an item first."))
            return
        if self._seismic is None or not self._seismic.has_image():
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "View a line first, then zoom to the area you want."))
            return
        # Snapshot the visible ViewBox window NOW (Qt objects aren't thread-safe).
        x_range, y_range = self._seismic.current_view_range()
        # Snapshot the EXACT amplitude levels the live image is using right now —
        # the HQ raster must reuse these verbatim (not recompute its own vmin/vmax
        # from the crop) or its colors drift from what's on screen, most visibly
        # when "[ -1 to 1 ]" (diverging) clipping is active.
        live_vmin, live_vmax = self._seismic.current_levels()

        params = dict(self.controls.display_params())
        params["clip_lo"] = 0.0
        params["fill_value"] = 0.0
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.active_nodes()]
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled
        handler = self._handler
        ppt = max(self._HQ_MIN_PX_PER_TRACE, float(self.controls.px_per_trace()))

        def job(progress, cancel) -> tuple:
            import numpy as np
            from sbp_studio.core.constants import CMAPS
            from sbp_studio.viz.render import _colorize_for_target
            if getattr(obj, "data", None) is None:
                progress(float("nan"), "Loading traces…")
            _obj = handler.load_full(obj, cancel)
            progress(float("nan"), "")
            # Full-resolution DSP (identical to the export), then crop to the view.
            proc, t0_full = _process_full_array(
                _obj, params, node_cfg, align_enabled, cancel)
            clone, crop = _crop_for_viewport(_obj, proc, t0_full, x_range, y_range)
            if crop.size == 0 or clone.n_traces < 1 or clone.ns < 1:
                raise RuntimeError("The visible area is empty — zoom onto the data.")
            cancel.check()

            # HQ colourise — the SAME smooth bilinear raster the Matplotlib export
            # embeds, upsampled so zoomed-in traces are sharp instead of blocky.
            # vmin/vmax are the live view's CURRENT levels (snapshotted above), not
            # recomputed from the crop — keeps colors pixel-identical to what's on
            # screen, including under fixed [-1, 1] diverging clipping.
            vmin, vmax = live_vmin, live_vmax
            cmap_name = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
            if params.get("inv_cmap"):
                cmap_name += "_r"
            tw = int(np.clip(clone.n_traces * ppt, 1000, self._HQ_MAX_PX))
            th = int(np.clip(clone.ns * 2, 1000, self._HQ_MAX_PX))
            rgba = _colorize_for_target(crop, cmap_name, vmin, vmax, (tw, th),
                                       interp=params.get("interp", "nearest"))

            # Crop bounds (snapped to the data grid) for an exact setRect mapping.
            dt_ms = clone.dt_us / 1000.0
            x0 = float(clone.dist_km[0]); x1 = float(clone.dist_km[-1])
            t0 = float(clone.min_delay); t1 = t0 + clone.ns * dt_ms
            return rgba, x0, x1, t0, t1

        self.tasks.run_task(
            job, self._on_hq_overlay_ready,
            QCoreApplication.translate("SubTabbedTab", "Rendering HQ preview…"))

    def _on_hq_overlay_ready(self, result: tuple) -> None:
        rgba, x0, x1, t0, t1 = result
        self._seismic.show_hq_overlay(rgba, x0, x1, t0, t1)
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "HQ preview shown — pan or zoom to dismiss."))

    # ── Batch export (many profiles / chains in one background pass) ──────────

    def export_batch(self, items: list, cfg: dict, handler: "SourceHandler") -> None:
        """Export every item in *items* to its own source folder, in ONE worker.

        Inherits THIS tab's current DSP/filter settings (controls + pipeline) and
        applies them uniformly to all items, via the SAME render pipeline as the
        single export (``_render_export_figure``) — so batch quality is pristine
        and identical. Each output lands in the item's source directory, named
        after that folder (``<folder>/<folder>.<fmt>``). Memory-safe: heavy data is
        loaded JIT and released after each item; the figure is closed every step.

        ``handler`` is the source strategy for the BATCH items, supplied by the
        caller (which sidebar list the batch came from). It is decoupled from the
        live view's current handler — a batch of profiles always exports as
        profiles even if a chain is on screen — and it is the ONLY thing that
        knows the source kind here: this method itself is fully type-agnostic.
        """
        fmt = cfg["format"]
        # Optional single output folder for the whole batch (empty = each item to
        # its own source folder). The per-line naming convention is preserved by
        # _batch_output_path — see its docstring.
        out_dir = (cfg.get("out_dir") or "").strip()
        # Snapshot the side-panel DSP/presentation state on the GUI thread (Qt
        # widgets are not thread-safe) — applied uniformly to every batch item.
        scale_cfg = self.controls.scale_config()
        params = dict(self.controls.display_params())
        params["clip_lo"] = 0.0
        params["fill_value"] = 0.0 if cfg["fill_zero"] else float("nan")
        params["draw_file_boundaries"] = bool(cfg["draw_file_boundaries"])
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.active_nodes()]
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled

        valid = [o for o in items
                 if o is not None and not getattr(o, "error", None)]
        if not valid:
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "Select an item first."))
            return

        def job(progress, cancel) -> tuple:
            from sbp_studio.viz.render import save_figure

            used: set = set()

            def make_out(obj):
                # Folder-named output (in out_dir when set, else the source
                # dir), de-duped so two items never silently overwrite.
                out = _batch_output_path(handler.source_path(obj), fmt, used, out_dir)
                used.add(str(out))
                return out

            def render_save(render_obj, out):
                fig = None
                try:
                    fig, render_dpi = _render_export_figure(
                        render_obj, cfg, params, node_cfg, scale_cfg,
                        align_enabled, handler, cancel)
                    save_figure(fig, str(out), dpi=render_dpi, fmt=fmt,
                                pdf_page=cfg["pdf_page"])
                finally:
                    if fig is not None:
                        fig.clear()  # releases imshow raster; Figure() unmanaged by pyplot

            def _report(frac, name):
                progress(frac, QCoreApplication.translate(
                    "SubTabbedTab", "Exporting {0}…").format(name) if name else "")

            # ── Process-pool branch (roadmap #1, GUI side) ────────────────────
            # Profiles only (parallel_export_safe: one item == one file path a
            # worker can rebuild from scratch), ≥2 items, and only when the
            # RAM budget affords ≥2 workers EACH holding a full item — a
            # worker keeps both the trace matrix and the render raster
            # resident, hence 2× the in-process per-item estimate. Every
            # worker runs the exact same render_export_figure the sequential
            # path delegates to (gui.export_headless — Qt-free by contract),
            # so pool output is pixel-identical. On a low-RAM machine
            # plan_workers resolves to 1 → the prefetch/sequential path below,
            # i.e. prior behaviour unchanged.
            if getattr(handler, "parallel_export_safe", False) and len(valid) >= 2:
                from sbp_studio.core._backends import plan_workers
                from sbp_studio.gui.export_headless import (
                    divide_mem_budget, run_batch_export_pool)
                n_workers = plan_workers(_estimate_batch_item_bytes(valid) * 2.0)
                if n_workers >= 2:
                    per_worker_cfg = divide_mem_budget(cfg, n_workers)
                    payloads = [dict(
                        path=handler.source_path(obj),
                        out=str(make_out(obj)),
                        name=(getattr(obj, "label", None)
                              or getattr(obj, "name", "item")),
                        fmt=fmt, pdf_page=cfg["pdf_page"],
                        cfg=per_worker_cfg, params=params, node_cfg=node_cfg,
                        scale_cfg=scale_cfg, align_enabled=align_enabled,
                    ) for obj in valid]
                    return run_batch_export_pool(
                        payloads, n_workers, progress=_report, cancel=cancel)

            # In-process fallback (chains, a single item, or a 1-worker RAM
            # budget): the prefetch-one pipeline + RAM-flat release in the
            # shared, unit-tested engine (see _run_batch_export_loop).
            return _run_batch_export_loop(
                valid, handler, make_out=make_out, render_save=render_save,
                progress=_report, cancel=cancel)

        self.tasks.run_task(
            job, self._on_batch_exported,
            QCoreApplication.translate("SubTabbedTab", "Batch export…"))

    def _on_batch_exported(self, result: tuple) -> None:
        saved, failed = result
        if failed:
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "Batch export: {0} saved, {1} failed.").format(
                    len(saved), len(failed)))
        else:
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "Batch export complete: {0} file(s) saved.").format(
                    len(saved)))

    def _on_export_fix_requested(self) -> None:
        obj = self._active_object()
        if obj is None or getattr(obj, "error", None):
            self.tasks.notify(QCoreApplication.translate("SubTabbedTab", "Select an item first."))
            return
        out, _ = QFileDialog.getSaveFileName(
            self, QCoreApplication.translate("SubTabbedTab", "Export FIX marks"),
            f"{self._export_basename()}_fix.shp",
            QCoreApplication.translate("SubTabbedTab", "Shapefile (*.shp);;GeoJSON (*.geojson);;CSV (*.csv)"))
        if not out:
            return
        interval = int(self.controls.fix_interval.value())
        ts, dist, lons, lats = obj.timestamps, obj.dist_km, obj.lons, obj.lats

        def job(progress, cancel) -> tuple:
            from sbp_studio.core import (
                compute_fix_positions, fixes_to_wgs84, write_fix_points_csv,
                write_fix_points_geojson, write_fix_points_shp,
            )
            progress(float("nan"), "")
            pts = compute_fix_positions(ts, dist, lons, lats, interval)
            # GIS boundary: projected native metres → the WGS84 every FIX
            # writer declares (same fix as the interpretation-marks export —
            # see core.geometry_export.fixes_to_wgs84). Plain attribute
            # reads on obj, worker-safe.
            pts = fixes_to_wgs84(pts, obj)
            low = out.lower()
            if low.endswith((".geojson", ".json")):
                write_fix_points_geojson(out, pts)
            elif low.endswith(".csv"):
                write_fix_points_csv(out, pts)
            else:
                write_fix_points_shp(out, pts)
            return out, len(pts)

        self.tasks.run_task(
            job, self._on_fix_exported,
            QCoreApplication.translate("SubTabbedTab", "Exporting FIX marks…"))

    def _on_fix_exported(self, result: tuple) -> None:
        _out, n = result
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "{0} FIX marks exported.").format(n))

    # ── Interpretation & Picking (Phase 3) ──────────────────────────────────

    def _on_picking_export_import(self) -> None:
        """'Exportar/Importar' clicked: show the chooser, then drive whichever
        native QFileDialog the user picked — same dispatch-by-extension
        convention as _on_export_fix_requested above."""
        picks = self._seismic.get_picks()
        dlg = PickingExportImportDialog(self, has_existing_picks=bool(picks))
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.choice() is None:
            return
        if dlg.choice() == PickingExportImportDialog.EXPORT:
            self._export_picking_session(picks)
        else:
            self._import_picking_session()

    def _export_picking_session(self, picks: list) -> None:
        if not picks:
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "No interpretation markers to export."))
            return
        out, _ = QFileDialog.getSaveFileName(
            self, QCoreApplication.translate("SubTabbedTab", "Export interpretation markers"),
            f"{self._export_basename()}_picks.shp",
            QCoreApplication.translate(
                "SubTabbedTab",
                "Shapefile (*.shp);;GeoJSON (*.geojson);;CSV (*.csv);;"
                "Internal session (*.tps)"))
        if not out:
            return

        # Captured on the GUI thread: the profile/chain the picks were made on
        # — export_picks uses its coord_unit/detected_crs to convert projected
        # native coordinates to the WGS84 the .shp/.geojson formats declare
        # (core.picking.picks_to_wgs84). Plain attribute reads, worker-safe.
        source = self._seismic.get_picking_source()

        def job(progress, cancel) -> tuple:
            from sbp_studio.core import export_picks, save_picks_tps
            progress(float("nan"), "")
            if out.lower().endswith(".tps"):
                save_picks_tps(out, picks)
                return out, len(picks)
            written = export_picks(out, picks, source=source)
            return written, len(picks)

        self.tasks.run_task(
            job, self._on_picking_exported,
            QCoreApplication.translate("SubTabbedTab", "Exporting interpretation markers…"))

    def _on_picking_exported(self, result: tuple) -> None:
        _out, n = result
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "{0} interpretation marker(s) exported.").format(n))

    def _import_picking_session(self) -> None:
        """Import is only reachable via the dialog when the current list is
        EMPTY (see PickingExportImportDialog) — set_picks still replaces the
        list wholesale, defensively, even if called some other way."""
        path, _ = QFileDialog.getOpenFileName(
            self, QCoreApplication.translate("SubTabbedTab", "Import interpretation markers"),
            "", QCoreApplication.translate(
                "SubTabbedTab", "Internal session (*.tps)"))
        if not path:
            return
        try:
            from sbp_studio.core import load_picks_tps
            picks = load_picks_tps(path)
        except (OSError, ValueError) as exc:
            self.tasks.show_error(QCoreApplication.translate(
                "SubTabbedTab", "Could not read {0}: {1}").format(path, exc))
            return
        self._seismic.set_picks(picks)
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "{0} interpretation marker(s) imported.").format(len(picks)))

    # ── Construction ────────────────────────────────────────────────────────

    _DOCK_FEATURES = (QDockWidget.DockWidgetFeature.DockWidgetFloatable
                      | QDockWidget.DockWidgetFeature.DockWidgetMovable)

    def _build_controls_dock(self) -> QDockWidget:
        # Controls dock = the DSP node pipeline (the processing source) + the
        # retained presentation/output controls (palette, clip, FIX, scale,
        # export). The static DSP filter sections are hidden — the pipeline
        # replaces them. (The Link Views toggle used to live here too; it now
        # lives at the top of the Profile/Seismic dock — see _build_view_docks.)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # User-resizable, but NEVER a greedy space-consumer: a MINIMUM width
        # so every control stays readable, and a MAXIMUM so the dock layout
        # engine cannot hand it half of whatever new space appears when the
        # window is maximized (Qt's QDockAreaLayout otherwise splits newly
        # freed space ~50/50 between competing dock areas — wasting it here
        # on a column of static-width controls instead of the seismic
        # section). The user can still narrow it below this by dragging the
        # splitter; only growth beyond a comfortable working width is capped.
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(380)
        scroll.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        # ProcessingControls' stacked tab PAGES are the SOLE root content
        # here — the pipeline panel is embedded INSIDE the "Controles y
        # Procesado" page (at its very top, above Palette), not stacked
        # above the tabs in a separate wrapper layout, so every control
        # genuinely lives inside one of the two tabs. The tab STRIP itself
        # is promoted into the dock's custom title bar below (DockTitleBar)
        # rather than living here too — see processing_controls.py's
        # _DraggableTabBar/tab_bar() for why the strip and the pages are two
        # separate widgets.
        self.pipeline_panel = PipelinePanel()
        self.controls = ProcessingControls()
        self.controls.set_dsp_sections_visible(False)
        self.controls.embed_pipeline_panel(self.pipeline_panel)

        scroll.setWidget(self.controls)

        dock = QDockWidget()
        dock.setFeatures(self._DOCK_FEATURES)
        dock.setWidget(scroll)
        # Custom title bar: promotes the "Controles y Procesado"/"Marcas y
        # Exportación" tabs to the dock's absolute top, replacing the
        # native bar (which would otherwise show a now-redundant
        # "Controles" label directly above them).
        tab_bar = self.controls.tab_bar()
        tab_bar.set_dock(dock)
        dock.setTitleBarWidget(DockTitleBar(dock, tab_bar))
        self._install_restore_menu(dock)
        self._dock_controls = dock
        return dock

    def _build_view_docks(self) -> None:
        """Profile (Seismic) / Map / Spectrum / Headers, each wrapped in its
        own QDockWidget — multi-monitor support: DockWidgetFloatable +
        DockWidgetMovable let the user drag any one of them out into a
        floating window on a second screen while the rest of the app stays
        put. ``tabifyDockWidget`` then groups all four into the SAME tab
        strip Qt renders for tabified docks, so by default the workspace
        looks and behaves exactly like the old QTabWidget sub-notebook —
        the difference is purely that each "tab" can now be torn off."""
        self.pages: list[_Page] = []
        self._docks: list[QDockWidget] = []
        for i in range(4):
            page = _Page()
            # The greedy space-consumer: Expanding (not the QStackedWidget
            # default of Preferred) so this is the side QMainWindow's dock
            # layout hands ALL newly freed width to on maximize, instead of
            # splitting it ~50/50 with the now width-capped Controls dock
            # (see _build_controls_dock).
            page.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.pages.append(page)
            dock = QDockWidget()
            dock.setFeatures(self._DOCK_FEATURES)
            if i == PROFILE:
                # Cross-module sync toggle (Link Views) lives at the top of
                # the Profile/Seismic dock — its old home, a row above the
                # tab strip, no longer exists now that the views are docks
                # rather than tab pages. Governs ONLY the live navigation
                # cursor + double-click POI features — the existing
                # always-on visible-segment/click-to-jump sync is unrelated
                # and stays unconditional. Unchecked by default (see
                # _on_link_views_toggled).
                profile_host = QWidget()
                profile_host.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                profile_col = QVBoxLayout(profile_host)
                profile_col.setContentsMargins(0, 0, 0, 0)
                profile_col.setSpacing(0)
                row = QHBoxLayout()
                row.setContentsMargins(4, 2, 4, 2)
                self.chk_link_views = QCheckBox()
                self.chk_link_views.setChecked(False)
                self.chk_link_views.toggled.connect(self._on_link_views_toggled)
                row.addWidget(self.chk_link_views)
                row.addStretch(1)
                profile_col.addLayout(row)
                profile_col.addWidget(page, 1)
                dock.setWidget(profile_host)
            else:
                dock.setWidget(page)
            self._install_restore_menu(dock)
            self._docks.append(dock)

        self._dock_host.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._docks[PROFILE])
        for dock in self._docks[1:]:
            self._dock_host.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self._dock_host.tabifyDockWidget(self._docks[PROFILE], dock)
        self._docks[PROFILE].raise_()           # Profile active by default
        # Just-in-time CRS prompting (Task 2): the old subtabs.currentChanged
        # equivalent. A tabified QDockWidget has no currentChanged(index)
        # signal, but visibilityChanged(bool) fires exactly when it becomes
        # the raised/visible tab (or is hidden) — the correct substitute.
        self._docks[MAP].visibilityChanged.connect(self._on_map_dock_visible)

    def _install_restore_menu(self, dock: QDockWidget) -> None:
        """Right-click a FLOATING panel → 'Restaurar a la ventana principal'
        re-docks it; Qt's own layout engine snaps it back to wherever it was
        pulled from. The menu only appears while the dock is actually
        floating — right-clicking a docked panel does nothing here."""
        dock.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        dock.customContextMenuRequested.connect(
            lambda pos, d=dock: self._show_dock_restore_menu(d, pos))

    def _show_dock_restore_menu(self, dock: QDockWidget, pos) -> None:
        if not dock.isFloating():
            return
        menu = QMenu(dock)
        # Exact literal label requested by spec — intentionally NOT routed
        # through self.tr()/translate() like every other string in this
        # file, which all use an English source + a Spanish .ts entry.
        act = menu.addAction("Restaurar a la ventana principal")
        act.triggered.connect(lambda: dock.setFloating(False))
        menu.exec(dock.mapToGlobal(pos))

    # ── i18n ────────────────────────────────────────────────────────────────

    def retranslate_ui(self) -> None:
        # Fixed context: literals defined here but invoked from subclass
        # instances, so self.tr() would resolve the wrong context. Inline
        # translate() calls so pylupdate6 can extract them.
        titles = (
            QCoreApplication.translate("SubTabbedTab", "Profile"),
            QCoreApplication.translate("SubTabbedTab", "Map"),
            QCoreApplication.translate("SubTabbedTab", "Spectrum"),
            QCoreApplication.translate("SubTabbedTab", "Headers"),
        )
        for i, title in enumerate(titles):
            self._docks[i].setWindowTitle(title)
        # Not shown anywhere visually any more (the custom DockTitleBar
        # replaced the native bar that would have displayed this) — kept
        # only as the OS-level window label (taskbar/Alt-Tab) for when this
        # dock is torn off into a floating window.
        self._dock_controls.setWindowTitle(
            QCoreApplication.translate("SubTabbedTab", "Controls"))
        self.chk_link_views.setText(QCoreApplication.translate(
            "SubTabbedTab", "Link Views"))
        self.chk_link_views.setToolTip(QCoreApplication.translate(
            "SubTabbedTab",
            "When active: hovering the seismic section shows a live cursor on "
            "the map, and double-clicking adds a Point of Interest marker."))
        msg = self.empty_message()
        for page in self.pages:
            page.placeholder.set_text(msg)
