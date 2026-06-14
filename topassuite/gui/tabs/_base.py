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

from typing import Optional

from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QFrame, QScrollArea, QSplitter, QStackedWidget,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..components import (
    ExportDialog, HeaderView, MapView, PipelinePanel, PlaceholderView,
    ProcessingControls, SeismicView, SpectrumView,
)
from ..dsp import DSPContext, PreviewController
from ..i18n import language_manager
from ..state import AppState
from ._render import compute_figsize, effective_export_dpi

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


def _batch_output_path(src_path: str, fmt: str, used: set) -> "Path":
    """Folder-named export path for a source SEG-Y file: ``<dir>/<dir>.<fmt>``.

    e.g. ``Z:/data/Line_01/1.sgy`` → ``Z:/data/Line_01/Line_01.pdf``. If that path
    was already chosen in this batch (two items share a folder) it is de-duped
    with the file stem so nothing is silently overwritten. Returns a ``Path``; the
    caller adds ``str(path)`` to ``used``.
    """
    from pathlib import Path
    src_dir = Path(src_path).parent
    out = src_dir / f"{src_dir.name}.{fmt}"
    if str(out) in used:
        out = src_dir / f"{src_dir.name}_{Path(src_path).stem}.{fmt}"
    return out


def _render_export_figure(obj, cfg, params, node_cfg, aspect, align_enabled,
                          is_chain, cancel):
    """Shared per-item export render — used by BOTH the single export and every
    batch item so their quality can never drift.

    Runs the FULL-resolution DSP pipeline (static delay-alignment + dynamic nodes)
    on ``obj.data`` (the 100 % native matrix — never the live view's decimated
    ``_arr``), then renders to a Matplotlib Figure at a DPI floored by
    ``effective_export_dpi`` so the embedded raster is never decimated, then
    applies the WYSIWYG aspect fit. Returns ``(fig, render_dpi)``; the caller
    saves and closes the figure.
    """
    from topassuite.core import apply_delay_alignment
    from topassuite.gui.dsp import DSPContext, make_node
    from topassuite.viz.render import (
        build_theme, render_chain_figure, render_profile_figure,
    )
    # 1) STATIC geometry: delay alignment on the FULL array first.
    data = obj.data.copy()
    if align_enabled and getattr(obj, "delays", None) is not None:
        data = apply_delay_alignment(
            data, obj.delays, obj.min_delay, obj.dt_us,
            fill_value=params["fill_value"])
    # 2) Dynamic DSP node pipeline over the FULL-resolution array.
    ctx = DSPContext.from_source(obj)
    for key, npar in node_cfg:
        cancel.check()
        data = make_node(key, npar).apply(data, ctx)
    cancel.check()
    # Figure size from native dims; DPI floored so target ≥ (n_traces, ns) → the
    # core never decimates the matrix (the historical pristine-CLI fidelity).
    figsize = compute_figsize(obj, int(cfg["dpi"]), None, aspect, cfg["velocity"])
    render_dpi = effective_export_dpi(figsize, data.shape, int(cfg["dpi"]))
    render_opts = dict(
        x_tick_km=cfg["x_tick"], t_tick_ms=cfg["t_tick"], show_grid=cfg["grid"],
        title_override=None, clip_lo=0.0, time_tick_min=cfg["time_ticks"],
        margin_top_ms=cfg["margin_top"], margin_bottom_ms=cfg["margin_bottom"],
        time_fmt=cfg["time_fmt"], time_font_size=cfg["time_font_size"],
        time_align=cfg["time_align"], fix_font_size=5.0,
        fix_bbox_alpha=cfg["fix_bbox_alpha"], fix_color=cfg["fix_color"],
        colors=build_theme(theme=cfg["theme"]))
    render = render_chain_figure if is_chain else render_profile_figure
    fig = render(obj, data, params, figsize=figsize, dpi=render_dpi, **render_opts)
    # WYSIWYG aspect fit: grow the figure so the DATA box hits `aspect` at full
    # size (decorations take a fixed inch margin) — restores the data-area pixels.
    if aspect:
        seis = next((a for a in fig.axes if a.get_images()), None)
        if seis is not None:
            for _ in range(4):
                fig.canvas.draw()
                pos = seis.get_position()
                fw, fh = fig.get_size_inches()
                if pos.width <= 0 or pos.height <= 0:
                    break
                data_w = pos.width * fw
                margin_v = fh - pos.height * fh   # absolute non-data height
                fig.set_size_inches(fw, data_w / aspect + margin_v)
                try:
                    fig.tight_layout(pad=1.2)
                except Exception:
                    pass
    return fig, render_dpi


class SubTabbedTab(QWidget):
    """Controls column + four-view sub-notebook with real Render dispatch."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.tasks = tasks  # task service (MainWindow.run_task / notify / show_error)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_subtabs())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 1000])
        root.addWidget(split)

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

        # Render button → re-fit the whole section (the live loop is automatic).
        self.controls.render_requested.connect(self.preview.fit)
        self.controls.export_image_requested.connect(self._on_export_image_requested)
        self.controls.export_fix_requested.connect(self._on_export_fix_requested)
        self.controls.scale_changed.connect(self._on_scale_changed)
        self.controls.boundaries_toggled.connect(self._on_boundaries_toggled)
        self.controls.align_toggled.connect(lambda *_: self.preview.alignment_changed())
        self.controls.display_changed.connect(self.preview.display_changed)
        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

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
        self.subtabs.setCurrentIndex(MAP)

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

    def _is_chain(self) -> bool:
        """True if the active object is a ProfileChain (Chains tab)."""
        return False

    def _on_scale_changed(self) -> None:
        """Apply the new display aspect to the on-screen section live."""
        if self._seismic is not None and self._seismic.has_image():
            self._seismic.set_aspect(self.controls.aspect())

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

    def _on_map_layer_requested(self, path: str) -> None:
        """Read a GIS overlay file in the core (off the GUI thread) and add the
        parsed WGS84 layer to the map."""
        def job(progress, cancel):
            from topassuite.core import read_gis_layer
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
            from topassuite.core import compute_spectrum
            from topassuite.gui.dsp import DSPContext, make_node
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
    # (topassuite.viz.render) on a worker thread — the same high-quality,
    # PDF-capable path as the CLI `export-image` — NOT a PyQtGraph screenshot.

    def _on_export_image_requested(self) -> None:
        obj = self._active_object()
        if obj is None or getattr(obj, "error", None):
            self.tasks.notify(QCoreApplication.translate("SubTabbedTab", "Select an item first."))
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config()
        fmt = cfg["format"]
        out, _ = QFileDialog.getSaveFileName(
            self, QCoreApplication.translate("SubTabbedTab", "Export image"),
            f"{self._export_basename()}.{fmt}",
            f"{fmt.upper()} (*.{fmt})")
        if not out:
            return

        is_chain = self._is_chain()
        aspect = self.controls.aspect()  # scale == the visualizer's aspect ratio
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
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.nodes()]
        # Static delay-alignment (geometry) is applied to the base BEFORE the
        # dynamic nodes — exactly as the preview does — read from the static
        # checkbox. ``params["align"]`` drives the renderer's time axis
        # (t0 = min_delay when aligned).
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled

        def job(progress, cancel) -> str:
            from topassuite.core import load_profile as _lp
            from topassuite.viz.render import save_figure
            # Lazy-load safety: profiles are header-only stubs and CHAINS assemble
            # their matrix lazily. Load/assemble JIT on the worker thread so the
            # export never fails (race, or chain never viewed).
            _obj = obj
            if getattr(_obj, "data", None) is None:
                progress(float("nan"), "Loading traces for export…")
                if is_chain:
                    _obj.load_chain_traces(cancel=cancel)
                else:
                    _obj = _lp(_obj.path, load_traces=True)
                    if getattr(_obj, "error", None):
                        raise RuntimeError(f"Cannot load {_obj.name}: {_obj.error}")
            progress(float("nan"), "")
            # Shared render pipeline (identical to every batch item): full-res DSP
            # + decimation-free DPI floor + WYSIWYG aspect fit.
            fig, render_dpi = _render_export_figure(
                _obj, cfg, params, node_cfg, aspect, align_enabled, is_chain, cancel)
            try:
                save_figure(fig, out, dpi=render_dpi, fmt=fmt, pdf_page=cfg["pdf_page"])
            except OSError as exc:
                # Most common cause: the target PDF/PNG is open in another app
                # (Acrobat, image viewer) → Windows locks it (PermissionError).
                # Surface a clear, actionable message instead of a raw traceback.
                from pathlib import Path
                from topassuite.core.tasks import ExportError
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
                fig.clear()
            return out

        self.tasks.run_task(
            job, self._on_image_exported,
            QCoreApplication.translate("SubTabbedTab", "Exporting image…"))

    def _on_image_exported(self, out: str) -> None:
        from pathlib import Path
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "Image saved: {0}").format(Path(out).name))

    # ── Batch export (many profiles / chains in one background pass) ──────────

    def export_batch(self, items: list, cfg: dict) -> None:
        """Export every item in *items* to its own source folder, in ONE worker.

        Inherits THIS tab's current DSP/filter settings (controls + pipeline) and
        applies them uniformly to all items, via the SAME render pipeline as the
        single export (``_render_export_figure``) — so batch quality is pristine
        and identical. Each output lands in the item's source directory, named
        after that folder (``<folder>/<folder>.<fmt>``). Memory-safe: heavy data is
        loaded JIT and released after each item; the figure is closed every step.
        """
        is_chain = self._is_chain()
        fmt = cfg["format"]
        # Snapshot the side-panel DSP/presentation state on the GUI thread (Qt
        # widgets are not thread-safe) — applied uniformly to every batch item.
        aspect = self.controls.aspect()
        params = dict(self.controls.display_params())
        params["clip_lo"] = 0.0
        params["fill_value"] = 0.0 if cfg["fill_zero"] else float("nan")
        params["draw_file_boundaries"] = bool(cfg["draw_file_boundaries"])
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.nodes()]
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled

        valid = [o for o in items
                 if o is not None and not getattr(o, "error", None)]
        if not valid:
            self.tasks.notify(QCoreApplication.translate(
                "SubTabbedTab", "Select an item first."))
            return

        def job(progress, cancel) -> tuple:
            import matplotlib.pyplot as plt
            from topassuite.core import load_profile as _lp
            from topassuite.core.tasks import Cancelled
            from topassuite.viz.render import save_figure

            n = len(valid)
            saved: list = []
            failed: list = []
            used: set = set()
            for i, obj in enumerate(valid):
                cancel.check()
                name = getattr(obj, "label", None) or getattr(obj, "name", "item")
                progress(i / n, QCoreApplication.translate(
                    "SubTabbedTab", "Exporting {0}…").format(name))

                # Source dir → folder-named output, de-duped so two items from the
                # same folder never silently overwrite each other.
                src_path = obj.profiles[0].path if is_chain else obj.path
                out = _batch_output_path(src_path, fmt, used)
                used.add(str(out))

                was_loaded = getattr(obj, "data", None) is not None
                render_obj = obj
                fig = None
                try:
                    if not was_loaded:                       # JIT lazy-load
                        if is_chain:
                            obj.load_chain_traces(cancel=cancel)
                        else:
                            render_obj = _lp(obj.path, load_traces=True)
                            if getattr(render_obj, "error", None):
                                failed.append((name, render_obj.error))
                                continue
                    fig, render_dpi = _render_export_figure(
                        render_obj, cfg, params, node_cfg, aspect,
                        align_enabled, is_chain, cancel)
                    save_figure(fig, str(out), dpi=render_dpi, fmt=fmt,
                                pdf_page=cfg["pdf_page"])
                    saved.append(str(out))
                except Cancelled:
                    raise                                    # abort the whole batch
                except Exception as exc:                     # one bad item ≠ kill batch
                    failed.append((name, str(exc)))
                finally:
                    if fig is not None:
                        # CRITICAL: free the figure every step. render.py builds
                        # UNMANAGED Figure() objects, so plt.close alone is a no-op
                        # for them — fig.clear() releases the heavy imshow raster.
                        fig.clear()
                        plt.close(fig)
                    # Release JIT-loaded heavy data → batch stays RAM-flat. Chains
                    # mutate in place, so revert to a stub; transient profiles GC.
                    if not was_loaded and is_chain:
                        obj.data = None
                        obj.clip_p99 = None
            progress(1.0, "")
            return saved, failed

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
            from topassuite.core import (
                compute_fix_positions, write_fix_points_csv,
                write_fix_points_geojson, write_fix_points_shp,
            )
            progress(float("nan"), "")
            pts = compute_fix_positions(ts, dist, lons, lats, interval)
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

    # ── Construction ────────────────────────────────────────────────────────

    def _build_controls(self) -> QWidget:
        # Left column = the DSP node pipeline (the new processing source) on top,
        # and the retained presentation/output controls below (palette, clip,
        # FIX, scale, export). The static DSP filter sections are hidden — the
        # pipeline replaces them.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(280)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        host = QWidget()
        col = QVBoxLayout(host)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        self.pipeline_panel = PipelinePanel()
        col.addWidget(self.pipeline_panel)

        self.controls = ProcessingControls()
        self.controls.set_dsp_sections_visible(False)
        col.addWidget(self.controls)

        scroll.setWidget(host)
        return scroll

    def _build_subtabs(self) -> QWidget:
        self.subtabs = QTabWidget()
        self.pages: list[_Page] = []
        for _ in range(4):
            page = _Page()
            self.pages.append(page)
            self.subtabs.addTab(page, "")
        return self.subtabs

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
            self.subtabs.setTabText(i, title)
        msg = self.empty_message()
        for page in self.pages:
            page.placeholder.set_text(msg)
