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
    QDialog, QFileDialog, QFrame, QMessageBox, QScrollArea, QSplitter,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)

from ..components import (
    ExportDialog, HeaderView, MapView, PipelinePanel, PlaceholderView,
    ProcessingControls, SeismicView, SpectrumView,
)
from ..dsp import DSPContext, PreviewController
from ..i18n import language_manager
from ..state import AppState
from ._render import (
    dpi_for_budget, effective_aspect, effective_export_dpi, figsize_for_scale,
)

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


def _process_full_array(obj, params, node_cfg, align_enabled, cancel):
    """Run static delay-alignment + the dynamic DSP nodes on the FULL native
    matrix. Returns ``(processed, t0_ms)`` where ``t0_ms`` is the time of the
    processed array's row 0 (min_delay when aligned, else delay_ms)."""
    from sbp_studio.core import apply_delay_alignment
    from sbp_studio.gui.dsp import DSPContext, make_node
    data = obj.data.copy()
    t0 = float(getattr(obj, "delay_ms", 0.0) or 0.0)
    if align_enabled and getattr(obj, "delays", None) is not None:
        data = apply_delay_alignment(
            data, obj.delays, obj.min_delay, obj.dt_us,
            fill_value=params["fill_value"])
        t0 = float(getattr(obj, "min_delay", 0.0) or 0.0)
    ctx = DSPContext.from_source(obj)
    for key, npar in node_cfg:
        cancel.check()
        data = make_node(key, npar).apply(data, ctx)
    cancel.check()
    return data, t0


def _crop_for_viewport(obj, proc, t0_full, x_range, y_range):
    """Crop the processed matrix to the visible ViewBox window and return a
    lightweight source clone the headless renderer can consume.

    ``x_range`` is (km, km), ``y_range`` is (ms, ms) straight from the PyQtGraph
    ViewBox. Columns are mapped via the per-trace distance axis (searchsorted) and
    rows via the processed array's time origin ``t0_full``. The clone carries only
    the attributes ``render_profile_figure`` + ``figsize_for_scale`` read, with the
    per-trace arrays sliced so nothing is misaligned."""
    import numpy as np
    from types import SimpleNamespace
    dist = np.asarray(obj.dist_km, dtype=float)
    n_rows, n_cols = proc.shape
    xa, xb = sorted((float(x_range[0]), float(x_range[1])))
    c0 = int(np.clip(np.searchsorted(dist, xa, side="left"), 0, n_cols - 1))
    c1 = int(np.clip(np.searchsorted(dist, xb, side="right"), c0 + 1, n_cols))
    dt_ms = obj.dt_us / 1000.0
    ya, yb = sorted((float(y_range[0]), float(y_range[1])))
    r0 = int(np.clip(round((ya - t0_full) / dt_ms), 0, n_rows - 1))
    r1 = int(np.clip(round((yb - t0_full) / dt_ms), r0 + 1, n_rows))
    crop = np.ascontiguousarray(proc[r0:r1, c0:c1])

    def _sl(name):
        arr = getattr(obj, name, None)
        return None if arr is None else np.asarray(arr)[c0:c1]

    dk = dist[c0:c1]
    t0_crop = t0_full + r0 * dt_ms
    base_name = getattr(obj, "name", None) or getattr(obj, "label", "viewport")
    clone = SimpleNamespace(
        name=f"{base_name} · viewport",
        n_traces=int(c1 - c0), ns=int(r1 - r0), dt_us=int(obj.dt_us),
        dist_km=dk, total_km=float(dk[-1] - dk[0]) if dk.size else 0.0,
        timestamps=list(getattr(obj, "timestamps", []) or [])[c0:c1],
        lons=_sl("lons"), lats=_sl("lats"), delays=_sl("delays"),
        water_depth=_sl("water_depth"),
        # Already-processed data → render with align=False; the time origin is the
        # crop's first row, so both align branches of time_window yield t0_crop.
        min_delay=t0_crop, delay_ms=t0_crop, max_delay=t0_crop,
    )
    return clone, crop


def _render_export_figure(obj, cfg, params, node_cfg, scale_cfg, align_enabled,
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
    from sbp_studio.viz.render import (
        build_theme, render_chain_figure, render_profile_figure,
    )
    # Full-resolution DSP pipeline (alignment + nodes) — shared with the viewport
    # HQ export so the crop is processed identically.
    data, _t0_full = _process_full_array(
        obj, params, node_cfg, align_enabled, cancel)
    # ORIGINAL scaling: figsize from compute_figsize (n_traces·px/dpi width floored
    # at 8in, height from the aspect ratio) — the proven, normal export sizing.
    # effective_export_dpi is the ONLY retained safeguard: it raises the render DPI
    # just enough that target ≥ (n_traces, ns) so Matplotlib never silently
    # DOWNSAMPLES the matrix (which softened deep-profile PDFs). It adjusts pixel
    # density only — the physical figsize (scaling) is untouched.
    # Figsize from the SELECTED scale mode (aspect / VE / hybrid). The exported
    # aspect ratio (and VE) are fixed by the mode; effective_export_dpi then raises
    # the DPI so the embedded raster carries the full native grid (no decimation).
    figsize = figsize_for_scale(obj, scale_cfg, int(cfg["dpi"]), cfg["velocity"])
    render_dpi = effective_export_dpi(figsize, data.shape, int(cfg["dpi"]))
    # RAM cap (parity with the CLI): never let the raster exceed the memory
    # budget. effective_export_dpi can raise the DPI toward 2400 to hit the native
    # grid; for a long/deep line that is gigapixels → the GUI export hangs. Cap the
    # DPI to what the budget allows. figsize (the aspect/VE proportions) is kept
    # exact — only pixel density drops, exactly like the CLI's coupled down-scale.
    dpi_cap = dpi_for_budget(figsize, cfg.get("mem_budget_gb"))
    if dpi_cap is not None and render_dpi > dpi_cap:
        render_dpi = max(50, dpi_cap)
    eff_aspect = figsize[0] / figsize[1] if figsize[1] > 0 else None
    render_opts = dict(
        x_tick_km=cfg["x_tick"], t_tick_ms=cfg["t_tick"], show_grid=cfg["grid"],
        title_override=None, clip_lo=0.0, time_tick_min=cfg["time_ticks"],
        margin_top_ms=cfg["margin_top"], margin_bottom_ms=cfg["margin_bottom"],
        time_fmt=cfg["time_fmt"], time_font_size=cfg["time_font_size"],
        time_align=cfg["time_align"], fix_font_size=5.0,
        fix_bbox_alpha=cfg["fix_bbox_alpha"], fix_color=cfg["fix_color"],
        axis_font_size=cfg.get("axis_font_size", 7.0),
        grid_alpha=cfg.get("grid_alpha", 0.18), grid_lw=cfg.get("grid_lw", 0.5),
        colors=build_theme(theme=cfg["theme"]))
    render = render_chain_figure if is_chain else render_profile_figure
    fig = render(obj, data, params, figsize=figsize, dpi=render_dpi, **render_opts)
    # WYSIWYG aspect fit: grow the figure so the DATA box hits the mode's effective
    # aspect at full size (decorations take a fixed inch margin) — restores pixels.
    aspect = eff_aspect
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
        # Wider default so every filter control is visible without scrolling, and a
        # grippy, non-collapsible handle so the user can drag the panel edge freely.
        split.setChildrenCollapsible(False)
        split.setHandleWidth(6)
        split.setSizes([360, 1000])
        self._main_split = split
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

        # Render Full → re-fit the whole section (the live loop is automatic).
        self.controls.render_requested.connect(self.preview.fit)
        # Render Viewport HQ → HQ Matplotlib-quality render of the visible crop,
        # shown as an ephemeral overlay INSIDE the live viewer (no file export).
        self.controls.render_viewport_requested.connect(
            self._on_hq_preview_requested)
        self.controls.export_image_requested.connect(self._on_export_image_requested)
        self.controls.export_fix_requested.connect(self._on_export_fix_requested)
        self.controls.scale_changed.connect(self._on_scale_changed)
        self.controls.boundaries_toggled.connect(self._on_boundaries_toggled)
        self.controls.align_toggled.connect(lambda *_: self.preview.alignment_changed())
        self.controls.display_changed.connect(self.preview.display_changed)
        # Live export-DPI readout (Part 2): refresh when the scale settings OR the
        # active source change, so the user always sees the resolution that will be
        # generated for the current configuration.
        self.state.active_profile_changed.connect(lambda *_: self._update_dpi_estimate())
        self.state.active_chain_changed.connect(lambda *_: self._update_dpi_estimate())
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
                           velocity=1500.0)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.config()
        # RAM validation: if the requested memory budget exceeds the machine's
        # actual free RAM, alert before rendering so the user can cancel and lower
        # it rather than risk an out-of-memory crash.
        if not self._confirm_memory_budget(cfg.get("mem_budget_gb", 6.0)):
            return
        fmt = cfg["format"]
        out, _ = QFileDialog.getSaveFileName(
            self, QCoreApplication.translate("SubTabbedTab", "Export image"),
            f"{self._export_basename()}.{fmt}",
            f"{fmt.upper()} (*.{fmt})")
        if not out:
            return

        is_chain = self._is_chain()
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
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.nodes()]
        # Static delay-alignment (geometry) is applied to the base BEFORE the
        # dynamic nodes — exactly as the preview does — read from the static
        # checkbox. ``params["align"]`` drives the renderer's time axis
        # (t0 = min_delay when aligned).
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled

        def job(progress, cancel) -> str:
            import matplotlib.pyplot as plt
            from sbp_studio.core import load_profile as _lp
            from sbp_studio.viz.render import save_figure
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
                _obj, cfg, params, node_cfg, scale_cfg, align_enabled, is_chain, cancel)
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
                # Rule 7 (Matplotlib memory): release the figure every job.
                fig.clear()
                plt.close(fig)
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

        params = dict(self.controls.display_params())
        params["clip_lo"] = 0.0
        params["fill_value"] = 0.0
        node_cfg = [(n.KEY, dict(n.params)) for n in self.pipeline_panel.nodes()]
        align_enabled = self.controls.align_enabled()
        params["align"] = align_enabled
        is_chain = self._is_chain()
        ppt = max(self._HQ_MIN_PX_PER_TRACE, float(self.controls.px_per_trace()))

        def job(progress, cancel) -> tuple:
            import numpy as np
            from sbp_studio.core import load_profile as _lp
            from sbp_studio.core.constants import CMAPS
            from sbp_studio.viz.render import _colorize_for_target, _vmin_vmax
            _obj = obj
            if getattr(_obj, "data", None) is None:
                progress(float("nan"), "Loading traces…")
                if is_chain:
                    _obj.load_chain_traces(cancel=cancel)
                else:
                    _obj = _lp(_obj.path, load_traces=True)
                    if getattr(_obj, "error", None):
                        raise RuntimeError(f"Cannot load {_obj.name}: {_obj.error}")
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
            vmin, vmax = _vmin_vmax(crop, params.get("clip", 99.6),
                                    params.get("clip_lo", 0.0))
            cmap_name = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
            if params.get("inv_cmap"):
                cmap_name += "_r"
            tw = int(np.clip(clone.n_traces * ppt, 1000, self._HQ_MAX_PX))
            th = int(np.clip(clone.ns * 2, 1000, self._HQ_MAX_PX))
            rgba = _colorize_for_target(crop, cmap_name, vmin, vmax, (tw, th))

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
        scale_cfg = self.controls.scale_config()
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
            from sbp_studio.core import load_profile as _lp
            from sbp_studio.core.tasks import Cancelled
            from sbp_studio.viz.render import save_figure

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
                        render_obj, cfg, params, node_cfg, scale_cfg,
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
            from sbp_studio.core import (
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
        # User-resizable: only a MINIMUM width is set (was a hard setFixedWidth that
        # pinned the column and blocked the splitter handle). The user can now drag
        # the splitter edge to widen/narrow the filter panel freely; the scroll
        # bar remains only as a fallback for very short windows.
        scroll.setMinimumWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
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
