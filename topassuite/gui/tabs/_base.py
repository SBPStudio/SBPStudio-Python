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

from ..components import ExportDialog, PlaceholderView, ProcessingControls
from ..i18n import language_manager
from ..state import AppState
from ._render import compute_figsize

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


class SubTabbedTab(QWidget):
    """Controls column + four-view sub-notebook with real Render dispatch."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.tasks = tasks  # task service (MainWindow.run_task / notify / show_error)
        self._seismic: Optional[QWidget] = None  # SeismicView, created on first render

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_subtabs())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([252, 1000])
        root.addWidget(split)

        self.controls.render_requested.connect(self._on_render_requested)
        self.controls.export_image_requested.connect(self._on_export_image_requested)
        self.controls.export_fix_requested.connect(self._on_export_fix_requested)
        self.controls.scale_changed.connect(self._on_scale_changed)
        self.controls.boundaries_toggled.connect(self._on_boundaries_toggled)
        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    # ── Overridable hooks ───────────────────────────────────────────────────

    def empty_message(self) -> str:
        """Translated message shown when nothing is selected."""
        return ""

    def _on_render_requested(self) -> None:
        """Subclasses dispatch a CoreWorker render here."""

    def _active_object(self):
        """The SegyProfile / ProfileChain currently driving this tab (or None)."""
        return None

    def _is_chain(self) -> bool:
        """True if the active object is a ProfileChain (Chains tab)."""
        return False

    def _on_scale_changed(self) -> None:
        """Apply the new display aspect to the on-screen section live."""
        if self._seismic is not None and self._seismic.has_image():
            self._seismic.set_aspect(self.controls.aspect())

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
        dpi = int(cfg["dpi"])
        aspect = self.controls.aspect()  # scale == the visualizer's aspect ratio
        params = dict(self.controls.params())
        params["clip_lo"] = 0.0
        params["fill_value"] = 0.0 if cfg["fill_zero"] else float("nan")
        # File-boundary lines in the export come from the EXPORT DIALOG's own
        # checkbox — independent of the interactive viewer's toggle. The core
        # renderer reads params["draw_file_boundaries"] to gate ax.axvline.
        params["draw_file_boundaries"] = bool(cfg["draw_file_boundaries"])

        def job(progress, cancel) -> str:
            from topassuite.core import process_chain_data, process_profile_data
            from topassuite.viz.render import (
                build_theme, render_chain_figure, render_profile_figure, save_figure,
            )
            # Lazy-load safety: profiles are added as header-only stubs and
            # traces are loaded on selection. If the user triggers an export
            # before that background load has finished (race condition), load
            # the traces here on the worker thread so the export never fails.
            _obj = obj
            if not is_chain and getattr(_obj, "data", None) is None:
                from topassuite.core import load_profile as _lp
                progress(float("nan"), "Loading traces for export…")
                _obj = _lp(_obj.path, load_traces=True)
                if getattr(_obj, "error", None):
                    raise RuntimeError(f"Cannot load {_obj.name}: {_obj.error}")
            progress(float("nan"), "")
            data = (process_chain_data(_obj, params) if is_chain
                    else process_profile_data(_obj, params))
            cancel.check()
            figsize = compute_figsize(_obj, dpi, None, aspect, cfg["velocity"])
            render_opts = dict(
                x_tick_km=cfg["x_tick"], t_tick_ms=cfg["t_tick"], show_grid=cfg["grid"],
                title_override=None, clip_lo=0.0, time_tick_min=cfg["time_ticks"],
                margin_top_ms=cfg["margin_top"], margin_bottom_ms=cfg["margin_bottom"],
                time_fmt=cfg["time_fmt"], time_font_size=cfg["time_font_size"],
                time_align=cfg["time_align"], fix_font_size=5.0,
                fix_bbox_alpha=cfg["fix_bbox_alpha"], fix_color=cfg["fix_color"],
                colors=build_theme(theme=cfg["theme"]))
            render = render_chain_figure if is_chain else render_profile_figure
            fig = render(_obj, data, params, figsize=figsize, dpi=dpi, **render_opts)
            # Make the seismic DATA AREA have the SAME aspect as the on-screen
            # view (WYSIWYG scale). Matplotlib decorations (title, labels,
            # colorbar) take a roughly fixed margin in inches, so we iterate:
            # keep the data width, set figure height = data_w/aspect + margins,
            # and re-run tight_layout until the data box converges to `aspect`.
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
            save_figure(fig, out, dpi=dpi, fmt=fmt, pdf_page=cfg["pdf_page"])
            fig.clear()
            return out

        self.tasks.run_task(
            job, self._on_image_exported,
            QCoreApplication.translate("SubTabbedTab", "Exporting image…"))

    def _on_image_exported(self, out: str) -> None:
        from pathlib import Path
        self.tasks.notify(QCoreApplication.translate(
            "SubTabbedTab", "Image saved: {0}").format(Path(out).name))

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(252)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls = ProcessingControls()
        scroll.setWidget(self.controls)
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
