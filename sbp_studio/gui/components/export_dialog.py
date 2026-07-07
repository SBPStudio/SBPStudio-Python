"""
export_dialog.py — Compact options dialog for the high-quality (matplotlib) export.

The interactive views use PyQtGraph, but image/PDF export goes through the core
headless matplotlib renderer (``sbp_studio.viz.render``) — the same path as the
CLI ``export-image`` — for proper axes, ticks, time labels, FIX styling and
vector PDF output.

Only format, DPI and theme are exposed here. The display SCALE (aspect ratio)
is taken from the visualizer's own scale control so the export matches what is
on screen — it is NOT duplicated in this dialog. The finer styling (ticks, time
label format, FIX styling, margins, fill, PDF page) is baked to sensible
defaults in :meth:`config`. The DSP/processing comes from the main controls
panel — the export reflects how the section is processed (WYSIWYG).
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

# DPI choices offered in the dropdown (editable — any value can be typed).
DPI_CHOICES = ["96", "150", "200", "300", "450", "600", "900", "1200", "1800", "2400"]
DEFAULT_DPI = "1200"

# Paper sizes imported here so the export dialog can show the budget estimate
# using the selected paper dimensions instead of the view-scale figsize.
# Defined in _render.py (the shared export-computation module) to avoid duplication.
try:
    from ..tabs._render import PAPER_SIZES
except ImportError:
    PAPER_SIZES = {"A4": (11.69, 8.27), "A3": (16.54, 11.69), "A0": (46.81, 33.11)}


class ExportDialog(QDialog):
    """Collects the essential presentation options for a high-quality export."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                 source: object = None, scale_cfg: Optional[dict] = None,
                 velocity: float = 1500.0, batch: bool = False,
                 view_scale: Optional[tuple] = None) -> None:
        super().__init__(parent)
        # Optional live-estimate context: the active profile/chain + the selected
        # scale mode let the dialog show the real output size and the forced DPI
        # adjustment when a setting would exceed the memory budget (Part 1).
        self._source = source
        self._scale_cfg = scale_cfg
        self._velocity = float(velocity)
        # The live viewport's physical scale (in_per_km, in_per_ms) — the 'Auto'
        # page is the FULL data extent at this scale (the viewport authors the
        # scale, never the extent — see export_headless._page_from_scale), so
        # the estimate labels reflect the real dynamic page. None (no live
        # view) → the estimate falls back to figsize_for_scale.
        self._view_scale = view_scale
        # ``batch`` mode adds the optional "custom output directory" row (only
        # meaningful for a multi-item batch — a single export picks its path in
        # the following Save dialog instead). Kept off for single export so that
        # dialog stays unchanged.
        self._batch = bool(batch)
        self.setMinimumWidth(360)
        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self.cb_format = QComboBox()
        for label, ext in (("PDF (vector)", "pdf"), ("PNG", "png"),
                            ("TIFF", "tiff"), ("SVG (vector)", "svg")):
            self.cb_format.addItem(label, ext)
        # Paper size: 'Auto' (dynamic page — matches the on-screen view for a
        # single export, the scale formula for a batch) is FIRST and the
        # DEFAULT. Fixed sheets (A4/A3/A0) remain as explicit choices. 'Auto'
        # was previously missing entirely and the default was A4 — the 'A4
        # trap': every export was squeezed onto a fixed sheet and the scale
        # mode / zoom never reached the page geometry.
        self.cb_papersize = QComboBox()
        self.cb_papersize.addItem("Auto")
        for key in PAPER_SIZES:
            self.cb_papersize.addItem(key)
        self.cb_papersize.setCurrentText("Auto")
        self.cb_papersize.currentTextChanged.connect(lambda *_: self._update_budget_status())
        self.cb_dpi = QComboBox()
        self.cb_dpi.setEditable(True)
        self.cb_dpi.addItems(DPI_CHOICES)
        self.cb_dpi.setCurrentText(DEFAULT_DPI)
        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["dark", "light", "print"])
        self.cb_theme.setCurrentText("print")

        # ── Geophysical printout: grid, axis fonts, margins ──────────────────
        # X / Y grid-cell spacing (0 = that grid off). The grid itself is gated by
        # the checkbox; spacing 0 falls back to matplotlib's automatic ticks.
        self.sp_xtick = QDoubleSpinBox()
        self.sp_xtick.setRange(0.0, 1000.0); self.sp_xtick.setDecimals(2)
        self.sp_xtick.setSingleStep(0.5); self.sp_xtick.setValue(0.0)
        self.sp_xtick.setSpecialValueText(self.tr("auto"))
        self.sp_ttick = QDoubleSpinBox()
        self.sp_ttick.setRange(0.0, 10000.0); self.sp_ttick.setDecimals(0)
        self.sp_ttick.setSingleStep(10.0); self.sp_ttick.setValue(0.0)
        self.sp_ttick.setSpecialValueText(self.tr("auto"))
        self.cb_grid = QCheckBox()
        self.cb_grid.setChecked(False)
        self.sp_axisfont = QDoubleSpinBox()
        self.sp_axisfont.setRange(3.0, 36.0); self.sp_axisfont.setDecimals(1)
        self.sp_axisfont.setSingleStep(0.5); self.sp_axisfont.setValue(7.0)
        self.sp_timefont = QDoubleSpinBox()
        self.sp_timefont.setRange(3.0, 36.0); self.sp_timefont.setDecimals(1)
        self.sp_timefont.setSingleStep(0.5); self.sp_timefont.setValue(6.0)
        self.sp_mtop = QDoubleSpinBox()
        self.sp_mtop.setRange(0.0, 2000.0); self.sp_mtop.setDecimals(0)
        self.sp_mtop.setSingleStep(5.0); self.sp_mtop.setValue(20.0)
        self.sp_mbot = QDoubleSpinBox()
        self.sp_mbot.setRange(0.0, 2000.0); self.sp_mbot.setDecimals(0)
        self.sp_mbot.setSingleStep(5.0); self.sp_mbot.setValue(20.0)
        # Peak RAM the export raster may use (validated against free RAM before
        # rendering — see SubTabbedTab._on_export_image_requested).
        self.sp_membudget = QDoubleSpinBox()
        self.sp_membudget.setRange(0.5, 256.0); self.sp_membudget.setDecimals(1)
        self.sp_membudget.setSingleStep(1.0); self.sp_membudget.setValue(6.0)

        self._rows = [
            ("Format", self.cb_format),
            ("Tamaño de Papel", self.cb_papersize),
            ("Resolution (DPI)", self.cb_dpi),
            ("Theme", self.cb_theme),
            ("X grid spacing (km)", self.sp_xtick),
            ("Y grid spacing (ms)", self.sp_ttick),
            ("Axis font size (pt)", self.sp_axisfont),
            ("Time-label font size (pt)", self.sp_timefont),
            ("Top margin (ms)", self.sp_mtop),
            ("Bottom margin (ms)", self.sp_mbot),
            ("Memory budget (GB)", self.sp_membudget),
        ]
        self._label_widgets: list[QLabel] = []
        for _text, widget in self._rows:
            label = QLabel()
            self._label_widgets.append(label)
            form.addRow(label, widget)

        # File-boundary (red seam) lines in the export. Independent of the
        # interactive viewer's own "Show file boundaries" toggle. Unchecked
        # by default — boundaries are only drawn in the export if requested.
        self.cb_boundaries = QCheckBox()
        self.cb_boundaries.setChecked(False)
        root.addWidget(self.cb_boundaries)

        # Draw the grid (cells) using the X/Y spacings above. Off → no grid.
        root.addWidget(self.cb_grid)

        # Reflector-safe downscale: when the export must shrink below native (a
        # RAM-capped DPI), pool by max-|amplitude| so thin high-amplitude
        # reflectors survive instead of being averaged out by bilinear. Default on.
        self.cb_maxabs = QCheckBox()
        self.cb_maxabs.setChecked(True)
        root.addWidget(self.cb_maxabs)

        # Burn the active interpretation picks (markers) into the export raster
        # at full export resolution — see viz.render._draw_picks. Off by default:
        # most exports are a clean section, not an annotated interpretation map.
        self.cb_picks = QCheckBox()
        self.cb_picks.setChecked(False)
        root.addWidget(self.cb_picks)

        # ── Optional custom output directory (batch only) ────────────────────
        # When enabled, EVERY file in the batch is written into this one folder
        # instead of each item's own source directory. The per-line naming
        # convention (file named after the folder containing that line's SEG-Y
        # files) is preserved by the batch writer — see _batch_output_path. When
        # the checkbox is off or the field is blank, config()['out_dir'] is ""
        # and the batch falls back to the original per-source-folder behaviour.
        if self._batch:
            self.chk_out_dir = QCheckBox()
            self.chk_out_dir.setChecked(False)
            root.addWidget(self.chk_out_dir)
            out_row = QHBoxLayout()
            self.ed_out_dir = QLineEdit()
            self.btn_out_dir = QPushButton()
            self.btn_out_dir.setMaximumWidth(110)
            self.btn_out_dir.clicked.connect(self._browse_out_dir)
            out_row.addWidget(self.ed_out_dir, 1)
            out_row.addWidget(self.btn_out_dir, 0)
            root.addLayout(out_row)
            # The field + Browse button only accept input once the checkbox is
            # ticked. The handler reads isChecked() as the single source of
            # truth and ignores the signal's own argument, so it behaves
            # identically no matter what emits it (toggled's bool, stateChanged's
            # int, or the direct call below for the initial state) — belt-and-
            # suspenders against any signal-signature edge case.
            self.chk_out_dir.toggled.connect(self._sync_out_dir_enabled)
            self._sync_out_dir_enabled()

        # ── Dynamic info panel (Part 1): native DPI, quality %, RAM estimate ──
        # Recomputed live as DPI / memory budget change so the user sees exactly
        # what the engine will do BEFORE exporting.
        self.lbl_quality = QLabel()
        self.lbl_quality.setWordWrap(True)
        root.addWidget(self.lbl_quality)

        # ── Live budget / output-size status (Part 1) ────────────────────────
        # Shows the estimated raster size for the current DPI + scale, and — when
        # the chosen DPI would exceed the memory budget — the forced-safe DPI the
        # export will actually use. Updates as DPI / memory budget change.
        self.lbl_budget = QLabel()
        self.lbl_budget.setWordWrap(True)
        root.addWidget(self.lbl_budget)
        self.cb_dpi.currentTextChanged.connect(lambda *_: self._update_budget_status())
        self.sp_membudget.valueChanged.connect(lambda *_: self._update_budget_status())
        # On finishing DPI entry, hard-clamp it to the budget-safe maximum so the
        # conflicting setting is locked to a value that cannot hang the export.
        self.cb_dpi.lineEdit().editingFinished.connect(self._clamp_dpi_to_budget)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self._update_budget_status()

        self.retranslate_ui()

    # ── Result ──────────────────────────────────────────────────────────────

    def _sync_out_dir_enabled(self, *_args) -> None:
        """Enable the custom-output path field + Browse button only while the
        'custom folder' checkbox is ticked. ``*_args`` absorbs whatever the
        signal passes (a bool from ``toggled``, an int from ``stateChanged``,
        or nothing from the direct init call) — the checkbox's own
        ``isChecked()`` is the single source of truth."""
        on = self.chk_out_dir.isChecked()
        self.ed_out_dir.setEnabled(on)
        self.btn_out_dir.setEnabled(on)

    def _browse_out_dir(self) -> None:
        start = self.ed_out_dir.text().strip()
        d = QFileDialog.getExistingDirectory(
            self, self.tr("Select output directory for the batch"), start)
        if d:
            self.ed_out_dir.setText(d)

    def _out_dir(self) -> str:
        """The chosen batch output directory, or "" (fall back to each item's
        own source folder). Only ever non-empty in batch mode with the box
        ticked AND a non-blank path."""
        if not self._batch or not self.chk_out_dir.isChecked():
            return ""
        return self.ed_out_dir.text().strip()

    def _dpi(self) -> int:
        try:
            return max(50, min(4800, int(float(self.cb_dpi.currentText()))))
        except ValueError:
            return int(DEFAULT_DPI)

    # ── Live budget validation (Part 1) ──────────────────────────────────────

    def _estimate(self) -> Optional[tuple]:
        """Return (figsize, chosen_dpi, eff_dpi, safe_dpi, mpx, native_dpi, ram_gb)
        for the current settings, or None when there is no source to estimate
        against. ``eff_dpi`` is what the export will REALLY use (native-need raise,
        capped by budget); ``safe_dpi`` is the budget ceiling; ``native_dpi`` is
        the DPI for a 1:1 raster (1 px per trace horizontally / sample vertically);
        ``ram_gb`` is the RGBA matrix footprint (w·h·4 bytes) at ``eff_dpi``."""
        if self._source is None or self._scale_cfg is None:
            return None
        try:
            import math
            from ..tabs._render import (figsize_for_scale, effective_export_dpi,
                                        dpi_for_budget)
            chosen = self._dpi()
            paper_key = self.cb_papersize.currentText()
            figsize = None
            if paper_key in PAPER_SIZES:
                figsize = PAPER_SIZES[paper_key]
            elif self._view_scale is not None and self._view_scale[0] > 0 \
                    and self._view_scale[1] > 0:
                # 'Auto' with a live view → the FULL data extent at the
                # on-screen scale (same rule as _page_from_scale: width from
                # the X scale × line length, height from the Y scale × record
                # + margins — the axes are independent).
                ipk, ipm = float(self._view_scale[0]), float(self._view_scale[1])
                total_km = float(getattr(self._source, "total_km", 0.0) or 0.0)
                ns_ = int(getattr(self._source, "ns", 0) or 0)
                dt_us = float(getattr(self._source, "dt_us", 0) or 0)
                rec_ms = ns_ * dt_us / 1000.0 + float(self.sp_mtop.value()) \
                    + float(self.sp_mbot.value())
                if total_km > 0 and rec_ms > 0:
                    figsize = (max(0.5, total_km * ipk), max(0.5, rec_ms * ipm))
            if figsize is None:
                figsize = figsize_for_scale(self._source, self._scale_cfg, chosen,
                                            self._velocity)
            ns = int(getattr(self._source, "ns", 0))
            nt = int(getattr(self._source, "n_traces", 0))
            eff = effective_export_dpi(figsize, (ns, nt), chosen)
            safe = dpi_for_budget(figsize, float(self.sp_membudget.value()))
            if safe is not None and eff > safe:
                eff = max(50, safe)
            mpx = (figsize[0] * eff) * (figsize[1] * eff) / 1e6
            # Native (1:1) DPI: 1 output px per trace (x) / sample (y).
            w_in, h_in = float(figsize[0]), float(figsize[1])
            native = (int(math.ceil(max(nt / w_in, ns / h_in)))
                      if w_in > 0 and h_in > 0 else eff)
            # RGBA matrix footprint at the effective DPI (w·h·4 bytes).
            ram_gb = (w_in * eff) * (h_in * eff) * 4 / 1024 ** 3
            return figsize, chosen, eff, safe, mpx, native, ram_gb
        except Exception:
            return None

    def _update_budget_status(self) -> None:
        est = self._estimate()
        if est is None:
            self.lbl_budget.setText("")
            self.lbl_quality.setText("")
            return
        _figsize, chosen, eff, safe, mpx, native, ram_gb = est

        # ── Info panel: native DPI · quality · RAM ──
        native = max(1, int(native))
        quality = min(100.0, 100.0 * eff / native)
        ram_txt = (self.tr("{0:.2f} GB").format(ram_gb) if ram_gb >= 1.0
                   else self.tr("{0:.0f} MB").format(ram_gb * 1024))
        if quality >= 99.5:
            self.lbl_quality.setStyleSheet("color:#3a8a3a;")
            q_line = self.tr("Quality: {0:.0f}% (native resolution)").format(quality)
        else:
            self.lbl_quality.setStyleSheet("color:#cc7000;")
            q_line = self.tr("Quality: {0:.0f}% (downsampled)").format(quality)
        self.lbl_quality.setText(
            self.tr("Native DPI: {0} (1 px / trace · sample)").format(native)
            + "\n" + q_line
            + "\n" + self.tr("Estimated RAM: {0}").format(ram_txt))

        # ── Budget line (unchanged behaviour) ──
        if safe is not None and eff < chosen:
            self.lbl_budget.setStyleSheet("color:#cc7000;")
            self.lbl_budget.setText(self.tr(
                "⚠ {0} DPI would exceed the {1:.0f} GB budget — the export will be "
                "limited to {2} DPI ({3:.0f} Mpx) to avoid running out of memory.")
                .format(chosen, self.sp_membudget.value(), eff, mpx))
        else:
            self.lbl_budget.setStyleSheet("color:#3a8a3a;")
            self.lbl_budget.setText(self.tr(
                "Output ≈ {0:.0f} Mpx at {1} DPI — within the {2:.0f} GB budget.")
                .format(mpx, eff, self.sp_membudget.value()))

    def _clamp_dpi_to_budget(self) -> None:
        """Lock the DPI field to the budget-safe maximum if the user typed a value
        that would be forced down anyway — so the conflicting setting can't stay on
        screen as if it were honoured."""
        est = self._estimate()
        if est is None:
            return
        _figsize, chosen, eff, safe, *_ = est
        if safe is not None and chosen > safe:
            self.cb_dpi.setCurrentText(str(int(eff)))
            self._update_budget_status()

    def config(self) -> dict:
        """Exposed options + baked defaults (the latter mirror a good print recipe).

        Scale (aspect ratio) is intentionally absent — the export reads it from
        the visualizer's scale control so the output matches the on-screen view.
        """
        # 0 in the spacing spin-boxes means "auto / off" → None to the renderer.
        x_tick = self.sp_xtick.value() or None
        t_tick = self.sp_ttick.value() or None
        return dict(
            # ── Exposed ──
            format=self.cb_format.currentData(),
            paper_size=self.cb_papersize.currentText(),
            dpi=self._dpi(),
            theme=self.cb_theme.currentText(),
            draw_file_boundaries=self.cb_boundaries.isChecked(),
            grid=self.cb_grid.isChecked(),
            x_tick=x_tick,
            t_tick=t_tick,
            axis_font_size=float(self.sp_axisfont.value()),
            time_font_size=float(self.sp_timefont.value()),
            margin_top=float(self.sp_mtop.value()),
            margin_bottom=float(self.sp_mbot.value()),
            mem_budget_gb=float(self.sp_membudget.value()),
            max_abs_pool=self.cb_maxabs.isChecked(),
            overlay_picks=self.cb_picks.isChecked(),
            # Batch-only: single target folder for every exported file ("" =
            # each item goes to its own source folder, the historical default).
            # Ignored by the single export path (which picks its own Save path).
            out_dir=self._out_dir(),
            # ── Baked defaults ──
            velocity=1500.0,
            pdf_page="auto",
            # Top UTC time axis (the "#nn …Z" markers + their top ticks) is
            # intentionally OFF for the GUI export → a perfectly clean top axis.
            # None gates _add_time_axis in the core renderer (see render.py). The
            # CLI keeps its own opt-in --time-ticks feature (zero regression).
            time_ticks=None,
            time_fmt="full",
            time_align="left",
            fix_color=None,        # use theme FIX colour
            fix_bbox_alpha=0.0,
            grid_alpha=0.18,
            grid_lw=0.5,
            fill_zero=True,
        )

    # ── i18n ────────────────────────────────────────────────────────────────

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Export options"))
        labels = [self.tr("Format"), self.tr("Tamaño de Papel"),
                  self.tr("Resolution (DPI)"), self.tr("Theme"),
                  self.tr("X grid spacing (km)"), self.tr("Y grid spacing (ms)"),
                  self.tr("Axis font size (pt)"), self.tr("Time-label font size (pt)"),
                  self.tr("Top margin (ms)"), self.tr("Bottom margin (ms)"),
                  self.tr("Memory budget (GB)")]
        for label_widget, text in zip(self._label_widgets, labels):
            label_widget.setText(text)
        self.cb_boundaries.setText(self.tr("Include red file boundary lines in export"))
        self.cb_grid.setText(self.tr("Draw grid (uses the X/Y spacings above)"))
        self.cb_maxabs.setText(self.tr("Use Max-Abs pooling when downsampling "
                                       "(preserves amplitude peaks)"))
        self.cb_maxabs.setToolTip(self.tr(
            "When the export must shrink below native resolution, keep the "
            "strongest sample per block instead of averaging — preserves thin, "
            "high-amplitude reflectors."))
        self.cb_picks.setText(self.tr("Overlay interpretation markers"))
        self.cb_picks.setToolTip(self.tr(
            "Burn the active interpretation picks (markers + ID labels) into "
            "the exported image at full resolution, exactly where they sit on "
            "the live section."))
        if self._batch:
            self.chk_out_dir.setText(self.tr("Save all files to a custom folder"))
            self.chk_out_dir.setToolTip(self.tr(
                "When enabled, every file in the batch is written to this one "
                "folder. Each file keeps its automatic name (the name of the "
                "folder containing that line's SEG-Y files). When disabled, "
                "each file goes to its own source folder."))
            self.ed_out_dir.setPlaceholderText(self.tr("Custom output folder for the batch…"))
            self.btn_out_dir.setText(self.tr("Browse…"))
        # Standard buttons render with no visible text under the dark QSS — set
        # explicit, translated text so they're always readable.
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("Accept"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("Cancel"))
