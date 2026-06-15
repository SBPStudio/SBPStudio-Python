"""
processing_controls.py — DSP controls panel for the seismic views.

Builds the left-hand controls column (palette, predictive deconvolution,
bandpass filter, clip/gain, TVG, AGC, delay alignment, preset filters, FIX
marks) and exposes :meth:`params` — the exact dict consumed by the core
``process_profile_data`` / ``process_chain_data`` pipeline, plus the render-only
extras (clip, cmap, inv_cmap, fix, fix_iv). It performs NO processing; pressing
Render simply emits :pyattr:`ProcessingControls.render_requested`.

Combo contents (colormap and preset names) come verbatim from the core domain
constants — they are data, not UI chrome, and the core resolves them directly,
so they are intentionally not run through translation.

Every translatable label is set with an explicit literal in
:meth:`retranslate_ui` (not via a stored variable key) so ``pylupdate6`` can
extract the English source strings.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..i18n import language_manager

# Core domain constants (labels for the combos + preset descriptions).
from sbp_studio.core.constants import (
    CMAPS, FILTER_DESCRIPTIONS, FILTER_PRESETS,
)


class LabeledSlider(QWidget):
    """Horizontal slider with arbitrary float resolution and a value label."""

    valueChanged = pyqtSignal(float)

    def __init__(self, lo: float, hi: float, res: float = 1.0,
                 init: Optional[float] = None, fmt: str = "{:.0f}",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._lo = float(lo)
        self._res = float(res)
        self._fmt = fmt
        steps = max(1, int(round((hi - lo) / res)))

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, steps)
        self._lbl = QLabel()
        self._lbl.setObjectName("sub")
        self._lbl.setFixedWidth(54)   # fits 2-decimal values like "100.00"
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if init is None:
            init = lo
        self._slider.setValue(int(round((float(init) - self._lo) / self._res)))
        self._slider.valueChanged.connect(self._update_label)
        self._update_label()

        lay.addWidget(self._slider, 1)
        lay.addWidget(self._lbl, 0)

    def _update_label(self, *_) -> None:
        self._lbl.setText(self._fmt.format(self.value()))
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        return self._lo + self._slider.value() * self._res


class ProcessingControls(QWidget):
    """Scrollable DSP controls; emits :pyattr:`render_requested` on Render."""

    render_requested = pyqtSignal()
    render_viewport_requested = pyqtSignal()   # HQ export of the visible ViewBox crop
    export_image_requested = pyqtSignal()
    export_fix_requested = pyqtSignal()
    scale_changed = pyqtSignal()  # aspect ratio changed (live)
    boundaries_toggled = pyqtSignal(bool)  # show/hide file-seam lines (live)
    align_toggled = pyqtSignal(bool)  # delay-alignment geometry toggled (rebuild base)
    display_changed = pyqtSignal()  # cmap / clip / FIX changed → recolour preview

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        # Compact, fully-adjusted layout from startup: tight outer margins and a
        # small inter-row spacing so every control is visible without scrolling.
        v.setContentsMargins(6, 6, 6, 4)
        v.setSpacing(2)
        self._v = v

        # ── Palette ──
        self.sec_palette = self._section()
        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(list(CMAPS.keys()))
        v.addWidget(self.cmap_cb)
        self.inv_cmap = QCheckBox()
        v.addWidget(self.inv_cmap)

        # ── Predictive deconvolution ──
        self.sec_decon = self._section()
        self.decon = QCheckBox()
        v.addWidget(self.decon)
        self.cap_decon_op = self._caption()
        self.decon_op = LabeledSlider(1.0, 50.0, 1.0, 10.0, "{:.0f}")
        v.addWidget(self.decon_op)
        self.cap_decon_gap = self._caption()
        self.decon_gap = LabeledSlider(0.1, 20.0, 0.1, 2.0, "{:.1f}")
        v.addWidget(self.decon_gap)
        self.cap_decon_wn = self._caption()
        self.decon_wn = LabeledSlider(0.1, 10.0, 0.1, 1.0, "{:.1f}")
        v.addWidget(self.decon_wn)

        # ── Bandpass filter ──
        self.sec_filter = self._section()
        self.filt = QCheckBox()
        v.addWidget(self.filt)
        self.cap_flo = self._caption()
        self.flo = LabeledSlider(500, 15000, 100, 2000, "{:.0f}")
        v.addWidget(self.flo)
        self.cap_fhi = self._caption()
        self.fhi = LabeledSlider(500, 15000, 100, 7000, "{:.0f}")
        v.addWidget(self.fhi)

        # ── Clip / gain ──
        self.sec_clip = self._section()
        self.cap_clip = self._caption()
        # Fine-grained amplitude clip: 0.05-% resolution, 2 decimals, so the user
        # can gently dial back "too strong" sections (e.g. 99.50 → 99.05) instead
        # of jumping in whole percent. The fractional percentile flows untouched
        # to np.percentile (no int rounding anywhere downstream).
        self.clip = LabeledSlider(80.0, 100.0, 0.05, 99.6, "{:.2f}")
        v.addWidget(self.clip)
        self.tvg = QCheckBox()
        v.addWidget(self.tvg)
        self.cap_tvg_alpha = self._caption()
        self.tvg_alpha = LabeledSlider(0.0, 150.0, 1.0, 15.0, "{:.0f}")
        v.addWidget(self.tvg_alpha)
        self.agc = QCheckBox()
        v.addWidget(self.agc)
        self.cap_agc_win = self._caption()
        self.agc_win = LabeledSlider(5, 100, 5, 20, "{:.0f}")
        v.addWidget(self.agc_win)

        # ── Preset filters ──
        self.sec_preset = self._section()
        self.preset_cb = QComboBox()
        self.preset_cb.addItems(list(FILTER_PRESETS.keys()))
        self.preset_cb.currentTextChanged.connect(self._update_preset_desc)
        v.addWidget(self.preset_cb)
        self.lbl_preset_desc = QLabel()
        self.lbl_preset_desc.setObjectName("sub")
        self.lbl_preset_desc.setWordWrap(True)
        v.addWidget(self.lbl_preset_desc)

        # ── FIX marks ──
        self.sec_fix = self._section()
        self.fix = QCheckBox()
        v.addWidget(self.fix)
        fix_row = QHBoxLayout()
        self.cap_fix_interval = QLabel()
        self.cap_fix_interval.setObjectName("sub")
        fix_row.addWidget(self.cap_fix_interval)
        self.fix_interval = QSpinBox()
        self.fix_interval.setRange(1, 120)
        self.fix_interval.setValue(15)
        fix_row.addWidget(self.fix_interval)
        fix_row.addStretch(1)
        v.addLayout(fix_row)

        # ── Geometry & Presentation (STATIC controls — NOT pipeline nodes) ──
        # Delay alignment is a static geometry correction (applied to the base
        # array before the dynamic DSP nodes), not a movable filter. File-seam
        # boundaries are a presentation overlay toggled live in the view.
        self.sec_geometry = self._section()
        self.align_delays = QCheckBox()
        self.align_delays.setChecked(True)
        self.align_delays.toggled.connect(self.align_toggled.emit)
        v.addWidget(self.align_delays)
        self.show_boundaries = QCheckBox()
        self.show_boundaries.setChecked(True)
        self.show_boundaries.toggled.connect(self.boundaries_toggled.emit)
        v.addWidget(self.show_boundaries)

        # ── Scale / proportions — THREE mutually-exclusive export modes ──────
        # The live PyQtGraph preview AND the matplotlib export both follow the
        # selected mode (see _base.SubTabbedTab._on_scale_changed / export). Tight,
        # compact rows: the radio sits right next to its inline value editor.
        self.sec_scale = self._section()
        self.scale_group = QButtonGroup(self)

        def _tight_row(*widgets) -> QHBoxLayout:
            """A compact row: zero margins, minimal spacing, radio hugging its
            inline editor (Part 2 — buttons fit close to their labels)."""
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            for w, stretch in widgets:
                row.addWidget(w, stretch)
            v.addLayout(row)
            return row

        # Mode 1 — Lock aspect ratio (constant W:H shape; VE floats with length).
        # Its value is the 'horizontal deformation' set by the slider + spin DIRECTLY
        # underneath it.
        self.rb_aspect = QRadioButton()
        self.rb_aspect.setChecked(True)
        self.scale_group.addButton(self.rb_aspect, 0)
        _tight_row((self.rb_aspect, 1))

        # ── Horizontal deformation (aspect ratio W:H) — sensitive slider + spin ──
        # Placed DIRECTLY under the Aspect-ratio button (Part 1). The slider is kept
        # SHORT (max width) on the left and a stretch pushes the QDoubleSpinBox to
        # the right so it lines up under the aspect button's right edge. Synced
        # bidirectionally; this IS the aspect-mode ratio (0.5–20.0 in 0.01 steps).
        self._DEF_SCALE = 100.0          # slider int ↔ ratio float (×0.01)
        self.sld_deform = QSlider(Qt.Orientation.Horizontal)
        self.sld_deform.setRange(int(0.5 * self._DEF_SCALE), int(20.0 * self._DEF_SCALE))
        self.sld_deform.setSingleStep(1)            # 0.01 — highly sensitive
        self.sld_deform.setPageStep(10)
        self.sld_deform.setMaximumWidth(120)        # shorter slider (Part 1)
        self.sp_ratio = QDoubleSpinBox()
        self.sp_ratio.setRange(0.5, 100.0)
        self.sp_ratio.setDecimals(2)
        self.sp_ratio.setSingleStep(0.1)
        self.sp_ratio.setValue(3.0)
        self.sp_ratio.setMaximumWidth(72)
        self.sld_deform.setValue(int(3.0 * self._DEF_SCALE))
        _deform_row = QHBoxLayout()
        _deform_row.setContentsMargins(16, 0, 0, 0)   # indent under the aspect label
        _deform_row.setSpacing(4)
        _deform_row.addWidget(self.sld_deform, 0)
        _deform_row.addStretch(1)                     # push the spin to the right edge
        _deform_row.addWidget(self.sp_ratio, 0)
        v.addLayout(_deform_row)
        self._syncing_deform = False
        self.sld_deform.valueChanged.connect(self._on_deform_slider)
        self.sp_ratio.valueChanged.connect(self._on_deform_spin)

        # Mode 2 — Lock vertical exaggeration (constant, geologically comparable).
        self.rb_ve = QRadioButton()
        self.scale_group.addButton(self.rb_ve, 1)
        self.sp_ve = QDoubleSpinBox()
        self.sp_ve.setRange(1.0, 2000.0)
        self.sp_ve.setSingleStep(1.0)
        self.sp_ve.setValue(67.0)
        self.sp_ve.setMaximumWidth(72)
        _tight_row((self.rb_ve, 1), (self.sp_ve, 0))

        # Mode 3 — Hybrid: lock VE (the value above) but cap the aspect so an
        # extremely long line never deforms into an 'infinite noodle'.
        self.rb_hybrid = QRadioButton()
        self.scale_group.addButton(self.rb_hybrid, 2)
        self.sp_maxasp = QDoubleSpinBox()
        self.sp_maxasp.setRange(1.0, 100.0)
        self.sp_maxasp.setSingleStep(0.5)
        self.sp_maxasp.setValue(5.0)
        self.sp_maxasp.setMaximumWidth(72)
        _tight_row((self.rb_hybrid, 1), (self.sp_maxasp, 0))
        self.cap_scale_hint = self._caption()

        # ── Resolution: pixels per trace ─────────────────────────────────────
        # Drives the EXPORT horizontal density (aspect-mode figure width, == the
        # CLI --px-per-trace) AND the live VIEWER detail (raises the preview column
        # cap so zooming in stays sharp). Higher = sharper / heavier.
        self.cap_pxtrace = QLabel()
        self.cap_pxtrace.setObjectName("sub")
        self.sp_pxtrace = QDoubleSpinBox()
        self.sp_pxtrace.setRange(1.0, 200.0)
        self.sp_pxtrace.setSingleStep(1.0)
        self.sp_pxtrace.setDecimals(0)
        self.sp_pxtrace.setValue(20.0)
        self.sp_pxtrace.setMaximumWidth(72)
        _tight_row((self.cap_pxtrace, 1), (self.sp_pxtrace, 0))

        # ── Dynamic export-DPI readout (Part 2) ──────────────────────────────
        # Shows the resolution the export will actually generate for the CURRENT
        # configuration (scale mode + px/trace + figure size), computed live by the
        # tab (SubTabbedTab._update_dpi_estimate) so the user knows the result
        # before rendering. Set to a placeholder until a profile is active.
        self.lbl_dpi_estimate = self._caption()
        self.lbl_dpi_estimate.setWordWrap(True)

        # ── Reset (Part 3 of the previous task): restore aspect defaults ───────
        self.btn_scale_reset = QPushButton()
        self.btn_scale_reset.clicked.connect(self.reset_scale_defaults)
        _tight_row((self.btn_scale_reset, 1))

        # Any mode switch or value edit → live preview re-aspect (debounced by the
        # cheap downstream work) and is read at export time.
        for rb in (self.rb_aspect, self.rb_ve, self.rb_hybrid):
            rb.toggled.connect(lambda *_: self.scale_changed.emit())
        for sp in (self.sp_ratio, self.sp_ve, self.sp_maxasp):
            sp.valueChanged.connect(lambda *_: self.scale_changed.emit())
        # px/trace changes both the export width AND the live viewer detail.
        self.sp_pxtrace.valueChanged.connect(lambda *_: self.scale_changed.emit())
        self.sp_pxtrace.valueChanged.connect(lambda *_: self.display_changed.emit())

        # ── Action — two side-by-side render buttons (Part 3) ──
        line = QFrame()
        line.setObjectName("hline")
        v.addWidget(line)
        render_row = QHBoxLayout()
        render_row.setContentsMargins(0, 0, 0, 0)
        render_row.setSpacing(4)
        # Button 1 — Render Full: re-render the WHOLE line (current behaviour).
        self.btn_render = QPushButton()
        self.btn_render.clicked.connect(self.render_requested.emit)
        render_row.addWidget(self.btn_render, 1)
        # Button 2 — Render Viewport HQ: export ONLY the zoomed-in ViewBox crop at
        # a forced high-DPI preset (handled in _base via the SeismicView range).
        self.btn_render_viewport = QPushButton()
        self.btn_render_viewport.clicked.connect(self.render_viewport_requested.emit)
        render_row.addWidget(self.btn_render_viewport, 1)
        v.addLayout(render_row)

        # ── Export ── (high-quality matplotlib export via a dedicated dialog)
        self.sec_export = self._section()
        self.btn_export_img = QPushButton()
        self.btn_export_img.clicked.connect(self.export_image_requested.emit)
        v.addWidget(self.btn_export_img)
        self.btn_export_fix = QPushButton()
        self.btn_export_fix.clicked.connect(self.export_fix_requested.emit)
        v.addWidget(self.btn_export_fix)

        v.addStretch(1)

        # Presentation changes drive a live recolour of the preview (debounced
        # downstream by the controller's own work being cheap).
        self.cmap_cb.currentTextChanged.connect(lambda *_: self.display_changed.emit())
        self.inv_cmap.toggled.connect(lambda *_: self.display_changed.emit())
        self.clip.valueChanged.connect(lambda *_: self.display_changed.emit())
        self.fix.toggled.connect(lambda *_: self.display_changed.emit())
        self.fix_interval.valueChanged.connect(lambda *_: self.display_changed.emit())

        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    # ── Builders ─────────────────────────────────────────────────────────────

    def _section(self) -> QLabel:
        line = QFrame()
        line.setObjectName("hline")
        self._v.addWidget(line)
        lbl = QLabel()
        lbl.setObjectName("section")
        self._v.addWidget(lbl)
        return lbl

    def _caption(self) -> QLabel:
        lbl = QLabel()
        lbl.setObjectName("sub")
        self._v.addWidget(lbl)
        return lbl

    # ── Parameters consumed by the core pipeline + render extras ────────────

    def params(self) -> dict:
        return dict(
            decon=self.decon.isChecked(), decon_op=self.decon_op.value(),
            decon_gap=self.decon_gap.value(), decon_wn=self.decon_wn.value(),
            filt=self.filt.isChecked(),
            flo=int(self.flo.value()), fhi=int(self.fhi.value()),
            preset=self.preset_cb.currentText(),
            tvg=self.tvg.isChecked(), tvg_alpha=self.tvg_alpha.value(),
            agc=self.agc.isChecked(), agc_win=int(self.agc_win.value()),
            align=self.align_delays.isChecked(),
            clip=float(self.clip.value()), cmap=self.cmap_cb.currentText(),
            inv_cmap=self.inv_cmap.isChecked(),
            fix=self.fix.isChecked(), fix_iv=int(self.fix_interval.value()),
        )

    def display_params(self) -> dict:
        """Presentation + static-geometry params for the live preview.

        ``align`` is the STATIC delay-alignment toggle (a geometry correction,
        not a pipeline node): the controller applies it to the base array before
        the dynamic nodes. ``boundaries`` is the file-seam overlay toggle.
        """
        return dict(
            cmap=self.cmap_cb.currentText(),
            inv_cmap=self.inv_cmap.isChecked(),
            clip=float(self.clip.value()),
            fix=self.fix.isChecked(),
            fix_iv=int(self.fix_interval.value()),
            align=self.align_delays.isChecked(),
            boundaries=self.show_boundaries.isChecked(),
            px_per_trace=float(self.sp_pxtrace.value()),
        )

    def align_enabled(self) -> bool:
        """Whether the static delay-alignment correction is on."""
        return self.align_delays.isChecked()

    def set_dsp_sections_visible(self, visible: bool) -> None:
        """Hide the static DSP FILTER sections (superseded by the node pipeline).

        The DSP chain (decon/bandpass/preset/TVG/AGC) now comes from the
        PipelinePanel, so those controls are hidden. The palette, clip, FIX,
        the static Geometry & Presentation controls (delay alignment + file
        boundaries), scale and export remain. Hidden DSP checkboxes are also
        unchecked so any residual ``params()`` read is a safe no-op. NOTE:
        ``align_delays`` is NOT hidden — it is a static geometry control now.
        """
        dsp_widgets = [
            self.sec_decon, self.decon, self.cap_decon_op, self.decon_op,
            self.cap_decon_gap, self.decon_gap, self.cap_decon_wn, self.decon_wn,
            self.sec_filter, self.filt, self.cap_flo, self.flo,
            self.cap_fhi, self.fhi,
            self.sec_preset, self.preset_cb, self.lbl_preset_desc,
            self.tvg, self.cap_tvg_alpha, self.tvg_alpha,
            self.agc, self.cap_agc_win, self.agc_win,
        ]
        for w in dsp_widgets:
            w.setVisible(visible)
        if not visible:
            for cb in (self.decon, self.filt, self.tvg, self.agc):
                cb.setChecked(False)

    # ── i18n ────────────────────────────────────────────────────────────────

    def scale_config(self) -> dict:
        """The selected scaling mode + its values, consumed by the live preview
        (``effective_aspect``) and the export (``figsize_for_scale``).

        mode ∈ {'aspect','ve','hybrid'}; ratio (mode 1), ve (modes 2/3),
        max_aspect (mode 3). Mirrors the three finalised CLI export recipes."""
        if self.rb_ve.isChecked():
            mode = "ve"
        elif self.rb_hybrid.isChecked():
            mode = "hybrid"
        else:
            mode = "aspect"
        return dict(mode=mode,
                    ratio=float(self.sp_ratio.value()),
                    ve=float(self.sp_ve.value()),
                    max_aspect=float(self.sp_maxasp.value()),
                    px_per_trace=float(self.sp_pxtrace.value()))

    def px_per_trace(self) -> float:
        """Pixels per trace — export horizontal density + live viewer detail."""
        return float(self.sp_pxtrace.value())

    def set_dpi_estimate(self, text: str) -> None:
        """Set the live 'resulting export DPI' readout (computed by the tab)."""
        self.lbl_dpi_estimate.setText(text)

    # ── Horizontal-deformation slider ↔ spin sync (Part 4) ───────────────────

    def _on_deform_slider(self, val: int) -> None:
        if self._syncing_deform:
            return
        self._syncing_deform = True
        self.sp_ratio.setValue(val / self._DEF_SCALE)
        self._syncing_deform = False

    def _on_deform_spin(self, val: float) -> None:
        if self._syncing_deform:
            return
        self._syncing_deform = True
        self.sld_deform.setValue(int(round(val * self._DEF_SCALE)))
        self._syncing_deform = False

    def reset_scale_defaults(self) -> None:
        """Part 3: restore every aspect/scale setting to its default in one click."""
        self.rb_aspect.setChecked(True)
        self.sp_ratio.setValue(3.0)        # syncs the deformation slider too
        self.sp_ve.setValue(67.0)
        self.sp_maxasp.setValue(5.0)
        self.sp_pxtrace.setValue(20.0)
        self.scale_changed.emit()

    def aspect(self) -> Optional[float]:
        """Back-compat: the fixed W:H ratio when in aspect mode, else None.

        VE / hybrid aspects depend on the active line's length, so they are
        resolved per-source by ``_render.effective_aspect`` at the call site."""
        return float(self.sp_ratio.value()) if self.rb_aspect.isChecked() else None

    def boundaries_visible(self) -> bool:
        """Whether file-seam boundary lines should be shown in the live view."""
        return self.show_boundaries.isChecked()

    def _update_preset_desc(self, *_) -> None:
        key = FILTER_PRESETS.get(self.preset_cb.currentText(), "none")
        self.lbl_preset_desc.setText(FILTER_DESCRIPTIONS.get(key, ""))

    def retranslate_ui(self) -> None:
        # Sections
        self.sec_palette.setText(self.tr("PALETTE"))
        self.sec_decon.setText(self.tr("PREDICTIVE DECONVOLUTION"))
        self.sec_filter.setText(self.tr("BANDPASS FILTER"))
        self.sec_clip.setText(self.tr("CLIP / GAIN"))
        self.sec_preset.setText(self.tr("PRESET FILTERS"))
        self.sec_fix.setText(self.tr("FIX MARKS"))
        # Captions
        self.cap_decon_op.setText(self.tr("Operator length (ms):"))
        self.cap_decon_gap.setText(self.tr("Prediction gap/lag (ms):"))
        self.cap_decon_wn.setText(self.tr("Pre-whitening noise (%):"))
        self.cap_flo.setText(self.tr("F low (Hz):"))
        self.cap_fhi.setText(self.tr("F high (Hz):"))
        self.cap_clip.setText(self.tr("Amplitude clip (%):"))
        self.cap_tvg_alpha.setText(self.tr("Attenuation coef. α:"))
        self.cap_agc_win.setText(self.tr("AGC window (ms):"))
        self.cap_fix_interval.setText(self.tr("Interval (min):"))
        # Checkboxes / button
        self.inv_cmap.setText(self.tr("Invert colors"))
        self.decon.setText(self.tr("Enable deconvolution"))
        self.filt.setText(self.tr("Enable filter"))
        self.tvg.setText(self.tr("Adaptive TVG (compensate α)"))
        self.agc.setText(self.tr("Apply AGC"))
        self.fix.setText(self.tr("Show FIX marks"))
        self.sec_geometry.setText(self.tr("GEOMETRY & PRESENTATION"))
        self.align_delays.setText(self.tr("Compensate delays (align groups)"))
        self.show_boundaries.setText(self.tr("Show file boundaries (red lines)"))
        self.show_boundaries.setToolTip(self.tr("Show file boundaries (red lines)"))
        self.sec_scale.setText(self.tr("SCALE / PROPORTIONS"))
        self.rb_aspect.setText(self.tr("Lock aspect ratio (W:H)"))
        self.rb_aspect.setToolTip(self.tr(
            "Constant figure shape for every line; vertical exaggeration varies "
            "with line length."))
        self.rb_ve.setText(self.tr("Lock vertical exaggeration"))
        self.rb_ve.setToolTip(self.tr(
            "Constant VE → geologically comparable across all lines; very long "
            "lines become wide/short."))
        self.rb_hybrid.setText(self.tr("Hybrid (VE, capped)"))
        self.rb_hybrid.setToolTip(self.tr(
            "Locks the VE above, but caps the width:height ratio so extreme lines "
            "don't deform into an 'infinite noodle'."))
        self.cap_scale_hint.setText(self.tr(
            "Hybrid uses the VE value, limited by the max-aspect on its right."))
        self.sld_deform.setToolTip(self.tr(
            "Horizontal deformation (stretch / vertical exaggeration): drag for a "
            "fine, live adjustment of the aspect ratio. Synced with the box on its "
            "right."))
        self.sp_ratio.setToolTip(self.sld_deform.toolTip())
        self.btn_scale_reset.setText(self.tr("↺  Reset aspect settings"))
        self.cap_pxtrace.setText(self.tr("Pixels / trace (resolution)"))
        self.sp_pxtrace.setToolTip(self.tr(
            "Horizontal sharpness: higher = more pixels per trace in the export "
            "and more detail in the live view (heavier). Matches CLI --px-per-trace."))
        self.btn_render.setText(self.tr("⟳  Render Full"))
        self.btn_render.setToolTip(self.tr("Re-render the whole seismic line."))
        self.btn_render_viewport.setText(self.tr("🔍  Render Viewport HQ"))
        self.btn_render_viewport.setToolTip(self.tr(
            "Render the visible (zoomed-in) area at high quality and overlay it on "
            "the live view. Pan or zoom to dismiss it."))
        self.sec_export.setText(self.tr("EXPORT"))
        self.btn_export_img.setText(self.tr("💾  Export image"))
        self.btn_export_fix.setText(self.tr("🗺  Export FIX → SHP / GeoJSON / CSV"))
        self._update_preset_desc()
