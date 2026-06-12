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
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..i18n import language_manager

# Core domain constants (labels for the combos + preset descriptions).
from topassuite.core.constants import (
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
        self._lbl.setFixedWidth(48)
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
    export_image_requested = pyqtSignal()
    export_fix_requested = pyqtSignal()
    scale_changed = pyqtSignal()  # aspect ratio changed (live)
    boundaries_toggled = pyqtSignal(bool)  # show/hide file-seam lines (live)
    align_toggled = pyqtSignal(bool)  # delay-alignment geometry toggled (rebuild base)
    display_changed = pyqtSignal()  # cmap / clip / FIX changed → recolour preview

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(3)
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
        self.clip = LabeledSlider(80, 100, 1, 98, "{:.0f}")
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

        # ── Scale (display aspect; the export reproduces it) ──
        self.sec_scale = self._section()
        self.cap_aspect = self._caption()
        self.aspect_sp = QDoubleSpinBox()
        self.aspect_sp.setRange(0.0, 100.0)   # 0 = free (fill panel)
        self.aspect_sp.setSingleStep(0.5)
        self.aspect_sp.setValue(3.0)
        self.aspect_sp.valueChanged.connect(lambda *_: self.scale_changed.emit())
        v.addWidget(self.aspect_sp)

        # ── Action ──
        line = QFrame()
        line.setObjectName("hline")
        v.addWidget(line)
        self.btn_render = QPushButton()
        self.btn_render.clicked.connect(self.render_requested.emit)
        v.addWidget(self.btn_render)

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
            clip=int(self.clip.value()), cmap=self.cmap_cb.currentText(),
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
            clip=int(self.clip.value()),
            fix=self.fix.isChecked(),
            fix_iv=int(self.fix_interval.value()),
            align=self.align_delays.isChecked(),
            boundaries=self.show_boundaries.isChecked(),
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

    def aspect(self) -> Optional[float]:
        """Display aspect ratio W:H, or None when free (fill panel)."""
        v = self.aspect_sp.value()
        return None if v == 0 else v

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
        self.sec_scale.setText(self.tr("SCALE"))
        self.cap_aspect.setText(self.tr("Aspect ratio (W:H, 0=free)"))
        self.btn_render.setText(self.tr("⟳  RENDER"))
        self.sec_export.setText(self.tr("EXPORT"))
        self.btn_export_img.setText(self.tr("💾  Export image"))
        self.btn_export_fix.setText(self.tr("🗺  Export FIX → SHP / GeoJSON / CSV"))
        self._update_preset_desc()
