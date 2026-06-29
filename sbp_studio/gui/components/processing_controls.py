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
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QSlider, QSpinBox, QStackedWidget, QTabBar, QVBoxLayout, QWidget,
)

from ..i18n import language_manager

# Core domain constants (labels for the combos + preset descriptions).
from sbp_studio.core.constants import (
    CMAPS, FILTER_DESCRIPTIONS, FILTER_PRESETS,
)

# Live-preview horizontal detail (px per trace). The user-facing px/trace control
# was replaced by the physical 'traces per cm' scale; this fixed value keeps the
# preview column cap + HQ overlay sharpness at their historical level.
DEFAULT_PX_PER_TRACE = 20.0

# Preset combo grouping: a standard marine-seismic workflow order (complex-
# trace attributes → structural attributes → 2D image filters → frequency/
# smoothing) rather than FILTER_PRESETS' flat dict order. "none" (the
# no-filter sentinel) always stays first, ungrouped — see
# _populate_preset_combo. Imported (not redefined) from gui/dsp/nodes.py,
# which now ALSO uses this exact grouping for PresetNode's dynamic "Type"
# combo (_ChoiceRow in pipeline_panel.py) — one canonical category list for
# both widgets, so they can never drift out of sync. Re-exported under the
# original local names so existing internal references/tests are unaffected.
from ..dsp import PRESET_CATEGORIES as _PRESET_CATEGORIES
from ..dsp import tr_preset_category as _tr_preset_category

# The header-row paint fix (QStyledItemDelegate bypassing the active
# QStyle/QSS entirely for disabled rows — see theme.py's docstring) is
# shared with PresetNode's dynamic "Type" combo (_ChoiceRow in
# pipeline_panel.py) so both categorized combos render identically.
# Re-exported under the original local name for existing references/tests.
from ..theme import CategoryHeaderItemDelegate as _PresetHeaderDelegate


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


class DockDragMixin:
    """Shared mouse-drag logic so a widget embedded INSIDE a custom
    QDockWidget title bar can still 'tear off'/move the dock by dragging it.

    Replacing a dock's NATIVE title bar (``setTitleBarWidget``) loses Qt's
    own built-in drag-to-float handling for free — any custom widget placed
    there needs to reimplement it. This distinguishes a plain click (e.g.
    switching tabs) from an actual drag using Qt's own
    ``QApplication.startDragDistance()`` threshold, then floats + moves the
    dock directly. ``self._dock`` is ``None`` until a caller assigns the
    QDockWidget instance (it doesn't exist yet when the embedded widget is
    first constructed) — every method below is then a safe no-op.
    """
    _dock = None
    _drag_origin = None        # QPoint, global position at mouse-down
    _dock_press_offset = None  # QPoint, drag_origin − dock.pos() at mouse-down

    def _dock_press(self, event) -> None:
        if self._dock is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint()
            self._dock_press_offset = self._drag_origin - self._dock.pos()

    def _dock_move(self, event) -> None:
        if self._dock is None or self._drag_origin is None:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        pos = event.globalPosition().toPoint()
        if (pos - self._drag_origin).manhattanLength() < QApplication.startDragDistance():
            return
        if not self._dock.isFloating():
            # Floating changes dock.pos()'s coordinate origin — recompute
            # the press offset fresh the instant it happens so the dock
            # doesn't jump under the cursor.
            self._dock.setFloating(True)
            self._dock_press_offset = pos - self._dock.pos()
        self._dock.move(pos - self._dock_press_offset)

    def _dock_release(self, _event) -> None:
        self._drag_origin = None
        self._dock_press_offset = None


class _DraggableTabBar(QTabBar, DockDragMixin):
    """The "Controles y Procesado" / "Marcas y Exportación" tab strip,
    promoted into the dock's custom title bar (see _base.py's
    _build_controls_dock) — draggable via DockDragMixin so the user can
    still tear the panel off by dragging the tabs themselves, exactly as
    they could with Qt's native title bar."""

    def set_dock(self, dock) -> None:
        self._dock = dock

    def mousePressEvent(self, event) -> None:
        self._dock_press(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._dock_move(event)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dock_release(event)
        super().mouseReleaseEvent(event)


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
    interp_changed = pyqtSignal(str)  # 'nearest' | 'bilinear' → live ImageItem paint hint
    # Interpretation & Picking (Phase 3): checkable — True while picking mode
    # is active (double-click on the section places a marker).
    picking_toggled = pyqtSignal(bool)
    # "Exportar/Importar" clicked — the tab shows the export/import dialog.
    export_import_picking_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Two tabs so the left dock isn't one long, cluttered column:
        # "Controles y Procesado" keeps the DSP pipeline / palette / scale /
        # render controls; "Marcas y Exportación" groups everything about
        # FIX marks, file-boundary lines, and the export/marker buttons.
        #
        # The tab STRIP and the tab CONTENT are deliberately split into two
        # separate widgets — a QTabBar (``self.tabs_controls``) driving a
        # plain QStackedWidget — rather than one QTabWidget. A QTabWidget
        # bundles both into a single widget, which would make it impossible
        # to promote JUST the strip into the dock's custom title bar (see
        # _base.py's _build_controls_dock/DockTitleBar) while leaving the
        # actual page content in the dock's normal scrollable body below.
        # ``self._v`` is the CURRENT build target — every ``_section()``/
        # ``_caption()`` call and most inline ``v.addWidget(...)`` calls
        # below read/write through it, so simply reassigning
        # ``v = self._v = <tab's layout>`` at a handful of points routes the
        # REST of this unchanged construction code into whichever tab it
        # belongs to, with zero risk to any signal/slot connection (those
        # are independent of widget parentage/layout membership). Every
        # single control, including the externally-owned PipelinePanel (see
        # embed_pipeline_panel), ends up inside one of the two stacked
        # pages — none of them sit "above" or outside the tab strip.
        self.tabs_controls = _DraggableTabBar()
        self._stack = QStackedWidget()
        outer.addWidget(self._stack)
        self.tabs_controls.currentChanged.connect(self._stack.setCurrentIndex)

        tab1 = QWidget()
        v = QVBoxLayout(tab1)
        # Compact, fully-adjusted layout from startup: tight outer margins and a
        # small inter-row spacing so every control is visible without scrolling.
        v.setContentsMargins(6, 6, 6, 4)
        v.setSpacing(2)
        self._v = self._v1 = v

        tab2 = QWidget()
        self._v2 = QVBoxLayout(tab2)
        self._v2.setContentsMargins(6, 6, 6, 4)
        self._v2.setSpacing(2)

        self.tabs_controls.addTab("")
        self._stack.addWidget(tab1)
        self.tabs_controls.addTab("")
        self._stack.addWidget(tab2)

        # ── Palette ──
        self.sec_palette = self._section()
        self.cmap_cb = QComboBox()
        self.cmap_cb.addItems(list(CMAPS.keys()))
        v.addWidget(self.cmap_cb)
        # Invert + amplitude-range mode (diverging −1..1 vs sequential 0..1), one
        # row of checkboxes. The two ranges are mutually exclusive.
        self.inv_cmap = QCheckBox()
        self.amp_diverging = QCheckBox()       # [-1 to 1] — diverging amplitudes
        self.amp_sequential = QCheckBox()      # [ 0 to 1] — sequential amplitudes
        self.amp_sequential.setChecked(True)   # default: |amp| 0..1 (historical)
        _amp_row = QHBoxLayout()
        _amp_row.setContentsMargins(0, 0, 0, 0)
        _amp_row.setSpacing(8)
        _amp_row.addWidget(self.inv_cmap)
        _amp_row.addWidget(self.amp_diverging)
        _amp_row.addWidget(self.amp_sequential)
        _amp_row.addStretch(1)
        v.addLayout(_amp_row)
        self.amp_diverging.toggled.connect(self._on_amp_diverging)
        self.amp_sequential.toggled.connect(self._on_amp_sequential)

        # ── Render style (directly under the palette) ──
        # Show Wiggles / Show Variable Area are INDEPENDENT visibility toggles
        # (either can be on with the other off); Raster below, interactive only
        # while one of them is on. Wiggle traces render in BLACK. Both off =
        # pure density. Both default OFF so the initial preview stays the
        # historical density/raster-only view.
        self.wiggle_cb = QCheckBox()
        self.va_cb = QCheckBox()
        _wig_row = QHBoxLayout()
        _wig_row.setContentsMargins(0, 0, 0, 0)
        _wig_row.setSpacing(8)
        _wig_row.addWidget(self.wiggle_cb)
        _wig_row.addWidget(self.va_cb)
        _wig_row.addStretch(1)
        v.addLayout(_wig_row)
        self.raster_cb = QCheckBox()
        self.raster_cb.setChecked(True)
        self.raster_cb.setEnabled(False)       # enabled only while the overlay is on
        v.addWidget(self.raster_cb)
        self.wiggle_cb.toggled.connect(self._on_overlay_toggled)
        self.va_cb.toggled.connect(self._on_overlay_toggled)
        for _cb in (self.wiggle_cb, self.va_cb, self.raster_cb):
            _cb.toggled.connect(lambda *_: self.display_changed.emit())

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
        self.preset_cb.setItemDelegate(_PresetHeaderDelegate(self.preset_cb))
        self._populate_preset_combo()
        self.preset_cb.currentTextChanged.connect(self._update_preset_desc)
        v.addWidget(self.preset_cb)
        self.lbl_preset_desc = QLabel()
        self.lbl_preset_desc.setObjectName("sub")
        self.lbl_preset_desc.setWordWrap(True)
        v.addWidget(self.lbl_preset_desc)

        # ── FIX marks (moved to the "Marcas y Exportación" tab) ──
        v = self._v = self._v2
        self.sec_fix = self._section()
        self.fix = QCheckBox()
        self.fix.setChecked(False)   # default OFF (CRITICAL per spec)
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
        v = self._v = self._v1   # back to "Controles" for Geometry/Scale/Render

        # ── Geometry & Presentation (STATIC controls — NOT pipeline nodes) ──
        # Delay alignment is a static geometry correction (applied to the base
        # array before the dynamic DSP nodes), not a movable filter. File-seam
        # boundaries are a presentation overlay toggled live in the view —
        # moved to the "Marcas y Exportación" tab below (added straight to
        # self._v2 regardless of the current build target, since align_delays/
        # ab_compare on either side of it stay on this tab).
        self.sec_geometry = self._section()
        self.align_delays = QCheckBox()
        self.align_delays.setChecked(True)
        self.align_delays.toggled.connect(self.align_toggled.emit)
        v.addWidget(self.align_delays)
        self.show_boundaries = QCheckBox()
        self.show_boundaries.setChecked(False)   # default OFF (CRITICAL per spec)
        self.show_boundaries.toggled.connect(self.boundaries_toggled.emit)
        self._v2.addWidget(self.show_boundaries)   # lives on "Marcas y Exportación"
        # A/B Compare: split the section into RAW (left) vs the live DSP
        # pipeline output (right), with a labeled divider, so a filter's effect
        # is judged side-by-side. A presentation toggle → drives display_changed.
        self.ab_compare = QCheckBox()
        self.ab_compare.setChecked(False)
        self.ab_compare.toggled.connect(lambda *_: self.display_changed.emit())
        v.addWidget(self.ab_compare)

        # ── Scale / proportions — FOUR mutually-exclusive vertical modes ─────
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

        # Mode 0 — Free (fit to window): pyqtgraph's native unlocked, free-aspect
        # auto-fit, with NO ViewBox aspect lock at all (see SeismicView.set_aspect,
        # which already treats aspect=None as 'unlocked + autoRange'). THE DEFAULT
        # mode on construction (i.e. when a fresh file is loaded into a new panel) —
        # no surprising stretch/squash before the user has chosen a rule. It has no
        # value controls of its own.
        self.rb_free = QRadioButton()
        self.rb_free.setChecked(True)
        self.scale_group.addButton(self.rb_free, 3)
        _tight_row((self.rb_free, 1))

        # Mode 1 — Lock aspect ratio (constant W:H shape; VE floats with length).
        # Its value is the 'horizontal deformation' set by the slider + spin DIRECTLY
        # underneath it.
        self.rb_aspect = QRadioButton()
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
        # The radio is alone on its row; an indented slider + numeric box below it
        # set the VE value (mirrors the aspect-ratio slider/spin pattern above).
        self.rb_ve = QRadioButton()
        self.scale_group.addButton(self.rb_ve, 1)
        _tight_row((self.rb_ve, 1))
        self._VE_SCALE = 1.0             # slider int ↔ VE float (1:1, integer VE)
        self.sld_ve = QSlider(Qt.Orientation.Horizontal)
        self.sld_ve.setRange(1, 2000)
        self.sld_ve.setSingleStep(1)
        self.sld_ve.setPageStep(25)
        self.sld_ve.setMaximumWidth(120)
        self.sp_ve = QDoubleSpinBox()
        self.sp_ve.setRange(1.0, 2000.0)
        self.sp_ve.setDecimals(0)
        self.sp_ve.setSingleStep(1.0)
        self.sp_ve.setValue(67.0)
        self.sp_ve.setMaximumWidth(72)
        self.sld_ve.setValue(67)
        _ve_row = QHBoxLayout()
        _ve_row.setContentsMargins(16, 0, 0, 0)   # indent under the VE label
        _ve_row.setSpacing(4)
        _ve_row.addWidget(self.sld_ve, 0)
        _ve_row.addStretch(1)
        _ve_row.addWidget(self.sp_ve, 0)
        v.addLayout(_ve_row)
        self._syncing_ve = False
        self.sld_ve.valueChanged.connect(self._on_ve_slider)
        self.sp_ve.valueChanged.connect(self._on_ve_spin)

        # Mode 3 — Hybrid: lock VE (the value above) but cap the aspect so an
        # extremely long line never deforms into an 'infinite noodle'.
        self.rb_hybrid = QRadioButton()
        self.scale_group.addButton(self.rb_hybrid, 2)
        _tight_row((self.rb_hybrid, 1))
        self._MAXASP_SCALE = 10.0        # slider int ↔ max-aspect float (×0.1)
        self.sld_maxasp = QSlider(Qt.Orientation.Horizontal)
        self.sld_maxasp.setRange(10, 1000)        # 1.0 .. 100.0
        self.sld_maxasp.setSingleStep(5)          # 0.5
        self.sld_maxasp.setPageStep(25)
        self.sld_maxasp.setMaximumWidth(120)
        self.sp_maxasp = QDoubleSpinBox()
        self.sp_maxasp.setRange(1.0, 100.0)
        self.sp_maxasp.setSingleStep(0.5)
        self.sp_maxasp.setValue(5.0)
        self.sp_maxasp.setMaximumWidth(72)
        self.sld_maxasp.setValue(50)
        _maxasp_row = QHBoxLayout()
        _maxasp_row.setContentsMargins(16, 0, 0, 0)   # indent under the Hybrid label
        _maxasp_row.setSpacing(4)
        _maxasp_row.addWidget(self.sld_maxasp, 0)
        _maxasp_row.addStretch(1)
        _maxasp_row.addWidget(self.sp_maxasp, 0)
        v.addLayout(_maxasp_row)
        self._syncing_maxasp = False
        self.sld_maxasp.valueChanged.connect(self._on_maxasp_slider)
        self.sp_maxasp.valueChanged.connect(self._on_maxasp_spin)

        # ── Horizontal scale: traces per cm (label, then slider + numeric) ────
        # Replaces the old px/trace control. STRICTLY horizontal: sets the export
        # width (n_traces / tpc / 2.54) and the live horizontal density; the
        # vertical (time) scale is held constant (decoupled — see figsize_for_scale
        # and SeismicView.set_aspect). Higher = compressed; lower = stretched.
        #
        # Performance note: a rigorous VE formula that ties height to this
        # control's live width was tried and REVERTED — it made VE/hybrid mode
        # recompute the full vertical geometry on every drag tick, which made
        # dragging this slider laggy. VE/hybrid deliberately stay decoupled
        # (fixed GUI_X_SCALE constant, see figsize_for_scale) so this control
        # stays cheap and instantaneous, at the cost of VE not being perfectly
        # physically rigorous relative to the chosen trace density.
        self.cap_tpc = QLabel()
        self.cap_tpc.setObjectName("sub")
        v.addWidget(self.cap_tpc)                 # label on its own line, above
        self.sld_tpc = QSlider(Qt.Orientation.Horizontal)
        self.sld_tpc.setRange(1, 2000)
        self.sld_tpc.setSingleStep(1)
        self.sld_tpc.setPageStep(25)
        self.sp_tpc = QDoubleSpinBox()
        self.sp_tpc.setRange(1.0, 2000.0)
        self.sp_tpc.setDecimals(0)
        self.sp_tpc.setSingleStep(1.0)
        self.sp_tpc.setValue(40.0)
        self.sp_tpc.setMaximumWidth(72)
        self.sld_tpc.setValue(40)
        _tpc_row = QHBoxLayout()
        _tpc_row.setContentsMargins(0, 0, 0, 0)
        _tpc_row.setSpacing(4)
        _tpc_row.addWidget(self.sld_tpc, 1)       # slider fills the row width
        _tpc_row.addWidget(self.sp_tpc, 0)
        v.addLayout(_tpc_row)
        self._syncing_tpc = False
        self.sld_tpc.valueChanged.connect(self._on_tpc_slider)
        self.sp_tpc.valueChanged.connect(self._on_tpc_spin)

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

        # Any mode switch or value edit → live preview re-aspect immediately.
        for rb in (self.rb_free, self.rb_aspect, self.rb_ve, self.rb_hybrid):
            rb.toggled.connect(lambda *_: self.scale_changed.emit())
            # Strict state machine: a mode switch immediately enables ONLY that
            # mode's own controls and disables (greys out) everyone else's — see
            # _update_scale_mode_enabled. Fixes the cross-talk where an inactive
            # mode's slider/spin could still be dragged/edited and silently
            # mutate the view even though it had no visible effect.
            rb.toggled.connect(lambda *_: self._update_scale_mode_enabled())
        for sp in (self.sp_ratio, self.sp_ve, self.sp_maxasp):
            sp.valueChanged.connect(lambda *_: self.scale_changed.emit())
        # The master Trazas/cm control is intentionally DECOUPLED from VE's
        # height: VE/hybrid use a fixed constant (GUI_X_SCALE), not the live
        # width, precisely so dragging this slider is cheap and instant — no
        # vertical-geometry recompute is triggered by it (see figsize_for_scale
        # in _render.py for the rationale; rigorous-but-expensive VE math tied
        # to live width was tried and reverted for performance).
        self.sp_tpc.valueChanged.connect(lambda *_: self.scale_changed.emit())
        # Set the correct INITIAL enabled state (Free is checked by default, so
        # the other three modes' controls must start disabled, not just on the
        # first toggle).
        self._update_scale_mode_enabled()

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

        # ── Pixel interpolation for HQ Render / Image Export (and, live, the
        # on-screen raster) — nearest keeps hard pixel edges (historical look);
        # bilinear/bicubic smooth the upsampled raster. Nearest is the default so
        # existing renders are unaffected until the user opts in. Compact pill
        # toggles (QRadioButton, indicator-less per theme.py's segmented-control
        # style) — same look as the amplitude-range / scale-mode rows above.
        interp_row = QHBoxLayout()
        interp_row.setContentsMargins(0, 0, 0, 0)
        interp_row.setSpacing(4)
        self.rb_interp_nearest = QRadioButton()
        self.rb_interp_bilinear = QRadioButton()
        self.rb_interp_bicubic = QRadioButton()
        self.rb_interp_nearest.setChecked(True)
        self._interp_group = QButtonGroup(self)
        self._interp_group.addButton(self.rb_interp_nearest)
        self._interp_group.addButton(self.rb_interp_bilinear)
        self._interp_group.addButton(self.rb_interp_bicubic)
        interp_row.addWidget(self.rb_interp_nearest)
        interp_row.addWidget(self.rb_interp_bilinear)
        interp_row.addWidget(self.rb_interp_bicubic)
        interp_row.addStretch(1)
        v.addLayout(interp_row)
        for rb in (self.rb_interp_nearest, self.rb_interp_bilinear, self.rb_interp_bicubic):
            rb.toggled.connect(lambda *_: self.display_changed.emit())
            # Only the button that just became checked carries the new mode —
            # the other(s) in the exclusive group fire toggled(False) too, but
            # interp_mode() at that instant would already report the new
            # selection, so only emit on the checked=True transition.
            rb.toggled.connect(lambda checked: self.interp_changed.emit(self.interp_mode())
                               if checked else None)

        # "Controles" content ends here — anchor it to the top of its tab.
        v.addStretch(1)

        # ── Marcas y Exportación tab: FIX marks (above) + file boundaries
        # (above, on self._v2) + the export/marker buttons (below) ──────────
        v = self._v = self._v2

        # ── Export ── (high-quality matplotlib export via a dedicated dialog)
        self.sec_export = self._section()
        self.btn_export_img = QPushButton()
        self.btn_export_img.clicked.connect(self.export_image_requested.emit)
        v.addWidget(self.btn_export_img)
        self.btn_export_fix = QPushButton()
        self.btn_export_fix.clicked.connect(self.export_fix_requested.emit)
        v.addWidget(self.btn_export_fix)

        # ── Interpretation & Picking (Phase 3) ──
        # Checkable toggle: green while ACTIVE (double-click places a marker
        # on the section), red while inactive — a strong, unmistakable
        # on/off signal distinct from this panel's neutral button palette,
        # since accidentally leaving picking mode on is an easy way to place
        # unwanted markers while just trying to inspect the section.
        self.btn_toggle_picking = QPushButton()
        self.btn_toggle_picking.setCheckable(True)
        self.btn_toggle_picking.toggled.connect(self._on_picking_toggled)
        self.btn_toggle_picking.toggled.connect(self.picking_toggled.emit)
        v.addWidget(self.btn_toggle_picking)
        self._on_picking_toggled(False)   # paint the initial RED (inactive) state

        self.btn_export_import_picking = QPushButton()
        self.btn_export_import_picking.clicked.connect(
            self.export_import_picking_requested.emit)
        v.addWidget(self.btn_export_import_picking)

        v.addStretch(1)

        # Clean numeric inputs: strip the clunky up/down arrows from every spin
        # box in the panel. Values are set via the paired sliders, by typing, or
        # the scroll wheel — and dropping the arrow buttons stops the spin text
        # from being clipped in the narrow side panel.
        for _sb in self.findChildren((QSpinBox, QDoubleSpinBox)):
            _sb.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

        # Presentation changes drive a live recolour of the preview (debounced
        # downstream by the controller's own work being cheap).
        self.cmap_cb.currentTextChanged.connect(lambda *_: self.display_changed.emit())
        self.inv_cmap.toggled.connect(lambda *_: self.display_changed.emit())
        self.amp_diverging.toggled.connect(lambda *_: self.display_changed.emit())
        self.amp_sequential.toggled.connect(lambda *_: self.display_changed.emit())
        self.clip.valueChanged.connect(lambda *_: self.display_changed.emit())
        self.fix.toggled.connect(lambda *_: self.display_changed.emit())
        self.fix_interval.valueChanged.connect(lambda *_: self.display_changed.emit())

        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    # ── External embedding ───────────────────────────────────────────────────

    def embed_pipeline_panel(self, panel: QWidget) -> None:
        """Insert ``panel`` (the externally-owned PipelinePanel — "Flujo de
        procesado" / "Parámetros del módulo") at the very TOP of the
        "Controles y Procesado" tab's page, above Palette. The caller
        (_base.py's _build_controls_dock) owns the panel's lifetime/
        instance; this widget only places it — keeping every control inside
        one of the two stacked tab pages, with nothing left sitting above
        or outside the tab strip."""
        self._v1.insertWidget(0, panel)

    def tab_bar(self) -> QTabBar:
        """The "Controles y Procesado" / "Marcas y Exportación" tab strip —
        deliberately NOT added to this widget's own layout (see __init__);
        the caller (_base.py's _build_controls_dock) promotes it into the
        dock's custom title bar instead."""
        return self.tabs_controls

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
        show_wiggle_line = self.wiggle_cb.isChecked()
        show_va = self.va_cb.isChecked()
        # Either toggle alone is enough to enter wiggle-overlay mode — they are
        # independent visibility switches, not a coupled master/sub pair: with
        # both off the section falls back to pure density (no overlay at all).
        overlay_on = show_wiggle_line or show_va
        return dict(
            cmap=self.cmap_cb.currentText(),
            inv_cmap=self.inv_cmap.isChecked(),
            clip=float(self.clip.value()),
            fix=self.fix.isChecked(),
            fix_iv=int(self.fix_interval.value()),
            align=self.align_delays.isChecked(),
            boundaries=self.show_boundaries.isChecked(),
            px_per_trace=DEFAULT_PX_PER_TRACE,
            # Render style derived from the checkboxes: either Show Wiggles or
            # Show Variable Area being on enters 'wiggle' mode; the raster
            # underlay is user-toggleable only while the overlay is active
            # (Density always keeps the raster).
            style="wiggle" if overlay_on else "density",
            va_fill=show_va,
            show_wiggle_line=show_wiggle_line,
            show_raster=(self.raster_cb.isChecked() if overlay_on else True),
            # Amplitude range: diverging (−1..1, signed) vs sequential (0..1, |amp|).
            amp_range=("diverging" if self.amp_diverging.isChecked()
                       else "sequential"),
            # A/B Compare: raw|processed split render (see PreviewController).
            ab_compare=self.ab_compare.isChecked(),
            # Pixel-scaling mode for HQ Render / Image Export rasters.
            interp=self.interp_mode(),
        )

    def align_enabled(self) -> bool:
        """Whether the static delay-alignment correction is on."""
        return self.align_delays.isChecked()

    def interp_mode(self) -> str:
        """'nearest' | 'bilinear' | 'bicubic' — the pixel-scaling mode for HQ
        Render / Image Export (and the live raster); see the toggle row under
        the render buttons."""
        if self.rb_interp_bicubic.isChecked():
            return "bicubic"
        if self.rb_interp_bilinear.isChecked():
            return "bilinear"
        return "nearest"

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

        mode ∈ {'free','aspect','ve','hybrid'}; ratio (mode 'aspect'), ve (modes
        've'/'hybrid'), max_aspect (mode 'hybrid'). 'free' has no value of its
        own — it means 'no aspect lock', and the others are reported anyway so
        the export still has a sane fallback if free's None can't apply."""
        if self.rb_free.isChecked():
            mode = "free"
        elif self.rb_ve.isChecked():
            mode = "ve"
        elif self.rb_hybrid.isChecked():
            mode = "hybrid"
        else:
            mode = "aspect"
        # layout_mode is FORCED to 'decoupled' (the Layout-mode dropdown was
        # removed): traces/cm is strictly horizontal, the vertical scale comes
        # from VE and never changes with the trace spacing (kept cheap/instant
        # — see the performance note on the traces/cm widget above).
        return dict(mode=mode,
                    ratio=float(self.sp_ratio.value()),
                    ve=float(self.sp_ve.value()),
                    max_aspect=float(self.sp_maxasp.value()),
                    traces_per_cm=float(self.sp_tpc.value()),
                    layout_mode="decoupled")

    def px_per_trace(self) -> float:
        """Live-preview horizontal detail (px per trace) — a fixed value now that
        the physical 'traces per cm' control governs export width. Kept for the
        preview column cap + HQ overlay sharpness."""
        return DEFAULT_PX_PER_TRACE

    def render_style(self) -> str:
        """Section render style: 'wiggle' while either Show Wiggles or Show
        Variable Area is checked, else 'density'."""
        return ("wiggle" if (self.wiggle_cb.isChecked() or self.va_cb.isChecked())
               else "density")

    def show_raster(self) -> bool:
        """Whether the raster base layer is drawn (always True unless the wiggle
        overlay is active and the Raster checkbox is cleared = 'Wiggle Only')."""
        overlay_on = self.wiggle_cb.isChecked() or self.va_cb.isChecked()
        return self.raster_cb.isChecked() if overlay_on else True

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

    # ── Traces-per-cm slider ↔ spin sync ─────────────────────────────────────

    def _on_tpc_slider(self, val: int) -> None:
        if self._syncing_tpc:
            return
        self._syncing_tpc = True
        self.sp_tpc.setValue(float(val))
        self._syncing_tpc = False

    def _on_tpc_spin(self, val: float) -> None:
        if self._syncing_tpc:
            return
        self._syncing_tpc = True
        self.sld_tpc.setValue(int(round(val)))
        self._syncing_tpc = False

    # ── VE slider ↔ spin sync ────────────────────────────────────────────────

    def _on_ve_slider(self, val: int) -> None:
        if self._syncing_ve:
            return
        self._syncing_ve = True
        self.sp_ve.setValue(val / self._VE_SCALE)
        self._syncing_ve = False

    def _on_ve_spin(self, val: float) -> None:
        if self._syncing_ve:
            return
        self._syncing_ve = True
        self.sld_ve.setValue(int(round(val * self._VE_SCALE)))
        self._syncing_ve = False

    # ── Max-aspect slider ↔ spin sync ────────────────────────────────────────

    def _on_maxasp_slider(self, val: int) -> None:
        if self._syncing_maxasp:
            return
        self._syncing_maxasp = True
        self.sp_maxasp.setValue(val / self._MAXASP_SCALE)
        self._syncing_maxasp = False

    def _on_maxasp_spin(self, val: float) -> None:
        if self._syncing_maxasp:
            return
        self._syncing_maxasp = True
        self.sld_maxasp.setValue(int(round(val * self._MAXASP_SCALE)))
        self._syncing_maxasp = False

    # ── Scale-mode strict state machine ──────────────────────────────────────

    def _update_scale_mode_enabled(self) -> None:
        """Enable ONLY the active mode's own controls; grey out (disable) the
        other modes' sliders/spinboxes so they cannot be dragged/edited or fire
        a stray view-update while inactive — this is the fix for the reported
        UI cross-talk. The Hybrid mode shares the VE slider+spin with VE mode
        (it locks/edits the SAME 'vertical exaggeration' value, then caps the
        resulting aspect with its own max-aspect control), so VE's controls are
        enabled for EITHER mode. Free has no controls of its own. The master
        horizontal-scale control is NOT touched here — it stays enabled in
        every mode (wired once in __init__, never disabled)."""
        is_aspect = self.rb_aspect.isChecked()
        is_ve = self.rb_ve.isChecked()
        is_hybrid = self.rb_hybrid.isChecked()
        for w in (self.sld_deform, self.sp_ratio):
            w.setEnabled(is_aspect)
        for w in (self.sld_ve, self.sp_ve):
            w.setEnabled(is_ve or is_hybrid)
        for w in (self.sld_maxasp, self.sp_maxasp):
            w.setEnabled(is_hybrid)

    def _on_overlay_toggled(self, *_args) -> None:
        """The Raster checkbox is interactive only while the wiggle overlay is
        active — Show Wiggles or Show Variable Area checked (Density always
        keeps the raster). Its checked state is preserved across toggles."""
        self.raster_cb.setEnabled(self.wiggle_cb.isChecked() or self.va_cb.isChecked())

    def _on_amp_diverging(self, on: bool) -> None:
        """Diverging (−1..1) and sequential (0..1) amplitude ranges are mutually
        exclusive — at least one is always set (default sequential)."""
        if on:
            self.amp_sequential.setChecked(False)
        elif not self.amp_sequential.isChecked():
            self.amp_sequential.setChecked(True)

    def _on_amp_sequential(self, on: bool) -> None:
        if on:
            self.amp_diverging.setChecked(False)
        elif not self.amp_diverging.isChecked():
            self.amp_diverging.setChecked(True)

    def reset_scale_defaults(self) -> None:
        """Part 3: restore every aspect/scale setting to its default in one click.

        Free is the panel's actual default mode now (see __init__); resetting
        restores it too, alongside every sub-mode's own default value (so
        switching to Aspect/VE/Hybrid afterwards starts from a known state)."""
        self.rb_free.setChecked(True)      # also re-enables/disables via toggled
        self.sp_ratio.setValue(3.0)        # syncs the deformation slider too
        self.sp_ve.setValue(67.0)
        self.sp_maxasp.setValue(5.0)
        self.sp_tpc.setValue(40.0)         # syncs the traces/cm slider too
        self.scale_changed.emit()

    def aspect(self) -> Optional[float]:
        """Back-compat: the fixed W:H ratio when in aspect mode, else None.

        VE / hybrid aspects depend on the active line's length, so they are
        resolved per-source by ``_render.effective_aspect`` at the call site."""
        return float(self.sp_ratio.value()) if self.rb_aspect.isChecked() else None

    def boundaries_visible(self) -> bool:
        """Whether file-seam boundary lines should be shown in the live view."""
        return self.show_boundaries.isChecked()

    def _populate_preset_combo(self) -> None:
        """Build the categorized preset list: 'none' first (ungrouped, the
        no-filter sentinel), then each category as a DISABLED header item
        followed by its presets, in a standard marine-seismic workflow
        order (see _PRESET_CATEGORIES). Disabled items are automatically
        skipped by Qt when navigating the popup with click or keyboard, so
        _update_preset_desc's existing currentText()-based lookup needs no
        change — a header can never become the current selection.

        Re-callable on a language switch (see retranslate_ui) to refresh
        just the (translatable) header text — the preset LABELS themselves
        are core domain data, never translated (see this module's
        docstring), so the current selection survives a rebuild unchanged."""
        current = self.preset_cb.currentText()
        self.preset_cb.blockSignals(True)
        self.preset_cb.clear()
        key_to_label = {key: label for label, key in FILTER_PRESETS.items()}

        def _add_preset_item(key: str) -> None:
            label = key_to_label.get(key)
            if label is None:
                return
            self.preset_cb.addItem(label)
            idx = self.preset_cb.count() - 1
            desc = FILTER_DESCRIPTIONS.get(key, "")
            if desc:
                self.preset_cb.setItemData(idx, desc, Qt.ItemDataRole.ToolTipRole)

        def _add_header(name: str) -> None:
            # Clean text, no "---" decoration — _PresetHeaderDelegate's own
            # disabled/bold/accent-colour rendering is what marks this row
            # as a header, matching the "Add module" menu's plain-text
            # QLabel headers exactly (one visual system, not two).
            self.preset_cb.addItem(_tr_preset_category(name))
            idx = self.preset_cb.count() - 1
            item = self.preset_cb.model().item(idx)
            item.setEnabled(False)
            # The actual on-screen rendering of this row is handled by
            # _PresetHeaderDelegate (set as preset_cb's item delegate),
            # which paints disabled rows itself rather than trusting the
            # active QStyle/QSS to dim or distinguish them — that trust
            # turned out to be misplaced on at least one OS/style
            # combination. The bold/Black font set here is kept anyway as
            # the model-level source of truth for "this row is a header"
            # (also asserted by tests) and as a harmless fallback should
            # this item ever render through a different delegate. Size is
            # NOT bumped here (only the delegate does that, via the safe
            # bump_font_size) to avoid double-scaling the same row twice.
            font = item.font()
            font.setBold(True)
            font.setWeight(QFont.Weight.Black)
            item.setFont(font)

        _add_preset_item("none")
        categorized: set = set()
        for header_en, keys in _PRESET_CATEGORIES:
            _add_header(header_en)
            for key in keys:
                categorized.add(key)
                _add_preset_item(key)

        # Defensive catch-all: a preset NOT listed in _PRESET_CATEGORIES
        # (e.g. a future addition nobody re-categorised yet) still appears
        # here rather than silently vanishing from the combo —
        # FILTER_PRESETS stays the single source of truth for "what's
        # selectable".
        leftover = [k for k in FILTER_PRESETS.values()
                    if k != "none" and k not in categorized]
        if leftover:
            _add_header("Other")
            for key in leftover:
                _add_preset_item(key)

        restore_idx = self.preset_cb.findText(current) if current else -1
        self.preset_cb.setCurrentIndex(restore_idx if restore_idx >= 0 else 0)
        self.preset_cb.blockSignals(False)

    def _update_preset_desc(self, *_) -> None:
        key = FILTER_PRESETS.get(self.preset_cb.currentText(), "none")
        self.lbl_preset_desc.setText(FILTER_DESCRIPTIONS.get(key, ""))

    def _on_picking_toggled(self, checked: bool) -> None:
        """Paint btn_toggle_picking GREEN while active, RED while inactive —
        an explicit colour, not a theme token, since this is meant to stand
        out from every other (neutral) button in the panel — AND flip its
        text to name the action the next click will take ("Deactivate"
        while active, "Activate" while inactive), so the button's own label
        never lags behind its colour/checked state."""
        if checked:
            self.btn_toggle_picking.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; }"
                "QPushButton:hover { background-color: #388e3c; }")
            self.btn_toggle_picking.setText(self.tr("📍 Deactivate marker"))
        else:
            self.btn_toggle_picking.setStyleSheet(
                "QPushButton { background-color: #c62828; color: white; font-weight: bold; }"
                "QPushButton:hover { background-color: #d32f2f; }")
            self.btn_toggle_picking.setText(self.tr("📍 Activate marker"))

    def retranslate_ui(self) -> None:
        # Tabs — the dock's own conceptual naming ("Controls") is folded
        # straight into the tab header itself rather than repeated above it,
        # so the tab IS the primary, human-readable label for this section.
        self.tabs_controls.setTabText(0, self.tr("Controls and Processing"))
        self.tabs_controls.setTabText(1, self.tr("Marks and Export"))
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
        self.amp_diverging.setText(self.tr("[ -1 to 1 ]"))
        self.amp_diverging.setToolTip(self.tr(
            "Diverging amplitude range (−1..1) — colormaps centred on zero for "
            "signed amplitudes."))
        self.amp_sequential.setText(self.tr("[ 0 to 1 ]"))
        self.amp_sequential.setToolTip(self.tr(
            "Sequential amplitude range (0..1) — colormaps for |amplitude|."))
        self.wiggle_cb.setText(self.tr("Show Wiggles"))
        self.wiggle_cb.setToolTip(self.tr(
            "Draw traces as black wiggle lines over the raster (zoom in for more "
            "native detail). Independent of Show Variable Area — either can be "
            "on while the other is off."))
        self.va_cb.setText(self.tr("Show Variable Area"))
        self.va_cb.setToolTip(self.tr(
            "Fill the positive lobes of each wiggle (classic variable-area look). "
            "Independent of Show Wiggles — either can be on while the other is off."))
        self.raster_cb.setText(self.tr("Raster"))
        self.raster_cb.setToolTip(self.tr(
            "Show the colour raster underneath the wiggle/variable-area overlay "
            "(only while one of them is on; off = overlay only)."))
        self.decon.setText(self.tr("Enable deconvolution"))
        self.filt.setText(self.tr("Enable filter"))
        self.tvg.setText(self.tr("Adaptive TVG (compensate α)"))
        self.agc.setText(self.tr("Apply AGC"))
        self.fix.setText(self.tr("Show FIX marks"))
        self.sec_geometry.setText(self.tr("GEOMETRY & PRESENTATION"))
        self.align_delays.setText(self.tr("Compensate delays (align groups)"))
        self.show_boundaries.setText(self.tr("Show file boundaries"))
        self.show_boundaries.setToolTip(self.tr("Show file boundaries (red lines)"))
        self.ab_compare.setText(self.tr("A/B Compare"))
        self.ab_compare.setToolTip(self.tr(
            "Split the section: raw data on the left, the DSP pipeline output "
            "on the right, with a labeled divider."))
        self.sec_scale.setText(self.tr("SCALE / PROPORTIONS"))
        self.rb_free.setText(self.tr("Free (Fit to window)"))
        self.rb_free.setToolTip(self.tr(
            "No aspect lock — pyqtgraph's native free-fit view, exactly as a "
            "freshly loaded file looks. The default mode."))
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
        # Descriptive hint moved off-panel into the max-aspect spin's tooltip to
        # keep the side panel compact (no inline prose label).
        self.sp_maxasp.setToolTip(self.tr(
            "Hybrid uses the VE value, limited by the max-aspect on its right."))
        self.sld_deform.setToolTip(self.tr(
            "Horizontal deformation (stretch / vertical exaggeration): drag for a "
            "fine, live adjustment of the aspect ratio. Synced with the box on its "
            "right."))
        self.sp_ratio.setToolTip(self.sld_deform.toolTip())
        self.btn_scale_reset.setText(self.tr("↺ Reset"))
        self.btn_scale_reset.setToolTip(self.tr("Reset aspect settings"))
        self.cap_tpc.setText(self.tr("Traces / cm (horizontal scale)"))
        self.sp_tpc.setToolTip(self.tr(
            "Horizontal trace spacing: higher = more traces per cm (compressed), "
            "lower = stretched. Strictly horizontal — the vertical (time) scale "
            "never changes."))
        self.sld_tpc.setToolTip(self.sp_tpc.toolTip())
        self.btn_render.setText(self.tr("⟳  Render Full"))
        self.btn_render.setToolTip(self.tr("Re-render the whole seismic line."))
        self.btn_render_viewport.setText(self.tr("🔍 Viewport HQ"))
        self.btn_render_viewport.setToolTip(self.tr(
            "Render the visible (zoomed-in) area at high quality and overlay it on "
            "the live view. Pan or zoom to dismiss it."))
        self.rb_interp_nearest.setText(self.tr("Nearest"))
        self.rb_interp_bilinear.setText(self.tr("Bilinear"))
        self.rb_interp_bicubic.setText(self.tr("Bicubic"))
        self.sec_export.setText(self.tr("EXPORT"))
        self.btn_export_img.setText(self.tr("💾  Export image"))
        self.btn_export_fix.setText(self.tr("🗺 Export FIX"))
        self.btn_export_fix.setToolTip(self.tr("Export FIX → SHP / GeoJSON / CSV"))
        # Text (and colour) depend on the CURRENT checked state — re-derive
        # both from the single source of truth in _on_picking_toggled so a
        # language switch never reverts an active picker's label back to
        # "Activate" while it's still actually active.
        self._on_picking_toggled(self.btn_toggle_picking.isChecked())
        if self.btn_toggle_picking.isChecked():
            self.btn_toggle_picking.setToolTip(self.tr(
                "Picking mode is ON — double-click the section to place a marker. "
                "Click to turn off."))
        else:
            self.btn_toggle_picking.setToolTip(self.tr(
                "Turn on picking mode: double-click the section to place an "
                "interpretation marker."))
        self.btn_export_import_picking.setText(self.tr("Export/Import marker"))
        self.btn_export_import_picking.setToolTip(self.tr(
            "Export interpretation markers to SHP/GeoJSON/CSV, or import a "
            "previously saved session (.tps)."))
        # Preset combo: only the category HEADERS are translatable UI chrome
        # (the preset labels themselves are domain data — see this module's
        # docstring) — rebuild to refresh them, preserving the current
        # selection (see _populate_preset_combo).
        self._populate_preset_combo()
        self._update_preset_desc()
