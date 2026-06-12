"""
spectrum_view.py — Advanced Spectrum QC panel (PyQtGraph, on-demand).

Faithful PyQtGraph re-implementation of the original ``TopasSUITE.py``
``_draw_spectrum_figure`` (which used Matplotlib). Heavy Welch analysis
(``core.compute_spectrum``) runs ONLY on the "Generate" button, for the scope
selected in the combo (current ViewBox vs full profile).

Layout (mirrors the original 3-panel figure):
  ┌ control bar:  [ Generate ]   scope: (ViewBox | Full profile) ┐
  │ 1-D PSD   — Welch mean + P50 + filled P10–P90 + peak/centroid │
  │             + BW-3 dB band (shaded) + BW-6 dB edges           │
  │ 2-D spectrogram (freq×dist, inferno −50…0 dB)  │ band-energy  │
  │   + BW/peak guide lines + chain boundaries     │ distribution │
  │ stats: Peak·Centroid·BW-3/6·Roll-off·SNR·NFFT·windows         │
  └──────────────────────────────────────────────────────────────┘

No DSP/FFT here — the tab passes in a core ``SpectrumResult``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ..i18n import language_manager
from ..theme import MONO, theme

SCOPE_VIEWBOX = "viewbox"
SCOPE_FULL = "full"

# Explicit, high-contrast spectrum colours (independent of the app theme so the
# legend always reads as the user expects). Mean = cyan; indicators red/yellow/green.
MEAN_COLOR = "#2dd4ee"     # bright cyan  — Welch mean
PEAK_COLOR = "#ff4d4d"     # red          — peak frequency
CENTROID_COLOR = "#ffd23f"  # yellow      — spectral centroid
BW_COLOR = "#35d07f"       # green        — BW-3 dB / BW-6 dB


class SpectrumView(QWidget):
    """On-demand advanced spectrum QC (Welch PSD + spectrogram + band energy + stats)."""

    generate_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Control bar ──
        bar = QHBoxLayout()
        self.btn_generate = QPushButton()
        self.btn_generate.setObjectName("primary")
        self.btn_generate.clicked.connect(
            lambda: self.generate_requested.emit(self.scope_data()))
        self.lbl_scope = QLabel()
        self.lbl_scope.setObjectName("sub")
        self.cb_scope = QComboBox()
        self.cb_scope.addItem("", SCOPE_VIEWBOX)
        self.cb_scope.addItem("", SCOPE_FULL)
        bar.addWidget(self.btn_generate)
        bar.addSpacing(10)
        bar.addWidget(self.lbl_scope)
        bar.addWidget(self.cb_scope)
        bar.addStretch(1)
        root.addLayout(bar)

        # ── Graphics layout (PSD on top; spectrogram + band bars below) ──
        self.glw = pg.GraphicsLayoutWidget()
        root.addWidget(self.glw, 1)

        # 1-D PSD plot (spans both columns).
        self.psd = self.glw.addPlot(row=0, col=0, colspan=2)
        self.psd.setMenuEnabled(False)
        self.psd.showGrid(x=True, y=True, alpha=0.15)
        self.psd.setYRange(-80, 3, padding=0)
        for ax in ("left", "bottom"):
            self.psd.getAxis(ax).enableAutoSIPrefix(False)
        self.psd_legend = self.psd.addLegend(offset=(-10, 8))
        self.region_bw3 = pg.LinearRegionItem(
            values=[0, 0], orientation="vertical", movable=False)
        self.region_bw3.setZValue(-10)
        self.psd.addItem(self.region_bw3)
        self._fill = None
        self.c_p10 = self.psd.plot([], [])
        self.c_p90 = self.psd.plot([], [])
        self.c_p50 = self.psd.plot([], [], name="")
        self.c_mean = self.psd.plot([], [], name="")
        self.line_peak = pg.InfiniteLine(angle=90, movable=False)
        self.line_centroid = pg.InfiniteLine(angle=90, movable=False)
        self.line_bw6_lo = pg.InfiniteLine(angle=90, movable=False)
        self.line_bw6_hi = pg.InfiniteLine(angle=90, movable=False)
        for ln in (self.line_peak, self.line_centroid, self.line_bw6_lo, self.line_bw6_hi):
            self.psd.addItem(ln)
        self.txt_peak = pg.TextItem(anchor=(0, 1))
        self.psd.addItem(self.txt_peak)

        # Invisible proxy curves carrying the indicator pens, purely so the
        # legend can show a coloured line sample for Peak / Centroid / BW-3 / -6.
        self.proxy_peak = pg.PlotDataItem([], [])
        self.proxy_centroid = pg.PlotDataItem([], [])
        self.proxy_bw3 = pg.PlotDataItem([], [])
        self.proxy_bw6 = pg.PlotDataItem([], [])

        # 2-D spectrogram.
        self.spectro = self.glw.addPlot(row=1, col=0)
        self.spectro.setMenuEnabled(False)
        self.spectro.invertY(False)
        for ax in ("left", "bottom"):
            self.spectro.getAxis(ax).enableAutoSIPrefix(False)
        self.img = pg.ImageItem()
        self.spectro.addItem(self.img)
        self._cbar = None
        self._spectro_lines: list = []

        # Band-energy distribution (horizontal bars).
        self.bandplot = self.glw.addPlot(row=1, col=1)
        self.bandplot.setMenuEnabled(False)
        self.bandplot.setMaximumWidth(220)
        self.bandplot.showGrid(x=True, alpha=0.15)
        for ax in ("left", "bottom"):
            self.bandplot.getAxis(ax).enableAutoSIPrefix(False)
        self._bars = None
        self._band_peak_idx = -1     # which band carries the most energy
        self._n_bands = 0

        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)

        # ── Stats readout ──
        self.stats = QLabel()
        self.stats.setObjectName("sub")
        self.stats.setWordWrap(True)
        root.addWidget(self.stats)

        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)

    # ── Public API ──────────────────────────────────────────────────────────

    def scope_data(self) -> str:
        return self.cb_scope.currentData()

    def show_result(self, sp, fs: float, dist_km=None,
                    boundaries: Sequence[float] = ()) -> None:
        """Render a core ``SpectrumResult``."""
        fkhz = sp.freqs / 1000.0
        f_max = min(20.0, fs / 2000.0)
        bw3_lo, bw3_hi = sp.bw_3db_lo / 1000.0, sp.bw_3db_hi / 1000.0
        bw6_lo, bw6_hi = sp.bw_6db_lo / 1000.0, sp.bw_6db_hi / 1000.0
        pk, ct = sp.peak_hz / 1000.0, sp.centroid_hz / 1000.0

        # ── 1-D PSD ──
        self.c_p10.setData(fkhz, sp.spec_p10_db)
        self.c_p90.setData(fkhz, sp.spec_p90_db)
        self.c_p50.setData(fkhz, sp.spec_p50_db)
        self.c_mean.setData(fkhz, sp.spec_mean_db)
        if self._fill is not None:
            self.psd.removeItem(self._fill)
        # Translucent DARK band — a subtle backdrop that never overpowers the
        # bright mean / median lines drawn on top of it.
        fcol = pg.mkColor("#46566f"); fcol.setAlpha(60)
        self._fill = pg.FillBetweenItem(self.c_p10, self.c_p90, brush=fcol)
        self._fill.setZValue(-5)
        self.psd.addItem(self._fill)
        self.region_bw3.setRegion([bw3_lo, bw3_hi])
        self.line_peak.setValue(pk)
        self.line_centroid.setValue(ct)
        self.line_bw6_lo.setValue(bw6_lo)
        self.line_bw6_hi.setValue(bw6_hi)
        pk_db = float(sp.spec_mean_db[np.argmax(sp.spec_mean_db)])
        self.txt_peak.setText(f"{self.tr('Peak')} {pk:.2f} kHz")
        self.txt_peak.setColor(PEAK_COLOR)
        self.txt_peak.setPos(min(pk + 0.05 * f_max, 0.8 * f_max), pk_db)
        self.psd.setXRange(0, f_max, padding=0)

        # ── 2-D spectrogram ──
        f_mask = sp.freqs <= f_max * 1000.0
        spec2d = np.asarray(sp.spec_2d_db)[f_mask, :]
        if dist_km is not None and len(dist_km):
            x0, x1 = float(dist_km[0]), float(dist_km[-1])
        else:
            x0, x1 = 0.0, float(spec2d.shape[1])
        self.img.setImage(spec2d, autoLevels=False)
        self.img.setLevels([-50.0, 0.0])
        self.img.setLookupTable(self._inferno_lut())
        self.img.setRect(QRectF(x0, 0.0, max(1e-6, x1 - x0), f_max))
        self._draw_spectro_lines(boundaries, bw3_lo, bw3_hi, pk)
        self._update_cbar()

        # ── Band-energy distribution ──
        self._draw_bands(sp)

        # ── Stats ──
        self.stats.setText(self._stats_text(sp))

    def clear(self) -> None:
        for c in (self.c_p10, self.c_p90, self.c_p50, self.c_mean):
            c.setData([], [])
        self.img.clear()
        self.txt_peak.setText("")
        if self._bars is not None:
            self.bandplot.removeItem(self._bars); self._bars = None
        self.stats.setText("")

    # ── Internals ───────────────────────────────────────────────────────────

    def _draw_bands(self, sp) -> None:
        if self._bars is not None:
            self.bandplot.removeItem(self._bars)
            self._bars = None
        labels = list(sp.band_labels)
        pcts = list(sp.band_pcts)
        if not labels:
            self._n_bands = 0
            self._band_peak_idx = -1
            return
        # The band with the most energy is highlighted; remember it so a later
        # theme switch (_restyle) can re-tint the bars without recomputing.
        self._n_bands = len(labels)
        self._band_peak_idx = int(np.argmax(pcts)) if pcts else -1
        y = np.arange(len(labels))
        self._bars = pg.BarGraphItem(
            x0=0, y=y, height=0.7, width=pcts,
            brushes=self._bar_brushes(), pen=pg.mkPen(theme.color("bg")))
        self.bandplot.addItem(self._bars)
        self.bandplot.getAxis("left").setTicks(
            [[(i, lab) for i, lab in enumerate(labels)]])
        self.bandplot.setYRange(-0.6, len(labels) - 0.4, padding=0)
        self.bandplot.setXRange(0, (max(pcts) * 1.18) if pcts else 1.0, padding=0)

    def _default_bar_brush(self):
        """Contrasting grey for non-peak bars: light grey on a dark background,
        dark grey on a light one (chosen by the panel's luminance, so it works
        for any theme)."""
        bg = pg.mkColor(theme.color("panel"))
        lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        return pg.mkBrush("#566070" if lum > 128 else "#bcc6d4")

    def _bar_brushes(self):
        """Brush list: bright cyan for the peak band, adaptive grey for the rest."""
        default = self._default_bar_brush()
        brushes = [default for _ in range(self._n_bands)]
        if 0 <= self._band_peak_idx < self._n_bands:
            brushes[self._band_peak_idx] = pg.mkBrush(MEAN_COLOR)   # theme-agnostic
        return brushes

    def _stats_text(self, sp) -> str:
        bw3 = (sp.bw_3db_hi - sp.bw_3db_lo) / 1000.0
        bw6 = (sp.bw_6db_hi - sp.bw_6db_lo) / 1000.0
        warn = ("   ⚠ " + self.tr("low resolution (short profile)")
                if getattr(sp, "low_res", False) else "")
        return (f"  {self.tr('Peak')} {sp.peak_hz/1000:.3f} kHz    "
                f"{self.tr('Centroid')} {sp.centroid_hz/1000:.3f} kHz    "
                f"BW-3dB {bw3:.2f} kHz ({sp.bw_3db_lo/1000:.2f}–{sp.bw_3db_hi/1000:.2f})    "
                f"BW-6dB {bw6:.2f} kHz    "
                f"{self.tr('Roll-off 85%')} {sp.roll_off_hz/1000:.2f} kHz    "
                f"SNR {sp.snr_db:.1f} dB    "
                f"NFFT {sp.nfft}    "
                f"{sp.n_frames} {self.tr('windows/trace (Welch 50%)')}{warn}")

    @staticmethod
    def _inferno_lut() -> np.ndarray:
        try:
            return pg.colormap.get("inferno", source="matplotlib").getLookupTable(
                0.0, 1.0, 256, alpha=False)
        except Exception:
            return pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256, alpha=False)

    def _draw_spectro_lines(self, boundaries: Sequence[float],
                            bw3_lo: float, bw3_hi: float, pk: float) -> None:
        for ln in self._spectro_lines:
            self.spectro.removeItem(ln)
        self._spectro_lines.clear()
        # Horizontal guide lines: BW-3 dB edges (green) and peak (red).
        pen_bw = pg.mkPen(BW_COLOR, width=0.8, style=Qt.PenStyle.DashLine)
        pen_pk = pg.mkPen(PEAK_COLOR, width=0.8, style=Qt.PenStyle.DotLine)
        for yv, pen in ((bw3_lo, pen_bw), (bw3_hi, pen_bw), (pk, pen_pk)):
            ln = pg.InfiniteLine(pos=float(yv), angle=0, pen=pen)
            self.spectro.addItem(ln)
            self._spectro_lines.append(ln)
        # Vertical chain-boundary lines.
        pen_b = pg.mkPen(theme.color("warn"), width=1, style=Qt.PenStyle.DashLine)
        for x in boundaries:
            ln = pg.InfiniteLine(pos=float(x), angle=90, pen=pen_b)
            self.spectro.addItem(ln)
            self._spectro_lines.append(ln)

    def _update_cbar(self) -> None:
        if self._cbar is not None:
            return
        try:
            cm = pg.colormap.get("inferno", source="matplotlib")
        except Exception:
            cm = pg.colormap.get("viridis")
        self._cbar = pg.ColorBarItem(values=(-50.0, 0.0), colorMap=cm,
                                     label=self.tr("dB"))
        self._cbar.setImageItem(self.img, insert_in=self.spectro)

    def _restyle(self, *_) -> None:
        self.glw.setBackground(theme.color("panel"))

        # ── Distinct PSD curves ──
        # Mean  = bright CYAN, solid, prominent.
        # Median= high-contrast text colour (white on dark / dark on light), dashed.
        # P10/P90 = faint dotted lines — the translucent band is the real backdrop.
        self.c_mean.setPen(pg.mkPen(MEAN_COLOR, width=2))
        self.c_p50.setPen(pg.mkPen(theme.color("text"), width=1.3, style=Qt.PenStyle.DashLine))
        faint = pg.mkColor(theme.color("sub")); faint.setAlpha(110)
        self.c_p10.setPen(pg.mkPen(faint, width=1, style=Qt.PenStyle.DotLine))
        self.c_p90.setPen(pg.mkPen(faint, width=1, style=Qt.PenStyle.DotLine))

        # ── Indicator pens (explicit red/yellow/green) → applied to the live
        # line AND its legend proxy so the legend shows exactly what each means. ──
        peak_pen     = pg.mkPen(PEAK_COLOR,     width=1.5, style=Qt.PenStyle.DashLine)
        centroid_pen = pg.mkPen(CENTROID_COLOR, width=1.5, style=Qt.PenStyle.DotLine)
        bw3_pen      = pg.mkPen(BW_COLOR,       width=1.5)
        bw6_pen      = pg.mkPen(BW_COLOR,       width=1.1, style=Qt.PenStyle.DashDotLine)
        self.line_peak.setPen(peak_pen);         self.proxy_peak.setPen(peak_pen)
        self.line_centroid.setPen(centroid_pen); self.proxy_centroid.setPen(centroid_pen)
        self.proxy_bw3.setPen(bw3_pen)
        for ln in (self.line_bw6_lo, self.line_bw6_hi):
            ln.setPen(bw6_pen)
        self.proxy_bw6.setPen(bw6_pen)
        reg = pg.mkColor(BW_COLOR); reg.setAlpha(40)
        self.region_bw3.setBrush(reg)

        # Re-tint the band-energy bars so a theme switch keeps the peak band
        # bright-cyan and the rest a contrasting grey for the new background.
        if self._bars is not None and self._n_bands:
            self._bars.setOpts(brushes=self._bar_brushes(),
                               pen=pg.mkPen(theme.color("bg")))

        for plot in (self.psd, self.spectro, self.bandplot):
            for ax in ("left", "bottom"):
                axis = plot.getAxis(ax)
                axis.setPen(theme.color("sub"))
                axis.setTextPen(theme.color("text"))
                axis.setStyle(tickFont=QFont(MONO, 8))

    def _retranslate(self, *_) -> None:
        self.btn_generate.setText(self.tr("⚙  Generate advanced analysis"))
        self.lbl_scope.setText(self.tr("Scope:"))
        self.cb_scope.setItemText(0, self.tr("Current ViewBox"))
        self.cb_scope.setItemText(1, self.tr("Full profile"))
        self.psd.setLabel("bottom", self.tr("Frequency (kHz)"))
        self.psd.setLabel("left", self.tr("Amplitude (dB re. max)"))
        self.psd.setTitle(self.tr("Welch spectrum + percentiles"),
                          color=theme.color("text"), size="9pt")
        self.spectro.setLabel("bottom", self.tr("Distance (km)"))
        self.spectro.setLabel("left", self.tr("Frequency (kHz)"))
        self.bandplot.setLabel("bottom", self.tr("Energy (%)"))
        self.bandplot.setTitle(self.tr("Band distribution"),
                               color=theme.color("text"), size="8pt")
        if self.psd_legend is not None:
            self.psd_legend.clear()
            # Curves + every indicator line, so the user sees what the cyan,
            # white, red, yellow and green lines each represent.
            self.psd_legend.addItem(self.c_mean, self.tr("Welch mean"))
            self.psd_legend.addItem(self.c_p50, self.tr("Median"))
            self.psd_legend.addItem(self.c_p90, self.tr("P10–P90"))
            self.psd_legend.addItem(self.proxy_peak, self.tr("Peak"))
            self.psd_legend.addItem(self.proxy_centroid, self.tr("Centroid"))
            self.psd_legend.addItem(self.proxy_bw3, self.tr("BW -3 dB"))
            self.psd_legend.addItem(self.proxy_bw6, self.tr("BW -6 dB"))
            try:
                self.psd_legend.setColumnCount(2)   # compact 2-column layout
            except Exception:
                pass
