"""
seismic_view.py — Interactive seismic section (PyQtGraph, GPU-accelerated).

Renders a float32 amplitude raster (produced off-thread by
``tabs._render.compute_section``) into a ``pg.ImageItem`` with a Look-Up
Table so colorisation happens entirely in C++/GPU — no Python RGBA loop.

Time grows downward (Y inverted), distance along X. Optional chain-join
boundaries and FIX marks are drawn as vertical dashed lines.

Dynamic zoom
------------
The display buffer (``_arr``) is a decimated float32 array that covers the
full [dist0, dist1] × [t0, t1] extent. When the user zooms in, the
``ViewBox.sigRangeChanged`` handler re-slices the buffer to the visible
column range and passes the sub-array to ``ImageItem`` — giving full buffer
resolution for the zoomed window with no extra I/O.

Export separation
-----------------
``_arr`` is strictly a display cache. Exports call ``render_profile_figure``
(the core headless matplotlib path) with the original full-resolution data
matrix. Neither ``_arr`` nor the LUT is involved in any export path.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..i18n import language_manager
from ..theme import MONO, theme

# ── Global PyQtGraph config ───────────────────────────────────────────────────
# row-major: arr[row, col] → row = Y (time), col = X (distance).
pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("antialias", False)
pg.setConfigOption("useOpenGL", True)   # hardware-accelerated pan/zoom

# FIX mark tuple: (number, distance_km, label)
FixMark = Tuple[int, float, str]


class SeismicView(QWidget):
    """PyQtGraph seismic section fed with a pre-computed float32 amplitude array."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.invertY(True)  # time downward
        self.img = pg.ImageItem(autoDownsample=True)
        self.plot.addItem(self.img)
        # Disable SI auto-prefix so axes show real km/ms (not "(×0.001)").
        for _ax in ("left", "bottom"):
            self.plot.getAxis(_ax).enableAutoSIPrefix(False)

        self._cbar: Optional[pg.ColorBarItem] = None
        self._overlay_lines: List[pg.InfiniteLine] = []
        self._boundary_lines: List[pg.InfiniteLine] = []  # toggled live
        self._boundaries_visible: bool = True

        # Display buffer state (set by show_image; used by _on_range_changed).
        self._arr: Optional[np.ndarray] = None   # (rows, cols) float32
        self._lut: Optional[np.ndarray] = None   # (256, 3) uint8
        self._vmax: float = 1.0
        self._cmap_name: str = "viridis"
        self._rect: tuple = (0.0, 1.0, 0.0, 1.0)   # (dist0, dist1, t0, t1)
        self._aspect: Optional[float] = None
        self._last_zoom_key: Optional[tuple] = None  # (c0, c1, stride) dedup

        # Debounce zoom updates: accumulate rapid range-change events and apply
        # 60 ms after the last one to avoid re-slicing on every mouse-wheel tick.
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(60)
        self._zoom_timer.timeout.connect(self._apply_zoom_update)
        self._pending_ranges: Optional[list] = None

        # Connect dynamic-zoom handler.
        self.plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)

        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)

    # ── Public API ──────────────────────────────────────────────────────────

    def show_image(self, arr: np.ndarray,
                   dist0: float, dist1: float, t0: float, t1: float,
                   *, cmap_name: str, vmax: float,
                   title: str = "",
                   boundaries: Sequence[float] = (),
                   fixes: Sequence[FixMark] = (),
                   aspect: Optional[float] = None,
                   boundaries_visible: bool = True) -> None:
        """Display a float32 amplitude raster with LUT colorisation.

        ``arr``   — (rows, cols) float32 display buffer from ``compute_section``.
        ``vmax``  — amplitude value that maps to the top of the colourmap.
        ``aspect``— on-screen W:H ratio of the data box (None = free).

        The LUT is applied in C++/GPU by ``pg.ImageItem``. The same ``aspect``
        is reproduced in the exported PDF/image via the iterative tight_layout
        in ``_base._on_export_image_requested``.
        """
        self._vmax = float(vmax) or 1.0
        self._cmap_name = cmap_name
        self._rect = (float(dist0), float(dist1), float(t0), float(t1))
        self._last_zoom_key = None  # force a zoom refresh on the new data
        self._boundaries_visible = bool(boundaries_visible)

        # Store the display buffer for viewport-adaptive re-slicing on zoom.
        self._arr = arr

        # Build LUT and push the initial full-buffer image.
        self._update_colorbar()
        self._push_full_image()
        self._draw_overlays(boundaries, fixes)
        if title:
            self.plot.setTitle(title, color=theme.color("text"), size="9pt")
        self.set_aspect(aspect)

    def set_aspect(self, aspect: Optional[float]) -> None:
        """Lock the data box to a W:H ratio (None = free), then fit."""
        self._aspect = aspect
        vb = self.plot.getViewBox()
        d0, d1, t0, t1 = self._rect
        x_ext, y_ext = (d1 - d0), (t1 - t0)
        if aspect and x_ext > 0 and y_ext > 0:
            vb.setAspectLocked(True, ratio=aspect * y_ext / x_ext)
        else:
            vb.setAspectLocked(False)
        self.plot.autoRange()

    def has_image(self) -> bool:
        return self._arr is not None

    def set_boundaries_visible(self, visible: bool) -> None:
        """Show/hide the red file-seam boundary lines live (no re-render)."""
        self._boundaries_visible = bool(visible)
        for ln in self._boundary_lines:
            ln.setVisible(self._boundaries_visible)

    def export_image(self, path: str, dpi: int = 300) -> None:
        """Fallback pyqtgraph screenshot export (not used by the main export path)."""
        from pyqtgraph.exporters import ImageExporter, SVGExporter
        if path.lower().endswith(".svg"):
            SVGExporter(self.plot).export(path)
        else:
            exp = ImageExporter(self.plot)
            exp.parameters()["width"] = int(max(1200, dpi * 10))
            exp.export(path)

    # ── Dynamic zoom ────────────────────────────────────────────────────────

    def _on_range_changed(self, _vb, ranges) -> None:
        """Schedule a display-buffer re-slice when the viewport changes."""
        if self._arr is None:
            return
        self._pending_ranges = ranges
        if not self._zoom_timer.isActive():
            self._zoom_timer.start()

    def _apply_zoom_update(self) -> None:
        """Slice the display buffer to the visible column window and push to GPU."""
        ranges = self._pending_ranges
        arr = self._arr
        if ranges is None or arr is None:
            return

        d0, d1, t0, t1 = self._rect
        total_km = d1 - d0
        total_cols = arr.shape[1]
        if total_km <= 0 or total_cols < 2:
            return

        xv0, xv1 = ranges[0]
        vis0 = max(d0, xv0)
        vis1 = min(d1, xv1)
        if vis1 <= vis0:
            return

        # Map the visible km range to buffer column indices.
        c0 = max(0, int((vis0 - d0) / total_km * total_cols))
        c1 = min(total_cols, int(math.ceil((vis1 - d0) / total_km * total_cols)))
        visible_cols = max(1, c1 - c0)

        # Match stride to the viewport pixel width so we never over-render.
        vb_px = max(1, int(self.plot.sceneBoundingRect().width()))
        stride = max(1, visible_cols // vb_px)

        key = (c0, c1, stride)
        if key == self._last_zoom_key:
            return   # nothing to do
        self._last_zoom_key = key

        sub = arr[:, c0:c1:stride]
        sub_d0 = d0 + (c0 / total_cols) * total_km
        sub_d1 = d0 + (c1 / total_cols) * total_km

        self.img.setImage(sub, autoLevels=False)
        self.img.setLevels([0.0, self._vmax])
        if self._lut is not None:
            self.img.setLookupTable(self._lut)
        self.img.setRect(QRectF(sub_d0, t0, sub_d1 - sub_d0, t1 - t0))

    # ── Internals ───────────────────────────────────────────────────────────

    def _push_full_image(self) -> None:
        """Push the complete display buffer to ``ImageItem`` (initial render)."""
        arr = self._arr
        if arr is None:
            return
        d0, d1, t0, t1 = self._rect
        self.img.setImage(arr, autoLevels=False)
        self.img.setLevels([0.0, self._vmax])
        if self._lut is not None:
            self.img.setLookupTable(self._lut)
        self.img.setRect(QRectF(float(d0), float(t0),
                                float(d1) - float(d0),
                                float(t1) - float(t0)))
        self._last_zoom_key = None

    def _colormap(self) -> pg.ColorMap:
        try:
            return pg.colormap.get(self._cmap_name, source="matplotlib")
        except Exception:
            return pg.colormap.get("viridis", source="matplotlib")

    def _build_lut(self) -> np.ndarray:
        """Return a (256, 3) uint8 LUT for the current colormap."""
        return self._colormap().getLookupTable(0.0, 1.0, 256, alpha=False)

    def _update_colorbar(self) -> None:
        if self._cbar is not None:
            self.glw.removeItem(self._cbar)
            self._cbar = None
        cm = self._colormap()
        self._lut = cm.getLookupTable(0.0, 1.0, 256, alpha=False)
        self._cbar = pg.ColorBarItem(
            values=(0.0, self._vmax),
            colorMap=cm,
            label=self.tr("Amplitude"))
        self.glw.addItem(self._cbar, 0, 1)

    def _draw_overlays(self, boundaries: Sequence[float],
                       fixes: Sequence[FixMark]) -> None:
        for ln in self._overlay_lines:
            self.plot.removeItem(ln)
        self._overlay_lines.clear()
        self._boundary_lines.clear()

        pen_b = pg.mkPen(theme.color("warn"), width=1, style=Qt.PenStyle.DashLine)
        for x in boundaries:
            ln = pg.InfiniteLine(pos=float(x), angle=90, pen=pen_b)
            ln.setVisible(self._boundaries_visible)   # honour the live toggle
            self.plot.addItem(ln)
            self._overlay_lines.append(ln)
            self._boundary_lines.append(ln)

        pen_f = pg.mkPen(theme.color("highlight"), width=1, style=Qt.PenStyle.DashLine)
        for num, dist, label in fixes:
            ln = pg.InfiniteLine(
                pos=float(dist), angle=90, pen=pen_f,
                label=f"{num}·{label}",
                labelOpts={"color": theme.color("highlight"), "position": 0.96,
                           "rotateAxis": (1, 0)})
            self.plot.addItem(ln)
            self._overlay_lines.append(ln)

    def _restyle(self, *_) -> None:
        self.glw.setBackground(theme.color("panel"))
        for ax in ("left", "bottom"):
            axis = self.plot.getAxis(ax)
            axis.setPen(theme.color("sub"))
            axis.setTextPen(theme.color("text"))
            axis.setStyle(tickFont=QFont(MONO, 8))
        if self._cbar is not None:
            self._update_colorbar()
            if self._arr is not None:
                self._push_full_image()

    def _retranslate(self, *_) -> None:
        self.plot.setLabel("bottom", self.tr("Distance (km)"))
        self.plot.setLabel("left", self.tr("Time (ms)"))
        if self._cbar is not None:
            self._cbar.setLabel("right", self.tr("Amplitude"))
