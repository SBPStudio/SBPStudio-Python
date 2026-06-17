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
from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal
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

    # Emitted (debounced) when the user pans/zooms while in preview mode. A
    # PreviewController listens and re-runs the DSP pipeline on the new window.
    view_range_changed = pyqtSignal()

    # Emitted IMMEDIATELY on every pan/zoom (no debounce) with the absolute
    # trace COLUMN indices [trace0, trace1) currently on screen. Resolved with
    # np.searchsorted against the per-trace distance axis. That axis is built in
    # the core parser from the CLEANED (median-filtered) navigation track, so it
    # is smooth and monotonically non-decreasing — searchsorted is exact and
    # O(log n). Explicit side='left'/'right' makes the half-open range correct
    # even across distance plateaus (a stationary vessel): the upper bound still
    # reaches the true end. The map slices its coordinate arrays with these.
    visible_traces_changed = pyqtSignal(int, int)

    # Emitted when the user CLICKS (not drags) a point on the section, carrying
    # the absolute trace index under the cursor (resolved via searchsorted on the
    # distance axis). Drives the Header Inspector's row selection.
    trace_clicked = pyqtSignal(int)

    # Viewport-settle debounce: DSP recompute fires this long after the LAST
    # pan/zoom event, so dragging never triggers a pipeline run mid-gesture.
    SETTLE_MS = 300

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
        # Ephemeral high-quality render overlay (Render Viewport HQ): a pre-coloured
        # RGBA ImageItem laid exactly over the data area, auto-removed on any
        # pan/zoom so the fast PyQtGraph view returns seamlessly.
        self._hq_item: Optional[pg.ImageItem] = None
        self._hq_active: bool = False
        self._hq_view_key: Optional[tuple] = None
        self._overlay_lines: List[pg.InfiniteLine] = []
        self._boundary_lines: List[pg.InfiniteLine] = []  # toggled live
        self._boundaries_visible: bool = True

        # Full per-trace distance array (km) of the active profile/chain, set by
        # the controller. The ViewBox X-range is resolved to absolute trace
        # indices by masking THIS array directly (plateau-proof; see
        # _emit_visible_traces). None until a source is shown.
        self._dist_km: Optional[np.ndarray] = None

        # Display buffer state (set by show_image; used by _on_range_changed).
        self._arr: Optional[np.ndarray] = None   # (rows, cols) float32
        self._lut: Optional[np.ndarray] = None   # (256, 3) uint8
        self._vmax: float = 1.0
        self._cmap_name: str = "viridis"
        self._rect: tuple = (0.0, 1.0, 0.0, 1.0)   # (dist0, dist1, t0, t1)
        self._aspect: Optional[float] = None
        self._last_zoom_key: Optional[tuple] = None  # (c0, c1, stride) dedup
        # Preview mode: a PreviewController owns image updates; the internal
        # display-buffer re-slice is bypassed and range changes are forwarded
        # as view_range_changed instead.
        self._preview_mode: bool = False

        # Non-preview internal re-slice: light 60 ms throttle (fires 60 ms after
        # the FIRST event in a burst) — fine for the cheap display-buffer slice.
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(60)
        self._zoom_timer.timeout.connect(self._apply_zoom_update)
        self._pending_ranges: Optional[list] = None

        # Preview-mode SETTLE debounce: restarted on EVERY range-change event so
        # it only fires once the viewport has been still for SETTLE_MS. This
        # decouples the (expensive) DSP recompute from the live drag — while the
        # user pans, PyQtGraph natively moves the existing ImageItem buffer and
        # NO pipeline runs; the DSP recomputes only after the viewport settles.
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.setInterval(self.SETTLE_MS)
        self._settle_timer.timeout.connect(self.view_range_changed.emit)

        # Connect dynamic-zoom handler.
        self.plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)
        # Single-click (non-drag) → emit the trace index under the cursor.
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)

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
        self.clear_hq_overlay()     # a new section supersedes any HQ overlay

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
        # True for both paths: the static display buffer (_arr) and the
        # controller-driven preview (which sets only the ImageItem).
        return self._arr is not None or self.img.image is not None

    # ── Preview mode (driven by a PreviewController) ─────────────────────────

    def enable_preview(self, on: bool = True) -> None:
        """Hand image updates to an external PreviewController.

        In preview mode the internal display-buffer re-slice is bypassed and
        pan/zoom is forwarded as :pyattr:`view_range_changed` so the controller
        can re-run the DSP pipeline on the newly visible window.
        """
        self._preview_mode = bool(on)

    def current_view_range(self) -> tuple:
        """Return ((x0_km, x1_km), (y0_ms, y1_ms)) of the current ViewBox."""
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        return (float(x0), float(x1)), (float(y0), float(y1))

    # ── Ephemeral HQ render overlay ──────────────────────────────────────────

    def show_hq_overlay(self, rgba: np.ndarray,
                        x0: float, x1: float, t0: float, t1: float) -> None:
        """Lay a pre-coloured RGBA raster (the HQ Matplotlib-quality crop) exactly
        over the data area, mapped to the crop's km × ms bounds with the SAME
        ``setRect`` convention as the live image so it covers it without scaling
        distortion. Sits above the standard view and is auto-removed on any
        pan/zoom (see :meth:`_on_range_changed`)."""
        if self._hq_item is None:
            self._hq_item = pg.ImageItem()
            self._hq_item.setZValue(20)            # above the live image + overlays
            self.plot.addItem(self._hq_item)
        # autoLevels=False: the array is already RGBA uint8, no LUT/levels needed.
        self._hq_item.setImage(rgba, autoLevels=False)
        lo_t, hi_t = (t0, t1) if t1 >= t0 else (t1, t0)
        self._hq_item.setRect(QRectF(float(x0), float(lo_t),
                                     float(x1) - float(x0), float(hi_t) - float(lo_t)))
        self._hq_item.setVisible(True)
        self._hq_active = True
        # Remember the view it was rendered for, so the very next genuine pan/zoom
        # (a DIFFERENT range) removes it — but merely placing it does not.
        (vx0, vx1), (vy0, vy1) = self.plot.getViewBox().viewRange()
        self._hq_view_key = (round(vx0, 6), round(vx1, 6), round(vy0, 6), round(vy1, 6))

    def clear_hq_overlay(self) -> None:
        """Remove the HQ overlay from the scene, reverting to the fast PyQtGraph view."""
        if self._hq_item is not None:
            self.plot.removeItem(self._hq_item)
            self._hq_item = None
        self._hq_active = False
        self._hq_view_key = None

    def has_hq_overlay(self) -> bool:
        return self._hq_active

    def center_on_distance(self, km: float) -> None:
        """Scroll the ViewBox horizontally to centre on a distance (km), keeping
        the current zoom width. Used by the map's click-to-jump."""
        vb = self.plot.getViewBox()
        (x0, x1), _ = vb.viewRange()
        half = (x1 - x0) / 2.0
        vb.setXRange(km - half, km + half, padding=0)

    def _on_scene_click(self, ev) -> None:
        """Map a single left-click to the trace index under the cursor and emit
        it. pyqtgraph fires sigMouseClicked only for clicks (drags pan the view),
        so this never interferes with panning."""
        dist = self._dist_km
        if dist is None or dist.size < 1:
            return
        try:
            if ev.button() != Qt.MouseButton.LeftButton:
                return
            x = float(self.plot.getViewBox().mapSceneToView(ev.scenePos()).x())
        except (AttributeError, TypeError):
            return
        idx = int(np.searchsorted(dist, x, side="left"))
        idx = max(0, min(idx, dist.size - 1))
        self.trace_clicked.emit(idx)

    def set_colormap(self, cmap_name: str, vmax: float) -> None:
        """Set the LUT + colorbar for preview updates (cheap; no image reset)."""
        self._cmap_name = cmap_name
        self._vmax = float(vmax) or 1.0
        self._lut = self._build_lut()
        if self._cbar is None:
            self._cbar = pg.ColorBarItem(values=(0.0, self._vmax),
                                         colorMap=self._colormap(),
                                         label=self.tr("Amplitude"))
            self.glw.addItem(self._cbar, 0, 1)
        else:
            self._cbar.setColorMap(self._colormap())
            self._cbar.setLevels((0.0, self._vmax))

    def show_preview(self, arr: np.ndarray, dist0: float, dist1: float,
                     t0: float, t1: float, *, vmax: float,
                     fit: bool = False) -> None:
        """Lean image update for the live preview — NO autoRange unless ``fit``.

        ``set_colormap`` must have been called first (LUT ready). On ``fit`` the
        view is auto-ranged once (initial display); subsequent pan/zoom-driven
        previews leave the user's viewport untouched.
        """
        self.clear_hq_overlay()   # a fresh preview supersedes any HQ overlay
        self._vmax = float(vmax) or 1.0
        self._rect = (float(dist0), float(dist1), float(t0), float(t1))
        self.img.setImage(arr, autoLevels=False)
        self.img.setLevels([0.0, self._vmax])
        if self._lut is not None:
            self.img.setLookupTable(self._lut)
        self.img.setRect(QRectF(float(dist0), float(t0),
                                float(dist1) - float(dist0),
                                float(t1) - float(t0)))
        if self._cbar is not None:
            self._cbar.setLevels((0.0, self._vmax))
        if fit:
            self.set_aspect(self._aspect)   # autoRanges to fit the new section

    def set_overlays(self, boundaries: Sequence[float] = (),
                     fixes: Sequence[FixMark] = ()) -> None:
        """Public hook so the controller can (re)draw FIX/boundary lines."""
        self._draw_overlays(boundaries, fixes)

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

    def set_distance_axis(self, dist_km) -> None:
        """Give the view the full per-trace distance array (km) of the active
        source (called by the controller). The ViewBox X-range is resolved to
        absolute trace indices by masking this array directly. Re-broadcasts the
        visible range for the present ViewBox so the map updates on new data."""
        self._dist_km = (np.asarray(dist_km, dtype=float)
                         if dist_km is not None else None)
        self._emit_visible_traces(self.plot.getViewBox().viewRange())

    def _emit_visible_traces(self, ranges) -> None:
        """Resolve the ViewBox X-range to absolute trace indices and broadcast.

        The distance axis (built in core from the cleaned navigation track) is
        monotonically non-decreasing, so ``np.searchsorted`` maps the X-range to
        the half-open trace window [t0, t1) in O(log n). ``side='left'`` for the
        lower bound and ``side='right'`` for the upper bound make the window
        correct across distance plateaus — the upper bound steps PAST a frozen-
        GPS plateau rather than stopping at its start, so the far edge reaches
        the true end."""
        dist = self._dist_km
        if dist is None or dist.size < 1:
            return
        try:
            xv0, xv1 = float(ranges[0][0]), float(ranges[0][1])
        except (TypeError, IndexError):
            return
        x_min, x_max = (xv0, xv1) if xv0 <= xv1 else (xv1, xv0)
        n = dist.size
        t0 = int(np.searchsorted(dist, x_min, side="left"))
        t1 = int(np.searchsorted(dist, x_max, side="right"))
        t0 = max(0, min(t0, n - 1))
        t1 = max(t0 + 1, min(t1, n))
        self.visible_traces_changed.emit(t0, t1)

    def _on_range_changed(self, _vb, ranges) -> None:
        """Schedule a display-buffer re-slice (or a preview refresh) on pan/zoom."""
        # Ephemeral HQ overlay: any GENUINE view change (a range different from the
        # one it was rendered for) removes the overlay and reverts to the fast view.
        # Merely placing the overlay does not change the range, so it survives until
        # the user actually pans/zooms.
        if self._hq_active:
            key = (round(ranges[0][0], 6), round(ranges[0][1], 6),
                   round(ranges[1][0], 6), round(ranges[1][1], 6))
            if key != self._hq_view_key:
                self.clear_hq_overlay()
        # Immediate (un-debounced) trace-index broadcast for the navigation map.
        self._emit_visible_traces(ranges)
        if self._preview_mode:
            # SETTLE debounce: restart on every event so the DSP recompute only
            # fires once the viewport stops moving. During the drag PyQtGraph
            # natively pans the existing ImageItem — no pipeline runs.
            self._settle_timer.start()
            return
        if self._arr is None:
            return
        self._pending_ranges = ranges
        if not self._zoom_timer.isActive():
            self._zoom_timer.start()

    def _apply_zoom_update(self) -> None:
        """Slice the display buffer to the visible column window and push to GPU
        (non-preview path only; preview uses the settle timer → view_range_changed)."""
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
