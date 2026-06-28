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
from PyQt6.QtGui import QAction, QFont, QPainterPath
from PyQt6.QtWidgets import QInputDialog, QMenu, QVBoxLayout, QWidget

from ..i18n import language_manager
from ..theme import MONO, theme

# ── Global PyQtGraph config ───────────────────────────────────────────────────
# row-major: arr[row, col] → row = Y (time), col = X (distance).
pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("antialias", False)
pg.setConfigOption("useOpenGL", True)   # hardware-accelerated pan/zoom

# FIX mark tuple: (number, distance_km, label)
FixMark = Tuple[int, float, str]


class _PickScatterPlotItem(pg.ScatterPlotItem):
    """ScatterPlotItem variant that ALSO intercepts right-clicks on its
    points — the base class's ``mouseClickEvent`` only ever handles
    ``Qt.MouseButton.LeftButton`` (every other button is explicitly
    ``ev.ignore()``'d, see pyqtgraph's own source), which would let a
    right-click on a marker bubble straight through to the ViewBox's own
    context menu instead of this marker's Edit/Delete menu. Removing the
    button check (and forwarding whichever button matched) is the entire
    change — clicks that miss every point still ``ev.ignore()`` exactly as
    before, so the normal Measure/Hide Ruler/Add Anomaly menu is untouched
    everywhere else on the section."""

    def mouseClickEvent(self, ev) -> None:
        pts = self.pointsAt(ev.pos())
        if len(pts) > 0:
            self.ptsClicked = pts
            ev.accept()
            self.sigClicked.emit(self, self.ptsClicked, ev)
        else:
            ev.ignore()


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

    # Emitted by the right-click "Add Anomaly to Map" menu action, carrying the
    # trace index under the cursor at the time of the right-click. Drives the
    # "Anomaly Waypoint" cross-module feature: when Link Views is active, the
    # tab adds a POI marker to the map at this trace. Deliberately an EXPLICIT
    # menu action rather than a double-click — a double-click is too easy to
    # trigger by accident while zooming/measuring.
    add_anomaly_requested = pyqtSignal(int)

    # Emitted on EVERY mouse move over the section (no debounce — mirrors the
    # ruler's own live readout), carrying the trace index under the cursor.
    # Only consumed when Link Views is active; drives the map's live
    # "Navigation Marker". Cheap: same searchsorted lookup as a click.
    cursor_trace_changed = pyqtSignal(int)

    # Emitted while the user DRAGS the A/B Compare divider (the wiper),
    # carrying its new X position (km). The PreviewController recomposes the
    # raw|processed split from CACHED arrays — no DSP pipeline rerun — so the
    # wipe is fluid. sigDragged fires only on user drag, never on the
    # controller's own programmatic setPos (pan/zoom), so there's no feedback
    # loop. See update_ab / _on_ab_divider_dragged.
    ab_split_changed = pyqtSignal(float)

    # Viewport-settle debounce: DSP recompute fires this long after the LAST
    # pan/zoom event, so dragging never triggers a pipeline run mid-gesture.
    SETTLE_MS = 300

    # Max visible traces drawn as live wiggle before auto-falling back to the
    # raster base (keeps pan/zoom fluid). Mirrors the export budget
    # ``viz.render.WIGGLE_MAX_TRACES``; the PreviewController reads this as the
    # guard threshold so the budget lives with the view.
    WIGGLE_BUDGET = 1200

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.plot = self.glw.addPlot(row=0, col=0)
        # Drop PlotItem's own "Plot Options" submenu (grid/log-scale clutter not
        # relevant here) but KEEP the ViewBox's own menu (View All / Mouse Mode /
        # Export...) — the Measure / Hide Ruler actions below are injected into
        # that same ViewBox menu, so right-click still offers pyqtgraph's normal
        # zoom/export entries alongside them.
        self.plot.setMenuEnabled(False, enableViewBoxMenu=True)
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
        # Batched wiggle/VA overlay: ONE PlotCurveItem for all visible trace lines
        # (NaN-separated segments, connect='finite') → a single GPU draw call, plus
        # ONE filled path item for the variable-area lobes (also batched).
        self._wiggle_item: Optional[pg.PlotCurveItem] = None
        self._va_item = None    # QGraphicsPathItem (lazy) — variable-area fill
        self._overlay_lines: List[pg.InfiniteLine] = []
        self._boundary_lines: List[pg.InfiniteLine] = []  # toggled live
        # Parallel to _boundary_lines: each seam's TRUE trace-index position
        # (resolved once via searchsorted against self._dist_km when the
        # lines are (re)built — see _draw_overlays), so _reposition_boundaries
        # can keep them glued to the image's own per-window linear
        # approximation exactly like picks (_pick_x_km) instead of sitting at
        # a fixed/raw km value that mismatches the actual seam pixel
        # whenever the window's local stretch diverges from the chain's
        # true (non-uniform) distance axis — the same bug picks had.
        self._boundary_trace_indices: List[int] = []
        self._boundaries_visible: bool = True

        # A/B Compare overlay: a vertical divider + "A (raw)" / "B (filtered)"
        # labels drawn over the raw|processed composite raster (the controller
        # composites the image; this view only draws the marker). Lazily
        # created on first activation, then shown/hidden — see update_ab.
        self._ab_divider = None
        self._ab_label_a = None
        self._ab_label_b = None

        # Full per-trace distance array (km) of the active profile/chain, set by
        # the controller. The ViewBox X-range is resolved to absolute trace
        # indices by masking THIS array directly (plateau-proof; see
        # _emit_visible_traces). None until a source is shown.
        self._dist_km: Optional[np.ndarray] = None

        # Interactive ruler (Measure tool): a pg.LineSegmentROI with two
        # free-dragging handles, plus a pg.TextItem that tracks its midpoint
        # showing the live ΔX/ΔT readout. Both are None while inactive (see
        # _set_ruler_active) so an idle ruler holds no scene items. Activated
        # via the ViewBox right-click menu (self._measure_action /
        # self._hide_ruler_action below), not a toolbar button.
        self._ruler_roi: Optional[pg.LineSegmentROI] = None
        self._ruler_label: Optional[pg.TextItem] = None

        # Interpretation & Picking (Phase 3): double-click (while active)
        # drops a numbered marker + a free-text description, drawn as ONE
        # shared ScatterPlotItem (efficient: a single GPU draw call however
        # many points exist) plus one pg.TextItem per point for its ID label
        # (pyqtgraph has no batched text item). ``_next_pick_id`` is a
        # monotonically increasing counter that is NEVER reset by toggling
        # picking on/off, nor by deleting a point — only a brand-new
        # SeismicView instance (i.e. app restart) starts it back at 1.
        self._pick_mode: bool = False
        self._picks: List["PickPoint"] = []
        self._next_pick_id: int = 1
        self._pick_source = None   # SegyProfile/ProfileChain for coord lookup
        self._pick_scatter: Optional[_PickScatterPlotItem] = None
        self._pick_labels: dict = {}   # pick id -> pg.TextItem
        # Markers live directly in the ViewBox (self.plot.addItem — see
        # _ensure_pick_scatter), NOT parented to self.img. Their X position
        # is NOT a fixed dist_km[trace_index] lookup, though — the image
        # itself never renders a trace AT its true km position to begin
        # with. self.img's array is always a UNIFORM pixel resample
        # (scipy.ndimage.zoom / PIL, see viz.render._colorize_for_target,
        # and pyqtgraph's own ImageItem.setRect does the equivalent), with
        # no notion of dist_km's real (non-uniform — ship speed varies) per-
        # trace spacing: it stretches LINEARLY across whatever km window is
        # currently shown. A pick must therefore use the SAME linear
        # interpolation within the CURRENT window's trace-index bounds
        # (_img_trace_lo/_img_trace_hi, set by show_preview/_apply_zoom_
        # update/_push_full_image) to land on the pixel its data actually
        # occupies — see _redraw_picks's docstring for the full derivation,
        # and viz.render._draw_picks for the identical fix on the export
        # side (where there's only ever one "window" — the whole exported
        # raster — so no cross-zoom drift question arises there).
        self._img_trace_lo: Optional[int] = None
        self._img_trace_hi: Optional[int] = None
        # The km span the CURRENTLY DISPLAYED image actually covers — tracked
        # SEPARATELY from self._rect, which (for the static show_image/
        # _apply_zoom_update path) keeps meaning "the full dataset's extent"
        # even while _apply_zoom_update's re-slice shows only a sub-window
        # of it (self._rect is still relied on elsewhere — e.g. set_aspect —
        # for that full-extent meaning, so it can't double as this).
        self._img_km_lo: Optional[float] = None
        self._img_km_hi: Optional[float] = None

        # Display buffer state (set by show_image; used by _on_range_changed).
        self._arr: Optional[np.ndarray] = None   # (rows, cols) float32
        self._lut: Optional[np.ndarray] = None   # (256, 3) uint8
        self._lut_dirty: bool = True             # True → upload LUT to GPU on next show_preview
        self._img_levels: Optional[tuple] = None # (vmin, vmax) last applied to ImageItem
        self._vmax: float = 1.0
        self._vmin: float = 0.0                  # 0 (sequential) or −vmax (diverging)
        self._cmap_name: str = "viridis"
        # Cached wiggle pen/brush — rebuilt only when color changes. Color is
        # dynamic by default (see _contrast_color): black/white chosen to
        # maximise contrast against whatever's showing underneath. An explicit
        # update_wiggle(color=...) pins it until the next update_wiggle call.
        self._wiggle_pen = None
        self._wiggle_pen_col: str = ""
        self._wiggle_brush = None
        self._wiggle_brush_col: str = ""
        self._wiggle_color_pinned: bool = False
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

        # Cross-module sync (Link Views) state: set by the owning tab whenever
        # the "Link Views" toggle changes (see set_link_views_enabled). Drives
        # whether the live hover marker/anomaly menu action are meaningful —
        # the signals themselves still fire either way; the TAB is what
        # actually no-ops while unchecked (same split as trace_clicked).
        self._link_views_enabled: bool = False
        self._last_hover_idx: Optional[int] = None

        # Connect dynamic-zoom handler.
        self.plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)
        # Single-click (non-drag) → emit the trace index under the cursor.
        self.plot.scene().sigMouseClicked.connect(self._on_scene_click)
        # Live hover → cursor_trace_changed (cross-module navigation cursor).
        self.plot.scene().sigMouseMoved.connect(self._on_scene_hover)

        # ── Ruler actions, replacing the ViewBox's own right-click menu
        # entirely (industry-standard "right-click to measure", no toolbar
        # button). pyqtgraph's stock entries (View All / X axis / Y axis /
        # Mouse Mode) are pure workspace clutter for this interpretation view,
        # so the menu is stripped down to ONLY Measure / Hide Ruler. We keep
        # the existing ViewBoxMenu INSTANCE (just empty it with .clear())
        # rather than swapping in a plain QMenu, because ViewBox internals
        # call menu.setViewList(...) whenever any ViewBox in the app is
        # registered/unregistered (see ViewBox.updateViewLists) — a plain
        # QMenu lacks that method and would crash the first time some other
        # tab's plot is created. clear() leaves that method intact while
        # removing every default action. ──────────────────────────────────
        vb = self.plot.getViewBox()
        # Keep pyqtgraph's stock corner "A" auto-range button. A custom
        # instant "Reset View" was tried here (both hand-rolled aspect-lock
        # math and the plain pg.ViewBox.autoRange() one-liner) but both fight
        # this view's OWN zoom-adaptive re-slicing: _apply_zoom_update shrinks
        # self.img's rect to whatever column range is currently visible (see
        # its docstring), so anything that fits to CURRENT item geometry
        # (autoRange included) ends up re-fitting to the last zoomed sub-
        # window instead of the true full dataset — confirmed reproducible,
        # not a one-off. Reverted; the stock button is the supported reset.
        self.plot.showButtons()
        self._measure_action = QAction(self)
        self._measure_action.triggered.connect(lambda: self._set_ruler_active(True))
        self._hide_ruler_action = QAction(self)
        self._hide_ruler_action.triggered.connect(lambda: self._set_ruler_active(False))
        # "Add Anomaly to Map" (Part 1): an EXPLICIT menu action rather than a
        # double-click — see add_anomaly_requested's docstring. Uses the last
        # hovered trace (set on every mouse move, same index resolution as a
        # click) as "the trace under the cursor" at the moment of right-click.
        self._anomaly_action = QAction(self)
        self._anomaly_action.triggered.connect(self._on_add_anomaly_triggered)
        vb.menu.clear()
        vb.menu.addAction(self._measure_action)
        vb.menu.addAction(self._hide_ruler_action)
        vb.menu.addSeparator()
        vb.menu.addAction(self._anomaly_action)
        # Grey out "Hide Ruler"/"Add Anomaly" when there's nothing to act on.
        vb.menu.aboutToShow.connect(self._update_ruler_menu_state)

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
        """Lock the data box to a W:H ratio (None = free).

        STRICT vertical preservation: when an aspect is set we anchor the VERTICAL
        (time) extent to the full record and let the locked aspect derive the
        horizontal window. This pins ms-per-pixel, so changing the horizontal
        scale (traces/cm → a different aspect) compresses/spreads the traces
        horizontally WITHOUT ever rescaling time. ``autoRange`` is used only in the
        free (unlocked) case, since it would letterbox and shrink the vertical when
        the figure is wide."""
        self._aspect = aspect
        vb = self.plot.getViewBox()
        d0, d1, t0, t1 = self._rect
        x_ext, y_ext = (d1 - d0), (t1 - t0)
        if aspect and x_ext > 0 and y_ext > 0:
            vb.setAspectLocked(True, ratio=aspect * y_ext / x_ext)
            lo_t, hi_t = (t0, t1) if t1 >= t0 else (t1, t0)
            vb.setYRange(lo_t, hi_t, padding=0)   # anchor time; X WIDTH follows the lock
            # The lock derives X's width from Y + the widget's pixel aspect, but
            # never recenters X's pan position — so loading a brand-new source
            # (a different distance domain) would otherwise stay parked over the
            # PREVIOUS file's old window. Recenter X on the new data's midpoint;
            # passing only xRange (no yRange) leaves the just-set Y range intact.
            (cur_x0, cur_x1), _ = vb.viewRange()
            half_width = (cur_x1 - cur_x0) / 2.0
            x_center = (d0 + d1) / 2.0
            vb.setXRange(x_center - half_width, x_center + half_width, padding=0)
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

    def current_levels(self) -> tuple:
        """Return (vmin, vmax) currently applied to the live image's colormap —
        the SAME amplitude bounds an HQ/Matplotlib render must reuse verbatim to
        avoid a clipping mismatch with what's on screen."""
        return self._vmin, self._vmax

    def set_image_interpolation(self, mode: str) -> None:
        """Toggle pixel-scaling smoothing for the live raster (and the HQ overlay)
        to approximate the Render/Export interpolation choice. PyQtGraph's
        ImageItem scales via QPainter.drawImage, which only offers an on/off
        smoothing hint (no distinct bicubic mode) — so 'bilinear' AND 'bicubic'
        both map to smooth=True here; the EXACT mode ('nearest'/'bilinear'/
        'bicubic') is still passed verbatim to Matplotlib for HQ/Export, which
        does support all three. This is a paint-time hint, not a data change,
        so it's cheap to flip on every toggle."""
        from PyQt6.QtGui import QPainter
        smooth = mode in ("bilinear", "bicubic")
        self.glw.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)

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

    # ── Live wiggle / variable-area overlay ──────────────────────────────────

    def within_wiggle_budget(self, count: int) -> bool:
        """Whether ``count`` visible traces is within the live wiggle budget.

        The PreviewController calls this BEFORE building a wiggle so an
        over-budget window auto-falls back to the raster base (see
        :meth:`disable_wiggle`). Owning the threshold here keeps the budget with
        the view that draws it."""
        return int(count) <= self.WIGGLE_BUDGET

    def update_wiggle(self, xs: np.ndarray, ys: np.ndarray,
                      xs_va: Optional[np.ndarray] = None,
                      ys_va: Optional[np.ndarray] = None, *,
                      show_raster: bool = True, va_fill: bool = True,
                      show_line: bool = True,
                      color: Optional[str] = None) -> None:
        """Draw the batched wiggle line (``xs``/``ys``) and, when ``va_fill`` and
        ``xs_va``/``ys_va`` are given, the variable-area fill (one batched filled
        path), then set the raster base layer's visibility.

        ``show_raster`` True keeps the density raster underneath (Raster +
        Wiggle); False hides it (Wiggle Only). The raster is only HIDDEN, never
        cleared, so a later transition re-shows it instantly (flicker-free).

        ``show_line`` and ``va_fill`` are independent: either the wiggle line
        or the VA fill can be hidden while the other stays fully visible — the
        line is a separate ``PlotCurveItem`` from the VA fill's
        ``QGraphicsPathItem``, so toggling one's visibility never touches the
        other's geometry or paint state.

        Wiggle line + VA fill render in a dynamically-chosen black/white for
        maximum contrast against whatever's showing underneath — the active
        colormap if the raster is visible, else the theme's panel background
        (see ``_contrast_color``). Pass ``color`` to pin an explicit override;
        it stays pinned (ignoring theme/colormap changes) until the next
        ``update_wiggle`` call, including one with ``color=None``."""
        self._wiggle_color_pinned = color is not None
        # Apply show_raster BEFORE picking the color: _contrast_color reads
        # the raster's CURRENT visibility to decide what it's contrasting against.
        self.img.setVisible(bool(show_raster))
        col = color if color is not None else self._contrast_color()

        # Rebuild pen/brush only when the color changes (so these objects are
        # constructed once and reused on every subsequent frame with the same
        # color). Flags are captured BEFORE updating so the item-level setPen
        # guard below can still branch on whether the pen actually changed.
        pen_changed   = col != self._wiggle_pen_col
        brush_changed = col != self._wiggle_brush_col
        if pen_changed:
            self._wiggle_pen     = pg.mkPen(col, width=1)
            self._wiggle_pen_col = col
        if brush_changed:
            self._wiggle_brush     = pg.mkBrush(col)
            self._wiggle_brush_col = col

        # ── Variable-area fill (below the line, above the raster) ──
        if va_fill and xs_va is not None and ys_va is not None:
            try:
                path = pg.arrayToQPath(np.asarray(xs_va, dtype=float),
                                       np.asarray(ys_va, dtype=float),
                                       connect="finite")
                if self._va_item is None:
                    from PyQt6.QtWidgets import QGraphicsPathItem
                    self._va_item = QGraphicsPathItem()
                    self._va_item.setPen(pg.mkPen(None))
                    self._va_item.setZValue(15)        # above raster, below line
                    # NO cache mode (default). DeviceCoordinateCache was tried
                    # here to blit a rasterized pixmap on pan instead of
                    # re-filling the path, but it caches the path's footprint
                    # in DEVICE pixels — when the ViewBox applies a fresh
                    # X/Y transform (zoom), Qt must regenerate that pixmap from
                    # scratch anyway, and during/just after an anisotropic
                    # (non-uniform X vs Y) zoom the regenerated cache visibly
                    # lagged the sibling wiggle line (a PlotCurveItem, which
                    # has no cache and always re-strokes in true vector data
                    # coordinates every frame) by a frame or more — the fill
                    # would detach from the line until the next repaint. The
                    # VA path is already capped at VA_MAX_ROWS, so re-filling
                    # it every frame in plain vector mode is cheap, and it
                    # keeps the fill pixel-exact with the line at all times.
                    self.plot.getViewBox().addItem(self._va_item)
                self._va_item.setBrush(self._wiggle_brush)
                self._va_item.setPath(path)
                self._va_item.setVisible(True)
            except Exception:                          # fill is best-effort
                if self._va_item is not None:
                    self._va_item.setVisible(False)
        elif self._va_item is not None:
            self._va_item.setVisible(False)

        # ── Wiggle line (on top of the fill) ──
        if self._wiggle_item is None:
            self._wiggle_item = pg.PlotCurveItem(
                connect="finite", antialias=True, pen=self._wiggle_pen)
            self._wiggle_item.setZValue(16)   # above the VA fill, below HQ (20)
            self.plot.addItem(self._wiggle_item)
        elif pen_changed:
            self._wiggle_item.setPen(self._wiggle_pen)
        self._wiggle_item.setData(xs, ys)
        self._wiggle_item.setVisible(bool(show_line))

    def set_wiggle_line_visible(self, visible: bool) -> None:
        """Show/hide the wiggle line only, independent of the VA fill and the
        raster — used by the geometry-unchanged fast path (only the toggle
        changed, not the underlying trace data)."""
        if self._wiggle_item is not None:
            self._wiggle_item.setVisible(bool(visible))

    def set_raster_visible(self, visible: bool) -> None:
        """Show/hide the raster base layer without touching wiggle geometry.

        Used by the PreviewController's geometry-unchanged fast path (only
        the colormap/show_raster toggle changed, not the wiggle shape itself)
        — ``update_wiggle`` handles its own ``show_raster`` directly since it
        already recomputes color on every call. Refreshes the dynamic wiggle/
        VA contrast color when the visibility actually flips, since what's
        showing underneath (colormap vs. bare theme background) just changed."""
        visible = bool(visible)
        changed = self.img.isVisible() != visible
        self.img.setVisible(visible)
        if changed:
            self._refresh_dynamic_color()

    def _contrast_color(self) -> str:
        """Black or white — whichever maximises contrast against whatever is
        currently showing underneath the wiggle line + VA fill: the perceived
        brightness of the active colormap's LUT when the raster is visible,
        otherwise the current theme's panel background color."""
        if self.img.isVisible() and self._lut is not None and self._lut.size:
            lut = self._lut.astype(np.float64)
            # A flat mean over all 256 rows is invariant to "Invert Colors"
            # (same palette, just reordered) and would never react to it.
            # Weight the low-amplitude end (row 0 = vmin) most heavily: most
            # of a seismic section sits near the quiet/background amplitude,
            # so that's the color the wiggle/VA actually overlays most of the
            # time — and this weighting correctly flips with the LUT's order.
            n = lut.shape[0]
            weights = np.linspace(1.0, 0.0, n)
            weights /= weights.sum()
            lum = 0.299 * lut[:, 0] + 0.587 * lut[:, 1] + 0.114 * lut[:, 2]
            avg_lum = float(np.dot(weights, lum) / 255.0)
        else:
            avg_lum = self._hex_luminance(theme.color("panel"))
        return "#000000" if avg_lum > 0.5 else "#ffffff"

    @staticmethod
    def _rgb_luminance(r: float, g: float, b: float) -> float:
        """Perceived (ITU-R BT.601) luminance of an 0..255 RGB triple, 0..1."""
        return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

    @classmethod
    def _hex_luminance(cls, hex_color: str) -> float:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return cls._rgb_luminance(r, g, b)

    def _refresh_dynamic_color(self) -> None:
        """Recompute and re-apply the wiggle/VA contrast color on the existing
        items with no geometry change. Called on theme switch, colormap
        switch, or a raster-visibility toggle that bypassed update_wiggle.
        No-op while a color is pinned via ``update_wiggle(color=...)``."""
        if self._wiggle_color_pinned:
            return
        col = self._contrast_color()
        pen_changed   = col != self._wiggle_pen_col
        brush_changed = col != self._wiggle_brush_col
        if pen_changed:
            self._wiggle_pen     = pg.mkPen(col, width=1)
            self._wiggle_pen_col = col
            if self._wiggle_item is not None:
                self._wiggle_item.setPen(self._wiggle_pen)
        if brush_changed:
            self._wiggle_brush     = pg.mkBrush(col)
            self._wiggle_brush_col = col
            if self._va_item is not None:
                self._va_item.setBrush(self._wiggle_brush)

    def disable_wiggle(self) -> None:
        """Hide the wiggle + VA fill and ensure the raster base is visible — the
        density state. The raster is kept ready underneath so the transition never
        flickers. Geometry is explicitly cleared so the NaN-separated float arrays
        and QPainterPath are garbage-collected immediately rather than held until
        the next wiggle draw."""
        if self._wiggle_item is not None:
            self._wiggle_item.setData(x=[], y=[])   # release NaN-separated float arrays
            self._wiggle_item.setVisible(False)
        if self._va_item is not None:
            self._va_item.setPath(QPainterPath())   # release QPainterPath geometry
            self._va_item.setVisible(False)
        self.img.setVisible(True)

    def center_on_distance(self, km: float) -> None:
        """Scroll the ViewBox horizontally to centre on a distance (km), keeping
        the current zoom width. Used by the map's click-to-jump."""
        vb = self.plot.getViewBox()
        (x0, x1), _ = vb.viewRange()
        half = (x1 - x0) / 2.0
        vb.setXRange(km - half, km + half, padding=0)

    # ── Interactive ruler (Measure tool) ──────────────────────────────────────

    def _update_ruler_menu_state(self) -> None:
        """Grey out 'Hide Ruler' when no ruler is active, and 'Add Anomaly to
        Map' when Link Views is off or there's no trace under the cursor
        (menu aboutToShow)."""
        self._hide_ruler_action.setEnabled(self._ruler_roi is not None)
        self._anomaly_action.setEnabled(
            self._link_views_enabled and self._last_hover_idx is not None)

    def _set_ruler_active(self, active: bool) -> None:
        """Create or fully tear down the ruler ROI + readout label.

        Driven by the ViewBox right-click menu's 'Measure' / 'Hide Ruler'
        actions (see __init__). The ViewBox's X axis is already real distance
        (km) and its Y axis is already real time (ms) — see module docstring /
        show_image — so the two handle positions can be read straight off the
        ROI with no extra trace-index/sample-index lookup. Tearing down on
        deactivation (rather than just hiding) frees the ROI's scene items
        immediately. 'Measure' on an already-active ruler replaces it with a
        fresh one centred on the current view."""
        if active:
            if self._ruler_roi is not None:
                self._set_ruler_active(False)
            (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            half_x = (x1 - x0) * 0.125
            p1, p2 = (cx - half_x, cy), (cx + half_x, cy)
            pen = pg.mkPen(theme.color("highlight"), width=2)
            self._ruler_roi = pg.LineSegmentROI([p1, p2], pen=pen)
            self.plot.addItem(self._ruler_roi)
            self._ruler_label = pg.TextItem(
                anchor=(0.5, 1.0), color=theme.color("highlight"))
            self.plot.addItem(self._ruler_label)
            self._ruler_roi.sigRegionChanged.connect(self._update_ruler_label)
            self._update_ruler_label()
        else:
            if self._ruler_roi is not None:
                self.plot.removeItem(self._ruler_roi)
                self._ruler_roi.deleteLater()
                self._ruler_roi = None
            if self._ruler_label is not None:
                self.plot.removeItem(self._ruler_label)
                self._ruler_label.deleteLater()
                self._ruler_label = None

    def _update_ruler_label(self, *_) -> None:
        """Recompute ΔX (m) / ΔT (ms) from the ROI's two handles and move the
        label to their midpoint. Reads handle positions via the scene (the
        same mapSceneToView idiom as _on_scene_click) so the result is
        correct regardless of the ROI's own local coordinate frame."""
        roi, label = self._ruler_roi, self._ruler_label
        if roi is None or label is None:
            return
        vb = self.plot.getViewBox()
        (_, scene_p1), (_, scene_p2) = roi.getSceneHandlePositions()
        p1 = vb.mapSceneToView(scene_p1)
        p2 = vb.mapSceneToView(scene_p2)
        dx_m = abs(p2.x() - p1.x()) * 1000.0   # km → m
        dt_ms = abs(p2.y() - p1.y())           # already ms
        label.setText(f"ΔX: {dx_m:,.1f} m | ΔT: {dt_ms:,.1f} ms")
        label.setPos((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)

    # ── Interpretation & Picking (Phase 3) ──────────────────────────────────

    def set_pick_mode(self, active: bool) -> None:
        """Turn picking on/off (driven by ProcessingControls.btn_toggle_picking).
        Existing markers are left exactly as they are — only NEW double-click
        creation is gated by this flag."""
        self._pick_mode = bool(active)

    def set_picking_source(self, obj) -> None:
        """Called by the owning tab whenever the active profile/chain changes
        (alongside set_distance_axis) — used ONLY to resolve x_coord/y_coord
        for a NEWLY created pick (core.resolve_pick_coords reads obj.lons/
        obj.lats). Re-rendering the SAME profile (a DSP tweak, a Render Full)
        never calls this with a different object, so existing picks survive
        that; switching to a genuinely DIFFERENT profile/chain does, and that
        is the one case where the on-screen markers are no longer meaningful
        (their trace_index would point into a different file's geometry) —
        so the visible list is cleared then, but the id counter is NOT reset
        (see __init__): ids stay unique for the lifetime of this widget."""
        if obj is not self._pick_source:
            self.clear_picks()
        self._pick_source = obj

    def get_picks(self) -> List["PickPoint"]:
        """A shallow copy — callers (export) must not mutate the live list."""
        return list(self._picks)

    def set_picks(self, picks: List["PickPoint"]) -> None:
        """Replace the current pick list wholesale (Import) and redraw.
        ``_next_pick_id`` is advanced past the highest imported id so newly
        created picks never collide with imported ones."""
        self._picks = list(picks)
        if self._picks:
            self._next_pick_id = max(self._next_pick_id,
                                     max(p.id for p in self._picks) + 1)
        self._redraw_picks()

    def clear_picks(self) -> None:
        """Remove every marker from the screen AND the data list (the id
        counter is untouched — see set_picking_source)."""
        self._picks = []
        self._redraw_picks()

    def _ensure_pick_scatter(self) -> _PickScatterPlotItem:
        if self._pick_scatter is None:
            self._pick_scatter = _PickScatterPlotItem(
                size=12, pen=pg.mkPen("#000000", width=1),
                brush=pg.mkBrush("#ffd54f"), symbol="o")
            self._pick_scatter.setZValue(60)   # above raster/wiggle, below A/B (70)
            self._pick_scatter.sigClicked.connect(self._on_pick_scatter_clicked)
            # Added directly to the ViewBox (NOT parented to self.img) and
            # positioned in absolute (km, ms) view coordinates — see
            # _redraw_picks's docstring for why. ignoreBounds=True matches
            # the A/B divider/labels' own convention: a handful of marker
            # glyphs must never skew autoRange()'s fit to the actual data.
            self.plot.addItem(self._pick_scatter, ignoreBounds=True)
        return self._pick_scatter

    def _pick_x_km(self, trace_index: int, n_traces: Optional[int]) -> float:
        """The km position a pick at ``trace_index`` must render at to land
        on the SAME pixel self.img's data actually occupies there.

        self.img's array is always a UNIFORM resample (pyqtgraph's
        ImageItem.setRect stretches it linearly across whatever km window
        is currently shown — see viz.render._colorize_for_target for the
        identical mechanism on the export side), with no notion of
        dist_km's real (non-uniform — ship speed varies) per-trace spacing.
        So trace_index's pixel position is the LINEAR FRACTION of its
        offset within the CURRENT window's trace-index bounds
        (self._img_trace_lo/_img_trace_hi, set by show_preview/
        _apply_zoom_update/_push_full_image), mapped into that SAME
        window's own km bounds (self._img_km_lo/_img_km_hi — tracked
        separately from self._rect, which keeps meaning "the full
        dataset's extent" even while a zoom re-slice shows only part of
        it) — NOT a direct dist_km[trace_index] lookup, which would
        decouple from the image whenever spacing is non-uniform (this was
        the actual reported bug: markers rendering far from their true
        geological location).

        This means a pick's RENDERED km position can shift slightly between
        different zoom windows (each window's own linear approximation of
        non-uniform spacing differs a little) — an intentional, unavoidable
        trade-off: it is what keeps the marker glued to the image's actual
        pixels in any GIVEN view, at the cost of not being a perfectly
        fixed absolute position across views. Falls back to a direct
        dist_km lookup when no window bounds are known yet (e.g. before the
        first preview/show_image call).
        """
        lo, hi = self._img_trace_lo, self._img_trace_hi
        d0, d1 = self._img_km_lo, self._img_km_hi
        if lo is not None and hi is not None and hi > lo and d0 is not None and d1 != d0:
            frac = (trace_index - lo) / (hi - lo)
            return d0 + frac * (d1 - d0)
        dist = self._dist_km
        if n_traces:
            return float(dist[trace_index])
        return float(trace_index)

    def _redraw_picks(self) -> None:
        """Rebuild the shared scatter's spots + every ID TextItem from
        ``self._picks``. Both the scatter and every label live directly in
        the ViewBox (self.plot.addItem), never parented to self.img.

        Called whenever the pick LIST changes (add/edit/delete/import/
        clear) AND whenever the CURRENT zoom window changes
        (show_preview/_apply_zoom_update/_push_full_image) — unlike a fixed
        absolute position, the window-relative X computed by _pick_x_km
        needs re-evaluating every time that window's own linear
        approximation shifts. See _pick_x_km's docstring for the full
        derivation. Y is the pick's own time_ms verbatim — the sample
        interval is constant by construction, so it never has this problem.
        """
        for label in self._pick_labels.values():
            self.plot.removeItem(label)
            label.deleteLater()
        self._pick_labels.clear()

        if not self._picks:
            if self._pick_scatter is not None:
                self._pick_scatter.setData([])
            return

        scatter = self._ensure_pick_scatter()
        dist = self._dist_km
        n_traces = dist.size if dist is not None else None
        spots = []
        for p in self._picks:
            i = p.trace_index
            if n_traces:   # clamp defensively (e.g. an imported session from a longer line)
                i = max(0, min(i, n_traces - 1))
            x_km = self._pick_x_km(i, n_traces)
            y_ms = p.time_ms
            spots.append({"pos": (x_km, y_ms), "data": p.id})
            label = pg.TextItem(str(p.id), anchor=(0.5, 1.2),
                                color=theme.color("highlight"))
            label.setZValue(61)
            self.plot.addItem(label, ignoreBounds=True)
            label.setPos(x_km, y_ms)
            self._pick_labels[p.id] = label
        scatter.setData(spots)

    def _reposition_boundaries(self) -> None:
        """Keep file-boundary seam lines glued to the image's own current
        per-window linear approximation, exactly like picks (_pick_x_km) —
        called whenever the displayed window changes (show_preview/
        _apply_zoom_update/_push_full_image), not just when the boundary
        list itself is rebuilt (_draw_overlays)."""
        if not self._boundary_lines:
            return
        n_traces = self._dist_km.size if self._dist_km is not None else None
        for ln, trace_idx in zip(self._boundary_lines, self._boundary_trace_indices):
            ln.setPos(self._pick_x_km(trace_idx, n_traces))

    def _create_pick_at(self, scene_pos, trace_index: int) -> None:
        """Double-click handler: resolve the grid position, ask for a
        description, assign the next id, append + draw. Cancelling the
        QInputDialog creates nothing (no id is consumed for a cancelled pick
        — the counter only advances on an actual created point)."""
        from sbp_studio.core import PickPoint, resolve_pick_coords
        try:
            time_ms = float(self.plot.getViewBox().mapSceneToView(scene_pos).y())
        except (AttributeError, TypeError):
            return
        text, ok = QInputDialog.getText(
            self, self.tr("New interpretation marker"),
            self.tr("Name or brief description"))
        if not ok:
            return
        x_coord, y_coord = resolve_pick_coords(self._pick_source, trace_index)
        pick = PickPoint(id=self._next_pick_id, trace_index=trace_index,
                         time_ms=time_ms, x_coord=x_coord, y_coord=y_coord,
                         description=text.strip())
        self._next_pick_id += 1
        self._picks.append(pick)
        self._redraw_picks()

    def _on_pick_scatter_clicked(self, _plot, points, ev) -> None:
        """Only RIGHT-clicks open the Edit/Delete menu (a left-click on a
        marker is a no-op here — see _PickScatterPlotItem's docstring for
        why right-clicks reach this handler at all)."""
        try:
            if ev.button() != Qt.MouseButton.RightButton or not points:
                return
        except AttributeError:
            return
        pick_id = points[0].data()
        pick = next((p for p in self._picks if p.id == pick_id), None)
        if pick is None:
            return
        menu = QMenu(self)
        act_edit = menu.addAction(self.tr("Edit description"))
        act_delete = menu.addAction(self.tr("Delete marker"))
        chosen = menu.exec(ev.screenPos().toPoint())
        if chosen is act_edit:
            self._edit_pick_description(pick)
        elif chosen is act_delete:
            self._delete_pick(pick)

    def _edit_pick_description(self, pick: "PickPoint") -> None:
        text, ok = QInputDialog.getText(
            self, self.tr("Edit marker"),
            self.tr("Name or brief description"), text=pick.description)
        if ok:
            pick.description = text.strip()
            self._redraw_picks()

    def _delete_pick(self, pick: "PickPoint") -> None:
        self._picks = [p for p in self._picks if p.id != pick.id]
        self._redraw_picks()

    def _on_scene_click(self, ev) -> None:
        """Map a single left-click to the trace index under the cursor and
        emit it. pyqtgraph fires sigMouseClicked only for clicks (drags pan
        the view), so this never interferes with panning.

        Also the picking entry point: while picking mode is active, a
        DOUBLE left-click places a new marker instead of (not in addition
        to) emitting trace_clicked. ``ev.isAccepted()`` is checked first —
        GraphicsScene.sendClickEvent ALWAYS emits sigMouseClicked, even for
        clicks an item (e.g. an existing pick marker) already accepted for
        itself — so a double-click that lands ON an existing marker is
        correctly ignored here rather than stacking a second point on it."""
        idx = self._index_at_scene_pos(ev.scenePos())
        if idx is None:
            return
        try:
            if ev.button() != Qt.MouseButton.LeftButton:
                return
        except AttributeError:
            return
        if self._pick_mode and ev.double() and not ev.isAccepted():
            # NOT the same idx resolved above: _index_at_scene_pos finds the
            # trace whose TRUE dist_km is closest to the clicked km, but
            # _pick_x_km renders a pick via the CURRENT window's linear
            # approximation, not a true dist_km lookup — using idx here
            # would create a pick that immediately "snaps" away from the
            # exact pixel just clicked whenever spacing is non-uniform. See
            # _picking_trace_index_at_scene_pos's docstring.
            pick_idx = self._picking_trace_index_at_scene_pos(ev.scenePos())
            if pick_idx is not None:
                self._create_pick_at(ev.scenePos(), pick_idx)
            return
        self.trace_clicked.emit(idx)

    def _picking_trace_index_at_scene_pos(self, scene_pos) -> Optional[int]:
        """Resolve a scene position to a trace_index for PICKING
        specifically — the exact inverse of ``_pick_x_km``'s window-relative
        linear interpolation, NOT ``_index_at_scene_pos``'s dist_km-
        searchsorted resolution (used by hover/single-click/the ruler).

        This is what guarantees a newly created pick renders at EXACTLY the
        pixel it was clicked on, with zero snap, even when the line's real
        trace spacing is non-uniform: _redraw_picks will place this same
        trace_index via the identical forward mapping, so click → render is
        an exact round trip. Falls back to ``_index_at_scene_pos`` when the
        current window's trace-index bounds aren't known yet (e.g. before
        the first preview/show_image call)."""
        try:
            x_km = float(self.plot.getViewBox().mapSceneToView(scene_pos).x())
        except (AttributeError, TypeError):
            return None
        lo, hi = self._img_trace_lo, self._img_trace_hi
        d0, d1 = self._img_km_lo, self._img_km_hi
        n_traces = self._dist_km.size if self._dist_km is not None else None
        if lo is not None and hi is not None and hi > lo and d0 is not None and d1 != d0:
            frac = (x_km - d0) / (d1 - d0)
            i = int(round(lo + frac * (hi - lo)))
        else:
            i = self._index_at_scene_pos(scene_pos)
            if i is None:
                return None
        if n_traces:
            i = max(0, min(i, n_traces - 1))
        return max(0, i)

    def set_link_views_enabled(self, enabled: bool) -> None:
        """Called by the owning tab whenever the "Link Views" toggle changes
        (see _base.py). Only gates the "Add Anomaly to Map" menu action's
        enabled state — the hover/click signals themselves always fire; the
        tab is what actually no-ops them while unchecked."""
        self._link_views_enabled = bool(enabled)

    def _on_add_anomaly_triggered(self) -> None:
        if self._last_hover_idx is not None:
            self.add_anomaly_requested.emit(self._last_hover_idx)

    def _on_scene_hover(self, scene_pos) -> None:
        """Live hover → cursor_trace_changed. Fires on every mouse move over
        the scene; cheap (one searchsorted call), mirrors the ruler's own
        live-readout idiom. Also remembers the index for "Add Anomaly to
        Map" (the right-click menu has no cursor position of its own)."""
        idx = self._index_at_scene_pos(scene_pos)
        self._last_hover_idx = idx
        if idx is not None:
            self.cursor_trace_changed.emit(idx)

    def _index_at_scene_pos(self, scene_pos) -> Optional[int]:
        """Shared scene-position → absolute trace index resolution, used by
        both click and hover handlers."""
        dist = self._dist_km
        if dist is None or dist.size < 1:
            return None
        try:
            x = float(self.plot.getViewBox().mapSceneToView(scene_pos).x())
        except (AttributeError, TypeError):
            return None
        idx = int(np.searchsorted(dist, x, side="left"))
        return max(0, min(idx, dist.size - 1))

    def set_colormap(self, cmap_name: str, vmax: float, vmin: float = 0.0) -> None:
        """Set the LUT + colorbar for preview updates (cheap; no image reset).

        ``vmin`` is 0 for the sequential (0..1, |amp|) range or −vmax for the
        diverging (−1..1, signed) range, so a diverging colormap centres on zero."""
        cmap_changed = cmap_name != self._cmap_name
        self._cmap_name = cmap_name
        self._vmax = float(vmax) or 1.0
        self._vmin = float(vmin)
        if cmap_changed:
            self._lut = self._build_lut()
            self._lut_dirty = True   # schedule GPU upload on next show_preview
            self._refresh_dynamic_color()  # new colormap → re-check wiggle contrast
        if self._cbar is None:
            self._cbar = pg.ColorBarItem(values=(self._vmin, self._vmax),
                                         colorMap=self._colormap(),
                                         label=self.tr("Amplitude"))
            self.glw.addItem(self._cbar, 0, 1)
        else:
            if cmap_changed:
                self._cbar.setColorMap(self._colormap())
            self._cbar.setLevels((self._vmin, self._vmax))

    def show_preview(self, arr: np.ndarray, dist0: float, dist1: float,
                     t0: float, t1: float, *, vmax: float, vmin: float = 0.0,
                     fit: bool = False,
                     c0: Optional[int] = None, c1: Optional[int] = None) -> None:
        """Lean image update for the live preview — NO autoRange unless ``fit``.

        ``set_colormap`` must have been called first (LUT ready). On ``fit`` the
        view is auto-ranged once (initial display); subsequent pan/zoom-driven
        previews leave the user's viewport untouched.

        ``c0``/``c1`` — the FULL-RESOLUTION trace-index bounds (half-open)
        this window's ``arr`` actually represents (e.g. PreviewController's
        ``win.c_vis0``/``win.c_vis1``) — used ONLY to keep interpretation
        picks glued to the image's own linear pixel-grid approximation (see
        ``_pick_x_km``'s docstring). ``None`` leaves any existing picks on
        their last-known position rather than guessing.
        """
        self.clear_hq_overlay()   # a fresh preview supersedes any HQ overlay
        vmax_f = float(vmax) or 1.0
        vmin_f = float(vmin)
        self._vmax = vmax_f
        self._vmin = vmin_f
        self._rect = (float(dist0), float(dist1), float(t0), float(t1))
        self._img_km_lo, self._img_km_hi = float(dist0), float(dist1)
        self.img.setImage(arr, autoLevels=False)
        new_levels = (vmin_f, vmax_f)
        if new_levels != self._img_levels:
            self.img.setLevels([vmin_f, vmax_f])
            self._img_levels = new_levels
        if self._lut_dirty and self._lut is not None:
            self.img.setLookupTable(self._lut)
            self._lut_dirty = False
        self.img.setRect(QRectF(float(dist0), float(t0),
                                float(dist1) - float(dist0),
                                float(t1) - float(t0)))
        if c0 is not None and c1 is not None:
            self._img_trace_lo = int(c0)
            self._img_trace_hi = int(c1)
        if fit:
            self.set_aspect(self._aspect)   # autoRanges to fit the new section
        if self._picks:
            self._redraw_picks()
        self._reposition_boundaries()

    def set_overlays(self, boundaries: Sequence[float] = (),
                     fixes: Sequence[FixMark] = ()) -> None:
        """Public hook so the controller can (re)draw FIX/boundary lines."""
        self._draw_overlays(boundaries, fixes)

    def update_ab(self, active: bool, split_km: float = 0.0,
                  t_top: float = 0.0, t_bot: float = 0.0,
                  x0: float = 0.0, x1: float = 0.0) -> None:
        """Draw / hide the A/B Compare divider (the draggable wiper) over the
        composite raster.

        The controller has already spliced raw|processed into the shown image;
        this only marks the boundary at ``split_km`` with a vertical DRAGGABLE
        line and the "A (raw)" / "B (filtered)" captions, and clamps the line
        to the visible window [x0, x1]. Lazily creates the items the first
        time A/B is engaged, then toggles their visibility. Dragging the line
        emits :pyattr:`ab_split_changed`; the controller recomposes the split
        from cached arrays (no DSP rerun)."""
        if not active:
            for it in (self._ab_divider, self._ab_label_a, self._ab_label_b):
                if it is not None:
                    it.setVisible(False)
            return

        if self._ab_divider is None:
            self._ab_divider = pg.InfiniteLine(
                angle=90, movable=True,        # ← the user can drag this wiper
                pen=pg.mkPen(theme.color("highlight"), width=2,
                             style=Qt.PenStyle.DashLine),
                hoverPen=pg.mkPen(theme.color("highlight"), width=3))
            self._ab_divider.setZValue(70)
            self._ab_divider.setCursor(Qt.CursorShape.SplitHCursor)
            # sigDragged fires ONLY for a user drag (not the programmatic
            # setPos below), so this can't feed back into the controller's
            # pan/zoom repositioning.
            self._ab_divider.sigDragged.connect(self._on_ab_divider_dragged)
            self.plot.addItem(self._ab_divider, ignoreBounds=True)
            self._ab_label_a = pg.TextItem(anchor=(1.0, 0.0),
                                           color=theme.color("highlight"))
            self._ab_label_b = pg.TextItem(anchor=(0.0, 0.0),
                                           color=theme.color("highlight"))
            for lb in (self._ab_label_a, self._ab_label_b):
                lb.setZValue(71)
                self.plot.addItem(lb, ignoreBounds=True)

        # Clamp the wiper to the visible window so it can't be dragged into the
        # axis margins where there is no data to reveal.
        if x1 > x0:
            self._ab_divider.setBounds((float(x0), float(x1)))
        self._ab_label_top = min(float(t_top), float(t_bot))   # cached for drag
        self._ab_divider.setPos(float(split_km))   # programmatic → no sigDragged
        self._ab_divider.setVisible(True)
        self._ab_label_a.setText(self.tr("A (raw)"))
        self._ab_label_b.setText(self.tr("B (filtered)"))
        self._position_ab_labels(float(split_km))
        self._ab_label_a.setVisible(True)
        self._ab_label_b.setVisible(True)

    def _position_ab_labels(self, split_km: float) -> None:
        """Pin the A/B captions to either side of the divider at the top."""
        y_top = getattr(self, "_ab_label_top", 0.0)
        if self._ab_label_a is not None:
            self._ab_label_a.setPos(split_km, y_top)
        if self._ab_label_b is not None:
            self._ab_label_b.setPos(split_km, y_top)

    def _on_ab_divider_dragged(self, line) -> None:
        """User dragged the wiper → move the captions with it and ask the
        controller to recompose the raw|processed split at the new position."""
        x = float(line.value())
        self._position_ab_labels(x)
        self.ab_split_changed.emit(x)

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
        # The re-slice's column window IS this view's CURRENT linear
        # approximation — picks must track it (see _pick_x_km's docstring).
        self._img_trace_lo, self._img_trace_hi = c0, c1
        self._img_km_lo, self._img_km_hi = sub_d0, sub_d1
        if self._picks:
            self._redraw_picks()
        self._reposition_boundaries()

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
        # Full buffer, 1:1 — the window IS the whole trace range.
        self._img_trace_lo, self._img_trace_hi = 0, arr.shape[1]
        self._img_km_lo, self._img_km_hi = float(d0), float(d1)
        self._last_zoom_key = None
        if self._picks:
            self._redraw_picks()
        self._reposition_boundaries()

    def _colormap(self) -> pg.ColorMap:
        try:
            return pg.colormap.get(self._cmap_name, source="matplotlib")
        except Exception:
            return pg.colormap.get("viridis", source="matplotlib")

    def _build_lut(self) -> np.ndarray:
        """Return a (256, 3) uint8 LUT for the current colormap."""
        return self._colormap().getLookupTable(0.0, 1.0, 256, alpha=False)

    def _update_colorbar(self) -> None:
        old_cbar = self._cbar
        if old_cbar is not None:
            self.glw.removeItem(old_cbar)
            old_cbar.setParentItem(None)  # detach from scene so Qt can reap the C++ object
            old_cbar.deleteLater()
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
            ln.deleteLater()  # destroy C++ InfiniteLine + its child TextItem label
        self._overlay_lines.clear()
        self._boundary_lines.clear()
        self._boundary_trace_indices.clear()

        # boundaries are RAW km values (e.g. ProfileChain.boundaries_km).
        # Resolve each to its TRUE trace-index position once here (via
        # searchsorted against self._dist_km) and position the line via
        # _pick_x_km — the SAME window-relative linear interpolation picks
        # use — rather than the raw km value directly, so the seam stays
        # glued to the actual column boundary the image renders, not an
        # approximate absolute position (see _boundary_trace_indices'
        # comment in __init__ for the full story).
        dist = self._dist_km
        n_traces = dist.size if dist is not None else None
        pen_b = pg.mkPen(theme.color("warn"), width=1, style=Qt.PenStyle.DashLine)
        for x in boundaries:
            trace_idx = int(np.searchsorted(dist, float(x))) if dist is not None else 0
            if n_traces:
                trace_idx = max(0, min(trace_idx, n_traces - 1))
            x_km = self._pick_x_km(trace_idx, n_traces)
            ln = pg.InfiniteLine(pos=x_km, angle=90, pen=pen_b)
            ln.setVisible(self._boundaries_visible)   # honour the live toggle
            self.plot.addItem(ln)
            self._overlay_lines.append(ln)
            self._boundary_lines.append(ln)
            self._boundary_trace_indices.append(trace_idx)

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
        if self._ruler_roi is not None:
            self._ruler_roi.setPen(pg.mkPen(theme.color("highlight"), width=2))
        if self._ruler_label is not None:
            self._ruler_label.setColor(theme.color("highlight"))
        if self._ab_divider is not None:
            self._ab_divider.setPen(pg.mkPen(theme.color("highlight"), width=2,
                                             style=Qt.PenStyle.DashLine))
            self._ab_divider.setHoverPen(pg.mkPen(theme.color("highlight"), width=3))
        for lb in (self._ab_label_a, self._ab_label_b):
            if lb is not None:
                lb.setColor(theme.color("highlight"))
        self._refresh_dynamic_color()  # new panel background → re-check wiggle contrast

    def _retranslate(self, *_) -> None:
        self.plot.setLabel("bottom", self.tr("Distance (km)"))
        self.plot.setLabel("left", self.tr("Time (ms)"))
        if self._cbar is not None:
            self._cbar.setLabel("right", self.tr("Amplitude"))
        self._measure_action.setText(self.tr("Measure"))
        self._hide_ruler_action.setText(self.tr("Hide Ruler"))
        self._anomaly_action.setText(self.tr("Add Anomaly to Map"))
