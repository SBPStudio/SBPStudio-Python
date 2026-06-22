"""
map_view.py — Navigation map (vessel trackline) with bi-directional sync.

Plots the cruise track from the per-trace Source X/Y coordinates (already
scaled by the SEG-Y coordinate scalar and unit-converted at load time, i.e.
``profile.lons`` / ``profile.lats``). A strict 1:1 aspect lock keeps the track
geometry undistorted.

Two overlaid lines:
  * full trackline   — thin, muted (the whole profile path)
  * visible segment  — thick, bright (the traces currently on screen)

Two STATIC anchor dots mark the line endpoints — green = Start Of Line (SOL,
``_x[0]``/``_y[0]``), red = End Of Line (EOL, ``_x[-1]``/``_y[-1]``). They never
move; only the thick cyan segment tracks the visible window.

A lightweight OFFLINE world basemap is drawn underneath for geographic context —
100 % local, no tiles/network. It is loaded from a bundled GeoJSON coastline
asset (``assets/coastlines_highres.geojson``) with the stdlib ``json`` module; if
that file is missing it falls back to a plain lat/lon graticule (and warns on the
console). It is shown only when the track looks like geographic degrees (bbox
within ±180 lon / ±90 lat); projected/UTM metres hide it. GeoJSON parsing +
rendering live here in the GUI — the core never deals with display assets.

Bi-directional sync (by absolute TRACE INDEX). The seismic view resolves the
visible indices by masking the real distance array, so it stays in lock-step
with the data even when GPS dropouts plateau the distance:
  * Profile → Map: ``set_visible_range(trace0, trace1)`` re-highlights the cyan
    segment as the user pans/zooms the seismic ViewBox.
  * Map → Profile: clicking the track emits :pyattr:`trace_clicked(idx)`; the
    tab scrolls the SeismicView to that trace.
"""
from __future__ import annotations

import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QMarginsF, Qt, QPointF, QRectF, QSizeF, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QImage, QPageSize, QPainter, QPainterPath,
    QPdfWriter, QPen, QPolygonF,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QColorDialog, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QGraphicsItem, QGraphicsLineItem,
    QGraphicsPathItem, QGraphicsRectItem, QHBoxLayout, QInputDialog, QLabel,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QSlider,
    QSplitter, QVBoxLayout, QWidget,
)

from ..i18n import language_manager
from ..theme import MONO, theme

# Z-order bands. Basemap sits locked at the very bottom; the SEG-Y track/markers
# at z ≥ 0 on top. Custom GIS layers live strictly in between, ordered by the
# Layer Manager list (top of list = nearest the top of the band).
_BASEMAP_Z = -1000
_LAYER_Z_TOP = -100
_LAYER_Z_BOTTOM = -999
# Distinct outline colours cycled across vector layers.
_LAYER_COLORS = ["#ffb000", "#ff5e5e", "#b388ff", "#35d07f", "#ff8ad6",
                 "#7fd1ff", "#ffe066", "#ff7043"]
# Marks the special SEG-Y track row so it's protected from removal.
_ROLE_IS_TRACK = int(Qt.ItemDataRole.UserRole) + 1
# Marks the basemap row — also protected from removal, and excluded from the
# normal list-order → zValue mapping (it stays pinned at _BASEMAP_Z).
_ROLE_IS_BASEMAP = int(Qt.ItemDataRole.UserRole) + 2
# The user-facing name WITHOUT its leading tag glyph (▦/▤/≈/🌐) — kept separate
# from QListWidgetItem.text() so renaming never corrupts the tag, and so a
# language switch can refresh a built-in row's name without touching a
# user-supplied rename (see _ROLE_RENAMED).
_ROLE_NAME = int(Qt.ItemDataRole.UserRole) + 3
_ROLE_TAG = int(Qt.ItemDataRole.UserRole) + 4
_ROLE_RENAMED = int(Qt.ItemDataRole.UserRole) + 5
# Best-effort (x, y) map position to anchor this layer's on-map name label —
# set once at registration time from the layer's own geometry (see
# _register_layer / _update_track_anchor). None = no meaningful anchor
# (e.g. the basemap), so "Toggle Label on Map" is a no-op for that row.
_ROLE_ANCHOR = int(Qt.ItemDataRole.UserRole) + 6
# Whether the row's name should be rendered on the map (independent of the
# row's own checkbox visibility — see _set_label_visible).
_ROLE_LABEL_ON = int(Qt.ItemDataRole.UserRole) + 7
# The live pg.TextItem for this row's label, lazily created on first toggle-on
# and reused afterwards (never recreated, just shown/hidden/repositioned).
_ROLE_LABEL_ITEM = int(Qt.ItemDataRole.UserRole) + 8
# User-edited legend caption (None = fall back to the layer's own name). Lives
# on the layer row, not the legend, so it survives the legend being toggled
# off/on and repopulated.
_ROLE_LEGEND_TEXT = int(Qt.ItemDataRole.UserRole) + 9
# "Format Labels…" persisted properties (see _LabelFormatDialog / _show_label).
_ROLE_LABEL_ROTATION = int(Qt.ItemDataRole.UserRole) + 10   # degrees, default 0.0
_ROLE_LABEL_OFFSET = int(Qt.ItemDataRole.UserRole) + 11     # (dx, dy) data units, default (0,0)
_ROLE_LABEL_SIZE = int(Qt.ItemDataRole.UserRole) + 12       # font point size, default _LABEL_DEFAULT_SIZE
# User-chosen override colour ("Change Color…", hex string) — None = keep the
# layer's auto-assigned colour from _LAYER_COLORS.
_ROLE_COLOR = int(Qt.ItemDataRole.UserRole) + 13
# The row's start/end location markers (SOL/EOL dots), if it's a track-type
# layer — [] / None for anything else. "Show Points" toggles THESE, not
# vertices along the line (see _apply_show_points).
_ROLE_MARKERS = int(Qt.ItemDataRole.UserRole) + 14
# A user-drawn shape's raw geometry, kept for "Export as .shp…":
# {"geom_type": "point"|"line", "coords": [(x, y), ...]}.
_ROLE_DRAWN_GEOM = int(Qt.ItemDataRole.UserRole) + 15
# Marks a row as user-drawn (via the map canvas's right-click Draw Point /
# Draw Polyline) — gates the "Export as .shp…" context-menu action.
_ROLE_IS_DRAWN = int(Qt.ItemDataRole.UserRole) + 16
# "Hide from legend" — the layer itself stays fully visible on the map; this
# only excludes its row from the LEGEND rebuild (_refresh_legend), letting
# the remaining entries pack tightly with no gap. Restored via the legend's
# background right-click "Hidden items…" submenu.
_ROLE_LEGEND_HIDDEN = int(Qt.ItemDataRole.UserRole) + 17
# File-switching (Part 1): the source identity of an "Add to map" reference
# track layer — a profile's path string, or "chain:<label>" for a chain. None
# for every other kind of layer (drawn shapes, GIS overlays, the live track).
# See add_track_layer / layer_source_clicked.
_ROLE_SOURCE_ID = int(Qt.ItemDataRole.UserRole) + 18

_LABEL_DEFAULT_SIZE = 9

# Bundled local basemap asset (NO network ever). A high-resolution decimated
# coastline GeoJSON (e.g. Natural Earth 1:110m) dropped here is rendered as the
# land backdrop; if it is absent we fall back to a plain lat/lon graticule.
_BASEMAP_ASSET = "coastlines_highres.geojson"
_basemap_warned = False   # warn at most once per process


def _warn_missing_basemap(path: Path, exc: Optional[Exception] = None) -> None:
    """One-time console warning when the local GeoJSON basemap can't be loaded.
    No network fallback — the map just draws a graticule instead."""
    global _basemap_warned
    if _basemap_warned:
        return
    _basemap_warned = True
    detail = f" ({exc})" if exc else ""
    print(
        f"[MapView] Local basemap '{_BASEMAP_ASSET}' not loaded{detail}; "
        f"drawing a lat/lon graticule instead.\n"
        f"          Place a decimated coastline GeoJSON (e.g. Natural Earth "
        f"1:110m) at:\n            {path}\n"
        f"          for an offline geographic backdrop (no internet required).",
        file=sys.stderr,
    )


class _DraggableLegend(pg.LegendItem):
    """A :class:`pg.LegendItem` with editable entries.

    Free dragging is already built into the base class (``mouseDragEvent``).
    This subclass adds the two things pyqtgraph doesn't: double-click (or
    right-click) an entry to edit its caption, and a right-click menu to snap
    the legend to a corner (handy after a manual drag, and the standard
    bottom-left/bottom-right placement expected in map exports).

    All editing logic lives on the owning :class:`MapView` (``_map_view``) —
    this class only translates raw mouse events into row indices."""

    def __init__(self, map_view: "MapView", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._map_view = map_view

    def _row_at(self, pos) -> Optional[int]:
        """Row index whose LABEL cell contains ``pos`` (legend-local coords,
        already supplied pre-mapped by pyqtgraph's click-event dispatch)."""
        for row in range(self.layout.rowCount()):
            item = self.layout.itemAt(row, 1)
            if item is not None and item.geometry().contains(pos):
                return row
        return None

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._map_view._show_legend_context_menu(self._row_at(ev.pos()), ev.screenPos())
            return
        if ev.double():
            ev.accept()
            row = self._row_at(ev.pos())
            if row is not None:
                self._map_view._edit_legend_entry(row)
            return
        super().mouseClickEvent(ev)


class _RasterItemSample(pg.graphicsItems.LegendItem.ItemSample):
    """Legend swatch for a raster (GeoTIFF) layer: a small thumbnail of the
    actual image instead of pyqtgraph's default line-style swatch — a plain
    pen/line is meaningless for a raster (there's no line to draw)."""

    _THUMB_PX = 40   # downsample target before colorizing; plenty for a 20x20 swatch

    def __init__(self, image_item: pg.ImageItem) -> None:
        super().__init__(image_item)
        self._qimg = self._build_thumbnail(image_item)

    @classmethod
    def _build_thumbnail(cls, image_item: pg.ImageItem) -> Optional[QImage]:
        arr = image_item.image
        if arr is None or arr.size == 0:
            return None
        h, w = arr.shape[:2]
        step = max(1, max(h, w) // cls._THUMB_PX)
        small = np.ascontiguousarray(arr[::step, ::step])
        try:
            argb, _has_alpha = pg.functions.makeARGB(
                small, lut=image_item.lut, levels=image_item.levels)
        except Exception:
            return None
        img = QImage(argb.data, argb.shape[1], argb.shape[0],
                    argb.strides[0], QImage.Format.Format_ARGB32)
        return img.copy()   # detach from the transient `argb` buffer

    def paint(self, p, *args) -> None:
        if self._qimg is None or self._qimg.isNull():
            super().paint(p, *args)
            return
        scaled = self._qimg.scaled(18, 18, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        x = 1.0 + (18 - scaled.width()) / 2.0
        y = 1.0 + (18 - scaled.height()) / 2.0
        p.drawImage(QPointF(x, y), scaled)


class _ScaleBarItem(pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    """Custom graphic-scale replacement for stock ``pg.ScaleBar``.

    Stock ScaleBar only supports a single filled rect + a label set once at
    construction, and (like ``pg.LegendItem``) is fully non-interactive — no
    right-click hook. This subclass is modeled on its anchor/update plumbing
    (``changeParent``/``updateBar``/``setParentItem`` below are copied from
    pg.ScaleBar's own source, including the SAME setParentItem(None)-on-detach
    quirk worked around in ``MapView._toggle_scale_bar``) but adds:

      * three visual styles — simple line / segmented (tick) line / alternating
        black-white blocks (``self.style``)
      * two zoom behaviors (``self.zoom_mode``):
          "size"  — the real-world distance (``size_m``) is fixed; the bar's
                    ON-SCREEN width grows/shrinks with zoom (stock behavior).
          "value" — the on-screen pixel width (``pixel_width``) is fixed; the
                    TEXT updates to whatever real distance that now represents.
      * a right-click menu (style / zoom mode / "Set Scale (1:X)…")

    ``calibration_fn()`` returns how many real-world METRES one data-unit-X
    represents at the CURRENT view (1.0 for projected/UTM; latitude-corrected
    for geographic degrees — see ``MapView._scale_calibration_factor``), so
    this class itself never needs to know about coordinate systems.
    """

    N_SEGMENTS = 4

    def __init__(self, calibration_fn: Callable[[], float], *, width: float = 5,
                color="#ffffff", style: str = "line", zoom_mode: str = "size",
                size_m: float = 100.0, pixel_width: float = 120.0,
                offset=(-20, -20),
                on_style_changed: Optional[Callable[[str], None]] = None,
                on_zoom_mode_changed: Optional[Callable[[str], None]] = None,
                on_set_scale: Optional[Callable[[], None]] = None,
                on_reset_scale: Optional[Callable[[], None]] = None,
                is_locked_fn: Optional[Callable[[], bool]] = None) -> None:
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton)
        self._calibration_fn = calibration_fn
        self._bar_width = width
        self._color = color
        self.style = style
        self.zoom_mode = zoom_mode
        self.size_m = size_m
        self.pixel_width = pixel_width
        self.offset = offset
        self._on_style_changed = on_style_changed
        self._on_zoom_mode_changed = on_zoom_mode_changed
        self._on_set_scale = on_set_scale
        self._on_reset_scale = on_reset_scale
        self._is_locked_fn = is_locked_fn

        self._rect = QRectF(0, 0, 0, 0)   # current rendered footprint (hit area)
        self._last_w_px = 0.0
        self._last_meters = size_m
        # The ViewBox we currently have sigRangeChanged → updateBar wired to,
        # so changeParent can DISCONNECT it on detach/reparent (Bug #5) — the
        # stock pg.ScaleBar never disconnects, leaking the item + accumulating
        # dead callbacks on every toggle-off/on cycle.
        self._connected_view = None

        self._segments: list = []   # transient children for "segmented"/"blocks"
        self.bar = QGraphicsRectItem()
        self.bar.setParentItem(self)
        self.text = pg.TextItem(anchor=(0.5, 1))
        self.text.setParentItem(self)
        self._restyle_pens()

    # ── pg.ScaleBar-derived anchor plumbing (see class docstring) ───────────

    def changeParent(self) -> None:
        # itemChange calls this on EVERY parent/scene change (attach, detach,
        # reparent). Always tear down the previous connection first (Bug #5):
        # on detach the new parentItem() is None so the stock pg.ScaleBar would
        # silently leave sigRangeChanged → updateBar connected, keeping this
        # C++ item alive and firing dead callbacks forever. Disconnecting the
        # remembered view here makes detach AND reparent both clean.
        if self._connected_view is not None:
            try:
                self._connected_view.sigRangeChanged.disconnect(self.updateBar)
            except (TypeError, RuntimeError):
                pass        # already gone / never connected — fine
            self._connected_view = None
        view = self.parentItem()
        if view is None:
            return
        view.sigRangeChanged.connect(self.updateBar)
        self._connected_view = view
        self.updateBar()

    def boundingRect(self) -> QRectF:
        return self._rect

    def setParentItem(self, p):
        ret = pg.GraphicsObject.setParentItem(self, p)
        if self.offset is not None:
            offset = pg.Point(self.offset)
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret

    def paint(self, p, *args) -> None:
        pass   # all visible content lives in child items (bar/segments/text)

    # ── Style / color ────────────────────────────────────────────────────────

    def set_color(self, color) -> None:
        self._color = color
        self._restyle_pens()

    def _restyle_pens(self) -> None:
        pen = pg.mkPen(self._color)
        brush = pg.mkBrush(self._color)
        self.bar.setPen(pen)
        self.bar.setBrush(brush)
        self.text.setColor(self._color)
        if self.style != "blocks":   # blocks are deliberately fixed black/white
            for seg in self._segments:
                seg.setPen(pen)

    def set_style(self, style: str) -> None:
        if style == self.style:
            return
        self.style = style
        if self._on_style_changed:
            self._on_style_changed(style)
        self.updateBar()

    def set_zoom_mode(self, mode: str) -> None:
        if mode == self.zoom_mode:
            return
        # Seed the new mode from the bar's CURRENT effective value so the
        # switch doesn't visibly jump the bar's size or label.
        if mode == "value":
            self.pixel_width = self._last_w_px or self.pixel_width
        else:
            self.size_m = self._last_meters or self.size_m
        self.zoom_mode = mode
        if self._on_zoom_mode_changed:
            self._on_zoom_mode_changed(mode)
        self.updateBar()

    # ── Live geometry/label recompute (called on every pan/zoom) ────────────

    def updateBar(self, *_args) -> None:
        view = self.parentItem()
        if view is None:
            return
        m_per_unit = max(self._calibration_fn(), 1e-12)
        p0 = view.mapFromViewToItem(self, QPointF(0, 0))
        p1 = view.mapFromViewToItem(self, QPointF(1.0, 0))
        px_per_unit = abs((p1 - p0).x()) or 1e-9
        if self.zoom_mode == "value":
            w_px = self.pixel_width
            meters = (w_px / px_per_unit) * m_per_unit
        else:
            data_span = self.size_m / m_per_unit
            w_px = data_span * px_per_unit
            meters = self.size_m
        self._last_w_px = w_px
        self._last_meters = meters
        self.prepareGeometryChange()
        self._rect = QRectF(-w_px, 0, w_px, max(self._bar_width, 1))
        self._redraw(w_px)
        self.text.setText(pg.siFormat(meters, suffix="m"))
        self.text.setPos(-w_px / 2.0, 0)

    def _redraw(self, w_px: float) -> None:
        for seg in self._segments:
            seg.setParentItem(None)
        self._segments = []
        self.bar.setVisible(self.style == "line")

        if self.style == "line":
            self.bar.setRect(QRectF(-w_px, 0, w_px, self._bar_width))
            return

        n = self.N_SEGMENTS
        seg_w = w_px / n if n else w_px
        if self.style == "blocks":
            for i in range(n):
                rect = QGraphicsRectItem(-w_px + i * seg_w, 0, seg_w, self._bar_width)
                rect.setBrush(QBrush(QColor("#000000" if i % 2 == 0 else "#ffffff")))
                rect.setPen(QPen(QColor("#000000"), 1))
                rect.setParentItem(self)
                self._segments.append(rect)
        elif self.style == "segmented":
            pen = pg.mkPen(self._color, width=2)
            baseline = QGraphicsLineItem(-w_px, self._bar_width / 2.0, 0, self._bar_width / 2.0)
            baseline.setPen(pen)
            baseline.setParentItem(self)
            self._segments.append(baseline)
            for i in range(n + 1):
                x = -w_px + i * seg_w
                tick = QGraphicsLineItem(x, 0, x, self._bar_width)
                tick.setPen(pen)
                tick.setParentItem(self)
                self._segments.append(tick)

    # ── Right-click menu ─────────────────────────────────────────────────────

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._show_context_menu(ev.screenPos())
        else:
            ev.ignore()

    def _show_context_menu(self, screen_pos) -> None:
        menu = QMenu()
        style_menu = menu.addMenu(self.tr("Scale Style"))
        styles = (("line", self.tr("Simple Line")),
                 ("segmented", self.tr("Segmented Line")),
                 ("blocks", self.tr("Alternating Black/White Blocks")))
        for key, label in styles:
            act = style_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.style == key)
            act.triggered.connect(lambda _checked, k=key: self.set_style(k))

        zoom_menu = menu.addMenu(self.tr("Zoom Behavior"))
        modes = (("size", self.tr("Dynamic Size (fixed value)")),
                ("value", self.tr("Dynamic Value (fixed width)")))
        for key, label in modes:
            act = zoom_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.zoom_mode == key)
            act.triggered.connect(lambda _checked, k=key: self.set_zoom_mode(k))

        menu.addSeparator()
        scale_action = menu.addAction(self.tr("Set Scale (1:X)…"))
        scale_action.triggered.connect(lambda: self._on_set_scale and self._on_set_scale())
        reset_action = menu.addAction(self.tr("Reset Scale Lock"))
        # Only meaningful when an absolute 1:X lock is currently armed —
        # greyed out otherwise so the menu states the truth.
        reset_action.setEnabled(bool(self._is_locked_fn and self._is_locked_fn()))
        reset_action.triggered.connect(lambda: self._on_reset_scale and self._on_reset_scale())
        menu.exec(screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else screen_pos)


class _NorthArrowItem(pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    """Custom North Arrow overlay: 4 visual styles, 2 corner positions, and
    an adjustable pixel size, with a right-click menu for all three.

    Screen-space anchored (the SAME ``GraphicsWidgetAnchor`` mechanism as
    ``_ScaleBarItem`` — see that class's docstring for why this is exempt
    from the ``ItemIgnoresTransformations`` export-scaling bug that affects
    ``pg.LegendItem``) rather than living in DATA space like the previous
    plain ``pg.TextItem`` implementation: this gets a CONSTANT on-screen
    size for free, with no per-range-change repositioning needed at all —
    the anchor mechanism keeps it pinned to its corner automatically.

    Unlike ``pg.ScaleBar``/``pg.LegendItem``, this class's own
    ``setParentItem()`` only re-anchors when the new parent is NOT None, so
    it never hits their "Cannot anchor; parent is not set" detach bug — no
    offset-clearing workaround is needed before ``removeItem()``."""

    def __init__(self, *, style: str = "simple", position: str = "top-right",
                size_px: float = 36.0, color="#ffffff", offset_px: float = 16.0,
                on_style_changed: Optional[Callable[[str], None]] = None,
                on_position_changed: Optional[Callable[[str], None]] = None,
                on_size_changed: Optional[Callable[[float], None]] = None,
                on_set_size: Optional[Callable[[], None]] = None) -> None:
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton)
        self.style = style
        self.position = position
        self.size_px = size_px
        self._color = color
        self._offset_px = offset_px
        self._on_style_changed = on_style_changed
        self._on_position_changed = on_position_changed
        self._on_size_changed = on_size_changed
        self._on_set_size = on_set_size
        self._rect = QRectF(0, 0, size_px, size_px)

    def _apply_anchor(self) -> None:
        self.prepareGeometryChange()
        s = self.size_px
        o = self._offset_px
        if self.position == "top-left":
            self._rect = QRectF(0, 0, s, s)
            anchor, offset = (0.0, 0.0), (o, o)
        else:
            self._rect = QRectF(-s, 0, s, s)
            anchor, offset = (1.0, 0.0), (-o, o)
        if self.parentItem() is not None:
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)

    def boundingRect(self) -> QRectF:
        return self._rect

    def setParentItem(self, p):
        ret = pg.GraphicsObject.setParentItem(self, p)
        if p is not None:
            self._apply_anchor()
        return ret

    def set_style(self, style: str) -> None:
        if style != self.style:
            self.style = style
            if self._on_style_changed:
                self._on_style_changed(style)
            self.update()

    def set_position(self, position: str) -> None:
        if position != self.position:
            self.position = position
            self._apply_anchor()
            if self._on_position_changed:
                self._on_position_changed(position)

    def set_size(self, size_px: float) -> None:
        self.size_px = max(16.0, float(size_px))
        self._apply_anchor()
        if self._on_size_changed:
            self._on_size_changed(self.size_px)

    def set_color(self, color) -> None:
        self._color = color
        self.update()

    def paint(self, p, *args) -> None:
        rect = self._rect
        p.save()
        pen = QPen(QColor(self._color))
        pen.setWidthF(max(1.0, rect.width() * 0.05))
        p.setPen(pen)
        p.setBrush(QBrush(QColor(self._color)))
        cx = rect.center().x()
        top, bottom, w = rect.top(), rect.bottom(), rect.width()
        label_rect = QRectF(rect.left(), bottom - w * 0.32, w, w * 0.32)
        font = QFont()
        font.setPointSizeF(max(7.0, w * 0.26))
        p.setFont(font)

        if self.style == "triangle":
            tri = QPolygonF([QPointF(cx, top),
                            QPointF(rect.left() + w * 0.15, bottom - w * 0.34),
                            QPointF(rect.right() - w * 0.15, bottom - w * 0.34)])
            p.drawPolygon(tri)
        elif self.style == "minimal":
            small = QFont(); small.setPointSizeF(max(6.0, w * 0.2)); p.setFont(small)
            p.drawLine(QPointF(cx, bottom - w * 0.32), QPointF(cx, top + w * 0.10))
            head = QPolygonF([QPointF(cx, top), QPointF(cx - w * 0.07, top + w * 0.16),
                             QPointF(cx + w * 0.07, top + w * 0.16)])
            p.drawPolygon(head)
        elif self.style == "compass":
            radius = w * 0.30
            center = QPointF(cx, top + radius + w * 0.06)
            p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            p.drawEllipse(center, radius, radius)
            for ang, length in ((90, radius * 1.6), (270, radius * 0.6),
                                (0, radius * 0.6), (180, radius * 0.6)):
                rad = math.radians(ang)
                tip = QPointF(center.x() + length * math.cos(rad),
                              center.y() - length * math.sin(rad))
                p.drawLine(center, tip)
        else:   # "simple" — the original "N ↑" glyph, redrawn as a real arrow
            p.drawLine(QPointF(cx, bottom - w * 0.32), QPointF(cx, top + w * 0.05))
            head = QPolygonF([QPointF(cx, top), QPointF(cx - w * 0.12, top + w * 0.22),
                             QPointF(cx + w * 0.12, top + w * 0.22)])
            p.drawPolygon(head)
        p.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "N")
        p.restore()

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._show_context_menu(ev.screenPos())
        else:
            ev.ignore()

    def _show_context_menu(self, screen_pos) -> None:
        menu = QMenu()
        pos_menu = menu.addMenu(self.tr("Position"))
        for key, label in (("top-right", self.tr("Top-Right")),
                          ("top-left", self.tr("Top-Left"))):
            act = pos_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.position == key)
            act.triggered.connect(lambda _c, k=key: self.set_position(k))

        style_menu = menu.addMenu(self.tr("Style"))
        for key, label in (("simple", self.tr("Simple N↑")),
                          ("compass", self.tr("Compass Rose")),
                          ("minimal", self.tr("Minimalist")),
                          ("triangle", self.tr("Filled Triangle"))):
            act = style_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self.style == key)
            act.triggered.connect(lambda _c, k=key: self.set_style(k))

        menu.addSeparator()
        size_action = menu.addAction(self.tr("Size…"))
        size_action.triggered.connect(lambda: self._on_set_size and self._on_set_size())
        menu.exec(screen_pos.toPoint() if hasattr(screen_pos, "toPoint") else screen_pos)


class _LabelFormatDialog(QDialog):
    """Rotation / offset / font size for one or more selected layers' on-map
    name labels ("Format Labels…"), all as sliders. For a multi-selection,
    the fields are pre-filled from the FIRST selected row; OK applies the
    chosen values uniformly to every selected row (standard batch-edit UX).

    Offset is entered as a slider over ‰ (per-mille) of the map's CURRENT
    view extent rather than raw coordinate units — a slider needs a small,
    fixed integer range, and "per-mille of what's currently visible" stays
    meaningful regardless of whether the underlying coordinates are
    geographic degrees or projected metres. ``values()`` converts that back
    to the same data-unit offset that was always persisted in
    ``_ROLE_LABEL_OFFSET``, so the stored representation is unchanged.

    Every slider drag calls ``on_change(rotation, offset, size)`` LIVE (on
    every ``valueChanged``, not just on release/accept) so the caller can
    push the in-progress values straight onto the map's TextItem(s)."""

    def __init__(self, parent, rotation: float, offset: Tuple[float, float], size: int,
                view_extent: Tuple[float, float], on_change=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Format Labels"))
        self._extent = (view_extent[0] or 1.0, view_extent[1] or 1.0)
        self._on_change = None   # guarded until all sliders exist — see below
        form = QFormLayout(self)

        self.sl_rotation = self._slider_row(
            form, self.tr("Rotation (angle):"), -180, 180, int(round(rotation)), "°")

        ew, eh = self._extent
        ox_permille = max(-200, min(200, int(round(offset[0] / ew * 1000))))
        oy_permille = max(-200, min(200, int(round(offset[1] / eh * 1000))))
        self.sl_offset_x = self._slider_row(
            form, self.tr("Offset X (left / right):"), -200, 200, ox_permille, "‰")
        self.sl_offset_y = self._slider_row(
            form, self.tr("Offset Y (down / up):"), -200, 200, oy_permille, "‰")

        self.sl_size = self._slider_row(
            form, self.tr("Font size:"), 4, 72, size, "pt")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        # Only NOW are sl_rotation/sl_offset_x/sl_offset_y/sl_size all set —
        # _notify_change() calls self.values(), which reads every one of
        # them, so wiring the live callback any earlier (while the sliders'
        # own setValue() calls above are still firing valueChanged) would hit
        # an AttributeError on attributes that don't exist yet.
        self._on_change = on_change

    def _slider_row(self, form: QFormLayout, label_text: str, lo: int, hi: int,
                    val: int, suffix: str) -> QSlider:
        row = QHBoxLayout()
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(val)
        readout = QLabel(f"{val}{suffix}")
        readout.setMinimumWidth(40)

        def _changed(v: int) -> None:
            readout.setText(f"{v}{suffix}")
            self._notify_change()

        sl.valueChanged.connect(_changed)
        row.addWidget(sl, 1)
        row.addWidget(readout)
        form.addRow(label_text, row)
        return sl

    def _notify_change(self) -> None:
        if self._on_change is not None:
            self._on_change(*self.values())

    def values(self) -> Tuple[float, Tuple[float, float], int]:
        ew, eh = self._extent
        offset = (self.sl_offset_x.value() / 1000.0 * ew,
                 self.sl_offset_y.value() / 1000.0 * eh)
        return float(self.sl_rotation.value()), offset, self.sl_size.value()


class MapView(QWidget):
    """Vessel-track navigation map (PyQtGraph, 1:1 isometric)."""

    # Emitted with the trace index nearest to a click on the track.
    trace_clicked = pyqtSignal(int)
    # Emitted with a local file path when the user picks "Add Layer". The owning
    # tab reads it off-thread (core readers) and calls back :meth:`add_layer`.
    layer_file_requested = pyqtSignal(str)
    # File Switching (Part 1): emitted when the user clicks an "Add to map"
    # reference track layer (see add_track_layer), carrying its source_id — a
    # profile path, or "chain:<label>" for a chain — and the absolute trace
    # index nearest the click (within THAT track's own per-trace arrays, so
    # it indexes directly into the same profile/chain once activated).
    # MainWindow resolves source_id back to a sidebar row, force-activates
    # it (even if a stale multi-selection would otherwise swallow a plain
    # row-click), and centers the seismic view on the trace index once the
    # profile/chain's data has actually finished loading.
    layer_source_clicked = pyqtSignal(str, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        # Resizable divider between the map canvas and the layers sidebar so the
        # user can widen the layer list (long names) or reclaim canvas space.
        map_split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(map_split)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self._grid_visible = True              # Reference Grid toggle (Part 3)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setAspectLocked(True)        # strict 1:1 → no track distortion
        for ax in ("left", "bottom"):
            self.plot.getAxis(ax).enableAutoSIPrefix(False)
        # Right axis: tick line only (no duplicate value labels) so the plot's
        # bounding box reads as a closed rectangle — otherwise the right edge
        # looks "cut off", especially with the legend now sitting in the side
        # panel rather than inside that open edge.
        self.plot.showAxis("right")
        self.plot.getAxis("right").setStyle(showValues=False)

        # Plot + live cursor-coordinate readout, docked at the bottom of the
        # map (Part 3) — wrapped together so the readout always tracks the
        # plot's width regardless of the side panel's size.
        plot_col = QWidget()
        plot_lay = QVBoxLayout(plot_col)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.setSpacing(2)
        plot_lay.addWidget(self.plot, 1)
        self.lbl_cursor_coords = QLabel("")
        self.lbl_cursor_coords.setStyleSheet(f"font-family: {MONO};")
        plot_lay.addWidget(self.lbl_cursor_coords)
        map_split.addWidget(plot_col)
        self._layer_count = 0                  # for cycling layer colours
        map_split.addWidget(self._build_layer_panel())
        # Canvas absorbs resize; the sidebar keeps its width and never collapses
        # fully (always grab-able to re-expand).
        map_split.setStretchFactor(0, 1)
        map_split.setStretchFactor(1, 0)
        map_split.setCollapsible(1, False)
        map_split.setSizes([1000, 190])

        # Track data (per trace).
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        # Authoritative geographic/projected flag derived from the ACTUAL CRS
        # (set_track / add_layer pass it down from the core; Bug #10). None =
        # unknown → fall back to the value-magnitude heuristic in
        # _coords_are_geographic. True/False overrides it.
        self._geographic_hint: Optional[bool] = None
        # Cached list of currently-VISIBLE raster ImageItems for the live
        # cursor Z-readout (Bug #12). None = dirty → rebuilt lazily on the next
        # hover. Invalidated whenever a layer is added/removed or its
        # visibility toggles, so the 60 Hz mouse-move handler never walks the
        # whole layer tree.
        self._raster_cache: Optional[list] = None
        # The SEG-Y track is a MANAGED layer (injected lazily on set_track).
        self._track_item: Optional[QListWidgetItem] = None

        # Map-canvas legend (Feature: Interactive and Editable Legend). None
        # while "Show Legend" is unchecked — created/destroyed, not just
        # hidden, so a toggle-off fully releases its scene items.
        self._legend: Optional[_DraggableLegend] = None
        # Parallel to self._legend.items — row i's owning QListWidgetItem, for
        # click-to-row lookups (double-click / right-click to edit a caption).
        self._legend_rows: list = []
        # Whether the legend is "docked" into the side panel (dock_legend_view)
        # instead of floating on the map canvas — see _dock_legend/_undock_legend.
        self._legend_docked: bool = False
        # Global "Show Points" toggle: shows/hides the start/end location
        # markers (SOL/EOL dots) on every track-type layer. Defaults to True
        # — those markers always appeared unconditionally before this toggle
        # existed, so the default behaviour is unchanged; unchecking lets the
        # user hide them. Not per-layer — see _apply_show_points.
        self._show_points: bool = True

        # Interactive drawing (right-click the map → Draw Point / Draw
        # Polyline / Draw Polygon / Measure Tool). None = normal mode (clicks
        # jump to the nearest trace, as always); a mode name while a draw is
        # in progress — see _start_draw/_on_draw_click/_finish_draw/_cancel_draw.
        # "measure" reuses the same point-accumulation/preview machinery but
        # never creates a permanent layer (see _finish_draw) and additionally
        # tracks the live cursor position for a running-distance readout.
        self._draw_mode: Optional[str] = None
        self._draw_points: list = []
        self._draw_preview_item: Optional[pg.PlotDataItem] = None
        self._measure_text: Optional[pg.TextItem] = None
        self._drawn_seq = 0   # naming counter for "Drawn Point N" / "Drawn Line N"
        self._poi_seq = 0     # naming counter for "POI N" (see add_poi_marker)

        # Cross-module sync (Link Views): a single transient, non-layer marker
        # tracking the seismic section's live hover position. Lazily created on
        # first use; repositioned (not recreated) on every hover. Independent
        # of the drawn/POI layers — never appears in the layer list or legend.
        self._nav_marker: Optional[pg.ScatterPlotItem] = None

        # Cartographic overlays — persistent toggles, independent of any
        # specific layer (see _toggle_scale_bar / _toggle_north_arrow). The
        # style/zoom-mode/size settings below live on MapView (not the item
        # itself) so they survive a toggle-off/on cycle — see _toggle_scale_bar.
        self._scale_bar: Optional[_ScaleBarItem] = None
        self._scale_bar_style: str = "line"
        self._scale_bar_zoom_mode: str = "size"
        self._scale_bar_size_m: float = 0.0    # 0 → recompute a nice value on first show
        self._scale_bar_pixel_width: float = 120.0
        # Export Scale Lock (Part 2): set by _apply_absolute_scale whenever
        # the user successfully applies a "Set Scale (1:X)" — export_image
        # then derives its output pixel size from THIS ratio (at the
        # current view's ground span) instead of the normal dpi*10
        # heuristic, so the exported file reproduces the exact physical
        # scale at the target print DPI. None = no lock (normal export).
        self._locked_scale_ratio: Optional[float] = None
        # North Arrow settings persist on MapView (not the item itself) so
        # they survive a toggle-off/on cycle — same convention as the scale
        # bar's own _scale_bar_* attributes.
        self._north_arrow: Optional[_NorthArrowItem] = None
        self._north_arrow_style: str = "simple"
        self._north_arrow_position: str = "top-right"
        self._north_arrow_size_px: float = 36.0

        # Offline world basemap (geographic context). One grouped path item under
        # everything; shown only when the track looks like degrees (see
        # _update_basemap). Built before the curves so it sits at the bottom.
        self._basemap_is_graticule: bool = False
        self._basemap_item: Optional[QListWidgetItem] = None
        self._basemap: QGraphicsPathItem = self._build_basemap()
        self._ensure_basemap_layer()           # checkable row, like any other layer

        # Items: full track (under) + visible segment (over) + start/end markers.
        self.curve_full = self.plot.plot([], [])
        self.curve_seg = self.plot.plot([], [])
        self.marker_start = pg.ScatterPlotItem(size=9, pen=None)
        self.marker_end = pg.ScatterPlotItem(size=9, pen=None)
        self.plot.addItem(self.marker_start)
        self.plot.addItem(self.marker_end)

        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved_coords)
        # Auto-clear the absolute scale lock on a USER pan/zoom (Bug #3).
        # sigRangeChangedManually fires ONLY for mouse drag/wheel — never for
        # programmatic setRange/autoRange — so _apply_absolute_scale's own
        # zoom (deferred + aspect-lock-nudged) can't false-trigger it. The
        # autoRange-on-new-track case is handled explicitly in set_track.
        self.plot.getViewBox().sigRangeChangedManually.connect(
            self._on_view_changed_clear_lock)
        self.plot.installEventFilter(self)   # Enter/Escape while drawing a polyline

        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)

    # ── Public API ──────────────────────────────────────────────────────────

    def set_track(self, x, y, is_geographic: Optional[bool] = None) -> None:
        """Set the full trackline from per-trace X/Y coordinates and pin the
        STATIC SOL/EOL anchors: green at the absolute start ``_x[0]``/``_y[0]``,
        red at the absolute end ``_x[-1]``/``_y[-1]``. These dots never move;
        only the cyan segment tracks the visible window. Resets the segment.

        ``is_geographic`` (Bug #10) is the authoritative coordinate-unit flag
        the caller derives from the source CRS (via
        ``core.crs_produces_geographic``): True = lon/lat, False = projected
        metres, None = unknown → fall back to the magnitude heuristic. Drives
        the basemap, scale-bar calibration, and cursor readout units."""
        self._geographic_hint = is_geographic
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self.curve_full.setData(self._x, self._y)
        if self._x.size:
            self.marker_start.setData([self._x[0]], [self._y[0]])   # green = SOL
            self.marker_end.setData([self._x[-1]], [self._y[-1]])   # red   = EOL
        self.curve_seg.setData([], [])
        self._ensure_track_layer()          # the track is a managed list layer
        self._update_track_anchor()
        self._update_basemap()
        # A new track reframes the view (autoRange below), so any armed
        # absolute scale lock no longer matches what's on screen — drop it
        # (Bug #3). autoRange is programmatic, so sigRangeChangedManually
        # never fires for it; clear explicitly here.
        self._reset_scale_lock()
        self.plot.getViewBox().autoRange()
        self._legend_dirty()

    def set_visible_range(self, trace0: int, trace1: int) -> None:
        """Highlight the cyan segment for the absolute trace-index window
        [trace0, trace1). Pure index slicing — NO distance lookup. The SOL/EOL
        anchor dots are NOT touched here: they stay fixed at the line ends."""
        if self._x is None or self._x.size < 1:
            return
        n = self._x.size
        t0 = max(0, min(int(trace0), n - 1))
        t1 = max(t0 + 1, min(int(trace1), n))
        self.curve_seg.setData(self._x[t0:t1], self._y[t0:t1])

    def clear(self) -> None:
        self._x = self._y = None
        self._geographic_hint = None     # next track re-establishes it (Bug #10)
        self.curve_full.setData([], [])
        self.curve_seg.setData([], [])
        self.marker_start.setData([], [])
        self.marker_end.setData([], [])
        self._basemap.setVisible(False)
        # Drop the managed track row (graphics items stay, just emptied). It is
        # re-injected at the top on the next set_track.
        if self._track_item is not None:
            self._remove_label(self._track_item)
            r = self.layer_list.row(self._track_item)
            if r >= 0:
                self.layer_list.takeItem(r)
            self._track_item = None
            self._reorder_layers()
            self._legend_dirty()

    # ── Cross-module sync (Link Views) ──────────────────────────────────────

    def show_navigation_marker(self, idx: int) -> None:
        """Move the live navigation cursor to the track point at absolute
        trace ``idx`` (called from the tab when Link Views is active and the
        user hovers the seismic section). Lazily creates a single reusable
        ScatterPlotItem — never a layer-list row, so it can't be hidden,
        renamed, or exported like a real layer."""
        if self._x is None or self._x.size < 1:
            return
        idx = max(0, min(int(idx), self._x.size - 1))
        if self._nav_marker is None:
            self._nav_marker = pg.ScatterPlotItem(
                size=12, brush=pg.mkBrush("#ff3030"), pen=pg.mkPen("w", width=1.5))
            self._nav_marker.setZValue(80)   # above every layer, including drawn shapes
            self.plot.getViewBox().addItem(self._nav_marker, ignoreBounds=True)
        self._nav_marker.setData([self._x[idx]], [self._y[idx]])
        self._nav_marker.setVisible(True)

    def hide_navigation_marker(self) -> None:
        """Hide the live navigation cursor (called when Link Views is turned off)."""
        if self._nav_marker is not None:
            self._nav_marker.setVisible(False)

    def add_poi_marker(self, idx: int) -> None:
        """Add a permanent "Point of Interest" layer at the track point for
        absolute trace ``idx`` (called from the tab on a double-click in the
        seismic section while Link Views is active). A real layer-list row —
        distinct star marker, tagged _ROLE_IS_DRAWN so it inherits the
        existing "Export as .shp…" support, exactly like a hand-drawn point."""
        if self._x is None or self._x.size < 1:
            return
        idx = max(0, min(int(idx), self._x.size - 1))
        pt = (float(self._x[idx]), float(self._y[idx]))
        self._poi_seq += 1
        item = pg.ScatterPlotItem(x=[pt[0]], y=[pt[1]], size=13, symbol="star",
                                  brush=pg.mkBrush("#ffd23f"), pen=pg.mkPen("w", width=1))
        name = self.tr("POI {0}").format(self._poi_seq)
        li = self._register_layer(name, [item], "vector", anchor=pt)
        li.setData(_ROLE_IS_DRAWN, True)
        li.setData(_ROLE_DRAWN_GEOM, {"geom_type": "point", "coords": [pt]})

    # ── Offline world basemap ─────────────────────────────────────────────────

    def _build_basemap(self) -> QGraphicsPathItem:
        """Build a single grouped path item for the basemap, in (lon, lat) data
        coordinates. One item = fluid pan/zoom. Added beneath the track, ignored
        by autoRange (the view frames the TRACK, not the planet), and
        click-transparent so it never steals the map's click-to-jump. Content is
        the local GeoJSON coastlines, or a graticule fallback if it's missing."""
        path, self._basemap_is_graticule = self._load_basemap_path()
        item = QGraphicsPathItem(path)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        item.setVisible(False)
        self.plot.getViewBox().addItem(item, ignoreBounds=True)
        item.setZValue(_BASEMAP_Z)     # set AFTER addItem so it isn't bumped up
        return item

    @staticmethod
    def _asset_path() -> Path:
        """Resolve the bundled GeoJSON path. Checks the repo-root ``assets/`` then
        a package-local ``sbp_studio/assets/`` so it works both in-tree and when
        the package ships its own assets. Returns the repo-root candidate if none
        exist (used only for the warning message)."""
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "assets" / _BASEMAP_ASSET,   # <repo>/assets/
            here.parents[2] / "assets" / _BASEMAP_ASSET,   # sbp_studio/assets/
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    def _load_basemap_path(self) -> Tuple[QPainterPath, bool]:
        """Return (path, is_graticule). Loads the local GeoJSON with the stdlib
        ``json`` module (no third-party GIS deps, no network). On any failure —
        file missing, unreadable, or empty — warns once and returns a lat/lon
        graticule instead."""
        asset = self._asset_path()
        if not asset.exists():
            _warn_missing_basemap(asset)
            return self._graticule_path(), True
        try:
            with open(asset, "r", encoding="utf-8") as fh:
                gj = json.load(fh)
            path = self._geojson_to_path(gj)
            if path.elementCount() == 0:
                raise ValueError("no renderable geometries")
            return path, False
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _warn_missing_basemap(asset, exc)
            return self._graticule_path(), True

    @staticmethod
    def _geojson_to_path(gj: dict) -> QPainterPath:
        """Flatten a GeoJSON object into one QPainterPath (lon=x, lat=y).
        Polygons/MultiPolygons → closed subpaths (filled land); LineStrings →
        open subpaths (coastline strokes). Robust to Feature/FeatureCollection/
        bare-geometry/GeometryCollection nesting."""
        path = QPainterPath()

        def add_ring(coords, close: bool) -> None:
            pts = [QPointF(float(c[0]), float(c[1])) for c in coords if len(c) >= 2]
            if len(pts) < 2:
                return
            path.addPolygon(QPolygonF(pts))
            if close:
                path.closeSubpath()

        def add_geom(geom) -> None:
            if not geom:
                return
            gtype, coords = geom.get("type"), geom.get("coordinates")
            if gtype == "Polygon":
                for ring in coords:
                    add_ring(ring, True)
            elif gtype == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        add_ring(ring, True)
            elif gtype == "LineString":
                add_ring(coords, False)
            elif gtype == "MultiLineString":
                for line in coords:
                    add_ring(line, False)
            elif gtype == "GeometryCollection":
                for g in geom.get("geometries", ()):
                    add_geom(g)

        gtype = gj.get("type") if isinstance(gj, dict) else None
        if gtype == "FeatureCollection":
            for feat in gj.get("features", ()):
                add_geom((feat or {}).get("geometry"))
        elif gtype == "Feature":
            add_geom(gj.get("geometry"))
        elif gtype:
            add_geom(gj)            # bare geometry object
        return path

    @staticmethod
    def _graticule_path() -> QPainterPath:
        """Fallback geographic context: a global lat/lon graticule every 15°.
        Drawn beneath the track; with autoRange framing the track only the
        nearby grid lines are visible."""
        path = QPainterPath()
        step = 15
        for lon in range(-180, 181, step):       # meridians
            path.moveTo(float(lon), -90.0)
            path.lineTo(float(lon), 90.0)
        for lat in range(-90, 91, step):          # parallels
            path.moveTo(-180.0, float(lat))
            path.lineTo(180.0, float(lat))
        return path

    def _coords_are_geographic(self) -> bool:
        """Whether the map's coordinates are geographic lon/lat (vs projected
        metres). Drives the basemap, scale-bar calibration, and cursor units.

        Bug #10: prefer the AUTHORITATIVE CRS-derived flag (``_geographic_hint``,
        passed down by set_track/add_layer from ``core.crs_produces_geographic``)
        whenever it is known. Only when the CRS is totally absent (hint is
        None) fall back to the value-magnitude heuristic — bbox within
        lon ∈ [-180, 180], lat ∈ [-90, 90] — which alone misclassifies a small
        projected/UTM survey whose coordinates happen to land in that range."""
        if self._geographic_hint is not None:
            return self._geographic_hint
        if self._x is None or self._x.size == 0:
            return False
        return (float(np.nanmin(self._x)) >= -180.0 and float(np.nanmax(self._x)) <= 180.0
                and float(np.nanmin(self._y)) >= -90.0 and float(np.nanmax(self._y)) <= 90.0)

    def _basemap_checked(self) -> bool:
        """Whether the user wants the basemap shown (its Layers-list checkbox).
        True before the row exists yet (e.g. during __init__'s first build)."""
        return (self._basemap_item is None
               or self._basemap_item.checkState() == Qt.CheckState.Checked)

    def _apply_basemap_visibility(self) -> None:
        """Actual on-screen visibility = user checkbox AND the geographic
        heuristic — unchecking the layer always hides it; checking it only
        shows it where it would be meaningful (geographic coordinates)."""
        self._basemap.setVisible(self._basemap_checked() and self._coords_are_geographic())

    def _update_basemap(self) -> None:
        self._apply_basemap_visibility()

    def _ensure_basemap_layer(self) -> None:
        """Add the basemap as a checkable row in the Layers list, behaving like
        any other layer (checkbox, rename, opacity) — except it can't be
        removed and its draw order is pinned at ``_BASEMAP_Z`` (excluded from
        the list-order → zValue mapping in ``_reorder_layers``)."""
        li = QListWidgetItem()
        li.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                   | Qt.ItemFlag.ItemIsUserCheckable)   # not draggable: stays put
        li.setCheckState(Qt.CheckState.Checked)
        li.setData(Qt.ItemDataRole.UserRole, [self._basemap])
        li.setData(_ROLE_IS_BASEMAP, True)
        self._basemap_item = li
        self.layer_list.addItem(li)
        self._set_basemap_item_text()

    def _set_basemap_item_text(self) -> None:
        if self._basemap_item is not None and not self._basemap_item.data(_ROLE_RENAMED):
            self._relabel(self._basemap_item, "🌐", self.tr("World Basemap"))

    # ── Layer Manager (custom GIS overlays) ───────────────────────────────────

    def _build_layer_panel(self) -> QWidget:
        """Side panel: a drag-reorderable, checkable list of custom layers plus
        Add/Remove buttons. List order = draw order (top row = top of the band)."""
        panel = QWidget()
        # Min/max (not fixed) so the QSplitter can resize it; sensible bounds keep
        # it from shrinking unusably small or hogging the whole tab.
        panel.setMinimumWidth(150)
        panel.setMaximumWidth(420)
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self.lbl_layers = QLabel()
        self.lbl_layers.setObjectName("section")
        v.addWidget(self.lbl_layers)
        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        # Extended (not Single) selection: the right-click context menu must
        # act on multiple selected layers at once (e.g. batch-toggling labels).
        self.layer_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.layer_list.itemChanged.connect(self._on_layer_item_changed)
        self.layer_list.currentItemChanged.connect(self._on_current_layer_changed)
        self.layer_list.itemDoubleClicked.connect(self._on_rename_layer)
        self.layer_list.model().rowsMoved.connect(lambda *_: self._reorder_layers())
        self.layer_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self._on_layer_context_menu)
        v.addWidget(self.layer_list, 1)
        # Docked-legend host: a small bare ViewBox (no axes/grid) that hosts the
        # LegendItem when the user docks it off the map canvas (see
        # _dock_legend). Hidden until then. A GraphicsLayoutWidget rather than
        # the QListWidget itself, per the dock design — the list stays usable
        # the whole time, this just appears alongside it while docked.
        # Height is NOT fixed — _fit_dock_legend() resizes it to match the
        # legend's actual content each time it changes, so longer legends
        # (more entries) are never clipped; a fixed magic-number height was
        # the previous (buggy) approach.
        self.dock_legend_view = pg.GraphicsLayoutWidget()
        self.dock_legend_view.setMinimumHeight(30)
        self.dock_vb = self.dock_legend_view.addViewBox(lockAspect=False)
        self.dock_vb.setMouseEnabled(False, False)
        self.dock_vb.setMenuEnabled(False)
        self.dock_vb.invertY(True)
        self.dock_legend_view.setVisible(False)
        v.addWidget(self.dock_legend_view)
        # Legend / Points — auto-populated from the currently visible (checked)
        # layers; Points shows/hides the SOL/EOL location markers.
        legend_row = QHBoxLayout()
        self.chk_show_legend = QCheckBox()
        self.chk_show_legend.toggled.connect(self._on_legend_toggled)
        self.chk_show_points = QCheckBox()
        self.chk_show_points.setChecked(True)
        self.chk_show_points.toggled.connect(self._on_show_points_toggled)
        legend_row.addWidget(self.chk_show_legend)
        legend_row.addWidget(self.chk_show_points)
        v.addLayout(legend_row)
        # Opacity (Transparencia) — applies to the currently selected layer.
        self.lbl_opacity = QLabel()
        v.addWidget(self.lbl_opacity)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setEnabled(False)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        v.addWidget(self.opacity_slider)
        row = QHBoxLayout()
        self.btn_add_layer = QPushButton()
        self.btn_add_layer.clicked.connect(self._on_add_layer_clicked)
        self.btn_remove_layer = QPushButton()
        self.btn_remove_layer.clicked.connect(self._remove_selected_layer)
        row.addWidget(self.btn_add_layer)
        row.addWidget(self.btn_remove_layer)
        v.addLayout(row)
        self.btn_export_map = QPushButton()
        self.btn_export_map.clicked.connect(self._on_export_clicked)
        v.addWidget(self.btn_export_map)
        return panel

    def _on_add_layer_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Add map layer"), "",
            self.tr("GIS layers (*.shp *.tif *.tiff *.geojson)"))
        if path:
            self.layer_file_requested.emit(path)

    # ── Export ──────────────────────────────────────────────────────────────

    _EXPORT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf")
    # Hard ceiling on the rendered output's pixel count, to bound the QImage /
    # QPdfWriter backing-store allocation (ARGB32 = 4 bytes/px, plus the
    # exporter's antialias buffers, so the real RAM cost is a few× this).
    # 150 Mpx ≈ a 600 MB final image — enough for a poster-size 300-DPI print
    # (e.g. ~14 000 × ~10 000), while preventing an extreme zoom-out at a small
    # 1:X ratio (or a huge --dpi) from requesting a multi-gigapixel image that
    # would OOM-crash the app. See _clamp_export_width.
    _EXPORT_MAX_PIXELS = 150_000_000

    def _on_export_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export map"), "map.png",
            self.tr("PNG image (*.png);;JPEG image (*.jpg);;TIFF image (*.tif);;"
                    "PDF document (*.pdf)"))
        if path:
            self.export_image(path)

    def export_image(self, path: str, dpi: int = 300) -> None:
        """Render the map (basemap + track + every visible layer, exactly as
        currently shown — hidden layers stay hidden) to PNG, JPEG, TIFF, or
        PDF — dispatched by ``path``'s extension (defaults to PNG if absent
        or unrecognised).

        Raster formats (PNG/JPEG/TIFF) go through pyqtgraph's
        ``ImageExporter`` (Qt's ``QImage.save()`` infers the file format from
        the extension — no extra code needed per format). PDF is the one
        exception: pyqtgraph's ``SVGExporter`` raises on this scene's grid/
        axis content regardless of which layers are visible (a pre-existing
        pyqtgraph bug, confirmed independent of the Layers list — see
        https://github.com/pyqtgraph/pyqtgraph SVGExporter coordinate
        parsing), so PDF is rendered directly via ``QPdfWriter`` instead (see
        ``_export_pdf``) — a real vector page, never SVGExporter.

        If the legend is DOCKED to the side panel, it's no longer part of the
        map scene at all — exported as a separate side-by-side composite
        instead (see ``_export_composite``); a floating legend exports inline
        with the map (see ``_export_single``). Either way, all four axes
        (top/bottom/left/right) are temporarily shown WITH their coordinate
        values for the export only, then restored to the live-view state.

        Export Scale Lock (Part 2): if an absolute "Set Scale (1:X)" is
        currently armed (``self._locked_scale_ratio``), the output pixel
        width is derived from that EXACT ratio at this call's ``dpi`` (see
        ``_scale_locked_export_width``) instead of the normal dpi*10
        heuristic, so the file physically reproduces that scale when printed
        at ``dpi``."""
        ext = Path(path).suffix.lower()
        if ext not in self._EXPORT_EXTENSIONS:
            path += ".png"
            ext = ".png"
        target_width = (self._scale_locked_export_width(dpi)
                        if self._locked_scale_ratio is not None
                        else int(max(1200, dpi * 10)))
        target_width = self._clamp_export_width(target_width)
        with self._full_axes_for_export():
            if ext == ".pdf":
                self._export_pdf(path, target_width, dpi)
            elif self._legend is not None and self._legend_docked:
                self._export_composite(path, target_width)
            else:
                self._export_single(path, target_width)

    def _clamp_export_width(self, target_width: int) -> int:
        """Cap ``target_width`` so the rendered output stays under
        ``_EXPORT_MAX_PIXELS``, downscaling while PRESERVING the aspect ratio
        (the height is derived from the width × the plot's source aspect in
        every export path, so clamping the width alone keeps the proportions).

        Guards against an OOM crash when an extreme zoom-out at a small 1:X
        scale, or a very high export DPI, would otherwise request a multi-
        gigapixel image. A no-op for ordinary exports."""
        target_width = max(1, int(target_width))
        src = self.plot.plotItem.sceneBoundingRect()
        aspect = (src.height() / src.width()) if src.width() > 0 else 1.0
        total_px = target_width * max(1.0, target_width * aspect)
        if total_px <= self._EXPORT_MAX_PIXELS:
            return target_width
        scale = math.sqrt(self._EXPORT_MAX_PIXELS / total_px)
        clamped = max(1200, int(target_width * scale))
        print(f"[Export] requested {target_width}px (~{total_px / 1e6:.0f} Mpx) "
              f"exceeds the {self._EXPORT_MAX_PIXELS / 1e6:.0f} Mpx ceiling; "
              f"clamped to {clamped}px (aspect preserved).")
        return clamped

    def _scale_locked_export_width(self, dpi: int) -> int:
        """Export Scale Lock (Part 2): compute the output pixel width so the
        exported file physically reproduces the EXACT armed "1:X" scale when
        printed/displayed at ``dpi`` — i.e. one physical inch of the printed
        page still represents exactly ``ratio`` ground inches, the same
        definition ``_apply_absolute_scale`` uses for the live screen. This
        is a SEPARATE DPI domain from the screen's (export print DPI vs.
        monitor physical DPI) — independent of the normal dpi*10 heuristic,
        because this is about a PHYSICAL print scale, not pixel density."""
        (x0, x1), _ = self.plot.getViewBox().viewRange()
        m_per_unit = self._scale_calibration_factor()
        view_span_m = abs(x1 - x0) * m_per_unit
        ratio = self._locked_scale_ratio
        paper_meters = view_span_m / ratio
        paper_inches = paper_meters / 0.0254
        width_px = max(1, int(round(paper_inches * dpi)))
        print(f"[Scale Math · Export] view_span={abs(x1 - x0):.4f} data units "
              f"({view_span_m:.2f} m ground), ratio=1:{ratio:.0f} -> "
              f"paper_width={paper_inches:.4f} in @ {dpi} DPI -> {width_px} px "
              f"(prints at exactly 1:{ratio:.0f})")
        return width_px

    @contextlib.contextmanager
    def _full_axes_for_export(self):
        """Temporarily show all 4 axes — uniformly styled and WITH coordinate
        values — for the duration of an export render. Restores the exact
        prior per-axis visibility/showValues state afterward, regardless of
        how the export finishes (including on an exception).

        Two real bugs fixed here, both confirmed by rendering an actual
        export and inspecting the resulting pixels (not just the Python
        state flags, which already looked correct):

        1. ``setStyle(showValues=True)`` alone did NOT take effect in time
           for the SYNCHRONOUS ``ImageExporter`` render that immediately
           follows — pyqtgraph's ``AxisItem.setStyle()`` invalidates its
           cached tick-label ``QPicture`` and calls the normal (deferred)
           ``QGraphicsItem.update()``, but the exporter renders the scene
           directly without first flushing that pending Qt update — so the
           right/top axes exported with NO visible numbers at all, even
           though ``showValues`` was correctly ``True`` in their style dict.
           ``QApplication.processEvents()`` (twice, empirically confirmed
           necessary) flushes the deferred update/layout before the render.
        2. Explicitly re-applying the SAME pen/textPen/tickFont to all four
           axes HERE (mirroring ``_restyle()``) guarantees top/right are
           styled identically to bottom/left even if this ever runs before
           the first ``_restyle()`` call."""
        axes = ("left", "bottom", "right", "top")
        prev_visible = {ax: self.plot.getAxis(ax).isVisible() for ax in axes}
        prev_show_values = {ax: self.plot.getAxis(ax).style.get("showValues", True)
                            for ax in axes}
        for ax in axes:
            self.plot.showAxis(ax)
            axis = self.plot.getAxis(ax)
            axis.setStyle(showValues=True, tickFont=QFont(MONO, 8))
            axis.setPen(theme.color("sub"))
            axis.setTextPen(theme.color("text"))
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
            app.processEvents()
        try:
            yield
        finally:
            for ax in axes:
                if prev_visible[ax]:
                    self.plot.showAxis(ax)
                else:
                    self.plot.hideAxis(ax)
                self.plot.getAxis(ax).setStyle(showValues=prev_show_values[ax])

    def _export_single(self, path: str, target_width: int) -> None:
        """The map, plus the legend inline if it's floating on the canvas."""
        from pyqtgraph.exporters import ImageExporter
        flag = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        if self._legend is not None:
            # Bug fix: LegendItem sets ItemIgnoresTransformations, so it always
            # paints at a fixed PHYSICAL size no matter the exporter's target
            # resolution — confirmed empirically (its paint-time transform
            # scale stayed 1.0 at both a ~screen-sized and a 4×-wider export)
            # — which is why it came out comically tiny next to a dpi*10-wide
            # export versus how it looks live. Clearing the flag for the
            # export's duration lets it inherit the SAME scale factor the
            # exporter already applies to the rest of the scene; the
            # subsequent verification showed the legend's transform scale then
            # matched the exporter's own width ratio exactly.
            self._legend.setFlag(flag, False)
        try:
            exp = ImageExporter(self.plot.plotItem)
            exp.parameters()["width"] = target_width
            exp.export(path)
        finally:
            if self._legend is not None:
                self._legend.setFlag(flag, True)

    def _export_composite(self, path: str, target_width: int) -> None:
        """Docked-legend export: a clean composite with the LEGEND on the
        RIGHT of the map, top-aligned (the legend panel pasted at
        x=map_width, y=0). Never the side panel's buttons or the layer
        list — those were never part of ``self.plot.plotItem`` to begin with
        (only the map canvas is), so excluding them needs no extra work;
        this just also renders the legend block that's no longer ON the map."""
        from pyqtgraph.exporters import ImageExporter
        exp = ImageExporter(self.plot.plotItem)
        exp.parameters()["width"] = target_width
        map_img = exp.export(toBytes=True)

        # Same scale the map just rendered at, so text/line weights match
        # between the two panels of the composite.
        scale = target_width / max(1, self.plot.width())
        flag = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        self._legend.setFlag(flag, False)
        try:
            br = self._legend.boundingRect()
            margin = 10
            legend_w_px = max(50, int(round((br.width() + 2 * margin) * scale)))
            exp2 = ImageExporter(self.dock_vb)
            exp2.parameters()["width"] = legend_w_px
            legend_img = exp2.export(toBytes=True)
        finally:
            self._legend.setFlag(flag, True)

        self._compose_top_right(map_img, legend_img).save(path)

    def _compose_top_right(self, map_img: QImage, legend_img: QImage) -> QImage:
        """Paste the legend block to the RIGHT of the map, top-aligned
        (legend pasted at x=map_width, y=0) — the standard 'legend panel
        beside the map' layout for an exported figure."""
        w = map_img.width() + legend_img.width()
        h = max(map_img.height(), legend_img.height())
        out = QImage(w, h, QImage.Format.Format_RGB32)
        out.fill(QColor(theme.color("panel")))
        p = QPainter(out)
        p.drawImage(0, 0, map_img)
        p.drawImage(map_img.width(), 0, legend_img)
        p.end()
        return out

    def _export_pdf(self, path: str, target_width: int, dpi: int) -> None:
        """PDF export via ``QPdfWriter``, rendering the map's QGraphicsScene
        directly with a QPainter — per the explicit requirement, NEVER
        pyqtgraph's ``SVGExporter`` (it raises on this scene's grid/axis
        content). The page's physical size is derived from
        ``target_width``/``dpi`` so a PDF export reproduces the exact same
        physical scale as a raster export at the same DPI (including under
        the Export Scale Lock — see ``_scale_locked_export_width``).

        A floating legend is part of the SAME scene as the map (it's a
        child item of the plot's ViewBox), so the single scene.render() call
        below already includes it, exactly like ``_export_single``'s
        ImageExporter call. A DOCKED legend lives in a SEPARATE scene
        (``self.dock_vb``) — rendered to its own raster image (reusing the
        same composite sizing as ``_export_composite``) and pasted onto the
        SAME PDF page via drawImage(), positioned right of the map, top-
        aligned — mixing one vector region with one raster region on a
        single PDF page is standard practice, not a workaround."""
        plot_item = self.plot.plotItem
        source_rect = plot_item.sceneBoundingRect()
        aspect = source_rect.height() / max(1.0, source_rect.width())
        map_h_px = max(1, int(round(target_width * aspect)))

        legend_w_px = 0
        legend_img: Optional[QImage] = None
        flag = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        if self._legend is not None and self._legend_docked:
            from pyqtgraph.exporters import ImageExporter
            scale = target_width / max(1, self.plot.width())
            br = self._legend.boundingRect()
            legend_w_px = max(50, int(round((br.width() + 20) * scale)))
            self._legend.setFlag(flag, False)
            try:
                exp = ImageExporter(self.dock_vb)
                exp.parameters()["width"] = legend_w_px
                legend_img = exp.export(toBytes=True)
            finally:
                self._legend.setFlag(flag, True)

        total_w_px = target_width + legend_w_px
        total_h_px = max(map_h_px, legend_img.height() if legend_img is not None else 0)
        print(f"[Scale Math · PDF page] target_width={target_width} px, "
              f"map_height={map_h_px} px, legend_width={legend_w_px} px @ {dpi} DPI "
              f"-> page size {total_w_px / dpi:.4f} x {total_h_px / dpi:.4f} in")

        writer = QPdfWriter(path)
        writer.setResolution(dpi)
        writer.setPageSize(QPageSize(QSizeF(total_w_px / dpi, total_h_px / dpi),
                                     QPageSize.Unit.Inch))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        painter = QPainter(writer)
        # Bug fix (PDF export colors): pyqtgraph items query self._exportOpts
        # to decide HOW they paint for an export render — in particular,
        # ScatterPlotItem (symbol markers: SOL/EOL dots, drawn points, POI
        # stars) only bypasses its cached/pre-rendered symbol pixmap and
        # recomputes its resolutionScale when _exportOpts is set; left at its
        # default (False, i.e. "live interactive paint"), it can reuse a
        # symbol pixmap cached at the LIVE view's screen scale/colors rather
        # than freshly painting at THIS export's resolution. ImageExporter
        # (the PNG/JPG/TIFF path) already calls this — scene.render() alone
        # (what PDF needs, since SVGExporter is unusable here) does not, so
        # it must be armed explicitly to match the raster paths' fidelity.
        from pyqtgraph.exporters import ImageExporter
        resolution_scale = target_width / max(1, self.plot.width())
        export_helper = ImageExporter(plot_item)
        export_helper.setExportMode(True, {
            "antialias": True,
            "background": self.plot.backgroundBrush().color(),
            "painter": painter,
            "resolutionScale": resolution_scale,
        })
        # Bug #6: a FLOATING legend is part of this same scene, so scene.render()
        # paints it — but LegendItem sets ItemIgnoresTransformations, pinning it
        # to a fixed PHYSICAL size that comes out comically tiny on a large PDF
        # page. Clear the flag for the render's duration (restored in finally),
        # exactly as _export_single does for the raster path, so it inherits
        # the export's resolution scale. A DOCKED legend isn't in this scene
        # (it was rendered to legend_img above), so it's unaffected.
        floating_legend = (self._legend if (self._legend is not None
                                            and not self._legend_docked) else None)
        if floating_legend is not None:
            floating_legend.setFlag(flag, False)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            target_rect = QRectF(0, 0, target_width, map_h_px)
            self.plot.scene().render(painter, target_rect, source_rect)
            if legend_img is not None:
                painter.drawImage(target_width, 0, legend_img)
        finally:
            if floating_legend is not None:
                floating_legend.setFlag(flag, True)
            export_helper.setExportMode(False)
            painter.end()

    # ── Layer-list context menu ────────────────────────────────────────────────

    def _on_layer_context_menu(self, pos) -> None:
        items = self.layer_list.selectedItems()
        if not items:
            return
        menu = QMenu(self)
        act_label = menu.addAction(self.tr("Toggle Label on Map"))
        act_label.triggered.connect(lambda: self._toggle_labels(items))
        act_format = menu.addAction(self.tr("Format Labels…"))
        act_format.triggered.connect(lambda: self._format_labels(items))
        menu.addSeparator()
        act_color = menu.addAction(self.tr("Change Color…"))
        act_color.triggered.connect(lambda: self._change_layer_color(items))
        drawn = [li for li in items if bool(li.data(_ROLE_IS_DRAWN))]
        if drawn:
            menu.addSeparator()
            act_shp = menu.addAction(self.tr("Export as .shp…"))
            act_shp.triggered.connect(lambda: self._export_drawn_shapefile(drawn))
        polygons = [li for li in drawn
                   if (li.data(_ROLE_DRAWN_GEOM) or {}).get("geom_type") == "polygon"]
        if len(polygons) == 1:
            act_props = menu.addAction(self.tr("Show Properties…"))
            act_props.triggered.connect(lambda: self._show_layer_properties(polygons[0]))
        menu.exec(self.layer_list.viewport().mapToGlobal(pos))

    # ── "Show Properties…" (Part 4: polygon area/perimeter) ────────────────────

    def _to_metric_points(self, points: list) -> list:
        """Scale a list of (x, y) points to real-world metres. Geographic
        (WGS84 degree) coordinates are scaled to local tangent-plane metres
        using the same latitude-cosine calibration as the scale bar (see
        _scale_calibration_factor) — 1° of longitude shrinks toward the
        poles while 1° of latitude stays ~constant (111 320 m). Projected/
        UTM coordinates are already metric and pass through unchanged."""
        if not points:
            return []
        if self._coords_are_geographic():
            center_lat = sum(p[1] for p in points) / len(points)
            m_x = 111_320.0 * max(0.01, math.cos(math.radians(center_lat)))
            m_y = 111_320.0
            return [(x * m_x, y * m_y) for x, y in points]
        return [(float(x), float(y)) for x, y in points]

    def _polygon_metrics(self, points: list) -> Tuple[float, float]:
        """Real-world (area m², perimeter m) of a polygon ring (UNCLOSED —
        matches the stored _ROLE_DRAWN_GEOM convention). Points are scaled
        to local tangent-plane metres first (see _to_metric_points) before
        applying the shoelace formula and summing segment lengths —
        otherwise the result would be in physically meaningless
        square-degrees for geographic coordinates."""
        if len(points) < 2:
            return 0.0, 0.0
        pts_m = self._to_metric_points(points)
        n = len(pts_m)
        area2 = 0.0
        for i in range(n):
            x1, y1 = pts_m[i]
            x2, y2 = pts_m[(i + 1) % n]
            area2 += x1 * y2 - x2 * y1
        area = abs(area2) / 2.0
        perimeter = self._path_length(pts_m + [pts_m[0]])
        return area, perimeter

    def _show_layer_properties(self, li: QListWidgetItem) -> None:
        geom = li.data(_ROLE_DRAWN_GEOM) or {}
        points = geom.get("coords") or []
        area_m2, perim_m = self._polygon_metrics(points)
        name = li.data(_ROLE_NAME) or li.text()
        area_text = (f"{area_m2 / 1_000_000.0:,.4f} km²" if area_m2 >= 1_000_000.0
                    else f"{area_m2:,.2f} m²")
        perim_text = (f"{perim_m / 1000.0:,.4f} km" if perim_m >= 1000.0
                      else f"{perim_m:,.2f} m")
        QMessageBox.information(
            self, self.tr("Layer Properties"),
            self.tr("{0}\n\nArea: {1}\nPerimeter: {2}").format(name, area_text, perim_text))

    # ── Map-canvas name labels ("Toggle Label on Map") ─────────────────────────

    def _toggle_labels(self, items: list) -> None:
        """Batch toggle for a multi-selection: if any selected row has its
        label off, turn labels ON for the whole selection; only turn them all
        OFF once every selected row already has one showing."""
        any_off = any(not bool(li.data(_ROLE_LABEL_ON)) for li in items)
        for li in items:
            self._set_label_visible(li, any_off)

    def _set_label_visible(self, li: QListWidgetItem, on: bool) -> None:
        """Arm/disarm a row's label. The label only actually appears while
        BOTH this is True AND the row's own checkbox is checked (see
        _on_layer_item_changed) — a hidden layer never shows a stray label."""
        if bool(li.data(_ROLE_IS_BASEMAP)):
            return   # no meaningful label anchor for the basemap
        li.setData(_ROLE_LABEL_ON, bool(on))
        layer_visible = li.checkState() == Qt.CheckState.Checked
        if on and layer_visible:
            self._show_label(li)
        else:
            self._hide_label(li)

    def _show_label(self, li: QListWidgetItem) -> None:
        anchor = li.data(_ROLE_ANCHOR)
        if anchor is None:
            return
        text_item = li.data(_ROLE_LABEL_ITEM)
        name = li.data(_ROLE_NAME) or li.text()
        if text_item is None:
            text_item = pg.TextItem(name, anchor=(0.5, 0.5))
            text_item.setZValue(50)        # always readable, above every layer
            self.plot.getViewBox().addItem(text_item, ignoreBounds=True)
            li.setData(_ROLE_LABEL_ITEM, text_item)
        else:
            text_item.setText(name)
        # Bug fix ("labels vanish when changing text size"): pg.TextItem keeps
        # text at a constant ON-SCREEN size by computing the inverse of its
        # parent's scene transform in updateTransform() — but that method's
        # FIRST line is `if not self.isVisible(): return`, silently skipping
        # the recompute while the item is still hidden. setVisible(True) used
        # to run LAST here, after setAngle() (which triggers
        # updateTransform()) had already run — so on a freshly re-shown label
        # the transform could be stale/never-computed, and a size change
        # (which only repositions via updateTextPos(), it doesn't itself
        # refresh the transform) rendered against that stale state. Making
        # the item visible FIRST, then changing font/position, then setting
        # the angle LAST (forcing one fresh, correct transform computation
        # with the final size/visibility already in effect) fixes it.
        text_item.setVisible(True)
        offset = li.data(_ROLE_LABEL_OFFSET) or (0.0, 0.0)
        text_item.setPos(anchor[0] + offset[0], anchor[1] + offset[1])
        font = text_item.textItem.font()
        font.setPointSize(int(li.data(_ROLE_LABEL_SIZE) or _LABEL_DEFAULT_SIZE))
        text_item.setFont(font)
        text_item.setAngle(float(li.data(_ROLE_LABEL_ROTATION) or 0.0))
        self._restyle_label(text_item)

    # ── "Format Labels…" (rotation / offset / size) ────────────────────────────

    def _format_labels(self, items: list) -> None:
        items = [li for li in items if not bool(li.data(_ROLE_IS_BASEMAP))]
        if not items:
            return
        first = items[0]
        rotation = float(first.data(_ROLE_LABEL_ROTATION) or 0.0)
        offset = first.data(_ROLE_LABEL_OFFSET) or (0.0, 0.0)
        size = int(first.data(_ROLE_LABEL_SIZE) or _LABEL_DEFAULT_SIZE)
        # Per-row originals (a multi-selection can have DIFFERING starting
        # values) — restored exactly on Cancel, since every slider drag now
        # applies live to the map, not just on accept.
        originals = {id(li): (float(li.data(_ROLE_LABEL_ROTATION) or 0.0),
                              li.data(_ROLE_LABEL_OFFSET) or (0.0, 0.0),
                              int(li.data(_ROLE_LABEL_SIZE) or _LABEL_DEFAULT_SIZE))
                    for li in items}

        def apply_to_all(rot: float, off: Tuple[float, float], sz: int) -> None:
            for li in items:
                li.setData(_ROLE_LABEL_ROTATION, rot)
                li.setData(_ROLE_LABEL_OFFSET, off)
                li.setData(_ROLE_LABEL_SIZE, sz)
                if li.data(_ROLE_LABEL_ON) and li.checkState() == Qt.CheckState.Checked:
                    self._show_label(li)

        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        view_extent = (abs(x1 - x0), abs(y1 - y0))
        dlg = _LabelFormatDialog(self, rotation, offset, size, view_extent,
                                 on_change=apply_to_all)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            for li in items:                       # revert each row's OWN original
                apply_to_all_one = originals[id(li)]
                li.setData(_ROLE_LABEL_ROTATION, apply_to_all_one[0])
                li.setData(_ROLE_LABEL_OFFSET, apply_to_all_one[1])
                li.setData(_ROLE_LABEL_SIZE, apply_to_all_one[2])
                if li.data(_ROLE_LABEL_ON) and li.checkState() == Qt.CheckState.Checked:
                    self._show_label(li)
            return
        apply_to_all(*dlg.values())   # already live-applied; re-applied for robustness

    def _hide_label(self, li: QListWidgetItem) -> None:
        text_item = li.data(_ROLE_LABEL_ITEM)
        if text_item is not None:
            text_item.setVisible(False)

    def _remove_label(self, li: QListWidgetItem) -> None:
        """Fully release a row's label item (on layer removal, not just hide)."""
        text_item = li.data(_ROLE_LABEL_ITEM)
        if text_item is not None:
            self.plot.getViewBox().removeItem(text_item)
            li.setData(_ROLE_LABEL_ITEM, None)

    def _restyle_label(self, text_item) -> None:
        """Pure text — no background fill, border, or bounding box (pg.TextItem
        defaults to fill=None/border=None already; this only ever sets the
        text colour, never overrides those back to a visible brush/pen)."""
        text_item.setColor(theme.color("text"))
        text_item.update()

    # ── "Change Color…" ─────────────────────────────────────────────────────────

    def _change_layer_color(self, items: list) -> None:
        items = [li for li in items if not bool(li.data(_ROLE_IS_BASEMAP))]
        if not items:
            return
        current = QColor(items[0].data(_ROLE_COLOR) or "#ffffff")
        color = QColorDialog.getColor(current, self, self.tr("Change Color"))
        if color.isValid():
            for li in items:
                self._apply_layer_color(li, color)

    def _apply_layer_color(self, li: QListWidgetItem, color: QColor) -> None:
        """Recolor a layer's primary visual item in place (line pen, and its
        point symbols if 'Show Points' is on) and persist the choice
        (_ROLE_COLOR) so re-renders / the legend swatch stay in sync. Only the
        layer's PRIMARY item (items[0]) is recoloured — secondary markers with
        their own fixed meaning (the SOL/EOL green/red dots) are untouched."""
        if bool(li.data(_ROLE_IS_BASEMAP)):
            return
        li.setData(_ROLE_COLOR, color.name())
        items = li.data(Qt.ItemDataRole.UserRole) or []
        if not items:
            return
        gi = items[0]
        if isinstance(gi, pg.PlotDataItem):
            old_pen = gi.opts.get("pen")
            width = old_pen.widthF() if old_pen is not None else 2.0
            gi.setPen(pg.mkPen(color, width=width))
            if gi.opts.get("symbol") is not None:
                gi.setSymbolBrush(color)
                gi.setSymbolPen(color)
        elif isinstance(gi, pg.ScatterPlotItem):
            gi.setBrush(pg.mkBrush(color))
            gi.setPen(pg.mkPen(color))
        elif isinstance(gi, QGraphicsPathItem):
            pen = gi.pen(); pen.setColor(color); gi.setPen(pen)
            brush = gi.brush()
            if brush.style() != Qt.BrushStyle.NoBrush:
                fill = QColor(color); fill.setAlpha(brush.color().alpha())
                gi.setBrush(QBrush(fill))
        # ImageItem (raster): colour doesn't apply — left untouched.
        self._legend_dirty()    # the swatch reads the item's pen fresh each refresh

    # ── "Show Points" (global) ──────────────────────────────────────────────────

    def _on_show_points_toggled(self, checked: bool) -> None:
        self._show_points = bool(checked)
        self._apply_show_points()

    def _apply_show_points(self) -> None:
        """Show/hide the start/end location markers (SOL/EOL dots) on every
        track-type layer — the live SEG-Y track's green/red anchors, and any
        add_track_layer row's own start/end dots (_ROLE_MARKERS). This does
        NOT draw vertices along the line itself — an earlier version of this
        toggle did that, which was the wrong behaviour: the request is to
        hide/show the existing default markers, not add new ones."""
        for row in range(self.layer_list.count()):
            for gi in (self.layer_list.item(row).data(_ROLE_MARKERS) or []):
                gi.setVisible(self._show_points)

    # ── Legend ──────────────────────────────────────────────────────────────────

    def _legend_dirty(self) -> None:
        """Repopulate the legend if it's currently shown; no-op otherwise."""
        if self._legend is not None:
            self._refresh_legend()

    def _on_legend_toggled(self, checked: bool) -> None:
        if checked:
            if self._legend is None:
                self._legend = _DraggableLegend(self, offset=(10, -10))   # bottom-left
                self._legend.setParentItem(self.plot.getViewBox())
                self._restyle_legend()
            self._refresh_legend()
        elif self._legend is not None:
            # ViewBox.removeItem() calls setParentItem(None) internally, and
            # pyqtgraph's LegendItem.setParentItem unconditionally re-anchors
            # using its stored 'offset' option whenever it's called — even to
            # detach — which raises once the parent is gone. Clearing the
            # option first short-circuits that branch (it's a plain `if
            # self.opts['offset'] is not None` check) without otherwise
            # touching the item.
            self._legend.opts["offset"] = None
            vb = self.dock_vb if self._legend_docked else self.plot.getViewBox()
            vb.removeItem(self._legend)
            self._legend = None
            self._legend_rows = []
            self._legend_docked = False
            self.dock_legend_view.setVisible(False)

    def _legend_sample(self, li: QListWidgetItem):
        """A legend-swatch-compatible 'sample' for this row. pyqtgraph's
        ItemSample.paint() reads ``item.opts['pen']`` — only PlotDataItem /
        ScatterPlotItem expose that, so a raw QGraphicsPathItem (polygon/line)
        gets a tiny throwaway PlotDataItem carrying the same pen, used only
        for the swatch icon. A raster (ImageItem) gets an actual image
        thumbnail instead — a line/pen swatch is meaningless for it."""
        items = li.data(Qt.ItemDataRole.UserRole) or []
        if not items:
            return None
        first = items[0]
        if isinstance(first, (pg.PlotDataItem, pg.ScatterPlotItem)):
            return first
        if isinstance(first, pg.ImageItem):
            return _RasterItemSample(first)
        pen = first.pen() if hasattr(first, "pen") else pg.mkPen(theme.color("text"))
        return pg.PlotDataItem([], [], pen=pen)

    def _refresh_legend(self) -> None:
        """Repopulate from the currently VISIBLE (checked), non-basemap,
        non-legend-hidden rows. A row's custom caption (``_ROLE_LEGEND_TEXT``,
        set via double-click / right-click on an entry) survives the
        repopulate; everything else falls back to the layer's own list name.

        Because this fully clears and rebuilds the LegendItem from scratch
        every time, a row hidden via ``_ROLE_LEGEND_HIDDEN`` is simply never
        re-added — there's no separate "remove + repack" step needed, the
        remaining entries are tightly packed by construction."""
        if self._legend is None:
            return
        self._legend.clear()
        self._legend_rows = []
        for row in range(self.layer_list.count()):
            li = self.layer_list.item(row)
            if (bool(li.data(_ROLE_IS_BASEMAP)) or li.checkState() != Qt.CheckState.Checked
                    or bool(li.data(_ROLE_LEGEND_HIDDEN))):
                continue
            sample = self._legend_sample(li)
            if sample is None:
                continue
            text = li.data(_ROLE_LEGEND_TEXT) or li.data(_ROLE_NAME) or li.text()
            self._legend.addItem(sample, text)
            self._legend_rows.append(li)
        self._fit_dock_legend()

    def _restyle_legend(self) -> None:
        if self._legend is None:
            return
        self._legend.setBrush(pg.mkBrush(theme.color("panel")))
        self._legend.setPen(pg.mkPen(theme.color("sub")))
        self._legend.setLabelTextColor(theme.color("text"))

    def _set_legend_anchor(self, side: str) -> None:
        """Snap the legend to a bottom corner — 'left' (default) or 'right'
        (the conventional placement for map exports). Free dragging still
        works afterwards; this is just a quick preset."""
        if self._legend is None:
            return
        anchor = (1, 1) if side == "right" else (0, 1)
        offset = (-10, -10) if side == "right" else (10, -10)
        self._legend.anchor(itemPos=anchor, parentPos=anchor, offset=offset)

    def _show_legend_context_menu(self, row: Optional[int], screen_pos) -> None:
        menu = QMenu(self)
        if row is not None:
            act_edit = menu.addAction(self.tr("Edit legend entry"))
            act_edit.triggered.connect(lambda: self._edit_legend_entry(row))
            act_hide = menu.addAction(self.tr("Hide from legend"))
            act_hide.triggered.connect(lambda: self._hide_legend_entry(row))
            menu.addSeparator()
        else:
            hidden = self._hidden_legend_layers()
            if hidden:
                sub = menu.addMenu(self.tr("Hidden items…"))
                for li in hidden:
                    name = li.data(_ROLE_LEGEND_TEXT) or li.data(_ROLE_NAME) or li.text()
                    act = sub.addAction(name)
                    act.triggered.connect(lambda _checked=False, li=li: self._restore_legend_entry(li))
                menu.addSeparator()
        act_left = menu.addAction(self.tr("Move to bottom-left"))
        act_right = menu.addAction(self.tr("Move to bottom-right"))
        act_left.triggered.connect(lambda: self._set_legend_anchor("left"))
        act_right.triggered.connect(lambda: self._set_legend_anchor("right"))
        menu.addSeparator()
        if self._legend_docked:
            act_dock = menu.addAction(self.tr("Float on map"))
            act_dock.triggered.connect(self._undock_legend)
        else:
            act_dock = menu.addAction(self.tr("Dock to side panel"))
            act_dock.triggered.connect(self._dock_legend)
        menu.exec(screen_pos.toPoint())

    def _hide_legend_entry(self, row: int) -> None:
        """Remove a row from the legend only — the layer itself stays fully
        visible on the map. _refresh_legend's full clear-and-rebuild means
        the remaining entries automatically repack with no gap."""
        if self._legend is None or row >= len(self._legend_rows):
            return
        self._legend_rows[row].setData(_ROLE_LEGEND_HIDDEN, True)
        self._refresh_legend()

    def _restore_legend_entry(self, li: QListWidgetItem) -> None:
        li.setData(_ROLE_LEGEND_HIDDEN, False)
        self._legend_dirty()

    def _hidden_legend_layers(self) -> list:
        return [self.layer_list.item(row) for row in range(self.layer_list.count())
               if bool(self.layer_list.item(row).data(_ROLE_LEGEND_HIDDEN))]

    def _dock_legend(self) -> None:
        """Move the legend off the map canvas into the side panel's bare host
        ViewBox (dock_legend_view).

        NOT a re-parent of the same item: the map's PlotWidget and the dock
        host's GraphicsLayoutWidget are two SEPARATE QGraphicsScenes, and
        directly calling setParentItem() to move a live item across scenes
        segfaults (confirmed empirically — Qt/pyqtgraph does not support it
        for a composite GraphicsWidget like LegendItem). So instead: cleanly
        detach the old instance (same safe path as turning the legend off)
        and construct a FRESH one native to the dock host's own scene. No
        data loss — captions live on the layer rows (_ROLE_LEGEND_TEXT), not
        on the legend object, so the repopulate restores them exactly.

        offset=(8, 8) gives it an explicit top-left anchor with a small
        inset margin — without an offset at all, LegendItem.setParentItem()
        never calls anchor() (its auto-anchor branch is gated on
        ``opts['offset'] is not None``), so the legend just sat at Qt's
        default (0, 0): flush against the host's edge with zero margin,
        which read as clipped/unpolished."""
        if self._legend is None or self._legend_docked:
            return
        self._legend.opts["offset"] = None
        self.plot.getViewBox().removeItem(self._legend)
        self._legend_docked = True
        self.dock_legend_view.setVisible(True)
        self._legend = _DraggableLegend(self, offset=(8, 8))   # top-left, inset
        self._legend.setParentItem(self.dock_vb)
        self._restyle_legend()
        self._refresh_legend()

    def _undock_legend(self) -> None:
        """Reverse of _dock_legend — see its docstring for why this rebuilds
        a fresh instance rather than re-parenting the same one."""
        if self._legend is None or not self._legend_docked:
            return
        self._legend.opts["offset"] = None
        self.dock_vb.removeItem(self._legend)
        self._legend_docked = False
        self.dock_legend_view.setVisible(False)
        self._legend = _DraggableLegend(self, offset=(10, -10))
        self._legend.setParentItem(self.plot.getViewBox())
        self._restyle_legend()
        self._refresh_legend()

    def _fit_dock_legend(self) -> None:
        """Frame the dock host's ViewBox so the legend (parented directly to
        it, not auto-tracked the way addItem()'d content is) is fully
        visible, with a margin matching its own top-left anchor inset (see
        _dock_legend) on every side — and resize the HOST WIDGET itself to
        match the legend's actual content height. The host has no fixed
        height (see _build_layer_panel), so a legend with many entries always
        fits instead of being squeezed/clipped into a fixed-size box."""
        if not self._legend_docked or self._legend is None:
            return
        br = self._legend.boundingRect()
        margin = 8.0   # same inset used for the legend's top-left anchor offset
        w = max(50.0, br.width() + 2 * margin)
        h = max(30.0, br.height() + 2 * margin)
        self.dock_legend_view.setFixedHeight(int(round(h)))
        self.dock_vb.setRange(QRectF(0, 0, w, h), padding=0)

    def _edit_legend_entry(self, row: int) -> None:
        """Rename a legend entry's caption WITHOUT touching the layer's own
        name in the Layers list — stored separately on the row
        (_ROLE_LEGEND_TEXT) so it persists across legend repopulates."""
        if self._legend is None or row >= len(self._legend_rows):
            return
        li = self._legend_rows[row]
        _sample, label = self._legend.items[row]
        new_text, ok = QInputDialog.getText(
            self, self.tr("Edit legend entry"), self.tr("Text:"), text=label.text)
        new_text = new_text.strip()
        if ok and new_text:
            li.setData(_ROLE_LEGEND_TEXT, new_text)
            label.setText(new_text)

    def add_layer(self, layer) -> None:
        """Render a core VectorLayer / RasterLayer overlay (called by the tab
        after the core reader runs off-thread). Coordinates are already WGS84."""
        from ...core.gis_io import RasterLayer, VectorLayer
        # Bug #10: with no SEG-Y track loaded, a standalone GIS overlay's own
        # CRS-derived flag classifies the map (so the basemap / scale bar /
        # cursor units are right even when viewing only an overlay). Never
        # overrides a track's already-established hint.
        if (self._x is None and self._geographic_hint is None
                and getattr(layer, "is_geographic", None) is not None):
            self._geographic_hint = layer.is_geographic
        if isinstance(layer, RasterLayer):
            self._add_raster(layer)
        elif isinstance(layer, VectorLayer):
            self._add_vector(layer)

    def _add_vector(self, layer) -> None:
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        anchor: Optional[Tuple[float, float]] = None
        if layer.geom_type == "point":
            pts = np.vstack(layer.paths) if layer.paths else np.empty((0, 2))
            item = pg.ScatterPlotItem(x=pts[:, 0], y=pts[:, 1], size=7,
                                      brush=pg.mkBrush(color), pen=None)
            if pts.size:
                anchor = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        else:
            path = QPainterPath()
            for arr in layer.paths:
                if len(arr) < 2:
                    continue
                path.addPolygon(QPolygonF([QPointF(float(x), float(y))
                                           for x, y in arr]))
                if layer.geom_type == "polygon":
                    path.closeSubpath()
                if anchor is None:           # label at the midpoint of the FIRST path
                    mid = arr[len(arr) // 2]
                    anchor = (float(mid[0]), float(mid[1]))
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(color)); pen.setCosmetic(True); pen.setWidthF(1.5)
            item.setPen(pen)
            if layer.geom_type == "polygon":
                fill = QColor(color); fill.setAlpha(60)
                item.setBrush(QBrush(fill))
            else:
                item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._register_layer(layer.name, [item], "vector", anchor=anchor)

    def add_track_layer(self, name: str, x, y, source_id: Optional[str] = None) -> None:
        """Inject an independent navigation track as a managed GIS layer.

        Used by the sidebar 'Add to map' batch loader: each selected profile /
        chain becomes its own reorderable, checkable, opacity-controlled overlay
        (a polyline with green SOL / red EOL anchors). This is a PINNED reference
        layer — it does NOT touch the active SEG-Y track row, the active profile,
        or any trace data. Coordinates are expected already in WGS84 (lon, lat),
        reprojected off-thread by the caller's CoreWorker.

        Separate files are NEVER stitched together here: each call draws its
        OWN independent polyline (one PlotDataItem per file/chain), so a real
        gap between two unrelated surveys is rendered as a real gap, never a
        straight line across it. A chain's own track is drawn continuous
        because its caller already concatenated it deliberately — see
        ProfileChain.track_lons in core/model.py — that decision belongs to
        whoever built the chain, not to this method.

        ``source_id`` (File Switching, Part 1) identifies which profile/chain
        this layer represents — a profile path, or "chain:<label>" — so a
        click on it can emit layer_source_clicked and let the tab activate
        that exact file. None for a layer with no such backing source.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            return
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        line = pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=2))
        if source_id is not None:
            # CRITICAL: the click only reaches sigClicked if the underlying
            # PlotCurveItem is made CLICKABLE — setAcceptedMouseButtons alone
            # is not enough, because PlotCurveItem.mouseClickEvent early-
            # returns on `not self.clickable` so sigClicked never fires. A
            # generous click width gives the thin polyline a forgiving hit
            # zone. The curve's own mouseClickEvent ACCEPTS the event, so the
            # scene-level _on_click bails (ev.isAccepted()) and never also
            # recentres the active profile (see _on_click).
            line.setCurveClickable(True, width=8)
            line.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

            def _on_track_clicked(_item, ev, _x=x, _y=y, sid=source_id) -> None:
                p = self.plot.getViewBox().mapSceneToView(ev.scenePos())
                idx = int(np.argmin((_x - p.x()) ** 2 + (_y - p.y()) ** 2))
                self.layer_source_clicked.emit(sid, idx)

            line.sigClicked.connect(_on_track_clicked)
        else:
            line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        start = pg.ScatterPlotItem([x[0]], [y[0]], size=8,
                                   brush=pg.mkBrush(theme.color("ok")), pen=None)
        end = pg.ScatterPlotItem([x[-1]], [y[-1]], size=8,
                                 brush=pg.mkBrush(theme.color("warn")), pen=None)
        for it in (start, end):
            it.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        mid = len(x) // 2
        li = self._register_layer(name, [line, start, end], "vector",
                                  anchor=(float(x[mid]), float(y[mid])), markers=[start, end])
        li.setData(_ROLE_SOURCE_ID, source_id)
        # Show the world basemap if these coordinates look geographic, then frame
        # the newly added geometry so the user sees it land (autoRange ignores the
        # basemap, which is added with ignoreBounds=True).
        if (float(np.nanmin(x)) >= -180.0 and float(np.nanmax(x)) <= 180.0
                and float(np.nanmin(y)) >= -90.0 and float(np.nanmax(y)) <= 90.0):
            self._basemap.setVisible(self._basemap_checked())   # respect the user's toggle
        self.plot.getViewBox().autoRange()

    def _add_raster(self, layer) -> None:
        lon0, lon1, lat0, lat1 = layer.bbox
        item = pg.ImageItem()
        # Row 0 of a GeoTIFF is the NORTH edge; flip so it maps to the top of the
        # geographic rect (MapView Y increases northward).
        item.setImage(np.flipud(np.asarray(layer.image)), autoLevels=True)
        item.setRect(QRectF(lon0, lat0, lon1 - lon0, lat1 - lat0))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._register_layer(layer.name, [item], "raster",
                             anchor=((lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0))

    def _relabel(self, li: QListWidgetItem, tag: str, name: str) -> None:
        """Single source of truth for a row's displayed text: ``tag + name``.
        Stores both parts separately (``_ROLE_TAG``/``_ROLE_NAME``) so a rename
        only ever touches the name, never the tag glyph. Also keeps an active
        on-map label (if shown) and an unedited legend caption in sync, since
        both default to this same display name."""
        li.setData(_ROLE_TAG, tag)
        li.setData(_ROLE_NAME, name)
        li.setText(f"{tag}  {name}" if tag else name)
        if li.data(_ROLE_LABEL_ON):
            self._show_label(li)
        if li.data(_ROLE_LEGEND_TEXT) is None:
            self._legend_dirty()

    def _register_layer(self, name: str, gitems: list, kind: str,
                        anchor: Optional[Tuple[float, float]] = None,
                        markers: Optional[list] = None) -> QListWidgetItem:
        for gi in gitems:
            self.plot.getViewBox().addItem(gi, ignoreBounds=True)
        self._layer_count += 1
        tag = "▦" if kind == "raster" else "▤"
        li = QListWidgetItem()
        li.setFlags(li.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        li.setCheckState(Qt.CheckState.Checked)
        li.setData(Qt.ItemDataRole.UserRole, list(gitems))
        li.setData(_ROLE_ANCHOR, anchor)
        li.setData(_ROLE_MARKERS, markers)
        self._relabel(li, tag, name)
        # Newest layer on top, but keep the track row on top by default (the user
        # can still drag a layer above it).
        row0_is_track = (self.layer_list.count() > 0
                         and bool(self.layer_list.item(0).data(_ROLE_IS_TRACK)))
        self.layer_list.insertItem(1 if row0_is_track else 0, li)
        self._reorder_layers()
        self._legend_dirty()
        self._apply_show_points()      # new layer respects the current global toggle
        self._invalidate_raster_cache()   # a new layer may be a raster (Bug #12)
        return li

    # ── The SEG-Y track as a managed layer ────────────────────────────────────

    def _ensure_track_layer(self) -> None:
        """Inject the 'Navigation / Track' row at the top of the list (once). Its
        four graphics items (full line, visible segment, SOL/EOL dots) are managed
        as one layer — reorderable and opacity-controlled like any GIS overlay."""
        if self._track_item is not None:
            return
        items = [self.curve_full, self.curve_seg, self.marker_start, self.marker_end]
        li = QListWidgetItem()
        li.setFlags(li.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        li.setCheckState(Qt.CheckState.Checked)
        li.setData(Qt.ItemDataRole.UserRole, items)
        li.setData(_ROLE_IS_TRACK, True)
        li.setData(_ROLE_MARKERS, [self.marker_start, self.marker_end])
        self._track_item = li
        self.layer_list.insertItem(0, li)
        self._set_track_item_text()
        self._reorder_layers()
        self._apply_show_points()

    def _set_track_item_text(self) -> None:
        if self._track_item is not None and not self._track_item.data(_ROLE_RENAMED):
            self._relabel(self._track_item, "≈", self.tr("Navigation / Track"))

    def _update_track_anchor(self) -> None:
        """Recompute the live SEG-Y track's label anchor (its midpoint) — the
        underlying ``_x``/``_y`` change on every ``set_track`` call, unlike a
        static GIS layer's geometry, so this can't be a one-time registration
        step like ``_register_layer``'s ``anchor=``."""
        if self._track_item is None or self._x is None or self._x.size == 0:
            return
        mid = self._x.size // 2
        self._track_item.setData(_ROLE_ANCHOR, (float(self._x[mid]), float(self._y[mid])))
        if self._track_item.data(_ROLE_LABEL_ON):
            self._show_label(self._track_item)   # reposition if currently shown

    def _reorder_layers(self) -> None:
        """Map list order → zValue in the custom band (top row highest), keeping
        every layer strictly between the basemap (-1000) and z=0. Multi-item
        layers (the track) get tiny within-group offsets so their internal
        stacking (line < segment < markers) is preserved. The basemap row is
        excluded — it stays pinned at ``_BASEMAP_Z`` regardless of its position
        in the list."""
        for row in range(self.layer_list.count()):
            li = self.layer_list.item(row)
            if bool(li.data(_ROLE_IS_BASEMAP)):
                continue
            items = li.data(Qt.ItemDataRole.UserRole) or []
            base = max(_LAYER_Z_BOTTOM, _LAYER_Z_TOP - row)
            for j, gi in enumerate(items):
                gi.setZValue(base + j * 0.1)

    def _on_layer_item_changed(self, li: QListWidgetItem) -> None:
        if bool(li.data(_ROLE_IS_BASEMAP)):
            # The basemap's visibility also depends on the geographic heuristic,
            # not just the checkbox — see _apply_basemap_visibility.
            self._apply_basemap_visibility()
            return
        visible = li.checkState() == Qt.CheckState.Checked
        for gi in (li.data(Qt.ItemDataRole.UserRole) or []):
            gi.setVisible(visible)
        # A hidden layer's label must disappear with it; a re-shown layer that
        # had its label armed (_ROLE_LABEL_ON) gets it back automatically.
        if visible and li.data(_ROLE_LABEL_ON):
            self._show_label(li)
        else:
            self._hide_label(li)
        self._legend_dirty()        # the legend only lists currently-visible layers
        self._invalidate_raster_cache()   # visibility change affects the Z-readout (Bug #12)

    def _on_current_layer_changed(self, current: Optional[QListWidgetItem],
                                  _previous: Optional[QListWidgetItem]) -> None:
        """Sync the opacity slider to the newly selected layer's opacity."""
        self.opacity_slider.setEnabled(current is not None)
        if current is None:
            return
        items = current.data(Qt.ItemDataRole.UserRole) or []
        op = items[0].opacity() if items else 1.0
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(int(round(op * 100)))
        self.opacity_slider.blockSignals(False)

    def _on_opacity_changed(self, value: int) -> None:
        li = self.layer_list.currentItem()
        if li is None:
            return
        op = max(0.0, min(1.0, value / 100.0))
        for gi in (li.data(Qt.ItemDataRole.UserRole) or []):
            gi.setOpacity(op)

    def _remove_selected_layer(self) -> None:
        li = self.layer_list.currentItem()
        if li is None or bool(li.data(_ROLE_IS_TRACK)) or bool(li.data(_ROLE_IS_BASEMAP)):
            return       # the track and the basemap are intrinsic — hide, don't orphan
        self.layer_list.takeItem(self.layer_list.row(li))
        for gi in (li.data(Qt.ItemDataRole.UserRole) or []):
            self.plot.getViewBox().removeItem(gi)
        self._remove_label(li)      # a removed layer's label must not linger
        self._reorder_layers()
        self._legend_dirty()
        self._invalidate_raster_cache()   # Bug #12

    # ── Rename ──────────────────────────────────────────────────────────────

    def _on_rename_layer(self, li: QListWidgetItem) -> None:
        """Double-click a row to rename it. Edits only the bare name (the tag
        glyph is reapplied untouched), and marks the row so a later language
        switch never overwrites a user-chosen name for the built-in rows."""
        current_name = li.data(_ROLE_NAME) or li.text()
        new_name, ok = QInputDialog.getText(
            self, self.tr("Rename layer"), self.tr("Name:"), text=current_name)
        new_name = new_name.strip()
        if ok and new_name:
            li.setData(_ROLE_RENAMED, True)
            self._relabel(li, li.data(_ROLE_TAG) or "", new_name)

    # ── Map → Profile (click to jump) / drawing / context menu ────────────────

    def _invalidate_raster_cache(self) -> None:
        """Mark the visible-raster cache stale (Bug #12) — called on any layer
        add/remove/visibility change. Rebuilt lazily on the next hover."""
        self._raster_cache = None

    def _visible_rasters(self) -> list:
        """Currently-visible raster ImageItems, cached so the per-move hover
        handler never re-walks the whole layer tree (Bug #12). Rebuilt only
        when the cache was invalidated."""
        if self._raster_cache is None:
            cache: list = []
            for row in range(self.layer_list.count()):
                li = self.layer_list.item(row)
                if li.checkState() != Qt.CheckState.Checked:
                    continue
                for it in (li.data(Qt.ItemDataRole.UserRole) or []):
                    if isinstance(it, pg.ImageItem) and it.image is not None:
                        cache.append(it)
            self._raster_cache = cache
        return self._raster_cache

    def _on_mouse_moved_coords(self, scene_pos) -> None:
        """Live Cursor Coordinates (Part 3): docked readout below the map,
        updated on every mouse move. Lon/Lat for geographic tracks, plain X/Y
        (already metric) for projected/UTM ones. If the cursor is over a
        visible GeoTIFF raster, the sampled Z-value is appended.

        Bug #13: stays blank until there's actually something loaded (a track
        or a raster), and renders a blank placeholder rather than a literal
        'nan' if the mapped coordinate is non-finite (e.g. a degenerate view
        transform / GPS-dropout vertex region)."""
        has_rasters = bool(self._visible_rasters())
        if self._x is None and not has_rasters:
            self.lbl_cursor_coords.setText("")
            return
        pt = self.plot.getViewBox().mapSceneToView(scene_pos)
        x, y = pt.x(), pt.y()
        if not (math.isfinite(x) and math.isfinite(y)):
            self.lbl_cursor_coords.setText("—")     # blank placeholder, never 'nan'
            return
        if self._coords_are_geographic():
            text = self.tr("Lon: {0:.5f}°  Lat: {1:.5f}°").format(x, y)
        else:
            text = self.tr("X: {0:,.2f} m  Y: {1:,.2f} m").format(x, y)
        z = self._sample_raster_z(pt) if has_rasters else None
        if z is not None:
            text += "  " + self.tr("Z: {0:,.3f}").format(z)
        self.lbl_cursor_coords.setText(text)

    def _sample_raster_z(self, pt: QPointF) -> Optional[float]:
        """GeoTIFF Z-value (Part 3): if ``pt`` (data-space) falls within a
        currently VISIBLE raster layer, map it to that raster's array
        row/col and return the sampled value — the elevation/amplitude/
        whatever scalar the GeoTIFF encodes. The TOPMOST visible raster
        (highest zValue) wins if several overlap, matching what's actually
        drawn on top. None if no raster covers the point, the layer is an
        RGB(A) image (no single scalar Z), or the sample is non-finite
        (nodata)."""
        best_item, best_rect, best_z = None, None, None
        # Bug #12: iterate only the cached visible-raster list, not the whole
        # layer tree, on every mouse move.
        for it in self._visible_rasters():
            if it.image is None:
                continue
            # ImageItem.boundingRect() is in its own LOCAL pixel space
            # (e.g. (0,0,W,H)) — setRect() only applies an internal
            # transform, it does NOT change what boundingRect() reports.
            # mapRectToParent() resolves the actual DATA-space rect.
            rect = it.mapRectToParent(it.boundingRect())
            # Bug #9: a degenerate GeoTIFF bbox (lon0==lon1 / lat0==lat1,
            # e.g. a 1-pixel raster or a pathological reprojection) gives a
            # zero-width/height rect — sampling it below would divide by
            # zero inside this high-frequency mouse-move handler. Skip it.
            if rect.width() <= 0 or rect.height() <= 0:
                continue
            if not rect.contains(pt):
                continue
            if best_z is None or it.zValue() > best_z:
                best_item, best_rect, best_z = it, rect, it.zValue()
        if best_item is None:
            return None
        arr = best_item.image
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]   # redundant single-band axis — treat as scalar
        h, w = arr.shape[0], arr.shape[1]
        col = int((pt.x() - best_rect.x()) / best_rect.width() * w)
        # Row 0 of this (already-flipped, see _add_raster) array sits at the
        # rect's BOTTOM (south); row increases toward the TOP (north) — so
        # row is a direct, non-inverted function of y here.
        row_idx = int((pt.y() - best_rect.y()) / best_rect.height() * h)
        col = max(0, min(col, w - 1))
        row_idx = max(0, min(row_idx, h - 1))
        sample = arr[row_idx, col]
        if np.ndim(sample) == 0:
            val = float(sample)
        else:
            # Bug fix: real-world GeoTIFFs are very often a rendered RGB(A)
            # image (e.g. a colorized bathymetry map — confirmed against
            # examples/_real_in/Batimetria.tif, shape (H, W, 4) uint8), not a
            # raw single-band scalar grid. The previous "RGB(A) -> no Z"
            # bail-out meant the readout silently never showed anything for
            # any such file — exactly the reported "still fails to append
            # the Z-value" behaviour. There's no original scalar to recover
            # from rendered colors, but the user's own framing — "Z-Value
            # (elevation/intensity)" — accepts an intensity proxy: standard
            # luminance (ignoring any alpha channel), the usual stand-in
            # cartographic tools show for RGB rasters.
            rgb = np.asarray(sample[:3], dtype=float)
            val = float(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        return val if math.isfinite(val) else None

    def _on_click(self, ev) -> None:
        # The scene emits sigMouseClicked unconditionally, even after an item
        # already consumed the click. A clickable reference-track curve (see
        # add_track_layer) ACCEPTS its own click to emit layer_source_clicked;
        # without this guard the same left-click would ALSO fall through here
        # and recentre the currently-active profile on its nearest vertex —
        # cross-talk that would jump the WRONG profile while switching files.
        # Right-clicks are handled here regardless (the context menu must open
        # even when the cursor is over a layer).
        if ev.button() != Qt.MouseButton.RightButton and ev.isAccepted():
            return
        if ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._on_map_right_click(ev)
            return
        if self._draw_mode is not None:
            ev.accept()
            self._on_draw_click(ev)
            return
        if self._x is None or not self._x.size:
            return
        vb = self.plot.getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        # Nearest track vertex (aspect is locked 1:1, so plain Euclidean works).
        idx = int(np.argmin((self._x - p.x()) ** 2 + (self._y - p.y()) ** 2))
        self.trace_clicked.emit(idx)

    # ── Interactive drawing (right-click → Draw Point / Draw Polyline) ────────

    def _on_map_right_click(self, ev) -> None:
        """Right-click context menu on bare map canvas (the legend, if
        floating there, intercepts its OWN right-clicks first — see
        _DraggableLegend.mouseClickEvent — so this only ever fires for clicks
        that land on the map itself)."""
        menu = QMenu(self)
        act_point = menu.addAction(self.tr("Draw Point"))
        act_line = menu.addAction(self.tr("Draw Polyline"))
        act_poly = menu.addAction(self.tr("Draw Polygon"))
        act_measure = menu.addAction(self.tr("Measure Tool"))
        act_point.triggered.connect(lambda: self._start_draw("point"))
        act_line.triggered.connect(lambda: self._start_draw("polyline"))
        act_poly.triggered.connect(lambda: self._start_draw("polygon"))
        act_measure.triggered.connect(lambda: self._start_draw("measure"))
        if self._draw_mode is not None:
            menu.addSeparator()
            act_cancel = menu.addAction(self.tr("Cancel Drawing"))
            act_cancel.triggered.connect(self._cancel_draw)
        # Cartographic overlays, grouped at the bottom — independent,
        # persistent toggles (no conflict with the draw-mode actions above).
        menu.addSeparator()
        act_scale = menu.addAction(self.tr("Toggle Graphic Scale"))
        act_scale.setCheckable(True)
        act_scale.setChecked(self._scale_bar is not None)
        act_scale.triggered.connect(self._toggle_scale_bar)
        act_north = menu.addAction(self.tr("Toggle North Arrow"))
        act_north.setCheckable(True)
        act_north.setChecked(self._north_arrow is not None)
        act_north.triggered.connect(self._toggle_north_arrow)
        act_grid = menu.addAction(self.tr("Toggle Grid"))
        act_grid.setCheckable(True)
        act_grid.setChecked(self._grid_visible)
        act_grid.triggered.connect(self._toggle_grid)
        menu.exec(ev.screenPos().toPoint())

    def _toggle_grid(self) -> None:
        """Reference Grid (Part 3): showGrid was previously called once,
        unconditionally, at construction — now a persistent toggle so the
        user can hide it. Re-applying showGrid (rather than touching the
        ViewBox directly) is what already renders correctly in exports, since
        the export path re-renders this same PlotItem."""
        self._grid_visible = not self._grid_visible
        self.plot.showGrid(x=self._grid_visible, y=self._grid_visible, alpha=0.15)

    def _start_draw(self, mode: str) -> None:
        """Enter draw mode. Pan/zoom are deliberately disabled for the
        duration (restored the instant drawing finishes or is cancelled) —
        this is the 'safely intercept mouse events' part: a click that has
        any drag component while panning is still enabled could otherwise
        both place a vertex AND shift the view, which would be confusing and
        could place the vertex at the wrong spot. Normal pan/zoom is
        completely unaffected outside an active draw."""
        self._cancel_draw()
        self._draw_mode = mode
        self._draw_points = []
        self.plot.getViewBox().setMouseEnabled(False, False)
        self.plot.setCursor(Qt.CursorShape.CrossCursor)
        if mode == "measure":
            self.plot.scene().sigMouseMoved.connect(self._on_measure_mouse_moved)

    def _cancel_draw(self) -> None:
        if self._draw_mode is None:
            return
        if self._draw_mode == "measure":
            try:
                self.plot.scene().sigMouseMoved.disconnect(self._on_measure_mouse_moved)
            except TypeError:
                pass
            if self._measure_text is not None:
                self.plot.getViewBox().removeItem(self._measure_text)
                self._measure_text = None
        self._draw_mode = None
        self._draw_points = []
        self.plot.getViewBox().setMouseEnabled(True, True)
        self.plot.unsetCursor()
        if self._draw_preview_item is not None:
            self.plot.getViewBox().removeItem(self._draw_preview_item)
            self._draw_preview_item = None

    def _on_draw_click(self, ev) -> None:
        vb = self.plot.getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        pt = (float(p.x()), float(p.y()))
        if self._draw_mode == "point":
            self._finish_draw([pt])
            return
        # Polyline / Polygon / Measure: the double-click's OWN position is
        # still the final vertex (the user is clicking AT that point to
        # finish there), so it must be appended before finishing — not
        # discarded in favour of whatever was already accumulated.
        finishing = ev.double() and bool(self._draw_points)
        self._draw_points.append(pt)
        if finishing:
            self._finish_draw(self._draw_points)
            return
        self._update_draw_preview()

    def _update_draw_preview(self, extra_point: Optional[Tuple[float, float]] = None) -> None:
        """``extra_point`` (the live cursor position, while measuring) is
        appended to the preview WITHOUT being added to ``_draw_points`` —
        only confirmed clicks become real vertices."""
        pts = list(self._draw_points)
        if extra_point is not None:
            pts.append(extra_point)
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if self._draw_preview_item is None:
            self._draw_preview_item = pg.PlotDataItem(
                xs, ys, pen=pg.mkPen(theme.color("highlight"), width=2,
                                     style=Qt.PenStyle.DashLine),
                symbol="o", symbolSize=6, symbolBrush=theme.color("highlight"))
            self._draw_preview_item.setZValue(60)
            self.plot.getViewBox().addItem(self._draw_preview_item, ignoreBounds=True)
        else:
            self._draw_preview_item.setData(xs, ys)

    def _on_measure_mouse_moved(self, scene_pos) -> None:
        """Live running-distance readout next to the cursor while the
        Measure Tool is active — the last segment rubber-bands to follow the
        mouse (unlike Draw Polyline/Polygon, which only update on a
        confirmed click); nothing here is ever persisted as a layer."""
        if self._draw_mode != "measure" or not self._draw_points:
            return
        vb = self.plot.getViewBox()
        p = vb.mapSceneToView(scene_pos)
        cur = (float(p.x()), float(p.y()))
        self._update_draw_preview(extra_point=cur)
        pts_m = self._to_metric_points(self._draw_points + [cur])
        dist = self._path_length(pts_m)
        text = (f"{dist / 1000.0:,.4f} km" if dist >= 1000.0
                else f"{dist:,.2f} m")
        if self._measure_text is None:
            self._measure_text = pg.TextItem(text, anchor=(0.0, 1.0),
                                             color=theme.color("highlight"))
            self._measure_text.setZValue(61)
            self.plot.getViewBox().addItem(self._measure_text, ignoreBounds=True)
        else:
            self._measure_text.setText(text)
        self._measure_text.setPos(cur[0], cur[1])

    @staticmethod
    def _path_length(points: list) -> float:
        total = 0.0
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return total

    def _finish_draw(self, points: list) -> None:
        mode = self._draw_mode
        self._cancel_draw()
        if mode == "point" and len(points) >= 1:
            self._add_drawn_point(points[0])
        elif mode == "polyline" and len(points) >= 2:
            self._add_drawn_polyline(points)
        elif mode == "polygon" and len(points) >= 3:
            self._add_drawn_polygon(points)
        # "measure": no layer created — _cancel_draw() above already cleared
        # the preview/readout, matching "Double-click or Escape cancels/
        # finishes the measurement, clearing the temporary drawing."

    def _add_drawn_point(self, pt: Tuple[float, float]) -> None:
        self._drawn_seq += 1
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        item = pg.ScatterPlotItem(x=[pt[0]], y=[pt[1]], size=10,
                                  brush=pg.mkBrush(color), pen=pg.mkPen("w", width=1))
        name = self.tr("Drawn Point {0}").format(self._drawn_seq)
        li = self._register_layer(name, [item], "vector", anchor=pt)
        li.setData(_ROLE_IS_DRAWN, True)
        li.setData(_ROLE_DRAWN_GEOM, {"geom_type": "point", "coords": [pt]})

    def _add_drawn_polyline(self, points: list) -> None:
        self._drawn_seq += 1
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        item = pg.PlotDataItem(xs, ys, pen=pg.mkPen(color, width=2))
        anchor = points[len(points) // 2]
        name = self.tr("Drawn Line {0}").format(self._drawn_seq)
        li = self._register_layer(name, [item], "vector", anchor=anchor)
        li.setData(_ROLE_IS_DRAWN, True)
        li.setData(_ROLE_DRAWN_GEOM, {"geom_type": "line", "coords": points})

    def _add_drawn_polygon(self, points: list) -> None:
        """Like a polyline, but the shape is explicitly closed (last vertex
        back to the first) and rendered filled. The STORED geometry keeps the
        points UNCLOSED — shapely's Polygon (used by write_vector) closes the
        ring itself, so storing it twice would just duplicate the vertex."""
        self._drawn_seq += 1
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        path = QPainterPath()
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        path.closeSubpath()
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(color)); pen.setCosmetic(True); pen.setWidthF(1.5)
        item.setPen(pen)
        fill = QColor(color); fill.setAlpha(60)
        item.setBrush(QBrush(fill))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        anchor = points[len(points) // 2]
        name = self.tr("Drawn Polygon {0}").format(self._drawn_seq)
        li = self._register_layer(name, [item], "vector", anchor=anchor)
        li.setData(_ROLE_IS_DRAWN, True)
        li.setData(_ROLE_DRAWN_GEOM, {"geom_type": "polygon", "coords": points})

    def eventFilter(self, obj, event) -> bool:
        if obj is self.plot and self._draw_mode is not None \
                and event.type() == QEvent.Type.KeyPress:
            if (self._draw_mode in ("polyline", "polygon", "measure")
                    and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                    and self._draw_points):
                self._finish_draw(self._draw_points)
                return True
            if event.key() == Qt.Key.Key_Escape:
                self._cancel_draw()      # cancels ANY active draw mode, including "point"
                return True
        return super().eventFilter(obj, event)

    # ── "Export as .shp…" (drawn layers) ───────────────────────────────────────

    def _export_drawn_shapefile(self, items: list) -> None:
        items = [li for li in items if bool(li.data(_ROLE_IS_DRAWN))]
        if not items:
            return
        # The map always displays drawn/loaded shapes in WGS84 lon/lat (see
        # add_layer's docstring), so that's the correct default for "the
        # map's current EPSG code" — the user can type any other target.
        target_epsg, ok = QInputDialog.getText(
            self, self.tr("Target EPSG"), self.tr("EPSG code:"), text="EPSG:4326")
        if not ok:
            return
        target_epsg = target_epsg.strip() or "EPSG:4326"
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export as Shapefile"), "drawn.shp",
            self.tr("Shapefile (*.shp)"))
        if not path:
            return
        if not path.lower().endswith(".shp"):
            path += ".shp"
        from ...core.gis_io import write_vector
        try:
            # One file per layer (a shapefile can't mix geometry types, and
            # each drawn row is its own independent shape) — single
            # selection writes exactly the chosen path; multiple selections
            # suffix the filename per layer.
            for i, li in enumerate(items):
                geom = li.data(_ROLE_DRAWN_GEOM)
                if geom is None:
                    continue
                out_path = path if len(items) == 1 else self._suffixed_path(path, i)
                write_vector(out_path, geom["geom_type"], [np.asarray(geom["coords"])],
                            target_epsg=target_epsg)
        except Exception as exc:
            QMessageBox.warning(self, self.tr("Export failed"), str(exc))

    @staticmethod
    def _suffixed_path(path: str, idx: int) -> str:
        p = Path(path)
        return str(p.with_name(f"{p.stem}_{idx + 1}{p.suffix}"))

    # ── Cartographic overlays (Scale Bar / North Arrow) ─────────────────────────
    #
    # Both are persistent, independent of any specific layer, and — unlike the
    # legend's LegendItem — neither needs the ItemIgnoresTransformations
    # work-around from export_image(): confirmed empirically that the scale
    # bar (its bar geometry is recomputed via sigRangeChanged → updateBar(),
    # which maps real-world `size` through the CURRENT view transform rather
    # than opting out of inherited transforms) and pg.TextItem (its own
    # updateTransform() inverse-scale trick, unrelated to the
    # ItemIgnoresTransformations flag LegendItem uses) both already scale
    # correctly when the exporter renders at a different resolution than the
    # live screen — verified by tracing each one's paint-time transform/pixel
    # footprint at two different export widths and seeing the SAME ratio.

    def _scale_calibration_factor(self) -> float:
        """Real-world METRES represented by one data-unit-X at the current
        view's center latitude. Projected/UTM coordinates are already metric
        (1 unit = 1 m). Geographic (WGS84 degree) coordinates need a
        latitude-cosine correction — 1° of longitude shrinks toward the
        poles, while 1° of latitude stays ~constant (111 320 m, mean WGS84
        ellipsoid value) — otherwise the bar would label raw degrees as if
        they were metres (the original miscalibration this fixes)."""
        if not self._coords_are_geographic():
            return 1.0
        (_x0, _x1), (y0, y1) = self.plot.getViewBox().viewRange()
        center_lat = (y0 + y1) / 2.0
        return 111_320.0 * max(0.01, math.cos(math.radians(center_lat)))

    def _toggle_scale_bar(self) -> None:
        if self._scale_bar is not None:
            # Same pyqtgraph quirk as the legend (see _on_legend_toggled):
            # ViewBox.removeItem() calls setParentItem(None) internally, and
            # the item's setParentItem() unconditionally re-anchors using its
            # stored 'offset' attribute whenever it's called — even to
            # detach — which raises once the parent is gone. Clearing the
            # attribute first short-circuits that branch.
            self._scale_bar.offset = None
            self.plot.getViewBox().removeItem(self._scale_bar)
            self._scale_bar = None
            return
        if self._scale_bar_size_m <= 0:
            (x0, x1), _ = self.plot.getViewBox().viewRange()
            span_m = abs(x1 - x0) * self._scale_calibration_factor()
            self._scale_bar_size_m = self._nice_scale_length(span_m)
        self._scale_bar = _ScaleBarItem(
            self._scale_calibration_factor, width=5, color=theme.color("text"),
            style=self._scale_bar_style, zoom_mode=self._scale_bar_zoom_mode,
            size_m=self._scale_bar_size_m, pixel_width=self._scale_bar_pixel_width,
            offset=(-20, -20),   # bottom-right inset
            on_style_changed=self._on_scale_bar_style_changed,
            on_zoom_mode_changed=self._on_scale_bar_zoom_mode_changed,
            on_set_scale=self._prompt_absolute_scale,
            on_reset_scale=self._reset_scale_lock,
            is_locked_fn=lambda: self._locked_scale_ratio is not None)
        self._scale_bar.setParentItem(self.plot.getViewBox())

    def _reset_scale_lock(self) -> None:
        """Disarm the absolute "Set Scale (1:X)" export lock (Bug #3): the
        next export reverts to the normal resolution heuristic. Triggered
        either by the scale-bar menu's "Reset Scale Lock" or automatically
        on any user pan/zoom / autoRange (see _on_view_changed_clear_lock)."""
        if self._locked_scale_ratio is not None:
            print(f"[Scale Lock] cleared (was 1:{self._locked_scale_ratio:.0f}); "
                  f"exports revert to the normal resolution.")
            self._locked_scale_ratio = None

    def _on_view_changed_clear_lock(self, *_args) -> None:
        """Auto-clear the export scale lock on a user pan/zoom (Bug #3).
        Wired to sigRangeChangedManually, which fires ONLY for mouse
        drag/wheel — never for the programmatic setXRange inside
        _apply_absolute_scale — so arming the lock can't self-clear. Once
        the view no longer matches the locked 1:X, keeping the lock would
        size exports wrong, so we drop it."""
        self._reset_scale_lock()

    def _on_scale_bar_style_changed(self, style: str) -> None:
        self._scale_bar_style = style

    def _on_scale_bar_zoom_mode_changed(self, mode: str) -> None:
        self._scale_bar_zoom_mode = mode
        if self._scale_bar is not None:
            if mode == "value":
                self._scale_bar_pixel_width = self._scale_bar.pixel_width
            else:
                self._scale_bar_size_m = self._scale_bar.size_m

    def _prompt_absolute_scale(self) -> None:
        """Right-click → "Set Scale (1:X)…": zoom the view so one physical
        screen inch depicts exactly X real-world inches. See
        _apply_absolute_scale for the DPI math."""
        text, ok = QInputDialog.getText(
            self, self.tr("Set Scale"), self.tr("Scale (e.g. 1:50000):"),
            text="1:50000")
        if not ok or not text.strip():
            return
        try:
            ratio = float(text.strip().replace(" ", "").split(":")[-1])
        except ValueError:
            QMessageBox.warning(self, self.tr("Invalid scale"),
                                self.tr("Enter a scale like 1:50000."))
            return
        if ratio > 0:
            self._apply_absolute_scale(ratio)

    def _apply_absolute_scale(self, ratio: float) -> None:
        """Zoom the ViewBox so the CURRENT screen's physical DPI makes one
        on-screen inch depict ``ratio`` real-world inches (the standard
        cartographic "1:X" definition). meters_per_pixel = (X·0.0254 m) / DPI;
        converted to data-units via the same geo/projected calibration as the
        scale bar, then multiplied by the ViewBox's current pixel width to
        get the new X-range. The map is strictly aspect-locked 1:1 (see
        __init__), so Y is derived automatically — only X is set explicitly.

        Also arms the Export Scale Lock (``self._locked_scale_ratio``): the
        NEXT ``export_image()`` call will derive its output pixel size from
        this same ratio at print DPI instead of the normal dpi*10 heuristic
        — see ``_scale_locked_export_width``. Screen and export are two
        independent DPI domains (the live monitor's physical DPI vs. the
        export's target print DPI), so each gets its own explicit math —
        printed here for verification."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        dpi = screen.physicalDotsPerInch() or 96.0
        meters_per_pixel = (ratio * 0.0254) / dpi
        m_per_unit = max(self._scale_calibration_factor(), 1e-12)
        data_per_pixel = meters_per_pixel / m_per_unit
        vb = self.plot.getViewBox()
        w_px = vb.size().width()
        if w_px <= 0:
            return
        new_span = data_per_pixel * w_px
        (x0, x1), _ = vb.viewRange()
        cx = (x0 + x1) / 2.0
        # setXRange is programmatic, so it does NOT emit sigRangeChangedManually
        # — the auto-clear (Bug #3) only watches that signal, so arming the
        # lock here can't self-clear even though setXRange fires the deferred
        # sigRangeChanged and the aspect lock may nudge the range afterwards.
        vb.setXRange(cx - new_span / 2.0, cx + new_span / 2.0, padding=0)
        self._locked_scale_ratio = ratio
        print(f"[Scale Math · Screen] 1:{ratio:.0f} @ screen_dpi={dpi:.2f} -> "
              f"meters_per_pixel={meters_per_pixel:.6f} m/px, "
              f"meters_per_data_unit={m_per_unit:.4f}, "
              f"view_pixel_width={w_px:.1f} px -> new_x_span={new_span:.4f} data units "
              f"({new_span * m_per_unit:.2f} m ground)")

    @staticmethod
    def _nice_scale_length(span: float) -> float:
        """A round 1/2/5×10ⁿ length close to 15% of the given REAL-WORLD span
        (always metres — callers convert data-units → metres via
        _scale_calibration_factor() first) — the usual rule of thumb for a
        readable scale bar."""
        raw = abs(span) * 0.15
        if raw <= 0:
            return 1.0
        exp = np.floor(np.log10(raw))
        base = raw / (10 ** exp)
        nice = 1.0 if base < 1.5 else 2.0 if base < 3.5 else 5.0 if base < 7.5 else 10.0
        return float(nice * (10 ** exp))

    def _toggle_north_arrow(self) -> None:
        if self._north_arrow is not None:
            # Unlike the scale bar/legend, _NorthArrowItem's own setParentItem()
            # only re-anchors when the new parent is NOT None (see its
            # docstring) — so no offset-clearing workaround is needed before
            # removeItem() detaches it.
            self.plot.getViewBox().removeItem(self._north_arrow)
            self._north_arrow = None
            return
        self._north_arrow = _NorthArrowItem(
            style=self._north_arrow_style, position=self._north_arrow_position,
            size_px=self._north_arrow_size_px, color=theme.color("text"),
            on_style_changed=self._on_north_arrow_style_changed,
            on_position_changed=self._on_north_arrow_position_changed,
            on_size_changed=self._on_north_arrow_size_changed,
            on_set_size=self._prompt_north_arrow_size)
        self._north_arrow.setZValue(55)
        self._north_arrow.setParentItem(self.plot.getViewBox())

    def _on_north_arrow_style_changed(self, style: str) -> None:
        self._north_arrow_style = style

    def _on_north_arrow_position_changed(self, position: str) -> None:
        self._north_arrow_position = position

    def _on_north_arrow_size_changed(self, size_px: float) -> None:
        self._north_arrow_size_px = size_px

    def _prompt_north_arrow_size(self) -> None:
        current = int(self._north_arrow.size_px) if self._north_arrow is not None \
            else int(self._north_arrow_size_px)
        val, ok = QInputDialog.getInt(
            self, self.tr("North Arrow Size"), self.tr("Size (pixels):"),
            current, 16, 200)
        if ok and self._north_arrow is not None:
            self._north_arrow.set_size(float(val))

    def _restyle_north_arrow(self) -> None:
        if self._north_arrow is not None:
            self._north_arrow.set_color(theme.color("text"))

    # ── Internals ───────────────────────────────────────────────────────────

    def _restyle(self, *_) -> None:
        self.plot.setBackground(theme.color("panel"))
        self.dock_legend_view.setBackground(theme.color("panel"))
        # World basemap: a very subtle land tint + faint cosmetic border (or, in
        # graticule-fallback mode, faint grid lines with no fill) so it reads as
        # context without competing with the cyan trackline. Adapts to the active
        # theme via the panel-background luminance.
        if getattr(self, "_basemap", None) is not None:
            dark = QColor(theme.color("panel")).lightnessF() < 0.5
            line = QColor("#39414d") if dark else QColor("#c4ccd6")
            pen = QPen(line)
            pen.setCosmetic(True)          # constant 1 px regardless of zoom
            pen.setWidthF(1.0)
            self._basemap.setPen(pen)
            if self._basemap_is_graticule:
                self._basemap.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            else:
                self._basemap.setBrush(QBrush(QColor("#1e222a") if dark
                                              else QColor("#e6eaef")))
        self.curve_full.setPen(pg.mkPen(theme.color("sub"), width=1))
        self.curve_seg.setPen(pg.mkPen("#2dd4ee", width=3))      # bright cyan segment
        self.marker_start.setBrush(pg.mkBrush(theme.color("ok")))    # green = start
        self.marker_end.setBrush(pg.mkBrush(theme.color("warn")))    # red = end
        for ax in ("left", "bottom", "right", "top"):
            axis = self.plot.getAxis(ax)
            axis.setPen(theme.color("sub"))
            axis.setTextPen(theme.color("text"))
            axis.setStyle(tickFont=QFont(MONO, 8))
        for row in range(self.layer_list.count()):
            text_item = self.layer_list.item(row).data(_ROLE_LABEL_ITEM)
            if text_item is not None:
                self._restyle_label(text_item)
        self._restyle_legend()
        if self._scale_bar is not None:
            self._scale_bar.set_color(theme.color("text"))
        self._restyle_north_arrow()

    def _retranslate(self, *_) -> None:
        self.plot.setLabel("bottom", self.tr("Easting / Longitude (X)"))
        self.plot.setLabel("left", self.tr("Northing / Latitude (Y)"))
        self.lbl_layers.setText(self.tr("Layers"))
        self.chk_show_legend.setText(self.tr("Show Legend"))
        self.chk_show_points.setText(self.tr("Show Points"))
        self.lbl_opacity.setText(self.tr("Opacity"))
        self.btn_add_layer.setText(self.tr("Add Layer"))
        self.btn_remove_layer.setText(self.tr("Remove"))
        self.btn_export_map.setText(self.tr("Export"))
        self._set_track_item_text()
        self._set_basemap_item_text()
