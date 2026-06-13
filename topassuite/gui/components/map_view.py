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

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView, QFileDialog, QGraphicsPathItem, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QSlider, QVBoxLayout, QWidget,
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


class MapView(QWidget):
    """Vessel-track navigation map (PyQtGraph, 1:1 isometric)."""

    # Emitted with the trace index nearest to a click on the track.
    trace_clicked = pyqtSignal(int)
    # Emitted with a local file path when the user picks "Add Layer". The owning
    # tab reads it off-thread (core readers) and calls back :meth:`add_layer`.
    layer_file_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setAspectLocked(True)        # strict 1:1 → no track distortion
        for ax in ("left", "bottom"):
            self.plot.getAxis(ax).enableAutoSIPrefix(False)
        root.addWidget(self.plot, 1)
        self._layer_count = 0                  # for cycling layer colours
        root.addWidget(self._build_layer_panel())

        # Track data (per trace).
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        # The SEG-Y track is a MANAGED layer (injected lazily on set_track).
        self._track_item: Optional[QListWidgetItem] = None

        # Offline world basemap (geographic context). One grouped path item under
        # everything; shown only when the track looks like degrees (see
        # _update_basemap). Built before the curves so it sits at the bottom.
        self._basemap_is_graticule: bool = False
        self._basemap: QGraphicsPathItem = self._build_basemap()

        # Items: full track (under) + visible segment (over) + start/end markers.
        self.curve_full = self.plot.plot([], [])
        self.curve_seg = self.plot.plot([], [])
        self.marker_start = pg.ScatterPlotItem(size=9, pen=None)
        self.marker_end = pg.ScatterPlotItem(size=9, pen=None)
        self.plot.addItem(self.marker_start)
        self.plot.addItem(self.marker_end)

        self.plot.scene().sigMouseClicked.connect(self._on_click)

        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)

    # ── Public API ──────────────────────────────────────────────────────────

    def set_track(self, x, y) -> None:
        """Set the full trackline from per-trace X/Y coordinates and pin the
        STATIC SOL/EOL anchors: green at the absolute start ``_x[0]``/``_y[0]``,
        red at the absolute end ``_x[-1]``/``_y[-1]``. These dots never move;
        only the cyan segment tracks the visible window. Resets the segment."""
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self.curve_full.setData(self._x, self._y)
        if self._x.size:
            self.marker_start.setData([self._x[0]], [self._y[0]])   # green = SOL
            self.marker_end.setData([self._x[-1]], [self._y[-1]])   # red   = EOL
        self.curve_seg.setData([], [])
        self._ensure_track_layer()          # the track is a managed list layer
        self._update_basemap()
        self.plot.getViewBox().autoRange()

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
        self.curve_full.setData([], [])
        self.curve_seg.setData([], [])
        self.marker_start.setData([], [])
        self.marker_end.setData([], [])
        self._basemap.setVisible(False)
        # Drop the managed track row (graphics items stay, just emptied). It is
        # re-injected at the top on the next set_track.
        if self._track_item is not None:
            r = self.layer_list.row(self._track_item)
            if r >= 0:
                self.layer_list.takeItem(r)
            self._track_item = None
            self._reorder_layers()

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
        a package-local ``topassuite/assets/`` so it works both in-tree and when
        the package ships its own assets. Returns the repo-root candidate if none
        exist (used only for the warning message)."""
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "assets" / _BASEMAP_ASSET,   # <repo>/assets/
            here.parents[2] / "assets" / _BASEMAP_ASSET,   # topassuite/assets/
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
        """Heuristic: only draw the world basemap when the track looks like
        geographic degrees — bounding box within lon ∈ [-180, 180],
        lat ∈ [-90, 90]. Projected/UTM metres blow past these, so the basemap
        stays hidden rather than rendering a speck next to huge coordinates."""
        if self._x is None or self._x.size == 0:
            return False
        return (float(np.nanmin(self._x)) >= -180.0 and float(np.nanmax(self._x)) <= 180.0
                and float(np.nanmin(self._y)) >= -90.0 and float(np.nanmax(self._y)) <= 90.0)

    def _update_basemap(self) -> None:
        self._basemap.setVisible(self._coords_are_geographic())

    # ── Layer Manager (custom GIS overlays) ───────────────────────────────────

    def _build_layer_panel(self) -> QWidget:
        """Side panel: a drag-reorderable, checkable list of custom layers plus
        Add/Remove buttons. List order = draw order (top row = top of the band)."""
        panel = QWidget()
        panel.setFixedWidth(190)
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)
        self.lbl_layers = QLabel()
        self.lbl_layers.setObjectName("section")
        v.addWidget(self.lbl_layers)
        self.layer_list = QListWidget()
        self.layer_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.layer_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.layer_list.itemChanged.connect(self._on_layer_item_changed)
        self.layer_list.currentItemChanged.connect(self._on_current_layer_changed)
        self.layer_list.model().rowsMoved.connect(lambda *_: self._reorder_layers())
        v.addWidget(self.layer_list, 1)
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
        return panel

    def _on_add_layer_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Add map layer"), "",
            self.tr("GIS layers (*.shp *.tif *.tiff *.geojson)"))
        if path:
            self.layer_file_requested.emit(path)

    def add_layer(self, layer) -> None:
        """Render a core VectorLayer / RasterLayer overlay (called by the tab
        after the core reader runs off-thread). Coordinates are already WGS84."""
        from ...core.gis_io import RasterLayer, VectorLayer
        if isinstance(layer, RasterLayer):
            self._add_raster(layer)
        elif isinstance(layer, VectorLayer):
            self._add_vector(layer)

    def _add_vector(self, layer) -> None:
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        if layer.geom_type == "point":
            pts = np.vstack(layer.paths) if layer.paths else np.empty((0, 2))
            item = pg.ScatterPlotItem(x=pts[:, 0], y=pts[:, 1], size=7,
                                      brush=pg.mkBrush(color), pen=None)
        else:
            path = QPainterPath()
            for arr in layer.paths:
                if len(arr) < 2:
                    continue
                path.addPolygon(QPolygonF([QPointF(float(x), float(y))
                                           for x, y in arr]))
                if layer.geom_type == "polygon":
                    path.closeSubpath()
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(color)); pen.setCosmetic(True); pen.setWidthF(1.5)
            item.setPen(pen)
            if layer.geom_type == "polygon":
                fill = QColor(color); fill.setAlpha(60)
                item.setBrush(QBrush(fill))
            else:
                item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._register_layer(layer.name, [item], "vector")

    def add_track_layer(self, name: str, x, y) -> None:
        """Inject an independent navigation track as a managed GIS layer.

        Used by the sidebar 'Add to map' batch loader: each selected profile /
        chain becomes its own reorderable, checkable, opacity-controlled overlay
        (a polyline with green SOL / red EOL anchors). This is a PINNED reference
        layer — it does NOT touch the active SEG-Y track row, the active profile,
        or any trace data. Coordinates are expected already in WGS84 (lon, lat),
        reprojected off-thread by the caller's CoreWorker.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0:
            return
        color = _LAYER_COLORS[self._layer_count % len(_LAYER_COLORS)]
        line = pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=2))
        line.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        start = pg.ScatterPlotItem([x[0]], [y[0]], size=8,
                                   brush=pg.mkBrush(theme.color("ok")), pen=None)
        end = pg.ScatterPlotItem([x[-1]], [y[-1]], size=8,
                                 brush=pg.mkBrush(theme.color("warn")), pen=None)
        for it in (start, end):
            it.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._register_layer(name, [line, start, end], "vector")
        # Show the world basemap if these coordinates look geographic, then frame
        # the newly added geometry so the user sees it land (autoRange ignores the
        # basemap, which is added with ignoreBounds=True).
        if (float(np.nanmin(x)) >= -180.0 and float(np.nanmax(x)) <= 180.0
                and float(np.nanmin(y)) >= -90.0 and float(np.nanmax(y)) <= 90.0):
            self._basemap.setVisible(True)
        self.plot.getViewBox().autoRange()

    def _add_raster(self, layer) -> None:
        lon0, lon1, lat0, lat1 = layer.bbox
        item = pg.ImageItem()
        # Row 0 of a GeoTIFF is the NORTH edge; flip so it maps to the top of the
        # geographic rect (MapView Y increases northward).
        item.setImage(np.flipud(np.asarray(layer.image)), autoLevels=True)
        item.setRect(QRectF(lon0, lat0, lon1 - lon0, lat1 - lat0))
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._register_layer(layer.name, [item], "raster")

    def _register_layer(self, name: str, gitems: list, kind: str) -> None:
        for gi in gitems:
            self.plot.getViewBox().addItem(gi, ignoreBounds=True)
        self._layer_count += 1
        tag = "▦" if kind == "raster" else "▤"
        li = QListWidgetItem(f"{tag}  {name}")
        li.setFlags(li.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        li.setCheckState(Qt.CheckState.Checked)
        li.setData(Qt.ItemDataRole.UserRole, list(gitems))
        # Newest layer on top, but keep the track row on top by default (the user
        # can still drag a layer above it).
        row0_is_track = (self.layer_list.count() > 0
                         and bool(self.layer_list.item(0).data(_ROLE_IS_TRACK)))
        self.layer_list.insertItem(1 if row0_is_track else 0, li)
        self._reorder_layers()

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
        self._track_item = li
        self.layer_list.insertItem(0, li)
        self._set_track_item_text()
        self._reorder_layers()

    def _set_track_item_text(self) -> None:
        if self._track_item is not None:
            self._track_item.setText("≈  " + self.tr("Navigation / Track"))

    def _reorder_layers(self) -> None:
        """Map list order → zValue in the custom band (top row highest), keeping
        every layer strictly between the basemap (-1000) and z=0. Multi-item
        layers (the track) get tiny within-group offsets so their internal
        stacking (line < segment < markers) is preserved."""
        for row in range(self.layer_list.count()):
            items = self.layer_list.item(row).data(Qt.ItemDataRole.UserRole) or []
            base = max(_LAYER_Z_BOTTOM, _LAYER_Z_TOP - row)
            for j, gi in enumerate(items):
                gi.setZValue(base + j * 0.1)

    def _on_layer_item_changed(self, li: QListWidgetItem) -> None:
        visible = li.checkState() == Qt.CheckState.Checked
        for gi in (li.data(Qt.ItemDataRole.UserRole) or []):
            gi.setVisible(visible)

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
        if li is None or bool(li.data(_ROLE_IS_TRACK)):
            return                                 # the track is intrinsic — never orphan it
        self.layer_list.takeItem(self.layer_list.row(li))
        for gi in (li.data(Qt.ItemDataRole.UserRole) or []):
            self.plot.getViewBox().removeItem(gi)
        self._reorder_layers()

    # ── Map → Profile (click to jump) ────────────────────────────────────────

    def _on_click(self, ev) -> None:
        if self._x is None or not self._x.size:
            return
        vb = self.plot.getViewBox()
        p = vb.mapSceneToView(ev.scenePos())
        # Nearest track vertex (aspect is locked 1:1, so plain Euclidean works).
        idx = int(np.argmin((self._x - p.x()) ** 2 + (self._y - p.y()) ** 2))
        self.trace_clicked.emit(idx)

    # ── Internals ───────────────────────────────────────────────────────────

    def _restyle(self, *_) -> None:
        self.plot.setBackground(theme.color("panel"))
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
        for ax in ("left", "bottom"):
            axis = self.plot.getAxis(ax)
            axis.setPen(theme.color("sub"))
            axis.setTextPen(theme.color("text"))
            axis.setStyle(tickFont=QFont(MONO, 8))

    def _retranslate(self, *_) -> None:
        self.plot.setLabel("bottom", self.tr("Easting / Longitude (X)"))
        self.plot.setLabel("left", self.tr("Northing / Latitude (Y)"))
        self.lbl_layers.setText(self.tr("Layers"))
        self.lbl_opacity.setText(self.tr("Opacity"))
        self.btn_add_layer.setText(self.tr("Add Layer"))
        self.btn_remove_layer.setText(self.tr("Remove"))
        self._set_track_item_text()
