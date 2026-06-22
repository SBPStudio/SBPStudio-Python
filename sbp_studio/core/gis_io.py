"""
gis_io.py — Local GIS layer readers/writers (vector + raster), WGS84.

Reads user-supplied overlay files for the navigation map and returns plain,
GUI-free data containers already in WGS84 lon/lat so the GUI can render them
directly against the geographic basemap; also writes user-drawn map shapes
back out to a shapefile:

  * read_vector(.shp …)  → :class:`VectorLayer`  (geometries as lon/lat paths)
  * read_geotiff(.tif …) → :class:`RasterLayer`  (image + WGS84 bounding box)
  * read_gis_layer(path) → dispatch by file extension
  * write_vector(path, geom_type, paths) → save lon/lat paths to a shapefile

ALL parsing and CRS reprojection happen here in the CORE (pyproj via
``spatial.reproject_points``). Heavy GIS deps (geopandas, tifffile) are imported
lazily inside the readers so importing this module stays cheap and the core
stays installable without them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .spatial import WGS84, reproject_points

# Largest raster dimension (px) kept on read. A real-world bathymetry/DEM
# GeoTIFF can be tens of thousands of pixels per side — loading it full-res
# bloats RAM and the live scene for a backdrop that's only ever a few hundred
# screen pixels wide. Above this, the array is strided-decimated on read; the
# WGS84 bounding box is derived from the ORIGINAL geotransform, so the spatial
# extent is unchanged (the GUI just stretches a smaller array over the same
# rect). 4096 keeps it crisp at any practical map zoom.
_MAX_RASTER_DIM = 4096


def _decimate_raster(image: np.ndarray, max_dim: int = _MAX_RASTER_DIM) -> np.ndarray:
    """Strided-downsample a raster so its largest side is ≤ ``max_dim``,
    leaving any channel axis (H, W, C) untouched. No-op when already small.
    Strided (not averaged) so it stays dependency-free and cheap — a map
    backdrop doesn't need anti-aliased resampling."""
    if image.ndim < 2:
        return image
    h, w = int(image.shape[0]), int(image.shape[1])
    step = int(np.ceil(max(h, w) / float(max_dim)))
    if step <= 1:
        return image
    return np.ascontiguousarray(image[::step, ::step])


@dataclass
class VectorLayer:
    """Vector overlay: a list of (N, 2) lon/lat paths (WGS84)."""
    name: str
    geom_type: str                       # 'line' | 'polygon' | 'point'
    paths: List[np.ndarray] = field(default_factory=list)
    src_crs: Optional[str] = None
    # Whether the RETURNED coordinates are geographic (lon/lat). True when a
    # source CRS was present (the paths were kept/reprojected to WGS84); None
    # when no CRS/.prj was available, so the GUI falls back to its magnitude
    # heuristic instead of guessing (Bug #10).
    is_geographic: Optional[bool] = None


@dataclass
class RasterLayer:
    """Raster overlay: an image array + its WGS84 bounding box."""
    name: str
    image: np.ndarray                    # (H, W) or (H, W, C)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # lon0, lon1, lat0, lat1
    src_crs: Optional[str] = None
    # As VectorLayer.is_geographic — True when the GeoTIFF carried a CRS (its
    # extent was kept/reprojected to a WGS84 bbox), None when georeferencing
    # gave no CRS (Bug #10).
    is_geographic: Optional[bool] = None


# ── Vector (shapefile / any OGR format geopandas can open) ─────────────────────

def _explode(geom, out: List[np.ndarray]) -> None:
    """Recursively flatten a shapely geometry into (N, 2) coordinate arrays."""
    gt = geom.geom_type
    if gt in ("LineString", "LinearRing"):
        out.append(np.asarray(geom.coords, dtype=float)[:, :2])
    elif gt == "Polygon":
        out.append(np.asarray(geom.exterior.coords, dtype=float)[:, :2])
        for ring in geom.interiors:
            out.append(np.asarray(ring.coords, dtype=float)[:, :2])
    elif gt == "Point":
        out.append(np.asarray([list(geom.coords)[0]], dtype=float)[:, :2])
    elif gt in ("MultiLineString", "MultiPolygon", "MultiPoint", "GeometryCollection"):
        for g in geom.geoms:
            _explode(g, out)


def _family(geom_type: str) -> str:
    if "Polygon" in geom_type:
        return "polygon"
    if "Point" in geom_type:
        return "point"
    return "line"


def read_vector(path: str) -> VectorLayer:
    """Read a vector file (e.g. shapefile) and return its geometries as WGS84
    lon/lat paths. The layer's source CRS is read from the file and reprojected
    to WGS84 with the core transform; a CRS-less file is assumed to already be
    in lon/lat."""
    try:
        import geopandas as gpd
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("Reading vector layers needs geopandas "
                          "(pip install geopandas).") from exc

    gdf = gpd.read_file(path)
    src_epsg = None
    if gdf.crs is not None:
        src_epsg = gdf.crs.to_epsg()
        if src_epsg != 4326:
            gdf = gdf.to_crs(epsg=4326)              # CRS transform via geopandas/pyproj

    paths: List[np.ndarray] = []
    fam = "line"
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if not paths:
            fam = _family(geom.geom_type)
        _explode(geom, paths)

    src = f"EPSG:{src_epsg}" if src_epsg else None
    # A CRS WAS present (so the paths are now WGS84 lon/lat) ⇒ geographic; no
    # CRS/.prj ⇒ unknown (None) so the GUI uses its heuristic — Bug #10.
    is_geographic = True if gdf.crs is not None else None
    return VectorLayer(name=Path(path).stem, geom_type=fam, paths=paths,
                       src_crs=src, is_geographic=is_geographic)


def write_vector(path: str, geom_type: str, paths: List[np.ndarray],
                 crs: str = "EPSG:4326", target_epsg: Optional[str] = None) -> None:
    """Write a list of (N, 2) lon/lat coordinate paths to a shapefile (or any
    other OGR format geopandas can write, selected by ``path``'s extension).
    Each path becomes one feature; ``geom_type`` is 'point' | 'line' |
    'polygon' (uniform per call — a shapefile can't mix geometry types in one
    file, so callers with mixed types must call this once per type).

    ``crs`` is the SOURCE CRS of ``paths`` (WGS84 lon/lat by default — how
    every drawn/loaded map shape in the GUI is stored). If ``target_epsg`` is
    given (e.g. ``"EPSG:32630"`` or bare ``"32630"``), the geometries are
    reprojected to it via ``GeoDataFrame.to_crs()`` (geopandas/pyproj) before
    saving — invalid codes raise naturally from ``to_crs()`` and propagate to
    the caller."""
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point, Polygon
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("Writing vector layers needs geopandas "
                          "(pip install geopandas).") from exc

    builders = {"point": lambda p: Point(p[0]), "line": LineString, "polygon": Polygon}
    builder = builders.get(geom_type, LineString)
    min_pts = 1 if geom_type == "point" else 2
    geoms = [builder(np.asarray(p, dtype=float)) for p in paths if len(p) >= min_pts]
    if not geoms:
        raise ValueError("No valid geometry to write (need at least "
                         f"{min_pts} point(s) per path for '{geom_type}').")
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=crs)
    if target_epsg:
        target_epsg = str(target_epsg).strip()
        if target_epsg.isdigit():
            target_epsg = f"EPSG:{target_epsg}"
        gdf = gdf.to_crs(target_epsg)
    gdf.to_file(path)


# ── Raster (GeoTIFF) ───────────────────────────────────────────────────────────

def _geotiff_crs(md: dict) -> Optional[str]:
    epsg = md.get("ProjectedCSTypeGeoKey") or md.get("GeographicTypeGeoKey")
    try:
        epsg = int(epsg)
    except (TypeError, ValueError):
        return None
    return f"EPSG:{epsg}" if 1024 <= epsg <= 32767 else None


def read_geotiff(path: str) -> RasterLayer:
    """Read a GeoTIFF and return its pixel array plus a WGS84 bounding box.

    The source bounding box (from the ModelPixelScale + ModelTiepoint tags, or a
    ModelTransformation matrix) is reprojected corner-by-corner to WGS84 and the
    extent taken — an axis-aligned approximation suitable for a map backdrop.
    Raises ValueError for a TIFF without georeferencing tags."""
    try:
        import tifffile
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("Reading GeoTIFFs needs tifffile "
                          "(pip install tifffile).") from exc

    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        image = page.asarray()
        md = tf.geotiff_metadata or {}

    h, w = int(image.shape[0]), int(image.shape[1])
    src_crs = _geotiff_crs(md)

    # Source-CRS corner coordinates of the image extent.
    if "ModelPixelScale" in md and "ModelTiepoint" in md:
        sx, sy = float(md["ModelPixelScale"][0]), float(md["ModelPixelScale"][1])
        i, j, _k, X, Y, _Z = (float(v) for v in md["ModelTiepoint"][:6])
        x0 = X - i * sx
        x1 = X + (w - i) * sx
        y_top = Y + j * sy
        y_bot = Y - (h - j) * sy
    elif "ModelTransformation" in md:
        m = np.asarray(md["ModelTransformation"], dtype=float).reshape(4, 4)
        def _xy(col, row):
            return (m[0, 0] * col + m[0, 1] * row + m[0, 3],
                    m[1, 0] * col + m[1, 1] * row + m[1, 3])
        c0 = _xy(0, 0); c1 = _xy(w, 0); c2 = _xy(0, h); c3 = _xy(w, h)
        xs = [c0[0], c1[0], c2[0], c3[0]]; ys = [c0[1], c1[1], c2[1], c3[1]]
        x0, x1, y_bot, y_top = min(xs), max(xs), min(ys), max(ys)
    else:
        raise ValueError(f"{Path(path).name}: no GeoTIFF georeferencing tags found.")

    # Corners → WGS84 (passthrough if already geographic / unknown).
    cx = np.array([x0, x1, x1, x0], dtype=float)
    cy = np.array([y_top, y_top, y_bot, y_bot], dtype=float)
    if src_crs and src_crs != WGS84:
        lon, lat = reproject_points(cx, cy, src_crs, WGS84)
    else:
        lon, lat = cx, cy
    bbox = (float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max()))
    # Decimate AFTER the bbox is derived (which used the original h, w +
    # geotransform), so the WGS84 extent is preserved exactly while RAM /
    # scene cost is bounded — see _decimate_raster (Bug #7).
    image = _decimate_raster(image)
    # CRS present ⇒ extent kept/reprojected to a WGS84 bbox (geographic); no
    # CRS ⇒ unknown (None), GUI heuristic decides — Bug #10.
    is_geographic = True if src_crs else None
    return RasterLayer(name=Path(path).stem, image=image, bbox=bbox,
                       src_crs=src_crs, is_geographic=is_geographic)


# ── Dispatch ───────────────────────────────────────────────────────────────────

def read_gis_layer(path: str):
    """Read a local GIS file by extension → VectorLayer or RasterLayer."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        return read_geotiff(path)
    if ext in (".shp", ".geojson", ".json", ".gpkg"):
        return read_vector(path)
    raise ValueError(f"Unsupported GIS file type: {ext}")
