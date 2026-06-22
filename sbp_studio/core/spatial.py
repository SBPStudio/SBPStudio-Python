"""
spatial.py — Point-array coordinate reprojection (CRS → CRS).

The per-FILE reprojector (``io_segy.reproject_one`` / ``reproject_chain``) writes
new SEG-Y files. This module extracts the SAME exact transform path — a single
vectorised pyproj ``Transformer`` call with ``always_xy=True`` — as a lightweight
reusable function for converting raw coordinate ARRAYS, e.g. to give the
navigation map a consistent geographic frame without touching trace data.

GUI-free by contract (pyproj + numpy only). The GUI imports and CALLS these; no
projection math ever lives in the GUI layer.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer

from .tasks import CRSError

WGS84 = "EPSG:4326"


# ── CRS preset catalog ────────────────────────────────────────────────────────
# Structured, categorised catalog for the Reprojector UI. Each entry is
# ``(category_key, [(label, code), …])``. The category_key is a stable English
# string the GUI maps to a translated, non-selectable separator row; the items
# are the selectable presets. UTM zones are generated for all 60 zones in each
# hemisphere (WGS 84): North = EPSG:326zz, South = EPSG:327zz.
def _utm_zones(hemi: str, base: int):
    return [(f"UTM Zone {z:02d}{hemi} (EPSG:{base + z})", f"EPSG:{base + z}")
            for z in range(1, 61)]


CRS_CATALOG = [
    ("Geographic", [
        ("WGS 84 (EPSG:4326)",            "EPSG:4326"),
        ("ED50 (EPSG:4230)",              "EPSG:4230"),
        ("ETRS89 (EPSG:4258)",            "EPSG:4258"),
        ("SIRGAS 2000 (EPSG:4674)",       "EPSG:4674"),
    ]),
    ("Polar Stereographic", [
        ("Antarctic Polar Stereographic (EPSG:3031)",      "EPSG:3031"),
        ("Arctic Polar Stereographic (EPSG:3995)",         "EPSG:3995"),
        ("NSIDC Sea Ice North / Arctic (EPSG:3413)",       "EPSG:3413"),
        ("NSIDC Sea Ice South / Antarctic (EPSG:3976)",    "EPSG:3976"),
    ]),
    ("UTM North (WGS 84)", _utm_zones("N", 32600)),
    ("UTM South (WGS 84)", _utm_zones("S", 32700)),
]

# Flat label→code map over the whole catalog (convenience lookup).
CRS_PRESETS = {label: code for _cat, items in CRS_CATALOG for label, code in items}


def reproject_points(lons, lats, src_crs: str, dst_crs: str
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Reproject point arrays from ``src_crs`` to ``dst_crs``.

    Returns ``(xs, ys)`` float64 arrays in the destination CRS axis order
    (``always_xy=True`` → x/easting/lon first, y/northing/lat second). One bulk
    ``Transformer.transform`` call (≈50× faster than per-point — the same path
    the file reprojector uses). Identity (``src == dst``) short-circuits to a
    passthrough copy. Raises :class:`CRSError` on an invalid CRS string."""
    xs = np.asarray(lons, dtype=float)
    ys = np.asarray(lats, dtype=float)
    try:
        s = CRS.from_user_input(src_crs)
        d = CRS.from_user_input(dst_crs)
    except Exception as exc:
        raise CRSError(f"Invalid CRS: {exc}") from exc
    if s == d:
        return xs.copy(), ys.copy()
    tf = Transformer.from_crs(s, d, always_xy=True)
    nx, ny = tf.transform(xs, ys)
    return np.asarray(nx, dtype=float), np.asarray(ny, dtype=float)


def crs_produces_geographic(src_crs) -> "Optional[bool]":
    """Whether :func:`to_geographic` will yield GEOGRAPHIC (WGS84 lon/lat)
    output for ``src_crs`` — used by the map to classify coordinate units
    from the ACTUAL CRS instead of a value-magnitude heuristic (Bug #10).

    * ``True``  — ``src_crs`` parses as a CRS (geographic stays lon/lat;
      projected is reprojected to WGS84), so the produced display
      coordinates are geographic.
    * ``None``  — ``src_crs`` is empty or unparseable, so ``to_geographic``
      passes the native coordinates through unchanged and their unit is
      unknown; the caller should fall back to its magnitude heuristic.

    Note: this reports the unit of the PRODUCED (post-``to_geographic``)
    coordinates, which is what the map actually plots — not whether the
    source CRS itself was geographic."""
    if not src_crs:
        return None
    try:
        CRS.from_user_input(src_crs)
        return True
    except Exception:
        return None


def to_geographic(lons, lats, src_crs) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience for map display: reproject native coordinates to WGS84
    geographic (lon, lat degrees).

    Passthrough (copy) when ``src_crs`` is empty/None or already geographic.
    Never raises — display must not crash, so any reprojection failure falls
    back to the input coordinates unchanged."""
    xs = np.asarray(lons, dtype=float)
    ys = np.asarray(lats, dtype=float)
    if not src_crs:
        return xs.copy(), ys.copy()
    try:
        if CRS.from_user_input(src_crs).is_geographic:
            return xs.copy(), ys.copy()
        return reproject_points(xs, ys, src_crs, WGS84)
    except Exception:
        return xs.copy(), ys.copy()
