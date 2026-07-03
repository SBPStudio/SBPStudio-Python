"""
geometry_export.py — FIX-point and navline geometry writers.

Naming convention
-----------------
write_fix_points_shp/geojson/csv  — point shapefile/GeoJSON/CSV of FIX marks
write_navline_shp/geojson/csv     — polyline/linestring navline export

The FIX-point writers are ported from topas_core.py (write_shp_pure,
write_geojson, write_csv). The navline writers are ported from
TopasSUITE._write_navline_shp/geojson/csv (~L2427-2597).

All writers use only the Python stdlib — no external dependencies.

parse_timestamp  / compute_fix_positions are ported from topas_core.py
verbatim. Their numeric behaviour is identical to the GUI monolith.
"""
from __future__ import annotations

import csv as _csv
import json as _json
import struct
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np


# ── WGS84 PRJ string ──────────────────────────────────────────────────────────

_WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",'
              'DATUM["D_WGS_1984",'
              'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
              'PRIMEM["Greenwich",0.0],'
              'UNIT["Degree",0.0174532925199433]]')


# ── Timestamp parsing ─────────────────────────────────────────────────────────

def parse_timestamp(ts: str) -> Optional[datetime]:
    """Convert "YYYY-DOYnnn HH:MM:SS" to a datetime. Returns None on failure."""
    try:
        date_part, time_part = ts.split(" ")
        year = int(date_part.split("-")[0])
        doy  = int(date_part.split("DOY")[1])
        h, m, s = (int(x) for x in time_part.split(":"))
        return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=h, minutes=m, seconds=s)
    except Exception:
        return None


# ── FIX-position computation ──────────────────────────────────────────────────

def compute_fix_positions(
    timestamps: List[str],
    dist_km: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    interval_min: int,
) -> List[Tuple[int, float, str, float, float]]:
    """
    Compute FIX marks at regular time intervals.

    Returns list of (fix_num, dist_km, "HH:MM", lon, lat).
    Port of topas_core.compute_fix_positions (identical numeric behaviour).
    """
    epoch = datetime(1970, 1, 1)
    t_sec = np.empty(len(timestamps), dtype=np.float64)
    for k, ts in enumerate(timestamps):
        dt = parse_timestamp(ts)
        t_sec[k] = (dt - epoch).total_seconds() if dt else np.nan

    valid = ~np.isnan(t_sec)
    if not valid.any():
        return []

    t0      = t_sec[valid][0]
    t_end   = t_sec[valid][-1]
    iv_sec  = interval_min * 60.0

    first_rem = t0 % iv_sec
    first_fix = t0 + (iv_sec - first_rem) if first_rem else t0 + iv_sec

    fix_times = np.arange(first_fix, t_end + 1e-3, iv_sec)
    if fix_times.size == 0:
        return []

    t_sorted_idx = np.argsort(t_sec, kind="stable")
    t_sorted     = t_sec[t_sorted_idx]

    fixes: List[Tuple[int, float, str, float, float]] = []
    half_iv = iv_sec / 2.0
    for num, ft in enumerate(fix_times, 1):
        pos      = np.searchsorted(t_sorted, ft)
        best_idx = None
        best_diff = np.inf
        for cand_pos in (pos - 1, pos):
            if 0 <= cand_pos < len(t_sorted_idx):
                orig_idx = t_sorted_idx[cand_pos]
                if np.isnan(t_sec[orig_idx]):
                    continue
                diff = abs(t_sec[orig_idx] - ft)
                if diff < best_diff:
                    best_diff = diff
                    best_idx  = orig_idx
        if best_idx is not None and best_diff < half_iv:
            hhmm = datetime.fromtimestamp(ft, tz=timezone.utc).strftime("%H:%M")
            fixes.append((num, float(dist_km[best_idx]), hhmm,
                          float(lons[best_idx]), float(lats[best_idx])))
    return fixes


def fixes_to_wgs84(fixes: list, source=None) -> list:
    """Return ``fixes`` (see :func:`compute_fix_positions`) with lon/lat
    resolved to WGS84 degrees — what ``write_fix_points_shp``/``geojson``
    declare (.prj / CRS84) and what the CSV's lon/lat column headers claim.

    Same GIS-boundary contract as ``core.picking.picks_to_wgs84`` (the
    interpretation-marks fix): ``compute_fix_positions`` reads the profile's
    NATIVE navigation (``obj.lons``/``obj.lats`` — raw UTM eastings/northings
    in METRES for a projected CoordinateUnits=1 file), so writing them under
    a WGS84 declaration put every FIX mark thousands of "degrees" outside the
    map. Conversion runs only when ``source`` is a projected file WITH a
    resolved CRS (``source.detected_crs``); geographic files, an unresolved
    CRS, or ``source=None`` (legacy callers, and the render/preview figure
    overlays, which never use the lon/lat fields) pass through unchanged.
    The on-figure fields (fix_num, dist_km, HH:MM) are never touched."""
    if source is None or not fixes:
        return fixes
    if getattr(source, "coord_unit", None) != 1:
        return fixes                     # geographic file — already degrees
    src_crs = getattr(source, "detected_crs", None)
    if not src_crs:
        return fixes                     # projected, CRS unresolved — see docstring
    from .spatial import to_geographic
    xs = np.array([f[3] for f in fixes], dtype=float)
    ys = np.array([f[4] for f in fixes], dtype=float)
    lon, lat = to_geographic(xs, ys, src_crs)
    return [(num, dist, hora, float(lo), float(la))
            for (num, dist, hora, _x, _y), lo, la in zip(fixes, lon, lat)]


# ── FIX-point writers (points: list of (fix_num, dist_km, fix_hora, lon, lat)) ─

def write_fix_points_shp(path: str, points: list) -> None:
    """
    Write a POINT shapefile for FIX marks using only the stdlib.
    points: list of (fix_num, dist_km_unused, fix_hora, lon, lat).
    Generates .shp / .shx / .dbf / .prj — WGS84.
    """
    records = points

    def _shp_record(lon: float, lat: float) -> bytes:
        return struct.pack("<i dd", 1, lon, lat)

    shp_records = []
    offsets     = []
    cur_offset  = 50

    for rec in records:
        lon, lat = rec[3], rec[4]
        content  = _shp_record(lon, lat)
        offsets.append(cur_offset)
        shp_records.append(content)
        cur_offset += 4 + len(content) // 2

    file_length = cur_offset
    lons_list   = [r[3] for r in records]
    lats_list   = [r[4] for r in records]
    xmin, xmax  = min(lons_list), max(lons_list)
    ymin, ymax  = min(lats_list), max(lats_list)

    def _file_header(file_len: int) -> bytes:
        return (struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, file_len) +
                struct.pack("<ii dddddddd", 1000, 1,
                            xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))

    shp_path = path if path.endswith(".shp") else path + ".shp"
    shx_path = shp_path[:-4] + ".shx"
    dbf_path = shp_path[:-4] + ".dbf"
    prj_path = shp_path[:-4] + ".prj"

    with open(shp_path, "wb") as shp_f, open(shx_path, "wb") as shx_f:
        shx_file_len = 50 + 4 * len(records)
        shp_f.write(_file_header(file_length))
        shx_f.write(_file_header(shx_file_len))
        for idx, (content, offset) in enumerate(zip(shp_records, offsets)):
            rec_num     = idx + 1
            content_len = len(content) // 2
            rec_hdr     = struct.pack(">ii", rec_num, content_len)
            shp_f.write(rec_hdr + content)
            shx_f.write(struct.pack(">ii", offset, content_len))

    fields = [
        (b"fix_num\x00\x00\x00\x00",  b"N", 6,  0),
        (b"fix_hora\x00\x00\x00",     b"C", 8,  0),
        (b"lon\x00\x00\x00\x00\x00\x00\x00\x00", b"N", 18, 8),
        (b"lat\x00\x00\x00\x00\x00\x00\x00\x00", b"N", 18, 8),
    ]
    record_len = 1 + sum(f[2] for f in fields)
    header_len = 32 + 32 * len(fields) + 1

    with open(dbf_path, "wb") as dbf:
        dbf.write(struct.pack("<B B B B I H H 20s",
                              3, 125, 1, 1, len(records),
                              header_len, record_len, b"\x00" * 20))
        for fname, ftype, flen, fdec in fields:
            dbf.write(fname[:11].ljust(11, b"\x00"))
            dbf.write(ftype)
            dbf.write(b"\x00" * 4)
            dbf.write(struct.pack("B", flen))
            dbf.write(struct.pack("B", fdec))
            dbf.write(b"\x00" * 14)
        dbf.write(b"\r")
        for rec in records:
            fix_num, _dist, fix_hora, lon, lat = rec
            dbf.write(b" ")
            dbf.write(str(fix_num).rjust(6).encode("ascii"))
            dbf.write(fix_hora.ljust(8).encode("ascii"))
            dbf.write(f"{lon:.8f}".rjust(18).encode("ascii"))
            dbf.write(f"{lat:.8f}".rjust(18).encode("ascii"))
        dbf.write(b"\x1a")

    with open(prj_path, "w") as prj:
        prj.write(_WGS84_PRJ)


def write_fix_points_geojson(path: str, points: list) -> None:
    """Write GeoJSON FeatureCollection of FIX points.
    points: list of (fix_num, dist_km_unused, fix_hora, lon, lat)."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"fix_num": num, "fix_hora": hora,
                           "lon": round(lon, 8), "lat": round(lat, 8)},
        }
        for num, _dist, hora, lon, lat in points
    ]
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name",
                  "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
          "features": features}
    out_path = path if path.endswith(".geojson") else path + ".geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(fc, f, ensure_ascii=False, indent=2)


def write_fix_points_csv(path: str, points: list) -> None:
    """Write CSV of FIX marks: fix_num, fix_hora, lon, lat.
    points: list of (fix_num, dist_km_unused, fix_hora, lon, lat)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["fix_num", "fix_hora", "lon", "lat"])
        for num, _dist, hora, lon, lat in points:
            w.writerow([num, hora, round(lon, 8), round(lat, 8)])


# ── Navline writers ────────────────────────────────────────────────────────────

def write_navline_shp(
    path: str,
    lons, lats, dist, wd, ts,
    include_attrs: bool = True,
    crs_str: Optional[str] = None,
) -> None:
    """
    Write a POLYLINE (type 3) shapefile of the navigation track.
    Ported from TopasSUITE._write_navline_shp (~L2427).

    The .dbf stores summary statistics for the single polyline record.
    Per-vertex attributes are best exported via GeoJSON or CSV.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    n    = len(lons)

    shp_path = path if path.lower().endswith(".shp") else path + ".shp"
    shx_path = shp_path[:-4] + ".shx"
    dbf_path = shp_path[:-4] + ".dbf"
    prj_path = shp_path[:-4] + ".prj"

    num_parts       = 1
    num_points      = n
    content_bytes   = 4 + 32 + 4 + 4 + 4 * num_parts + 16 * num_points
    content_words   = content_bytes // 2

    xmin, xmax = float(np.min(lons)), float(np.max(lons))
    ymin, ymax = float(np.min(lats)), float(np.max(lats))

    content  = struct.pack("<i", 3)
    content += struct.pack("<dddd", xmin, ymin, xmax, ymax)
    content += struct.pack("<ii", num_parts, num_points)
    content += struct.pack("<i", 0)
    for x, y in zip(lons, lats):
        content += struct.pack("<dd", float(x), float(y))

    file_len_words = 50 + 4 + content_words

    def _fhdr(flen: int) -> bytes:
        return (struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, flen) +
                struct.pack("<ii dddddddd", 1000, 3,
                            xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))

    with open(shp_path, "wb") as shp, open(shx_path, "wb") as shx:
        shp.write(_fhdr(file_len_words))
        shx.write(_fhdr(50 + 4))
        rec_hdr = struct.pack(">ii", 1, content_words)
        shp.write(rec_hdr + content)
        shx.write(struct.pack(">ii", 50, content_words))

    # .dbf — one record, summary statistics
    sum_fields = [
        (b"n_traces\x00\x00\x00",  b"N",  7, 0),
        (b"total_km\x00\x00\x00",  b"N", 12, 4),
        (b"lon_start\x00\x00",     b"N", 18, 8),
        (b"lat_start\x00\x00",     b"N", 18, 8),
        (b"lon_end\x00\x00\x00\x00", b"N", 18, 8),
        (b"lat_end\x00\x00\x00\x00", b"N", 18, 8),
        (b"ts_start\x00\x00\x00",  b"C", 24, 0),
        (b"ts_end\x00\x00\x00\x00\x00",   b"C", 24, 0),
    ]
    if include_attrs:
        sum_fields += [
            (b"wd_mean\x00\x00\x00\x00",  b"N", 10, 2),
            (b"wd_min\x00\x00\x00\x00\x00",  b"N", 10, 2),
            (b"wd_max\x00\x00\x00\x00\x00",  b"N", 10, 2),
        ]

    record_len = 1 + sum(f[2] for f in sum_fields)
    header_len = 32 + 32 * len(sum_fields) + 1

    with open(dbf_path, "wb") as dbf:
        dbf.write(struct.pack("<B B B B I H H 20s",
                              3, 125, 1, 1, 1,
                              header_len, record_len, b"\x00" * 20))
        for fname, ftype, flen, fdec in sum_fields:
            dbf.write(fname[:11].ljust(11, b"\x00"))
            dbf.write(ftype)
            dbf.write(b"\x00" * 4)
            dbf.write(struct.pack("B", flen))
            dbf.write(struct.pack("B", fdec))
            dbf.write(b"\x00" * 14)
        dbf.write(b"\r")

        dbf.write(b" ")
        dbf.write(str(n).rjust(7).encode("ascii"))
        dbf.write(f"{float(dist[-1]):.4f}".rjust(12).encode("ascii"))
        dbf.write(f"{float(lons[0]):.8f}".rjust(18).encode("ascii"))
        dbf.write(f"{float(lats[0]):.8f}".rjust(18).encode("ascii"))
        dbf.write(f"{float(lons[-1]):.8f}".rjust(18).encode("ascii"))
        dbf.write(f"{float(lats[-1]):.8f}".rjust(18).encode("ascii"))
        dbf.write((ts[0][:24]  if ts else "").ljust(24).encode("ascii"))
        dbf.write((ts[-1][:24] if ts else "").ljust(24).encode("ascii"))
        if include_attrs:
            dbf.write(f"{float(np.nanmean(wd)):.2f}".rjust(10).encode("ascii"))
            dbf.write(f"{float(np.nanmin(wd)):.2f}".rjust(10).encode("ascii"))
            dbf.write(f"{float(np.nanmax(wd)):.2f}".rjust(10).encode("ascii"))
        dbf.write(b"\x1a")

    with open(prj_path, "w") as prj:
        if crs_str and crs_str not in ("EPSG:4326", "4326"):
            try:
                from pyproj import CRS as _CRS
                prj.write(_CRS.from_user_input(crs_str).to_wkt())
            except Exception:
                prj.write(_WGS84_PRJ)
        else:
            prj.write(_WGS84_PRJ)


def write_navline_geojson(
    path: str,
    lons, lats, dist, wd, ts,
    include_attrs: bool = True,
    crs_str: Optional[str] = None,
) -> None:
    """
    Write a GeoJSON LineString of the navigation track.
    Ported from TopasSUITE._write_navline_geojson (~L2543).
    """
    lons  = np.asarray(lons, dtype=float)
    lats  = np.asarray(lats, dtype=float)
    coords = [[float(x), float(y)] for x, y in zip(lons, lats)]

    props: dict = {
        "n_traces":  len(lons),
        "total_km":  round(float(dist[-1]), 4) if len(dist) else 0.0,
        "lon_start": round(float(lons[0]),  8),
        "lat_start": round(float(lats[0]),  8),
        "lon_end":   round(float(lons[-1]), 8),
        "lat_end":   round(float(lats[-1]), 8),
        "ts_start":  ts[0]  if ts else "",
        "ts_end":    ts[-1] if ts else "",
    }
    if include_attrs:
        props["wd_mean"]     = round(float(np.nanmean(wd)), 2)
        props["wd_min"]      = round(float(np.nanmin(wd)), 2)
        props["wd_max"]      = round(float(np.nanmax(wd)), 2)
        props["dist_km"]     = [round(float(v), 4) for v in dist]
        props["water_depth"] = [round(float(v), 2) for v in wd]
        props["timestamps"]  = list(ts)

    crs_name = ("urn:ogc:def:crs:OGC:1.3:CRS84" if not crs_str
                else f"urn:ogc:def:crs:EPSG::{crs_str.upper().replace('EPSG:','')}")
    fc = {
        "type": "FeatureCollection",
        "crs":  {"type": "name", "properties": {"name": crs_name}},
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        }],
    }
    out_path = path if path.lower().endswith(".geojson") else path + ".geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(fc, f, ensure_ascii=False, indent=2)


# ── Interpretation pick-point writers (points: list of (id, x, y, description)) ─
#
# Reuses the EXACT same methodology as write_fix_points_shp/geojson/csv above
# (pure stdlib struct packing for .shp, no new GIS dependency) — only the DBF
# schema/attributes differ (id: numeric, description: free text, instead of
# fix_num/fix_hora).

def write_picks_shp(path: str, points: list) -> None:
    """
    Write a POINT shapefile for interpretation picks using only the stdlib.
    points: list of (id, x, y, description).
    Generates .shp / .shx / .dbf / .prj.
    """
    records = points

    def _shp_record(x: float, y: float) -> bytes:
        return struct.pack("<i dd", 1, x, y)

    shp_records = []
    offsets     = []
    cur_offset  = 50

    for rec in records:
        x, y    = rec[1], rec[2]
        content = _shp_record(x, y)
        offsets.append(cur_offset)
        shp_records.append(content)
        cur_offset += 4 + len(content) // 2

    file_length = cur_offset
    xs_list = [r[1] for r in records] or [0.0]
    ys_list = [r[2] for r in records] or [0.0]
    xmin, xmax = min(xs_list), max(xs_list)
    ymin, ymax = min(ys_list), max(ys_list)

    def _file_header(file_len: int) -> bytes:
        return (struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, file_len) +
                struct.pack("<ii dddddddd", 1000, 1,
                            xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0))

    shp_path = path if path.endswith(".shp") else path + ".shp"
    shx_path = shp_path[:-4] + ".shx"
    dbf_path = shp_path[:-4] + ".dbf"
    prj_path = shp_path[:-4] + ".prj"

    with open(shp_path, "wb") as shp_f, open(shx_path, "wb") as shx_f:
        shx_file_len = 50 + 4 * len(records)
        shp_f.write(_file_header(file_length))
        shx_f.write(_file_header(shx_file_len))
        for idx, (content, offset) in enumerate(zip(shp_records, offsets)):
            rec_num     = idx + 1
            content_len = len(content) // 2
            rec_hdr     = struct.pack(">ii", rec_num, content_len)
            shp_f.write(rec_hdr + content)
            shx_f.write(struct.pack(">ii", offset, content_len))

    # DBF text fields are fixed-width and ASCII-only (the .dbf spec predates
    # any encoding declaration) — non-ASCII description characters are
    # replaced rather than raising, so an accented/emoji description never
    # crashes the export; GeoJSON/CSV (UTF-8) are the full-fidelity formats.
    desc_len = 80
    fields = [
        (b"id\x00\x00\x00\x00\x00\x00\x00\x00\x00", b"N", 10, 0),
        (b"descr\x00\x00\x00\x00\x00\x00", b"C", desc_len, 0),
    ]
    record_len = 1 + sum(f[2] for f in fields)
    header_len = 32 + 32 * len(fields) + 1

    with open(dbf_path, "wb") as dbf:
        dbf.write(struct.pack("<B B B B I H H 20s",
                              3, 125, 1, 1, len(records),
                              header_len, record_len, b"\x00" * 20))
        for fname, ftype, flen, fdec in fields:
            dbf.write(fname[:11].ljust(11, b"\x00"))
            dbf.write(ftype)
            dbf.write(b"\x00" * 4)
            dbf.write(struct.pack("B", flen))
            dbf.write(struct.pack("B", fdec))
            dbf.write(b"\x00" * 14)
        dbf.write(b"\r")
        for rec in records:
            pick_id, _x, _y, descr = rec
            dbf.write(b" ")
            dbf.write(str(pick_id).rjust(10).encode("ascii"))
            dbf.write(str(descr)[:desc_len].ljust(desc_len)
                      .encode("ascii", errors="replace"))
        dbf.write(b"\x1a")

    with open(prj_path, "w") as prj:
        prj.write(_WGS84_PRJ)


def write_picks_geojson(path: str, points: list) -> None:
    """Write GeoJSON FeatureCollection of interpretation picks.
    points: list of (id, x, y, description)."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x, y]},
            "properties": {"id": pid, "description": descr},
        }
        for pid, x, y, descr in points
    ]
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name",
                  "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
          "features": features}
    out_path = path if path.endswith(".geojson") else path + ".geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(fc, f, ensure_ascii=False, indent=2)


def write_picks_csv(path: str, points: list) -> None:
    """Write CSV of interpretation picks: id, x, y, description.
    points: list of (id, x, y, description)."""
    out_path = path if path.lower().endswith(".csv") else path + ".csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["id", "x", "y", "description"])
        for pid, x, y, descr in points:
            w.writerow([pid, round(float(x), 8), round(float(y), 8), descr])


def write_navline_csv(path: str, lons, lats, dist, wd, ts) -> None:
    """
    Write a CSV of the navigation track, one row per trace.
    Columns: trace_idx, lon, lat, dist_km, water_depth_m, timestamp.
    Ported from TopasSUITE._write_navline_csv (~L2586).
    """
    out_path = path if path.lower().endswith(".csv") else path + ".csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["trace_idx", "lon", "lat", "dist_km", "water_depth_m", "timestamp"])
        for i, (x, y, d, wdv, t) in enumerate(zip(lons, lats, dist, wd, ts)):
            w.writerow([i + 1,
                        round(float(x), 8), round(float(y), 8),
                        round(float(d), 4), round(float(wdv), 2),
                        t])
