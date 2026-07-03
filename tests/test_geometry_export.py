"""
test_geometry_export.py — Tests for FIX-point and navline writers.
"""
from __future__ import annotations

import csv
import json
import os
import struct

import numpy as np
import pytest

from sbp_studio.core import load_profile, compute_fix_positions
from sbp_studio.core.geometry_export import (
    write_fix_points_shp, write_fix_points_geojson, write_fix_points_csv,
    write_navline_shp, write_navline_geojson, write_navline_csv,
    parse_timestamp,
)


class TestParseTimestamp:
    def test_valid(self):
        from datetime import datetime
        dt = parse_timestamp("2024-DOY100 10:05:30")
        assert dt is not None
        assert dt.year == 2024
        assert dt.hour == 10
        assert dt.minute == 5

    def test_invalid(self):
        assert parse_timestamp("bad string") is None
        assert parse_timestamp("") is None


class TestComputeFixPositions:
    def test_returns_list(self, simple_segy):
        sd    = load_profile(simple_segy)
        fixes = compute_fix_positions(sd.timestamps, sd.dist_km, sd.lons, sd.lats, 1)
        assert isinstance(fixes, list)

    def test_tuple_format(self, simple_segy):
        sd    = load_profile(simple_segy)
        fixes = compute_fix_positions(sd.timestamps, sd.dist_km, sd.lons, sd.lats, 1)
        if fixes:
            num, dist, hora, lon, lat = fixes[0]
            assert isinstance(num, int)
            assert ":" in hora
            assert isinstance(dist, float)


class TestFixWriters:
    def _sample_points(self):
        return [(1, 0.0, "10:00", -8.0, 43.0),
                (2, 1.5, "10:01", -7.9, 43.01)]

    def test_shp_files_created(self, tmp_path):
        pts  = self._sample_points()
        base = str(tmp_path / "fix_test")
        write_fix_points_shp(base, pts)
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            assert os.path.exists(base + ext)

    def test_geojson_valid(self, tmp_path):
        pts  = self._sample_points()
        out  = str(tmp_path / "fix.geojson")
        write_fix_points_geojson(out, pts)
        with open(out) as f:
            fc = json.load(f)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2

    def test_csv_columns(self, tmp_path):
        pts = self._sample_points()
        out = str(tmp_path / "fix.csv")
        write_fix_points_csv(out, pts)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert "fix_num" in rows[0]
        assert "lon" in rows[0]
        assert len(rows) == 2


class TestNavlineWriters:
    def _sample_nav(self):
        n    = 20
        lons = np.linspace(-8.0, -7.8, n)
        lats = np.full(n, 43.0)
        dist = np.linspace(0.0, 5.0, n)
        wd   = np.full(n, 100.0)
        ts   = [f"2024-DOY100 10:{i:02d}:00" for i in range(n)]
        return lons, lats, dist, wd, ts

    def test_navline_shp_created(self, tmp_path):
        lons, lats, dist, wd, ts = self._sample_nav()
        write_navline_shp(str(tmp_path / "nav"), lons, lats, dist, wd, ts)
        assert os.path.exists(str(tmp_path / "nav.shp"))

    def test_navline_geojson_linestring(self, tmp_path):
        lons, lats, dist, wd, ts = self._sample_nav()
        out = str(tmp_path / "nav.geojson")
        write_navline_geojson(out, lons, lats, dist, wd, ts)
        with open(out) as f:
            fc = json.load(f)
        feat = fc["features"][0]
        assert feat["geometry"]["type"] == "LineString"
        assert len(feat["geometry"]["coordinates"]) == 20

    def test_navline_csv_row_count(self, tmp_path):
        lons, lats, dist, wd, ts = self._sample_nav()
        out = str(tmp_path / "nav.csv")
        write_navline_csv(out, lons, lats, dist, wd, ts)
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20
        assert "trace_idx" in rows[0]
        assert "water_depth_m" in rows[0]


class TestFixesToWgs84:
    """GIS-boundary coordinate conversion for FIX marks (same class of bug as
    the interpretation-marks fix, tests/test_picking.py::TestPicksToWgs84):
    compute_fix_positions reads the profile's NATIVE navigation — raw UTM
    metres for a projected (CoordinateUnits=1) file — while the FIX writers
    declare WGS84 (.prj / CRS84 / lon,lat CSV headers). fixes_to_wgs84
    converts exactly when a projected source with a resolved CRS is supplied,
    and passes through otherwise."""

    TRUE_LON = (-60.51234, -60.49876)
    TRUE_LAT = (-62.98765, -62.97654)
    UTM_CRS = "EPSG:32720"

    class _Src:
        def __init__(self, coord_unit, detected_crs):
            self.coord_unit = coord_unit
            self.detected_crs = detected_crs

    def _utm_fixes(self):
        from sbp_studio.core.spatial import reproject_points
        ux, uy = reproject_points(np.asarray(self.TRUE_LON),
                                  np.asarray(self.TRUE_LAT),
                                  "EPSG:4326", self.UTM_CRS)
        return [(i + 1, 0.5 * i, f"10:0{i}", float(ux[i]), float(uy[i]))
                for i in range(len(ux))]

    def test_projected_source_converts_to_wgs84(self):
        from sbp_studio.core import fixes_to_wgs84
        out = fixes_to_wgs84(self._utm_fixes(), self._Src(1, self.UTM_CRS))
        for (num, dist, hora, lon, lat), tl, tt in zip(out, self.TRUE_LON, self.TRUE_LAT):
            assert abs(lon - tl) < 1e-8 and abs(lat - tt) < 1e-8

    def test_figure_fields_never_touched(self):
        from sbp_studio.core import fixes_to_wgs84
        fixes = self._utm_fixes()
        out = fixes_to_wgs84(fixes, self._Src(1, self.UTM_CRS))
        assert [f[:3] for f in out] == [f[:3] for f in fixes]

    def test_geographic_source_passes_through(self):
        from sbp_studio.core import fixes_to_wgs84
        fixes = [(1, 0.0, "10:00", -8.5, 43.1)]
        assert fixes_to_wgs84(fixes, self._Src(3, "EPSG:4326")) == fixes

    def test_projected_unresolved_crs_passes_through_raw(self):
        from sbp_studio.core import fixes_to_wgs84
        fixes = self._utm_fixes()
        assert fixes_to_wgs84(fixes, self._Src(1, None)) == fixes

    def test_none_source_and_empty(self):
        from sbp_studio.core import fixes_to_wgs84
        fixes = self._utm_fixes()
        assert fixes_to_wgs84(fixes, None) is fixes
        assert fixes_to_wgs84([], self._Src(1, self.UTM_CRS)) == []

    def test_end_to_end_shp_readable_at_true_position(self, tmp_path):
        """Converted fixes -> stdlib .shp writer -> geopandas reads real
        WGS84 lon/lat, matching the declared .prj — the file lands on the
        map at the track's position instead of millions of degrees away."""
        from sbp_studio.core import fixes_to_wgs84
        out_base = str(tmp_path / "fix_utm")
        write_fix_points_shp(out_base,
                             fixes_to_wgs84(self._utm_fixes(),
                                            self._Src(1, self.UTM_CRS)))
        gpd = pytest.importorskip("geopandas")
        gdf = gpd.read_file(out_base + ".shp")
        assert len(gdf) == 2
        assert abs(float(gdf.geometry.x.iloc[0]) - self.TRUE_LON[0]) < 1e-8
        assert abs(float(gdf.geometry.y.iloc[0]) - self.TRUE_LAT[0]) < 1e-8
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326
