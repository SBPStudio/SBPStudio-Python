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
