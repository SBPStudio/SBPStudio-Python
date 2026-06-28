"""
test_picking.py — Interpretation & Picking Module (Phase 3), core layer.

Covers core.picking (PickPoint, resolve_pick_coords, .tps save/load,
export_picks dispatch) and core.geometry_export's new pick writers
(write_picks_shp/geojson/csv). All pure Python/stdlib — no Qt.
"""
from __future__ import annotations

import json
import os

import pytest

from sbp_studio.core import (
    PickPoint, export_picks, load_picks_tps, resolve_pick_coords, save_picks_tps,
    write_picks_csv, write_picks_geojson, write_picks_shp,
)


class _FakeSource:
    """Minimal stand-in for a SegyProfile/ProfileChain's coordinate arrays."""
    def __init__(self, lons, lats):
        self.lons = lons
        self.lats = lats


class TestResolvePickCoords:
    def test_valid_coordinate_returned_directly(self):
        obj = _FakeSource([0.0, -8.5, 0.0], [0.0, 43.1, 0.0])
        assert resolve_pick_coords(obj, 1) == (-8.5, 43.1)

    def test_exact_zero_zero_falls_back_to_trace_number(self):
        """A real recorded position is never exactly (0, 0) — that pair is
        the reliable signature of 'no navigation', not a real fix on the
        equator/prime-meridian."""
        obj = _FakeSource([0.0, -8.5], [0.0, 43.1])
        assert resolve_pick_coords(obj, 0) == (0.0, 0.0)

    def test_nan_falls_back_to_trace_number(self):
        obj = _FakeSource([float("nan")], [float("nan")])
        assert resolve_pick_coords(obj, 0) == (0.0, 0.0)

    def test_out_of_range_index_falls_back(self):
        obj = _FakeSource([-8.5], [43.1])
        assert resolve_pick_coords(obj, 99) == (99.0, 0.0)

    def test_missing_lons_lats_attribute_falls_back(self):
        class Empty:
            pass
        assert resolve_pick_coords(Empty(), 7) == (7.0, 0.0)

    def test_one_axis_nonzero_is_kept_as_valid(self):
        """Only the EXACT (0,0) pair is treated as 'missing' — a real fix
        that happens to have lat or lon (but not both) at exactly 0 must
        not be discarded."""
        obj = _FakeSource([0.0], [43.1])
        assert resolve_pick_coords(obj, 0) == (0.0, 43.1)


class TestPicksTpsRoundTrip:
    def _sample(self):
        return [
            PickPoint(id=1, trace_index=10, time_ms=123.4, x_coord=-8.5,
                      y_coord=43.1, description="Fault A"),
            PickPoint(id=2, trace_index=250, time_ms=987.6, x_coord=-8.4,
                      y_coord=43.2, description=""),
        ]

    def test_round_trips_every_field_exactly(self, tmp_path):
        picks = self._sample()
        path = str(tmp_path / "session")
        save_picks_tps(path, picks)
        assert os.path.exists(path + ".tps")
        loaded = load_picks_tps(path + ".tps")
        assert loaded == picks

    def test_save_accepts_explicit_tps_extension(self, tmp_path):
        path = str(tmp_path / "session.tps")
        save_picks_tps(path, self._sample())
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tps")

    def test_empty_list_round_trips_to_empty_list(self, tmp_path):
        path = str(tmp_path / "empty")
        save_picks_tps(path, [])
        assert load_picks_tps(path + ".tps") == []

    def test_malformed_entries_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "bad.tps"
        path.write_text(json.dumps({
            "version": 1,
            "picks": [
                {"id": 1, "trace_index": "not-an-int"},          # bad: skip
                {"id": 2, "trace_index": 5, "time_ms": 1.0,
                 "x_coord": 0.0, "y_coord": 0.0, "description": "ok"},
                "not even a dict",                                 # bad: skip
            ],
        }), encoding="utf-8")
        loaded = load_picks_tps(str(path))
        assert len(loaded) == 1
        assert loaded[0].id == 2 and loaded[0].description == "ok"

    def test_non_list_picks_field_returns_empty(self, tmp_path):
        path = tmp_path / "wrong_shape.tps"
        path.write_text(json.dumps({"version": 1, "picks": "oops"}), encoding="utf-8")
        assert load_picks_tps(str(path)) == []

    def test_missing_description_defaults_to_empty_string(self, tmp_path):
        path = tmp_path / "no_desc.tps"
        path.write_text(json.dumps({"version": 1, "picks": [
            {"id": 1, "trace_index": 0, "time_ms": 0.0, "x_coord": 0.0, "y_coord": 0.0},
        ]}), encoding="utf-8")
        loaded = load_picks_tps(str(path))
        assert loaded[0].description == ""


class TestExportPicksDispatch:
    def _sample(self):
        return [
            PickPoint(id=1, trace_index=10, time_ms=123.4, x_coord=-8.5,
                      y_coord=43.1, description="Fault A"),
            PickPoint(id=2, trace_index=20, time_ms=200.0, x_coord=-8.4,
                      y_coord=43.2, description="Bright spot"),
        ]

    def test_csv_extension_dispatches_to_csv_writer(self, tmp_path):
        out = export_picks(str(tmp_path / "p.csv"), self._sample())
        assert out.endswith(".csv") and os.path.exists(out)
        text = open(out, encoding="utf-8").read()
        assert "Fault A" in text and "id,x,y,description" in text

    def test_geojson_extension_dispatches_to_geojson_writer(self, tmp_path):
        out = export_picks(str(tmp_path / "p.geojson"), self._sample())
        assert out.endswith(".geojson") and os.path.exists(out)
        fc = json.load(open(out, encoding="utf-8"))
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2
        assert fc["features"][0]["properties"]["description"] == "Fault A"

    def test_no_recognised_extension_defaults_to_shapefile(self, tmp_path):
        out = export_picks(str(tmp_path / "p"), self._sample())
        assert out.endswith(".shp")
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            assert os.path.exists(out[:-4] + ext)

    def test_shp_extension_dispatches_to_shapefile_writer(self, tmp_path):
        out = export_picks(str(tmp_path / "p.shp"), self._sample())
        assert out.endswith(".shp")
        assert os.path.exists(out)


class TestPickWritersDirect:
    """Direct coverage of the geometry_export writers export_picks dispatches to."""

    def _points(self):
        return [(1, -8.5, 43.1, "Fault A"), (2, -8.4, 43.2, "Bright spot")]

    def test_shp_files_created_with_correct_attributes(self, tmp_path):
        base = str(tmp_path / "picks")
        write_picks_shp(base, self._points())
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            assert os.path.exists(base + ext)
        # .prj is WGS84 by construction (no CRS reprojection for picks).
        assert "WGS_1984" in open(base + ".prj", encoding="utf-8").read()

    def test_shp_handles_non_ascii_description_without_crashing(self, tmp_path):
        """DBF text fields are fixed-width ASCII-only — a non-ASCII
        description must be replaced, not raise, since CSV/GeoJSON (UTF-8)
        already carry the full-fidelity text."""
        base = str(tmp_path / "picks_unicode")
        write_picks_shp(base, [(1, 0.0, 0.0, "Reflectór ñ 漢字")])
        assert os.path.exists(base + ".shp")

    def test_shp_handles_empty_points_list(self, tmp_path):
        base = str(tmp_path / "picks_empty")
        write_picks_shp(base, [])
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            assert os.path.exists(base + ext)

    def test_geojson_coordinates_and_properties(self, tmp_path):
        path = str(tmp_path / "picks.geojson")
        write_picks_geojson(path, self._points())
        fc = json.load(open(path, encoding="utf-8"))
        feat = fc["features"][0]
        assert feat["geometry"]["coordinates"] == [-8.5, 43.1]
        assert feat["properties"] == {"id": 1, "description": "Fault A"}

    def test_csv_header_and_rows(self, tmp_path):
        path = str(tmp_path / "picks.csv")
        write_picks_csv(path, self._points())
        lines = open(path, encoding="utf-8").read().splitlines()
        assert lines[0] == "id,x,y,description"
        assert lines[1] == "1,-8.5,43.1,Fault A"
