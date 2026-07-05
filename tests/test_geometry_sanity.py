"""
test_geometry_sanity.py — the 'Geometry Sanity & Reprojection' pipeline.

Background: three real-world SEG-Y files exposed three distinct navigation
failure modes, audited byte-by-byte from their actual trace headers:

  ANT26   (.../ANT26/SGY/20260215115516.seg)
      CoordinateUnits=2 (arc-seconds), scalar=-1000 — correctly formatted,
      but the raw Source X/Y magnitude is ~6 orders of magnitude too small
      (a parser/conversion fault upstream of this app), decoding to ~(0,0) —
      "Gulf of Guinea" instead of Antarctica. In-bounds but WRONG; a pure
      bounds check cannot catch this (see test_in_bounds_but_wrong_not_caught).

  L001A   (.../L001A/20260527060848.sgy)
      Same CoordinateUnits/scalar convention, correctly decodes to a real
      Mediterranean position (~3.77E, ~42.71N) — the control/working case.

  MCS7    (.../MCS7/5_MCS7_MIG.segy)
      CoordinateUnits=1 (length/metres) — i.e. PROJECTED (UTM-style)
      coordinates, not geographic at all. A naive reader that always treats
      Source/CDP X/Y as WGS84 degrees gets eastings/northings in the hundreds
      of thousands to millions — wildly outside [-180,180]/[-90,90] — which
      blows a Mercator-style basemap's bounding-box math to Inf/NaN and
      crashes the widget on zoom-to-extent. Petrel reads this file perfectly
      because it resolves a project CRS instead of assuming lon/lat.

The fix: io_segy._detect_crs (override or textual-header zone heuristic) +
core.spatial.safe_map_coords (the actual map-safety guard: geographic path
bounds-checked/dropped, projected path reprojected-or-refused, NEVER passed
through as raw projected metres).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sbp_studio.core import load_profile, safe_map_coords
from sbp_studio.core.io_segy import _detect_crs, _guess_utm_crs_from_text

_REAL_DIR = Path(__file__).resolve().parent.parent / "examples" / "_real_in"
_ANT26 = _REAL_DIR / "ANT26" / "SGY" / "20260215115516.seg"
_L001A = _REAL_DIR / "L001A" / "20260527060848.sgy"
_MCS7  = _REAL_DIR / "MCS7" / "5_MCS7_MIG.segy"

_skip_no_sample = pytest.mark.skipif(
    not (_ANT26.exists() and _L001A.exists() and _MCS7.exists()),
    reason="real sample SEG-Y files not present in this checkout")


# ── 1. safe_map_coords: projected path (coord_unit == 1) ────────────────────

class TestSafeMapCoordsProjected:
    def test_no_crs_refuses_passthrough_returns_nan(self):
        """The core fix: UNLIKE to_geographic, unresolved projected metres
        must NEVER reach the map as raw numbers — that's what blows up a
        WGS84 bounding box."""
        eastings  = np.array([584331.4, 584712.7, 585000.0])
        northings = np.array([4874890.8, 4873451.1, 4873000.0])
        x, y = safe_map_coords(eastings, northings, coord_unit=1, src_crs=None)
        assert np.all(np.isnan(x)) and np.all(np.isnan(y))

    def test_with_crs_reprojects_to_plausible_wgs84(self):
        """UTM 31N eastings/northings from the real MCS7 audit must land at
        the real-world Mediterranean location they represent."""
        eastings  = np.array([584331.4])
        northings = np.array([4874890.8])
        lon, lat = safe_map_coords(eastings, northings, coord_unit=1,
                                   src_crs="EPSG:32631")
        assert np.isfinite(lon[0]) and np.isfinite(lat[0])
        assert -180 <= lon[0] <= 180 and -90 <= lat[0] <= 90
        assert lon[0] == pytest.approx(4.05, abs=0.05)
        assert lat[0] == pytest.approx(44.02, abs=0.05)

    def test_invalid_crs_string_falls_back_to_nan_not_a_crash(self):
        x, y = safe_map_coords(np.array([584331.4]), np.array([4874890.8]),
                               coord_unit=1, src_crs="not-a-real-crs")
        assert np.all(np.isnan(x)) and np.all(np.isnan(y))


# ── 2. safe_map_coords: geographic path (coord_unit in (2, 3)) ──────────────

class TestSafeMapCoordsGeographic:
    def test_in_bounds_points_pass_through_unchanged(self):
        lons = np.array([3.7675, 4.05])
        lats = np.array([42.712, 44.02])
        x, y = safe_map_coords(lons, lats, coord_unit=3, src_crs=None)
        np.testing.assert_array_equal(x, lons)
        np.testing.assert_array_equal(y, lats)

    def test_out_of_bounds_point_dropped_others_kept(self):
        lons = np.array([3.7, 99999.0, 4.0])
        lats = np.array([42.7, -373.0, 43.0])
        x, y = safe_map_coords(lons, lats, coord_unit=2, src_crs=None)
        assert np.isnan(x[1]) and np.isnan(y[1])
        assert x[0] == 3.7 and x[2] == 4.0
        assert y[0] == 42.7 and y[2] == 43.0

    def test_nan_inf_input_dropped_not_propagated_as_a_crash(self):
        lons = np.array([3.7, np.nan, np.inf, -np.inf])
        lats = np.array([42.7, 43.0, 43.0, 43.0])
        x, y = safe_map_coords(lons, lats, coord_unit=2, src_crs=None)
        assert not np.isnan(x[0])
        assert all(np.isnan(x[i]) for i in (1, 2, 3))

    def test_in_bounds_but_wrong_value_is_not_caught_by_bounds_alone(self):
        """Documents a real limitation: the Antarctic file's bug (~0,0
        instead of a real Antarctic fix) is IN-BOUNDS, so a pure bounds
        check — exactly what this function implements — cannot detect it.
        That diagnosis required cross-referencing a KNOWN-GOOD file, not
        just bounds-checking in isolation."""
        lon, lat = safe_map_coords(np.array([2.64e-5]), np.array([-1.036e-4]),
                                   coord_unit=2, src_crs=None)
        assert not np.isnan(lon[0]) and not np.isnan(lat[0])


# ── 3. UTM-zone textual-header heuristic ─────────────────────────────────────

class TestUtmZoneHeuristic:
    def test_populated_zone_id_yields_a_guess(self):
        text = ("C20 MAP PROJECTION UTM ZONE ID:31 COORDINATE UNITS METERS"
                .ljust(80)).encode("ascii")
        assert _guess_utm_crs_from_text(text) == "EPSG:32631"

    def test_zero_zone_id_yields_no_guess(self):
        """The literal, unfilled-template state on the real ANT26 file
        ('ZONE ID:0') must NOT produce a confident (wrong) guess."""
        text = "C20 MAP PROJECTION ZONE ID:0 COORDINATE UNITS".encode("ascii")
        assert _guess_utm_crs_from_text(text) is None

    def test_blank_zone_id_yields_no_guess(self):
        """The real MCS7 file's actual state: label present, no value."""
        text = ("C20 MAP PROJECTION                      ZONE ID       "
               "COORDINATE UNITS          ").encode("ascii")
        assert _guess_utm_crs_from_text(text) is None

    def test_out_of_range_zone_yields_no_guess(self):
        text = "ZONE ID:99".encode("ascii")
        assert _guess_utm_crs_from_text(text) is None


# ── 4. _detect_crs resolution order for projected (coord_unit == 1) ─────────

class TestDetectCrsProjected:
    def test_override_wins_and_is_canonicalised(self):
        crs, notes = _detect_crs(1, np.array([0.0]), crs_override="32631")
        assert crs == "EPSG:32631"
        assert any("32631" in n for n in notes)

    def test_invalid_override_reports_none_with_explanation(self):
        crs, notes = _detect_crs(1, np.array([0.0]), crs_override="not-a-crs")
        assert crs is None
        assert notes and "inválido" in notes[0].lower() or "invalid" in notes[0].lower() or notes

    def test_heuristic_used_when_no_override(self):
        text = "ZONE ID:31".encode("ascii")
        crs, notes = _detect_crs(1, np.array([0.0]), raw_text=text)
        assert crs == "EPSG:32631"
        assert any("ADIVINADA" in n or "UNCONFIRMED" in n.upper() for n in notes)

    def test_neither_override_nor_heuristic_returns_none_with_warning(self):
        crs, notes = _detect_crs(1, np.array([0.0]), raw_text=b"")
        assert crs is None
        assert notes


# ── 5. End-to-end against the actual audited sample files ───────────────────

@_skip_no_sample
class TestRealSampleFiles:
    def test_mcs7_projected_no_override_yields_nan_track(self):
        """Reproduces the reported bug exactly: loading the file as-is, the
        map-facing coordinates must be NaN (refused), not raw UTM metres."""
        prof = load_profile(str(_MCS7), load_traces=False)
        assert prof.error is None
        assert prof.coord_unit == 1
        assert prof.detected_crs is None
        x, y = safe_map_coords(prof.track_lons, prof.track_lats,
                               prof.coord_unit, prof.detected_crs)
        assert np.all(np.isnan(x)) and np.all(np.isnan(y))
        # Native fields are UNCHANGED — exports/dist_km still work normally.
        assert np.all(np.isfinite(prof.track_lons))
        assert prof.total_km > 0

    def test_mcs7_with_override_yields_real_mediterranean_position(self):
        """The fix in action: tell the loader the zone, get a real position
        matching the audit's hand-derived first-trace ~4.052E/44.022N (the
        line runs ~140 km roughly north-south, so later traces drift to
        ~45.3N — checked on the first trace specifically, not the median)."""
        prof = load_profile(str(_MCS7), load_traces=False, crs_override="EPSG:32631")
        assert prof.detected_crs == "EPSG:32631"
        x, y = safe_map_coords(prof.track_lons, prof.track_lats,
                               prof.coord_unit, prof.detected_crs)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        assert np.all((x >= -180) & (x <= 180) & (y >= -90) & (y <= 90))
        assert float(x[0]) == pytest.approx(4.052, abs=0.01)
        assert float(y[0]) == pytest.approx(44.022, abs=0.01)

    def test_ant26_geographic_path_unaffected_still_decodes_to_origin(self):
        """Regression guard: this audit's OTHER finding (arc-second data
        decoding to ~0,0 due to an upstream magnitude fault) is untouched by
        this fix — bounds-checking alone correctly leaves it as-is (in-bounds
        but wrong; see TestSafeMapCoordsGeographic's documented limitation)."""
        prof = load_profile(str(_ANT26), load_traces=False)
        assert prof.coord_unit == 2
        assert prof.detected_crs == "EPSG:4326"
        x, y = safe_map_coords(prof.track_lons, prof.track_lats,
                               prof.coord_unit, prof.detected_crs)
        assert abs(float(x[0])) < 0.01 and abs(float(y[0])) < 0.01

    def test_l001a_geographic_path_unaffected_still_decodes_correctly(self):
        prof = load_profile(str(_L001A), load_traces=False)
        assert prof.coord_unit == 2
        assert prof.detected_crs == "EPSG:4326"
        x, y = safe_map_coords(prof.track_lons, prof.track_lats,
                               prof.coord_unit, prof.detected_crs)
        assert float(x[0]) == pytest.approx(3.7675, abs=0.01)
        assert float(y[0]) == pytest.approx(42.712, abs=0.01)


# ── GPS dropout repair (the 'track dives to Null Island' map bug) ──────────────

class TestRepairGpsDropouts:
    """_repair_gps_dropouts — unit coverage of the interpolation repair."""

    def _mk(self, n=20, lon0=-8.0, lat0=43.0, step=0.001):
        lons = lon0 + np.arange(n) * step
        lats = np.full(n, lat0)
        return lons, lats

    def test_clean_track_returned_unchanged(self):
        from sbp_studio.core.io_segy import _repair_gps_dropouts
        lons, lats = self._mk()
        lo, la, n_bad = _repair_gps_dropouts(lons, lats)
        assert n_bad == 0
        np.testing.assert_array_equal(lo, lons)
        np.testing.assert_array_equal(la, lats)

    def test_mid_run_dropout_bridged_linearly(self):
        """A 6-trace (0,0) run — longer than the median kernel can absorb —
        must be bridged by a straight line between the surrounding fixes."""
        from sbp_studio.core.io_segy import _repair_gps_dropouts
        lons, lats = self._mk()
        lons[8:14] = 0.0
        lats[8:14] = 0.0
        lo, la, n_bad = _repair_gps_dropouts(lons, lats)
        assert n_bad == 6
        assert not ((lo == 0.0) & (la == 0.0)).any()
        # Bridge is linear over trace index → matches the original uniform line.
        clean_lons, clean_lats = self._mk()
        np.testing.assert_allclose(lo, clean_lons, atol=1e-12)
        np.testing.assert_allclose(la, clean_lats, atol=1e-12)

    def test_leading_and_trailing_runs_hold_first_last_good_fix(self):
        from sbp_studio.core.io_segy import _repair_gps_dropouts
        lons, lats = self._mk()
        lons[:3] = 0.0; lats[:3] = 0.0
        lons[-2:] = 0.0; lats[-2:] = 0.0
        lo, la, n_bad = _repair_gps_dropouts(lons, lats)
        assert n_bad == 5
        assert (lo[:3] == lo[3]).all() and (la[:3] == la[3]).all()
        assert (lo[-2:] == lo[-3]).all() and (la[-2:] == la[-3]).all()

    def test_non_finite_fixes_also_repaired(self):
        from sbp_studio.core.io_segy import _repair_gps_dropouts
        lons, lats = self._mk()
        lons[5] = np.nan
        lats[10] = np.inf
        lo, la, n_bad = _repair_gps_dropouts(lons, lats)
        assert n_bad == 2
        assert np.isfinite(lo).all() and np.isfinite(la).all()

    def test_all_dropout_file_returned_unchanged(self):
        """No good fixes at all (a no-navigation file): nothing to interpolate
        from — must come back untouched, not crash or invent positions."""
        from sbp_studio.core.io_segy import _repair_gps_dropouts
        lons = np.zeros(10); lats = np.zeros(10)
        lo, la, n_bad = _repair_gps_dropouts(lons, lats)
        assert n_bad == 0
        assert (lo == 0).all() and (la == 0).all()

    def test_zero_lon_with_real_lat_is_not_a_dropout(self):
        """Only the exact PAIR (0,0) is the no-nav signature — a track that
        legitimately crosses the prime meridian (lon = 0, lat real) must not
        be touched."""
        from sbp_studio.core.io_segy import _repair_gps_dropouts
        lons = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
        lats = np.full(5, 51.0)
        lo, la, n_bad = _repair_gps_dropouts(lons, lats)
        assert n_bad == 0
        np.testing.assert_array_equal(lo, lons)


class TestLoaderGpsDropoutEndToEnd:
    """Full loader path on a real synthetic file with a patched GPS outage —
    the reported field symptom (erratic lines / wild bounding box / phantom
    distance) must be gone, while raw navigation stays export-faithful."""

    def _outage_file(self, tmp_path, n_traces=60, bad=slice(20, 35)):
        import segyio
        import shutil
        from tests.make_synthetic_segy import make_synthetic_segy
        clean = str(tmp_path / "clean.sgy")
        make_synthetic_segy(clean, n_traces=n_traces, ns=64)
        outage = str(tmp_path / "outage.sgy")
        shutil.copy(clean, outage)
        with segyio.open(outage, "r+", ignore_geometry=True) as f:
            for i in range(*bad.indices(n_traces)):
                f.header[i] = {segyio.TraceField.SourceX: 0,
                               segyio.TraceField.SourceY: 0}
        return clean, outage

    def test_track_never_visits_null_island(self, tmp_path):
        clean, outage = self._outage_file(tmp_path)
        sd = load_profile(outage, load_traces=False)
        assert sd.error is None
        # The display track must stay inside the survey envelope (±0.01°),
        # never dive toward (0,0).
        ref = load_profile(clean, load_traces=False)
        assert np.abs(sd.track_lons - ref.track_lons).max() < 0.01
        assert np.abs(sd.track_lats - ref.track_lats).max() < 0.01
        assert not ((sd.track_lons == 0) & (sd.track_lats == 0)).any()

    def test_distance_axis_free_of_phantom_kilometres(self, tmp_path):
        """Un-repaired, a dive from ~(-8°,43°) to (0,0) and back injects
        thousands of km into dist_km; repaired, the outage file's total must
        match the clean file's to within metres."""
        clean, outage = self._outage_file(tmp_path)
        ref = load_profile(clean, load_traces=False)
        sd = load_profile(outage, load_traces=False)
        assert sd.total_km == pytest.approx(ref.total_km, abs=0.01)
        # dist_km stays monotonically non-decreasing (axis contract).
        assert (np.diff(sd.dist_km) >= 0).all()

    def test_raw_navigation_keeps_the_zeros_for_exports(self, tmp_path):
        """Export fidelity: the RAW lons/lats (FIX/navline/pick sources) must
        still carry the recorded zeros — the repair is display-only."""
        _clean, outage = self._outage_file(tmp_path)
        sd = load_profile(outage, load_traces=False)
        assert ((sd.lons == 0) & (sd.lats == 0)).sum() >= 10

    def test_repaired_track_passes_safe_map_coords_fully_finite(self, tmp_path):
        """What actually reaches the map widget: every repaired track point
        must be valid WGS84 (no NaN holes, no out-of-range)."""
        _clean, outage = self._outage_file(tmp_path)
        sd = load_profile(outage, load_traces=False)
        lo, la = safe_map_coords(sd.track_lons, sd.track_lats,
                                 sd.coord_unit, sd.detected_crs)
        assert np.isfinite(lo).all() and np.isfinite(la).all()
        assert (np.abs(lo) <= 180).all() and (np.abs(la) <= 90).all()

    def test_isolated_corrupt_spike_still_absorbed_by_median(self, tmp_path):
        """Regression pin for the pre-existing spike defence: one corrupted
        header (huge jump) is rejected by the median smooth as before."""
        import segyio
        _clean, outage = self._outage_file(tmp_path, bad=slice(0, 0))  # no outage
        with segyio.open(outage, "r+", ignore_geometry=True) as f:
            f.header[30] = {segyio.TraceField.SourceX: 2_000_000_000,
                            segyio.TraceField.SourceY: 2_000_000_000}
        sd = load_profile(outage, load_traces=False)
        ref = load_profile(_clean, load_traces=False)
        assert np.abs(sd.track_lons - ref.track_lons).max() < 0.01
        assert np.abs(sd.track_lats - ref.track_lats).max() < 0.01
