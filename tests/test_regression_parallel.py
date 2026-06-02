"""
test_regression_parallel.py — Regression gates for all parallelised code paths.

Two-track contract
------------------
Every optimised path MUST satisfy:
  np.testing.assert_allclose(optimised_output, reference_output, atol=TOLERANCE)
before it ships as the default.

Tests in this file
------------------
1. Hilbert (envelope)    : _hilbert_parallel vs scipy.signal.hilbert sequential
2. Hilbert (inst_phase)  : same framework
3. AGC parallel          : _parallel_apply(_agc_block) vs _ref_agc sequential
4. Vectorised reprojection: _opt_reproject_coords_bulk vs _ref_reproject_trace
5. join_profiles         : pure copy vs reproject_chain with identity transform
                           (coord equality + preserved headers)
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import pytest
import segyio

from topassuite.core.processing import (
    _hilbert_parallel, _ref_agc, _parallel_apply,
)
from topassuite.core.io_segy import (
    _ref_reproject_trace, _opt_reproject_coords_bulk,
    _build_transformer, _safe_coord,
    join_profiles, reproject_chain, load_profile,
)
from topassuite.core import detect_chains


# ── Tolerances (documented) ────────────────────────────────────────────────────
#
# HILBERT_ATOL: float64 rounding differences only (scipy.signal.hilbert uses
#   the same FFT path internally; _parallel_apply splits traces but each block
#   uses identical arithmetic). Expected diff ≈ 1e-16 (machine eps), gate 1e-6.
#
# AGC_ATOL: uniform_filter1d is deterministic and identical in both paths;
#   only floating-point reassociation from block splitting. Gate 1e-6.
#
# REPROJ_ATOL: _opt_reproject_coords_bulk uses identical arithmetic to
#   _ref_reproject_trace, just vectorised. Gate 1e-10 (float64 precision).
#
HILBERT_ATOL = 1e-6
AGC_ATOL     = 1e-6
REPROJ_ATOL  = 1e-10


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def random_data():
    """Synthetic (ns=256, n_traces=100) float32 array with realistic amplitudes."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((256, 100)).astype(np.float32)


@pytest.fixture(scope="module")
def large_data():
    """Larger array to catch memory-layout edge cases."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((512, 200)).astype(np.float32)


# ── 1. Hilbert: envelope ──────────────────────────────────────────────────────

class TestHilbertParallelEnvelope:
    """
    _hilbert_parallel(key='envelope') must match scipy.signal.hilbert sequential.
    Tolerance: allclose(atol=1e-6) — float64 rounding only.
    """

    def test_small_array(self, random_data):
        from scipy.signal import hilbert
        ref = np.abs(hilbert(random_data, axis=0)).astype(np.float32)
        opt = _hilbert_parallel(random_data, "envelope", dt_us=250,
                                fs=1e6/250)
        np.testing.assert_allclose(opt, ref, atol=HILBERT_ATOL,
                                   err_msg="envelope: parallel vs ref mismatch")

    def test_large_array(self, large_data):
        from scipy.signal import hilbert
        ref = np.abs(hilbert(large_data, axis=0)).astype(np.float32)
        opt = _hilbert_parallel(large_data, "envelope", dt_us=250,
                                fs=1e6/250)
        np.testing.assert_allclose(opt, ref, atol=HILBERT_ATOL)

    def test_output_non_negative(self, random_data):
        opt = _hilbert_parallel(random_data, "envelope", dt_us=250, fs=4000)
        assert np.all(opt >= 0), "Envelope must be non-negative"

    def test_output_dtype(self, random_data):
        opt = _hilbert_parallel(random_data, "envelope", dt_us=250, fs=4000)
        assert opt.dtype == np.float32

    def test_output_shape(self, random_data):
        opt = _hilbert_parallel(random_data, "envelope", dt_us=250, fs=4000)
        assert opt.shape == random_data.shape


# ── 2. Hilbert: inst_phase ────────────────────────────────────────────────────

class TestHilbertParallelInstPhase:
    """
    _hilbert_parallel(key='inst_phase') must match sequential.
    """

    def test_allclose(self, random_data):
        from scipy.signal import hilbert
        ref = np.angle(hilbert(random_data, axis=0)).astype(np.float32)
        opt = _hilbert_parallel(random_data, "inst_phase", dt_us=250,
                                fs=4000)
        np.testing.assert_allclose(opt, ref, atol=HILBERT_ATOL)

    def test_range(self, random_data):
        opt = _hilbert_parallel(random_data, "inst_phase", dt_us=250, fs=4000)
        assert np.all(opt >= -np.pi - 1e-5) and np.all(opt <= np.pi + 1e-5)


# ── 3. AGC parallel ──────────────────────────────────────────────────────────

class TestAGCParallel:
    """
    Parallel AGC (_parallel_apply over _agc_block) must match _ref_agc.
    Tolerance: allclose(atol=1e-6).
    """

    def _run_parallel_agc(self, data, win_ms, dt_us):
        """Run the same AGC block logic as _process_data_generic."""
        from scipy.ndimage import uniform_filter1d
        win_s = max(3, int(win_ms / (dt_us / 1000.0)))
        if win_s % 2 == 0:
            win_s += 1

        def _agc_block(blk, _w=win_s):
            env = np.abs(blk)
            rms = uniform_filter1d(env, size=_w, axis=0)
            return (blk / np.maximum(rms, 1e-9)).astype(np.float32)

        return _parallel_apply(_agc_block, data)

    def test_allclose_small(self, random_data):
        win_ms = 50.0; dt_us = 250
        ref = _ref_agc(random_data, win_ms, dt_us)
        opt = self._run_parallel_agc(random_data, win_ms, dt_us)
        np.testing.assert_allclose(opt, ref, atol=AGC_ATOL,
                                   err_msg="AGC parallel vs ref mismatch")

    def test_allclose_large(self, large_data):
        win_ms = 30.0; dt_us = 125
        ref = _ref_agc(large_data, win_ms, dt_us)
        opt = self._run_parallel_agc(large_data, win_ms, dt_us)
        np.testing.assert_allclose(opt, ref, atol=AGC_ATOL)

    def test_output_dtype(self, random_data):
        opt = self._run_parallel_agc(random_data, 50.0, 250)
        assert opt.dtype == np.float32

    def test_rms_normalised(self, random_data):
        """After AGC, RMS over long windows should be roughly uniform."""
        from scipy.ndimage import uniform_filter1d
        opt = self._run_parallel_agc(random_data, 50.0, 250)
        col_rms = np.sqrt(np.mean(opt**2, axis=0))
        # All columns should have similar RMS after AGC
        assert np.std(col_rms) < np.mean(col_rms) * 0.5


# ── 4. Vectorised reprojection ────────────────────────────────────────────────

class TestReprojectVectorised:
    """
    _opt_reproject_coords_bulk must produce coordinates that match the
    reference per-trace _ref_reproject_trace within float64 rounding.
    Tolerance: allclose(atol=1e-10).
    """

    def _make_segy_with_coords(self, tmp_dir, n_traces=30,
                                scalar=-100, coord_unit=3,
                                base_lon=-8.0, base_lat=43.0):
        """Write a synthetic SEG-Y with known geographic coordinates."""
        import sys; sys.path.insert(0, str(
            __import__('pathlib').Path(__file__).parent.parent))
        from tests.make_synthetic_segy import make_synthetic_segy
        path = str(__import__('pathlib').Path(tmp_dir) / "reproj_test.sgy")
        make_synthetic_segy(path, n_traces=n_traces, ns=64, dt_us=250,
                            scalar_coord=scalar, coord_unit=coord_unit,
                            base_lon=base_lon, base_lat=base_lat)
        return path

    def test_geographic_to_projected(self, tmp_path):
        """Reproject WGS84 → UTM30N; opt coords match ref within 1e-10."""
        path = self._make_segy_with_coords(str(tmp_path))
        tf, out_sc, new_uc, _, div = _build_transformer("EPSG:4326", "EPSG:32630")

        with segyio.open(path, ignore_geometry=True) as f:
            # Reference: per-trace
            ref_xs = []; ref_ys = []
            for i in range(f.tracecount):
                nx, ny = _ref_reproject_trace(
                    f.header[i], tf, 2, out_sc, new_uc, div)
                ref_xs.append(nx); ref_ys.append(ny)
            ref_xs = np.array(ref_xs); ref_ys = np.array(ref_ys)

            # Optimised: bulk
            opt_xs, opt_ys = _opt_reproject_coords_bulk(f, tf, 2)

        np.testing.assert_allclose(opt_xs, ref_xs, atol=REPROJ_ATOL,
                                   err_msg="Vectorised X coords mismatch ref")
        np.testing.assert_allclose(opt_ys, ref_ys, atol=REPROJ_ATOL,
                                   err_msg="Vectorised Y coords mismatch ref")

    def test_arcsec_coords(self, tmp_path):
        """Arc-second input (coord_unit=2) handled correctly by both paths."""
        from tests.make_synthetic_segy import make_synthetic_segy
        path = str(tmp_path / "arcsec.sgy")
        make_synthetic_segy(path, n_traces=20, ns=64, dt_us=250,
                            scalar_coord=-100, coord_unit=2)
        tf, out_sc, new_uc, _, div = _build_transformer("EPSG:4326", "EPSG:32630")

        with segyio.open(path, ignore_geometry=True) as f:
            ref_xs = [_ref_reproject_trace(f.header[i], tf, 2, out_sc, new_uc, div)[0]
                      for i in range(f.tracecount)]
            opt_xs, _ = _opt_reproject_coords_bulk(f, tf, 2)

        np.testing.assert_allclose(opt_xs, ref_xs, atol=REPROJ_ATOL)

    def test_scalar_zero_treated_as_one(self, tmp_path):
        """scalar_coord=0 → fac=1.0 in both reference and vectorised path."""
        from tests.make_synthetic_segy import make_synthetic_segy
        path = str(tmp_path / "sc0.sgy")
        make_synthetic_segy(path, n_traces=10, ns=64, dt_us=250,
                            scalar_coord=0, coord_unit=3)
        tf, out_sc, new_uc, _, div = _build_transformer("EPSG:4326", "EPSG:4258")

        with segyio.open(path, ignore_geometry=True) as f:
            ref_xs = [_ref_reproject_trace(f.header[i], tf, 3, out_sc, new_uc, div)[0]
                      for i in range(f.tracecount)]
            opt_xs, _ = _opt_reproject_coords_bulk(f, tf, 3)

        np.testing.assert_allclose(opt_xs, ref_xs, atol=REPROJ_ATOL)


# ── 5. join_profiles (pure copy) ──────────────────────────────────────────────

class TestJoinProfiles:
    """
    join_profiles (fast copy path) must produce the same coordinate values
    as the source files and preserve DelayRecordingTime verbatim.
    """

    def test_output_trace_count(self, chain_pair, tmp_path):
        path1, path2 = chain_pair
        src1 = str(tmp_path / "jc1.sgy"); src2 = str(tmp_path / "jc2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        ch       = chains[0]

        out = join_profiles(ch)
        with segyio.open(out, ignore_geometry=True) as f:
            assert f.tracecount == ch.n_traces
        os.remove(out)

    def test_delay_preserved(self, chain_pair, tmp_path):
        """DelayRecordingTime must be identical in source and joined file."""
        from tests.make_synthetic_segy import make_synthetic_segy
        p1 = str(tmp_path / "d1.sgy")
        p2 = str(tmp_path / "d2.sgy")
        make_synthetic_segy(p1, delay_ms=20, delay_variable=True)
        make_synthetic_segy(p2, delay_ms=20, delay_variable=True,
                            base_lon=-7.9, doy_start=101)
        profiles = [load_profile(p1), load_profile(p2)]
        chains   = detect_chains(profiles, gap_km=5.0)
        if not chains:
            pytest.skip("profiles too far apart for chain detection")
        ch = chains[0]

        out = join_profiles(ch)
        src_delays = np.concatenate([
            np.asarray([load_profile(p.path).delays for p in ch.profiles]
                       [0].tolist() if hasattr(load_profile(p.path).delays, 'tolist')
                       else load_profile(p.path).delays)
            for p in ch.profiles])
        with segyio.open(out, ignore_geometry=True) as f:
            out_delays = np.asarray(
                f.attributes(segyio.TraceField.DelayRecordingTime)[:])
        np.testing.assert_array_equal(
            out_delays, ch.delays, err_msg="DelayRecordingTime not preserved")
        os.remove(out)

    def test_trace_number_sequential(self, chain_pair, tmp_path):
        path1, path2 = chain_pair
        src1, src2 = str(tmp_path / "jc1.sgy"), str(tmp_path / "jc2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        ch       = chains[0]
        out      = join_profiles(ch)
        with segyio.open(out, ignore_geometry=True) as f:
            tnums = np.asarray(f.attributes(segyio.TraceField.TraceNumber)[:])
        np.testing.assert_array_equal(
            tnums, np.arange(1, ch.n_traces + 1),
            err_msg="TraceNumber not sequential in join_profiles output")
        os.remove(out)

    def test_coords_identical_to_source(self, chain_pair, tmp_path):
        """Source coordinates must be preserved unchanged (no transformation)."""
        path1, path2 = chain_pair
        src1, src2 = str(tmp_path / "jc1.sgy"), str(tmp_path / "jc2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        ch       = chains[0]
        out      = join_profiles(ch)

        # Collect source X values
        src_xs = []
        for p in ch.profiles:
            with segyio.open(p.path, ignore_geometry=True) as f:
                src_xs.extend(f.attributes(segyio.TraceField.SourceX)[:])

        with segyio.open(out, ignore_geometry=True) as f:
            out_xs = list(f.attributes(segyio.TraceField.SourceX)[:])

        assert src_xs == out_xs, "SourceX values must be identical after join_profiles"
        os.remove(out)


# ── 6. out_sc fix: geographic scalar is -10_000 (not -10_000_000) ─────────────

class TestOutScFix:
    """
    Regression: after the OQ-3 fix, geographic reprojection must write
    out_sc = -10_000, not the overflowing -10_000_000.
    """

    def test_geographic_scalar_is_minus_10000(self, simple_segy, tmp_path):
        from topassuite.core.io_segy import reproject_one as _ro
        shutil.copy(simple_segy, str(tmp_path / "src.sgy"))
        sd  = load_profile(str(tmp_path / "src.sgy"))
        out = _ro(sd, "EPSG:4326", "EPSG:4258")   # geographic → geographic
        with segyio.open(out, ignore_geometry=True) as f:
            sc = int(f.header[0][segyio.TraceField.SourceGroupScalar])
        assert sc == -10_000, (
            f"Expected out_sc=-10000 (OQ-3 fix), got {sc}. "
            "If -10000000, the overflow bug was not fixed.")
        os.remove(out)

    def test_projected_scalar_unchanged(self, simple_segy, tmp_path):
        from topassuite.core.io_segy import reproject_one as _ro
        shutil.copy(simple_segy, str(tmp_path / "src.sgy"))
        sd  = load_profile(str(tmp_path / "src.sgy"))
        out = _ro(sd, "EPSG:4326", "EPSG:32630")   # geographic → projected
        with segyio.open(out, ignore_geometry=True) as f:
            sc = int(f.header[0][segyio.TraceField.SourceGroupScalar])
        assert sc == -100, f"Projected out_sc should still be -100, got {sc}"
        os.remove(out)
