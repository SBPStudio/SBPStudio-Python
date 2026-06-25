"""
test_dc_removal.py — mandatory DC-offset removal.

Covers core.apply_dc_removal directly (unit-level: mean exactly zeroed,
inplace vs new-array, dtype/no-mutation contract, NaN/Inf-safety on
degenerate traces, float64-accumulator precision on long traces) AND the
end-to-end load path (io_segy._populate_profile_from_file applies it as the
very first thing done to a freshly-loaded trace matrix, before amp_max /
clip_p99 are computed from it).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import segyio

from sbp_studio.core import apply_dc_removal, load_profile


# ── Unit tests: apply_dc_removal ────────────────────────────────────────────

class TestApplyDcRemovalMath:
    def test_centers_every_trace_on_zero(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((500, 12)).astype(np.float32)
        bias = np.arange(12, dtype=np.float32) * 37.5 - 200.0   # distinct DC per trace
        data += bias
        out = apply_dc_removal(data)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-4)

    def test_default_does_not_mutate_input(self):
        data = (np.ones((10, 4), dtype=np.float32) * 5.0)
        original = data.copy()
        out = apply_dc_removal(data)
        assert np.array_equal(data, original)          # untouched
        assert not np.array_equal(out, original)        # new array is zeroed

    def test_inplace_mutates_and_returns_same_buffer(self):
        data = (np.ones((10, 4), dtype=np.float32) * 5.0)
        out = apply_dc_removal(data, inplace=True)
        assert out is data
        assert np.allclose(data, 0.0)

    def test_preserves_shape_and_dtype(self):
        data = np.random.default_rng(1).standard_normal((300, 7)).astype(np.float32) + 50.0
        out = apply_dc_removal(data)
        assert out.shape == data.shape
        assert out.dtype == data.dtype

    def test_constant_dead_trace_is_safe(self):
        """A degenerate all-constant (dead-channel) trace must collapse to
        exactly zero, never NaN/Inf — pure subtraction, no division anywhere."""
        data = np.full((50, 1), 1234.5, dtype=np.float32)
        out = apply_dc_removal(data)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)

    def test_all_zero_trace_is_safe(self):
        data = np.zeros((50, 3), dtype=np.float32)
        out = apply_dc_removal(data)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)

    def test_robust_for_long_traces(self):
        """float64 accumulator: a long trace (thousands of samples, large
        common bias) must still be centred to a tight tolerance."""
        ns = 20000
        rng = np.random.default_rng(2)
        data = (rng.standard_normal((ns, 3)).astype(np.float32) * 0.01 + 1e5)
        out = apply_dc_removal(data)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-2)

    def test_single_sample_trace(self):
        data = np.array([[42.0]], dtype=np.float32)
        out = apply_dc_removal(data)
        assert out[0, 0] == 0.0


# ── End-to-end: load path applies DC removal as the very first stage ───────

def _write_biased_segy(path: str, biases) -> str:
    """Minimal synthetic SEG-Y: each trace is a constant value (its own
    'DC bias'), so post-load the entire array must be exactly zero once
    apply_dc_removal has run — an unambiguous, exact end-to-end check."""
    ns = 64
    n_traces = len(biases)
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1
    spec.samples = np.arange(ns, dtype=np.float32)
    spec.tracecount = n_traces
    with segyio.create(path, spec) as f:
        f.bin.update(tsort=segyio.TraceSortingFormat.UNKNOWN_SORTING, hdt=250, dto=250)
        for i, b in enumerate(biases):
            f.header[i] = {
                segyio.TraceField.SourceX: 0, segyio.TraceField.SourceY: 0,
                segyio.TraceField.GroupX: 0, segyio.TraceField.GroupY: 0,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CoordinateUnits: 3,
                segyio.TraceField.YearDataRecorded: 2024,
                segyio.TraceField.DayOfYear: 100,
                segyio.TraceField.HourOfDay: 0,
                segyio.TraceField.MinuteOfHour: i % 60,
                segyio.TraceField.SecondOfMinute: 0,
            }
            f.trace[i] = np.full(ns, float(b), dtype=np.float32)
    return path


class TestLoadPathAppliesDcRemoval:
    def test_constant_biased_traces_load_as_exact_zero(self, tmp_path):
        biases = [1000.0, -2500.0, 0.0, 37.25, -1.0]
        p = _write_biased_segy(str(tmp_path / "biased.sgy"), biases)
        sd = load_profile(p, load_traces=True)
        assert np.allclose(sd.data, 0.0)

    def test_amp_max_and_clip_p99_reflect_post_removal_signal(self, tmp_path):
        """amp_max / clip_p99 are computed from data AFTER DC removal — for
        a pure-DC (no-AC) trace this must be ~0, not the raw bias magnitude."""
        p = _write_biased_segy(str(tmp_path / "biased2.sgy"), [5000.0, 5000.0])
        sd = load_profile(p, load_traces=True)
        assert np.allclose(sd.amp_max, 0.0, atol=1e-3)
        assert sd.clip_p99 < 1.0

    def test_real_signal_plus_dc_bias_round_trips_to_ac_only(self, tmp_path):
        """A trace with a real AC waveform riding on a large DC bias must,
        post-load, match the AC waveform's own zero-mean shape (bias fully
        removed, signal shape untouched) — compared against ac - ac.mean()
        rather than ac itself, since an arbitrary sine window need not
        average to exactly zero over a non-integer number of cycles."""
        ns, dt_us = 200, 250
        # 5 complete periods of a 100 Hz tone across the 200-sample window
        # (200 * 250us = 50ms = 5 periods of 10ms) -> ac.mean() ~ 0 already,
        # so the assertion is a meaningful, near-exact round-trip check.
        ac = (5.0 * np.sin(2 * np.pi * 100.0 * np.arange(ns) * dt_us * 1e-6)).astype(np.float32)
        bias = 837.0
        spec = segyio.spec()
        spec.sorting = None
        spec.format = 1
        spec.samples = np.arange(ns, dtype=np.float32)
        spec.tracecount = 1
        path = str(tmp_path / "ac_plus_dc.sgy")
        with segyio.create(path, spec) as f:
            f.bin.update(tsort=segyio.TraceSortingFormat.UNKNOWN_SORTING, hdt=dt_us, dto=dt_us)
            f.header[0] = {
                segyio.TraceField.SourceX: 0, segyio.TraceField.SourceY: 0,
                segyio.TraceField.GroupX: 0, segyio.TraceField.GroupY: 0,
                segyio.TraceField.SourceGroupScalar: 1,
                segyio.TraceField.CoordinateUnits: 3,
                segyio.TraceField.YearDataRecorded: 2024,
                segyio.TraceField.DayOfYear: 100,
                segyio.TraceField.HourOfDay: 0,
                segyio.TraceField.MinuteOfHour: 0,
                segyio.TraceField.SecondOfMinute: 0,
            }
            f.trace[0] = ac + bias
        sd = load_profile(path, load_traces=True)
        assert np.allclose(sd.data[:, 0], ac, atol=1e-2)

    def test_header_only_load_is_unaffected(self, tmp_path):
        """load_traces=False never touches `.data` (None) — DC removal only
        runs when there is trace data to act on."""
        p = _write_biased_segy(str(tmp_path / "biased3.sgy"), [10.0, 20.0])
        sd = load_profile(p, load_traces=False)
        assert sd.data is None
