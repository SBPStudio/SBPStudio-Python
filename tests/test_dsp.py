"""
test_dsp.py — Validation scaffold for the dynamic DSP node pipeline.

Two concerns, kept separate:

1. MATH PURITY — generate a mathematically clean synthetic 1-D seismic trace
   (Ricker wavelet at known reflector times + additive white noise) and push it
   through individual DSP nodes. The output NumPy arrays are asserted against
   analytic expectations, isolating the algorithm so it can be cross-validated
   against commercial standards (Petrel / SeisSpace / ProMAX).

2. PIPELINE MECHANICS (GUI-side, no Qt) — prove the prefix memoization reuses
   upstream node outputs, and that ViewBox extraction returns the correct
   bounding box with halo.

Run:  pytest tests/test_dsp.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from sbp_studio.core import apply_agc, apply_log_compression
from sbp_studio.gui.dsp import (
    AGCNode, DSPContext, LogCompressionNode, Pipeline, extract_visible_window,
)


# ── Synthetic signal generators (mathematically pure) ───────────────────────────

def ricker(n: int, dt_s: float, f0: float) -> np.ndarray:
    """Zero-phase Ricker (Mexican-hat) wavelet, peak at the centre.

    r(t) = (1 - 2 π² f0² t²) · exp(-π² f0² t²)
    """
    t = (np.arange(n) - n // 2) * dt_s
    a = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def synthetic_trace(ns: int = 1000, dt_us: int = 250, f0: float = 800.0,
                    reflectors=((200, 1.0), (500, -0.6), (800, 0.3)),
                    noise_std: float = 0.0, seed: int = 0) -> np.ndarray:
    """A 1-D trace: Ricker wavelets at given (sample_index, amplitude) reflectors,
    optionally plus additive white Gaussian noise. Returns (ns,) float32."""
    dt_s = dt_us / 1e6
    wav = ricker(101, dt_s, f0)
    half = len(wav) // 2
    tr = np.zeros(ns, dtype=np.float32)
    for idx, amp in reflectors:
        lo, hi = idx - half, idx + half + 1
        if lo >= 0 and hi <= ns:
            tr[lo:hi] += amp * wav
    if noise_std > 0:
        tr += np.random.default_rng(seed).normal(0.0, noise_std, ns).astype(np.float32)
    return tr


def as_matrix(trace: np.ndarray, n_traces: int = 8) -> np.ndarray:
    """Tile a 1-D trace into an (ns, n_traces) matrix (the core array contract)."""
    return np.repeat(trace[:, None], n_traces, axis=1).astype(np.float32)


# ── 1. MATH: AGC correctness ────────────────────────────────────────────────────

class TestAGCMath:
    DT_US = 250

    def test_agc_equalises_amplitude_envelope(self):
        """After AGC, a strong early reflector and a weak late reflector should
        have comparable peak magnitude (gain is automatic)."""
        tr = synthetic_trace(ns=1000, dt_us=self.DT_US,
                             reflectors=((200, 1.0), (800, 0.1)), noise_std=0.0)
        data = as_matrix(tr)
        out = apply_agc(data, win_ms=20.0, dt_us=self.DT_US)

        early = float(np.max(np.abs(out[150:250, 0])))
        late  = float(np.max(np.abs(out[750:850, 0])))
        # Pre-AGC ratio is 10:1; post-AGC the weak event is lifted to the
        # same order of magnitude (ratio well under 3:1).
        assert early / max(late, 1e-9) < 3.0

    def test_agc_preserves_shape_and_dtype(self):
        data = as_matrix(synthetic_trace())
        out = apply_agc(data, win_ms=20.0, dt_us=self.DT_US)
        assert out.shape == data.shape
        assert out.dtype == np.float32
        assert out is not data            # never mutates input

    def test_agc_zero_input_safe(self):
        """RMS flooring (1e-9) must keep all-zero input finite (no div-by-zero)."""
        data = np.zeros((256, 4), dtype=np.float32)
        out = apply_agc(data, win_ms=20.0, dt_us=self.DT_US)
        assert np.all(np.isfinite(out))

    def test_agc_matches_core_reference(self):
        """The node must produce EXACTLY the core's apply_agc output (no GUI math)."""
        ctx = DSPContext(dt_us=self.DT_US, ns=1000, n_traces=8)
        data = as_matrix(synthetic_trace(noise_std=0.05))
        node = AGCNode({"win_ms": 25.0})
        np.testing.assert_array_equal(
            node.apply(data, ctx),
            apply_agc(data, 25.0, self.DT_US))


class TestLogCompressionMath:
    DT_US = 250

    def test_log_compression_shrinks_dynamic_range(self):
        """A strong early reflector and a weak late reflector should have their
        peak-amplitude ratio shrink after log compression (weak events boosted
        relative to strong ones), without flipping polarity."""
        tr = synthetic_trace(ns=1000, dt_us=self.DT_US,
                             reflectors=((200, 1.0), (800, 0.1)), noise_std=0.0)
        data = as_matrix(tr)
        out = apply_log_compression(data, k=10.0)

        strong_in  = float(np.max(np.abs(data[150:250, 0])))
        weak_in    = float(np.max(np.abs(data[750:850, 0])))
        strong_out = float(np.max(np.abs(out[150:250, 0])))
        weak_out   = float(np.max(np.abs(out[750:850, 0])))

        ratio_in  = strong_in / max(weak_in, 1e-9)
        ratio_out = strong_out / max(weak_out, 1e-9)
        assert ratio_out < ratio_in

        # Polarity (sign) of each reflector's peak must be preserved.
        assert np.sign(tr[200]) == np.sign(out[200, 0])
        assert np.sign(tr[800]) == np.sign(out[800, 0])

    def test_log_compression_preserves_shape_dtype_and_peak(self):
        data = as_matrix(synthetic_trace())
        out = apply_log_compression(data, k=10.0)
        assert out.shape == data.shape
        assert out.dtype == np.float32
        assert out is not data            # never mutates input
        # Rescaled back to the original peak amplitude (k=0 would be exact
        # identity; for k>0 the peak sample itself maps to +/-max_amp).
        assert np.isclose(float(np.max(np.abs(out))), float(np.max(np.abs(data))), rtol=1e-4)

    def test_log_compression_zero_input_safe(self):
        """The 1e-12 epsilon on max_amp must keep all-zero input finite."""
        data = np.zeros((256, 4), dtype=np.float32)
        out = apply_log_compression(data, k=10.0)
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)

    def test_log_compression_matches_core_reference(self):
        """The node must produce EXACTLY the core's apply_log_compression output."""
        ctx = DSPContext(dt_us=self.DT_US, ns=1000, n_traces=8)
        data = as_matrix(synthetic_trace(noise_std=0.05))
        node = LogCompressionNode({"k": 25.0})
        np.testing.assert_array_equal(
            node.apply(data, ctx),
            apply_log_compression(data, 25.0))


# ── 2. PIPELINE: prefix memoization ─────────────────────────────────────────────

class _CountingAGC(AGCNode):
    """AGC node that counts how many times apply() actually runs (cache probe)."""
    KEY = "agc_count"
    def __init__(self, params=None):
        super().__init__(params)
        self.calls = 0
    def apply(self, data, ctx):
        self.calls += 1
        return super().apply(data, ctx)


class TestMemoization:
    DT_US = 250

    def _ctx(self):
        return DSPContext(dt_us=self.DT_US, ns=512, n_traces=8)

    def test_unchanged_rerun_is_fully_cached(self):
        data = as_matrix(synthetic_trace(ns=512))
        pipe = Pipeline([_CountingAGC({"win_ms": 20.0}),
                         _CountingAGC({"win_ms": 40.0})])
        pipe.process(data, self._ctx(), input_token="W")
        assert pipe.last_compute_count == 2          # cold: both run
        pipe.process(data, self._ctx(), input_token="W")
        assert pipe.last_compute_count == 0          # warm: nothing recomputed

    def test_editing_downstream_reuses_upstream(self):
        """Editing node #2 must reuse node #1's cached output (recompute only #2)."""
        data = as_matrix(synthetic_trace(ns=512))
        n0 = _CountingAGC({"win_ms": 20.0})
        n1 = _CountingAGC({"win_ms": 40.0})
        pipe = Pipeline([n0, n1])
        pipe.process(data, self._ctx(), input_token="W")
        assert (n0.calls, n1.calls) == (1, 1)

        n1.params["win_ms"] = 55.0                   # edit the SECOND node only
        pipe.process(data, self._ctx(), input_token="W")
        assert pipe.last_compute_count == 1          # exactly one node recomputed
        assert (n0.calls, n1.calls) == (1, 2)        # node #1 was NOT re-run

    def test_cache_bounded_to_chain_length(self):
        data = as_matrix(synthetic_trace(ns=256))
        node = AGCNode({"win_ms": 20.0})
        pipe = Pipeline([node])
        for w in (20.0, 30.0, 40.0, 50.0):           # many edits
            node.params["win_ms"] = w
            pipe.process(data, self._ctx(), input_token="W")
        # Cache pruned to the current chain → never grows unbounded.
        assert len(pipe._cache) <= len(pipe.nodes)


# ── 3. PIPELINE: ViewBox-limited extraction ─────────────────────────────────────

class TestViewBoxExtraction:
    DT_US = 250

    def _full(self):
        ns, nt = 1000, 200
        data = as_matrix(synthetic_trace(ns=ns), n_traces=nt)
        dist_km = np.linspace(0.0, 10.0, nt)         # 0..10 km
        return data, dist_km

    def test_window_maps_to_correct_bbox(self):
        data, dist_km = self._full()
        # Visible: 2–4 km, 100–150 ms. t0=0, dt=0.25 ms → rows 400..600.
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(2.0, 4.0), y_range=(100.0, 150.0),
            time_halo=0, trace_halo=0)
        # 2 km → idx 40, 4 km → idx 80 (200 pts / 10 km = 20 pts/km).
        assert win.c0 == 40 and 80 <= win.c1 <= 82
        assert win.s0 == 400 and 600 <= win.s1 <= 601
        assert win.sub.shape[1] == win.c1 - win.c0

    def test_halo_expands_then_crops_back(self):
        data, dist_km = self._full()
        halo = 30
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(2.0, 4.0), y_range=(100.0, 150.0),
            time_halo=halo, trace_halo=0)
        visible_rows = win.s1 - win.s0
        # sub includes halo rows; crop_visible returns exactly the visible band.
        assert win.sub.shape[0] >= visible_rows
        cropped = win.crop_visible(win.sub)
        assert cropped.shape[0] == visible_rows

    def test_clamps_to_array_bounds(self):
        data, dist_km = self._full()
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(-5.0, 99.0), y_range=(-100.0, 1e6),
            time_halo=50, trace_halo=50)
        assert win.c0 == 0 and win.c1 == data.shape[1]
        assert win.sub.shape[0] <= data.shape[0]

    def test_viewbox_preview_matches_fullarray_in_window(self):
        """The crucial guarantee: ViewBox-limited AGC (with halo) equals the
        full-array AGC restricted to the same visible region (edges correct)."""
        data, dist_km = self._full()
        ctx = DSPContext(dt_us=self.DT_US, ns=data.shape[0], n_traces=data.shape[1])
        node = AGCNode({"win_ms": 20.0})
        pipe = Pipeline([node])

        full = apply_agc(data, 20.0, self.DT_US)     # reference: whole array

        halo = node.time_halo_samples(ctx)
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(3.0, 5.0), y_range=(120.0, 160.0),
            time_halo=halo, trace_halo=0)
        preview = pipe.process(win.sub, ctx, input_token=win.token)
        preview_visible = win.crop_visible(preview)
        full_visible = full[win.s0:win.s1, win.c0:win.c1]

        # With the half-window halo the windowed AGC reproduces the full-array
        # result inside the visible band to tight tolerance.
        np.testing.assert_allclose(preview_visible, full_visible, atol=1e-5, rtol=1e-3)

    def test_zoomed_in_is_full_resolution_no_decimation(self):
        """Deep zoom: small window stays under the caps → stride 1, original dt."""
        data, dist_km = self._full()
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(3.0, 4.0), y_range=(120.0, 140.0),
            time_halo=10, max_rows=4000, max_cols=8000)
        assert win.row_stride == 1 and win.col_stride == 1
        assert win.effective_dt_us == self.DT_US


# ── 4. PIPELINE: row decimation + effective-dt (preview perf fix) ────────────────

class TestRowDecimation:
    DT_US = 250

    def _big(self):
        # A tall profile: 40 000 samples × 200 traces (forces row decimation).
        ns, nt = 40_000, 200
        data = np.random.default_rng(1).standard_normal((ns, nt)).astype(np.float32)
        dist_km = np.linspace(0.0, 10.0, nt)
        return data, dist_km

    def test_rows_capped_and_effective_dt_scales(self):
        data, dist_km = self._big()
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(0.0, 10.0), y_range=(0.0, 1e9),   # whole profile visible
            time_halo=0, max_rows=4000, max_cols=8000)
        # Rows decimated under the cap; effective dt scaled by the row stride.
        assert win.sub.shape[0] <= 4000
        assert win.row_stride > 1
        assert win.effective_dt_us == self.DT_US * win.row_stride

    def test_cols_capped_independently(self):
        ns, nt = 1000, 50_000                          # very wide
        data = np.random.default_rng(2).standard_normal((ns, nt)).astype(np.float32)
        dist_km = np.linspace(0.0, 100.0, nt)
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(0.0, 100.0), y_range=(0.0, 1e9),
            time_halo=0, max_rows=4000, max_cols=8000)
        assert win.sub.shape[1] <= 8000
        assert win.col_stride > 1
        assert win.row_stride == 1                      # rows already small
        assert win.effective_dt_us == self.DT_US        # no row decimation → original dt

    def test_crop_indices_valid_after_row_decimation(self):
        data, dist_km = self._big()
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=self.DT_US,
            x_range=(2.0, 5.0), y_range=(1000.0, 4000.0),
            time_halo=200, max_rows=4000, max_cols=8000)
        # Crop indices live in the DECIMATED row space and stay in bounds.
        assert 0 <= win.r0 <= win.r1 <= win.sub.shape[0]
        cropped = win.crop_visible(win.sub)
        assert cropped.shape[0] >= 1
        assert cropped.shape[1] == win.sub.shape[1]

    def test_agc_uses_effective_dt_on_decimated_preview(self):
        """AGC on the decimated preview, fed effective_dt, normalises a known
        amplitude contrast just like the full-res path (physically consistent)."""
        ns, nt = 40_000, 64
        dt = self.DT_US
        # Strong early band, weak late band (10:1) across the whole height.
        tr = np.zeros(ns, dtype=np.float32)
        tr[:ns // 2] = 1.0
        tr[ns // 2:] = 0.1
        tr += np.random.default_rng(3).normal(0, 0.01, ns).astype(np.float32)
        data = as_matrix(tr, nt)
        dist_km = np.linspace(0.0, 5.0, nt)

        node = AGCNode({"win_ms": 50.0})
        pipe = Pipeline([node])
        ctx = DSPContext(dt_us=dt, ns=ns, n_traces=nt)
        halo = node.time_halo_samples(ctx)
        win = extract_visible_window(
            data, dist_km, t0_ms=0.0, dt_us=dt,
            x_range=(0.0, 5.0), y_range=(0.0, ns * dt / 1000.0),
            time_halo=halo, max_rows=4000, max_cols=8000)
        assert win.row_stride > 1                       # decimation actually happened

        sub_ctx = DSPContext(dt_us=win.effective_dt_us,
                             ns=win.sub.shape[0], n_traces=win.sub.shape[1])
        out = pipe.process(win.sub, sub_ctx, input_token=win.token)
        half = out.shape[0] // 2
        early = float(np.median(np.abs(out[:half, 0])))
        late = float(np.median(np.abs(out[half:, 0])))
        # AGC equalised the 10:1 contrast to roughly 1:1 on the decimated preview.
        assert 0.4 < early / max(late, 1e-9) < 2.5


# ── 5. PHASE 3: migrated nodes wrap the core math exactly ───────────────────────

class TestNodeMigration:
    DT_US = 250

    def _data(self):
        return as_matrix(synthetic_trace(ns=400, noise_std=0.05), n_traces=80)

    def _ctx(self, delays=None):
        from sbp_studio.gui.dsp import DSPContext
        return DSPContext(dt_us=self.DT_US, ns=400, n_traces=80,
                          delays=delays,
                          min_delay=float(delays.min()) if delays is not None else 0.0)

    def test_bandpass_node_matches_core(self):
        from sbp_studio.gui.dsp import BandpassNode
        from sbp_studio.core import apply_bandpass
        d = self._data()
        out = BandpassNode({"flo": 600, "fhi": 5000}).apply(d, self._ctx())
        np.testing.assert_array_equal(out, apply_bandpass(d, 600, 5000, self.DT_US))

    def test_tvg_node_matches_core(self):
        from sbp_studio.gui.dsp import TVGNode
        from sbp_studio.core import apply_tvg
        d = self._data()
        out = TVGNode({"alpha": 12.0}).apply(d, self._ctx())
        np.testing.assert_array_equal(out, apply_tvg(d, 12.0, self.DT_US))

    def test_decon_node_matches_core(self):
        from sbp_studio.gui.dsp import PredictiveDeconNode
        from sbp_studio.core import apply_predictive_decon
        d = self._data()
        out = PredictiveDeconNode({"op_ms": 12, "gap_ms": 2, "white_pct": 1.0}).apply(d, self._ctx())
        np.testing.assert_array_equal(out, apply_predictive_decon(d, self.DT_US, 12, 2, 1.0))

    def test_preset_envelope_node_matches_core(self):
        from sbp_studio.gui.dsp import PresetNode
        from sbp_studio.core import apply_filter_preset
        d = self._data()
        out = PresetNode({"preset": "envelope"}).apply(d, self._ctx())
        np.testing.assert_array_equal(out, apply_filter_preset(d, "envelope", self.DT_US))

    def test_delay_alignment_is_static_not_a_node(self):
        """Delay alignment is a STATIC geometry correction — it must NOT be a
        movable pipeline node (the controller/export apply core directly)."""
        from sbp_studio.core import apply_delay_alignment
        from sbp_studio.gui.dsp import NODE_REGISTRY
        assert "align" not in {c.KEY for c in NODE_REGISTRY}
        # The core function still exists and grows the matrix correctly.
        delays = np.random.default_rng(1).integers(20, 60, 80).astype(np.float32)
        d = self._data()
        out = apply_delay_alignment(d, delays, float(delays.min()), self.DT_US, fill_value=0.0)
        assert out.shape[0] >= d.shape[0]
        assert out.shape[1] == d.shape[1]

    def test_full_pipeline_in_order_matches_standalone_chain(self):
        """A mixed node pipeline applied in order == the standalone core chain."""
        from sbp_studio.gui.dsp import Pipeline, BandpassNode, TVGNode, AGCNode
        from sbp_studio.core import apply_bandpass, apply_tvg, apply_agc
        d = self._data()
        ctx = self._ctx()
        pipe = Pipeline([BandpassNode({"flo": 500, "fhi": 6000}),
                         TVGNode({"alpha": 8.0}),
                         AGCNode({"win_ms": 25})])
        out = pipe.process(d, ctx, input_token="t")
        ref = apply_agc(apply_tvg(apply_bandpass(d, 500, 6000, self.DT_US),
                                  8.0, self.DT_US), 25, self.DT_US)
        np.testing.assert_array_equal(out, ref)

    def test_registry_has_filter_nodes_only(self):
        """Registry holds the reorderable DSP FILTERS only — NOT static geometry."""
        from sbp_studio.gui.dsp import NODE_REGISTRY
        keys = {c.KEY for c in NODE_REGISTRY}
        assert keys == {"swell", "fk", "water_mute", "demultiple", "decon",
                        "bandpass", "notch", "whiten", "preset", "tvg", "agc",
                        "log_compress"}
        assert "align" not in keys


# ── 6. PHASE 3.5: marine geophysics filters (water mute + swell) ────────────────

class TestMarineFilters:
    DT_US = 250

    def _seabed_trace(self, ns=500, pick=200):
        tr = np.zeros(ns, dtype=np.float32)
        tr[pick:pick + 8] = np.array([1.0, .7, -.5, .4, -.3, .2, -.1, .05], dtype=np.float32)
        return tr

    # ── Water mute ──
    def test_water_mute_zeros_above_seabed_keeps_below(self):
        from sbp_studio.core import apply_water_mute
        tr = self._seabed_trace()
        tr[300] = 0.3                              # a fainter deeper reflector
        data = np.repeat(tr[:, None], 30, axis=1).astype(np.float32)
        out = apply_water_mute(data, threshold_pct=30, margin_ms=2.0, dt_us=self.DT_US)
        # pick≈200 (first ≥30% of peak 1.0); margin 2 ms = 8 samples → mute up to ~192
        assert np.all(out[:185, :] == 0.0)         # water column muted
        assert np.any(out[200:208, :] != 0.0)      # seabed reflector preserved
        assert out.dtype == np.float32 and out is not data

    def test_water_mute_node_matches_core(self):
        from sbp_studio.gui.dsp import WaterMuteNode, DSPContext
        from sbp_studio.core import apply_water_mute
        data = np.repeat(self._seabed_trace()[:, None], 20, axis=1).astype(np.float32)
        ctx = DSPContext(dt_us=self.DT_US, ns=data.shape[0], n_traces=data.shape[1])
        out = WaterMuteNode({"threshold_pct": 25, "margin_ms": 4}).apply(data, ctx)
        np.testing.assert_array_equal(out, apply_water_mute(data, 25, 4, self.DT_US))

    def test_water_mute_all_zero_trace_is_safe(self):
        from sbp_studio.core import apply_water_mute
        data = np.zeros((256, 8), dtype=np.float32)
        out = apply_water_mute(data, 30, 5, self.DT_US)
        assert np.all(np.isfinite(out)) and out.shape == data.shape

    # ── Swell ──
    def _heaved(self, nt=200, seed=7, frac=0.25, amp=5):
        base = self._seabed_trace()
        rng = np.random.default_rng(seed)
        jit = np.zeros(nt, int)
        idx = rng.choice(nt, nt // 4, replace=False)
        jit[idx] = rng.integers(-amp, amp + 1, len(idx))
        data = np.stack([np.roll(base, jit[j]) for j in range(nt)], axis=1).astype(np.float32)
        return data

    def test_swell_removes_incoherent_heave(self):
        from sbp_studio.core import apply_swell_filter
        data = self._heaved()
        pk_b = np.argmax(np.abs(data), axis=0)
        out = apply_swell_filter(data, window_traces=31, max_shift_ms=2.0, dt_us=self.DT_US)
        pk_a = np.argmax(np.abs(out), axis=0)
        assert pk_a.std() < 0.5 * pk_b.std()       # seabed flattened
        assert out.shape == data.shape and out.dtype == np.float32

    def test_swell_preserves_coherent_topography(self):
        from sbp_studio.core import apply_swell_filter
        nt = 200
        base = self._seabed_trace()
        trend = (5 * np.sin(2 * np.pi * np.arange(nt) / 120)).round().astype(int)
        data = np.stack([np.roll(base, trend[j]) for j in range(nt)], axis=1).astype(np.float32)
        out = apply_swell_filter(data, window_traces=21, max_shift_ms=2.0, dt_us=self.DT_US)
        pk_a = np.argmax(np.abs(out), axis=0)
        # Coherent seabed topography must survive (not be flattened away).
        assert np.corrcoef(pk_a, 200 + trend)[0, 1] > 0.9

    def test_swell_direction_pulls_outlier_to_consensus(self):
        from sbp_studio.core import apply_swell_filter
        base = self._seabed_trace()
        data = np.repeat(base[:, None], 60, axis=1).astype(np.float32)
        data[:, 30] = np.roll(base, 5)             # one trace heaved +5
        out = apply_swell_filter(data, window_traces=21, max_shift_ms=2.0, dt_us=self.DT_US)
        assert abs(int(np.argmax(np.abs(out[:, 30]))) - 200) <= 1

    def test_swell_node_matches_core(self):
        from sbp_studio.gui.dsp import SwellFilterNode, DSPContext
        from sbp_studio.core import apply_swell_filter
        data = self._heaved(nt=80)
        ctx = DSPContext(dt_us=self.DT_US, ns=data.shape[0], n_traces=data.shape[1])
        out = SwellFilterNode({"window_traces": 21, "max_shift_ms": 3.0}).apply(data, ctx)
        np.testing.assert_array_equal(out, apply_swell_filter(data, 21, 3.0, self.DT_US))

    def test_water_mute_is_precrop_swell_is_not(self):
        """Water mute needs the full trace → flagged PRECROP (run pre-crop in the
        preview); swell is a windowed filter → not PRECROP."""
        from sbp_studio.gui.dsp import WaterMuteNode, SwellFilterNode, AGCNode
        assert WaterMuteNode.PRECROP is True
        assert SwellFilterNode.PRECROP is False
        assert AGCNode.PRECROP is False


# ── 7. PHASE 4: amplitude spectrum (FFT) ────────────────────────────────────────

class TestAmplitudeSpectrum:
    DT_US = 125   # fs = 8 kHz, Nyquist 4 kHz

    def _tones(self, ns=2048, nt=20, freqs=(1000.0, 3000.0)):
        t = np.arange(ns) * self.DT_US / 1e6
        sig = sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)
        return (np.repeat(sig[:, None], nt, axis=1)
                + np.random.default_rng(0).normal(0, 0.02, (ns, nt)).astype(np.float32))

    def test_freq_axis_and_length(self):
        from sbp_studio.core import compute_amplitude_spectrum
        data = self._tones()
        f, a = compute_amplitude_spectrum(data, self.DT_US)
        assert f.shape == a.shape
        assert len(f) == data.shape[0] // 2 + 1
        assert abs(f[-1] - (1e6 / self.DT_US) / 2) < 1.0    # Nyquist = 4000 Hz
        assert f[0] == 0.0

    def test_db_scale_normalised_to_zero_peak(self):
        from sbp_studio.core import compute_amplitude_spectrum
        f, db = compute_amplitude_spectrum(self._tones(), self.DT_US)
        assert abs(float(db.max())) < 1e-4          # peak normalised to 0 dB
        assert float(db.min()) <= 0.0               # everything else ≤ 0 dB

    def test_dc_bin_forced_to_floor(self):
        """The 0 Hz spike must be eradicated so it can't compress the plot."""
        from sbp_studio.core import compute_amplitude_spectrum
        # Strong DC offset on top of a tone — classic skew case.
        data = self._tones(freqs=(1500.0,)) + 50.0
        f, db = compute_amplitude_spectrum(data, self.DT_US)
        assert db[0] < -60.0                        # DC driven far below the peak
        # The real tone is the peak, not DC.
        assert abs(f[np.argmax(db)] - 1500) < 60

    def test_peaks_at_injected_tones(self):
        from sbp_studio.core import compute_amplitude_spectrum
        f, db = compute_amplitude_spectrum(self._tones(freqs=(1000.0, 3000.0)), self.DT_US)
        at = lambda hz: float(db[np.argmin(np.abs(f - hz))])
        # Tones sit near the 0 dB peak; the noise floor is tens of dB down.
        assert at(1000) > at(500) + 15 and at(3000) > at(3800) + 15

    def test_bandpass_attenuation_visible_in_spectrum(self):
        from sbp_studio.core import compute_amplitude_spectrum, apply_bandpass
        data = self._tones(freqs=(1000.0, 3000.0))
        bp = apply_bandpass(data, 500, 1500, self.DT_US)
        f, db = compute_amplitude_spectrum(bp, self.DT_US)
        at = lambda hz: float(db[np.argmin(np.abs(f - hz))])
        assert at(3000) < at(1000) - 30             # 3 kHz ≥30 dB down vs passband

    def test_1d_trace_accepted(self):
        from sbp_studio.core import compute_amplitude_spectrum
        t = np.arange(1024) * self.DT_US / 1e6
        tr = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
        f, db = compute_amplitude_spectrum(tr, self.DT_US)
        assert abs(f[np.argmax(db)] - 2000) < 50    # peak at 2 kHz

    def test_degenerate_input_safe(self):
        from sbp_studio.core import compute_amplitude_spectrum
        f, a = compute_amplitude_spectrum(np.zeros((2, 4), np.float32), self.DT_US)
        assert np.all(np.isfinite(a)) and len(f) == len(a)


# ── 8. PHASE 4.5: advanced Welch spectrum (core) + panel render ─────────────────

class TestAdvancedSpectrum:
    DT_US = 125

    def _data(self, ns=2048, nt=120, hz=2000.0):
        t = np.arange(ns) * self.DT_US / 1e6
        sig = np.sin(2 * np.pi * hz * t).astype(np.float32)
        return (np.repeat(sig[:, None], nt, axis=1)
                + np.random.default_rng(0).normal(0, 0.05, (ns, nt)).astype(np.float32))

    def test_welch_result_fields(self):
        from sbp_studio.core import compute_spectrum
        sp = compute_spectrum(self._data(), 1e6 / self.DT_US)
        # Percentile band ordering and spectrogram shape.
        assert np.all(sp.spec_p90_db >= sp.spec_p10_db - 1e-6)
        assert sp.spec_2d_db.shape[0] == len(sp.freqs)
        assert abs(sp.peak_hz - 2000) < 60           # peak at the injected tone
        assert sp.snr_db > 10                         # tone >> HF noise

    def test_panel_renders_result_on_demand(self):
        """The advanced panel stays empty until show_result() is called."""
        from PyQt6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.spectrum_view import (
            SpectrumView, SCOPE_VIEWBOX, SCOPE_FULL)
        from sbp_studio.core import compute_spectrum
        sv = SpectrumView()
        assert len(sv.c_mean.getData()[0] or []) == 0          # empty until generated
        sp = compute_spectrum(self._data(), 1e6 / self.DT_US)
        sv.show_result(sp, 1e6 / self.DT_US, np.linspace(0, 5, 120), ())
        assert len(sv.c_mean.getData()[1]) == len(sp.freqs)    # PSD drawn
        assert sv.img.image is not None                         # spectrogram drawn
        assert "kHz" in sv.stats.text()                         # stats drawn
        assert {sv.cb_scope.itemData(0), sv.cb_scope.itemData(1)} == {SCOPE_VIEWBOX, SCOPE_FULL}

    def test_band_energy_sums_to_100(self):
        from sbp_studio.core import compute_spectrum
        from sbp_studio.core.spectrum import TOPAS_BANDS
        sp = compute_spectrum(self._data(), 1e6 / self.DT_US)
        assert sp.band_labels and len(sp.band_labels) == len(sp.band_pcts)
        assert abs(sum(sp.band_pcts) - 100.0) < 0.5
        # The 2 kHz tone → the 2–4 kHz band carries the most energy.
        assert sp.band_labels[int(np.argmax(sp.band_pcts))] in ("1–2 kHz", "2–4 kHz")

    def test_panel_renders_full_advanced_layout(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.spectrum_view import SpectrumView
        from sbp_studio.core import compute_spectrum
        sv = SpectrumView()
        sp = compute_spectrum(self._data(), 1e6 / self.DT_US)
        sv.show_result(sp, 1e6 / self.DT_US, np.linspace(0, 5, 120), ())
        assert sv._bars is not None                       # band-energy bars
        assert len(sv._spectro_lines) >= 3                # BW-3dB ×2 + peak guide lines
        assert "NFFT" in sv.stats.text() and "SNR" in sv.stats.text()

    def test_band_bars_peak_distinct_and_theme_adaptive(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.theme import theme
        from sbp_studio.gui.components.spectrum_view import SpectrumView, MEAN_COLOR
        from sbp_studio.core import compute_spectrum
        sv = SpectrumView()
        sp = compute_spectrum(self._data(), 1e6 / self.DT_US)

        def colors():
            return [b.color().name().lower() for b in sv._bars.opts["brushes"]]

        theme.set_theme("dark")
        sv.show_result(sp, 1e6 / self.DT_US, np.linspace(0, 5, 120), ())
        pk = sv._band_peak_idx
        dark = colors()
        assert dark[pk] == MEAN_COLOR                      # peak = bright cyan
        dark_default = {c for i, c in enumerate(dark) if i != pk}
        assert len(dark_default) == 1 and MEAN_COLOR not in dark_default

        theme.set_theme("light")                           # _restyle re-tints
        light = colors()
        assert light[pk] == MEAN_COLOR                     # peak stays cyan
        light_default = {c for i, c in enumerate(light) if i != pk}
        assert dark_default != light_default               # default grey adapts
        theme.set_theme("dark")


# ── 9. PHASE 5: navigation map + bi-directional sync ────────────────────────────

class TestNavigationMap:
    def _qt(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def _track(self, n=300):
        x = np.concatenate([np.linspace(0, 10, n // 2), np.full(n - n // 2, 10.0)])
        y = np.concatenate([np.zeros(n // 2), np.linspace(0, 8, n - n // 2)])
        dist = np.linspace(0, 18, n)
        return x, y, dist

    def test_set_track_and_aspect_locked(self):
        self._qt()
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        assert mv.plot.getViewBox().state["aspectLocked"] == 1.0   # strict 1:1
        x, y, dist = self._track()
        mv.set_track(x, y)                                  # index-based: no dist
        assert len(mv.curve_full.getData()[0]) == len(x)
        assert mv.marker_start.getData()[0][0] == x[0]
        assert mv.marker_end.getData()[0][0] == x[-1]

    def test_visible_segment_is_index_subset(self):
        self._qt()
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        x, y, _ = self._track()
        mv.set_track(x, y)
        mv.set_visible_range(80, 160)                       # pure trace indices
        sx, sy = mv.curve_seg.getData()
        np.testing.assert_array_equal(sx, x[80:160])
        # SOL/EOL anchors are STATIC: they stay at the absolute line ends,
        # NOT at the moving segment's ends.
        assert mv.marker_start.getData()[0][0] == x[0]      # green = SOL (fixed)
        assert mv.marker_end.getData()[0][0] == x[-1]       # red   = EOL (fixed)

    def test_sync_robust_to_gps_plateaus(self):
        """The whole point: indices must reach the true end even when the
        distance array has huge frozen-GPS plateaus. Resolved by masking the
        REAL distance array (boolean mask), not searchsorted nor interpolation."""
        self._qt()
        from sbp_studio.gui.components.seismic_view import SeismicView
        n = 400
        # Distance with a long plateau in the middle AND a frozen end.
        dist = np.linspace(0, 10, n)
        dist[150:250] = dist[150]            # mid plateau (GPS dropout)
        dist[-40:] = dist[-40]              # frozen at the end
        mv_traces = []
        sv = SeismicView()
        sv.visible_traces_changed.connect(lambda a, b: mv_traces.append((a, b)))
        sv.set_distance_axis(dist)
        # Scroll to the far edge: ViewBox right edge == the profile's max distance.
        sv._emit_visible_traces(((dist[0], dist[-1]), (0.0, 100.0)))
        t0, t1 = mv_traces[-1]
        assert t0 == 0 and t1 == n          # reaches the TRUE end despite plateaus
        # A window covering the right 25% by DISTANCE: the last in-view trace is
        # the exact final element of the frozen-end plateau (n), proving the mask
        # captures the whole plateau where searchsorted would stop at its start.
        d0, d1 = float(dist[0]), float(dist[-1])
        sv._emit_visible_traces(((d0 + 0.75 * (d1 - d0), d1), (0.0, 100.0)))
        t0, t1 = mv_traces[-1]
        # Right-quarter distance window: start at the first trace ≥ 0.75·extent,
        # end exactly at the last trace (the frozen-end plateau is fully included).
        want_t0 = int(np.where(dist >= d0 + 0.75 * (d1 - d0))[0][0])
        assert t1 == n and t0 == want_t0

    def test_click_emits_nearest_trace_index(self):
        self._qt()
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        x, y, _ = self._track()
        mv.set_track(x, y)
        got = []
        mv.trace_clicked.connect(got.append)
        # Fake a click event whose scene pos maps near track vertex 200.
        class _Ev:
            def __init__(self, pt): self._p = pt
            def scenePos(self): return self._p
        from PyQt6.QtCore import QPointF
        target = mv.plot.getViewBox().mapViewToScene(QPointF(float(x[200]), float(y[200])))
        mv._on_click(_Ev(target))
        assert got and abs(got[0] - 200) <= 1

    def test_basemap_conditional_on_geographic_coords(self):
        self._qt()
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        assert mv._basemap.isVisible() is False          # nothing loaded yet
        # Geographic degrees (Antarctic) → basemap shown.
        mv.set_track(np.linspace(-70, -69, 50), np.linspace(-65, -64, 50))
        assert mv._basemap.isVisible() is True
        # Projected UTM metres → basemap hidden (out of ±180/±90).
        mv.set_track(np.linspace(4.5e5, 4.6e5, 50), np.linspace(7.1e6, 7.11e6, 50))
        assert mv._basemap.isVisible() is False
        mv.clear()
        assert mv._basemap.isVisible() is False

    def test_track_injected_as_managed_layer(self):
        self._qt()
        from PyQt6.QtCore import Qt
        from sbp_studio.gui.components.map_view import MapView, _ROLE_IS_TRACK
        mv = MapView()
        assert mv._track_item is None
        mv.set_track(np.linspace(-70, -69, 20), np.linspace(-65, -64, 20))
        assert mv._track_item is not None
        assert mv.layer_list.item(0).data(_ROLE_IS_TRACK) is True
        assert mv.layer_list.item(0) is mv._track_item
        # All four track graphics items are managed under the one row.
        items = mv.layer_list.item(0).data(Qt.ItemDataRole.UserRole)
        assert set(items) == {mv.curve_full, mv.curve_seg, mv.marker_start, mv.marker_end}
        # set_track again does NOT duplicate the row.
        mv.set_track(np.linspace(-70, -69, 10), np.linspace(-65, -64, 10))
        assert sum(1 for r in range(mv.layer_list.count())
                   if mv.layer_list.item(r).data(_ROLE_IS_TRACK)) == 1
        mv.clear()
        assert mv._track_item is None

    def test_track_reorder_vs_raster_and_remove_guard(self):
        self._qt()
        from PyQt6.QtCore import Qt
        from sbp_studio.gui.components.map_view import MapView, _ROLE_IS_TRACK
        from sbp_studio.core import RasterLayer
        mv = MapView()
        mv.add_layer(RasterLayer(name="r", image=np.zeros((6, 6), np.uint8),
                                 bbox=(-3.0, -2.9, 43.0, 43.1)))
        mv.set_track(np.linspace(-3, -2.9, 10), np.linspace(43, 43.1, 10))
        track = mv.layer_list.item(0).data(Qt.ItemDataRole.UserRole)
        raster = next(mv.layer_list.item(r).data(Qt.ItemDataRole.UserRole)[0]
                      for r in range(mv.layer_list.count())
                      if not mv.layer_list.item(r).data(_ROLE_IS_TRACK))
        assert min(i.zValue() for i in track) > raster.zValue()      # track on top by default
        # Drag track below the raster → z flips.
        mv.layer_list.insertItem(1, mv.layer_list.takeItem(0))
        mv._reorder_layers()
        assert min(i.zValue() for i in track) < raster.zValue()
        # Track is protected from removal.
        mv.layer_list.setCurrentItem(mv._track_item)
        mv._remove_selected_layer()
        assert mv._track_item is not None

    def test_opacity_slider_drives_selected_layer(self):
        self._qt()
        from PyQt6.QtCore import Qt
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        mv.set_track(np.linspace(-70, -69, 10), np.linspace(-65, -64, 10))
        mv.layer_list.setCurrentItem(mv._track_item)        # currentItemChanged → sync
        assert mv.opacity_slider.isEnabled() and mv.opacity_slider.value() == 100
        mv.opacity_slider.setValue(40)
        items = mv._track_item.data(Qt.ItemDataRole.UserRole)
        assert all(abs(i.opacity() - 0.40) < 1e-6 for i in items)
        # Re-selecting reflects the layer's stored opacity.
        mv._on_current_layer_changed(mv._track_item, None)
        assert mv.opacity_slider.value() == 40

    def test_basemap_is_below_track_and_click_transparent(self):
        self._qt()
        from PyQt6.QtCore import Qt
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        assert mv._basemap.zValue() < 0                  # under the trackline
        assert mv._basemap.acceptedMouseButtons() == Qt.MouseButton.NoButton

    def test_basemap_loads_local_geojson(self):
        self._qt()
        from sbp_studio.gui.components.map_view import MapView
        mv = MapView()
        # The bundled assets/coastlines_highres.geojson is present → real coast.
        assert mv._basemap_is_graticule is False
        assert mv._basemap.path().elementCount() > 0

    def test_basemap_graticule_fallback_when_asset_missing(self, monkeypatch, tmp_path):
        self._qt()
        from pathlib import Path
        from sbp_studio.gui.components.map_view import MapView
        missing = tmp_path / "nope.geojson"
        monkeypatch.setattr(MapView, "_asset_path", staticmethod(lambda: Path(missing)))
        mv = MapView()
        assert mv._basemap_is_graticule is True            # graticule fallback
        assert mv._basemap.path().elementCount() > 0        # grid lines drawn

    def test_geojson_parser_handles_polygon_and_linestring(self):
        self._qt()
        from sbp_studio.gui.components.map_view import MapView
        gj = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Polygon",
             "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
            {"type": "Feature", "geometry": {"type": "LineString",
             "coordinates": [[2, 2], [3, 3], [4, 2]]}},
        ]}
        path = MapView._geojson_to_path(gj)
        assert path.elementCount() > 0

    def test_seismic_center_on_distance(self):
        self._qt()
        from sbp_studio.gui.components.seismic_view import SeismicView
        sv = SeismicView()
        sv.plot.getViewBox().setXRange(0, 20, padding=0)
        sv.center_on_distance(12.0)
        (x0, x1), _ = sv.plot.getViewBox().viewRange()
        assert abs((x0 + x1) / 2 - 12.0) < 0.5            # recentred, width kept
        assert abs((x1 - x0) - 20.0) < 1.0


class TestCoordinateExtraction:
    """Core SEG-Y coordinate parsing — strict SIGNED extraction + standard
    scalar scaling (the Antarctic 'collapse to 0,0' guard). Core-only, no Qt."""

    def test_to_signed_reinterprets_unsigned_bit_patterns(self):
        from sbp_studio.core.io_segy import _to_signed
        # 16-bit: 65436 (unsigned) == -100 (signed >h).
        assert int(_to_signed(65436, 16)) == -100
        assert int(_to_signed(-100, 16)) == -100        # already signed → unchanged
        # 32-bit: 4294967196 (unsigned) == -100 (signed >i).
        assert int(_to_signed(4294967196, 32)) == -100
        assert int(_to_signed(-650000, 32)) == -650000

    def test_scalar_fac_standard_rule(self):
        from sbp_studio.core.io_segy import _scalar_fac
        assert _scalar_fac(-100) == 0.01      # negative → divide by |scalar|
        assert _scalar_fac(100) == 100.0      # positive → multiply
        assert _scalar_fac(0) == 1.0          # zero → no scaling

    def test_negative_antarctic_coords_not_collapsed(self, tmp_path):
        from tests.make_synthetic_segy import make_synthetic_segy
        from sbp_studio.core import load_profile
        p = make_synthetic_segy(
            str(tmp_path / "antarctic.sgy"), n_traces=30,
            scalar_coord=-100, coord_unit=3,
            base_lon=-64.5, base_lat=-70.3, lon_step=0.01, lat_step=-0.01)
        prof = load_profile(p, load_traces=False)
        assert prof.scalar_coord == -100                 # signed scalar preserved
        assert prof.lons[0] == pytest.approx(-64.5, abs=0.02)
        assert prof.lats[0] == pytest.approx(-70.3, abs=0.02)
        assert not np.allclose(prof.lons, 0.0)           # NOT collapsed to 0,0
        assert np.all(prof.lats < 0)                      # stays in the deep south


class TestHeaderInspector:
    """Phase 6 — core header exposure + the NumPy-backed trace-header model and
    the HeaderView selection sync."""

    def _qt(self):
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def test_core_exposes_text_and_trace_headers(self, tmp_path):
        from tests.make_synthetic_segy import make_synthetic_segy
        from sbp_studio.core import load_profile
        p = make_synthetic_segy(str(tmp_path / "h.sgy"), n_traces=40,
                                scalar_coord=-100, coord_unit=3,
                                base_lon=-64.5, base_lat=-70.3, lon_step=0.01)
        prof = load_profile(p, load_traces=False)          # header-only stub
        assert isinstance(prof.text_header, str) and prof.text_header
        assert len(prof.text_header.splitlines()) == 40    # 40 × 80 cards
        th = prof.trace_headers
        # Dynamic: ALL standard SEG-Y trace words exposed, by segyio field name.
        import segyio
        assert len(th) == len(segyio.tracefield.keys)      # complete header (91)
        for key in ("SourceX", "SourceY", "SourceGroupScalar", "DelayRecordingTime"):
            assert key in th and th[key].shape[0] == prof.n_traces
        # Ordered by byte offset (240-byte header order).
        expected_order = [n for n, _ in sorted(segyio.tracefield.keys.items(),
                                               key=lambda kv: kv[1])]
        assert list(th.keys()) == expected_order
        # RAW signed value (degrees × |scalar|), NOT the scaled lon.
        assert int(th["SourceX"][0]) == -6450

    def test_table_model_dims_data_and_headers(self):
        self._qt()
        from sbp_studio.gui.components.header_view import TraceHeaderModel
        from PyQt6.QtCore import Qt
        cols = {"A": np.array([10, 20, 30]), "B": np.array([-1, -2, -3])}
        m = TraceHeaderModel(cols)
        assert m.rowCount() == 3 and m.columnCount() == 2
        assert m.data(m.index(2, 0)) == "30"
        assert m.data(m.index(0, 1)) == "-1"
        assert m.headerData(1, Qt.Orientation.Horizontal) == "B"
        assert m.headerData(2, Qt.Orientation.Vertical) == "2"   # trace index

    def test_table_model_scales_to_50k_rows_lazily(self):
        self._qt()
        from sbp_studio.gui.components.header_view import TraceHeaderModel
        big = {"X": np.arange(50_000, dtype=np.int32)}
        m = TraceHeaderModel(big)
        assert m.rowCount() == 50_000
        assert m.data(m.index(49_999, 0)) == "49999"   # only this cell is touched

    def test_headerview_select_is_silent_click_emits(self):
        self._qt()
        from sbp_studio.gui.components.header_view import HeaderView

        class _Obj:
            text_header = "C 1 HELLO"
            trace_headers = {"X": np.arange(100), "Y": np.arange(100) * 2}

        hv = HeaderView()
        hv.set_source(_Obj())
        got = []
        hv.trace_selected.connect(got.append)
        hv.select_trace(42)                            # programmatic → no emit
        assert [i.row() for i in hv.table.selectionModel().selectedRows()] == [42]
        assert got == []
        hv._on_clicked(hv.model.index(7, 0))           # user click → emit
        assert got == [7]

    def test_seismic_click_emits_trace_via_searchsorted(self):
        self._qt()
        from PyQt6.QtCore import Qt, QPointF
        from sbp_studio.gui.components.seismic_view import SeismicView
        sv = SeismicView()
        dist = np.linspace(0.0, 10.0, 101)
        sv.set_distance_axis(dist)
        sv.plot.getViewBox().setXRange(0.0, 10.0, padding=0)
        got = []
        sv.trace_clicked.connect(got.append)

        class _Click:
            def __init__(self, sp): self._sp = sp
            def button(self): return Qt.MouseButton.LeftButton
            def scenePos(self): return self._sp

        scene_pt = sv.plot.getViewBox().mapViewToScene(QPointF(5.0, 0.0))
        sv._on_scene_click(_Click(scene_pt))
        assert got and abs(got[-1] - int(np.searchsorted(dist, 5.0))) <= 1

    def test_chain_concatenates_trace_headers(self, tmp_path):
        from tests.make_synthetic_segy import make_chain_pair
        from sbp_studio.core import load_profile
        from sbp_studio.core.model import ProfileChain
        p1, p2 = make_chain_pair(str(tmp_path))
        profs = [load_profile(p1), load_profile(p2)]
        ch = ProfileChain(profs)
        assert ch.text_header == profs[0].text_header
        assert ch.trace_headers["SourceX"].shape[0] == profs[0].n_traces + profs[1].n_traces


class TestTrackCleaning:
    """Core-level GPS track cleaning (median filter) + its effect on the
    distance axis. All maths run in core/io_segy — nothing here touches Qt."""

    def test_median_filter_rejects_gps_spikes(self):
        from sbp_studio.core import smooth_track
        n = 200
        lon = np.linspace(-70.0, -69.5, n)        # high-latitude track
        lat = np.linspace(-65.0, -64.7, n)
        lon_noisy, lat_noisy = lon.copy(), lat.copy()
        spikes = [20, 21, 80, 150]                 # sparse frozen/jumped fixes
        lon_noisy[spikes] += np.array([0.4, -0.5, 0.6, -0.45])
        lat_noisy[spikes] += np.array([-0.3, 0.5, -0.4, 0.5])
        slon, slat = smooth_track(lon_noisy, lat_noisy, kernel=11)
        # Spikes removed → cleaned track hugs the underlying smooth line.
        assert np.max(np.abs(slon - lon)) < 0.05
        assert np.max(np.abs(slat - lat)) < 0.05
        # Endpoints preserved — NOT dragged toward 0 by zero-padding (mode=nearest).
        assert abs(slon[0]  - lon[0])  < 1e-6 and abs(slon[-1] - lon[-1]) < 1e-6
        assert abs(slat[0]  - lat[0])  < 1e-6 and abs(slat[-1] - lat[-1]) < 1e-6

    def test_smoothing_tames_step_to_step_jitter(self):
        from sbp_studio.core import smooth_track
        rng = np.random.default_rng(0)
        n = 300
        lon = np.linspace(0.0, 1.0, n) + rng.normal(0, 0.02, n)
        lat = np.linspace(0.0, 0.5, n) + rng.normal(0, 0.02, n)
        slon, slat = smooth_track(lon, lat, kernel=11)
        assert np.std(np.diff(slon)) < np.std(np.diff(lon))
        assert np.std(np.diff(slat)) < np.std(np.diff(lat))

    def test_short_track_returned_unchanged(self):
        from sbp_studio.core import smooth_track
        lon, lat = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        slon, slat = smooth_track(lon, lat)
        np.testing.assert_array_equal(slon, lon)
        np.testing.assert_array_equal(slat, lat)

    def test_loaded_profile_has_clean_track_and_monotonic_distance(self, tmp_path):
        from tests.make_synthetic_segy import make_synthetic_segy
        from sbp_studio.core import load_profile
        p = make_synthetic_segy(
            str(tmp_path / "arc.sgy"), n_traces=80,
            scalar_coord=-10000, coord_unit=3,
            base_lon=-8.5, base_lat=43.2, arc_deg=60.0, arc_radius_deg=0.08)
        prof = load_profile(p, load_traces=False)
        # Cleaned display track exists and is the same length as the raw record.
        assert prof.track_lons is not None and prof.track_lats is not None
        assert prof.track_lons.shape == prof.lons.shape
        # dist_km (built from the cleaned track) is monotonically non-decreasing.
        assert np.all(np.diff(prof.dist_km) >= 0)
        # Raw recorded coordinates are PRESERVED (exports must see them untouched).
        assert prof.lons is not prof.track_lons

    def test_distance_axis_searchsorted_window(self):
        """The distance axis is monotonic, so the SeismicView resolves the
        visible window with searchsorted — including across a plateau."""
        from PyQt6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.seismic_view import SeismicView
        n = 300
        dist = np.linspace(0.0, 15.0, n)
        dist[-50:] = dist[-50]                      # frozen-GPS plateau at the end
        got = []
        sv = SeismicView()
        sv.visible_traces_changed.connect(lambda a, b: got.append((a, b)))
        sv.set_distance_axis(dist)
        sv._emit_visible_traces(((dist[0], dist[-1]), (0.0, 1.0)))
        assert got[-1] == (0, n)                    # far edge reaches the true end
        # A window up to a mid distance maps to searchsorted-right of that value.
        sv._emit_visible_traces(((0.0, 6.0), (0.0, 1.0)))
        t0, t1 = got[-1]
        assert t0 == 0 and t1 == int(np.searchsorted(dist, 6.0, side="right"))


class TestReprojection:
    """Core point-array reprojection (core/spatial.py). No Qt — pure pyproj."""

    def test_roundtrip_wgs84_utm(self):
        from sbp_studio.core import reproject_points
        lons = np.array([-3.0, -2.5, -2.0])
        lats = np.array([43.0, 43.2, 43.4])
        xs, ys = reproject_points(lons, lats, "EPSG:4326", "EPSG:32630")
        assert np.all(xs > 100_000) and np.all(ys > 4_000_000)   # UTM metres
        # back to geographic ≈ original
        blon, blat = reproject_points(xs, ys, "EPSG:32630", "EPSG:4326")
        np.testing.assert_allclose(blon, lons, atol=1e-6)
        np.testing.assert_allclose(blat, lats, atol=1e-6)

    def test_identity_is_passthrough(self):
        from sbp_studio.core import reproject_points
        lons = np.array([10.0, 11.0]); lats = np.array([50.0, 51.0])
        xs, ys = reproject_points(lons, lats, "EPSG:4326", "EPSG:4326")
        np.testing.assert_array_equal(xs, lons)
        np.testing.assert_array_equal(ys, lats)

    def test_invalid_crs_raises(self):
        from sbp_studio.core import reproject_points
        from sbp_studio.core.tasks import CRSError
        with pytest.raises(CRSError):
            reproject_points([0.0], [0.0], "EPSG:4326", "NOT-A-CRS")

    def test_to_geographic_passthrough_when_geographic(self):
        from sbp_studio.core import to_geographic
        lons = np.array([-8.0, -7.9]); lats = np.array([43.0, 43.1])
        gx, gy = to_geographic(lons, lats, "EPSG:4326")
        np.testing.assert_array_equal(gx, lons)
        np.testing.assert_array_equal(gy, lats)
        # None CRS → passthrough too.
        gx2, gy2 = to_geographic(lons, lats, None)
        np.testing.assert_array_equal(gx2, lons)

    def test_to_geographic_converts_projected(self):
        from sbp_studio.core import to_geographic, reproject_points
        # Start from a known geographic point, project to UTM metres…
        lon0, lat0 = -3.0, 43.0
        xs, ys = reproject_points([lon0], [lat0], "EPSG:4326", "EPSG:32630")
        # …then ask to_geographic to bring the UTM metres back to degrees.
        gx, gy = to_geographic(xs, ys, "EPSG:32630")
        assert -180 <= gx[0] <= 180 and -90 <= gy[0] <= 90
        np.testing.assert_allclose([gx[0], gy[0]], [lon0, lat0], atol=1e-6)

    def test_to_geographic_never_raises_on_bad_crs(self):
        from sbp_studio.core import to_geographic
        lons = np.array([5.0]); lats = np.array([5.0])
        gx, gy = to_geographic(lons, lats, "BOGUS")     # falls back to input
        np.testing.assert_array_equal(gx, lons)
        np.testing.assert_array_equal(gy, lats)

    def test_crs_catalog_structure(self):
        from sbp_studio.core import CRS_CATALOG, CRS_PRESETS
        cats = [c for c, _ in CRS_CATALOG]
        assert cats == ["Geographic", "Polar Stereographic",
                        "UTM North (WGS 84)", "UTM South (WGS 84)"]
        counts = {c: len(items) for c, items in CRS_CATALOG}
        assert counts["UTM North (WGS 84)"] == 60 and counts["UTM South (WGS 84)"] == 60
        codes = set(CRS_PRESETS.values())
        assert "EPSG:3031" in codes            # Antarctic Polar Stereographic
        assert "EPSG:32660" in codes and "EPSG:32701" in codes   # UTM 60N / 01S
        assert len(CRS_PRESETS) == 4 + 4 + 60 + 60

    def test_reproject_one_honors_out_path(self, tmp_path):
        import os
        from tests.make_synthetic_segy import make_synthetic_segy
        from sbp_studio.core import load_profile, reproject_one
        p = make_synthetic_segy(str(tmp_path / "in.sgy"), n_traces=10,
                                base_lon=-3.0, base_lat=43.0)
        prof = load_profile(p, load_traces=False)
        custom = str(tmp_path / "explicit_dest.sgy")
        out = reproject_one(prof, "EPSG:4326", "EPSG:32630", 2, out_path=custom)
        assert out == custom and os.path.exists(custom)
        # Default path still works when out_path is omitted (backward compat).
        out2 = reproject_one(prof, "EPSG:4326", "EPSG:32630", 2)
        assert out2.endswith("_REPROY.sgy")


class TestCliConsole:
    """Integrated CLI console — command parsing/dispatch (GUI, no heavy deps)."""

    def _console(self):
        from PyQt6.QtWidgets import QApplication, QWidget
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.cli_console import CliConsole
        return CliConsole(QWidget())

    def _send(self, c, line):
        c.input.setText(line)
        c._run()
        return c.output.toPlainText()

    def test_echo_and_unknown(self):
        c = self._console()
        out = self._send(c, "echo hello world")
        assert "hello world" in out
        out = self._send(c, "definitely_not_a_cmd")
        assert "Unknown command" in out

    def test_clear_and_help(self):
        c = self._console()
        self._send(c, "echo x")
        self._send(c, "clear")
        assert c.output.toPlainText() == ""
        out = self._send(c, "help")
        assert "echo" in out and "theme" in out and "profiles" in out

    def test_register_external_command(self):
        c = self._console()
        hits = []
        c.register("ping", lambda args: hits.append(args), "test")
        self._send(c, "ping a b")
        assert hits == [["a", "b"]]

    def test_history_recall(self):
        c = self._console()
        self._send(c, "echo one")
        self._send(c, "echo two")
        c.recall_history(-1)
        assert c.input.text() == "echo two"
        c.recall_history(-1)
        assert c.input.text() == "echo one"

    def test_command_error_does_not_crash(self):
        c = self._console()
        c.register("boom", lambda args: (_ for _ in ()).throw(RuntimeError("kaboom")))
        out = self._send(c, "boom")
        assert "Error: kaboom" in out          # caught and printed, UI survives

    def test_tokenize_keeps_windows_paths_and_strips_quotes(self):
        from sbp_studio.gui.components.cli_console import CliConsole
        toks = CliConsole._tokenize(
            r'export-image .\examples\L01.sgy --fix-color "#cc4444" --preset envelope')
        assert toks == ["export-image", r".\examples\L01.sgy",
                        "--fix-color", "#cc4444", "--preset", "envelope"]

    def test_core_commands_are_wired(self):
        c = self._console()
        for name in ("info", "export-image", "reproject", "spectrum",
                     "navline", "fix", "accel", "join-chain"):
            assert name in c._commands

    def test_core_command_runs_through_console(self, tmp_path):
        from tests.make_synthetic_segy import make_synthetic_segy
        c = self._console()                    # parent is a plain QWidget → inline path
        p = make_synthetic_segy(str(tmp_path / "i.sgy"), n_traces=12,
                                base_lon=-3.0, base_lat=43.0)
        out = self._send(c, f"info {p}")
        assert "info finished" in out and c._busy is False   # streamed + reset

    def test_argparse_error_is_shown_not_raised(self):
        c = self._console()
        out = self._send(c, "reproject")       # missing required --src/--dst
        # argparse usage/error captured into the console, no crash.
        assert ("usage" in out.lower() or "error" in out.lower())
        assert c._busy is False


class TestCliPaths:
    """Phase 9.2 — portable relative-path resolution + pwd/cd."""

    def _console(self):
        from PyQt6.QtWidgets import QApplication, QWidget
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.cli_console import CliConsole
        return CliConsole(QWidget())

    def _send(self, c, line):
        c.input.setText(line); c._run()
        return c.output.toPlainText()

    def test_app_root_frozen_uses_executable_dir(self, monkeypatch):
        import os
        import sys
        from sbp_studio.gui.components import cli_console as cc
        # Frozen → folder containing the .exe (next to it), NOT sys._MEIPASS.
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable",
                            os.path.join("C:" + os.sep, "app", "SBPStudio.exe"),
                            raising=False)
        assert cc._app_root() == os.path.join("C:" + os.sep, "app")

    def test_app_root_source_is_project_root(self, monkeypatch):
        import os
        import sys
        from sbp_studio.gui.components import cli_console as cc
        monkeypatch.delattr(sys, "frozen", raising=False)
        root = cc._app_root()
        # The project root actually contains the package AND the examples dir,
        # so user paths like .\examples\… resolve correctly.
        assert os.path.isdir(os.path.join(root, "sbp_studio"))
        assert os.path.isdir(os.path.join(root, "examples"))

    def test_path_heuristic_skips_non_paths(self):
        c = self._console()
        for path_like in (r".\examples\x.sgy", "examples/out", "L01.seg",
                          r"data\a.tif"):
            assert c._looks_like_path(path_like) is True
        for not_path in ("--cmap", "Greys", "epsg:4326", "#cc4444", "0.0", "5",
                         "envelope", r"C:\abs\x.sgy"):
            assert c._looks_like_path(not_path) is False

    def test_relative_resolves_against_cwd_only_for_paths(self, tmp_path):
        import os
        c = self._console()
        c._cwd = str(tmp_path)
        r = c._resolve_path(r".\data\f.sgy")
        assert os.path.isabs(r) and r.startswith(str(tmp_path))
        assert c._resolve_path("epsg:4326") == "epsg:4326"     # untouched
        assert c._resolve_path("--out") == "--out"
        assert c._resolve_path(r"C:\abs\x.sgy") == r"C:\abs\x.sgy"   # absolute kept

    def test_pwd_and_cd(self, tmp_path):
        c = self._console()
        sub = tmp_path / "sub"; sub.mkdir()
        c._cwd = str(tmp_path)
        assert self._send(c, "pwd").splitlines()[-1] == str(tmp_path)
        self._send(c, "cd sub")
        assert c._cwd == str(sub)
        self._send(c, "cd ..")
        assert c._cwd == str(tmp_path)
        out = self._send(c, "cd definitely_missing_dir")
        assert "not a directory" in out and c._cwd == str(tmp_path)


class TestCliGuide:
    def test_guide_dialog_has_spanish_content_and_example(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.cli_console import CliGuideDialog
        dlg = CliGuideDialog()
        text = dlg.browser.toPlainText()
        assert "Guía de Comandos" in text
        assert "export-image" in text and "--fix-bbox-alpha" in text   # exact example
        assert "python -m sbp_studio.cli.main" in text                 # explains no prefix

    def test_guide_has_practical_examples_with_real_file(self):
        from PyQt6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.cli_console import CliGuideDialog, _GUIDE_FILE
        text = CliGuideDialog().browser.toPlainText()
        assert "Ejemplos Prácticos" in text
        assert _GUIDE_FILE in text                                     # real ANT26 file
        assert f"info {_GUIDE_FILE}" in text                           # quick inspect
        assert "--format shp" in text                                  # navline → shp
        assert "--src epsg:4326 --dst epsg:32630" in text              # reproject
        # The export-image example is ONE copy-pasteable line (no shell continuation).
        big = [ln for ln in text.splitlines() if ln.startswith("export-image")][0]
        assert "--out" in big and "`" not in big and not big.endswith("\\")
        # Examples use PORTABLE relative paths (no absolute drive letter).
        assert ".\\examples\\_real_in" in text and "Z:\\workspace" not in text
        # Explains app-root resolution + pwd/cd.
        assert "carpeta de la aplicación" in text
        assert "pwd" in text and "cd " in text


class TestGisReaders:
    """Core GIS layer readers (core/gis_io.py) — vector + raster → WGS84."""

    def _write_geotiff(self, path, epsg, x0, y0, px, h=8, w=12):
        import tifffile
        arr = (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8)
        model = 1 if epsg >= 32600 else 1
        key = 3072 if epsg >= 32600 else 2048      # Projected vs Geographic key
        gk = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, key, 0, 1, epsg]
        et = [(33550, "d", 3, (px, px, 0.0), True),
              (33922, "d", 6, (0, 0, 0, x0, y0, 0), True),
              (34735, "H", len(gk), tuple(gk), True)]
        tifffile.imwrite(path, arr, extratags=et)
        return arr

    def test_read_vector_reprojects_to_wgs84(self, tmp_path):
        import geopandas as gpd
        from shapely.geometry import LineString
        from sbp_studio.core import read_gis_layer, VectorLayer
        import pyproj
        tf = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
        xs, ys = tf.transform([-3.0, -2.9], [43.0, 43.1])
        p = str(tmp_path / "ln.shp")
        gpd.GeoDataFrame({"id": [1]}, geometry=[LineString(zip(xs, ys))],
                         crs="EPSG:32630").to_file(p)
        layer = read_gis_layer(p)
        assert isinstance(layer, VectorLayer) and layer.geom_type == "line"
        assert layer.src_crs == "EPSG:32630"
        np.testing.assert_allclose(layer.paths[0][0], [-3.0, 43.0], atol=1e-6)

    def test_read_geotiff_bbox_in_wgs84(self, tmp_path):
        from sbp_studio.core import read_gis_layer, RasterLayer
        p = str(tmp_path / "r.tif")
        arr = self._write_geotiff(p, 32630, 5e5, 4.8e6, 100.0, h=8, w=12)
        layer = read_gis_layer(p)
        assert isinstance(layer, RasterLayer)
        assert layer.image.shape == arr.shape and layer.src_crs == "EPSG:32630"
        lon0, lon1, lat0, lat1 = layer.bbox
        assert -180 <= lon0 < lon1 <= 180 and -90 <= lat0 < lat1 <= 90
        assert abs(lon0 - (-3.0)) < 0.05 and abs(lat1 - 43.35) < 0.05

    def test_geographic_geotiff_passthrough(self, tmp_path):
        from sbp_studio.core import read_gis_layer
        p = str(tmp_path / "geo.tif")
        # WGS84 raster: origin lon=-8, lat=43, 0.01° pixels.
        self._write_geotiff(p, 4326, -8.0, 43.0, 0.01, h=10, w=10)
        layer = read_gis_layer(p)
        lon0, lon1, lat0, lat1 = layer.bbox
        assert abs(lon0 - (-8.0)) < 1e-6 and abs(lat1 - 43.0) < 1e-6
        assert abs(lon1 - (-7.9)) < 1e-6                  # -8 + 10*0.01

    def test_plain_tiff_without_georef_raises(self, tmp_path):
        import tifffile
        from sbp_studio.core import read_geotiff
        p = str(tmp_path / "plain.tif")
        tifffile.imwrite(p, (np.zeros((4, 4))).astype(np.uint8))
        with pytest.raises(ValueError):
            read_geotiff(p)

    def test_unsupported_extension_raises(self):
        from sbp_studio.core import read_gis_layer
        with pytest.raises(ValueError):
            read_gis_layer("foo.xyz")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
