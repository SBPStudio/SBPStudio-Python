"""
test_processing.py — Characterisation + regression tests for processing pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbp_studio.core import (
    load_profile, process_profile_data, apply_filter_preset,
    apply_predictive_decon, time_window,
)


_BLANK_PARAMS = dict(
    decon=False, decon_op=10.0, decon_gap=1.0, decon_wn=0.1,
    filt=False,  flo=100.0, fhi=5000.0,
    preset="── Sin filtro preestablecido ──",
    tvg=False, tvg_alpha=0.0,
    agc=False,  agc_win=50.0,
    align=False,
    clip=99.0, cmap="Viridis", inv_cmap=False, fix=False, fix_iv=5,
)


class TestProcessData:
    def test_none_path_returns_copy(self, simple_segy):
        """process_data with all-off params should return a copy of .data."""
        sd   = load_profile(simple_segy)
        out  = process_profile_data(sd, _BLANK_PARAMS)
        assert out is not sd.data
        np.testing.assert_array_equal(out, sd.data)

    def test_output_shape_matches_input(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = process_profile_data(sd, _BLANK_PARAMS)
        assert out.shape == sd.data.shape

    def test_output_dtype_float32(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = process_profile_data(sd, _BLANK_PARAMS)
        assert out.dtype == np.float32

    def test_bandpass_filter(self, simple_segy):
        sd     = load_profile(simple_segy)
        params = {**_BLANK_PARAMS, "filt": True, "flo": 200.0, "fhi": 3000.0}
        out    = process_profile_data(sd, params)
        assert out.shape == sd.data.shape
        assert out.dtype == np.float32

    def test_tvg_amplifies(self, simple_segy):
        sd     = load_profile(simple_segy)
        params = {**_BLANK_PARAMS, "tvg": True, "tvg_alpha": 1.0}
        out    = process_profile_data(sd, params)
        # Late samples should be amplified vs input
        assert float(np.max(np.abs(out[-10:, :]))) > float(np.max(np.abs(sd.data[-10:, :])))

    def test_agc_normalises(self, simple_segy):
        sd     = load_profile(simple_segy)
        params = {**_BLANK_PARAMS, "agc": True, "agc_win": 10.0}
        out    = process_profile_data(sd, params)
        assert out.shape == sd.data.shape

    def test_align_extends_ns(self, delay_segy):
        """Delay alignment should produce more samples than the original."""
        sd     = load_profile(delay_segy)
        params = {**_BLANK_PARAMS, "align": True}
        out    = process_profile_data(sd, params)
        assert out.shape[0] > sd.ns
        assert out.shape[1] == sd.n_traces

    def test_input_not_mutated(self, simple_segy):
        sd      = load_profile(simple_segy)
        orig    = sd.data.copy()
        params  = {**_BLANK_PARAMS, "filt": True, "flo": 100.0, "fhi": 4000.0}
        process_profile_data(sd, params)
        np.testing.assert_array_equal(sd.data, orig)


class TestFilterPreset:
    @pytest.mark.parametrize("key", [
        "envelope", "inst_phase", "cos_phase",
        "similarity", "sobel_v", "laplacian", "highboost",
        "median5", "wiener7", "gauss1",
        "topas_narrow", "topas_wide", "topas_hires",
        "derivative", "integral",
    ])
    def test_shape_preserved(self, simple_segy, key):
        sd  = load_profile(simple_segy)
        out = apply_filter_preset(sd.data, key, sd.dt_us)
        assert out.shape == sd.data.shape
        assert out.dtype == np.float32

    def test_none_returns_copy(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = apply_filter_preset(sd.data, "none", sd.dt_us)
        assert out is not sd.data
        np.testing.assert_array_equal(out, sd.data)

    def test_none_empty_string(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = apply_filter_preset(sd.data, "", sd.dt_us)
        np.testing.assert_array_equal(out, sd.data)


class TestFilterPresetFastFlag:
    """Phase-2 engine audit: ``fast=True`` (live-preview only — see
    PresetNode._apply passing ``ctx.preview``) skips a float64 upcast several
    branches used unconditionally before, trading it for the memory/bandwidth
    saving of one fewer full-size float64 temporary per refresh.

    IMPORTANT correction made while writing this test: dtype PRESERVATION
    (scipy.ndimage keeps float32 in -> float32 out) is NOT the same claim as
    BIT-EXACT equivalence to the float64-then-downcast path — computing the
    same stencil arithmetic in float32 throughout rounds differently at each
    intermediate step than computing in float64 and truncating at the end.
    Measured directly: ``gauss1``/``derivative`` ARE exact (0 diff — their
    arithmetic happens not to accumulate rounding here); ``sobel_v``/
    ``laplacian``/``highboost``/``integral`` have a tiny (<1e-3 relative,
    confirmed below) rounding difference; ``wiener7`` is the one branch where
    scipy genuinely computes differently at float32 (still <1e-3 relative).
    Every one of these is utterly invisible on a percentile-clipped seismic
    display and NEVER reaches export/CLI (which never pass ``fast=True``)."""

    EXACT_KEYS = ["gauss1", "derivative"]
    TOLERANT_KEYS = ["sobel_v", "laplacian", "highboost", "integral", "wiener7"]

    def _data(self, ns=300, nt=20, seed=4):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((ns, nt)) * 3.0).astype(np.float32)

    @pytest.mark.parametrize("key", EXACT_KEYS)
    def test_fast_is_bit_exact_for_these_branches(self, key):
        data = self._data()
        out_default = apply_filter_preset(data, key, dt_us=200, fast=False)
        out_fast = apply_filter_preset(data, key, dt_us=200, fast=True)
        np.testing.assert_array_equal(out_default, out_fast)

    @pytest.mark.parametrize("key", TOLERANT_KEYS)
    def test_fast_stays_within_a_tiny_bounded_tolerance(self, key):
        """The accepted precision/speed tradeoff — bounded well under what
        could ever be geologically meaningful on a clipped display raster."""
        data = self._data()
        out_default = apply_filter_preset(data, key, dt_us=200, fast=False)
        out_fast = apply_filter_preset(data, key, dt_us=200, fast=True)
        np.testing.assert_allclose(out_default, out_fast, rtol=1e-3, atol=1e-4)

    def test_fast_defaults_to_false(self):
        """Omitting ``fast`` must behave exactly like ``fast=False`` — export/
        CLI call sites never pass it, so they must be untouched by this flag."""
        data = self._data()
        for key in self.EXACT_KEYS + self.TOLERANT_KEYS:
            out_omitted = apply_filter_preset(data, key, dt_us=200)
            out_explicit = apply_filter_preset(data, key, dt_us=200, fast=False)
            np.testing.assert_array_equal(out_omitted, out_explicit)


class TestDecon:
    def test_shape_preserved(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = apply_predictive_decon(sd.data, sd.dt_us, 10.0, 1.0, 0.1)
        assert out.shape == sd.data.shape

    def test_output_finite(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = apply_predictive_decon(sd.data, sd.dt_us, 10.0, 1.0, 0.1)
        assert np.all(np.isfinite(out))


def _decon_reference_per_trace_loop(data, dt_us, op_len_ms, gap_ms,
                                    white_noise_pct):
    """Independent reimplementation of the ORIGINAL per-trace algorithm
    (per-trace fft/ifft + solve_toeplitz + lfilter, full ``2*ns-1`` FFT size)
    — deliberately NOT importing anything from processing.py's current
    _ref_apply_predictive_decon, so this test verifies the Phase-2 engine
    refactor (batched rfft/irfft autocorrelation, smaller n_fft, GIL-
    releasing loop body) reproduces the PRE-refactor algorithm's output,
    independent of whatever processing.py contains now or in the future."""
    import scipy.linalg
    from scipy import signal as sp_signal
    ns, nt = data.shape
    dt_ms = dt_us / 1000.0
    nl = max(2, int(op_len_ms / dt_ms))
    gap = max(1, int(gap_ms / dt_ms))
    mu = white_noise_pct / 100.0
    max_lag = nl + gap
    n_fft = 2 ** int(np.ceil(np.log2(2 * ns - 1)))
    out = np.zeros_like(data)
    for i in range(nt):
        tr = data[:, i]
        X = np.fft.fft(tr, n_fft)
        r = np.fft.ifft(X * np.conj(X)).real[:max_lag]
        if r[0] == 0:
            out[:, i] = tr
            continue
        r[0] *= (1.0 + mu)
        try:
            a = scipy.linalg.solve_toeplitz(r[0:nl], r[gap:max_lag])
        except scipy.linalg.LinAlgError:
            out[:, i] = tr
            continue
        f = np.zeros(max_lag)
        f[0] = 1.0
        f[gap:] = -a
        out[:, i] = sp_signal.lfilter(f, [1.0], tr)
    return out


class TestDeconEngineRefactorEquivalence:
    """Phase-2 engine audit: the predictive-decon Python per-trace loop was
    rewritten to batch the autocorrelation (one rfft/irfft call across all
    traces, smaller n_fft) instead of an fft/ifft pair inside the loop — see
    _ref_apply_predictive_decon's docstring for the GIL/allocation rationale.
    This must be a pure performance refactor: the OUTPUT is verified against
    an independent reimplementation of the original algorithm, not just
    'shape preserved' / 'finite'."""

    def _synthetic(self, ns=600, nt=12, seed=3):
        rng = np.random.default_rng(seed)
        t = np.arange(ns)
        # A few traces with a clear reflectivity-like spike train (decon's
        # actual use case) plus broadband noise — not pure noise, so the
        # Toeplitz solve exercises a realistic, well-conditioned system.
        data = (rng.standard_normal((ns, nt)) * 0.05).astype(np.float32)
        for i in range(nt):
            spikes = rng.choice(ns, size=8, replace=False)
            data[spikes, i] += rng.normal(1.0, 0.3, size=8)
        return data.astype(np.float32)

    def test_matches_independent_reference_implementation(self):
        from sbp_studio.core import apply_predictive_decon
        data = self._synthetic()
        out_new = apply_predictive_decon(data, dt_us=250, op_len_ms=8.0,
                                         gap_ms=1.0, white_noise_pct=1.0)
        out_ref = _decon_reference_per_trace_loop(
            data, dt_us=250, op_len_ms=8.0, gap_ms=1.0, white_noise_pct=1.0)
        np.testing.assert_allclose(out_new, out_ref, atol=1e-3, rtol=1e-3)

    def test_dead_channel_passes_through_unchanged(self):
        """An all-zero trace must short-circuit to a passthrough, exactly
        like the original's ``if r[0] == 0: out[:, i] = tr``."""
        from sbp_studio.core import apply_predictive_decon
        data = self._synthetic()
        data[:, 3] = 0.0
        out = apply_predictive_decon(data, dt_us=250, op_len_ms=8.0,
                                     gap_ms=1.0, white_noise_pct=1.0)
        np.testing.assert_array_equal(out[:, 3], data[:, 3])

    def test_matches_reference_across_a_range_of_operator_lengths(self):
        """Sweep op/gap lengths (changes nl/gap/max_lag/n_fft together) to
        catch an off-by-one in the smaller n_fft sizing specifically."""
        from sbp_studio.core import apply_predictive_decon
        data = self._synthetic(ns=400, nt=6)
        for op_ms, gap_ms in [(2.0, 0.5), (15.0, 3.0), (40.0, 8.0)]:
            out_new = apply_predictive_decon(data, dt_us=200, op_len_ms=op_ms,
                                             gap_ms=gap_ms, white_noise_pct=2.0)
            out_ref = _decon_reference_per_trace_loop(
                data, dt_us=200, op_len_ms=op_ms, gap_ms=gap_ms,
                white_noise_pct=2.0)
            np.testing.assert_allclose(out_new, out_ref, atol=1e-3, rtol=1e-3)

    def test_matches_reference_on_a_wide_block_above_parallel_threshold(self):
        """_parallel_apply splits into a ThreadPoolExecutor above 64 traces /
        4MB — verify the batched-per-chunk autocorrelation still matches the
        reference across that boundary (chunking must not change results)."""
        from sbp_studio.core import apply_predictive_decon
        data = self._synthetic(ns=14000, nt=80)   # > 64 traces, > 4MB
        assert data.nbytes >= 4 * 1024 * 1024
        out_new = apply_predictive_decon(data, dt_us=200, op_len_ms=10.0,
                                         gap_ms=2.0, white_noise_pct=1.0)
        out_ref = _decon_reference_per_trace_loop(
            data, dt_us=200, op_len_ms=10.0, gap_ms=2.0, white_noise_pct=1.0)
        np.testing.assert_allclose(out_new, out_ref, atol=1e-3, rtol=1e-3)


class TestTimeWindow:
    def test_no_align(self, simple_segy):
        sd = load_profile(simple_segy)
        i0, i1, t0, t1 = time_window(sd, sd.ns, align=False)
        assert i0 == 0
        assert i1 == sd.ns
        assert t0 == sd.delay_ms
        assert t1 == pytest.approx(sd.delay_ms + sd.dur_ms)

    def test_align_uses_min_delay(self, delay_segy):
        sd = load_profile(delay_segy)
        _  , __, t0, ___ = time_window(sd, sd.ns, align=True)
        assert t0 == sd.min_delay
