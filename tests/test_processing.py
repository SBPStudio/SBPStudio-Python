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


class TestDecon:
    def test_shape_preserved(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = apply_predictive_decon(sd.data, sd.dt_us, 10.0, 1.0, 0.1)
        assert out.shape == sd.data.shape

    def test_output_finite(self, simple_segy):
        sd  = load_profile(simple_segy)
        out = apply_predictive_decon(sd.data, sd.dt_us, 10.0, 1.0, 0.1)
        assert np.all(np.isfinite(out))


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
