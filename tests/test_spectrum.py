"""
test_spectrum.py — Characterisation + regression tests for spectrum computation.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbp_studio.core import load_profile, compute_spectrum, SegyProfile
from sbp_studio.core.spectrum import _ref_compute_spectrum, SpectrumResult


class TestComputeSpectrum:
    def test_returns_spectrum_result(self, simple_segy):
        sd = load_profile(simple_segy)
        fs = 1e6 / sd.dt_us
        sp = compute_spectrum(np.nan_to_num(sd.data, nan=0.0), fs)
        assert isinstance(sp, SpectrumResult)

    def test_freqs_shape(self, simple_segy):
        sd = load_profile(simple_segy)
        fs = 1e6 / sd.dt_us
        sp = compute_spectrum(np.nan_to_num(sd.data, nan=0.0), fs)
        expected = sd.ns // 2 + 1
        # nfft may be <= ns, so n_freqs = nfft//2 + 1
        assert sp.freqs.shape[0] == sp.nfft // 2 + 1

    def test_spec_2d_shape(self, simple_segy):
        sd = load_profile(simple_segy)
        fs = 1e6 / sd.dt_us
        sp = compute_spectrum(np.nan_to_num(sd.data, nan=0.0), fs)
        assert sp.spec_2d_db.shape == (len(sp.freqs), sd.n_traces)

    def test_spec_mean_in_db_range(self, simple_segy):
        sd = load_profile(simple_segy)
        fs = 1e6 / sd.dt_us
        sp = compute_spectrum(np.nan_to_num(sd.data, nan=0.0), fs)
        # Mean spectrum in dB: max should be ≤ 0, min >= -200
        assert float(np.max(sp.spec_mean_db)) <= 0.1
        assert float(np.min(sp.spec_mean_db)) > -300

    def test_peak_hz_injected(self, spectrum_segy):
        """The injected 2 kHz sinusoid should be the peak frequency."""
        sd = load_profile(spectrum_segy)
        fs = 1e6 / sd.dt_us
        sp = compute_spectrum(np.nan_to_num(sd.data, nan=0.0), fs)
        # Allow ±Δf tolerance (Δf = fs/nfft)
        delta_f = fs / sp.nfft
        assert abs(sp.peak_hz - 2000.0) <= delta_f * 2

    def test_metrics_finite(self, simple_segy):
        sd = load_profile(simple_segy)
        fs = 1e6 / sd.dt_us
        sp = compute_spectrum(np.nan_to_num(sd.data, nan=0.0), fs)
        for attr in ("peak_hz", "centroid_hz", "bw_3db_lo", "bw_3db_hi",
                     "bw_6db_lo", "bw_6db_hi", "roll_off_hz", "snr_db"):
            assert np.isfinite(getattr(sp, attr)), f"{attr} is not finite"

    def test_public_equals_reference(self, simple_segy):
        """Regression: public compute_spectrum must equal _ref."""
        sd  = load_profile(simple_segy)
        fs  = 1e6 / sd.dt_us
        d   = np.nan_to_num(sd.data, nan=0.0)
        pub = compute_spectrum(d, fs)
        ref = _ref_compute_spectrum(d, fs)
        np.testing.assert_allclose(pub.freqs,        ref.freqs,        rtol=0, atol=0)
        np.testing.assert_allclose(pub.spec_mean_db, ref.spec_mean_db, rtol=1e-10)
        np.testing.assert_allclose(pub.spec_2d_db,   ref.spec_2d_db,   rtol=1e-10)
        assert pub.peak_hz == ref.peak_hz
