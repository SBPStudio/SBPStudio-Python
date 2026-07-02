"""
test_metadata.py — Characterisation tests for load_metadata and load_profile.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbp_studio.core import load_metadata, load_profile, SegyProfile, SegyMetadata


class TestLoadMetadata:
    def test_returns_metadata_type(self, simple_segy):
        md = load_metadata(simple_segy)
        assert isinstance(md, SegyMetadata)

    def test_no_trace_data_loaded(self, simple_segy):
        md = load_metadata(simple_segy)
        assert md.error is None
        # load_metadata must NOT read trace data
        assert md.clip_p99 is None

    def test_field_counts(self, simple_segy):
        md = load_metadata(simple_segy)
        assert md.n_traces == 50
        assert md.ns == 256
        assert md.dt_us == 250

    def test_duration(self, simple_segy):
        md = load_metadata(simple_segy)
        assert abs(md.dur_ms - 256 * 250 / 1000.0) < 1e-6

    def test_error_on_missing_file(self, tmp_path):
        md = load_metadata(str(tmp_path / "nonexistent.sgy"))
        assert md.error is not None

    def test_to_dict_is_json_serialisable(self, simple_segy):
        import json
        md = load_metadata(simple_segy)
        d  = md.to_dict()
        # Should not raise
        json.dumps(d, default=str)

    def test_arcsec_conversion(self, arcsec_segy):
        md = load_metadata(arcsec_segy)
        assert md.error is None
        # After arc-sec / 3600 division, should be in degree range
        assert md.lons is not None
        assert np.all(np.abs(md.lons) <= 180.0)
        assert np.all(np.abs(md.lats) <= 90.0)

    def test_scalar_negative(self, simple_segy):
        md = load_metadata(simple_segy)
        assert md.scalar_coord == -100

    def test_scalar_fac_applied(self, simple_segy):
        md = load_metadata(simple_segy)
        # scalar=-100 → fac=0.01; stored lon is lon_deg/0.01 → after fac: lon_deg
        assert md.lons is not None
        assert abs(float(md.lons[0]) - (-8.0)) < 1e-4

    def test_dist_km_shape(self, simple_segy):
        md = load_metadata(simple_segy)
        assert md.dist_km is not None
        assert md.dist_km.shape == (md.n_traces,)
        assert md.dist_km[0] == pytest.approx(0.0)
        assert md.total_km > 0.0


class TestLoadProfile:
    def test_data_shape(self, simple_segy):
        sd = load_profile(simple_segy)
        assert sd.data is not None
        assert sd.data.shape == (256, 50)
        assert sd.data.dtype == np.float32

    def test_amp_max_shape(self, simple_segy):
        sd = load_profile(simple_segy)
        assert sd.amp_max is not None
        assert sd.amp_max.shape == (50,)

    def test_clip_p99_positive(self, simple_segy):
        sd = load_profile(simple_segy)
        assert sd.clip_p99 is not None
        assert sd.clip_p99 > 0.0

    def test_no_load_traces(self, simple_segy):
        sd = load_profile(simple_segy, load_traces=False)
        assert sd.data is None
        assert sd.clip_p99 is None

    def test_timestamps_format(self, simple_segy):
        sd = load_profile(simple_segy)
        assert len(sd.timestamps) == 50
        for ts in sd.timestamps[:3]:
            assert "-DOY" in ts
            assert ":" in ts

    def test_delay_ms_from_trace0(self, delay_segy):
        sd = load_profile(delay_segy)
        assert sd.delay_ms == 20     # base delay
        assert sd.min_delay == 20.0
        assert sd.max_delay == 20.0 + (sd.n_traces - 1)  # variable delay

    def test_water_depth_shape(self, simple_segy):
        sd = load_profile(simple_segy)
        assert sd.water_depth is not None
        assert sd.water_depth.shape == (50,)
        assert np.all(np.isfinite(sd.water_depth))

    def test_summary_string(self, simple_segy):
        sd = load_profile(simple_segy)
        s  = sd.summary()
        assert "trazas" in s
        assert "km" in s

    def test_to_metadata(self, simple_segy):
        sd = load_profile(simple_segy)
        md = sd.to_metadata()
        assert isinstance(md, SegyMetadata)
        assert md.n_traces == sd.n_traces

    def test_active_ns_populated_when_traces_loaded(self, simple_segy):
        """simple_segy's data is full-depth random noise (no real cutoff),
        so active_ns should resolve near the full ns — proving the normal/
        no-meaningful-cutoff case never wrongly truncates."""
        sd = load_profile(simple_segy)
        assert sd.active_ns is not None
        assert 0 < sd.active_ns <= sd.ns

    def test_active_ns_none_for_header_only_stub(self, simple_segy):
        sd = load_profile(simple_segy, load_traces=False)
        assert sd.active_ns is None
        assert sd.active_ns_for_traces(0, sd.n_traces) is None

    def test_active_lo_populated_when_traces_loaded(self, simple_segy):
        """simple_segy's data is full-depth random noise (no real top
        cutoff), so active_lo should resolve near 0 — same 'normal case
        never wrongly truncates' guarantee as active_ns, just at the top."""
        sd = load_profile(simple_segy)
        assert sd.active_lo is not None
        assert 0 <= sd.active_lo < sd.active_ns

    def test_active_lo_none_for_header_only_stub(self, simple_segy):
        sd = load_profile(simple_segy, load_traces=False)
        assert sd.active_lo is None
        assert sd.active_band_for_traces(0, sd.n_traces) is None

    def test_active_band_for_traces_returns_lo_hi_pair(self, simple_segy):
        sd = load_profile(simple_segy)
        band = sd.active_band_for_traces(0, sd.n_traces)
        assert band == (sd.active_lo, sd.active_ns)


class TestDetectActiveNs:
    """_detect_active_ns (io_segy.py): the real listening-window depth — the
    last row with amplitude above a small fraction of clip_p99 — used to cap
    full_depth=True row processing below a file's nominal ns when most of
    that depth is recorded dead/padding (see SegyProfile.active_ns)."""

    def test_detects_real_signal_cutoff(self):
        from sbp_studio.core.io_segy import _detect_active_ns, ACTIVE_DEPTH_MARGIN_SAMPLES
        ns, n_traces = 1000, 20
        data = np.zeros((ns, n_traces), dtype=np.float32)
        data[:300, :] = 5.0          # real signal in the first 300 rows
        clip_p99 = float(np.percentile(np.abs(data), 99))
        active_ns = _detect_active_ns(data, clip_p99)
        assert active_ns == 300 + ACTIVE_DEPTH_MARGIN_SAMPLES

    def test_falls_back_to_full_ns_when_entirely_blank(self):
        from sbp_studio.core.io_segy import _detect_active_ns
        data = np.zeros((500, 10), dtype=np.float32)
        assert _detect_active_ns(data, clip_p99=0.0) == 500

    def test_falls_back_to_full_ns_when_signal_fills_the_whole_depth(self):
        from sbp_studio.core.io_segy import _detect_active_ns
        rng = np.random.default_rng(0)
        data = rng.standard_normal((400, 15)).astype(np.float32) * 5.0
        clip_p99 = float(np.percentile(np.abs(data), 99))
        assert _detect_active_ns(data, clip_p99) == 400

    def test_margin_overflow_clamps_to_ns(self):
        """A cutoff detected right near the bottom + margin must clamp at ns,
        never return a value past the array's own row count."""
        from sbp_studio.core.io_segy import _detect_active_ns
        ns = 200
        data = np.zeros((ns, 5), dtype=np.float32)
        data[:ns - 5, :] = 5.0       # real signal almost to the very last row
        clip_p99 = float(np.percentile(np.abs(data), 99))
        assert _detect_active_ns(data, clip_p99) == ns


class TestDetectActiveBand:
    """_detect_active_band (io_segy.py): the symmetric extension of
    _detect_active_ns — detects the TOP of the real signal band too, not
    just the bottom. The classic SBP case: a deep-water travel-time delay
    means every trace has DEAD leading samples before the sub-bottom
    reflectors of interest arrive — see SegyProfile.active_lo."""

    def test_detects_both_top_and_bottom_cutoff(self):
        from sbp_studio.core.io_segy import _detect_active_band, ACTIVE_DEPTH_MARGIN_SAMPLES
        ns, n_traces = 2000, 20
        data = np.zeros((ns, n_traces), dtype=np.float32)
        data[500:800, :] = 5.0       # real signal only in rows [500, 800)
        clip_p99 = float(np.percentile(np.abs(data), 99))
        lo, hi = _detect_active_band(data, clip_p99)
        assert lo == 500 - ACTIVE_DEPTH_MARGIN_SAMPLES
        assert hi == 800 + ACTIVE_DEPTH_MARGIN_SAMPLES

    def test_detect_active_ns_matches_the_band_hi(self):
        """Backward-compat: _detect_active_ns must be exactly the bottom
        half of _detect_active_band — same detection, same margin."""
        from sbp_studio.core.io_segy import _detect_active_band, _detect_active_ns
        rng = np.random.default_rng(2)
        data = np.zeros((1500, 12), dtype=np.float32)
        data[200:900, :] = rng.normal(0.0, 5.0, size=(700, 12))
        clip_p99 = float(np.percentile(np.abs(data), 99))
        lo, hi = _detect_active_band(data, clip_p99)
        assert _detect_active_ns(data, clip_p99) == hi

    def test_top_cutoff_clamps_at_zero_when_margin_overflows(self):
        """Real signal starting near row 0 must clamp lo at 0, never negative."""
        from sbp_studio.core.io_segy import _detect_active_band
        ns = 300
        data = np.zeros((ns, 8), dtype=np.float32)
        data[5:200, :] = 5.0          # signal starts almost at the very top
        clip_p99 = float(np.percentile(np.abs(data), 99))
        lo, hi = _detect_active_band(data, clip_p99)
        assert lo == 0

    def test_falls_back_to_full_band_when_entirely_blank(self):
        from sbp_studio.core.io_segy import _detect_active_band
        data = np.zeros((500, 10), dtype=np.float32)
        assert _detect_active_band(data, clip_p99=0.0) == (0, 500)

    def test_falls_back_to_full_band_when_signal_fills_the_whole_depth(self):
        from sbp_studio.core.io_segy import _detect_active_band
        rng = np.random.default_rng(0)
        data = rng.standard_normal((400, 15)).astype(np.float32) * 5.0
        clip_p99 = float(np.percentile(np.abs(data), 99))
        assert _detect_active_band(data, clip_p99) == (0, 400)
