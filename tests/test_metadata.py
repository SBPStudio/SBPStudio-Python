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
