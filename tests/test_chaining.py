"""
test_chaining.py — Characterisation tests for ProfileChain detection and concat.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbp_studio.core import load_profile, detect_chains, ProfileChain


class TestDetectChains:
    def test_two_contiguous_form_one_chain(self, chain_pair):
        path1, path2 = chain_pair
        profiles = [load_profile(path1), load_profile(path2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        assert len(chains) == 1
        assert len(chains[0].profiles) == 2

    def test_far_apart_form_separate_chains(self, tmp_path):
        from tests.make_synthetic_segy import make_synthetic_segy
        p1 = make_synthetic_segy(str(tmp_path / "a.sgy"), base_lon=-8.0, base_lat=43.0)
        p2 = make_synthetic_segy(str(tmp_path / "b.sgy"), base_lon=10.0, base_lat=55.0)
        profiles = [load_profile(p1), load_profile(p2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        assert len(chains) == 2

    def test_different_dt_forms_separate_chains(self, tmp_path):
        from tests.make_synthetic_segy import make_synthetic_segy
        p1 = make_synthetic_segy(str(tmp_path / "dt250.sgy"), dt_us=250)
        p2 = make_synthetic_segy(str(tmp_path / "dt500.sgy"), dt_us=500,
                                  base_lon=-8.001)
        profiles = [load_profile(p1), load_profile(p2)]
        chains   = detect_chains(profiles)
        assert len(chains) == 2

    def test_empty_input(self):
        chains = detect_chains([])
        assert chains == []

    def test_single_profile_wrapped(self, simple_segy):
        sd     = load_profile(simple_segy)
        chains = detect_chains([sd])
        assert len(chains) == 1
        assert len(chains[0].profiles) == 1


class TestProfileChainConcat:
    def test_chain_init_is_lazy(self, chain_pair):
        """A freshly-detected chain must NOT hold a trace matrix (RAM-flat) —
        only lightweight header-derived metadata, available immediately."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        assert ch.data is None
        assert ch.clip_p99 is None
        # n_traces is summed from the stubs, not the (absent) matrix.
        assert ch.n_traces == sd1.n_traces + sd2.n_traces

    def test_data_shape(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        ch.load_chain_traces()                       # explicit, lazy assembly
        assert ch.data.shape == (sd1.ns, sd1.n_traces + sd2.n_traces)
        assert ch.data.dtype == np.float32
        assert ch.clip_p99 is not None

    def test_load_chain_traces_from_evicted_stubs(self, chain_pair):
        """The crash that bit us: constituent profiles were evicted to stubs
        (data=None). load_chain_traces must re-read them from disk and stitch
        correctly instead of np.concatenate-ing 0-D Nones."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        # Simulate LRU eviction AFTER the chain was built.
        sd1.data = sd2.data = None
        ch.load_chain_traces()
        assert ch.data.shape == (sd1.ns, sd1.n_traces + sd2.n_traces)

    def test_load_chain_traces_idempotent(self, chain_pair):
        path1, path2 = chain_pair
        ch = ProfileChain([load_profile(path1), load_profile(path2)])
        first = ch.load_chain_traces().data
        assert ch.load_chain_traces().data is first   # no rebuild on 2nd call

    def test_dist_km_continuous(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        # dist_km should be monotonically non-decreasing
        assert np.all(np.diff(ch.dist_km) >= 0)
        assert ch.dist_km[0] == pytest.approx(0.0)

    def test_haversine_km_known(self):
        # 1° latitude ≈ 111.19 km; small tolerance
        d = ProfileChain._haversine_km(0.0, 0.0, 0.0, 1.0)
        assert 110.0 < d < 112.0

    def test_timestamps_length(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        assert len(ch.timestamps) == sd1.n_traces + sd2.n_traces

    def test_boundaries_km_count(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        assert len(ch.boundaries_km) == 1   # one boundary for 2 profiles
