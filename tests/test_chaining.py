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

    def test_load_chain_traces_clip_p99_is_max_of_per_file_values(self, chain_pair):
        """Phase 7: clip_p99 is now the MAX of the already-computed per-file
        values (never an underestimate, same 'safe union' philosophy as
        active_band_for_traces) instead of a true whole-array percentile —
        clip_p99 is metadata only (no DSP/display-stability path reads it),
        so this approximation is a deliberate, documented trade for skipping
        the O(ns*total_traces) full-array np.abs+percentile pass."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        ch.load_chain_traces()
        assert ch.clip_p99 == max(sd1.clip_p99, sd2.clip_p99)


class TestReadColumns:
    """ProfileChain.read_columns — the segmented alternative to
    load_chain_traces. NEVER builds/touches the whole-chain self.data; reads
    only the (typically 1, rarely a handful at a seam) segment(s) the
    requested column range actually overlaps."""

    def test_never_touches_chain_data(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        ch.read_columns(0, sd1.n_traces)
        assert ch.data is None   # the monolithic array was never built

    def test_range_within_a_single_segment_matches_that_profiles_data(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        out = ch.read_columns(0, sd1.n_traces)
        np.testing.assert_array_equal(out, sd1.data)

        out2 = ch.read_columns(sd1.n_traces, sd1.n_traces + sd2.n_traces)
        np.testing.assert_array_equal(out2, sd2.data)

    def test_range_spanning_the_seam_matches_the_stitched_array(self, chain_pair):
        """The decisive equivalence check: a read across the boundary must
        match exactly what load_chain_traces' full concat would give for the
        same absolute column range — proving the bounded multi-segment join
        is correct, not just the single-segment fast path."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch_segmented = ProfileChain([sd1, sd2])
        c0, c1 = sd1.n_traces - 3, sd1.n_traces + 3
        spanning = ch_segmented.read_columns(c0, c1)

        ch_full = ProfileChain([load_profile(path1), load_profile(path2)])
        ch_full.load_chain_traces()
        expected = ch_full.data[:, c0:c1]
        np.testing.assert_array_equal(spanning, expected)
        assert ch_segmented.data is None   # still never materialised

    def test_reloads_an_evicted_segment_from_disk(self, chain_pair):
        """Mirrors load_chain_traces_from_evicted_stubs — an evicted
        constituent must be transparently re-read from disk, and the reload
        lands in THAT PROFILE's own .data slot (the same object AppState's
        per-profile LRU manages), not a separate chain-level allocation."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        original = sd1.data.copy()
        ch = ProfileChain([sd1, sd2])
        sd1.data = None                          # simulate LRU eviction
        out = ch.read_columns(0, sd1.n_traces)
        np.testing.assert_array_equal(out, original)
        assert sd1.data is not None              # reload landed on the profile itself
        np.testing.assert_array_equal(sd1.data, original)

    def test_out_of_range_clamps_rather_than_errors(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        total = sd1.n_traces + sd2.n_traces
        out = ch.read_columns(-50, total + 999)
        assert out.shape[1] == total

    def test_empty_range_returns_zero_width_array(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        out = ch.read_columns(5, 5)
        assert out.shape == (sd1.ns, 0)


class TestSegyProfileReadColumns:
    """SegyProfile.read_columns — the trivial single-file case, duck-typed
    alongside ProfileChain.read_columns so GUI code can call either
    uniformly."""

    def test_matches_a_direct_slice(self, simple_segy):
        sd = load_profile(simple_segy)
        out = sd.read_columns(2, 5)
        np.testing.assert_array_equal(out, sd.data[:, 2:5])

    def test_reloads_when_evicted(self, simple_segy):
        sd = load_profile(simple_segy)
        original = sd.data.copy()
        sd.data = None
        out = sd.read_columns(0, sd.n_traces)
        np.testing.assert_array_equal(out, original)
        assert sd.data is not None

    def test_clamps_out_of_range(self, simple_segy):
        sd = load_profile(simple_segy)
        out = sd.read_columns(-10, sd.n_traces + 500)
        assert out.shape[1] == sd.n_traces


class TestActiveNsForTraces:
    """ProfileChain.active_ns_for_traces: resolves the real listening-window
    depth (SegyProfile.active_ns) for whichever constituent file(s) overlap
    a column range — the per-FILE granularity the live preview's true-depth
    cap needs for a stitched cadena where one segment's real signal ends far
    short of another's (see extract_visible_window's active_ns_lookup)."""

    def _chain_with_active_ns(self, chain_pair, ns1, ns2):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        sd1.active_ns, sd2.active_ns = ns1, ns2
        ch = ProfileChain([sd1, sd2])
        return ch, sd1, sd2

    def test_resolves_to_the_single_overlapping_profile(self, chain_pair):
        ch, sd1, sd2 = self._chain_with_active_ns(chain_pair, 100, 800)
        # A range entirely within the first profile's columns.
        assert ch.active_ns_for_traces(0, sd1.n_traces) == 100
        # A range entirely within the second profile's columns.
        assert ch.active_ns_for_traces(sd1.n_traces, ch.n_traces) == 800

    def test_resolves_to_the_max_when_range_spans_both_profiles(self, chain_pair):
        """The deeper of the overlapping files governs — the preview must
        never under-cut whichever constituent needs more depth."""
        ch, sd1, sd2 = self._chain_with_active_ns(chain_pair, 100, 800)
        spanning = ch.active_ns_for_traces(sd1.n_traces - 1, sd1.n_traces + 1)
        assert spanning == 800

    def test_returns_none_when_an_overlapping_profile_is_unloaded(self, chain_pair):
        """A stub (active_ns=None) in the overlapping range means 'don't
        know' — the caller must fall back to the full ns, never guess."""
        ch, sd1, sd2 = self._chain_with_active_ns(chain_pair, 100, None)
        assert ch.active_ns_for_traces(0, sd1.n_traces) == 100
        assert ch.active_ns_for_traces(sd1.n_traces, ch.n_traces) is None
        # A range spanning both: still None, since one overlapping file is unknown.
        assert ch.active_ns_for_traces(sd1.n_traces - 1, sd1.n_traces + 1) is None


class TestActiveBandForTraces:
    """ProfileChain.active_band_for_traces: the symmetric (lo, hi) extension
    of active_ns_for_traces — Strategy A's top+bottom real-signal band,
    resolved per overlapping constituent file the same way."""

    def _chain_with_active_band(self, chain_pair, band1, band2):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        if band1 is None:
            sd1.active_lo = sd1.active_ns = None
        else:
            sd1.active_lo, sd1.active_ns = band1
        if band2 is None:
            sd2.active_lo = sd2.active_ns = None
        else:
            sd2.active_lo, sd2.active_ns = band2
        ch = ProfileChain([sd1, sd2])
        return ch, sd1, sd2

    def test_resolves_to_the_single_overlapping_profile(self, chain_pair):
        ch, sd1, sd2 = self._chain_with_active_band(chain_pair, (50, 100), (200, 800))
        assert ch.active_band_for_traces(0, sd1.n_traces) == (50, 100)
        assert ch.active_band_for_traces(sd1.n_traces, ch.n_traces) == (200, 800)

    def test_spanning_range_takes_the_safe_union(self, chain_pair):
        """The combined range must never under-cut EITHER overlapping file:
        the SHALLOWEST top (min of the los) and the DEEPEST bottom (max of
        the his) — the safe union of both bands."""
        ch, sd1, sd2 = self._chain_with_active_band(chain_pair, (50, 100), (200, 800))
        spanning = ch.active_band_for_traces(sd1.n_traces - 1, sd1.n_traces + 1)
        assert spanning == (50, 800)

    def test_returns_none_when_an_overlapping_profile_is_unloaded(self, chain_pair):
        ch, sd1, sd2 = self._chain_with_active_band(chain_pair, (50, 100), None)
        assert ch.active_band_for_traces(0, sd1.n_traces) == (50, 100)
        assert ch.active_band_for_traces(sd1.n_traces, ch.n_traces) is None
        assert ch.active_band_for_traces(sd1.n_traces - 1, sd1.n_traces + 1) is None


class TestColumnsInCache:
    """ProfileChain.columns_in_cache — the disk-I/O gate used by Phase 10's
    BasePrepWorker to decide whether to dispatch off-thread before calling
    _prepared_base."""

    def test_true_when_all_segments_are_in_memory(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        assert ch.columns_in_cache(0, ch.n_traces) is True

    def test_false_when_first_segment_evicted(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        sd1.data = None
        assert ch.columns_in_cache(0, sd1.n_traces) is False

    def test_false_when_second_segment_evicted(self, chain_pair):
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        sd2.data = None
        assert ch.columns_in_cache(sd1.n_traces, ch.n_traces) is False

    def test_true_for_range_that_only_touches_resident_segment(self, chain_pair):
        """sd2 evicted but the query only overlaps sd1 → True."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        sd2.data = None
        assert ch.columns_in_cache(0, sd1.n_traces) is True

    def test_false_when_spanning_range_includes_evicted_segment(self, chain_pair):
        """A seam-crossing query hitting the evicted sd2 → False."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        sd2.data = None
        assert ch.columns_in_cache(sd1.n_traces - 2, sd1.n_traces + 2) is False

    def test_out_of_range_clamps_and_returns_true(self, chain_pair):
        """Requesting a range that clamps to nothing or just the resident
        portion must not raise and must return True."""
        path1, path2 = chain_pair
        sd1, sd2 = load_profile(path1), load_profile(path2)
        ch = ProfileChain([sd1, sd2])
        # Completely out of range — clamps to empty → no segments → True.
        assert ch.columns_in_cache(ch.n_traces + 100, ch.n_traces + 200) is True
        # Identical endpoints → empty range → True.
        assert ch.columns_in_cache(5, 5) is True


class TestHeterogeneousNsChain:
    """Heterogeneous per-file record lengths (the field crash: 'array at
    index 0 has size 6427 ... index 1 has size 32767'). The chain's vertical
    extent must be the DEEPEST constituent file, with shorter segments
    bottom-padded with zeros in BOTH assembly paths (load_chain_traces and
    read_columns)."""

    NS_SHORT, NS_DEEP = 96, 256

    def _hetero_chain(self, base_dir):
        """Two contiguous files (same dt, chain-linkable geometry/timestamps
        — mirrors make_chain_pair) with DIFFERENT sample counts."""
        from tests.make_synthetic_segy import make_synthetic_segy
        from pathlib import Path
        p1 = str(Path(base_dir) / "het_1.sgy")
        p2 = str(Path(base_dir) / "het_2.sgy")
        n_traces, lon_step = 40, 0.001
        end_lon = -8.0 + n_traces * lon_step
        gap_deg = 0.05 / 111.32
        make_synthetic_segy(p1, n_traces=n_traces, ns=self.NS_SHORT, dt_us=500,
                            base_lon=-8.0, doy_start=100, minute_start=0)
        make_synthetic_segy(p2, n_traces=n_traces, ns=self.NS_DEEP, dt_us=500,
                            base_lon=end_lon + gap_deg, doy_start=100,
                            minute_start=n_traces)
        profiles = [load_profile(p1), load_profile(p2)]
        chains = detect_chains(profiles, gap_km=1.0)
        assert len(chains) == 1 and len(chains[0].profiles) == 2
        return chains[0], profiles

    def test_chain_ns_is_deepest_file(self, tmp_path):
        ch, _ = self._hetero_chain(tmp_path)
        assert ch.ns == self.NS_DEEP
        assert ch.dur_ms == pytest.approx(self.NS_DEEP * ch.dt_us / 1000.0)

    def test_load_chain_traces_pads_instead_of_crashing(self, tmp_path):
        """The exact reported crash path: np.concatenate over mismatched
        row counts raised ValueError before the fix."""
        ch, profiles = self._hetero_chain(tmp_path)
        ch.load_chain_traces()
        assert ch.data.shape == (self.NS_DEEP, ch.n_traces)
        n1 = profiles[0].n_traces
        # Segment 1 (short file): real samples on top, zero pad below.
        np.testing.assert_array_equal(
            ch.data[:self.NS_SHORT, :n1], profiles[0].data)
        assert not ch.data[self.NS_SHORT:, :n1].any()      # pad is all zeros
        # Segment 2 (deep file): full height, byte-identical.
        np.testing.assert_array_equal(ch.data[:, n1:], profiles[1].data)

    def test_read_columns_pads_across_the_seam(self, tmp_path):
        """read_columns must agree with load_chain_traces column-for-column,
        including a window straddling the short→deep seam."""
        ch, profiles = self._hetero_chain(tmp_path)
        n1 = profiles[0].n_traces
        win = ch.read_columns(n1 - 5, n1 + 5)              # straddles the seam
        assert win.shape == (self.NS_DEEP, 10)
        ch.load_chain_traces()
        np.testing.assert_array_equal(win, ch.data[:, n1 - 5:n1 + 5])
        # Single-segment window inside the SHORT file must also be full height.
        win_short = ch.read_columns(0, 5)
        assert win_short.shape == (self.NS_DEEP, 5)
        assert not win_short[self.NS_SHORT:, :].any()

    def test_homogeneous_chain_unchanged(self, chain_pair):
        """Equal-ns chains (the normal case) must behave exactly as before —
        no padding rows, data equal to the plain concatenation."""
        profiles = [load_profile(chain_pair[0]), load_profile(chain_pair[1])]
        ch = detect_chains(profiles, gap_km=1.0)[0]
        ch.load_chain_traces()
        assert ch.ns == profiles[0].ns
        expected = np.concatenate([p.data for p in profiles], axis=1)
        np.testing.assert_array_equal(ch.data, expected)
