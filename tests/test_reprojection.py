"""
test_reprojection.py — Regression + contract tests for reproject_one.

Checks:
- DelayRecordingTime preserved (round-trip read)
- SourceGroupScalar / CoordinateUnits match expected values
- bin and text[0] copied
- Partial output deleted on failure (ReprojectionError)
- CRSError raised on invalid CRS
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import segyio

from sbp_studio.core import (
    load_profile, load_metadata, reproject_one, reproject_chain,
    detect_chains, CRSError, ReprojectionError,
)
from sbp_studio.core.io_segy import _ref_reproject_trace


class TestReprojectOne:
    def test_creates_output_file(self, simple_segy, tmp_path):
        sd      = load_profile(simple_segy)
        out_sgy = str(tmp_path / "out.sgy")
        import shutil
        shutil.copy(simple_segy, out_sgy)
        # Use the reference segy as source; reproject into UTM 30N
        sd2 = load_profile(out_sgy)
        out = reproject_one(sd2, "EPSG:4326", "EPSG:32630")
        assert os.path.exists(out)
        os.remove(out)

    def test_delay_preserved(self, delay_segy, tmp_path):
        """DelayRecordingTime must be unchanged after reprojection."""
        import shutil
        src_path = str(tmp_path / "src.sgy")
        shutil.copy(delay_segy, src_path)
        sd     = load_profile(src_path)
        out    = reproject_one(sd, "EPSG:4326", "EPSG:32630")

        # Read the reproj file and compare delays
        with segyio.open(out, ignore_geometry=True) as f:
            delays_out = np.asarray(f.attributes(segyio.TraceField.DelayRecordingTime)[:])
        np.testing.assert_array_equal(delays_out, sd.delays)
        os.remove(out)

    def test_scalar_and_unit_geographic(self, simple_segy, tmp_path):
        """Geographic dst → new_uc=3; out_sc=-10_000_000 truncated to 16-bit in file.

        SEG-Y SourceGroupScalar is a 16-bit field. -10_000_000 overflows,
        so segyio reads back the 16-bit two's-complement truncation (27008).
        This is the monolith's actual behaviour — preserved, NOT fixed.
        """
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd  = load_profile(src)
        out = reproject_one(sd, "EPSG:4326", "EPSG:4258")  # to ETRS89 geographic
        with segyio.open(out, ignore_geometry=True) as f:
            sc = int(f.header[0][segyio.TraceField.SourceGroupScalar])
            uc = int(f.header[0][segyio.TraceField.CoordinateUnits])
        # CoordinateUnits = 3 (decimal degrees) is the reliable check
        assert uc == 3
        # OQ-3 fix: out_sc = -10_000 (no longer overflows 16-bit field)
        assert sc == -10_000, (
            f"Expected out_sc=-10000 (OQ-3 fix applied), got {sc}. "
            "The previous -10_000_000 overflowed the 16-bit SourceGroupScalar field.")
        os.remove(out)

    def test_scalar_and_unit_projected(self, simple_segy, tmp_path):
        """Projected dst → out_sc=-100, new_uc=1."""
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd  = load_profile(src)
        out = reproject_one(sd, "EPSG:4326", "EPSG:32630")
        with segyio.open(out, ignore_geometry=True) as f:
            sc = int(f.header[0][segyio.TraceField.SourceGroupScalar])
            uc = int(f.header[0][segyio.TraceField.CoordinateUnits])
        assert sc == -100
        assert uc == 1
        os.remove(out)

    def test_text_header_copied(self, simple_segy, tmp_path):
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd  = load_profile(src)
        out = reproject_one(sd, "EPSG:4326", "EPSG:32630")
        with segyio.open(simple_segy, ignore_geometry=True) as src_f, \
             segyio.open(out,         ignore_geometry=True) as out_f:
            assert src_f.text[0] == out_f.text[0]
        os.remove(out)

    def test_invalid_crs_raises(self, simple_segy, tmp_path):
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd  = load_profile(src)
        with pytest.raises(CRSError):
            reproject_one(sd, "NOT_A_CRS", "EPSG:32630")

    def test_partial_output_deleted_on_error(self, simple_segy, tmp_path):
        """Partial output must be deleted on failure.
        Inject error via the bulk-coords function (optimised path used by default).
        """
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd  = load_profile(src)
        import sbp_studio.core.io_segy as _io
        orig = _io._opt_reproject_coords_bulk
        def _bad(*a, **k):
            raise RuntimeError("forced test error")
        _io._opt_reproject_coords_bulk = _bad
        from pathlib import Path
        expected_out = str(Path(src).with_name(Path(src).stem + "_REPROY" + Path(src).suffix))
        try:
            with pytest.raises(ReprojectionError):
                reproject_one(sd, "EPSG:4326", "EPSG:32630")
        finally:
            _io._opt_reproject_coords_bulk = orig
        assert not os.path.exists(expected_out)


class TestReprojectChain:
    def test_creates_joined_file(self, chain_pair, tmp_path):
        import shutil
        path1, path2 = chain_pair
        src1 = str(tmp_path / "c1.sgy")
        src2 = str(tmp_path / "c2.sgy")
        shutil.copy(path1, src1)
        shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        assert len(chains) == 1
        ch  = chains[0]
        out = reproject_chain(ch, "EPSG:4326", "EPSG:32630")
        assert os.path.exists(out)
        # Trace count must equal sum of source profiles
        md = load_metadata(out)
        assert md.n_traces == ch.n_traces
        os.remove(out)

    def test_trace_number_sequential(self, chain_pair, tmp_path):
        import shutil
        path1, path2 = chain_pair
        src1, src2 = str(tmp_path / "c1.sgy"), str(tmp_path / "c2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        chains   = detect_chains(profiles, gap_km=1.0)
        out = reproject_chain(chains[0], "EPSG:4326", "EPSG:32630")
        with segyio.open(out, ignore_geometry=True) as f:
            tnums = np.asarray(f.attributes(segyio.TraceField.TraceNumber)[:])
        expected = np.arange(1, chains[0].n_traces + 1)
        np.testing.assert_array_equal(tnums, expected)
        os.remove(out)


class TestAmplitudeTransform:
    """``amplitude_transform`` — the 'Aplicar filtros' filtered-export
    feature. Headers (text/binary/every trace header field) must stay
    byte-identical to the un-filtered path; ONLY the trace amplitude
    payload may differ, by exactly the given transform."""

    def test_default_none_is_byte_identical_to_unfiltered(self, simple_segy, tmp_path):
        """amplitude_transform=None must reproduce the EXACT pre-existing
        behaviour — no regression for the common (filters off) case."""
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd = load_profile(src)
        out_a = reproject_one(sd, "EPSG:4326", "EPSG:32630",
                              out_path=str(tmp_path / "a.sgy"))
        out_b = reproject_one(sd, "EPSG:4326", "EPSG:32630",
                              amplitude_transform=None,
                              out_path=str(tmp_path / "b.sgy"))
        with segyio.open(out_a, ignore_geometry=True) as fa, \
             segyio.open(out_b, ignore_geometry=True) as fb:
            np.testing.assert_array_equal(fa.trace.raw[:], fb.trace.raw[:])
            for i in range(fa.tracecount):
                assert dict(fa.header[i]) == dict(fb.header[i])

    def test_only_amplitude_payload_changes_headers_cloned_exactly(
            self, simple_segy, tmp_path):
        """The core contract: every header field (coordinates included,
        since amplitude_transform doesn't touch the coordinate-overwrite
        logic) is identical to a plain (unfiltered) reprojection. Two
        deliberate exceptions on the FILTERED side (the WYSIWYG fixes):
        the binary header's Format word becomes 5 (IEEE float32, so the
        filtered float payload isn't quantised back into IBM/int), and the
        transform's input is the DC-removed matrix — the same stage-0 input
        the live preview filters — not the raw file bytes."""
        import shutil
        from sbp_studio.core import apply_dc_removal
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd = load_profile(src)

        plain = reproject_one(sd, "EPSG:4326", "EPSG:32630",
                              out_path=str(tmp_path / "plain.sgy"))
        filtered = reproject_one(
            sd, "EPSG:4326", "EPSG:32630",
            amplitude_transform=lambda d: d * 2.0,
            out_path=str(tmp_path / "filtered.sgy"))

        with segyio.open(plain, ignore_geometry=True) as fp, \
             segyio.open(filtered, ignore_geometry=True) as ff:
            bin_p, bin_f = dict(fp.bin), dict(ff.bin)
            assert bin_f.pop(segyio.BinField.Format) == 5     # IEEE float32
            bin_p.pop(segyio.BinField.Format)
            assert bin_p == bin_f                             # rest identical
            assert fp.text[0] == ff.text[0]
            for i in range(fp.tracecount):
                assert dict(fp.header[i]) == dict(ff.header[i])   # EVERY field
            expected = apply_dc_removal(
                fp.trace.raw[:].T.astype(np.float32)) * 2.0
            np.testing.assert_allclose(
                ff.trace.raw[:].T, expected, rtol=1e-3, atol=1e-3)

    def test_shape_mismatch_raises_reprojection_error(self, simple_segy, tmp_path):
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd = load_profile(src)
        with pytest.raises(ReprojectionError):
            reproject_one(sd, "EPSG:4326", "EPSG:32630",
                          amplitude_transform=lambda d: d[:-1, :],
                          out_path=str(tmp_path / "bad.sgy"))

    def test_chain_applies_transform_per_constituent_file(self, chain_pair, tmp_path):
        import shutil
        path1, path2 = chain_pair
        src1, src2 = str(tmp_path / "c1.sgy"), str(tmp_path / "c2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        chains = detect_chains(profiles, gap_km=1.0)

        plain = reproject_chain(chains[0], "EPSG:4326", "EPSG:32630",
                                out_path=str(tmp_path / "plain_chain.sgy"))
        filtered = reproject_chain(
            chains[0], "EPSG:4326", "EPSG:32630",
            amplitude_transform=lambda d: d * 3.0,
            out_path=str(tmp_path / "filtered_chain.sgy"))

        with segyio.open(plain, ignore_geometry=True) as fp, \
             segyio.open(filtered, ignore_geometry=True) as ff:
            from sbp_studio.core import apply_dc_removal
            for i in range(fp.tracecount):
                assert dict(fp.header[i]) == dict(ff.header[i])
            # Transform input is DC-removed per constituent file (WYSIWYG —
            # same stage-0 input the preview filters); output is IEEE float32.
            assert int(ff.bin[segyio.BinField.Format]) == 5
            expected = apply_dc_removal(
                fp.trace.raw[:].T.astype(np.float32)) * 3.0
            np.testing.assert_allclose(
                ff.trace.raw[:].T, expected, rtol=1e-3, atol=1e-3)


class TestWysiwygFilteredExport:
    """Full WYSIWYG contract for filtered exports (the 'exported filtered
    chain looks washed out on re-import' bug). Three coupled guarantees:
    (1) filtered samples are written as IEEE float32 (format 5) — never
    quantised back into the source's IBM/integer sample format; (2) the
    transform filters the SAME stage-0 DC-removed input the live preview
    shows; (3) the loader does NOT DC-remove single-signed (envelope-like)
    data on re-import. Together: preview -> export -> re-import is
    BIT-exact."""

    @staticmethod
    def _envelope_like(d):
        a = np.abs(d)
        return (a / (a.max() + 1e-30)).astype(np.float32)

    def test_envelope_export_round_trips_bit_exact(self, simple_segy, tmp_path):
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd = load_profile(src)
        preview = self._envelope_like(sd.data)     # what the live preview shows

        out = reproject_one(sd, "EPSG:4326", "EPSG:32630",
                            amplitude_transform=self._envelope_like,
                            out_path=str(tmp_path / "env.sgy"))
        with segyio.open(out, ignore_geometry=True) as f:
            assert int(f.bin[segyio.BinField.Format]) == 5   # IEEE float32

        sd2 = load_profile(out)                    # the user's actual re-import
        assert sd2.error is None
        np.testing.assert_array_equal(sd2.data, preview)     # BIT-exact WYSIWYG

    def test_loader_preserves_single_signed_offset(self, simple_segy, tmp_path):
        """An all-positive file's per-trace mean is signal, not bias — the
        loader must NOT zero-centre it (that warped [0,1] envelopes into
        ±-swinging data no display range could fix)."""
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd = load_profile(src)
        out = reproject_one(sd, "EPSG:4326", "EPSG:32630",
                            amplitude_transform=self._envelope_like,
                            out_path=str(tmp_path / "env2.sgy"))
        sd2 = load_profile(out)
        assert float(sd2.data.min()) >= 0.0        # positivity preserved
        assert float(sd2.data.mean()) > 0.01       # offset (the signal) intact

    def test_loader_still_dc_removes_mixed_sign_raw(self, simple_segy):
        """Genuine raw (mixed-sign) seismic keeps mandatory stage-0 DC
        removal — every trace zero-centred exactly as before the guard."""
        sd = load_profile(simple_segy)
        assert float(sd.data.min()) < 0.0 < float(sd.data.max())
        col_means = sd.data.mean(axis=0)
        assert float(np.abs(col_means).max()) < 1e-4

    def test_unfiltered_export_keeps_source_format(self, simple_segy, tmp_path):
        """transform=None stays fully byte-faithful — including the source's
        own sample format word (the IEEE forcing is filtered-path only)."""
        import shutil
        src = str(tmp_path / "src.sgy")
        shutil.copy(simple_segy, src)
        sd = load_profile(src)
        out = reproject_one(sd, "EPSG:4326", "EPSG:32630",
                            out_path=str(tmp_path / "plain2.sgy"))
        with segyio.open(src, ignore_geometry=True) as fs, \
             segyio.open(out, ignore_geometry=True) as fo:
            assert int(fo.bin[segyio.BinField.Format]) == \
                   int(fs.bin[segyio.BinField.Format])


class TestHeterogeneousChainExport:
    """Chain export with heterogeneous per-file record lengths (the second
    half of the field crash fix — model.py's viewer-side padding landed
    separately). A single SEG-Y has ONE trace length, so the output is sized
    by the DEEPEST constituent file; shorter segments are bottom-padded with
    zeros, their per-trace sample-count word updated, and the binary header's
    Samples word re-patched. Homogeneous chains stay byte-identical."""

    NS_SHORT, NS_DEEP = 96, 256

    def _hetero_chain(self, base_dir):
        from tests.make_synthetic_segy import make_synthetic_segy
        from pathlib import Path
        p1 = str(Path(base_dir) / "hx_1.sgy")
        p2 = str(Path(base_dir) / "hx_2.sgy")
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
        return chains[0], p1, p2

    def _assert_padded_output(self, out, p1, p2, n1):
        """Shared contract: deepest-ns output, byte-faithful payload on top,
        provably-zero pad below, headers consistent with the written size."""
        with segyio.open(out, ignore_geometry=True) as f, \
             segyio.open(p1, ignore_geometry=True) as f1, \
             segyio.open(p2, ignore_geometry=True) as f2:
            assert len(f.samples) == self.NS_DEEP
            assert int(f.bin[segyio.BinField.Samples]) == self.NS_DEEP
            raw = f.trace.raw[:]
            # Short segment: source bytes on top, zeros below.
            np.testing.assert_array_equal(raw[:n1, :self.NS_SHORT],
                                          f1.trace.raw[:])
            assert not raw[:n1, self.NS_SHORT:].any()
            # Deep segment: byte-identical, full height.
            np.testing.assert_array_equal(raw[n1:], f2.trace.raw[:])
            # Per-trace sample-count word: PATCHED to the written length for
            # the padded segment; byte-faithfully UNCHANGED from the source
            # for the native-height segment (the fixture leaves it 0 there —
            # the clone must preserve whatever the source had, not invent it).
            counts = np.asarray(
                f.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[:])
            src_counts = np.asarray(
                f2.attributes(segyio.TraceField.TRACE_SAMPLE_COUNT)[:])
            assert (counts[:n1] == self.NS_DEEP).all()      # patched (padded)
            np.testing.assert_array_equal(counts[n1:], src_counts)

    def test_reproject_chain_pads_short_segment(self, tmp_path):
        ch, p1, p2 = self._hetero_chain(tmp_path)
        n1 = ch.profiles[0].n_traces
        out = reproject_chain(ch, "EPSG:4326", "EPSG:32630",
                              out_path=str(tmp_path / "hx_reproy.sgy"))
        self._assert_padded_output(out, p1, p2, n1)

    def test_join_profiles_pads_short_segment(self, tmp_path):
        from sbp_studio.core import join_profiles
        ch, p1, p2 = self._hetero_chain(tmp_path)
        n1 = ch.profiles[0].n_traces
        out = join_profiles(ch, out_path=str(tmp_path / "hx_joined.sgy"))
        self._assert_padded_output(out, p1, p2, n1)

    def test_filtered_hetero_export_and_reimport(self, tmp_path):
        """Filtered path over a heterogeneous chain: per-file transform, then
        padding, then IEEE float32 — and the result re-imports through the
        real loader at the full (deepest) height, matching the viewer."""
        ch, p1, p2 = self._hetero_chain(tmp_path)
        n1 = ch.profiles[0].n_traces
        out = reproject_chain(ch, "EPSG:4326", "EPSG:32630",
                              amplitude_transform=lambda d: d * 2.0,
                              out_path=str(tmp_path / "hx_filt.sgy"))
        with segyio.open(out, ignore_geometry=True) as f:
            assert int(f.bin[segyio.BinField.Format]) == 5
            assert len(f.samples) == self.NS_DEEP
            raw = f.trace.raw[:]
            assert not raw[:n1, self.NS_SHORT:].any()       # pad survived filter
        sd2 = load_profile(out)                              # user's re-import
        assert sd2.error is None
        assert sd2.ns == self.NS_DEEP
        assert sd2.data.shape[0] == self.NS_DEEP

    def test_homogeneous_chain_export_unchanged(self, chain_pair, tmp_path):
        """Equal-ns chains must produce byte-identical trace payloads and an
        untouched sample-count word — the padding machinery is a strict no-op."""
        import shutil
        path1, path2 = chain_pair
        src1, src2 = str(tmp_path / "h1.sgy"), str(tmp_path / "h2.sgy")
        shutil.copy(path1, src1); shutil.copy(path2, src2)
        profiles = [load_profile(src1), load_profile(src2)]
        ch = detect_chains(profiles, gap_km=1.0)[0]
        out = reproject_chain(ch, "EPSG:4326", "EPSG:32630",
                              out_path=str(tmp_path / "h_reproy.sgy"))
        with segyio.open(out, ignore_geometry=True) as f, \
             segyio.open(src1, ignore_geometry=True) as f1, \
             segyio.open(src2, ignore_geometry=True) as f2:
            n1 = f1.tracecount
            assert len(f.samples) == len(f1.samples)
            np.testing.assert_array_equal(f.trace.raw[:n1], f1.trace.raw[:])
            np.testing.assert_array_equal(f.trace.raw[n1:], f2.trace.raw[:])
