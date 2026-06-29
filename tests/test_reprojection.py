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
        logic) is identical to a plain (unfiltered) reprojection; only the
        trace samples differ, by exactly the transform applied."""
        import shutil
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
            assert fp.bin == ff.bin
            assert fp.text[0] == ff.text[0]
            for i in range(fp.tracecount):
                assert dict(fp.header[i]) == dict(ff.header[i])   # EVERY field
            np.testing.assert_allclose(
                ff.trace.raw[:], fp.trace.raw[:] * 2.0, rtol=1e-3, atol=1e-3)

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
            for i in range(fp.tracecount):
                assert dict(fp.header[i]) == dict(ff.header[i])
            np.testing.assert_allclose(
                ff.trace.raw[:], fp.trace.raw[:] * 3.0, rtol=1e-3, atol=1e-3)
