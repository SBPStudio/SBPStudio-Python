"""
test_dedup_timestamps.py — duplicate-timestamp artifact cleanup in the core loader.

The TOPAS acquisition system stamps blocks of CONSECUTIVE traces with an identical
DayOfYear/Hour/Minute/Second. io_segy purges those duplicates and applies the SAME
keep-mask to EVERY per-trace array so nothing is left spatially misaligned. These
tests pin:

  * the (n_traces, 4) diff detection + first-trace-always-kept rule,
  * surgical trimming of data + coords + delays + water depth + timestamps +
    inspector headers (alignment proven via a unique-per-trace SourceX tag),
  * original_n_traces / n_purged / n_traces bookkeeping,
  * the all-zero safety bypass (nothing purged),
  * a clean file is an exact pass-through (Zero-regression),
  * file-copy ops (join_profiles) stay byte-faithful to the SOURCE trace count.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import segyio

from sbp_studio.core import load_profile, load_metadata


def _write_segy(path: str, times: list, *, ns: int = 32, dt_us: int = 250) -> str:
    """Write a tiny SEG-Y where trace i has timestamp ``times[i]`` (doy,h,m,s).

    SourceX is set to the UNIQUE tag ``1000 + i`` (coord_unit=1, scalar=1) so a
    kept trace's identity is verifiable after trimming — proving the coordinate
    arrays were filtered by the exact same mask as the data matrix.
    """
    n = len(times)
    spec = segyio.spec()
    spec.sorting = None
    spec.format = 1
    spec.samples = np.arange(ns, dtype=np.float32)
    spec.tracecount = n
    with segyio.create(path, spec) as f:
        f.bin.update(hdt=dt_us, dto=dt_us)
        for i, (doy, h, m, s) in enumerate(times):
            f.header[i] = {
                segyio.TraceField.SourceX:            1000 + i,
                segyio.TraceField.SourceY:            2000 + i,
                segyio.TraceField.GroupX:             1000 + i,
                segyio.TraceField.GroupY:             2000 + i,
                segyio.TraceField.SourceGroupScalar:  1,
                segyio.TraceField.CoordinateUnits:    1,
                segyio.TraceField.SourceWaterDepth:   100 + i,
                segyio.TraceField.ElevationScalar:    1,
                segyio.TraceField.DelayRecordingTime: i,
                segyio.TraceField.YearDataRecorded:   2026,
                segyio.TraceField.DayOfYear:          doy,
                segyio.TraceField.HourOfDay:          h,
                segyio.TraceField.MinuteOfHour:       m,
                segyio.TraceField.SecondOfMinute:     s,
                segyio.TraceField.TraceNumber:        i + 1,
            }
            f.trace[i] = np.full(ns, float(i), dtype=np.float32)  # value == index
    return path


# Trace 0 unique; 1,2 duplicate 0's neighbour-change then repeat; 4 repeats 3.
# Expected kept indices: 0, 1, 3 (traces 2 and 4 are consecutive duplicates).
_TIMES = [
    (147, 5, 25, 45),   # 0 keep (first)
    (147, 5, 25, 46),   # 1 keep (differs from 0)
    (147, 5, 25, 46),   # 2 DROP (== 1)
    (147, 5, 25, 47),   # 3 keep (differs from 2)
    (147, 5, 25, 47),   # 4 DROP (== 3)
]
_KEEP_IDX = [0, 1, 3]


def test_purges_consecutive_duplicates_and_keeps_alignment(tmp_path):
    p = _write_segy(str(tmp_path / "dup.sgy"), _TIMES)
    sd = load_profile(p, load_traces=True)

    assert sd.original_n_traces == 5
    assert sd.n_traces == 3
    assert sd.n_purged == 2

    # Data matrix: each trace's sample value == its ORIGINAL index → kept set.
    assert sd.data.shape == (32, 3)
    assert list(sd.data[0, :].astype(int)) == _KEEP_IDX

    # Coordinates were filtered by the SAME mask (SourceX tag == 1000 + idx).
    assert list((sd.lons - 1000).astype(int)) == _KEEP_IDX
    assert list((sd.lats - 2000).astype(int)) == _KEEP_IDX

    # Every per-trace array has the cleaned length and stays aligned.
    assert sd.delays.shape[0] == 3 and list(sd.delays.astype(int)) == _KEEP_IDX
    assert sd.water_depth.shape[0] == 3
    assert len(sd.timestamps) == 3
    assert sd.amp_max.shape[0] == 3
    assert sd.dist_km.shape[0] == 3
    # Inspector headers: SourceX column trimmed identically.
    sx = sd.trace_headers["SourceX"]
    assert sx.shape[0] == 3 and list((sx - 1000).astype(int)) == _KEEP_IDX


def test_metadata_only_load_also_dedups(tmp_path):
    """Header-only load (no trace data) must report the same cleaned counts —
    the mask is derived from the time headers, available without trace data."""
    p = _write_segy(str(tmp_path / "dup2.sgy"), _TIMES)
    md = load_metadata(p)
    assert md.original_n_traces == 5
    assert md.n_traces == 3
    assert md.n_purged == 2
    assert md.lons.shape[0] == 3            # coords trimmed even header-only


def test_all_zero_time_headers_bypass(tmp_path):
    """A file that never populates the time words must NOT be collapsed to one
    trace — the all-zero safety bypass keeps every trace."""
    p = _write_segy(str(tmp_path / "zeros.sgy"), [(0, 0, 0, 0)] * 6)
    sd = load_profile(p, load_traces=True)
    assert sd.original_n_traces == 6
    assert sd.n_traces == 6
    assert sd.n_purged == 0
    assert sd.data.shape[1] == 6


def test_clean_file_is_exact_passthrough(tmp_path):
    """No consecutive duplicates → nothing purged, arrays unchanged."""
    times = [(147, 5, 25, s) for s in range(5)]   # strictly increasing seconds
    p = _write_segy(str(tmp_path / "clean.sgy"), times)
    sd = load_profile(p, load_traces=True)
    assert sd.n_purged == 0
    assert sd.n_traces == sd.original_n_traces == 5
    assert sd.data.shape[1] == 5


def test_join_is_byte_faithful_to_source_count(tmp_path):
    """File-copy ops size the output by the SOURCE trace count, NOT the cleaned
    in-memory count — the duplicate purge is a display/analysis feature only."""
    from sbp_studio.core import join_profiles
    from sbp_studio.core.model import ProfileChain

    p1 = _write_segy(str(tmp_path / "j1.sgy"), _TIMES)              # 5 → 3 cleaned
    sd = load_profile(p1, load_traces=False)
    assert sd.n_traces == 3 and sd.original_n_traces == 5

    out = str(tmp_path / "joined.sgy")
    join_profiles(ProfileChain([sd]), out_path=out)
    with segyio.open(out, ignore_geometry=True) as f:
        assert f.tracecount == 5            # all original traces preserved on disk
