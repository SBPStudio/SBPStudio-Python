"""
io_segy.py — SEG-Y loading and reprojection writers.

Public API
----------
load_metadata(path)                → SegyMetadata     (no trace data read)
load_profile(path, load_traces)    → SegyProfile      (traces loaded by default)
reproject_one(sd, src, dst, ...)   → str              (output path on success)
reproject_chain(ch, src, dst, ...) → str              (output path on success)
join_profiles(ch, ...)             → str              (pure copy, no CRS change)

Two-track design
----------------
reproject_one / reproject_chain dispatch to the OPTIMIZED path by default.
The reference (per-trace) path is kept as _ref_reproject_trace for tests.

Optimised reprojection (OQ-1 resolved)
---------------------------------------
_opt_reproject_coords_bulk reads ALL SourceX/Y/Scalar/Unit fields at once
using segyio.attributes, applies scalar factors and arc-second correction
vectorised, then calls Transformer.transform ONCE on the full coordinate
arrays.  For a 1347-trace file this reduces 1347 tf.transform() calls to 1.

Regression gate: allclose(opt_nxs, ref_nxs, atol=1e-10) — float64 rounding
only (documented in tests/test_regression_parallel.py).

out_sc fix (OQ-3 resolved)
--------------------------
Previous value -10_000_000 overflowed the SEG-Y 16-bit SourceGroupScalar
field. Fixed to -10_000 (maximum valid SEG-Y scalar for geographic CRS),
giving 4 decimal places ≈ 11 m accuracy — adequate for TOPAS survey data.

join_profiles fast path (OQ-1 resolved)
-----------------------------------------
join_profiles copies headers verbatim (no Transformer) and only updates
TraceNumber. Used by CLI join-chain --no-reproject or when src == dst.

CRS unit detection (OQ-2 partially resolved)
----------------------------------------------
_check_crs_units warns at load time when a projected CRS has non-metre
axis units, flagging that dist_km may be incorrect.

Behaviour preserved verbatim from the monolith
-----------------------------------------------
- Scalar factor: fac = 1/abs(sc) if sc<0, sc if sc>0, else 1.0
- Arc-sec: coord_unit==2 → divide by 3600 to get degrees
- Elevation scalar: same fac formula applied to SourceWaterDepth
- Timestamp format: "YYYY-DOYnnn HH:MM:SS" (DOY zero-padded to 3 digits)
- dist_km heuristic: geographic if coord_unit in (2,3) or range check; else
  Euclidean / 1000.0 (assumes metres — known limitation)
- delay_ms = int(delays[0])  (only first trace — known limitation)
- Reprojection header contract:
    copy full trace header (preserves DelayRecordingTime);
    overwrite ONLY SourceX/Y, GroupX/Y, SourceGroupScalar, CoordinateUnits
    (+ TraceNumber for chains/join);
    out_sc = -10_000 if dst geographic else -100  [FIXED from -10_000_000]
    new_uc = 3 (geographic) or 1 (projected);
    clip coords to ±INT32_MAX; copy bin + text[0].
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import segyio
from pyproj import CRS, Transformer

from .model import SegyMetadata, SegyProfile, ProfileChain
from .tasks import (
    SegyLoadError, CRSError, ReprojectionError, Cancelled,
    ProgressCallback, LogCallback, CancelToken,
    _noop_progress, _noop_log,
)

_INT32_MAX = 2_147_483_647

# Trace-order median-filter kernel for cleaning per-trace navigation. Large
# enough to reject multi-sample GPS spikes (frozen/jumping fixes common in
# high-latitude / Antarctic campaigns) yet small relative to a survey line.
TRACK_SMOOTH_KERNEL = 11


# ── Internal helpers ────────────────────────────────────────────────────────────

def smooth_track(lons: np.ndarray, lats: np.ndarray,
                 kernel: int = TRACK_SMOOTH_KERNEL) -> tuple:
    """Return median-filtered (lons, lats) in trace order — a cleaned DISPLAY
    track that rejects GPS spikes/outliers so the navigation map draws a smooth
    path and the along-track distance axis is well-behaved.

    This is a *display/geometry* track only: the caller keeps the raw recorded
    ``lons``/``lats`` for FIX and geometry exports (the smoothed values must
    never replace the authoritative recorded navigation).

    Uses ``scipy.ndimage.median_filter`` with ``mode="nearest"`` rather than
    ``scipy.signal.medfilt``: medfilt ZERO-pads its borders, which for absolute
    coordinates far from the origin (e.g. lat ≈ −70°) drags the first/last
    kernel//2 fixes toward 0 and wrecks the endpoints. Edge replication keeps the
    Start/End-Of-Line points exactly on the real track.

    Note: a median filter removes outliers; it does NOT *guarantee* strict
    monotonicity (a genuinely stationary vessel still yields repeated points).
    The cumulative distance built from these coordinates remains monotonically
    non-decreasing by construction (cumsum of non-negative steps)."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    n = lons.size
    if n < 3:
        return lons.copy(), lats.copy()
    # Clamp the kernel to an odd value that fits the trace count.
    k = min(int(kernel), n if n % 2 else n - 1)
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return lons.copy(), lats.copy()
    from scipy.ndimage import median_filter
    return (median_filter(lons, size=k, mode="nearest"),
            median_filter(lats, size=k, mode="nearest"))


def _to_signed(values, bits: int):
    """Reinterpret raw SEG-Y header values as two's-complement SIGNED integers of
    the given width. The coordinate fields are signed by the standard — the
    Coordinate Scalar (bytes 71-72) is 16-bit signed (struct ``>h``) and Source
    X/Y (bytes 73-76 / 77-80) are 32-bit signed (struct ``>i``). Forcing the sign
    here guards against a writer (or reader build) that emitted them UNSIGNED,
    which would otherwise turn large negatives — e.g. Antarctic lon/lat or a
    negative scalar — into bogus huge positives (the "collapse to 0,0" symptom).

    Widen to int64 first so any unsigned magnitude is preserved, then narrow to
    the signed width so out-of-range values wrap via two's complement."""
    dt = np.int16 if bits == 16 else np.int32
    return np.asarray(values, dtype=np.int64).astype(dt)


def _scalar_fac(sc: int) -> float:
    if sc < 0:
        return 1.0 / abs(sc)
    if sc > 0:
        return float(sc)
    return 1.0


def _decode_text_header(raw) -> str:
    """Decode the 3200-byte textual header to clean ASCII, wrapped to the
    standard 40 lines × 80 chars. Auto-detects EBCDIC (cp037) vs ASCII by
    printable-character ratio (legacy TOPAS files are often EBCDIC)."""
    if raw is None:
        return ""
    data = bytes(raw)
    if not data:
        return ""

    def _printable(s: str) -> float:
        if not s:
            return 0.0
        ok = sum(1 for ch in s if ch in "\r\n\t" or 32 <= ord(ch) < 127)
        return ok / len(s)

    ascii_txt = data.decode("ascii", errors="replace")
    try:
        ebcdic_txt = data.decode("cp037", errors="replace")
    except Exception:
        ebcdic_txt = ascii_txt
    txt = ascii_txt if _printable(ascii_txt) >= _printable(ebcdic_txt) else ebcdic_txt
    txt = txt.replace("\x00", " ")
    # Standard layout: 40 cards of 80 columns. Wrap accordingly and right-trim.
    lines = [txt[i:i + 80].rstrip() for i in range(0, min(len(txt), 3200), 80)]
    return "\n".join(lines)


def _extract_trace_headers(f: "segyio.SegyFile") -> dict:
    """Return an ordered ``{field_name: np.ndarray}`` of RAW values for EVERY
    standard SEG-Y trace header word (all 91), in 240-byte header order.

    Dynamic — driven by segyio's full field registry (``segyio.tracefield.keys``,
    ``{name: byte_offset}``) rather than a hardcoded subset, so the inspector
    exposes the complete header. This is what lets us hunt down non-standard /
    proprietary coordinate storage: acquisition systems like TOPAS sometimes
    write the true lat/lon into CDP_X/CDP_Y or the UnassignedInt1/2 words (bytes
    233/237) instead of the standard Source X/Y (73-80). Values are kept RAW (as
    stored) so byte content is visible verbatim.

    Header-attribute reads are vectorised and cheap even for 50k+ traces (no
    trace DATA is touched)."""
    out: dict = {}
    for name, offset in sorted(segyio.tracefield.keys.items(), key=lambda kv: kv[1]):
        try:
            out[name] = np.asarray(f.attributes(offset)[:])
        except Exception:
            continue        # skip any field segyio can't read on this file
    return out


def _dist_km(lons: np.ndarray, lats: np.ndarray, coord_unit: int) -> np.ndarray:
    """
    Compute cumulative along-track distance in km.

    Uses geographic (haversine-approximation) formula when coord_unit in (2,3)
    or when coordinates appear to be degrees. Falls back to Euclidean/1000
    otherwise (assumes metres — known limitation; OQ-2).
    """
    dlat = np.diff(lats)
    dlon = np.diff(lons)
    _is_geo = coord_unit in (2, 3) or (
        -180 <= float(lons[0]) <= 180 and -90 <= float(lats[0]) <= 90)
    if _is_geo:
        lat_m = np.mean(lats)
        d = np.sqrt((dlat * 111.32)**2 +
                    (dlon * 111.32 * np.cos(np.radians(lat_m)))**2)
    else:
        d = np.sqrt(dlat**2 + dlon**2) / 1000.0
    return np.concatenate([[0.0], np.cumsum(d)])


def _detect_crs(coord_unit: int, lons: np.ndarray) -> tuple:
    if coord_unit == 2:
        if -180 <= lons[0] <= 180:
            return "EPSG:4326", ["✔ Arc-seconds → WGS84 detectado"]
    elif coord_unit == 3:
        if -180 <= lons[0] <= 180:
            return "EPSG:4326", ["✔ Grados decimales → WGS84 detectado"]
    return None, ["⚠ No detectado automáticamente"]


def _check_crs_units(detected_crs: Optional[str], coord_unit: int) -> list:
    """
    OQ-2 partial fix: warn if a projected CRS has non-metre linear units.
    dist_km silently assumes metres when coord_unit not in (2, 3).

    Returns a list of warning strings (empty if no issue or exception).
    """
    if not detected_crs or coord_unit in (2, 3):
        return []
    try:
        crs_obj = CRS.from_user_input(detected_crs)
        if crs_obj.is_geographic:
            return []
        axis_info = crs_obj.axis_info
        if axis_info:
            unit = axis_info[0].unit_name.lower()
            if "metre" not in unit and "meter" not in unit:
                return [
                    f"⚠ Projected axis unit '{axis_info[0].unit_name}' is not metres "
                    "— dist_km may be incorrect (OQ-2)"
                ]
    except Exception:
        pass
    return []


def _populate_profile_from_file(prof: SegyProfile, f: "segyio.SegyFile",
                                load_traces: bool = True) -> None:
    """
    Fill a SegyProfile from an already-open segyio file handle.
    Shared by load_metadata and load_profile.
    """
    prof.n_traces = f.tracecount
    prof.ns       = f.samples.size
    prof.dt_us    = int(f.bin[segyio.BinField.Interval])

    h0 = f.header[0]
    # Coordinate + elevation scalars: 16-bit SIGNED (>h). Force the sign so a
    # negative scalar (the common case, meaning "divide by |scalar|") survives.
    prof.scalar_coord = int(_to_signed(h0[segyio.TraceField.SourceGroupScalar] or 0, 16))
    prof.scalar_elev  = int(_to_signed(h0[segyio.TraceField.ElevationScalar]   or 0, 16))
    prof.coord_unit   = int(h0[segyio.TraceField.CoordinateUnits]   or 0)

    _fields = [
        segyio.TraceField.DelayRecordingTime,
        segyio.TraceField.SourceX,
        segyio.TraceField.SourceY,
        segyio.TraceField.SourceWaterDepth,
        segyio.TraceField.YearDataRecorded,
        segyio.TraceField.DayOfYear,
        segyio.TraceField.HourOfDay,
        segyio.TraceField.MinuteOfHour,
        segyio.TraceField.SecondOfMinute,
    ]
    _hdr = {fld: np.asarray(f.attributes(fld)[:]) for fld in _fields}

    delays_raw     = _hdr[segyio.TraceField.DelayRecordingTime]
    prof.delays    = delays_raw
    prof.min_delay = float(np.min(delays_raw))
    prof.max_delay = float(np.max(delays_raw))
    prof.delay_ms  = int(delays_raw[0])  # known limitation: trace[0] only

    # Source X/Y: 32-bit SIGNED (>i), forced signed so negative Antarctic eastings
    # /northings don't wrap to huge positives. Standard SEG-Y scaling (via
    # _scalar_fac): val * scalar if scalar > 0, val / |scalar| if scalar < 0.
    sc  = prof.scalar_coord
    fac = _scalar_fac(sc)
    sxs  = _to_signed(_hdr[segyio.TraceField.SourceX], 32).astype(float) * fac
    sys_ = _to_signed(_hdr[segyio.TraceField.SourceY], 32).astype(float) * fac

    if prof.coord_unit == 2:   # arc-seconds → degrees
        prof.lons = sxs  / 3600.0
        prof.lats = sys_ / 3600.0
    else:
        prof.lons = sxs
        prof.lats = sys_

    # Cleaned DISPLAY track: median-filtered in trace order to reject GPS
    # spikes/outliers. The raw prof.lons/prof.lats above are PRESERVED for FIX
    # and geometry exports (authoritative recorded navigation); the smoothed
    # pair drives the navigation map and the along-track distance axis only.
    prof.track_lons, prof.track_lats = smooth_track(prof.lons, prof.lats)

    es   = prof.scalar_elev
    efac = _scalar_fac(es)
    prof.water_depth = _hdr[segyio.TraceField.SourceWaterDepth].astype(float) * efac

    yrs  = _hdr[segyio.TraceField.YearDataRecorded]
    doys = _hdr[segyio.TraceField.DayOfYear]
    hrs  = _hdr[segyio.TraceField.HourOfDay]
    mins = _hdr[segyio.TraceField.MinuteOfHour]
    secs = _hdr[segyio.TraceField.SecondOfMinute]

    prof.timestamps = [
        f"{int(y)}-DOY{int(d):03d} {int(h):02d}:{int(m):02d}:{int(s):02d}"
        for y, d, h, m, s in zip(yrs, doys, hrs, mins, secs)
    ]

    if load_traces:
        prof.data     = f.trace.raw[:].T.astype(np.float32)
        prof.amp_max  = np.max(np.abs(prof.data), axis=0)
        prof.clip_p99 = float(np.percentile(np.abs(prof.data), 99))
    else:
        prof.data    = None
        prof.amp_max = None
        prof.clip_p99 = None

    prof.dur_ms  = prof.ns * prof.dt_us / 1000.0
    prof.t_ms    = prof.delay_ms + np.arange(prof.ns) * prof.dt_us / 1000.0
    # Distance axis from the CLEANED track so it is smooth and well-behaved for
    # the seismic X-axis and the map↔profile sync (raw coords stay export-only).
    prof.dist_km = _dist_km(prof.track_lons, prof.track_lats, prof.coord_unit)
    prof.total_km = float(prof.dist_km[-1])

    # Header Inspector data — textual header + RAW per-trace header fields.
    # Cheap and available even header-only (no trace DATA needed).
    prof.text_header   = _decode_text_header(f.text[0])
    prof.trace_headers = _extract_trace_headers(f)
    prof.detected_crs, prof.crs_notes = _detect_crs(prof.coord_unit, prof.lons)

    # OQ-2: warn if projected CRS axis unit is not metres
    unit_warnings = _check_crs_units(prof.detected_crs, prof.coord_unit)
    prof.crs_notes.extend(unit_warnings)


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_metadata(path: str) -> SegyMetadata:
    """
    Load header-only metadata from a SEG-Y file WITHOUT reading trace data.
    Fast even for large files: reads only binary header + per-trace fields.
    """
    prof = SegyProfile(path)
    try:
        with segyio.open(path, ignore_geometry=True) as f:
            _populate_profile_from_file(prof, f, load_traces=False)
    except Exception as exc:
        prof.error = str(exc)
    return prof.to_metadata()


def load_profile(path: str, load_traces: bool = True) -> SegyProfile:
    """
    Load a SEG-Y file into a SegyProfile.
    On error, .error is set and other fields remain at defaults. Never raises.
    """
    prof = SegyProfile(path)
    try:
        with segyio.open(path, ignore_geometry=True) as f:
            _populate_profile_from_file(prof, f, load_traces=load_traces)
    except Exception as exc:
        prof.error = str(exc)
    return prof


# ── Reprojection helpers ────────────────────────────────────────────────────────

def _safe_coord(v: float, div: float) -> int:
    """Scale and clip a coordinate to fit in a SEG-Y int32 header field."""
    return int(np.clip(round(v / div), -_INT32_MAX, _INT32_MAX))


def _build_transformer(src_str: str, dst_str: str) -> tuple:
    """
    Build a pyproj Transformer and return (tf, out_sc, new_uc, dst_geo, div).

    out_sc values (OQ-3 fix applied):
      Geographic dst: -10_000   (4 decimal places ≈ 11 m — fits in 16-bit)
      Projected dst : -100      (2 decimal places — cm precision for metres)

    Previous value -10_000_000 overflowed the 16-bit SourceGroupScalar field,
    causing segyio to store truncated value 27008 and making reloaded
    coordinate scaling completely wrong.
    """
    try:
        src_crs = CRS.from_user_input(src_str)
        dst_crs = CRS.from_user_input(dst_str)
    except Exception as exc:
        raise CRSError(f"Invalid CRS: {exc}") from exc

    dst_geo = dst_crs.is_geographic
    out_sc  = -10_000 if dst_geo else -100   # FIXED: was -10_000_000
    new_uc  = 3 if dst_geo else 1
    div     = (1.0 / abs(out_sc)) if out_sc < 0 else float(out_sc)
    tf      = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tf, out_sc, new_uc, dst_geo, div


# ── Reference reprojection (oracle — one tf.transform call per trace) ──────────

def _ref_reproject_trace(h: dict, tf: Transformer, unit_hint: int,
                         out_sc: int, new_uc: int, div: float) -> tuple:
    """
    Reference per-trace coordinate transform.
    Path: REFERENCE. Used by regression tests as the ground truth.
    Returns (nx, ny) after scalar, arc-sec correction, and transform.
    """
    sc  = int(h[segyio.TraceField.SourceGroupScalar])
    uc  = int(h[segyio.TraceField.CoordinateUnits]) or unit_hint
    fac = _scalar_fac(sc)
    sx  = int(h[segyio.TraceField.SourceX]) * fac
    sy  = int(h[segyio.TraceField.SourceY]) * fac
    if uc == 2:
        sx /= 3600.0
        sy /= 3600.0
    return tf.transform(sx, sy)


# ── Optimised reprojection (one bulk tf.transform call for all traces) ─────────

def _opt_reproject_coords_bulk(src_f: "segyio.SegyFile",
                               tf: Transformer,
                               unit_hint: int) -> tuple:
    """
    Read ALL SourceX/Y + scalars from the file at once, apply scalar factors
    and arc-second corrections vectorised, then call tf.transform ONCE.

    Path: OPTIMIZED.
    Regression gate: allclose(opt, ref, atol=1e-10) — float64 rounding only.
    Documented in tests/test_regression_parallel.py::test_reproject_vectorised.

    Speedup: ~50× vs per-trace for pyproj overhead (dominated by Python
    call overhead, not computation). For 1347 traces: 1347 calls → 1 call.

    Returns
    -------
    (nxs, nys) : float64 arrays, length = f.tracecount
    """
    n = src_f.tracecount

    raw_scalars = np.asarray(
        src_f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=float)
    raw_units   = np.asarray(
        src_f.attributes(segyio.TraceField.CoordinateUnits)[:], dtype=int)
    raw_sxs     = np.asarray(
        src_f.attributes(segyio.TraceField.SourceX)[:], dtype=float)
    raw_sys     = np.asarray(
        src_f.attributes(segyio.TraceField.SourceY)[:], dtype=float)

    # Vectorised _scalar_fac: clip negative abs to ≥1 to avoid div-by-zero;
    # the np.where on scalars<0 branch is only used where scalars<0, but
    # numpy evaluates all branches, so we need the clip for safety.
    abs_sc = np.abs(raw_scalars).clip(1)
    facs   = np.where(raw_scalars < 0, 1.0 / abs_sc,
                      np.where(raw_scalars > 0, raw_scalars, 1.0))
    sxs = raw_sxs * facs
    sys_ = raw_sys * facs

    # Vectorised arc-second correction:
    # effective_unit = unit if unit != 0 else unit_hint
    eff_units = np.where(raw_units == 0, unit_hint, raw_units)
    arc_mask  = eff_units == 2
    sxs[arc_mask]  /= 3600.0
    sys_[arc_mask] /= 3600.0

    # Single bulk transform — the payoff
    nxs, nys = tf.transform(sxs, sys_)
    return nxs, nys


# ── Public reprojection: dispatches to optimised path ──────────────────────────

def reproject_one(
    sd: SegyProfile,
    src_str: str,
    dst_str: str,
    unit_hint: int = 2,
    log: LogCallback = _noop_log,
    progress: ProgressCallback = _noop_progress,
    cancel: Optional[CancelToken] = None,
    out_path: Optional[str] = None,
) -> str:
    """
    Reproject a single SEG-Y profile to a new CRS.

    Uses the optimised (vectorised) coordinate transform path.
    The reference trace-by-trace path is kept as _ref_reproject_trace.

    Header contract
    ---------------
    - Full trace header copied first (preserves DelayRecordingTime).
    - ONLY SourceX/Y, GroupX/Y, SourceGroupScalar, CoordinateUnits overwritten.
    - out_sc = -10_000 for geographic dst  [FIXED: was -10_000_000]
               -100    for projected dst
    - new_uc = 3 (geographic) or 1 (projected).
    - Coordinates clipped to ±INT32_MAX.
    - bin and text[0] copied from source.
    """
    if cancel is None:
        cancel = CancelToken.never()

    log("\n" + "-" * 55)
    log(f"Procesando : {sd.name}")
    log(f"CRS origen : {src_str}")
    log(f"CRS destino: {dst_str}")

    tf, out_sc, new_uc, _dst_geo, div = _build_transformer(src_str, dst_str)

    p       = Path(sd.path)
    # Caller-chosen destination (GUI file dialog) wins; else the legacy default
    # of a _REPROY-suffixed sibling beside the source file.
    outpath = out_path or str(p.with_name(p.stem + "_REPROY" + p.suffix))
    log(f"Salida     : {Path(outpath).name}")

    try:
        with segyio.open(sd.path, ignore_geometry=True) as src:
            spec = segyio.tools.metadata(src)

            # Optimised: compute ALL transformed coords in one bulk call
            progress(0.05, "computing coordinates…")
            nxs, nys = _opt_reproject_coords_bulk(src, tf, unit_hint)

            with segyio.create(outpath, spec) as dst:
                dst.bin    = src.bin
                dst.text[0] = src.text[0]
                for i in range(sd.n_traces):
                    cancel.check()
                    if i % 500 == 0:
                        log(f"  traza {i+1}/{sd.n_traces}…")
                        progress(0.1 + 0.9 * i / sd.n_traces,
                                 f"traza {i+1}/{sd.n_traces}")
                    dst.header[i] = src.header[i]
                    dst.header[i].update({
                        segyio.TraceField.SourceX:           _safe_coord(nxs[i], div),
                        segyio.TraceField.SourceY:           _safe_coord(nys[i], div),
                        segyio.TraceField.GroupX:            _safe_coord(nxs[i], div),
                        segyio.TraceField.GroupY:            _safe_coord(nys[i], div),
                        segyio.TraceField.SourceGroupScalar: out_sc,
                        segyio.TraceField.CoordinateUnits:   new_uc,
                    })
                    dst.trace[i] = src.trace[i]

        progress(1.0, "done")
        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except (Cancelled, CRSError):
        _try_delete(outpath)
        raise
    except Exception as exc:
        import traceback
        log(f"✘ Error: {exc}\n{traceback.format_exc()}")
        _try_delete(outpath)
        raise ReprojectionError(str(exc)) from exc


def reproject_chain(
    ch: ProfileChain,
    src_str: str,
    dst_str: str,
    unit_hint: int = 2,
    log: LogCallback = _noop_log,
    progress: ProgressCallback = _noop_progress,
    cancel: Optional[CancelToken] = None,
    out_path: Optional[str] = None,
) -> str:
    """
    Reproject and JOIN a ProfileChain into a single SEG-Y file.

    Uses the optimised (vectorised) coordinate path: per-profile bulk
    transform, then per-trace header write.

    Header contract: same as reproject_one, plus TraceNumber = global index.
    """
    if cancel is None:
        cancel = CancelToken.never()

    log("\n" + "-" * 55)
    log(f"Procesando CADENA: {ch.label}")
    log(f"CRS origen : {src_str}")
    log(f"CRS destino: {dst_str}")

    tf, out_sc, new_uc, _dst_geo, div = _build_transformer(src_str, dst_str)

    p0      = Path(ch.profiles[0].path)
    stem    = f"{p0.stem}_a_{Path(ch.profiles[-1].path).stem}_UNIDO_REPROY"
    # Caller-chosen destination wins; else the legacy default beside profile[0].
    outpath = out_path or str(p0.with_name(stem + p0.suffix))
    log(f"Salida     : {Path(outpath).name}")

    try:
        with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
            spec = segyio.tools.metadata(src0)
            spec.tracecount = ch.n_traces

        with segyio.create(outpath, spec) as dst:
            with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
                dst.bin    = src0.bin
                dst.text[0] = src0.text[0]

            global_idx = 0
            for p_idx, sd in enumerate(ch.profiles):
                log(f"  Integrando {p_idx+1}/{len(ch.profiles)}: {sd.name}…")
                with segyio.open(sd.path, ignore_geometry=True) as src:
                    # Bulk transform for this profile
                    nxs, nys = _opt_reproject_coords_bulk(src, tf, unit_hint)

                    for i in range(sd.n_traces):
                        cancel.check()
                        if global_idx % 1000 == 0:
                            log(f"    traza {global_idx+1}/{ch.n_traces}…")
                            progress(global_idx / ch.n_traces,
                                     f"traza {global_idx+1}/{ch.n_traces}")
                        dst.header[global_idx] = src.header[i]
                        dst.header[global_idx].update({
                            segyio.TraceField.SourceX:           _safe_coord(nxs[i], div),
                            segyio.TraceField.SourceY:           _safe_coord(nys[i], div),
                            segyio.TraceField.GroupX:            _safe_coord(nxs[i], div),
                            segyio.TraceField.GroupY:            _safe_coord(nys[i], div),
                            segyio.TraceField.SourceGroupScalar: out_sc,
                            segyio.TraceField.CoordinateUnits:   new_uc,
                            segyio.TraceField.TraceNumber:       global_idx + 1,
                        })
                        dst.trace[global_idx] = src.trace[i]
                        global_idx += 1

        progress(1.0, "done")
        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except (Cancelled, CRSError):
        _try_delete(outpath)
        raise
    except Exception as exc:
        import traceback
        log(f"✘ Error: {exc}\n{traceback.format_exc()}")
        _try_delete(outpath)
        raise ReprojectionError(str(exc)) from exc


# ── Pure join (no CRS transformation) ─────────────────────────────────────────

def join_profiles(
    ch: ProfileChain,
    out_path: Optional[str] = None,
    log: LogCallback = _noop_log,
    progress: ProgressCallback = _noop_progress,
    cancel: Optional[CancelToken] = None,
) -> str:
    """
    Join a ProfileChain into a single SEG-Y WITHOUT any coordinate transformation.

    All trace headers are copied verbatim (SourceX/Y, SourceGroupScalar,
    CoordinateUnits, DelayRecordingTime, etc. all preserved unchanged).
    Only TraceNumber is updated to the global sequential index (1-based).

    This is the fast path for join-chain --no-reproject or when src == dst.
    ~3× faster than reproject_chain with an identity transform because it
    avoids all pyproj / Transformer overhead.

    Parameters
    ----------
    ch       : ProfileChain (profiles in order)
    out_path : output file path; auto-generated if None
    log      : log callback (str → None)
    progress : progress callback (float, str → None)
    cancel   : cancellation token

    Returns
    -------
    Output file path on success.

    Raises
    ------
    ReprojectionError on failure (partial output deleted).
    Cancelled         if the cancel token fires.
    """
    if cancel is None:
        cancel = CancelToken.never()

    log("\n" + "-" * 55)
    log(f"Uniendo cadena (sin reproyectar): {ch.label}")

    p0 = Path(ch.profiles[0].path)
    if out_path:
        outpath = out_path
    else:
        stem    = f"{p0.stem}_a_{Path(ch.profiles[-1].path).stem}_UNIDO"
        outpath = str(p0.with_name(stem + p0.suffix))
    log(f"Salida     : {Path(outpath).name}")

    try:
        with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
            spec = segyio.tools.metadata(src0)
            spec.tracecount = ch.n_traces

        with segyio.create(outpath, spec) as dst:
            with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
                dst.bin    = src0.bin
                dst.text[0] = src0.text[0]

            global_idx = 0
            for p_idx, sd in enumerate(ch.profiles):
                log(f"  Copiando {p_idx+1}/{len(ch.profiles)}: {sd.name}…")
                with segyio.open(sd.path, ignore_geometry=True) as src:
                    for i in range(sd.n_traces):
                        cancel.check()
                        if global_idx % 1000 == 0:
                            progress(global_idx / ch.n_traces,
                                     f"traza {global_idx+1}/{ch.n_traces}")
                        # Copy header verbatim; only update TraceNumber
                        dst.header[global_idx] = src.header[i]
                        dst.header[global_idx].update({
                            segyio.TraceField.TraceNumber: global_idx + 1,
                        })
                        dst.trace[global_idx] = src.trace[i]
                        global_idx += 1

        progress(1.0, "done")
        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except Cancelled:
        _try_delete(outpath)
        raise
    except Exception as exc:
        import traceback
        log(f"✘ Error: {exc}\n{traceback.format_exc()}")
        _try_delete(outpath)
        raise ReprojectionError(str(exc)) from exc


def _try_delete(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
