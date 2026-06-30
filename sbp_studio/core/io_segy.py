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
giving 4 decimal places ≈ 11 m accuracy — adequate for SBP survey data.

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
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import segyio
from pyproj import CRS, Geod, Transformer

from .coordinates import resolve_crs
from .logger import get_logger
from .model import SegyMetadata, SegyProfile, ProfileChain
from .processing import apply_dc_removal
from .tasks import (
    SegyLoadError, CRSError, ReprojectionError, Cancelled,
    ProgressCallback, LogCallback, CancelToken,
    _noop_progress, _noop_log,
)

_LOG = get_logger("io_segy")

_INT32_MAX  = 2_147_483_647
_WGS84      = Geod(ellps="WGS84")   # #12: re-used for every geodesic dist_km call

# Trace-order median-filter kernel for cleaning per-trace navigation. Large
# enough to reject multi-sample GPS spikes (frozen/jumping fixes common in
# high-latitude / Antarctic campaigns) yet small relative to a survey line.
TRACK_SMOOTH_KERNEL = 11

# Active-depth detection (SegyProfile.active_lo/active_ns): many marine
# surveys record every shot with a FIXED listening window sized for the
# deepest expected water depth across the whole campaign, so a file covering
# a shallower stretch is mostly DEAD samples — trailing zeros below the real
# signal (active_ns, the original bottom-only detection) AND, just as often,
# LEADING zeros above it: a high-res SBP system starts recording at the ping
# but the sub-bottom reflectors of interest don't arrive until after the
# water-column travel time, which can be thousands of samples in deep water.
# See model.SegyProfile.active_lo/active_ns's docstring. A row counts as
# "live" once its max amplitude across all traces exceeds this fraction of
# the profile's own clip_p99 (a per-file noise/scale reference, never an
# absolute constant).
ACTIVE_DEPTH_REL_THRESH = 0.01
# Safety margin (samples) kept beyond the first/last detected live row on
# EACH side — guards against a slightly conservative detection and gives
# windowed filters with a small halo room past the cutoff.
ACTIVE_DEPTH_MARGIN_SAMPLES = 50
# Minimum position displacement (decimal degrees) treated as "real movement"
# in the dual-gate dedup (#20). ~9e-6° ≈ 1 m at the equator.
_DEDUP_POS_TOL_DEG = 9e-6
# CV of inter-trace spacing above which a reprojection/join op warns that
# true spatial regularization is recommended (#19).
_SPACING_CV_THRESH = 0.10


def _detect_active_band(data: np.ndarray, clip_p99: float) -> Tuple[int, int]:
    """``(lo, hi)`` — the first and last+1 row (depth) with amplitude above
    ``ACTIVE_DEPTH_REL_THRESH`` of ``clip_p99``, each padded by
    ``ACTIVE_DEPTH_MARGIN_SAMPLES`` and clamped to ``[0, ns]`` — see
    SegyProfile.active_lo/active_ns. Falls back to the full ``(0, ns)`` band
    if every row is at/under the threshold (a blank file, or pure noise with
    no real scale) — never returns something that would cut off real
    signal."""
    ns = data.shape[0]
    if clip_p99 <= 0:
        return 0, ns
    threshold = ACTIVE_DEPTH_REL_THRESH * clip_p99
    row_max = np.max(np.abs(data), axis=1)
    live_rows = np.flatnonzero(row_max > threshold)
    if live_rows.size == 0:
        return 0, ns
    lo = max(0, int(live_rows[0]) - ACTIVE_DEPTH_MARGIN_SAMPLES)
    hi = min(ns, int(live_rows[-1]) + 1 + ACTIVE_DEPTH_MARGIN_SAMPLES)
    return lo, hi


def _detect_active_ns(data: np.ndarray, clip_p99: float) -> int:
    """Backward-compatible bottom-only view of :func:`_detect_active_band` —
    see SegyProfile.active_ns."""
    return _detect_active_band(data, clip_p99)[1]


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


def _timestamp_dedup_mask(doy, hod, moh, som, sx, sy,
                          scalar: int = 0) -> Optional[np.ndarray]:
    """Return a boolean keep-mask (length n_traces) that drops consecutive traces
    sharing an identical acquisition timestamp AND position, or ``None`` when no
    cleaning is warranted.

    Dual-gate (#20): a trace is kept when EITHER gate fires —
      • Time gate:     any of the 4 SEG-Y time fields changed (Δt ≠ 0)
      • Position gate: raw SourceX/Y moved by more than ``_DEDUP_POS_TOL_DEG``
                       (converted to raw integer units via the coordinate scalar).

    Separating the two gates prevents over-aggressive purging of high-ping-rate
    files where a GPS with < 1 Hz update rate echoes the same integer coordinate
    for two consecutive 0.5 s pings: the time gate fires (different second or
    minute) and saves the trace even when the position gate cannot.

    Safety bypass: if ALL time fields across ALL traces are zero (a file that
    does not populate the time words) ``None`` is returned so nothing is purged.
    ``None`` is also returned when no consecutive duplicates are found (the common
    case), so the hot path stays a pass-through with zero extra cost."""
    tvec = np.stack([np.asarray(doy), np.asarray(hod),
                     np.asarray(moh), np.asarray(som),
                     np.asarray(sx),  np.asarray(sy)], axis=1).astype(np.int64)
    if tvec.shape[0] < 2 or not np.any(tvec[:, :4]):
        return None                      # <2 traces, or all-zero time headers → bypass

    # Time gate: any second/minute/hour/DOY field changed from the previous trace
    t_diff    = np.diff(tvec[:, :4], axis=0)
    t_changed = np.any(t_diff != 0, axis=1)

    # Position gate: raw coordinate integer moved by more than the 1-metre
    # tolerance (converted from degrees using the coordinate scalar, floor at 0
    # so any integer change counts when the scale is too coarse to represent 1 m).
    fac     = _scalar_fac(scalar) if scalar else 1.0
    tol_raw = max(0, int(_DEDUP_POS_TOL_DEG / fac))
    pos_diff  = np.diff(tvec[:, 4:], axis=0)
    p_changed = np.any(np.abs(pos_diff) > tol_raw, axis=1)

    changed = t_changed | p_changed
    keep = np.concatenate(([True], changed))                 # always keep trace[0]
    if keep.all():
        return None                      # no consecutive duplicates → pass-through
    return keep


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


# Codec map for the textual header. "ascii"/"ebcdic" are the two SEG-Y-relevant
# encodings the heuristic chooses between; "latin-1" is a never-fails fallback
# for extended-ASCII files (every byte 0x00-0xFF maps to a code point) exposed
# as a manual override in the GUI.
_TEXT_CODECS: Dict[str, str] = {
    "ascii":   "ascii",
    "ebcdic":  "cp037",
    "latin-1": "latin-1",
}


def _normalize_raw_text_bytes(raw) -> bytes:
    """Null-pad fix for the Kongsberg/TOPAS (and similar marine SBP) quirk:
    these acquisition systems pad the 3200-byte textual header with NUL bytes
    (``0x00``) instead of the SEG-Y-standard ASCII space (``0x20``). Swap them
    at the BYTE level *before* decoding so that (a) the encoding heuristic
    isn't skewed by a sea of non-printable nulls and (b) the decoded text
    shows clean trailing spaces under every codec. Idempotent and read-only."""
    if raw is None:
        return b""
    return bytes(raw).replace(b"\x00", b"\x20")


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for ch in s if ch in "\r\n\t" or 32 <= ord(ch) < 127)
    return ok / len(s)


def detect_text_header_encoding(raw) -> str:
    """Guess whether a raw 3200-byte textual-header block is ``"ascii"`` or
    ``"ebcdic"`` by comparing printable-character ratios under each codec.
    Many modern SBP acquisition systems violate the SEG-Y standard and write
    this block as plain ASCII rather than EBCDIC, so this is a heuristic, not
    a guarantee — the GUI exposes a manual override toggle (incl. a Latin-1
    fallback) for exactly that reason. Null padding is normalised to spaces
    first so the ratios reflect real content, not padding."""
    data = _normalize_raw_text_bytes(raw)
    if not data:
        return "ascii"
    ascii_txt = data.decode("ascii", errors="replace")
    try:
        ebcdic_txt = data.decode("cp037", errors="replace")
    except Exception:
        ebcdic_txt = ascii_txt
    return "ascii" if _printable_ratio(ascii_txt) >= _printable_ratio(ebcdic_txt) else "ebcdic"


def decode_text_header(raw, encoding: str = "auto") -> str:
    """Decode the 3200-byte textual header to clean text, wrapped to the
    standard 40 lines x 80 chars.

    *encoding* is one of ``"auto"`` (printable-ratio heuristic — the
    historical default), ``"ascii"``, ``"ebcdic"`` (cp037), or ``"latin-1"``
    (never-fails extended-ASCII fallback). NUL padding is converted to spaces
    at the byte level before decoding (the Kongsberg/TOPAS quirk). Read-only
    and side-effect-free: callers can re-decode the same raw bytes under a
    different encoding as many times as they like with zero file-corruption
    risk, since nothing is written.
    """
    data = _normalize_raw_text_bytes(raw)
    if not data:
        return ""

    enc = encoding if encoding in _TEXT_CODECS else detect_text_header_encoding(data)
    txt = data.decode(_TEXT_CODECS[enc], errors="replace")
    # Standard layout: 40 cards of 80 columns. Wrap accordingly and right-trim.
    lines = [txt[i:i + 80].rstrip() for i in range(0, min(len(txt), 3200), 80)]
    return "\n".join(lines)


def read_raw_text_header(path: str) -> bytes:
    """Read-only fetch of the TRUE raw, undecoded 3200-byte textual-header
    block — bytes 0-3199 of the file, read directly with plain file I/O.

    Deliberately does NOT use segyio's ``f.text[0]``: segyio unconditionally
    runs an internal EBCDIC-to-ASCII conversion table on the textual header,
    on the assumption that every SEG-Y file is standard-compliant EBCDIC.
    Many marine SBP acquisition systems (Kongsberg TOPAS and others) instead
    write this block as plain ASCII — running already-ASCII bytes through
    segyio's EBCDIC table double-translates them into garbage (confirmed by
    forensic byte comparison: ``segyio_text != raw_file_bytes`` for ANT26/
    L001A/MCS7 sample files, all of which are clean ASCII with zero NUL
    padding). Reading the bytes ourselves sidesteps that conversion entirely
    so our own ``decode_text_header``/``detect_text_header_encoding`` can
    choose ASCII vs EBCDIC (cp037) vs Latin-1 correctly, whatever the file
    actually contains.
    """
    with open(path, "rb") as fh:
        return fh.read(3200)


def _extract_trace_headers(f: "segyio.SegyFile") -> dict:
    """Return an ordered ``{field_name: np.ndarray}`` of RAW values for EVERY
    standard SEG-Y trace header word (all 91), in 240-byte header order.

    Dynamic — driven by segyio's full field registry (``segyio.tracefield.keys``,
    ``{name: byte_offset}``) rather than a hardcoded subset, so the inspector
    exposes the complete header. This is what lets us hunt down non-standard /
    proprietary coordinate storage: acquisition systems like SBP sometimes
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


def _dist_km(lons: np.ndarray, lats: np.ndarray, coord_unit: int,
             meters_per_unit: float = 1.0) -> np.ndarray:
    """
    Compute cumulative along-track distance in km.

    Geographic path (coord_unit 2 or 3, or values within degree range): uses
    pyproj.Geod.inv for true WGS-84 geodesic distances — replaces the former
    flat-Earth cosine approximation (#12). Projected path: Euclidean, scaled by
    *meters_per_unit* (1.0 for metres, 0.3048 for feet — #13).
    """
    _is_geo = coord_unit in (2, 3) or (
        -180 <= float(np.nanmedian(lons)) <= 180 and
        -90  <= float(np.nanmedian(lats)) <= 90)
    if _is_geo:
        if lons.size < 2:
            return np.zeros(lons.size)
        # _WGS84.inv returns (fwd_az, back_az, dist_m); vectorised over all gaps.
        _, _, dist_m = _WGS84.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        d = np.abs(dist_m) / 1000.0
    else:
        dlat = np.diff(lats)
        dlon = np.diff(lons)
        d = np.sqrt(dlat**2 + dlon**2) * meters_per_unit / 1000.0
    return np.concatenate([[0.0], np.cumsum(d)])


def _guess_utm_crs_from_text(raw_text: bytes) -> Optional[str]:
    """Best-effort heuristic for projected (CoordinateUnits=1) files: look for
    a POPULATED 'ZONE ID' card in the textual header (the standard SEG-Y C20
    line: ``MAP PROJECTION ... ZONE ID:<n> ... COORDINATE UNITS``) and turn a
    plausible UTM zone number into an EPSG guess.

    Deliberately conservative — SEG-Y has NO field for hemisphere anywhere in
    the standard header, so a found zone is assumed Northern (the more common
    default) purely as a fallback guess, never a confirmed detection; callers
    must flag it as unconfirmed (see ``_detect_crs``). Returns ``None`` (no
    guess) when the card is blank or ``0`` — the literal, unfilled-template
    state seen on real acquisition exports (confirmed on both ANT26 and MCS7
    sample files: ``ZONE ID:0`` / blank ``ZONE ID``), so a real EPSG override
    is the only reliable path for those.
    """
    try:
        text = _normalize_raw_text_bytes(raw_text).decode("ascii", errors="replace")
    except Exception:
        return None
    m = re.search(r"ZONE\s*ID\s*:?\s*(\d{1,2})\b", text, re.IGNORECASE)
    if not m:
        return None
    zone = int(m.group(1))
    if not (1 <= zone <= 60):
        return None
    return f"EPSG:{32600 + zone}"   # UTM zone, Northern hemisphere — UNCONFIRMED guess


def _detect_crs(coord_unit: int, lons: np.ndarray,
                crs_override: Optional[str] = None,
                raw_text: Optional[bytes] = None) -> tuple:
    """Resolve a CRS for the map, per CoordinateUnits — the 'Geometry Sanity
    & Reprojection' pipeline (see also :func:`sbp_studio.core.safe_map_coords`,
    which uses the result to guard what actually reaches the map widget).

    coord_unit == 1 (projected length, e.g. UTM metres): standard SEG-Y has
    NO field for the zone/projection, so it can never be auto-detected with
    certainty from the header alone. Resolution order:
      1. ``crs_override`` (explicit EPSG/WKT string, e.g. 'EPSG:32631') — the
         reliable path; exactly what a human operator (or Petrel's CRS
         prompt) would supply.
      2. ``_guess_utm_crs_from_text`` — a textual-header 'ZONE ID' heuristic,
         used ONLY when present and non-zero; flagged as unconfirmed.
      3. Neither → return None with an explicit warning. The raw
         eastings/northings must NEVER be silently treated as WGS84 degrees
         (that is exactly the "UTM meters fed to a Lat/Lon bounding box"
         crash) — refusing to guess here is what makes that refusal safe.
    """
    if coord_unit == 1:
        if crs_override:
            try:
                resolved = resolve_crs(crs_override)
                return resolved, [f"✔ Coordenadas proyectadas (units=1) → EPSG indicado: {resolved}"]
            except Exception as exc:
                return None, [f"⚠ EPSG indicado inválido ('{crs_override}'): {exc}"]
        guess = _guess_utm_crs_from_text(raw_text) if raw_text else None
        if guess:
            return guess, [
                f"⚠ Coordenadas proyectadas (units=1) — zona UTM ADIVINADA del "
                f"encabezado de texto ({guess}, hemisferio Norte asumido). "
                "VERIFICAR antes de confiar en el mapa; SEG-Y no registra el "
                "hemisferio."]
        return None, [
            "⚠ Coordenadas proyectadas (units=1, metros/pies) detectadas — SEG-Y "
            "no registra la zona/proyección. Indique un EPSG (p.ej. 'EPSG:32631' "
            "para UTM 31N) para reproyectar al mapa; sin él, el mapa OMITIRÁ esta "
            "pista en vez de graficar metros UTM como si fueran grados (lo que "
            "rompería el cuadro delimitador del mapa)."]
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
                                load_traces: bool = True,
                                crs_override: Optional[str] = None) -> None:
    """
    Fill a SegyProfile from an already-open segyio file handle.
    Shared by load_metadata and load_profile.

    ``crs_override`` is the reliable resolution path for projected
    (CoordinateUnits=1) files — e.g. 'EPSG:32631' for UTM 31N — whenever the
    SEG-Y header itself can't say which projection/zone was used (it never
    can; see ``_detect_crs``). Passed straight through to ``prof.detected_crs``
    resolution; ``prof.lons``/``prof.lats``/``prof.track_lons``/``track_lats``
    stay in their NATIVE recorded units regardless (unchanged contract — they
    feed exports, dist_km, and chain-gap geometry, none of which need or want
    a CRS to be known). Only the MAP-facing path
    (``sbp_studio.core.safe_map_coords``, fed by ``detected_crs``) changes.
    """
    prof.n_traces = f.tracecount
    prof.ns       = f.samples.size
    _dt = int(f.bin[segyio.BinField.Interval])
    if _dt <= 0:
        import warnings
        warnings.warn(
            f"SEG-Y binary header has Interval={_dt} µs (corrupt/missing); "
            "defaulting to 1 µs to prevent division-by-zero downstream.",
            RuntimeWarning, stacklevel=4,
        )
        _dt = 1
    prof.dt_us = _dt

    h0 = f.header[0]
    # Coordinate + elevation scalars: 16-bit SIGNED (>h). Force the sign so a
    # negative scalar (the common case, meaning "divide by |scalar|") survives.
    prof.scalar_coord = int(_to_signed(h0[segyio.TraceField.SourceGroupScalar] or 0, 16))
    prof.scalar_elev  = int(_to_signed(h0[segyio.TraceField.ElevationScalar]   or 0, 16))
    prof.coord_unit   = int(h0[segyio.TraceField.CoordinateUnits]   or 0)
    # #13: BinField.MeasurementSystem: 1=metres, 2=feet, 0=unknown (default metres)
    _meas = int(f.bin[segyio.BinField.MeasurementSystem] or 0)
    prof.meters_per_unit = 0.3048 if _meas == 2 else 1.0

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

    # ── Artifact cleanup: purge consecutive duplicate-timestamp traces ─────────
    # Computed from the time headers (available even header-only). The SAME mask
    # is applied below to EVERY per-trace array (coords, delays, water depth,
    # timestamps, the inspector headers and — when present — the trace matrix) so
    # nothing can become spatially misaligned. n_traces is updated to the cleaned
    # count; original_n_traces / n_purged record what was removed for the GUI.
    prof.original_n_traces = int(f.tracecount)
    _keep = _timestamp_dedup_mask(
        _hdr[segyio.TraceField.DayOfYear],
        _hdr[segyio.TraceField.HourOfDay],
        _hdr[segyio.TraceField.MinuteOfHour],
        _hdr[segyio.TraceField.SecondOfMinute],
        _hdr[segyio.TraceField.SourceX],
        _hdr[segyio.TraceField.SourceY],
        scalar=prof.scalar_coord,
    )
    if _keep is not None:
        for fld in list(_hdr):
            _hdr[fld] = _hdr[fld][_keep]
        prof.n_traces = int(_keep.sum())
        prof.n_purged = prof.original_n_traces - prof.n_traces
        _LOG.warning("Purged %d duplicate-timestamp trace(s) from %s (%d → %d)",
                     prof.n_purged, prof.name, prof.original_n_traces, prof.n_traces)
    else:
        prof.n_purged = 0

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
        data = f.trace.raw[:].T.astype(np.float32)
        if _keep is not None:                    # drop the duplicate columns
            data = np.ascontiguousarray(data[:, _keep])
        # Mandatory DC-offset removal — the VERY FIRST thing done to the raw
        # trace matrix, before anything else (DSP nodes, the legacy generic
        # pipeline, amp_max/clip_p99 stats) ever sees it. inplace=True is
        # safe here specifically: `data` was JUST allocated above (by
        # .astype(...) or by np.ascontiguousarray's copy) and has no other
        # reference yet, so skipping the extra allocation is a real memory
        # win on the multi-GB matrices a deep MCS line can produce.
        data = apply_dc_removal(data, inplace=True)
        prof.data     = data
        prof.amp_max  = np.max(np.abs(prof.data), axis=0)
        prof.clip_p99 = float(np.percentile(np.abs(prof.data), 99))
        prof.active_lo, prof.active_ns = _detect_active_band(prof.data, prof.clip_p99)
    else:
        prof.data    = None
        prof.amp_max = None
        prof.clip_p99 = None
        prof.active_lo = None
        prof.active_ns = None

    prof.dur_ms  = prof.ns * prof.dt_us / 1000.0
    prof.t_ms    = prof.delay_ms + np.arange(prof.ns) * prof.dt_us / 1000.0
    # Distance axis from the CLEANED track so it is smooth and well-behaved for
    # the seismic X-axis and the map↔profile sync (raw coords stay export-only).
    prof.dist_km = _dist_km(prof.track_lons, prof.track_lats, prof.coord_unit,
                             prof.meters_per_unit)
    prof.total_km = float(prof.dist_km[-1])

    # Header Inspector data — textual header is always decoded (single 3200-byte
    # block, negligible cost). The full 91-field per-trace extract is deferred
    # for header-only stubs (chain detection, pre-scan) where no inspector is
    # shown; it runs only when traces are loaded so the GUI inspector works.
    # Bytes come from read_raw_text_header (plain file I/O), NOT f.text[0] —
    # segyio's own text accessor force-converts EBCDIC->ASCII and corrupts
    # files (like these) that actually store the header as plain ASCII.
    _raw_text = read_raw_text_header(prof.path)
    prof.text_header = decode_text_header(_raw_text, encoding="auto")
    prof.text_header_encoding = detect_text_header_encoding(_raw_text)
    if load_traces:
        prof.trace_headers = _extract_trace_headers(f)
        if _keep is not None:                    # keep the inspector aligned too
            prof.trace_headers = {k: v[_keep] for k, v in prof.trace_headers.items()}
    prof.detected_crs, prof.crs_notes = _detect_crs(
        prof.coord_unit, prof.lons, crs_override=crs_override, raw_text=_raw_text)
    prof.crs_is_unknown = (prof.detected_crs is None)  # #21

    # OQ-2: warn if projected CRS axis unit is not metres
    unit_warnings = _check_crs_units(prof.detected_crs, prof.coord_unit)
    prof.crs_notes.extend(unit_warnings)


def set_crs_override(obj, crs_override: str) -> None:
    """Apply a user-chosen CRS to an ALREADY-LOADED ``SegyProfile`` or
    ``ProfileChain`` in place — no file re-read needed (``coord_unit`` and
    ``lons`` are already resident in memory). This is what the GUI's GIS-
    style CRS-selector dialog calls once the user picks a zone for a
    PROJECTED (CoordinateUnits=1) file that couldn't be auto-detected (see
    ``_detect_crs``).

    Updates ``obj.detected_crs``/``obj.crs_notes`` only. ``obj.lons``/
    ``obj.track_lons`` etc. are deliberately left untouched (native units —
    see ``_populate_profile_from_file``'s contract); the map picks up the
    new CRS on its next read via ``sbp_studio.core.safe_map_coords``.
    """
    crs, notes = _detect_crs(obj.coord_unit, obj.lons, crs_override=crs_override)
    notes = list(notes) + _check_crs_units(crs, obj.coord_unit)
    obj.detected_crs = crs
    obj.crs_notes = notes


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_metadata(path: str, crs_override: Optional[str] = None) -> SegyMetadata:
    """
    Load header-only metadata from a SEG-Y file WITHOUT reading trace data.
    Fast even for large files: reads only binary header + per-trace fields.

    ``crs_override``: EPSG/WKT string (e.g. 'EPSG:32631') to resolve the map
    CRS for PROJECTED files (CoordinateUnits=1) — SEG-Y has no zone/projection
    field, so this is the reliable way to tell the loader what UTM zone (or
    other projected CRS) the file's Source/CDP X/Y are actually in. See
    ``_detect_crs`` / ``sbp_studio.core.safe_map_coords``.
    """
    prof = SegyProfile(path)
    try:
        with segyio.open(path, ignore_geometry=True) as f:
            _populate_profile_from_file(prof, f, load_traces=False, crs_override=crs_override)
    except Exception as exc:
        prof.error = str(exc)
    return prof.to_metadata()


def load_profile(path: str, load_traces: bool = True,
                 crs_override: Optional[str] = None) -> SegyProfile:
    """
    Load a SEG-Y file into a SegyProfile.
    On error, .error is set and other fields remain at defaults. Never raises.

    ``crs_override``: see :func:`load_metadata`.
    """
    prof = SegyProfile(path)
    try:
        with segyio.open(path, ignore_geometry=True) as f:
            _populate_profile_from_file(prof, f, load_traces=load_traces, crs_override=crs_override)
    except Exception as exc:
        prof.error = str(exc)
    return prof


# ── In-place header patch ───────────────────────────────────────────────────────

def _format_text_header_for_write(text: str) -> str:
    """Format user-edited text into the SEG-Y 40 × 80 char layout (3200 chars)."""
    lines = text.split('\n')
    padded = [line[:80].ljust(80) for line in lines[:40]]
    while len(padded) < 40:
        padded.append(' ' * 80)
    return ''.join(padded)


def _encode_text_header_bytes(text: str, encoding: str = "ascii") -> bytes:
    """Format *text* to the 3200-char card layout and encode it to raw bytes
    using *encoding* ('ascii', 'ebcdic'/cp037, or 'latin-1') — the SAME codec
    the GUI is currently decoding/displaying with, so a round-trip Save
    never silently flips the file's textual-header convention. Characters
    outside the chosen codec degrade to ``?`` (errors="replace") rather than
    raising mid-write."""
    formatted = _format_text_header_for_write(text)
    codec = _TEXT_CODECS.get(encoding, "ascii")
    return formatted.encode(codec, errors="replace")


def patch_segy_headers(
    path: str,
    *,
    text_header: Optional[str] = None,
    text_encoding: str = "ascii",
    binary_updates: Optional[dict] = None,
) -> tuple:
    """Patch a SEG-Y file's text and/or binary header in-place.

    The text header is written with PLAIN file I/O (seek 0, write 3200
    raw bytes) — never via segyio's ``f.text[0] = ...`` setter. That setter
    unconditionally EBCDIC-encodes (cp037) whatever string it's given,
    regardless of the file's actual textual-header convention (confirmed by
    byte-level probe: writing an ASCII string through ``f.text[0]`` produces
    cp037 bytes on disk). Since many marine SBP systems (Kongsberg TOPAS and
    others) store this block as plain ASCII, that setter would silently flip
    the file's encoding on every save — exactly the kind of corruption this
    module exists to prevent. *text_encoding* should be whatever codec the
    GUI is currently displaying the text under (``profile.text_header_encoding``)
    so the write matches the read.

    Binary-header updates still go through segyio's ``r+`` mode (unaffected
    by this issue — only the text/EBCDIC accessor is special-cased by
    segyio). If ``BinField.Interval`` (dt_us) is in *binary_updates* the new
    value is also bulk-written to every trace's ``TRACE_SAMPLE_INTERVAL``
    field so that readers that use per-trace dt (rather than the binary
    header) are also corrected.

    Returns ``(True, "")`` on success or ``(False, error_message)`` on failure.
    """
    if text_header is None and not binary_updates:
        return True, ""

    if binary_updates:
        samples_key = int(segyio.BinField.Samples)
        if samples_key in {int(k) for k in binary_updates}:
            return False, (
                "Refusing to change ns (samples/trace) via header patch: this "
                "would change the binary header's declared trace length without "
                "resizing the actual trace data blocks on disk, corrupting every "
                "trace boundary for any reader. Resampling/truncating the data "
                "matrix and rewriting the file is a different, heavier operation "
                "not supported by this in-place patcher."
            )

    try:
        if text_header is not None:
            raw_bytes = _encode_text_header_bytes(text_header, text_encoding)
            with open(path, "r+b") as fh:
                fh.seek(0)
                fh.write(raw_bytes)

        if binary_updates:
            with segyio.open(path, mode='r+', ignore_geometry=True) as f:
                f.bin.update(binary_updates)
                interval_key = int(segyio.BinField.Interval)
                if interval_key in {int(k) for k in binary_updates}:
                    new_dt = int(binary_updates[
                        next(k for k in binary_updates if int(k) == interval_key)])
                    if new_dt > 0:
                        # Mass-propagate dt to every trace header. ``f.attributes()``
                        # is read-only in segyio; the portable write path is to
                        # update each trace header mapping in place.
                        ts_field = segyio.TraceField.TRACE_SAMPLE_INTERVAL
                        for i in range(f.tracecount):
                            f.header[i].update({ts_field: new_dt})
                f.flush()

        return True, ""
    except Exception as exc:
        _LOG.error("patch_segy_headers failed for %s: %s", path, exc)
        return False, str(exc)


# ── Trace-header field widths & safe bulk patching ──────────────────────────────

_TRACE_HEADER_BYTES = 240


def _build_trace_field_widths() -> Dict[str, int]:
    """Derive each standard trace-header field's byte width (2 or 4) from the
    gap to the next field's offset in segyio's byte-offset table. This
    reproduces the SEG-Y rev1 spec exactly — e.g. NSummedTraces (offset 31)
    to NStackedTraces (offset 33) is 2 bytes (int16); offset (37) to
    ReceiverGroupElevation (41) is 4 bytes (int32)."""
    offsets = sorted(segyio.tracefield.keys.items(), key=lambda kv: kv[1])
    widths: Dict[str, int] = {}
    for i, (name, off) in enumerate(offsets):
        nxt = offsets[i + 1][1] if i + 1 < len(offsets) else (_TRACE_HEADER_BYTES + 1)
        widths[name] = nxt - off
    return widths


_TRACE_FIELD_WIDTHS: Dict[str, int] = _build_trace_field_widths()


def trace_field_names() -> Tuple[str, ...]:
    """All standard trace-header field names, in on-disk byte order."""
    return tuple(sorted(_TRACE_FIELD_WIDTHS, key=lambda n: segyio.tracefield.keys[n]))


def trace_field_int_range(field_name: str) -> Optional[Tuple[int, int]]:
    """Signed-integer (min, max) for a standard trace-header field, derived
    from its byte width (2 -> int16, 4 -> int32) per the SEG-Y rev1 spec.
    Returns ``None`` if *field_name* isn't a recognised field — callers
    should treat that as "refuse to write", not "anything goes"."""
    width = _TRACE_FIELD_WIDTHS.get(field_name)
    if width == 2:
        return (-32768, 32767)
    if width == 4:
        return (-2_147_483_648, 2_147_483_647)
    return None


def patch_trace_header_field(
    path: str,
    field_name: str,
    values: np.ndarray,
    progress: Optional[ProgressCallback] = None,
) -> tuple:
    """Bulk-write *values* into a single named trace-header field, in-place.

    Safety contract (defense in depth — callers SHOULD validate first, but
    this is the function that actually touches the disk, so it validates
    again regardless):
      * *field_name* must be a recognised, fixed-width SEG-Y trace-header
        field (see :func:`trace_field_int_range`) — unknown fields are
        refused.
      * *values* must be an integer-dtype array — floats are refused, even
        if numerically whole, since the on-disk field is always integer.
      * Every value must fit the field's byte width (int16 or int32) —
        anything that would overflow/wrap on write is refused.
      * The array length must exactly match the file's trace count —
        otherwise headers would silently misalign.

    Writes use ``header[i].update({field: value})`` per trace, which patches
    ONLY that field's bytes — every other byte of every trace header is left
    untouched. Returns ``(True, "")`` on success or ``(False, error_message)``
    with NOTHING written on failure.
    """
    try:
        field_enum = getattr(segyio.TraceField, field_name)
    except AttributeError:
        return False, f"Unknown trace-header field: {field_name}"

    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        return False, (
            f"Refusing to write floating-point values into integer field "
            f"'{field_name}'."
        )
    if arr.dtype.kind not in "iub":
        return False, f"Unsupported value dtype for '{field_name}': {arr.dtype}"

    int_range = trace_field_int_range(field_name)
    if int_range is None:
        return False, f"'{field_name}' has no recognised fixed-width slot — refusing to write."
    lo, hi = int_range
    if arr.size and (int(arr.min()) < lo or int(arr.max()) > hi):
        return False, (
            f"Value range [{int(arr.min())}, {int(arr.max())}] overflows "
            f"'{field_name}' (valid range [{lo}, {hi}]) — refusing to write."
        )

    try:
        with segyio.open(path, mode='r+', ignore_geometry=True) as f:
            n = f.tracecount
            if arr.shape[0] != n:
                return False, (
                    f"Value count ({arr.shape[0]}) does not match trace count "
                    f"({n}) — refusing to write (would misalign headers)."
                )
            for i in range(n):
                if progress is not None and (i % 256 == 0):
                    progress(i / max(1, n), "patching trace headers")
                f.header[i].update({field_enum: int(arr[i])})
            f.flush()
        return True, ""
    except Exception as exc:
        _LOG.error("patch_trace_header_field failed for %s/%s: %s",
                   path, field_name, exc)
        return False, str(exc)


# ── Reprojection helpers ────────────────────────────────────────────────────────

def _safe_coord(v: float, div: float) -> int:
    """Scale and clip a coordinate to fit in a SEG-Y int32 header field."""
    return int(np.clip(round(v / div), -_INT32_MAX, _INT32_MAX))


def _safe_coords_bulk(arr: np.ndarray, div: float) -> np.ndarray:
    """Vectorised _safe_coord — returns an int32 array for bulk attribute writes."""
    return np.clip(np.round(arr / div), -_INT32_MAX, _INT32_MAX).astype(np.int32)


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
    raw_sxs     = _to_signed(
        src_f.attributes(segyio.TraceField.SourceX)[:], 32).astype(float)
    raw_sys     = _to_signed(
        src_f.attributes(segyio.TraceField.SourceY)[:], 32).astype(float)

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

    # #8: Guard against pyproj returning inf/nan (e.g. out-of-zone UTM coords).
    bad = ~(np.isfinite(nxs) & np.isfinite(nys))
    n_bad = int(bad.sum())
    if n_bad:
        _LOG.warning(
            "_opt_reproject_coords_bulk: %d/%d non-finite output coord(s); "
            "forward-filling from nearest valid neighbour.", n_bad, n)
        good = np.where(~bad)[0]
        if good.size == 0:
            raise ValueError(
                "All reprojected coordinates are non-finite — check the "
                "source/target CRS combination.")
        nxs[bad] = np.interp(np.where(bad)[0], good, nxs[good])
        nys[bad] = np.interp(np.where(bad)[0], good, nys[good])

    return nxs, nys


def _warn_irregular_spacing(dist_km: np.ndarray, label: str,
                             log_fn: LogCallback) -> None:
    """Emit a structured warning when inter-trace spacing CV exceeds 10% (#19).

    A high CV indicates that the SEG-Y file was recorded with variable ping
    rate or uneven vessel speed.  Writing it with uniform trace indices (as
    every SEG-Y writer does) is a silent form of spatial regularization: a
    25-m ping followed by a 250-m gap is rendered as two equally spaced
    columns, compressing the gap tenfold.  Callers should flag this so the
    geophysicist can decide whether true regularization is needed."""
    if dist_km is None or dist_km.size < 3:
        return
    spacing = np.diff(dist_km)
    mean_sp = float(spacing.mean())
    if mean_sp <= 0:
        return
    cv = float(spacing.std()) / mean_sp
    if cv > _SPACING_CV_THRESH:
        msg = (f"Espaciado irregular en '{label}': CV={cv * 100:.1f}% "
               f"(umbral {_SPACING_CV_THRESH * 100:.0f}%) — "
               "se recomienda regularización espacial real antes de interpretar.")
        _LOG.warning(
            "Irregular trace spacing in '%s': CV=%.1f%% (threshold %.0f%%). "
            "True spatial regularization is recommended before interpretation.",
            label, cv * 100, _SPACING_CV_THRESH * 100)
        log_fn(f"⚠  {msg}")


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
    amplitude_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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

    ``amplitude_transform`` — optional ``(ns, n_traces) float32 -> (ns,
    n_traces) float32`` callable (e.g. the active DSP filter chain, see
    ``gui.dsp.export_filter.apply_pipeline_to_matrix``) applied to the FULL
    trace matrix BEFORE writing. Every trace header is still cloned exactly
    as above — ONLY the trace amplitude payload is replaced by this
    function's output; everything else (EBCDIC text, binary header, every
    header field, coordinates) is byte-identical to the source. ``None``
    (the default) preserves the exact pre-existing byte-faithful behaviour
    (``dst.trace[i] = src.trace[i]``, no full-matrix read at all).
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

            # #19: warn if source trace spacing is highly irregular
            _warn_irregular_spacing(sd.dist_km, sd.name, log)

            # File-copy is byte-faithful to the SOURCE: iterate the file's true
            # trace count, NOT sd.n_traces (which may be the in-memory cleaned
            # count after duplicate-timestamp purging). The dedup is a display/
            # analysis feature; the reprojected file mirrors the original 1:1.
            n_src = int(src.tracecount)

            # Filtered amplitudes (optional): read the FULL matrix ONCE here
            # (NOT per-trace) so a 2-D/cross-trace filter sees every trace
            # together, exactly as the live preview's full_depth windowing
            # does. None ⇒ zero extra RAM, zero behaviour change (the
            # original byte-for-byte src.trace[i] copy below).
            processed = None
            if amplitude_transform is not None:
                progress(0.05, "applying filters…")
                full = src.trace.raw[:].T.astype(np.float32)   # (ns, n_traces)
                processed = amplitude_transform(full)
                if processed.shape != full.shape:
                    raise ReprojectionError(
                        "amplitude_transform changed the array shape: "
                        f"{full.shape} -> {processed.shape}")

            with segyio.create(outpath, spec) as dst:
                dst.bin    = src.bin
                dst.text[0] = src.text[0]

                cancel.check()
                progress(0.15, "copying headers…")

                # Phase 1: C-level bulk header clone — eliminates the Python
                # per-trace loop and every dict(src.header[i]) allocation.
                dst.header[:] = src.header[:]

                # Phase 2: bulk overwrite the 6 coordinate fields only.
                # One Python list-comprehension (6-key dicts, not 91) + one
                # C-level write; SourceGroupScalar / CoordinateUnits are
                # 2-byte fields — segyio truncates int32 → int16 on write.
                scaled_x = _safe_coords_bulk(nxs, div)
                scaled_y = _safe_coords_bulk(nys, div)
                dst.header[:] = [
                    {
                        segyio.TraceField.SourceX:           int(scaled_x[i]),
                        segyio.TraceField.SourceY:           int(scaled_y[i]),
                        segyio.TraceField.GroupX:            int(scaled_x[i]),
                        segyio.TraceField.GroupY:            int(scaled_y[i]),
                        segyio.TraceField.SourceGroupScalar: out_sc,
                        segyio.TraceField.CoordinateUnits:   new_uc,
                    }
                    for i in range(n_src)
                ]

                cancel.check()
                progress(0.70, "copying samples…")

                # Phase 3: bulk trace data copy.
                # .raw bypasses per-trace IBM/IEEE conversion; byte-faithful
                # for both float formats.  For the filtered path, processed
                # is (ns, n_traces) → transpose to (n_traces, ns) for segyio.
                if processed is None:
                    dst.trace.raw[:] = src.trace.raw[:]
                else:
                    dst.trace[:] = np.ascontiguousarray(
                        processed.T, dtype=np.float32)

        progress(1.0, "done")
        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except (Cancelled, CRSError):
        _try_delete(outpath)
        raise
    except Exception as exc:
        import traceback
        log(f"✘ Error: {exc}\n{traceback.format_exc()}")
        _LOG.exception("Reprojection failed for %s", sd.name)
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
    amplitude_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> str:
    """
    Reproject and JOIN a ProfileChain into a single SEG-Y file.

    Uses the optimised (vectorised) coordinate path: per-profile bulk
    transform, then per-trace header write.

    Header contract: same as reproject_one, plus TraceNumber = global index.

    ``amplitude_transform`` — see ``reproject_one``'s docstring. Applied
    PER CONSTITUENT FILE (its own full (ns, n_traces) matrix), since each
    file in the chain is a physically separate source — a 2-D filter sees
    full cross-trace continuity WITHIN one file, but not across the join
    seam between two chained files (no different from how the live preview
    only ever sees one profile's own matrix at a time).
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
        # #19: warn once for the whole chain before the write loop
        _warn_irregular_spacing(ch.dist_km, ch.label, log)

        # Byte-faithful copy: size the output by the SOURCE files' real trace
        # counts (original_n_traces), not ch.n_traces — which may be the cleaned
        # in-memory total after duplicate-timestamp purging.
        n_total = int(sum(getattr(p, "original_n_traces", 0) or p.n_traces
                          for p in ch.profiles))
        with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
            spec = segyio.tools.metadata(src0)
            spec.tracecount = n_total

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

                    # This constituent file's own full matrix (see
                    # reproject_one's docstring for why full-matrix-at-once,
                    # not per trace) — None ⇒ zero extra RAM, unchanged
                    # byte-for-byte behaviour.
                    processed = None
                    if amplitude_transform is not None:
                        full = src.trace.raw[:].T.astype(np.float32)
                        processed = amplitude_transform(full)
                        if processed.shape != full.shape:
                            raise ReprojectionError(
                                "amplitude_transform changed the array shape: "
                                f"{full.shape} -> {processed.shape}")

                    n_src_i   = int(src.tracecount)
                    seg_start = global_idx
                    seg_end   = global_idx + n_src_i

                    cancel.check()
                    log(f"    trazas {seg_start+1}–{seg_end}/{n_total}…")
                    progress(seg_start / n_total,
                             f"traza {seg_start+1}/{n_total}")

                    # Phase 1: C-level bulk header clone for this segment.
                    dst.header[seg_start:seg_end] = src.header[:]

                    # Phase 2: bulk overwrite coord + TraceNumber (7-key dicts).
                    scaled_x = _safe_coords_bulk(nxs, div)
                    scaled_y = _safe_coords_bulk(nys, div)
                    dst.header[seg_start:seg_end] = [
                        {
                            segyio.TraceField.SourceX:           int(scaled_x[i]),
                            segyio.TraceField.SourceY:           int(scaled_y[i]),
                            segyio.TraceField.GroupX:            int(scaled_x[i]),
                            segyio.TraceField.GroupY:            int(scaled_y[i]),
                            segyio.TraceField.SourceGroupScalar: out_sc,
                            segyio.TraceField.CoordinateUnits:   new_uc,
                            segyio.TraceField.TraceNumber:       seg_start + i + 1,
                        }
                        for i in range(n_src_i)
                    ]

                    # Phase 3: bulk trace data for this segment.
                    if processed is None:
                        dst.trace.raw[seg_start:seg_end] = src.trace.raw[:]
                    else:
                        dst.trace[seg_start:seg_end] = np.ascontiguousarray(
                            processed.T, dtype=np.float32)

                    global_idx = seg_end

        progress(1.0, "done")
        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except (Cancelled, CRSError):
        _try_delete(outpath)
        raise
    except Exception as exc:
        import traceback
        log(f"✘ Error: {exc}\n{traceback.format_exc()}")
        _LOG.exception("Reprojection failed for chain %s", ch.label)
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
        # Byte-faithful join: size by the SOURCE files' real trace counts
        # (original_n_traces), independent of in-memory duplicate-timestamp purging.
        n_total = int(sum(getattr(p, "original_n_traces", 0) or p.n_traces
                          for p in ch.profiles))
        with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
            spec = segyio.tools.metadata(src0)
            spec.tracecount = n_total

        with segyio.create(outpath, spec) as dst:
            with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
                dst.bin    = src0.bin
                dst.text[0] = src0.text[0]

            global_idx = 0
            for p_idx, sd in enumerate(ch.profiles):
                log(f"  Copiando {p_idx+1}/{len(ch.profiles)}: {sd.name}…")
                with segyio.open(sd.path, ignore_geometry=True) as src:
                    n_src_i   = int(src.tracecount)
                    seg_start = global_idx
                    seg_end   = global_idx + n_src_i

                    cancel.check()
                    progress(seg_start / n_total,
                             f"traza {seg_start+1}/{n_total}")

                    # Phase 1: C-level bulk header clone for this segment.
                    dst.header[seg_start:seg_end] = src.header[:]

                    # Phase 2: overwrite TraceNumber only (1-key dicts).
                    dst.header[seg_start:seg_end] = [
                        {segyio.TraceField.TraceNumber: seg_start + i + 1}
                        for i in range(n_src_i)
                    ]

                    # Phase 3: bulk trace data (byte-faithful raw copy).
                    dst.trace.raw[seg_start:seg_end] = src.trace.raw[:]

                    global_idx = seg_end

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
