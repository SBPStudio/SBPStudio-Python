"""
io_segy.py — SEG-Y loading and reprojection writers.

Public API
----------
load_metadata(path)                → SegyMetadata     (no trace data read)
load_profile(path, load_traces)    → SegyProfile      (traces loaded by default)
reproject_one(sd, src, dst, ...)   → str | None       (output path on success)
reproject_chain(ch, src, dst, ...) → str | None       (output path on success)

Behaviour preserved verbatim from the monolith
-----------------------------------------------
- Scalar factor: fac = 1/abs(sc) if sc<0, sc if sc>0, else 1.0
- Arc-sec: coord_unit==2 → divide by 3600 to get degrees
- Elevation scalar: same fac formula applied to SourceWaterDepth
- Timestamp format: "YYYY-DOYnnn HH:MM:SS" (DOY zero-padded to 3 digits)
- dist_km heuristic: geographic if coord_unit in (2,3) or range check; else
  Euclidean / 1000.0 (assumes metres — known limitation)
- delay_ms = int(delays[0])  (only first trace — known limitation)
- Reprojection header rules: copy full trace header (preserves
  DelayRecordingTime); overwrite only SourceX/Y, GroupX/Y,
  SourceGroupScalar, CoordinateUnits (+ TraceNumber for chains);
  out_sc = -10_000_000 if dst geographic else -100;
  new_uc = 3 (geographic) or 1 (projected);
  clip coords to ±INT32_MAX; copy bin + text[0].

Error handling
--------------
load_metadata / load_profile wrap all segyio errors in SegyLoadError.
reproject_one / reproject_chain raise ReprojectionError on failure and
delete partial output files. Both accept optional progress and cancel args.
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


# ── Internal helpers ────────────────────────────────────────────────────────────

def _scalar_fac(sc: int) -> float:
    if sc < 0:
        return 1.0 / abs(sc)
    if sc > 0:
        return float(sc)
    return 1.0


def _dist_km(lons: np.ndarray, lats: np.ndarray, coord_unit: int) -> np.ndarray:
    """
    Compute cumulative along-track distance in km.

    Uses geographic (haversine-approximation) formula when coord_unit in (2,3)
    or when coordinates appear to be degrees. Falls back to Euclidean/1000
    otherwise (assumes metres — known limitation).
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


def _populate_profile_from_file(prof: SegyProfile, f: "segyio.SegyFile",
                                load_traces: bool = True) -> None:
    """
    Fill a SegyProfile from an already-open segyio file handle.
    Extracted so that load_metadata and load_profile share one code path.
    """
    prof.n_traces = f.tracecount
    prof.ns       = f.samples.size
    prof.dt_us    = int(f.bin[segyio.BinField.Interval])

    h0 = f.header[0]
    prof.scalar_coord = int(h0[segyio.TraceField.SourceGroupScalar] or 0)
    prof.scalar_elev  = int(h0[segyio.TraceField.ElevationScalar]   or 0)
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

    sc  = prof.scalar_coord
    fac = _scalar_fac(sc)
    sxs  = _hdr[segyio.TraceField.SourceX].astype(float) * fac
    sys_ = _hdr[segyio.TraceField.SourceY].astype(float) * fac

    if prof.coord_unit == 2:   # arc-seconds → degrees
        prof.lons = sxs  / 3600.0
        prof.lats = sys_ / 3600.0
    else:
        prof.lons = sxs
        prof.lats = sys_

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
    prof.dist_km = _dist_km(prof.lons, prof.lats, prof.coord_unit)
    prof.total_km = float(prof.dist_km[-1])
    prof.detected_crs, prof.crs_notes = _detect_crs(prof.coord_unit, prof.lons)


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_metadata(path: str) -> SegyMetadata:
    """
    Load header-only metadata from a SEG-Y file WITHOUT reading trace data.

    This is fast even for large files since it reads only the binary header
    and per-trace header fields (one segyio attribute pass each).

    Returns
    -------
    SegyMetadata with error=None on success, or error set to exception string.

    Raises
    ------
    SegyLoadError on fatal I/O errors.
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

    Parameters
    ----------
    path        : absolute or relative path to the SEG-Y file
    load_traces : if True (default), loads the full trace matrix (ns, n_traces)
                  as float32. If False, data/amp_max/clip_p99 remain None.

    Returns
    -------
    SegyProfile; on error, the .error attribute is set and other fields
    have their default (zero/None) values. Never raises.
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
    """Scale and clip a coordinate value to fit in a SEG-Y int32 header field."""
    return int(np.clip(round(v / div), -_INT32_MAX, _INT32_MAX))


def _build_transformer(src_str: str, dst_str: str) -> tuple:
    """
    Build a pyproj Transformer and return (tf, out_sc, new_uc, dst_geo, div).

    Raises CRSError for invalid CRS strings.
    """
    try:
        src_crs = CRS.from_user_input(src_str)
        dst_crs = CRS.from_user_input(dst_str)
    except Exception as exc:
        raise CRSError(f"Invalid CRS: {exc}") from exc

    dst_geo = dst_crs.is_geographic
    out_sc  = -10_000_000 if dst_geo else -100
    new_uc  = 3 if dst_geo else 1
    div     = (1.0 / abs(out_sc)) if out_sc < 0 else float(out_sc)
    tf      = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tf, out_sc, new_uc, dst_geo, div


# ── Reference reprojection: trace-by-trace (the oracle) ───────────────────────

def _ref_reproject_trace(h: dict, tf: Transformer, unit_hint: int,
                         out_sc: int, new_uc: int, div: float) -> tuple:
    """
    Reference per-trace coordinate transform. Returns (nx, ny) after applying
    the scalar, arc-sec correction, and transform.
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


def reproject_one(
    sd: SegyProfile,
    src_str: str,
    dst_str: str,
    unit_hint: int = 2,
    log: LogCallback = _noop_log,
    progress: ProgressCallback = _noop_progress,
    cancel: Optional[CancelToken] = None,
) -> str:
    """
    Reproject a single SEG-Y profile to a new CRS.

    Header contract
    ---------------
    - Full trace header copied first (preserves DelayRecordingTime, bytes 109-110).
    - ONLY SourceX/Y, GroupX/Y, SourceGroupScalar, CoordinateUnits overwritten.
    - out_sc = -10_000_000 for geographic dst, -100 for projected dst.
    - new_uc = 3 (geographic) or 1 (projected).
    - Coordinates clipped to ±INT32_MAX.
    - bin and text[0] copied from source.

    Returns
    -------
    Output file path on success.

    Raises
    ------
    CRSError         if src_str or dst_str are invalid.
    ReprojectionError on any other failure (partial output deleted).
    Cancelled         if the cancel token fires.
    """
    if cancel is None:
        cancel = CancelToken.never()

    log("\n" + "-" * 55)
    log(f"Procesando : {sd.name}")
    log(f"CRS origen : {src_str}")
    log(f"CRS destino: {dst_str}")

    tf, out_sc, new_uc, _dst_geo, div = _build_transformer(src_str, dst_str)

    p       = Path(sd.path)
    outpath = str(p.with_name(p.stem + "_REPROY" + p.suffix))
    log(f"Salida     : {Path(outpath).name}")

    try:
        with segyio.open(sd.path, ignore_geometry=True) as src:
            spec = segyio.tools.metadata(src)
            with segyio.create(outpath, spec) as dst:
                dst.bin    = src.bin
                dst.text[0] = src.text[0]
                for i in range(sd.n_traces):
                    cancel.check()
                    if i % 200 == 0:
                        log(f"  traza {i+1}/{sd.n_traces}…")
                        progress(i / sd.n_traces, f"traza {i+1}/{sd.n_traces}")
                    h      = src.header[i]
                    nx, ny = _ref_reproject_trace(h, tf, unit_hint, out_sc, new_uc, div)
                    dst.header[i] = h
                    dst.header[i].update({
                        segyio.TraceField.SourceX:           _safe_coord(nx, div),
                        segyio.TraceField.SourceY:           _safe_coord(ny, div),
                        segyio.TraceField.GroupX:            _safe_coord(nx, div),
                        segyio.TraceField.GroupY:            _safe_coord(ny, div),
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
) -> str:
    """
    Reproject and JOIN a ProfileChain into a single SEG-Y file.

    Header contract: same as reproject_one, plus TraceNumber overwritten
    with the global sequential trace index (1-based) across all profiles.

    Returns
    -------
    Output file path on success.

    Raises
    ------
    CRSError, ReprojectionError, Cancelled — same semantics as reproject_one.
    """
    if cancel is None:
        cancel = CancelToken.never()

    log("\n" + "-" * 55)
    log(f"Procesando CADENA: {ch.label}")
    log(f"CRS origen : {src_str}")
    log(f"CRS destino: {dst_str}")

    tf, out_sc, new_uc, _dst_geo, div = _build_transformer(src_str, dst_str)

    p0     = Path(ch.profiles[0].path)
    stem   = f"{p0.stem}_a_{Path(ch.profiles[-1].path).stem}_UNIDO_REPROY"
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
                log(f"  Integrando perfil {p_idx+1}/{len(ch.profiles)}: {sd.name}...")
                with segyio.open(sd.path, ignore_geometry=True) as src:
                    for i in range(sd.n_traces):
                        cancel.check()
                        if global_idx % 500 == 0:
                            log(f"    traza global {global_idx+1}/{ch.n_traces}…")
                            progress(global_idx / ch.n_traces,
                                     f"traza {global_idx+1}/{ch.n_traces}")
                        h      = src.header[i]
                        nx, ny = _ref_reproject_trace(h, tf, unit_hint, out_sc, new_uc, div)
                        dst.header[global_idx] = h
                        dst.header[global_idx].update({
                            segyio.TraceField.SourceX:           _safe_coord(nx, div),
                            segyio.TraceField.SourceY:           _safe_coord(ny, div),
                            segyio.TraceField.GroupX:            _safe_coord(nx, div),
                            segyio.TraceField.GroupY:            _safe_coord(ny, div),
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


def _try_delete(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
