"""
cli/commands.py — Implementation of each CLI subcommand.

All commands are headless: no GUI imports, no DISPLAY required. Progress
is reported to stderr. Errors produce non-zero exit codes via SystemExit.

The CancelToken is connected to SIGINT (Ctrl-C) so that reprojection can
be interrupted cleanly.
"""
from __future__ import annotations

import concurrent.futures as _cf
import json
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..core import (
    load_metadata, load_profile, SegyProfile, ProfileChain,
    detect_chains, reproject_one, reproject_chain, join_profiles,
    process_profile_data, process_chain_data, time_window,
    compute_spectrum, colormapped_rgba,
    compute_fix_positions, fixes_to_wgs84,
    write_fix_points_shp, write_fix_points_geojson, write_fix_points_csv,
    write_navline_shp, write_navline_geojson, write_navline_csv,
    patch_segy_headers,
    apply_bandpass, apply_spectral_whitening, apply_agc, apply_tvg,
    apply_predictive_decon, apply_filter_preset,
    apply_swell_filter, apply_water_mute, apply_delay_alignment,
    CancelToken, Cancelled, ReprojectionError, CRSError,
    CMAPS,
)
from ..core.tasks import _noop_progress, PhasedTimer
from ..core._backends import accel_info


# ── Helpers ────────────────────────────────────────────────────────────────────

def _err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _progress(fraction: float, msg: str) -> None:
    bar_w  = 30
    filled = int(bar_w * max(0.0, min(1.0, fraction)))
    bar    = "#" * filled + "." * (bar_w - filled)
    pct    = int(fraction * 100)
    print(f"\r  [{bar}] {pct:3d}%  {msg}", end="", flush=True, file=sys.stderr)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


# ── Rasteriser RAM safety ───────────────────────────────────────────────────────

# Peak host-RAM the Matplotlib/Agg raster path holds *concurrently* per output
# pixel. A high-DPI seismic export keeps several full RGBA-sized buffers alive at
# once: the colourised float→RGBA source, the resized canvas image (the chain
# renderer scales the stitched native image up to figw·dpi), the Agg renderer
# buffer, and the PNG/PDF encode buffer. Each is 4 bytes/px, so ~4 live buffers
# ≈ 16 B/px; we budget 18 to leave slack for transient copies. This is what turns
# a hard numpy ArrayMemoryError into a graceful, proportion-preserving down-scale.
_RASTER_BYTES_PER_PX = 18.0

# Fraction of *currently-available* RAM we are willing to dedicate to one raster.
# Deliberately below 1.0 so the OS, the loaded SEG-Y trace matrices and other
# processes keep breathing room. Overridable per-run via --mem-budget-gb.
_RAM_SAFE_FRACTION = 0.55


def _available_ram_bytes() -> float:
    """Best-effort free-RAM probe. Uses psutil when present (accurate, cross-OS),
    else falls back to a conservative fixed estimate so the guard still engages on
    machines without psutil."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        # Conservative: assume ~4 GB free if we cannot measure (keeps the guard
        # protective rather than optimistic on an unknown box).
        return 4.0 * 1024 ** 3


def _safe_pixel_budget(mem_budget_gb: Optional[float]) -> tuple:
    """Return (max_pixels, budget_bytes) the final raster may occupy.

    ``mem_budget_gb`` (the --mem-budget-gb override) caps the peak rasteriser
    footprint directly; when None we take ``_RAM_SAFE_FRACTION`` of the currently
    available system RAM. The pixel budget is that byte budget divided by the
    per-pixel peak (``_RASTER_BYTES_PER_PX``)."""
    if mem_budget_gb is not None and mem_budget_gb > 0:
        budget_bytes = float(mem_budget_gb) * 1024 ** 3
    else:
        budget_bytes = _available_ram_bytes() * _RAM_SAFE_FRACTION
    max_px = max(1.0, budget_bytes / _RASTER_BYTES_PER_PX)
    return max_px, budget_bytes


def _make_cancel_token() -> CancelToken:
    tok = CancelToken()
    def _handler(_sig, _frame):
        print("\nInterrupted — cancelling…", file=sys.stderr)
        tok.cancel()
    signal.signal(signal.SIGINT, _handler)
    return tok


def _load_profiles(files: List[str], quiet: bool = False,
                   load_traces: bool = True) -> List[SegyProfile]:
    profiles = []
    for f in files:
        if not quiet:
            print(f"  Loading {Path(f).name}…", file=sys.stderr)
        sd = load_profile(f, load_traces=load_traces)
        if sd.error:
            print(f"  WARNING: {Path(f).name}: {sd.error}", file=sys.stderr)
        profiles.append(sd)
    return profiles


def _load_profiles_parallel(files: List[str],
                            load_traces: bool = True) -> List[SegyProfile]:
    """
    Load multiple SEG-Y files concurrently with ThreadPoolExecutor.
    Each file is a separate segyio handle → no shared state, thread-safe.
    I/O-bound: threading gives ~Nx speedup for N files on SSD/NVMe.
    """
    if len(files) <= 1:
        return _load_profiles(files, load_traces=load_traces)

    n_workers = min(len(files), 4)   # cap at 4 to avoid disk thrashing
    results:   List[Optional[SegyProfile]] = [None] * len(files)

    def _load_one(idx_path):
        idx, path = idx_path
        print(f"  Loading {Path(path).name}…", file=sys.stderr)
        sd = load_profile(path, load_traces=load_traces)
        if sd.error:
            print(f"  WARNING: {Path(path).name}: {sd.error}", file=sys.stderr)
        return idx, sd

    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for idx, sd in pool.map(_load_one, enumerate(files)):
            results[idx] = sd

    return results


# ── accel ──────────────────────────────────────────────────────────────────────

def cmd_accel(args) -> None:
    """
    accel — Display active hardware acceleration layers.

    Shows GPU (CuPy/CUDA), pyfftw, CPU workers, and library versions.
    Install instructions are printed for any missing accelerator.
    """
    from ..core._backends import GPU, FFTW, N_WORKERS, _GPU_NAME, _FFTW_MSG

    info = accel_info()

    def _ok(v):  return "[OK]" if v else "[--]"

    print("\nTopassuite — acceleration status")
    print("-" * 50)
    print(f"  CPU cores   : {info['cpu_cores']} total, {N_WORKERS} workers active")
    print(f"  numpy       : {info['numpy']}")
    print(f"  scipy       : {info['scipy']}")
    print()
    print(f"  {_ok(GPU)}  GPU (CuPy)  : {_GPU_NAME}")
    if not GPU:
        print("              Install: conda install -c conda-forge cupy cudatoolkit=<ver>")
        print("              Verify : nvidia-smi   then   python -c 'import cupy; cupy.array([1])'")
    elif "cupy" in info:
        print(f"              cupy {info['cupy']}  CUDA {info.get('cuda','?')}")
    print()
    print(f"  {_ok(FFTW)}  pyfftw      : {_FFTW_MSG}")
    if not FFTW:
        print("              Install: conda install -c conda-forge pyfftw")
        print("              Effect : 2-5x faster Hilbert (envelope) and bandpass")
    print()

    if GPU and FFTW:
        print("  All accelerators active — optimal performance.")
    elif GPU:
        print("  GPU active. Install pyfftw for additional CPU speedup.")
    elif FFTW:
        print("  pyfftw active. Install CuPy+CUDA for GPU acceleration.")
    else:
        print("  Running on CPU only. Both GPU and pyfftw would improve speed.")


# ── info ───────────────────────────────────────────────────────────────────────

def cmd_info(args) -> None:
    """
    info FILE... [--json]

    Print metadata for each SEG-Y file. With --json, output is a JSON array.
    """
    results = []
    for f in args.files:
        md = load_metadata(f)
        results.append(md.to_dict())

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for md in results:
            print("\n" + "-" * 60)
            print(f"File     : {md['name']}")
            print(f"Traces   : {md['n_traces']}")
            print(f"Samples  : {md['ns']}  dt={md['dt_us']} µs")
            print(f"Duration : {md['dur_ms']:.0f} ms  ({md['total_km']:.2f} km)")
            print(f"Delay    : {md['delay_ms']} ms  "
                  f"(range {md['min_delay']:.0f}–{md['max_delay']:.0f} ms)")
            print(f"CRS      : {md['detected_crs'] or '(not detected)'}")
            print(f"CoordUnit: {md['coord_unit']}  ScalarCoord={md['scalar_coord']}")
            if md['error']:
                print(f"ERROR    : {md['error']}")


# ── reproject ─────────────────────────────────────────────────────────────────

def cmd_reproject(args) -> None:
    """
    reproject FILE... --src EPSG --dst EPSG [--unit-hint N] [--out-dir DIR]
    """
    tok      = _make_cancel_token()
    out_dir  = Path(args.out_dir) if args.out_dir else None

    for f in args.files:
        sd = load_profile(f, load_traces=False)
        if sd.error:
            print(f"  SKIP {Path(f).name}: {sd.error}", file=sys.stderr)
            continue

        print(f"\nReprojecting {sd.name}…", file=sys.stderr)

        def _prog(frac, msg, _sd=sd):
            _progress(frac, f"{_sd.name}: {msg}")

        try:
            out = reproject_one(sd, args.src, args.dst,
                                unit_hint=args.unit_hint,
                                log=_log, progress=_prog, cancel=tok)
            if out_dir:
                dest = out_dir / Path(out).name
                out_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(out, str(dest))
                out = str(dest)
            print(f"\n  ✔ {Path(out).name}", file=sys.stderr)
        except Cancelled:
            print("\n  Cancelled.", file=sys.stderr)
            sys.exit(1)
        except (CRSError, ReprojectionError) as exc:
            _err(str(exc))


# ── join-chain ────────────────────────────────────────────────────────────────

def cmd_join_chain(args) -> None:
    """
    join-chain FILE... [--dst EPSG] [--src EPSG] [--no-reproject] [--gap-km K]

    Routing logic:
      --no-reproject          → join_profiles (fast copy, no CRS change)
      --dst omitted           → join_profiles (fast copy, no CRS change)
      --src X --dst X (same)  → join_profiles (fast copy, auto-detected)
      --src X --dst Y         → reproject_chain (full reprojection)
    """
    import shutil as _shutil
    tok      = _make_cancel_token()
    profiles = _load_profiles(args.files)
    valid    = [p for p in profiles if not p.error]

    if not valid:
        _err("No valid profiles to join.")

    gap_km = args.gap_km if args.gap_km else None
    chains = detect_chains(valid, gap_km=gap_km)

    if not chains:
        _err("No chains detected.")

    # Decide fast path vs reprojection
    no_reproj = getattr(args, "no_reproject", False)
    src_str   = args.src or chains[0].profiles[0].detected_crs or "EPSG:4326"
    dst_str   = getattr(args, "dst", None)

    if dst_str is None or no_reproj:
        use_fast = True
    else:
        # Auto-detect identity: resolve both CRS and compare
        try:
            from pyproj import CRS as _CRS
            use_fast = _CRS.from_user_input(src_str) == _CRS.from_user_input(dst_str)
        except Exception:
            use_fast = (src_str == dst_str)

    for ch in chains:
        print(f"\nJoining chain: {ch.label}", file=sys.stderr)
        if use_fast:
            print("  Mode: fast copy (no reprojection)", file=sys.stderr)

        def _prog(frac, msg, _ch=ch):
            _progress(frac, f"{_ch.name[:40]}: {msg}")

        try:
            if use_fast:
                out = join_profiles(ch, out_path=args.out,
                                    log=_log, progress=_prog, cancel=tok)
            else:
                out = reproject_chain(ch, src_str, dst_str,
                                      unit_hint=getattr(args, "unit_hint", 2),
                                      log=_log, progress=_prog, cancel=tok)
                if args.out:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    _shutil.move(out, args.out)
                    out = args.out
            print(f"\n  ✔ {Path(out).name}", file=sys.stderr)
        except Cancelled:
            print("\n  Cancelled.", file=sys.stderr)
            sys.exit(1)
        except (CRSError, ReprojectionError) as exc:
            _err(str(exc))


# ── check ───────────────────────────────────────────────────────────────────────

def _scan_anomalies(prof) -> List[str]:
    """Heuristic QC scan over a loaded SegyProfile → list of human-readable
    warnings. Catches the metadata problems the Headers editor exists to fix:
    a zero/implausible sample interval, missing geometry, an undetected CRS,
    duplicate-timestamp purging, and dead (all-zero) trace data."""
    issues: List[str] = []

    dt = int(getattr(prof, "dt_us", 0) or 0)
    if dt <= 0:
        issues.append("sample interval dt is 0 µs (binary header field unset) — "
                      "fix with `patch-header --dt <µs>`")
    elif dt > 20000:
        issues.append(f"sample interval dt={dt} µs is unusually large "
                      "(verify the header is in microseconds)")

    ns = int(getattr(prof, "ns", 0) or 0)
    if ns <= 0:
        issues.append("samples-per-trace ns is 0 — this cannot be safely patched "
                      "in place (it would corrupt trace boundaries); reprocess/"
                      "rewrite the file with the correct sample count instead")

    if not getattr(prof, "detected_crs", None):
        issues.append("no CRS detected from the coordinate headers — set one "
                      "before reprojecting (`reproject --src …`)")

    purged = int(getattr(prof, "n_purged", 0) or 0)
    if purged > 0:
        issues.append(f"{purged} consecutive duplicate-timestamp trace(s) were "
                      "auto-purged on load (acquisition double-stamping)")

    lons = np.asarray(getattr(prof, "lons", []), dtype=float)
    if lons.size == 0 or np.allclose(lons, 0.0):
        issues.append("source coordinates are all zero / missing — the navigation "
                      "track and map will be empty")

    data = getattr(prof, "data", None)
    if data is not None:
        if not np.any(np.isfinite(data)) or np.allclose(np.nan_to_num(data), 0.0):
            issues.append("trace data is entirely zero / non-finite (dead section)")
        elif np.isnan(data).any():
            issues.append("trace data contains NaN samples")

    return issues


def cmd_check(args) -> None:
    """
    check FILE... [--full-text]

    Diagnostic scan: prints the EBCDIC textual header, the key binary-header
    fields, basic statistics, and a heuristic list of detected anomalies. The
    headless counterpart of the GUI Headers tab.
    """
    for path in args.files:
        prof = load_profile(path, load_traces=not getattr(args, "no_stats", False))
        print("\n" + "=" * 70)
        print(f"FILE: {prof.name}")
        print("=" * 70)

        if prof.error:
            print(f"  ERROR: {prof.error}")
            continue

        # ── EBCDIC textual header (3200 bytes / 40 cards) ─────────────────────
        text = getattr(prof, "text_header", "") or ""
        if text:
            print("\n-- Textual header (EBCDIC 3200) " + "-" * 38)
            if getattr(args, "full_text", False):
                for line in text.split("\n"):
                    print(f"  {line}")
            else:
                # First 6 cards is plenty to recognise the survey; --full-text dumps all.
                for line in text.split("\n")[:6]:
                    print(f"  {line}")
                print("  …  (use --full-text for all 40 cards)")

        # ── Binary / file-level fields ────────────────────────────────────────
        print("\n-- Binary header & stats " + "-" * 45)
        print(f"  Traces (kept)     : {prof.n_traces}")
        if int(getattr(prof, "n_purged", 0) or 0) > 0:
            print(f"  Traces (original) : {prof.original_n_traces}  "
                  f"({prof.n_purged} purged)")
        print(f"  Samples / trace   : {prof.ns}")
        print(f"  Sample interval   : {prof.dt_us} µs  "
              f"({1e6 / prof.dt_us:.0f} Hz)" if prof.dt_us else
              f"  Sample interval   : {prof.dt_us} µs")
        print(f"  Record length     : {prof.dur_ms:.1f} ms")
        print(f"  Track length      : {prof.total_km:.2f} km")
        print(f"  Delay (first)     : {prof.delay_ms} ms")
        print(f"  CRS detected      : {prof.detected_crs or '(none)'}")
        print(f"  Coord unit / scal : {prof.coord_unit} / {prof.scalar_coord}")

        data = getattr(prof, "data", None)
        if data is not None:
            finite = np.nan_to_num(data)
            print(f"  Amplitude range   : [{finite.min():.4g}, {finite.max():.4g}]")
            print(f"  Amplitude RMS     : {np.sqrt(np.mean(finite ** 2)):.4g}")

        # ── Anomalies ─────────────────────────────────────────────────────────
        issues = _scan_anomalies(prof)
        print("\n-- Anomaly scan " + "-" * 54)
        if not issues:
            print("  OK — no anomalies detected.")
        else:
            for msg in issues:
                print(f"  ⚠ {msg}")


# ── patch-header ─────────────────────────────────────────────────────────────────

def cmd_patch_header(args) -> None:
    """
    patch-header FILE --dt µs [--text FILE] [--dry-run]

    In-place (segyio r+) patch of the binary header. Mirrors the GUI Headers
    editor's safe writer: changing --dt also mass-propagates the new sample
    interval to EVERY trace header (TRACE_SAMPLE_INTERVAL). MODIFIES THE FILE.

    ns (samples/trace) is intentionally NOT patchable here: changing it
    without resizing every trace's data block would corrupt the file (every
    trace boundary would misalign for any reader).
    """
    import segyio

    path = args.file
    md = load_metadata(path)
    if md.error:
        _err(f"Cannot read {path}: {md.error}")

    binary_updates: dict = {}
    if getattr(args, "dt", None) is not None:
        binary_updates[int(segyio.BinField.Interval)] = int(args.dt)

    text_header = None
    if getattr(args, "text", None):
        try:
            with open(args.text, "r", encoding="utf-8", errors="replace") as fh:
                text_header = fh.read()
        except OSError as exc:
            _err(f"Cannot read --text file: {exc}")

    if not binary_updates and text_header is None:
        _err("Nothing to patch. Provide --dt and/or --text FILE.")

    md_dict = md.to_dict()
    print(f"\nPatching header of {md_dict['name']}")
    print(f"  Current : dt={md_dict['dt_us']} µs, ns={md_dict['ns']}")
    if args.dt is not None:
        print(f"  → dt    : {args.dt} µs  (also propagated to all "
              f"{md_dict['n_traces']} trace headers)")
    if text_header is not None:
        print(f"  → text  : replaced from {Path(args.text).name}")

    if getattr(args, "dry_run", False):
        print("  DRY-RUN — no changes written.")
        return

    ok, errmsg = patch_segy_headers(
        path, text_header=text_header, binary_updates=binary_updates or None)
    if not ok:
        _err(f"Patch failed: {errmsg}")
    print(f"  ✔ {Path(path).name} patched in place.")


# ── process (headless DSP pipeline) ──────────────────────────────────────────────

# pipeline-token → (n_args, callable(data, obj, float_args) -> data). Each op
# delegates to the SAME core.apply_* function the GUI DSP nodes wrap, so the
# headless pipeline is bit-identical to the interactive one.
def _op_bandpass(data, obj, a):
    return apply_bandpass(data, a[0], a[1], obj.dt_us)

def _op_whiten(data, obj, a):
    return apply_spectral_whitening(data, obj.dt_us, a[0], a[1], a[2])

def _op_agc(data, obj, a):
    return apply_agc(data, a[0], obj.dt_us)

def _op_tvg(data, obj, a):
    return apply_tvg(data, a[0], obj.dt_us)

def _op_decon(data, obj, a):
    return apply_predictive_decon(data, obj.dt_us, a[0], a[1], a[2])

def _op_swell(data, obj, a):
    return apply_swell_filter(data, int(a[0]), a[1], obj.dt_us)

def _op_water_mute(data, obj, a):
    return apply_water_mute(data, a[0], a[1], obj.dt_us)

def _op_align(data, obj, a):
    return apply_delay_alignment(data, obj.delays, obj.min_delay, obj.dt_us,
                                 fill_value=0.0)

# name → (expected_arg_count, fn, signature_help)
_PIPELINE_OPS = {
    "bandpass":   (2, _op_bandpass,   "bandpass(flo_hz,fhi_hz)"),
    "whiten":     (3, _op_whiten,     "whiten(flo_hz,fhi_hz,smooth_hz)"),
    "agc":        (1, _op_agc,        "agc(window_ms)"),
    "tvg":        (1, _op_tvg,        "tvg(alpha)"),
    "decon":      (3, _op_decon,      "decon(op_ms,gap_ms,white_pct)"),
    "swell":      (2, _op_swell,      "swell(window_traces,max_shift_ms)"),
    "water_mute": (2, _op_water_mute, "water_mute(threshold_pct,margin_ms)"),
    "align":      (0, _op_align,      "align()"),
}


def _parse_pipeline(spec: str) -> list:
    """Parse ``"bandpass(1000,8000),whiten(...),agc(200)"`` into an ordered list
    of ``(name, [float, …], fn)`` tuples. Raises ValueError on any problem."""
    import re
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty --pipeline string")

    ops = []
    # Match name(args) groups; preset is handled separately (string arg).
    for m in re.finditer(r"(\w+)\s*\(([^)]*)\)", spec):
        name = m.group(1).lower()
        raw  = m.group(2).strip()

        if name == "preset":
            key = raw.strip().strip("'\"")
            ops.append((name, key, None))
            continue

        if name not in _PIPELINE_OPS:
            raise ValueError(
                f"unknown pipeline op {name!r}. Known: "
                + ", ".join(sorted(list(_PIPELINE_OPS) + ['preset'])))
        n_args, fn, sig = _PIPELINE_OPS[name]
        nums = [p for p in (x.strip() for x in raw.split(",")) if p != ""]
        if len(nums) != n_args:
            raise ValueError(f"{name}: expected {n_args} argument(s) — use {sig}")
        try:
            vals = [float(x) for x in nums]
        except ValueError:
            raise ValueError(f"{name}: arguments must be numeric — use {sig}")
        ops.append((name, vals, fn))

    if not ops:
        raise ValueError(
            "could not parse any op from --pipeline. Example: "
            "\"bandpass(1000,8000),whiten(1000,8000,300),agc(200)\"")
    return ops


def _apply_pipeline(data, obj, ops, progress=None) -> np.ndarray:
    """Run a parsed pipeline over ``data`` (ns × n_traces), returning a new array."""
    n = len(ops)
    for i, (name, arg, fn) in enumerate(ops):
        if progress:
            progress(i / max(1, n), f"{name}")
        if name == "preset":
            data = apply_filter_preset(data, arg, obj.dt_us)
        else:
            data = fn(data, obj, arg)
    if progress:
        progress(1.0, "done")
    return np.asarray(data, dtype=np.float32)


def _compute_keep_idx(src_f):
    """Re-derive the timestamp-dedup keep indices for a segyio handle, reusing the
    SAME core mask so a processed output stays aligned with the cleaned section.
    Returns a 1-D int array of kept trace indices, or None when nothing is purged."""
    import segyio
    from ..core.io_segy import _timestamp_dedup_mask
    TF = segyio.TraceField
    attr = src_f.attributes
    keep = _timestamp_dedup_mask(
        attr(TF.DayOfYear)[:], attr(TF.HourOfDay)[:],
        attr(TF.MinuteOfHour)[:], attr(TF.SecondOfMinute)[:],
        attr(TF.SourceX)[:], attr(TF.SourceY)[:])
    if keep is None:
        return None
    return np.nonzero(keep)[0]


def _write_processed_segy(src_path: str, out_path: str, processed: np.ndarray,
                          progress=None) -> None:
    """Write a processed (ns × n_traces) matrix to a new SEG-Y, preserving all
    geometry/headers. Two paths:

      * no duplicate-timestamp purge → copy the source byte-for-byte then
        overwrite trace samples in place (every header preserved exactly);
      * purge occurred → create a fresh file with only the kept traces, copying
        each kept trace's header from the source (keeps data ↔ header aligned).
    """
    import shutil
    import segyio

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_out = processed.shape[1]

    # First, resolve the keep-mask and (if purging) write the deduped file — all
    # while the source handle is open. The fast path defers the copy until the
    # handle is released, so Windows never blocks on an open read handle.
    with segyio.open(src_path, ignore_geometry=True) as src:
        keep_idx = _compute_keep_idx(src)
        if keep_idx is not None:
            spec = segyio.tools.metadata(src)
            spec.tracecount = n_out
            with segyio.create(out_path, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin.update(src.bin)
                for j, si in enumerate(keep_idx[:n_out]):
                    dst.header[j] = src.header[int(si)]
                    dst.trace[j] = np.ascontiguousarray(
                        processed[:, j].astype(np.float32))
            return

    # Fast path: structural copy (source handle now closed), then overwrite samples.
    shutil.copyfile(src_path, out_path)
    with segyio.open(out_path, mode="r+", ignore_geometry=True) as f:
        for i in range(n_out):
            if progress and (i % 256 == 0):
                progress(i / max(1, n_out), "writing")
            f.trace[i] = np.ascontiguousarray(processed[:, i].astype(np.float32))
        f.flush()


def cmd_process(args) -> None:
    """
    process INPUT OUTPUT --pipeline "op(args),op(args),…"

    Runs the headless DSP pipeline end-to-end on one SEG-Y file and writes a new
    SEG-Y with the processed samples (all geometry/headers preserved). The
    pipeline string uses the SAME stages as the GUI nodes.
    """
    tok = _make_cancel_token()
    timer = PhasedTimer(enabled=getattr(args, "timeit", False))

    try:
        ops = _parse_pipeline(args.pipeline)
    except ValueError as exc:
        _err(str(exc))

    print(f"\nPipeline: {args.pipeline}", file=sys.stderr)
    for name, arg, _fn in ops:
        print(f"  • {name}({arg if name == 'preset' else ', '.join(f'{v:g}' for v in arg)})",
              file=sys.stderr)

    with timer.phase("loading"):
        prof = load_profile(args.input, load_traces=True)
    if prof.error:
        _err(f"Cannot load {args.input}: {prof.error}")

    with timer.phase("processing"):
        processed = _apply_pipeline(prof.data, prof, ops, progress=_progress)
        print("", file=sys.stderr)

    with timer.phase("writing"):
        print(f"Writing {Path(args.output).name}…", file=sys.stderr)
        _write_processed_segy(args.input, args.output, processed, progress=_progress)
        print("", file=sys.stderr)

    print(f"  ✔ {Path(args.output).name}  "
          f"({processed.shape[1]} traces × {processed.shape[0]} samples)",
          file=sys.stderr)
    if getattr(args, "timeit", False):
        print(timer.report(prefix="\n  "), file=sys.stderr)


# ── batch-export ─────────────────────────────────────────────────────────────────

_SEGY_EXTS = (".sgy", ".segy", ".seg")


def _expand_segy_inputs(paths: List[str]) -> List[str]:
    """Expand a mix of directories and files into a sorted list of SEG-Y files.
    A directory contributes every *.sgy/*.segy/*.seg it directly contains."""
    out: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for ext in _SEGY_EXTS:
                out.extend(str(x) for x in sorted(pp.glob(f"*{ext}")))
                out.extend(str(x) for x in sorted(pp.glob(f"*{ext.upper()}")))
        elif pp.is_file():
            out.append(str(pp))
        else:
            print(f"  WARNING: no such path: {p}", file=sys.stderr)
    # De-duplicate while preserving order.
    seen, uniq = set(), []
    for f in out:
        k = os.path.normcase(os.path.abspath(f))
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


def cmd_batch_export(args) -> None:
    """
    batch-export DIR_OR_FILES… --out DIR [--format pdf] [--cmap …] [--ve …] …

    Convenience wrapper over the export-image engine: expands directories into
    SEG-Y files and renders each one into --out using the SAME vectorised
    max-abs-pooling renderer (custom colormaps, fixed VE, RAM safety, vector
    interpolation). One file's failure is reported but does not abort the batch.
    """
    files = _expand_segy_inputs(args.inputs)
    if not files:
        _err("No SEG-Y files found in the given path(s).")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format or "pdf"

    print(f"\nBatch export: {len(files)} file(s) → {out_dir}  "
          f"[format={fmt}, cmap={args.cmap or 'Viridis'}]", file=sys.stderr)

    ok = fail = 0
    for idx, f in enumerate(files, 1):
        stem = Path(f).stem
        out_path = out_dir / f"{stem}.{fmt}"
        print(f"\n[{idx}/{len(files)}] {Path(f).name}", file=sys.stderr)

        # Drive the existing single-file engine: one file, explicit output path.
        args.files = [f]
        args.out   = str(out_path)
        args.chain = False
        try:
            cmd_export_image(args)
            ok += 1
        except SystemExit as exc:               # per-file _err()/exit → keep going
            if exc.code not in (0, None):
                print(f"  SKIP {Path(f).name}: export failed (exit {exc.code})",
                      file=sys.stderr)
                fail += 1
        except Exception as exc:
            print(f"  SKIP {Path(f).name}: {exc}", file=sys.stderr)
            fail += 1

    print(f"\nBatch complete: {ok} ok, {fail} failed → {out_dir}", file=sys.stderr)
    if fail and not ok:
        sys.exit(1)


# ── Quality presets ────────────────────────────────────────────────────────────

_QUALITY_PRESETS = {
    # name   → DPI  (figheight always 7.0 unless --auto-height or --figheight)
    "screen": 150,
    "print":  300,
    "high":   600,
    "ultra":  900,
}

# ── export-image ──────────────────────────────────────────────────────────────

def cmd_export_image(args) -> None:
    """
    export-image FILE... [options]

    New in this version
    -------------------
    --quality screen|print|high|ultra  — DPI preset (150/300/600/900)
    --auto-height                      — figheight = ns/dpi (1 sample ≈ 1 px)
    --x-tick KM                        — distance tick every KM km
    --t-tick MS                        — time tick every MS ms
    --grid                             — overlay grid
    --no-axes                          — save raw RGBA pixels, no matplotlib frame
    --title TEXT                       — custom title
    --clip-lo P                        — lower clip percentile (default 0)
    """
    from ..viz.render import (render_profile_figure, render_chain_figure,
                              save_figure, save_raw_rgba)
    from ..core.constants import FILTER_PRESETS

    # ── Timer ──────────────────────────────────────────────────────────────
    timer = PhasedTimer(enabled=getattr(args, "timeit", False))

    # ── Load profiles header-only (parallel for multiple files) ───────────────
    # Trace matrices are JIT-loaded per file in the export loop below so that
    # only one profile's full matrix is resident at a time (peak RAM = 1 file,
    # not N files). All metadata needed for figsize / chain-detect is present
    # in header-only stubs. The parallel speedup for I/O is preserved.
    with timer.phase("loading"):
        if len(args.files) > 1:
            profiles = _load_profiles_parallel(args.files, load_traces=False)
        else:
            profiles = _load_profiles(args.files, load_traces=False)

    valid = [p for p in profiles if not p.error]
    if not valid:
        _err("No valid profiles to export.")

    # ── Resolve DPI (quality preset < explicit --dpi) ──────────────────────
    quality = getattr(args, "quality", None)
    explicit_dpi = args.dpi  # None if not set (we changed default to None)
    if explicit_dpi is not None:
        dpi = explicit_dpi
    elif quality:
        dpi = _QUALITY_PRESETS[quality]
    else:
        dpi = 150  # built-in default

    # ── Scale / proportion parameters ─────────────────────────────────────────
    x_scale  = getattr(args, "x_scale",  None)    # km per inch  (horizontal)
    y_scale  = getattr(args, "y_scale",  None)    # ms per inch  (vertical, legacy)
    velocity = getattr(args, "velocity", 1500.0)  # m/s for depth conversion
    ratio    = getattr(args, "ratio",    None)     # desired W:H display ratio
    ve_target = getattr(args, "ve",      None)     # FIXED vertical exaggeration
    max_aspect = getattr(args, "max_aspect", None) # noodle guard on the --ve path
    x_axis   = getattr(args, "x_axis", "distance")  # distance | trace | km

    # --ve sets a length-INDEPENDENT vertical exaggeration. It derives figheight
    # from the SAME physical km/in as the width (h = ve · depth_km / x_scale), so
    # it can only be expressed relative to a physical horizontal scale → it
    # requires --x-scale (a trace-based width has no fixed km/in to anchor VE to).
    if ve_target is not None and x_scale is None:
        _err("--ve requires --x-scale (the vertical exaggeration is defined "
             "relative to the horizontal km/in scale). Example: --x-scale 2 --ve 67")

    # figsize is deferred to _figsize() below; set defaults for fallback
    auto_height   = getattr(args, "auto_height", False)
    raw_figheight = getattr(args, "figheight", None)

    # figheight for the non-scale paths (resolved upfront)
    if raw_figheight is not None:
        _fallback_figheight = raw_figheight
    elif auto_height:
        _fallback_figheight = valid[0].ns / dpi
        print(f"  auto-height: ns={valid[0].ns}, dpi={dpi} "
              f"→ figheight={_fallback_figheight:.2f} in "
              f"(~{int(_fallback_figheight*dpi)} px)", file=sys.stderr)
    else:
        _fallback_figheight = 7.0

    px_per_trace = getattr(args, "px_per_trace", 2.0)
    no_axes      = getattr(args, "no_axes", False)
    fmt          = args.format or "png"

    # ── Build processing params ────────────────────────────────────────────
    params: dict = {
        "decon":      False,
        "decon_op":   10.0,
        "decon_gap":  1.0,
        "decon_wn":   0.1,
        "filt":       bool(args.bandpass),
        "flo":        args.bandpass[0] if args.bandpass else 100.0,
        "fhi":        args.bandpass[1] if args.bandpass else 5000.0,
        "preset":     args.preset or "",
        "tvg":        args.tvg is not None,
        "tvg_alpha":  args.tvg if args.tvg is not None else 0.0,
        "agc":        args.agc,
        "agc_win":    50.0,
        "align":      args.align,
        "fill_value": 0.0 if args.fill_zero else np.nan,
        "clip":       args.clip,
        "clip_lo":    getattr(args, "clip_lo", 0.0),
        "cmap":       _resolve_cmap(args.cmap),
        "inv_cmap":   args.invert,
        "fix":        args.fix is not None,
        "fix_iv":     args.fix if args.fix is not None else 5,
        # Preserve the CLI's historical behaviour: chain exports always draw
        # the file-seam boundary lines. (The GUI export dialog defaults OFF.)
        "draw_file_boundaries": True,
    }

    pdf_page = getattr(args, "pdf_page", "auto") or "auto"

    # ── Build colour theme ─────────────────────────────────────────────────
    from ..viz.render import build_theme as _build_theme
    colors = _build_theme(
        theme         = getattr(args, "theme",         "dark"),
        bg_color      = getattr(args, "bg_color",      None),
        text_color    = getattr(args, "text_color",    None),
        axes_bg_color = getattr(args, "axes_bg_color", None),
    )

    # ── Render options (axes, ticks, grid, title, time axis, margins, theme) ─
    render_opts: dict = {
        "x_tick_km":        getattr(args, "x_tick",          None),
        "t_tick_ms":        getattr(args, "t_tick",          None),
        "show_grid":        getattr(args, "grid",            False),
        "title_override":   getattr(args, "title",           None),
        "clip_lo":          getattr(args, "clip_lo",         0.0),
        "time_tick_min":    getattr(args, "time_ticks",      None),
        # New options
        "margin_top_ms":    getattr(args, "margin_top",      0.0),
        "margin_bottom_ms": getattr(args, "margin_bottom",   0.0),
        "time_fmt":         getattr(args, "time_fmt",        "hhmm"),
        "time_font_size":   getattr(args, "time_font_size",  6.0),
        "time_align":       getattr(args, "time_align",      "left"),
        "fix_font_size":    getattr(args, "fix_font_size",  5.0),
        "fix_bbox_alpha":   getattr(args, "fix_bbox_alpha", 0.12),
        "fix_color":        getattr(args, "fix_color",      None),
        "colors":           colors,
        "x_axis":           x_axis,          # distance | trace | km (Task 6/7)
    }

    # ── Rasteriser RAM budget (prevents ArrayMemoryError) ──────────────────
    mem_budget_gb = getattr(args, "mem_budget_gb", None)
    _max_px, _budget_bytes = _safe_pixel_budget(mem_budget_gb)
    _budget_src = (f"--mem-budget-gb {mem_budget_gb:g}" if mem_budget_gb
                   else f"{_RAM_SAFE_FRACTION:.0%} of {_available_ram_bytes()/1024**3:.1f} GB free")
    print(f"  RAM budget: {_budget_bytes/1024**3:.1f} GB ({_budget_src}) "
          f"→ max {_max_px/1e6:.0f} Mpx @ {_RASTER_BYTES_PER_PX:.0f} B/px peak",
          file=sys.stderr)

    # ── RAM validation: warn if the configured budget exceeds free RAM ─────────
    # Prominent terminal alert before rendering so the user can Ctrl-C and lower
    # --mem-budget-gb rather than risk an out-of-memory crash. Only an explicit
    # budget is checked (the auto 55%-of-free path is safe by construction).
    if mem_budget_gb and mem_budget_gb > 0:
        _free_gb = _available_ram_bytes() / 1024 ** 3
        if mem_budget_gb > _free_gb:
            print("  " + "!" * 68, file=sys.stderr)
            print(f"  WARNING: --mem-budget-gb {mem_budget_gb:g} GB EXCEEDS the "
                  f"{_free_gb:.1f} GB of free RAM detected on this machine.",
                  file=sys.stderr)
            print("  The export may run out of memory and crash. Lower "
                  "--mem-budget-gb (or close other apps) to be safe.",
                  file=sys.stderr)
            print("  " + "!" * 68, file=sys.stderr)

    # ── Figure size helper ─────────────────────────────────────────────────
    def _figsize(source) -> tuple:
        """
        Compute figure (width_in, height_in) applying this priority:

        WIDTH
          1. --x-scale X  → total_km / X
          2. fallback     → max(8, n_traces × px_per_trace / dpi)

        HEIGHT
          1. --ratio R         → width / R                  (any width mode)
          2. --x-scale + --velocity (no --ratio/--y-scale)
                               → depth_km / X  (VE=1, true physical scale)
          3. --y-scale Y       → record_ms / Y
          4. fallback          → _fallback_figheight (from --figheight/--auto-height/default)
        """
        total_km  = source.total_km
        record_ms = source.ns * source.dt_us / 1000.0
        depth_km  = record_ms * velocity / 2_000_000   # TWT → one-way → km

        # ── Width ─────────────────────────────────────────────────────────
        if x_axis != "distance":
            # PER-TRACE modes (--trace-axis / --km-no-stretch): draw exactly one
            # column per trace → trace-based width, the horizontal km scale (and
            # --x-scale) is intentionally ignored (Task 6/7).
            w = source.n_traces * px_per_trace / dpi
            _wmode = f"per-trace ({x_axis}, {px_per_trace:.0f} px/trace)"
        elif x_scale is not None:
            # PHYSICAL SCALE (geophysically correct): km per inch, honoured
            # EXACTLY — no clamps, no minimums. A short line therefore produces a
            # proportionally narrow figure (see the micro-figure warning below).
            w = total_km / x_scale
            _wmode = f"--x-scale {x_scale} km/in"
        else:
            # TRACE-BASED AUTO: every trace maps to exactly px_per_trace pixels
            # regardless of km (figwidth = n_traces · px_per_trace / dpi). No 8-in
            # floor — the width follows the trace count.
            w = source.n_traces * px_per_trace / dpi
            _wmode = f"auto {px_per_trace:.0f} px/trace"

        # ── Height ────────────────────────────────────────────────────────
        if ve_target is not None and x_scale is not None:
            # FIXED vertical exaggeration (length-INDEPENDENT, the correct way to
            # keep many lines comparable): h = VE · depth_km / x_scale. Because
            # depth_km = record_ms · v / 2e6 depends ONLY on the acquisition window
            # (constant across a survey) and x_scale is a constant, the height —
            # and therefore the VE — is identical for every line regardless of its
            # length or trace count. Overrides --ratio / --y-scale.
            h = max(0.5, ve_target * depth_km / x_scale)
            _mode = f"--ve {ve_target:g} (fixed, length-independent)"
            # "Infinite-noodle" guard: a constant VE makes very long lines extremely
            # wide vs tall. If the aspect would exceed --max-aspect, lock it there
            # by RAISING the height (overriding the VE only for that extreme line).
            if max_aspect is not None and max_aspect > 0 and w / h > max_aspect:
                h = w / max_aspect
                _mode = (f"--ve {ve_target:g} → clamped to --max-aspect "
                         f"{max_aspect:g}:1 (VE raised to save format)")

        elif ratio is not None:
            h = max(0.5, w / ratio)
            _mode = f"--ratio {ratio}"

        elif x_scale is not None and y_scale is None:
            # Physical scale: height uses same km/in as width → VE=1
            h = max(0.5, depth_km / x_scale)
            _mode = f"velocity {velocity:.0f} m/s, VE=1 (true scale)"

        elif y_scale is not None:
            h = max(0.5, record_ms / y_scale)
            _mode = f"--y-scale {y_scale} ms/in"

        else:
            h     = _fallback_figheight
            _mode = "default figheight"

        # ── VE info ───────────────────────────────────────────────────────
        # VE = (horizontal km/in) / (vertical km/in)
        # vertical km/in = depth_km / h
        h_km_per_in = depth_km / h if h > 0 else 1e-9
        w_km_per_in = total_km / w if w > 0 else 1e-9
        ve = h_km_per_in / w_km_per_in   # >1 means VE applied (depth stretched)
        # Note: VE>1 means vertical is exaggerated (depth appears deeper than real)
        # In seismic display convention: VE = (horiz_scale) / (vert_scale)
        # We print the inverse: how many times the depth is stretched
        ve_display = w_km_per_in / h_km_per_in  # < 1 for "squeezed depth"...
        # Convention: VE = horizontal_scale / vertical_scale
        # vertical_scale = depth_km / h (km/in)
        # horizontal_scale = total_km / w (km/in)
        # VE = vertical_scale / horizontal_scale  (>1 = depth squeezed, normal in seismic)
        # Actually seismic VE is: 1px vertical = dt*v/2 (metres), 1px horizontal = trace_spacing
        # Let's simplify: VE_display = (km/in horizontal) / (km/in vertical)
        # VE=1 → same km/in both axes (true scale, would look like a thin strip)
        # VE=50 → depth looks 50× more than reality
        true_ve = w_km_per_in / h_km_per_in  # how much vert is exaggerated vs horiz

        eff_ppt = (w * dpi / source.n_traces) if source.n_traces else 0.0
        print(
            f"  scale [W:{_wmode} | H:{_mode}]: "
            f"{total_km:.1f} km wide, {depth_km*1000:.0f} m deep → "
            f"figwidth {w:.2f} in × {h:.2f} in  "
            f"({eff_ppt:.1f} px/trace, ratio {w/h:.1f}:1, VE={true_ve:.0f}× "
            f"at v={velocity:.0f} m/s)",
            file=sys.stderr
        )

        # ── RAM-safe coupled down-scale (preserves aspect ratio AND VE) ─────
        # The peak rasteriser footprint scales with the OUTPUT pixel count
        # (figw·dpi × figh·dpi). When the requested figure would blow the RAM
        # budget we multiply BOTH dimensions by a single factor s = √(budget/req).
        # Because s hits width and height equally, w/h is unchanged → the aspect
        # ratio is identical AND the vertical exaggeration (VE = (total_km/depth_km)
        # · h/w) is mathematically identical; only the pixel density drops. This is
        # the maximum-quality fit: s is chosen to land exactly on the budget, never
        # lower, so a line is decimated by the smallest amount that avoids the
        # numpy ArrayMemoryError — and two different lines stay geologically
        # comparable (same proportions, same VE).
        req_px = (w * dpi) * (h * dpi)
        if req_px > _max_px:
            s = (_max_px / req_px) ** 0.5
            w_old, h_old = w, h
            w *= s
            h *= s
            print(
                f"  RAM-SAFE: {req_px/1e6:.0f} Mpx > budget {_max_px/1e6:.0f} Mpx "
                f"→ scaled ×{s:.3f} (both axes): "
                f"{w_old:.2f}×{h_old:.2f} in → {w:.2f}×{h:.2f} in "
                f"({int(w*dpi)}×{int(h*dpi)} px). Aspect & VE preserved; "
                f"px/trace {eff_ppt:.1f}→{(w*dpi/source.n_traces if source.n_traces else 0):.1f}. "
                f"More RAM? raise --mem-budget-gb.",
                file=sys.stderr
            )

        # ── Micro-figure guard: WARN, never clamp ──────────────────────────
        # matplotlib needs room for the title/axes/labels/colorbar; below ~2.5 in
        # those decorations overflow (the "Tight layout not applied" warning). We
        # keep the user's true figwidth and explain the options instead of
        # silently changing it (Rule: do not change the output format silently).
        if w < 2.5:
            if x_scale is not None:
                hint = ("drop --x-scale to use trace-based auto-scaling "
                        f"(would give {source.n_traces * px_per_trace / dpi:.2f} in), "
                        "or add --no-axes for a pure 1:1 raster export")
            else:
                hint = ("raise --px-per-trace, or add --no-axes for a pure 1:1 "
                        "raster export")
            print(
                f"  WARNING: figwidth {w:.2f} in < 2.5 in — too narrow for the "
                f"matplotlib axes/title/colorbar (decorations will overflow). The "
                f"figure is NOT clamped. To fix: {hint}.",
                file=sys.stderr
            )
        return (w, h)

    def _eff_px_per_trace(source, fs: tuple) -> float:
        """Effective px/trace for no-axes path (always computed from figsize)."""
        return fs[0] * dpi / source.n_traces

    def _eff_figheight(fs: tuple) -> float:
        return fs[1]

    def _px_str(w_in: float, h_in: float) -> str:
        return (f"{int(w_in*dpi)}x{int(h_in*dpi)} px, "
                f"{w_in:.1f}x{h_in:.1f} in @ {dpi} DPI")

    # ── Export helper ──────────────────────────────────────────────────────
    import traceback as _tb

    def _export_one(source, data, out_path, is_chain: bool) -> None:
        try:
            fs_tuple    = _figsize(source)
            eff_ppt     = _eff_px_per_trace(source, fs_tuple)
            eff_fheight = _eff_figheight(fs_tuple)

            if no_axes:
                # ── Raw RGBA path: zero margins, exact px/trace mapping ──
                save_raw_rgba(source, data, out_path, params, render_opts,
                              px_per_trace=eff_ppt, dpi=dpi,
                              figheight=eff_fheight, is_chain=is_chain,
                              pdf_page=pdf_page)
                w_px = int(source.n_traces * eff_ppt)
                h_px = int(eff_fheight * dpi)
                ext_ = Path(out_path).suffix.lower()
                if ext_ == ".pdf":
                    pg  = pdf_page.upper() if pdf_page else "auto"
                    tag = f"no-axes PDF, page={pg}"
                else:
                    tag = "no-axes, exact 1:1"
                print(f"  OK {Path(out_path).name}  "
                      f"({w_px}x{h_px} px, {tag})",
                      file=sys.stderr)
            else:
                # ── Matplotlib path: axes, labels, colorbar ─────────────
                if is_chain:
                    fig = render_chain_figure(
                        source, data, params,
                        figsize=fs_tuple, dpi=dpi, **render_opts)
                else:
                    fig = render_profile_figure(
                        source, data, params,
                        figsize=fs_tuple, dpi=dpi, **render_opts)
                sz = fig.get_size_inches()
                # Pass pdf_page for matplotlib path (rescales figure to fit paper)
                save_figure(fig, out_path, dpi=dpi, fmt=fmt, pdf_page=pdf_page)
                print(f"  OK {Path(out_path).name}  {_px_str(*sz)}",
                      file=sys.stderr)

        except Exception as exc:
            print(f"  ERROR exporting {Path(out_path).name}: {exc}",
                  file=sys.stderr)
            print(_tb.format_exc(), file=sys.stderr)
            sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────────────
    if args.chain:
        with timer.phase("chain-detect"):
            chains = detect_chains(valid, gap_km=None)
        for ch in chains:
            print(f"\nExporting chain: {ch.label}", file=sys.stderr)
            with timer.phase("processing"):
                # Chains assemble their stitched matrix lazily (memory-flat
                # detection). Build it now — constituents are already loaded in
                # the CLI, so this just concatenates the in-RAM arrays.
                ch.load_chain_traces()
                data = process_chain_data(ch, params)
            out = args.out or str(
                Path(ch.profiles[0].path).with_suffix(f".{fmt}"))
            with timer.phase("render+save"):
                _export_one(ch, data, out, is_chain=True)
    else:
        for sd in valid:
            print(f"\nExporting {sd.name}…", file=sys.stderr)
            with timer.phase("processing"):
                full = load_profile(sd.path, load_traces=True)
                if full.error:
                    print(f"  ERROR reloading {sd.name}: {full.error}",
                          file=sys.stderr)
                    sys.exit(1)
                data = process_profile_data(full, params)
                full.data = None    # release raw matrix; processed data stays
            out = args.out or str(Path(sd.path).with_suffix(f".{fmt}"))
            with timer.phase("render+save"):
                _export_one(full, data, out, is_chain=False)

    if getattr(args, "timeit", False):
        print(timer.report(prefix="\n  "), file=sys.stderr)


def _resolve_cmap(name: Optional[str]) -> str:
    if not name:
        return "Viridis"
    # Accept matplotlib name directly
    if name in CMAPS:
        return name
    # Try reverse lookup by value
    for k, v in CMAPS.items():
        if v.lower() == name.lower():
            return k
    return "Viridis"


# ── spectrum ──────────────────────────────────────────────────────────────────

def cmd_spectrum(args) -> None:
    """
    spectrum FILE [--format png|pdf] [--out PATH]
    """
    from ..viz.render import render_spectrum_figure, save_figure

    sd = load_profile(args.file)
    if sd.error:
        _err(f"Cannot load {args.file}: {sd.error}")

    print(f"Computing spectrum for {sd.name}…", file=sys.stderr)
    fs      = 1e6 / sd.dt_us
    d       = np.nan_to_num(sd.data, nan=0.0)
    sp_res  = compute_spectrum(d, fs)

    fig = render_spectrum_figure(
        sp_res, fs,
        title         = sd.name,
        n_traces      = sd.n_traces,
        dist_km       = sd.dist_km,
        boundaries_km = [],
    )

    fmt = args.format or "png"
    out = args.out or str(Path(sd.path).with_suffix(f".spectrum.{fmt}"))
    save_figure(fig, out, dpi=150, fmt=fmt)
    print(f"  ✔ {Path(out).name}", file=sys.stderr)
    print(f"  Peak:     {sp_res.peak_hz:.1f} Hz", file=sys.stderr)
    print(f"  Centroid: {sp_res.centroid_hz:.1f} Hz", file=sys.stderr)
    print(f"  SNR:      {sp_res.snr_db:.1f} dB", file=sys.stderr)


# ── navline ───────────────────────────────────────────────────────────────────

def cmd_navline(args) -> None:
    """
    navline FILE [--chain] --format shp|geojson|csv
                 [--crs EPSG] [--attrs] [--out PATH]
    """
    profiles = _load_profiles(args.files)
    valid    = [p for p in profiles if not p.error]
    if not valid:
        _err("No valid profiles.")

    fmt          = args.format
    include_attrs = getattr(args, "attrs", True)
    crs_str      = getattr(args, "crs",  None)

    tf = None
    if crs_str:
        try:
            from pyproj import CRS, Transformer
            src_crs = valid[0].detected_crs or "EPSG:4326"
            src_c   = CRS.from_user_input(src_crs)
            dst_c   = CRS.from_user_input(crs_str)
            if src_c != dst_c:
                tf = Transformer.from_crs(src_c, dst_c, always_xy=True)
        except Exception as exc:
            _err(f"CRS error: {exc}")

    def _write(source, out_base: str) -> None:
        lons = np.array(source.lons, dtype=float)
        lats = np.array(source.lats, dtype=float)
        dist = np.array(source.dist_km, dtype=float)
        wd   = np.array(source.water_depth, dtype=float)
        ts   = list(source.timestamps)

        if tf is not None:
            lons, lats = tf.transform(lons, lats)

        if fmt == "shp":
            write_navline_shp(out_base, lons, lats, dist, wd, ts, include_attrs, crs_str)
            out = out_base if out_base.lower().endswith(".shp") else out_base + ".shp"
        elif fmt == "geojson":
            write_navline_geojson(out_base, lons, lats, dist, wd, ts, include_attrs, crs_str)
            out = out_base if out_base.lower().endswith(".geojson") else out_base + ".geojson"
        else:
            write_navline_csv(out_base, lons, lats, dist, wd, ts)
            out = out_base if out_base.lower().endswith(".csv") else out_base + ".csv"
        print(f"  ✔ {Path(out).name}", file=sys.stderr)

    ext_map = {"shp": ".shp", "geojson": ".geojson", "csv": ".csv"}

    if args.chain:
        chains = detect_chains(valid)
        for ch in chains:
            print(f"\nNavline: {ch.label}", file=sys.stderr)
            stem    = f"{ch.profiles[0].stem}_a_{ch.profiles[-1].stem}_NAVLINE"
            out_base = args.out or str(Path(ch.profiles[0].path).with_name(stem + ext_map[fmt]))
            _write(ch, out_base)
    else:
        for sd in valid:
            print(f"\nNavline: {sd.name}", file=sys.stderr)
            out_base = args.out or str(Path(sd.path).with_name(sd.stem + "_NAVLINE" + ext_map[fmt]))
            _write(sd, out_base)


# ── fix ───────────────────────────────────────────────────────────────────────

def cmd_fix(args) -> None:
    """
    fix FILE [--chain] --interval MIN --format shp|geojson|csv [--out PATH]
    """
    profiles = _load_profiles(args.files)
    valid    = [p for p in profiles if not p.error]
    if not valid:
        _err("No valid profiles.")

    fmt    = args.format
    iv_min = args.interval

    ext_map = {"shp": ".shp", "geojson": ".geojson", "csv": ".csv"}

    def _write_fixes(source, out_base: str) -> None:
        fixes = compute_fix_positions(
            source.timestamps, source.dist_km, source.lons, source.lats, iv_min)
        if not fixes:
            print(f"  WARNING: no FIX marks found for {getattr(source, 'name', '')}",
                  file=sys.stderr)
            return

        # GIS boundary: projected native metres → the WGS84 every FIX writer
        # declares (see core.geometry_export.fixes_to_wgs84).
        pts = fixes_to_wgs84(fixes, source)

        if fmt == "shp":
            write_fix_points_shp(out_base, pts)
            out = out_base if out_base.lower().endswith(".shp") else out_base + ".shp"
        elif fmt == "geojson":
            write_fix_points_geojson(out_base, pts)
            out = out_base if out_base.lower().endswith(".geojson") else out_base + ".geojson"
        else:
            write_fix_points_csv(out_base, pts)
            out = out_base if out_base.lower().endswith(".csv") else out_base + ".csv"
        print(f"  ✔ {Path(out).name}  ({len(fixes)} marks)", file=sys.stderr)

    if args.chain:
        chains = detect_chains(valid)
        for ch in chains:
            print(f"\nFIX: {ch.label}", file=sys.stderr)
            stem     = f"{ch.profiles[0].stem}_a_{ch.profiles[-1].stem}_FIX"
            out_base = args.out or str(Path(ch.profiles[0].path).with_name(stem + ext_map[fmt]))
            _write_fixes(ch, out_base)
    else:
        for sd in valid:
            print(f"\nFIX: {sd.name}", file=sys.stderr)
            out_base = args.out or str(Path(sd.path).with_name(sd.stem + "_FIX" + ext_map[fmt]))
            _write_fixes(sd, out_base)
