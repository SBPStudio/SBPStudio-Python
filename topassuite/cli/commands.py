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
    detect_chains, reproject_one, reproject_chain,
    process_profile_data, process_chain_data, time_window,
    compute_spectrum, colormapped_rgba,
    compute_fix_positions,
    write_fix_points_shp, write_fix_points_geojson, write_fix_points_csv,
    write_navline_shp, write_navline_geojson, write_navline_csv,
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


def _make_cancel_token() -> CancelToken:
    tok = CancelToken()
    def _handler(_sig, _frame):
        print("\nInterrupted — cancelling…", file=sys.stderr)
        tok.cancel()
    signal.signal(signal.SIGINT, _handler)
    return tok


def _load_profiles(files: List[str], quiet: bool = False) -> List[SegyProfile]:
    profiles = []
    for f in files:
        if not quiet:
            print(f"  Loading {Path(f).name}…", file=sys.stderr)
        sd = load_profile(f)
        if sd.error:
            print(f"  WARNING: {Path(f).name}: {sd.error}", file=sys.stderr)
        profiles.append(sd)
    return profiles


def _load_profiles_parallel(files: List[str]) -> List[SegyProfile]:
    """
    Load multiple SEG-Y files concurrently with ThreadPoolExecutor.
    Each file is a separate segyio handle → no shared state, thread-safe.
    I/O-bound: threading gives ~Nx speedup for N files on SSD/NVMe.
    """
    if len(files) <= 1:
        return _load_profiles(files)

    n_workers = min(len(files), 4)   # cap at 4 to avoid disk thrashing
    results:   List[Optional[SegyProfile]] = [None] * len(files)

    def _load_one(idx_path):
        idx, path = idx_path
        print(f"  Loading {Path(path).name}…", file=sys.stderr)
        sd = load_profile(path)
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
    join-chain FILE... --dst EPSG [--src EPSG] [--gap-km K] [--out PATH]

    Joins contiguous profiles into one SEG-Y via reproject_chain.
    When --src == --dst, coordinates are reprojected through an identity
    transform (pure join with no coordinate change).
    """
    tok      = _make_cancel_token()
    profiles = _load_profiles(args.files)
    valid    = [p for p in profiles if not p.error]

    if not valid:
        _err("No valid profiles to join.")

    gap_km = args.gap_km if args.gap_km else None
    chains = detect_chains(valid, gap_km=gap_km)

    if not chains:
        _err("No chains detected.")

    src = args.src or chains[0].profiles[0].detected_crs or "EPSG:4326"
    dst = args.dst

    for ch in chains:
        print(f"\nJoining chain: {ch.label}", file=sys.stderr)

        def _prog(frac, msg, _ch=ch):
            _progress(frac, f"{_ch.name[:40]}: {msg}")

        try:
            out = reproject_chain(ch, src, dst,
                                  unit_hint=args.unit_hint if hasattr(args, "unit_hint") else 2,
                                  log=_log, progress=_prog, cancel=tok)
            if args.out:
                dest = args.out
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(out, dest)
                out = dest
            print(f"\n  ✔ {Path(out).name}", file=sys.stderr)
        except Cancelled:
            print("\n  Cancelled.", file=sys.stderr)
            sys.exit(1)
        except (CRSError, ReprojectionError) as exc:
            _err(str(exc))


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

    # ── Load profiles (parallel for multiple files) ────────────────────────
    with timer.phase("loading"):
        if len(args.files) > 1:
            profiles = _load_profiles_parallel(args.files)
        else:
            profiles = _load_profiles(args.files)

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

    # ── Resolve figheight ──────────────────────────────────────────────────
    # Priority: explicit --figheight > --auto-height > default 7.0
    auto_height  = getattr(args, "auto_height", False)
    raw_figheight = getattr(args, "figheight", None)  # None if not set

    if raw_figheight is not None:
        figheight = raw_figheight
    elif auto_height:
        # compute from data — use first valid profile's ns
        ref_ns    = valid[0].ns
        figheight = ref_ns / dpi
        print(f"  auto-height: ns={ref_ns}, dpi={dpi} "
              f"→ figheight={figheight:.2f} in "
              f"(output height ≈ {int(figheight*dpi)} px)", file=sys.stderr)
    else:
        figheight = 7.0

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
    }

    pdf_page = getattr(args, "pdf_page", "auto") or "auto"

    # ── Render options (axes, ticks, grid, title) ──────────────────────────
    render_opts: dict = {
        "x_tick_km":    getattr(args, "x_tick", None),
        "t_tick_ms":    getattr(args, "t_tick", None),
        "show_grid":    getattr(args, "grid", False),
        "title_override": getattr(args, "title", None),
        "clip_lo":      getattr(args, "clip_lo", 0.0),
    }

    # ── Figure size helper ─────────────────────────────────────────────────
    def _figsize(n_traces: int) -> tuple:
        w = max(8.0, n_traces * px_per_trace / dpi)
        return (w, figheight)

    def _px_str(w_in: float, h_in: float) -> str:
        return (f"{int(w_in*dpi)}x{int(h_in*dpi)} px, "
                f"{w_in:.1f}x{h_in:.1f} in @ {dpi} DPI")

    # ── Export helper ──────────────────────────────────────────────────────
    import traceback as _tb

    def _export_one(source, data, out_path, is_chain: bool) -> None:
        try:
            fs_tuple = _figsize(source.n_traces)

            if no_axes:
                # ── Raw RGBA path: zero margins, exact px/trace mapping ──
                save_raw_rgba(source, data, out_path, params, render_opts,
                              px_per_trace=px_per_trace, dpi=dpi,
                              figheight=figheight, is_chain=is_chain,
                              pdf_page=pdf_page)
                w_px = int(source.n_traces * px_per_trace)
                h_px = int(figheight * dpi)
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
                data = process_chain_data(ch, params)
            out = args.out or str(
                Path(ch.profiles[0].path).with_suffix(f".{fmt}"))
            with timer.phase("render+save"):
                _export_one(ch, data, out, is_chain=True)
    else:
        for sd in valid:
            print(f"\nExporting {sd.name}…", file=sys.stderr)
            with timer.phase("processing"):
                data = process_profile_data(sd, params)
            out = args.out or str(Path(sd.path).with_suffix(f".{fmt}"))
            with timer.phase("render+save"):
                _export_one(sd, data, out, is_chain=False)

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

        pts = [(num, dist, hora, lon, lat) for num, dist, hora, lon, lat in fixes]

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
