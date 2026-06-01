"""
viz/render.py — Headless figure builders (Matplotlib Agg backend).

GUI-independence contract
-------------------------
This module uses matplotlib with the non-interactive Agg backend only.

Public API
----------
render_profile_figure / render_chain_figure / render_spectrum_figure → Figure
save_figure(fig, path, dpi, fmt)                                      → None
save_raw_rgba(source, data, path, params, …)                          → None
                                                                        (no-axes path)

PDF export
----------
Two paths:
  • With-axes  (default)  : matplotlib PDF backend.  Text/lines are vector;
    seismic raster is embedded at the figure DPI.  Page = natural figure size.
  • No-axes (--no-axes)   : img2pdf backend.  Lossless PNG embedded with zero
    recompression.  Page size controllable via --pdf-page.

img2pdf is the optimal library for raster-only PDF:
  - No recompression: source PNG bytes copied verbatim into the PDF stream
  - Correct DPI metadata: plotter knows the physical print size
  - Grayscale detection: Greys images auto-converted to 'L' mode → 3-4× smaller

PDF page sizes (--pdf-page, no-axes only)
  auto  Natural size = px_width/dpi × px_height/dpi inches (default)
  A0    Fit within 1189×841 mm landscape, aspect preserved
  A1    Fit within 841×594 mm landscape
  A2    Fit within 594×420 mm landscape
  A3    Fit within 420×297 mm landscape
  A4    Fit within 297×210 mm landscape
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be called before importing pyplot/Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..core.constants import CMAPS, FILTER_PRESETS
from ..core.coloring import colormapped_rgba
from ..core.spectrum import SpectrumResult
from ..core.processing import time_window
from ..core.geometry_export import compute_fix_positions

# Theme colours (match the GUI monolith)
_C = {
    "bg":       "#12141a",
    "panel":    "#1a1d26",
    "accent":   "#1e2535",
    "highlight":"#2a3a5c",
    "bright":   "#4d9de0",
    "warn":     "#e94560",
    "ok":       "#3ddc97",
    "text":     "#dce3ee",
    "sub":      "#6a7a96",
    "entry":    "#0e1118",
}


# ── PDF page size table (landscape: wider × taller) ───────────────────────────

_PDF_PAPERS_MM: dict = {
    "AUTO": None,
    "A0":   (1189.0, 841.0),
    "A1":   (841.0,  594.0),
    "A2":   (594.0,  420.0),
    "A3":   (420.0,  297.0),
    "A4":   (297.0,  210.0),
}


def _save_pdf_raster(img: Any, path: str, dpi: int,
                     pdf_page: Optional[str] = None) -> None:
    """
    Embed a PIL Image into a PDF using img2pdf (lossless, zero recompression).
    Falls back to Pillow if img2pdf is not installed.

    Strategy
    --------
    1. If the image is RGBA with R==G==B everywhere (e.g. Greys colormap),
       convert to 'L' mode first.  Grayscale PDFs compress 3-4× better.
    2. Save PIL image to an in-memory PNG with embedded DPI metadata.
    3. Pass PNG bytes to img2pdf with a layout function:
         auto / None  → get_fixed_dpi_layout_fun((dpi, dpi))
                         page = px_w/dpi × px_h/dpi inches
         A0-A4        → get_layout_fun(pagesize, fit=FitMode.into)
                         image scaled to fit within the paper, centred,
                         aspect ratio preserved.

    Parameters
    ----------
    img      : PIL Image (RGBA, RGB or L)
    path     : output .pdf path
    dpi      : physical resolution for 'auto' sizing and PNG metadata
    pdf_page : 'auto' | 'A0' | 'A1' | 'A2' | 'A3' | 'A4'
    """
    import io as _io

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # ── Mode optimisation ──────────────────────────────────────────────────
    if img.mode == "RGBA":
        arr = np.array(img)
        if (np.array_equal(arr[:, :, 0], arr[:, :, 1]) and
                np.array_equal(arr[:, :, 1], arr[:, :, 2])):
            img = img.convert("L")    # true grayscale → lossless, compact
        else:
            img = img.convert("RGB")

    # ── img2pdf path (preferred) ───────────────────────────────────────────
    try:
        import img2pdf as _i2p

        buf = _io.BytesIO()
        img.save(buf, format="PNG", dpi=(dpi, dpi))
        png_bytes = buf.getvalue()

        page_key = (pdf_page or "auto").upper()
        paper_mm = _PDF_PAPERS_MM.get(page_key)

        if paper_mm is not None:
            pw_pt = _i2p.mm_to_pt(paper_mm[0])
            ph_pt = _i2p.mm_to_pt(paper_mm[1])
            layout = _i2p.get_layout_fun(
                pagesize=(pw_pt, ph_pt),
                fit=_i2p.FitMode.into,
            )
        else:
            # natural size: physical = px / dpi
            layout = _i2p.get_fixed_dpi_layout_fun((dpi, dpi))

        pdf_bytes = _i2p.convert(png_bytes, layout_fun=layout)
        with open(path, "wb") as fh:
            fh.write(pdf_bytes)
        return

    except ImportError:
        pass  # fall through to Pillow

    # ── Pillow fallback ────────────────────────────────────────────────────
    # Grayscale ('L'): lossless FlateDecode in PDF.
    # RGB: JPEG lossy — warn user.
    if img.mode not in ("L", "1"):
        import warnings
        warnings.warn(
            "img2pdf not installed — PDF will use JPEG compression (lossy). "
            "Install: pip install img2pdf",
            UserWarning, stacklevel=3)
    img.save(path, "PDF", resolution=dpi)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_colormap(name: str):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return matplotlib.cm.get_cmap(name)


def _draw_fix_marks(ax, fixes: list, color: str = "#FFD700") -> None:
    if not fixes:
        return
    for num, dist, label, _lon, _lat in fixes:
        ax.axvline(dist, color=color, lw=0.8, ls="--", alpha=0.65, zorder=3)
        ax.text(dist, 0.98, f" {num} · {label}", color=color,
                fontsize=5.5, fontfamily="monospace",
                rotation=90, va="top", ha="center", zorder=5, clip_on=True,
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none", alpha=0.20))


def _style_axes(ax, fig=None) -> None:
    """Apply the dark-theme style to an axes object."""
    ax.set_facecolor(_C["entry"])
    ax.tick_params(colors=_C["text"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_C["accent"])
    if fig is not None:
        fig.patch.set_facecolor(_C["panel"])


# ── Shared render helpers ──────────────────────────────────────────────────────

def _vmin_vmax(d: np.ndarray, clip_pct: float, clip_lo: float) -> tuple:
    """Compute (vmin, vmax) from data percentiles."""
    flat = np.abs(d[~np.isnan(d)])
    if flat.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(flat, clip_pct)) or 1.0
    vmin = float(np.percentile(flat, clip_lo)) if clip_lo > 0 else 0.0
    return vmin, vmax


def _apply_axes_options(ax, t0: float, t1: float,
                        dist_start: float, dist_end: float,
                        x_tick_km: Optional[float],
                        t_tick_ms: Optional[float],
                        show_grid: bool) -> None:
    """Apply optional tick marks and grid to a seismic axes."""
    if x_tick_km is not None and x_tick_km > 0:
        import numpy as _np
        ticks = _np.arange(
            _np.ceil(dist_start / x_tick_km) * x_tick_km,
            dist_end + x_tick_km * 0.01,
            x_tick_km)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks],
                           color=_C["text"], fontsize=7)

    if t_tick_ms is not None and t_tick_ms > 0:
        import numpy as _np
        t_lo = min(t0, t1); t_hi = max(t0, t1)
        ticks = _np.arange(
            _np.ceil(t_lo / t_tick_ms) * t_tick_ms,
            t_hi + t_tick_ms * 0.01,
            t_tick_ms)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.0f}" for t in ticks],
                           color=_C["text"], fontsize=7)

    if show_grid:
        ax.grid(True, color="white", alpha=0.18, lw=0.5, zorder=4)


def _resize_rgba(rgba: np.ndarray, target_px: tuple) -> np.ndarray:
    """Resize RGBA uint8 array to (w, h) target using LANCZOS; return numpy."""
    try:
        from PIL import Image as _PIL
        img = _PIL.fromarray(rgba)
        if img.size != target_px:
            # For downsampling use BILINEAR (faster, visually equivalent at
            # the downscale ratios used here). For upsampling keep LANCZOS.
            tw, th = target_px
            ih, iw = rgba.shape[:2]
            filt = _PIL.BILINEAR if (iw >= tw or ih >= th) else _PIL.LANCZOS
            img = img.resize(target_px, filt)
        return np.array(img)
    except ImportError:
        return rgba


def _colorize_for_target(d: np.ndarray, cmap_name: str,
                          vmin: float, vmax: float,
                          target_px: tuple) -> np.ndarray:
    """
    Colorize a 2-D float32 seismic array to a target pixel size using
    the faster of two paths:

    Path A — downsample float32 FIRST, then colorize (fast):
        Used when either dimension is downsampled > 2:1.
        scipy.ndimage.zoom(order=1) on float32 is much cheaper than
        PIL LANCZOS on an RGBA uint8 array that is 4× larger.
        Result: identical visual quality at typical display/print scales.

    Path B — colorize FIRST, then PIL resize (original path):
        Used for upsampling or near 1:1 scales.
    """
    src_h, src_w = d.shape
    tgt_w, tgt_h = target_px

    vh = src_h / max(tgt_h, 1)
    vw = src_w / max(tgt_w, 1)

    if vh > 2.0 or vw > 2.0:
        try:
            from scipy.ndimage import zoom as _zoom
            sh = tgt_h / src_h
            sw = tgt_w / src_w
            d_s = _zoom(d, (sh, sw), order=1, prefilter=False).astype(np.float32)
            return colormapped_rgba(d_s, cmap_name, vmin, vmax)
        except Exception:
            pass  # fall through to Path B

    # Path B
    rgba = colormapped_rgba(d, cmap_name, vmin, vmax)
    return _resize_rgba(rgba, target_px)


# ── Profile figure ─────────────────────────────────────────────────────────────

def render_profile_figure(
    sd: Any,
    data: np.ndarray,
    params: dict,
    figsize: tuple = (12, 7),
    dpi: int = 100,
    # ── new visualisation options ──
    x_tick_km: Optional[float] = None,
    t_tick_ms: Optional[float] = None,
    show_grid: bool = False,
    title_override: Optional[str] = None,
    clip_lo: float = 0.0,
) -> Figure:
    """
    Render a seismic profile as a headless Matplotlib figure.

    Parameters
    ----------
    sd, data, params : standard seismic inputs
    figsize          : (width_in, height_in)
    dpi              : figure resolution
    x_tick_km        : if set, place x-axis ticks every N km
    t_tick_ms        : if set, place y-axis ticks every N ms
    show_grid        : overlay semi-transparent grid
    title_override   : replace auto-generated title
    clip_lo          : lower percentile for colour range (default 0)
    """
    i0, i1, t0, t1 = time_window(sd, data.shape[0], params.get("align", False))
    d = data[i0:i1, :]

    clip_pct  = params.get("clip", 99)
    clip_lo_  = params.get("clip_lo", clip_lo)
    vmin, vmax = _vmin_vmax(d, clip_pct, clip_lo_)

    cmap_base = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
    cmap_name = cmap_base + "_r" if params.get("inv_cmap", False) else cmap_base

    target  = (int(figsize[0] * dpi), int(figsize[1] * dpi))
    resized = _colorize_for_target(d, cmap_name, vmin, vmax, target)

    fig = Figure(figsize=figsize, dpi=dpi, facecolor=_C["panel"])
    ax  = fig.add_subplot(111)
    _style_axes(ax, fig)

    ax.imshow(resized, aspect="auto", interpolation="none",
              extent=[sd.dist_km[0], sd.dist_km[-1], t1, t0])

    _apply_axes_options(ax, t0, t1, sd.dist_km[0], sd.dist_km[-1],
                        x_tick_km, t_tick_ms, show_grid)

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.015)
    cb.ax.yaxis.set_tick_params(color=_C["sub"], labelsize=7)
    cb.set_label("Amplitude", color=_C["sub"], fontsize=8)

    if title_override:
        title_str = title_override
    else:
        preset_name = params.get("preset", "")
        preset_key  = FILTER_PRESETS.get(preset_name, preset_name or "none")
        preset_lbl  = f"  ·  {preset_name}" if preset_key not in ("none", "") else ""
        delay_lbl   = "  ·  aligned" if params.get("align") else ""
        title_str   = (f"{sd.name}  ·  {sd.n_traces} tr  ·  "
                       f"{sd.dt_us} µs{preset_lbl}{delay_lbl}")
    ax.set_title(title_str, color=_C["text"], fontsize=10, pad=8)
    ax.set_xlabel("Distance (km)", color=_C["text"], fontsize=9)
    ax.set_ylabel("Time (ms)",     color=_C["text"], fontsize=9)

    if params.get("fix"):
        fixes = compute_fix_positions(
            sd.timestamps, sd.dist_km, sd.lons, sd.lats,
            int(params.get("fix_iv", 5)))
        _draw_fix_marks(ax, fixes, color=_C["highlight"])

    fig.tight_layout(pad=1.2)
    return fig


# ── Chain figure ───────────────────────────────────────────────────────────────

def render_chain_figure(
    ch: Any,
    data: np.ndarray,
    params: dict,
    figsize: tuple = (14, 7),
    dpi: int = 100,
    # ── new visualisation options ──
    x_tick_km: Optional[float] = None,
    t_tick_ms: Optional[float] = None,
    show_grid: bool = False,
    title_override: Optional[str] = None,
    clip_lo: float = 0.0,
) -> Figure:
    """
    Render a ProfileChain as a headless Matplotlib figure.

    Per-segment vmax normalisation, boundary vlines, colorbar = median vmax.
    """
    i0, i1, t0, t1 = time_window(ch, data.shape[0], params.get("align", False))
    d = data[i0:i1, :]

    cmap_base = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
    cmap_name = cmap_base + "_r" if params.get("inv_cmap", False) else cmap_base
    clip_pct  = params.get("clip", 99)
    clip_lo_  = params.get("clip_lo", clip_lo)

    seg_boundaries = [0] + [
        int(np.searchsorted(ch.dist_km, bk)) for bk in ch.boundaries_km
    ] + [ch.n_traces]

    tgt_h = int(figsize[1] * dpi)   # target height in pixels

    rgba_segs  = []
    seg_vmaxes = []
    for seg_start, seg_end in zip(seg_boundaries[:-1], seg_boundaries[1:]):
        seg_d = d[:, seg_start:seg_end]
        vm, vx = _vmin_vmax(seg_d, clip_pct, clip_lo_)
        seg_vmaxes.append(vx)
        # ── Key optimisation: each segment is resized to its correct number
        # of OUTPUT pixels before colourisation.  For figheight=7 @ 300 DPI,
        # tgt_h=2100 vs raw_h≈10000 → 4.76:1 vertical downsample on float32
        # (cheap) instead of on the 4× larger RGBA uint8 array (expensive).
        seg_tgt = (seg_end - seg_start, tgt_h)   # keep 1:1 horizontally
        rgba_segs.append(_colorize_for_target(seg_d, cmap_name, vm, vx, seg_tgt))

    vmax_cb = float(np.median(seg_vmaxes)) if seg_vmaxes else 1.0
    vmin_cb = 0.0
    # All segments already at target height; concatenate horizontally only.
    resized = np.concatenate(rgba_segs, axis=1)
    target  = (int(figsize[0] * dpi), int(figsize[1] * dpi))

    fig = Figure(figsize=figsize, dpi=dpi, facecolor=_C["panel"])
    ax  = fig.add_subplot(111)
    _style_axes(ax, fig)

    ax.imshow(resized, aspect="auto", interpolation="none",
              extent=[ch.dist_km[0], ch.dist_km[-1], t1, t0])

    for bk in ch.boundaries_km:
        ax.axvline(bk, color=_C["warn"], lw=1.0, ls="--", alpha=0.7, zorder=5)

    _apply_axes_options(ax, t0, t1, ch.dist_km[0], ch.dist_km[-1],
                        x_tick_km, t_tick_ms, show_grid)

    sm = plt.cm.ScalarMappable(cmap=cmap_name,
                                norm=mcolors.Normalize(vmin_cb, vmax_cb))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.015)
    cb.ax.yaxis.set_tick_params(color=_C["sub"], labelsize=7)
    cb.set_label("Amplitude (median vmax)", color=_C["sub"], fontsize=8)

    title_str = title_override or (
        f"{ch.label}  ·  {ch.n_traces} tr  ·  {ch.dt_us} µs  ·  "
        f"{ch.total_km:.1f} km")
    ax.set_title(title_str, color=_C["text"], fontsize=10, pad=8)
    ax.set_xlabel("Distance (km)", color=_C["text"], fontsize=9)
    ax.set_ylabel("Time (ms)",     color=_C["text"], fontsize=9)

    if params.get("fix"):
        fixes = compute_fix_positions(
            ch.timestamps, ch.dist_km, ch.lons, ch.lats,
            int(params.get("fix_iv", 5)))
        _draw_fix_marks(ax, fixes, color=_C["highlight"])

    fig.tight_layout(pad=1.2)
    return fig


# ── Raw RGBA export (no-axes mode) ─────────────────────────────────────────────

def save_raw_rgba(
    source: Any,
    data: np.ndarray,
    path: str,
    params: dict,
    render_opts: dict,
    px_per_trace: float = 1.0,
    dpi: int = 150,
    figheight: float = 7.0,
    is_chain: bool = False,
    pdf_page: Optional[str] = None,
) -> None:
    """
    Save seismic data as a pure pixel image — no axes, labels, colorbar,
    or matplotlib margins.

    Output pixel dimensions (guaranteed exact)
    ------------------------------------------
    width  = n_traces × px_per_trace
    height = figheight × dpi

    Format handling
    ---------------
    .png  → lossless Deflate, DPI metadata embedded
    .tif  → lossless LZW, DPI metadata embedded
    .pdf  → img2pdf (lossless PNG-in-PDF, no recompression) with DPI/page metadata.
            Grayscale images (Greys colormap) auto-converted to 'L' mode for
            3-4× smaller files.  --pdf-page controls physical page size.
            Falls back to Pillow PDF if img2pdf not installed.

    Parameters
    ----------
    source       : SegyProfile or ProfileChain
    data         : (ns_proc, n_traces) float32 — output of process_*_data
    path         : output file path (.png, .tif, .pdf)
    params       : processing params dict (clip, clip_lo, cmap, inv_cmap, align)
    render_opts  : dict (clip_lo used; x_tick/t_tick ignored in no-axes mode)
    px_per_trace : horizontal scale — pixels per trace
    dpi          : physical resolution (px → inches: px/dpi)
    figheight    : target image height in inches (height_px = figheight × dpi)
    is_chain     : True → per-segment vmax normalisation
    pdf_page     : 'auto'|'A0'|'A1'|'A2'|'A3'|'A4' — PDF page size (pdf only)
    """
    from pathlib import Path as _P
    try:
        from PIL import Image as _PIL
    except ImportError:
        raise RuntimeError("Pillow is required for --no-axes mode: pip install Pillow")

    i0, i1, _t0, _t1 = time_window(source, data.shape[0], params.get("align", False))
    d = data[i0:i1, :]

    cmap_base = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
    cmap_name = cmap_base + "_r" if params.get("inv_cmap", False) else cmap_base
    clip_pct  = params.get("clip", 99)
    clip_lo_  = params.get("clip_lo", render_opts.get("clip_lo", 0.0))

    target_w = max(1, int(source.n_traces * px_per_trace))
    target_h = max(1, int(figheight * dpi))

    if is_chain:
        seg_boundaries = [0] + [
            int(np.searchsorted(source.dist_km, bk))
            for bk in source.boundaries_km
        ] + [source.n_traces]
        segs = []
        for s, e in zip(seg_boundaries[:-1], seg_boundaries[1:]):
            seg = d[:, s:e]
            vm, vx = _vmin_vmax(seg, clip_pct, clip_lo_)
            seg_tgt = (e - s, target_h)   # keep 1:1 per trace
            segs.append(_colorize_for_target(seg, cmap_name, vm, vx, seg_tgt))
        img = _PIL.fromarray(np.concatenate(segs, axis=1))
    else:
        vmin, vmax = _vmin_vmax(d, clip_pct, clip_lo_)
        arr = _colorize_for_target(d, cmap_name, vmin, vmax, (target_w, target_h))
        img = _PIL.fromarray(arr)

    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), _PIL.BILINEAR)

    ext = _P(path).suffix.lower()

    if ext == ".pdf":
        _save_pdf_raster(img, path, dpi, pdf_page=pdf_page)
    else:
        _P(path).parent.mkdir(parents=True, exist_ok=True)
        save_kwargs: dict = {}
        if ext in (".tif", ".tiff"):
            save_kwargs["compression"] = "tiff_lzw"
            save_kwargs["dpi"] = (dpi, dpi)
        elif ext == ".png":
            save_kwargs["dpi"] = (dpi, dpi)
        img.save(path, **save_kwargs)


# ── Spectrum figure ────────────────────────────────────────────────────────────

def render_spectrum_figure(
    sp: SpectrumResult,
    fs: float,
    title: str,
    n_traces: int,
    dist_km: Optional[np.ndarray] = None,
    boundaries_km: Optional[List[float]] = None,
    figsize: tuple = (10, 8),
    dpi: int = 100,
) -> Figure:
    """
    Render a 3-panel spectrum figure (headless).

    Panel 0: Mean Welch spectrum + percentile fill + metrics annotation.
    Panel 1 left: 2-D spectrogram (frequency vs. trace/distance).
    Panel 1 right: Energy distribution by band.
    Panel 2: Metrics table.

    Mirrors TopasSUITE._draw_spectrum_figure (~L2819).

    Parameters
    ----------
    sp            : SpectrumResult from compute_spectrum()
    fs            : sample rate in Hz
    title         : label for the figure title
    n_traces      : number of valid traces
    dist_km       : (n_traces,) x-axis for the 2-D panel; uses trace index if None
    boundaries_km : list of profile-join km positions (chains only)
    """
    freqs     = sp.freqs
    freqs_khz = freqs / 1000.0
    f_max_khz = min(20.0, fs / 2000.0)

    fig = Figure(figsize=figsize, dpi=dpi, facecolor=_C["panel"])
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[2.8, 2.2, 0.9],
                   width_ratios=[3, 1],
                   hspace=0.52, wspace=0.35)

    # ── Panel 0: Mean spectrum + percentile fill ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style_axes(ax1, fig)

    ax1.fill_between(freqs_khz, sp.spec_p10_db, sp.spec_p90_db,
                     alpha=0.15, color=_C["bright"], label="P10–P90")
    ax1.plot(freqs_khz, sp.spec_p50_db,
             color=_C["sub"], lw=0.9, ls="--", alpha=0.8, label="Median (P50)")
    ax1.plot(freqs_khz, sp.spec_mean_db,
             color=_C["bright"], lw=1.5, label="Mean (Welch)")

    pk_khz = sp.peak_hz / 1000.0
    pk_db  = float(sp.spec_mean_db[np.argmax(sp.spec_mean_db)])
    ax1.axvline(pk_khz, color=_C["warn"], lw=1.0, ls="--", alpha=0.85)
    ax1.annotate(f" Peak\n {pk_khz:.2f} kHz",
                 xy=(pk_khz, pk_db), xytext=(pk_khz + 0.15, pk_db - 8),
                 color=_C["warn"], fontsize=7, fontfamily="monospace",
                 arrowprops=dict(arrowstyle="-", color=_C["warn"], lw=0.7))

    ct_khz = sp.centroid_hz / 1000.0
    ax1.axvline(ct_khz, color=_C["highlight"], lw=0.9, ls=":", alpha=0.8,
                label=f"Centroid {ct_khz:.2f} kHz")

    bw3_lo_k = sp.bw_3db_lo / 1000.0
    bw3_hi_k = sp.bw_3db_hi / 1000.0
    ax1.axvspan(bw3_lo_k, bw3_hi_k, alpha=0.07, color=_C["ok"],
                label=f"BW -3 dB  ({bw3_lo_k:.1f}–{bw3_hi_k:.1f} kHz)")
    bw6_lo_k = sp.bw_6db_lo / 1000.0
    bw6_hi_k = sp.bw_6db_hi / 1000.0
    ax1.axvline(bw6_lo_k, color=_C["ok"], lw=0.7, ls="-.", alpha=0.55)
    ax1.axvline(bw6_hi_k, color=_C["ok"], lw=0.7, ls="-.", alpha=0.55,
                label=f"BW -6 dB  ({bw6_lo_k:.1f}–{bw6_hi_k:.1f} kHz)")

    ax1.set_xlim(0, f_max_khz)
    ax1.set_ylim(-80, 3)
    ax1.set_xlabel("Frequency (kHz)",       color=_C["text"], fontsize=9)
    ax1.set_ylabel("Amplitude (dB re. max)", color=_C["text"], fontsize=9)
    _low_res_warn = "  ⚠ low resolution (short profile)" if sp.low_res else ""
    ax1.set_title(
        f"Welch Spectrum  ·  {title}  ·  {n_traces} traces  ·  "
        f"NFFT={sp.nfft}  ·  {sp.n_frames} windows/trace{_low_res_warn}",
        color=_C["text"] if not sp.low_res else _C["warn"], fontsize=9)
    ax1.grid(True, color=_C["accent"], alpha=0.3, lw=0.5)
    ax1.legend(facecolor=_C["panel"], edgecolor=_C["accent"],
               labelcolor=_C["text"], fontsize=6.5, loc="lower left", ncol=3)

    # ── Panel 1 left: 2-D spectrogram ─────────────────────────────────────
    ax2     = fig.add_subplot(gs[1, 0])
    _style_axes(ax2, fig)
    spec_2d = sp.spec_2d_db
    x_axis  = dist_km if dist_km is not None else np.arange(n_traces)
    x_label = "Distance (km)" if dist_km is not None else "Trace"
    f_mask  = freqs <= f_max_khz * 1000
    ax2.pcolormesh(x_axis, freqs_khz[f_mask], spec_2d[f_mask, :],
                   cmap="inferno", vmin=-50, vmax=0, shading="auto")
    ax2.axhline(bw3_lo_k, color=_C["ok"],  lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(bw3_hi_k, color=_C["ok"],  lw=0.8, ls="--", alpha=0.6)
    ax2.axhline(pk_khz,   color=_C["warn"], lw=0.7, ls=":",  alpha=0.7)
    if boundaries_km:
        for b in boundaries_km:
            ax2.axvline(b, color=_C["warn"], lw=0.7, ls="--", alpha=0.6)
    ax2.set_ylim(0, f_max_khz)
    ax2.set_xlabel(x_label,            color=_C["text"], fontsize=8)
    ax2.set_ylabel("Frequency (kHz)",  color=_C["text"], fontsize=8)
    ax2.set_title("2-D Spectrogram  (frequency vs. distance)",
                  color=_C["text"], fontsize=8)
    ax2.tick_params(colors=_C["text"], labelsize=7)

    sm2 = plt.cm.ScalarMappable(cmap="inferno", norm=mcolors.Normalize(-50, 0))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax2, pad=0.01, fraction=0.03)
    cb2.set_label("dB re. max", color=_C["sub"], fontsize=7)
    cb2.ax.yaxis.set_tick_params(color=_C["sub"], labelsize=6)

    # ── Panel 1 right: Energy by band ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style_axes(ax3, fig)

    bands = [
        ("< 1 kHz",   0,     1000),
        ("1–2 kHz",   1000,  2000),
        ("2–4 kHz",   2000,  4000),
        ("4–7 kHz",   4000,  7000),
        ("7–10 kHz",  7000,  10000),
        ("10–15 kHz", 10000, 15000),
        ("> 15 kHz",  15000, fs / 2),
    ]
    pwr_mean_lin = 10 ** (sp.spec_mean_db / 10)
    band_labels  = []
    band_powers  = []
    for label, flo, fhi in bands:
        mask = (freqs >= flo) & (freqs < fhi)
        if mask.any() and fhi <= fs / 2:
            band_labels.append(label)
            band_powers.append(pwr_mean_lin[mask].sum())

    total_bp  = sum(band_powers) + 1e-30
    band_pcts = [100 * p / total_bp for p in band_powers]
    colors_bar = [_C["accent"]] * len(band_labels)
    if band_pcts:
        colors_bar[int(np.argmax(band_pcts))] = _C["bright"]

    bars = ax3.barh(band_labels, band_pcts, color=colors_bar,
                    edgecolor=_C["bg"], linewidth=0.5)
    for bar, pct in zip(bars, band_pcts):
        if pct > 3:
            ax3.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%", va="center", ha="left",
                     color=_C["text"], fontsize=6, fontfamily="monospace")
    ax3.set_xlabel("Energy (%)", color=_C["text"], fontsize=7)
    ax3.set_title("Energy by\nband",  color=_C["text"], fontsize=8)
    ax3.tick_params(colors=_C["text"], labelsize=6.5)

    # ── Panel 2: Metrics table ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor(_C["accent"])
    ax4.axis("off")

    metrics = [
        ("Peak",     f"{sp.peak_hz / 1000:.2f} kHz"),
        ("Centroid", f"{sp.centroid_hz / 1000:.2f} kHz"),
        ("BW -3 dB", f"{bw3_lo_k:.1f}–{bw3_hi_k:.1f} kHz"),
        ("BW -6 dB", f"{bw6_lo_k:.1f}–{bw6_hi_k:.1f} kHz"),
        ("Roll-off", f"{sp.roll_off_hz / 1000:.2f} kHz  (85%)"),
        ("SNR",      f"{sp.snr_db:.1f} dB"),
        ("NFFT",     str(sp.nfft)),
        ("Windows",  f"{sp.n_frames}/trace (Welch 50%)"),
    ]

    col_x = [0.02, 0.14, 0.39, 0.52, 0.77, 0.89]
    row_y = [0.78, 0.44, 0.10]
    for col_idx, (lbl, val) in enumerate(metrics):
        cx = col_x[col_idx % 3 * 2]
        vx = col_x[col_idx % 3 * 2 + 1]
        cy = row_y[col_idx // 3]
        ax4.text(cx, cy, lbl + ":", transform=ax4.transAxes,
                 color=_C["sub"], fontsize=7, fontfamily="monospace",
                 ha="left", va="center")
        ax4.text(vx, cy, val, transform=ax4.transAxes,
                 color=_C["text"], fontsize=7, fontfamily="monospace",
                 ha="left", va="center", fontweight="bold")

    fig.tight_layout(pad=1.2)
    return fig


# ── Save helper ────────────────────────────────────────────────────────────────

def save_figure(
    fig: Figure,
    path: str,
    dpi: int = 150,
    fmt: Optional[str] = None,
    pdf_page: Optional[str] = None,
) -> None:
    """
    Save a Figure to disk using the Agg backend.

    For PDF output (with axes)
    --------------------------
    matplotlib renders a PDF where text/lines/colorbar are vector and the
    seismic raster is embedded at the figure's DPI.  Page size = figure size
    (natural physical dimensions).

    If pdf_page is given (A0-A4), the figure is rescaled to fit within that
    paper before saving so that the PDF can be desk-printed at the right size.
    Labels/ticks scale proportionally.

    Parameters
    ----------
    fig      : matplotlib Figure (Agg backend)
    path     : output file path
    dpi      : output DPI
    fmt      : override format ("png", "pdf", "tif", "svg")
    pdf_page : 'auto'|'A0'|'A1'|'A2'|'A3'|'A4' — paper fitting for PDF output
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    effective_fmt = fmt or Path(path).suffix.lstrip(".").lower()

    # ── PDF with page fitting ──────────────────────────────────────────────
    if effective_fmt == "pdf" and pdf_page and pdf_page.upper() != "AUTO":
        paper_mm = _PDF_PAPERS_MM.get(pdf_page.upper())
        if paper_mm:
            pw_in = paper_mm[0] / 25.4   # landscape width  in inches
            ph_in = paper_mm[1] / 25.4   # landscape height in inches
            fw, fh = fig.get_size_inches()
            scale  = min(pw_in / fw, ph_in / fh)
            fig.set_size_inches(fw * scale, fh * scale)

    kwargs: dict = {"dpi": dpi, "bbox_inches": "tight",
                    "facecolor": fig.get_facecolor()}
    if fmt:
        kwargs["format"] = fmt
    fig.savefig(path, **kwargs)
    plt.close(fig)
