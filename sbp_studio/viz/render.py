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
matplotlib.use("Agg")  # must be called before importing Figure
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..core.constants import CMAPS, FILTER_PRESETS
from ..core.coloring import colormapped_rgba
from ..core.spectrum import SpectrumResult
from ..core.processing import time_window
from ..core.geometry_export import compute_fix_positions

# ── Colour themes ─────────────────────────────────────────────────────────────
#
# Three built-in themes.  Pass one as --theme in the CLI; individual colours
# can be overridden with --bg-color / --text-color / --axes-bg-color.
#
# Keys used throughout render.py:
#   bg, panel    figure background
#   entry        seismic axes background
#   accent       spine / divider lines
#   highlight    FIX marks
#   text         labels and tick labels
#   sub          colorbar labels, minor text
#   warn         chain-join boundary lines
#   bright       spectrum lines
#   ok, grid     secondary colours

_THEMES: dict = {
    "dark": {                      # original dark-navy theme (matches the GUI)
        "bg":        "#12141a",
        "panel":     "#1a1d26",
        "entry":     "#0e1118",
        "accent":    "#1e2535",
        "highlight": "#4d9de0",
        "bright":    "#4d9de0",
        "warn":      "#e94560",
        "ok":        "#3ddc97",
        "text":      "#dce3ee",
        "sub":       "#6a7a96",
        "grid":      "white",
    },
    "light": {                     # light-grey, dark text — good for screen
        "bg":        "#f2f2f2",
        "panel":     "#f2f2f2",
        "entry":     "#ffffff",
        "accent":    "#bbbbbb",
        "highlight": "#0055cc",
        "bright":    "#0055cc",
        "warn":      "#cc2200",
        "ok":        "#006600",
        "text":      "#111111",
        "sub":       "#444444",
        "grid":      "#555555",
    },
    "print": {                     # pure white, black text — best for PDF/paper
        "bg":        "#ffffff",
        "panel":     "#ffffff",
        "entry":     "#ffffff",
        "accent":    "#cccccc",
        "highlight": "#003399",
        "bright":    "#003399",
        "warn":      "#990000",
        "ok":        "#004400",
        "text":      "#000000",
        "sub":       "#333333",
        "grid":      "#666666",
    },
}

# Default (backward-compat alias)
_C: dict = _THEMES["dark"]


def build_theme(
    theme: str = "dark",
    bg_color: Optional[str] = None,
    text_color: Optional[str] = None,
    axes_bg_color: Optional[str] = None,
) -> dict:
    """
    Return a colour dict for use by render functions.

    Parameters
    ----------
    theme         : 'dark' | 'light' | 'print' (default: 'dark')
    bg_color      : hex override for figure/panel background
    text_color    : hex override for all text and tick labels
    axes_bg_color : hex override for the seismic axes background
    """
    base = dict(_THEMES.get(theme, _THEMES["dark"]))
    if bg_color:
        base["bg"]    = bg_color
        base["panel"] = bg_color
    if text_color:
        base["text"] = text_color
        base["sub"]  = text_color
    if axes_bg_color:
        base["entry"] = axes_bg_color
    return base


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


def _pad_margins(d: np.ndarray, t0: float, t1: float,
                  dt_us: int,
                  margin_top_ms: float,
                  margin_bottom_ms: float) -> tuple:
    """
    Zero-pad the seismic matrix above and/or below the record and extend
    the time window accordingly.

    Returns
    -------
    (d_padded, t0_new, t1_new)
      d_padded : float32 array with extra zero rows at top and/or bottom
      t0_new   : adjusted start time in ms
      t1_new   : adjusted end time in ms
    """
    if margin_top_ms <= 0 and margin_bottom_ms <= 0:
        return d, t0, t1

    dt_ms  = dt_us / 1000.0
    top_s  = max(0, int(round(margin_top_ms    / dt_ms)))
    bot_s  = max(0, int(round(margin_bottom_ms / dt_ms)))
    d_padded = np.pad(d, ((top_s, bot_s), (0, 0)),
                      mode="constant", constant_values=0.0)
    return d_padded, t0 - top_s * dt_ms, t1 + bot_s * dt_ms


def _format_time_label(fix_tuple: tuple, timestamps: list,
                       dist_km_arr: np.ndarray, fmt: str) -> str:
    """
    Build the text label for a secondary-axis time tick.

    Parameters
    ----------
    fix_tuple     : (num, dist_km, "HH:MM", lon, lat)
    timestamps    : full timestamps list of the source ("YYYY-DOYnnn HH:MM:SS")
    dist_km_arr   : cumulative distance array (same length as timestamps)
    fmt           : 'hhmm' | 'fix' | 'position' | 'datetime' | 'full'
    """
    num, fix_dist, hhmm, lon, lat = fix_tuple

    # Nearest trace index by distance
    idx = int(np.clip(np.searchsorted(dist_km_arr, fix_dist),
                      0, len(dist_km_arr) - 1))
    ts_str = timestamps[idx] if timestamps else ""

    # Parse date from "YYYY-DOYnnn HH:MM:SS"  (DOY digits are at index 8:11)
    # Format: "2024-DOY100 10:05:30"
    #          01234567890123456789
    #                   ^^^  ← positions 8,9,10 = DOY digits
    date_str = ""
    try:
        from ..core.geometry_export import parse_timestamp as _pts
        dt = _pts(ts_str)
        if dt:
            date_str = dt.strftime("%d/%m/%Y")
    except Exception:
        try:
            from datetime import datetime, timedelta
            yr  = int(ts_str[:4])
            doy = int(ts_str[8:11])   # BUG FIX: was [6:9], DOY digits start at index 8
            dt  = datetime(yr, 1, 1) + timedelta(days=doy - 1)
            date_str = dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    lat_s = f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}"
    lon_s = f"{abs(lon):.4f}{'E' if lon >= 0 else 'W'}"

    if fmt == "fix":
        return f"#{num}  {hhmm}Z"
    if fmt == "position":
        return f"{hhmm}Z\n{lat_s}  {lon_s}"
    if fmt == "datetime":
        d = f"{date_str} " if date_str else ""
        return f"{d}{hhmm}Z"
    if fmt == "full":
        # Single first line with fix#, date and time; second line with coords
        head = f"#{num} {date_str} {hhmm}Z" if date_str else f"#{num} {hhmm}Z"
        return f"{head}\n{lat_s}  {lon_s}"
    # Default: 'hhmm'
    return f"{hhmm}Z"


def _draw_fix_marks(ax, fixes: list, color: str = "#FFD700",
                    font_size: float = 5.0,
                    bbox_alpha: float = 0.12,
                    axes_bg: str = "#0e1118") -> None:
    """
    Draw FIX-position marks: vertical dashed line + fix-number label.

    Parameters
    ----------
    fixes      : list of (num, dist_km, "HH:MM", lon, lat)
    color      : line and text colour (from theme highlight)
    font_size  : label font size in pt (default: 5.0)
    bbox_alpha : opacity of the label background (0 = no box, default: 0.12)
    axes_bg    : fill colour for label background (should match axes entry colour)
    """
    if not fixes:
        return
    for num, dist, _hhmm, _lon, _lat in fixes:
        ax.axvline(dist, color=color, lw=0.7, ls="--", alpha=0.55, zorder=3)
        kw: dict = dict(
            color=color, fontsize=font_size, fontfamily="monospace",
            rotation=90, va="top",
            # ha="right" → right edge of text at the tick line,
            # so the number appears to the LEFT of the mark.
            # Trailing spaces add a small gap between number and line.
            ha="right",
            zorder=5, clip_on=True,
            transform=ax.get_xaxis_transform(),
        )
        if bbox_alpha > 0:
            kw["bbox"] = dict(
                boxstyle="square,pad=0.1",
                fc=axes_bg,
                ec=color,
                alpha=bbox_alpha,
                linewidth=0.4,
            )
        ax.text(dist, 0.97, f"{num}  ", **kw)   # trailing spaces = gap from line


def _draw_picks(ax, picks: Optional[list], x_lo: float, x_hi: float, n_traces: int,
                color: str = "#FFA500", edge_color: str = "#000000",
                font_size: float = 7.0, marker_size: float = 42.0) -> None:
    """
    Burn interpretation picks (PickPoint: trace_index, time_ms, id) onto the
    export raster as a coloured marker + ID label, glued to the EXACT same
    linear pixel-grid approximation ax.imshow's own ``extent`` already uses
    for its columns — NOT the true (possibly non-uniform) dist_km value for
    a trace.

    Why this matters: ``_colorize_for_target`` resamples the seismic array
    to a DPI-driven target pixel width via a UNIFORM resample
    (``scipy.ndimage.zoom`` / PIL) — it has no notion of dist_km's real
    per-trace spacing. Column ``j`` of ``n_traces`` therefore always ends up
    at the LINEAR FRACTION ``j / n_traces`` of the image's full width,
    regardless of whether the real survey's trace spacing in km was uniform
    (it usually isn't — ship speed varies). A pick plotted at the TRUE
    ``dist_km[trace_index]`` would decouple from the column its data
    actually renders in whenever spacing is non-uniform — interpolating
    ``trace_index`` linearly into ``[x_lo, x_hi]`` (the SAME bounds the
    caller already passed to ``ax.imshow``'s own extent — see
    ``_x_axis_extent`` in render_profile_figure/render_chain_figure)
    reproduces the IDENTICAL approximation the raster itself is built from,
    so the marker always lands on the pixel its trace actually occupies.

    This single formula already covers every ``x_axis`` mode without a
    branch here: for "trace"/"km" mode the caller's ``x_lo, x_hi`` are
    ``(0, n_traces)`` (one column per trace, no resampling along that axis
    — see ``_x_axis_extent``), so the fraction collapses to plain
    ``trace_index`` unchanged.

    Y needs no equivalent correction: the sample interval (dt_ms) is
    constant by construction (unlike trace spacing), so ``time_ms`` already
    maps linearly and exactly onto the extent's Y bounds with no
    chunk/approximation error — see render_profile_figure/render_chain_figure's
    own t0/t1 extent construction, which ``p.time_ms`` is already expressed
    in (the SAME absolute-ms timeline the live SeismicView's
    ``_redraw_picks`` uses).
    """
    if not picks or n_traces <= 0:
        return
    xs: list = []
    ys: list = []
    for p in picks:
        i = max(0, min(int(p.trace_index), n_traces - 1))
        frac = i / n_traces
        x = x_lo + frac * (x_hi - x_lo)
        y = float(p.time_ms)
        xs.append(x)
        ys.append(y)
        ax.annotate(
            str(p.id), (x, y), xytext=(0, 7), textcoords="offset points",
            ha="center", va="bottom", color=color, fontsize=font_size,
            fontweight="bold", zorder=9, clip_on=True,
            path_effects=[patheffects.withStroke(linewidth=1.6, foreground=edge_color)],
        )
    # One batched scatter call for every pick (not one call per pick) — fewer
    # artists for large pick counts, and a single collection to query/restyle.
    ax.scatter(xs, ys, s=marker_size, c=color, edgecolors=edge_color,
              linewidths=0.9, zorder=8, clip_on=True)


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
    """Compute (vmin, vmax) from data percentiles.

    Subsamples to at most 200 000 elements before the percentile so that the
    cost stays flat regardless of matrix size (identical strategy to the
    preview's _estimate_vmax). The clip ceiling does not need exact precision —
    a uniform 2-D stride subsample of this size gives a stable estimate.
    """
    samp = d
    if samp.size > 200_000:
        step = int((samp.size / 200_000) ** 0.5) + 1
        samp = samp[::step, ::step]
    flat = np.abs(samp[np.isfinite(samp)])
    if flat.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(flat, clip_pct)) or 1.0
    vmin = float(np.percentile(flat, clip_lo)) if clip_lo > 0 else 0.0
    return vmin, vmax


def _apply_axes_options(ax, t0: float, t1: float,
                        dist_start: float, dist_end: float,
                        x_tick_km: Optional[float],
                        t_tick_ms: Optional[float],
                        show_grid: bool,
                        colors: Optional[dict] = None,
                        axis_font_size: float = 7.0,
                        grid_alpha: float = 0.18,
                        grid_lw: float = 0.5) -> None:
    """Apply optional tick marks and grid to a seismic axes.

    ``axis_font_size`` sizes the tick-number labels; ``grid_alpha`` / ``grid_lw``
    style the optional grid (exposed so the GUI export dialog can tune the
    geophysical printout). Defaults reproduce the historical look exactly."""
    C = colors if colors is not None else _C

    if x_tick_km is not None and x_tick_km > 0:
        ticks = np.arange(
            np.ceil(dist_start / x_tick_km) * x_tick_km,
            dist_end + x_tick_km * 0.01,
            x_tick_km)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks],
                           color=C["text"], fontsize=axis_font_size)

    if t_tick_ms is not None and t_tick_ms > 0:
        t_lo = min(t0, t1); t_hi = max(t0, t1)
        ticks = np.arange(
            np.ceil(t_lo / t_tick_ms) * t_tick_ms,
            t_hi + t_tick_ms * 0.01,
            t_tick_ms)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.0f}" for t in ticks],
                           color=C["text"], fontsize=axis_font_size)

    # Ensure AUTO tick labels (when no explicit ticks set) also use the theme colour
    # AND the requested size (matplotlib may generate them after this call).
    ax.tick_params(axis="both", labelcolor=C["text"], colors=C["text"],
                   labelsize=axis_font_size)

    if show_grid:
        ax.grid(True, color=C["grid"], alpha=grid_alpha, lw=grid_lw, zorder=4)


# ── X-axis modes (distance / trace / km-at-trace) ───────────────────────────────
# These are pure DATA-driven render options (no GUI/ephemeral state — Core/GUI
# contract preserved). They control ONLY the bottom-axis extent + labels:
#   "distance" — km extent (default/historical). The image is mapped uniformly
#                across [dist0, dist1]; with a non-uniform ping rate this stretches
#                (elongates) traces in sparse stretches.
#   "trace"    — trace-index extent [0, n]; ONE column per trace, no horizontal
#                stretch, axis read in trace number (km varies along the profile).
#   "km"       — trace-index extent [0, n] (no stretch) but km labels placed at the
#                REAL trace positions, so the bottom axis still reads in km with
#                tick spacing reflecting the true (variable) ping density.

def _nice_step(span: float) -> float:
    """A 1/2/5·10ᵏ 'nice' tick step giving ~6 ticks across `span`."""
    if span <= 0:
        return 1.0
    raw = span / 6.0
    p = 10.0 ** float(np.floor(np.log10(raw)))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * p >= raw:
            return float(m * p)
    return float(10.0 * p)


def _km_ticks_at_traces(ax, dist_km: np.ndarray, n_traces: int,
                        x_tick_km: Optional[float], colors: dict) -> None:
    """Place km labels at their TRACE-INDEX positions (for x_axis='km'): the data
    stays one-column-per-trace (no stretch) while the bottom axis reads in km."""
    dk = np.asarray(dist_km, dtype=float)
    if dk.size < 2 or n_traces < 2:
        return
    k0, k1 = float(dk[0]), float(dk[-1])
    step = x_tick_km if (x_tick_km and x_tick_km > 0) else _nice_step(k1 - k0)
    km_vals = np.arange(np.ceil(k0 / step) * step, k1 + step * 0.01, step)
    if km_vals.size == 0:
        return
    idx = np.clip(np.searchsorted(dk, km_vals), 0, n_traces - 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{k:.1f}" for k in km_vals],
                       color=colors["text"], fontsize=7)


def _x_axis_extent(x_axis: str, dist0: float, dist1: float, n_traces: int) -> tuple:
    """imshow X extent: km for 'distance', trace-index [0, n] for the per-trace
    modes (one column per trace → no horizontal stretching)."""
    if x_axis == "distance":
        return float(dist0), float(dist1)
    return 0.0, float(n_traces)


def _label_x_axis(ax, x_axis: str, dist_km: np.ndarray, n_traces: int,
                  x_tick_km: Optional[float], colors: dict,
                  label_font_size: float = 9.0) -> None:
    """Bottom-axis label + ticks for the chosen mode (called after imshow)."""
    if x_axis == "trace":
        ax.set_xlabel("Trace number", color=colors["text"], fontsize=label_font_size)
    elif x_axis == "km":
        _km_ticks_at_traces(ax, dist_km, n_traces, x_tick_km, colors)
        ax.set_xlabel("Distance (km)", color=colors["text"], fontsize=label_font_size)
    else:  # "distance"
        ax.set_xlabel("Distance (km)", color=colors["text"], fontsize=label_font_size)


def _add_time_axis(ax,
                   timestamps: list,
                   dist_km: np.ndarray,
                   lons: np.ndarray,
                   lats: np.ndarray,
                   time_tick_min: int,
                   time_fmt: str = "hhmm",
                   time_font_size: float = 6.0,
                   time_align: str = "left",
                   colors: Optional[dict] = None,
                   boundaries_km: Optional[List[float]] = None) -> None:
    """
    Add a secondary x-axis at the TOP of the seismic section showing UTC
    acquisition timestamps at regular time intervals.

    Parameters
    ----------
    ax             : seismic axes (bottom x = km)
    timestamps     : list of "YYYY-DOYnnn HH:MM:SS" strings, one per trace
    dist_km        : (n_traces,) cumulative distance array
    lons / lats    : (n_traces,) coordinates
    time_tick_min  : label interval in minutes
    time_fmt       : 'hhmm' | 'fix' | 'position' | 'datetime' | 'full'
    time_font_size : font size for labels (default: 6.0)
    time_align     : 'left' | 'center' | 'right' (horizontal alignment)
    colors         : theme dict; uses global _C if None
    """
    C = colors if colors is not None else _C

    fixes = compute_fix_positions(timestamps, dist_km, lons, lats, time_tick_min)
    if not fixes:
        return

    tick_km    = [f[1] for f in fixes]
    tick_label = [_format_time_label(f, timestamps, dist_km, time_fmt)
                  for f in fixes]

    # The user's intention when they say "left" / "right" is the POSITION of
    # the label relative to the tick mark, not the matplotlib ha alignment.
    # With rotation=90 (text reads bottom→top), the position semantics are:
    #   user "left"  → ha="right"  (right edge of text box at tick → text to the LEFT)
    #   user "right" → ha="left"   (left edge of text box at tick → text to the RIGHT)
    #   user "center" → ha="center"
    _ha_map = {"left": "right", "center": "center", "right": "left"}
    ha  = _ha_map.get(time_align, "right")
    va  = "bottom"   # bottom of text column at the axis line → text grows upward

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(tick_km)
    ax_top.set_xticklabels(
        tick_label,
        rotation=90,
        va=va,
        ha=ha,
        color=C["text"],
        fontsize=time_font_size,
        fontfamily="monospace",
    )
    ax_top.tick_params(
        axis="x", colors=C["text"],
        direction="out", length=4, width=0.8,
        labelsize=time_font_size,
    )
    ax_top.set_xlabel("UTC", color=C["sub"], fontsize=max(5, time_font_size - 1),
                      labelpad=2)
    for sp in ax_top.spines.values():
        sp.set_edgecolor(C["accent"])


def _resize_rgba(rgba: np.ndarray, target_px: tuple,
                  interp: Optional[str] = None) -> np.ndarray:
    """Resize RGBA uint8 array to (w, h) target.

    ``interp`` is the user's explicit HQ/Export pixel-interpolation choice:
    ``"nearest"`` forces hard-edged NEAREST scaling, ``"bicubic"`` forces
    PIL's BICUBIC. Any other value (the default, ``None``, or ``"bilinear"``)
    keeps the historical smooth behaviour — BILINEAR for downsampling (faster,
    visually equivalent at these ratios) and LANCZOS for upsampling —
    unaffected callers that don't pass ``interp`` see no change."""
    try:
        from PIL import Image as _PIL
        img = _PIL.fromarray(rgba)
        if img.size != target_px:
            if interp == "nearest":
                filt = _PIL.NEAREST
            elif interp == "bicubic":
                filt = _PIL.BICUBIC
            else:
                tw, th = target_px
                ih, iw = rgba.shape[:2]
                filt = _PIL.BILINEAR if (iw >= tw or ih >= th) else _PIL.LANCZOS
            img = img.resize(target_px, filt)
        return np.array(img)
    except ImportError:
        return rgba


def _maxabs_pool_1d(a: np.ndarray, tgt: int, axis: int = 0) -> np.ndarray:
    """Reduce ``a`` along ``axis`` to ``tgt`` bins keeping the MAX-|value| sample
    per bin. Preserves thin, high-amplitude reflectors that plain averaging /
    bilinear interpolation would smear out. ``tgt >= size`` (upsample) is a no-op
    on that axis.

    Fast path: when ``src`` is exactly divisible by ``tgt`` the entire reduction
    is a single reshape + argmax (no Python loop). Fallback: Python loop with
    variable-width bins for non-integer block ratios."""
    src = a.shape[axis]
    if tgt < 1 or tgt >= src:
        return a
    a = np.moveaxis(a, axis, 0)   # bring the target axis to front

    if src % tgt == 0:
        # ── Vectorized fast path: uniform block size ─────────────────────────
        blk     = src // tgt
        tail    = a.shape[1:]
        blocks  = a.reshape(tgt, blk, *tail)                            # (tgt, blk, ...)
        safe    = np.where(np.isfinite(blocks), np.abs(blocks), -1.0)  # mask NaN/Inf
        idx     = np.argmax(safe, axis=1)                               # (tgt, ...)
        out     = np.take_along_axis(blocks,
                                     np.expand_dims(idx, axis=1),
                                     axis=1)[:, 0]                      # (tgt, ...)
    else:
        # ── Fallback: Python loop for non-integer block ratios ───────────────
        edges = np.linspace(0, src, tgt + 1).astype(int)
        out   = np.empty((tgt,) + a.shape[1:], dtype=a.dtype)
        for i in range(tgt):
            s   = int(edges[i])
            e   = max(s + 1, int(edges[i + 1]))
            blk = a[s:e]
            # Mask non-finite entries so NaN/Inf never wins argmax.
            idx    = np.argmax(np.where(np.isfinite(blk), np.abs(blk), -1.0), axis=0)
            out[i] = np.take_along_axis(blk, idx[None], axis=0)[0]

    return np.moveaxis(out, 0, axis)


def _downsample_maxabs(d: np.ndarray, tgt_h: int, tgt_w: int) -> np.ndarray:
    """2-D max-|amplitude| downsample of a float array toward (tgt_h, tgt_w).
    Only axes that are actually shrinking are pooled (upscaled axes pass through,
    to be resized by the caller). Reflector-peak-preserving counterpart of the
    live view's ``_pool_rows_maxabs``."""
    d = _maxabs_pool_1d(d, tgt_h, axis=0)
    d = _maxabs_pool_1d(d, tgt_w, axis=1)
    return np.ascontiguousarray(d)


def _colorize_for_target(d: np.ndarray, cmap_name: str,
                          vmin: float, vmax: float,
                          target_px: tuple,
                          max_abs_pool: bool = False,
                          interp: Optional[str] = None) -> np.ndarray:
    """
    Colorize a 2-D float32 seismic array to a target pixel size using
    the faster of two paths:

    Path A — downsample float32 FIRST, then colorize:
        Used when either dimension is downsampled. With ``max_abs_pool`` the
        shrink uses MAX-|amplitude| pooling (reflector-safe — thin high-amplitude
        events survive); otherwise ``scipy.ndimage.zoom`` (order=0 nearest / 3
        bicubic per ``interp``, else order=1 bilinear — the default, unchanged
        for callers that don't pass ``interp``).
        Downsampling float32 first is much cheaper than PIL on the 4× RGBA array.

    Path B — colorize FIRST, then PIL resize (original path):
        Used for upsampling or near 1:1 scales. ``interp`` is forwarded to
        :func:`_resize_rgba` (``"nearest"`` forces hard edges; default keeps the
        historical BILINEAR/LANCZOS smooth behaviour).
    """
    # #10: failing filters (division-by-zero, sqrt of negative, etc.) leak
    # NaN/Inf into the array. Left unhandled, these map to transparent/black
    # pixels in the RGBA colormap instead of the neutral background — replace
    # them with 0.0 (mid-scale, harmless) before any resampling or colorizing.
    # Gated: np.nan_to_num always allocates a full copy (a ~1 GB transient on
    # large profiles), while np.isfinite(np.sum(...)) is an allocation-free
    # reduction — NaN/Inf propagate through the sum, so a finite total proves
    # the array is already clean. float64 accumulation keeps the sum itself
    # from overflowing on huge all-finite float32 arrays (false positive).
    if not np.isfinite(np.sum(d, dtype=np.float64)):
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    src_h, src_w = d.shape
    tgt_w, tgt_h = target_px

    vh = src_h / max(tgt_h, 1)
    vw = src_w / max(tgt_w, 1)

    if vh > 1.0 or vw > 1.0:
        if max_abs_pool:
            # Pool ONLY the shrinking axis/axes to the target; a final resize
            # (no-op or up-scale of the other axis) brings it to exact target_px.
            ph = tgt_h if vh > 1.0 else src_h
            pw = tgt_w if vw > 1.0 else src_w
            d_s = _downsample_maxabs(d, ph, pw).astype(np.float32)
            rgba = colormapped_rgba(d_s, cmap_name, vmin, vmax)
            return _resize_rgba(rgba, target_px, interp=interp)
        try:
            from scipy.ndimage import zoom as _zoom
            sh = tgt_h / src_h
            sw = tgt_w / src_w
            order = 0 if interp == "nearest" else (3 if interp == "bicubic" else 1)
            d_s = _zoom(d, (sh, sw), order=order, prefilter=False).astype(np.float32)
            return colormapped_rgba(d_s, cmap_name, vmin, vmax)
        except Exception:
            pass  # fall through to Path B

    # Path B
    rgba = colormapped_rgba(d, cmap_name, vmin, vmax)
    return _resize_rgba(rgba, target_px, interp=interp)


# ── Wiggle / variable-area overlay ──────────────────────────────────────────────

# Budget: max wiggle traces actually drawn. The raster base layer (when shown)
# keeps full trace detail; the vector overlay is decimated to this many columns
# so the wiggle stays legible and the export (and any vector PDF) stays light.
WIGGLE_MAX_TRACES = 1200

# Budget: max vertex rows per wiggle trace. Deep records (30k+ samples) would
# otherwise build native-resolution float64 polygon/line geometry — ~1 GB of
# sub-pixel detail no output medium can display. 4096 rows covers a 13.6-inch
# plot at 300 dpi with one vertex per pixel row; rows are MAX-|amplitude|
# pooled (same operator as the raster path) so reflector peaks survive.
WIGGLE_MAX_ROWS = 4096

# Default deflection gain — < 1.0, not 1.0: at gain=1.0 a fully-saturated sample
# (amp clipped to ±vmax) deflects by EXACTLY one full inter-trace spacing,
# landing precisely on the neighbouring trace's own anchor — zero margin. Near
# a strong reflector (several consecutive near-saturated samples) that fill
# visibly bleeds past where the neighbour's own data would justify any fill,
# reading as a "blob" with no local VA data. 0.9 keeps deflection strong/
# legible while guaranteeing a gap between adjacent traces' max excursions.
WIGGLE_GAIN_DEFAULT = 0.9


def _draw_wiggle_overlay(ax, d: np.ndarray, x_lo: float, x_hi: float,
                         t0: float, t1: float, *, vmax: float,
                         va_fill: bool = True, show_wiggle_line: bool = True,
                         wiggle_gain: float = WIGGLE_GAIN_DEFAULT,
                         max_traces: int = WIGGLE_MAX_TRACES,
                         color: str = "#000000", lw: float = 0.4) -> None:
    """Overlay budgeted wiggle (+ optional variable-area fill) traces onto ``ax``.

    ``d`` is the (ns, n_traces) float window already time-cropped/margined. Trace
    columns are decimated to ``max_traces`` and vertex rows max-|amp|-pooled to
    ``WIGGLE_MAX_ROWS`` so the vector overlay stays legible, light on RAM and
    fast regardless of the native record size (the raster base layer, if
    drawn, still carries full detail). Each drawn trace is plotted as a horizontal
    deflection x = centre + (amp/vmax)·spacing·gain; positive lobes are filled
    (classic variable-area look) when ``va_fill``; the deflection line itself is
    drawn when ``show_wiggle_line`` (the two are independent visibility toggles —
    either, both, or neither may be on). Also pins the axes limits so a 'Wiggle
    Only' figure (no imshow) is framed correctly (time downward)."""
    n_rows, n_cols = d.shape
    if n_cols < 1 or n_rows < 2:
        return
    vmax = float(vmax) or 1.0
    step = max(1, int(np.ceil(n_cols / max(1, max_traces))))
    cols = np.arange(0, n_cols, step)
    n_draw = cols.size
    span = float(x_hi - x_lo)
    centres = x_lo + (cols + 0.5) / n_cols * span         # native col → x position
    # Median inter-trace gap: robust when the vessel is stationary (span=0,
    # all centres identical → global-average would collapse deflect to zero).
    spacing = float(np.median(np.diff(centres))) if n_draw > 1 else span / max(1, n_cols)
    deflect = max(abs(spacing), 1e-6) * float(wiggle_gain)

    # Row-decimate the geometry to the display-resolution budget BEFORE any
    # vertex arrays are built (native 30k-row records → ~1 GB of float64
    # vertices otherwise). Max-|amp| pooling keeps thin reflectors; no-op when
    # the record is already within budget. All vertex arrays below are float32
    # — axis-space coordinates need far fewer than float32's ~7 significant
    # digits, and it halves the geometry RAM again.
    dw = d[:, cols]                                        # (n_rows, n_draw)
    if dw.shape[0] > WIGGLE_MAX_ROWS:
        dw = _maxabs_pool_1d(dw, WIGGLE_MAX_ROWS, axis=0)
    n_verts = dw.shape[0]
    t = np.linspace(t0, t1, n_verts).astype(np.float32)

    # #9: build ALL traces' geometry as vectorized NumPy arrays and hand them
    # to a single LineCollection / PolyCollection each, instead of calling
    # ax.plot()/ax.fill_betweenx() once per trace. Each of those per-trace
    # calls pays Matplotlib's Artist-creation + transform-registration
    # overhead; on a 1200-trace wiggle overlay that overhead dominates the
    # actual draw time. Two collections == two Artists, regardless of n_draw.
    amp_all = np.clip(dw / vmax, -1.0, 1.0).astype(np.float32, copy=False)
    centres32 = centres.astype(np.float32)
    deflect32 = np.float32(deflect)
    x_all = centres32[None, :] + amp_all * deflect32       # (n_verts, n_draw)

    if show_wiggle_line:
        line_segs = np.empty((n_draw, n_verts, 2), dtype=np.float32)
        line_segs[:, :, 0] = x_all.T
        line_segs[:, :, 1] = t[None, :]
        lc = LineCollection(line_segs, colors=color, linewidths=lw,
                            antialiaseds=True, zorder=6)
        ax.add_collection(lc)

    if va_fill:
        # Positive-lobe hull: clipping the deflection to >=0 collapses the
        # boundary onto the trace's own baseline (xc) at every negative
        # excursion, so the polygon area vanishes there automatically —
        # equivalent to the old where=(amp>0)+interpolate=True fill without
        # needing to hand-detect zero-crossings per sample.
        pos_all   = np.maximum(amp_all, 0.0)
        xhull_all = centres32[None, :] + pos_all * deflect32   # (n_verts, n_draw)
        t_rev = t[::-1]
        polys = []
        for j in range(n_draw):
            xc  = centres32[j]
            fwd  = np.column_stack((xhull_all[:, j], t))
            back = np.column_stack((np.full(n_verts, xc, dtype=np.float32), t_rev))
            polys.append(np.vstack((fwd, back)))
        pc = PolyCollection(polys, facecolors=color, edgecolors="none", zorder=6)
        ax.add_collection(pc)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(t1, t0)            # time downward — matches the imshow extent


# ── Profile figure ─────────────────────────────────────────────────────────────

def render_profile_figure(
    sd: Any,
    data: np.ndarray,
    params: dict,
    figsize: tuple = (12, 7),
    dpi: int = 100,
    # ── visualisation options ──
    x_tick_km: Optional[float] = None,
    t_tick_ms: Optional[float] = None,
    show_grid: bool = False,
    title_override: Optional[str] = None,
    clip_lo: float = 0.0,
    time_tick_min: Optional[int] = None,
    # ── new: margins, label config, colours ──
    margin_top_ms: float = 0.0,
    margin_bottom_ms: float = 0.0,
    time_fmt: str = "hhmm",
    time_font_size: float = 6.0,
    time_align: str = "left",
    fix_font_size: float = 5.0,
    fix_bbox_alpha: float = 0.12,
    fix_color: Optional[str] = None,
    colors: Optional[dict] = None,
    x_axis: str = "distance",
    axis_font_size: float = 7.0,
    grid_alpha: float = 0.18,
    grid_lw: float = 0.5,
    # ── layered rendering (density raster base + wiggle/VA overlay) ──
    show_raster: bool = True,
    style: str = "density",
    layout_mode: str = "aspect",
    va_fill: bool = True,
    show_wiggle_line: bool = True,
    wiggle_gain: float = WIGGLE_GAIN_DEFAULT,
    max_wiggles: int = WIGGLE_MAX_TRACES,
    max_abs_pool: bool = False,
    picks: Optional[list] = None,
    deco_scale: float = 1.0,
    box_aspect: Optional[float] = None,
) -> Figure:
    """
    Render a seismic profile as a headless Matplotlib figure.

    ``deco_scale`` — multiplier for the renderer-internal decoration sizes
    (title, colorbar text; the caller-supplied font sizes arrive pre-scaled),
    so a dynamically-sized WYSIWYG page keeps proportionate text. Default 1.0
    = the historical fixed sizes (CLI path unchanged).

    ``box_aspect`` — PHYSICAL height/width ratio to enforce on the AXES BOX
    (``ax.set_box_aspect``): the data area is letterboxed inside the page at
    exactly this proportion, with empty margins as needed — used by fixed
    paper-size exports to honour the user's custom on-screen scale instead of
    stretching the data to fill the sheet. Default ``None`` = historical fill
    (CLI path unchanged).

    Layered rendering
    -----------------
    ``style`` selects the section style: ``"density"`` (raster only) or
    ``"wiggle"`` (raster base + wiggle/variable-area overlay). ``show_raster``
    gates the base raster — set it False with ``style="wiggle"`` for a 'Wiggle
    Only' figure (density always keeps the raster regardless). ``va_fill`` toggles
    the black positive-lobe fill; ``wiggle_gain`` scales the deflection;
    ``max_wiggles`` budgets the drawn trace count. ``layout_mode`` is applied
    upstream via ``figsize`` (see ``_render.figsize_for_scale``) and accepted here
    for interface symmetry.

    ``x_axis`` selects the bottom-axis mode (see the module-level X-axis helpers):
    ``"distance"`` (default, km extent — historical), ``"trace"`` (one column per
    trace, no horizontal stretch, axis in trace number), or ``"km"`` (one column
    per trace but km labels at their real positions). Defaulting to ``"distance"``
    keeps the CLI/GUI behaviour unchanged.

    New parameters
    --------------
    margin_top_ms / margin_bottom_ms : zero-filled padding above/below record (ms)
    time_fmt       : top-axis label format ('hhmm'|'fix'|'position'|'datetime'|'full')
    time_font_size : font size for top time-axis labels
    time_align     : label position relative to tick — 'left' puts label to the LEFT
    fix_font_size  : font size for FIX-mark number labels inside the image
    fix_bbox_alpha : opacity of the FIX label background box (0 = no box)
    fix_color      : colour for FIX lines and labels (defaults to theme highlight)
    colors         : theme dict from build_theme(); uses dark theme if None
    """
    C = colors if colors is not None else _C

    i0, i1, t0, t1 = time_window(sd, data.shape[0], params.get("align", False))
    d = data[i0:i1, :]

    # ── Apply margins ──────────────────────────────────────────────────────
    d, t0, t1 = _pad_margins(d, t0, t1, sd.dt_us, margin_top_ms, margin_bottom_ms)

    clip_pct   = params.get("clip", 99.6)
    clip_lo_   = params.get("clip_lo", clip_lo)
    if params.get("vmax_override"):
        vmax = float(params["vmax_override"])
        vmin = -vmax if params.get("amp_range") == "diverging" else 0.0
    else:
        vmin, vmax = _vmin_vmax(d, clip_pct, clip_lo_)
        if params.get("amp_range") == "diverging":
            vmin = -vmax

    cmap_base = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
    cmap_name = cmap_base + "_r" if params.get("inv_cmap", False) else cmap_base

    fig = Figure(figsize=figsize, dpi=dpi, facecolor=C["panel"])
    ax  = fig.add_subplot(111)
    ax.set_facecolor(C["entry"])
    ax.tick_params(colors=C["text"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["accent"])
    if box_aspect:
        # Letterbox: pin the PHYSICAL axes-box proportion (never stretch the
        # user's custom scale to fill a fixed sheet). The two-pass raster
        # sizing below measures the axes box AFTER layout, so it colorizes to
        # the letterboxed box automatically.
        ax.set_box_aspect(box_aspect)

    x_lo, x_hi = _x_axis_extent(x_axis, sd.dist_km[0], sd.dist_km[-1], d.shape[1])
    # Layered: raster base (density) + optional wiggle/VA overlay. Density always
    # keeps the raster; 'Wiggle Only' (show_raster=False) applies in wiggle style.
    draw_raster = show_raster or style != "wiggle"

    # Add all axes decorations BEFORE the raster so tight_layout can settle the
    # final axes geometry, then measure the exact axes pixel box, and only THEN
    # colorize the data to that precise size.  This two-pass approach eliminates
    # the double-resample: previously the RGBA was oversized to the full figure
    # (figsize×dpi) and Matplotlib had to resample it again at draw time using
    # hardcoded "antialiased".  Now the RGBA matches the axes box 1:1 so the
    # imshow resample at draw time is a no-op and the user's interpolation choice
    # is honoured without the dark-background bias that afflicted nearest-neighbor.
    if style == "wiggle":
        _draw_wiggle_overlay(ax, d, x_lo, x_hi, t0, t1, vmax=vmax,
                             va_fill=va_fill, show_wiggle_line=show_wiggle_line,
                             wiggle_gain=wiggle_gain,
                             max_traces=max_wiggles, color="#000000")

    # km x-ticks only make sense on the km extent; the per-trace modes get their
    # own ticks via _label_x_axis below.
    _apply_axes_options(ax, t0, t1, x_lo, x_hi,
                        x_tick_km if x_axis == "distance" else None,
                        t_tick_ms, show_grid, colors=C,
                        axis_font_size=axis_font_size,
                        grid_alpha=grid_alpha, grid_lw=grid_lw)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.015)
    cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=7 * deco_scale)
    cb.ax.yaxis.set_tick_params(labelcolor=C["sub"])
    cb.set_label("Amplitude", color=C["sub"], fontsize=8 * deco_scale)

    if title_override:
        title_str = title_override
    else:
        preset_name = params.get("preset", "")
        preset_key  = FILTER_PRESETS.get(preset_name, preset_name or "none")
        preset_lbl  = f"  ·  {preset_name}" if preset_key not in ("none", "") else ""
        delay_lbl   = "  ·  aligned" if params.get("align") else ""
        title_str   = (f"{sd.name}  ·  {sd.n_traces} tr  ·  "
                       f"{sd.dt_us} µs{preset_lbl}{delay_lbl}")
    ax.set_title(title_str, color=C["text"], fontsize=10 * deco_scale,
                 pad=8 * deco_scale)
    _label_x_axis(ax, x_axis, sd.dist_km, d.shape[1], x_tick_km, C,
                  label_font_size=axis_font_size + 2)
    ax.set_ylabel("Time (ms)", color=C["text"], fontsize=axis_font_size + 2)

    if params.get("fix"):
        fixes = compute_fix_positions(
            sd.timestamps, sd.dist_km, sd.track_lons, sd.track_lats,
            int(params.get("fix_iv", 5)))
        _draw_fix_marks(ax, fixes,
                        color=fix_color or C["highlight"],
                        font_size=fix_font_size, bbox_alpha=fix_bbox_alpha,
                        axes_bg=C["entry"])

    if time_tick_min:
        _add_time_axis(ax, sd.timestamps, sd.dist_km, sd.track_lons, sd.track_lats,
                       time_tick_min, time_fmt=time_fmt,
                       time_font_size=time_font_size, time_align=time_align,
                       colors=C)

    _draw_picks(ax, picks, x_lo, x_hi, d.shape[1])

    # Layout pass: settle all decorations so ax.get_position() reflects the
    # real axes fraction after tight_layout trims margins for labels/colorbar.
    fig.tight_layout(pad=1.2)

    if draw_raster:
        # Two-pass raster path (always): colorize the float array to the EXACT
        # axes pixel box, then imshow it 1:1 (no Matplotlib re-resample, no
        # double-resample bias). ``extent`` places it on whichever X axis mode
        # is active — for "distance" the km extent is a UNIFORM linear mapping
        # of the trace columns (dist_km[0]..dist_km[-1]); genuinely non-uniform
        # ping spacing is intentionally NOT honoured here.
        #
        # The former WYSIWYG non-uniform-spacing path (pcolormesh, then a
        # NonUniformImage) was removed: it colorized the FULL native array
        # up-front (a 14k×20k chain = ~1.1 GiB RGBA — the reported export
        # MemoryError) and/or vector-drew one polygon per data cell, which was
        # both a stability risk and a severe slowdown. System stability and the
        # memory-bounded two-pass sizing overrule perfect distance scaling.
        ax_pos  = ax.get_position()
        axes_w  = max(1, int(round(ax_pos.width  * figsize[0] * dpi)))
        axes_h  = max(1, int(round(ax_pos.height * figsize[1] * dpi)))
        resized = _colorize_for_target(d, cmap_name, vmin, vmax,
                                       (axes_w, axes_h),
                                       max_abs_pool=max_abs_pool,
                                       interp=params.get("interp"))
        ax.imshow(resized, aspect="auto",
                  interpolation=params.get("interp") or "nearest",
                  extent=[x_lo, x_hi, t1, t0], rasterized=True)

        # Re-settle: cheap call that keeps the figure in a consistent state.
        fig.tight_layout(pad=1.2)

    return fig


# ── Chain figure ───────────────────────────────────────────────────────────────

def render_chain_figure(
    ch: Any,
    data: np.ndarray,
    params: dict,
    figsize: tuple = (14, 7),
    dpi: int = 100,
    # ── visualisation options ──
    x_tick_km: Optional[float] = None,
    t_tick_ms: Optional[float] = None,
    show_grid: bool = False,
    title_override: Optional[str] = None,
    clip_lo: float = 0.0,
    time_tick_min: Optional[int] = None,
    # ── new: margins, label config, colours ──
    margin_top_ms: float = 0.0,
    margin_bottom_ms: float = 0.0,
    time_fmt: str = "hhmm",
    time_font_size: float = 6.0,
    time_align: str = "left",
    fix_font_size: float = 5.0,
    fix_bbox_alpha: float = 0.12,
    fix_color: Optional[str] = None,
    colors: Optional[dict] = None,
    x_axis: str = "distance",
    axis_font_size: float = 7.0,
    grid_alpha: float = 0.18,
    grid_lw: float = 0.5,
    # ── layered rendering (density raster base + wiggle/VA overlay) ──
    show_raster: bool = True,
    style: str = "density",
    layout_mode: str = "aspect",
    va_fill: bool = True,
    show_wiggle_line: bool = True,
    wiggle_gain: float = WIGGLE_GAIN_DEFAULT,
    max_wiggles: int = WIGGLE_MAX_TRACES,
    max_abs_pool: bool = False,
    picks: Optional[list] = None,
    deco_scale: float = 1.0,
    box_aspect: Optional[float] = None,
) -> Figure:
    """
    Render a ProfileChain as a headless Matplotlib figure.

    ``deco_scale`` — internal-decoration multiplier (title, colorbar text);
    ``box_aspect`` — physical axes-box h/w letterbox for fixed paper sizes;
    see :func:`render_profile_figure`. Defaults = historical behaviour.

    Global vmax normalisation (max across segments), boundary vlines, colorbar
    shares the same global vmax — avoids brightness seams at file joins (#15).
    ``x_axis`` ("distance"|"trace"|"km") selects the bottom-axis mode — see the
    module-level X-axis helpers. Defaults to "distance" (historical behaviour).

    Layered rendering (``show_raster`` / ``style`` / ``va_fill`` / ``wiggle_gain``
    / ``max_wiggles`` / ``layout_mode``) mirrors :func:`render_profile_figure`:
    a density raster base with an optional budgeted wiggle/variable-area overlay,
    normalised by the median segment vmax. The per-segment raster is only built
    when it will be shown (skipped for 'Wiggle Only').
    """
    C = colors if colors is not None else _C

    i0, i1, t0, t1 = time_window(ch, data.shape[0], params.get("align", False))
    d = data[i0:i1, :]

    # ── Apply margins ──────────────────────────────────────────────────────
    d, t0, t1 = _pad_margins(d, t0, t1, ch.dt_us, margin_top_ms, margin_bottom_ms)

    cmap_base = CMAPS.get(params.get("cmap", "Viridis"), "viridis")
    cmap_name = cmap_base + "_r" if params.get("inv_cmap", False) else cmap_base
    clip_pct  = params.get("clip", 99.6)
    clip_lo_  = params.get("clip_lo", clip_lo)

    seg_boundaries = [0] + [
        int(np.searchsorted(ch.dist_km, bk)) for bk in ch.boundaries_km
    ] + [ch.n_traces]

    # Density always keeps the raster; 'Wiggle Only' (show_raster=False) skips it.
    draw_raster = show_raster or style != "wiggle"
    diverging   = params.get("amp_range") == "diverging"
    _vmax_ov    = float(params["vmax_override"]) if params.get("vmax_override") else None

    # Pass 1 — compute per-segment vmaxes only (cheap).  Colorization is deferred
    # to after tight_layout so the RGBA is sized to the real axes pixel box, not
    # the full figure (see the two-pass comment in render_profile_figure).
    seg_ns    = [e - s for s, e in zip(seg_boundaries[:-1], seg_boundaries[1:])]
    total_tr  = ch.n_traces or 1
    seg_data: list = []    # (seg_slice, vm, vx) — views of d, no extra allocation
    seg_vmaxes: list = []
    for seg_start, seg_end in zip(seg_boundaries[:-1], seg_boundaries[1:]):
        seg_d = d[:, seg_start:seg_end]
        if _vmax_ov is not None:
            vx = _vmax_ov
            vm = -_vmax_ov if diverging else 0.0
        else:
            vm, vx = _vmin_vmax(seg_d, clip_pct, clip_lo_)
            if diverging:
                vm = -vx
        seg_vmaxes.append(vx)
        seg_data.append((seg_d, vm, vx))

    # #15: normalise EVERY segment (and the colorbar) to the same GLOBAL max
    # instead of the per-segment/median vmax. Per-segment normalisation makes
    # each chain-joined file self-contrast-stretch to its own clip percentile,
    # so a quiet segment next to a loud one shows a visible brightness "seam"
    # at the file boundary even though nothing physically changed there.
    vmax_cb = _vmax_ov if _vmax_ov is not None else (float(max(seg_vmaxes)) if seg_vmaxes else 1.0)
    vmin_cb = -vmax_cb if diverging else 0.0
    n_traces_native = ch.n_traces   # trace count for axis labelling (≠ pixel width)

    fig = Figure(figsize=figsize, dpi=dpi, facecolor=C["panel"])
    ax  = fig.add_subplot(111)
    ax.set_facecolor(C["entry"])
    ax.tick_params(colors=C["text"], labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C["accent"])
    if box_aspect:
        # Letterbox for fixed paper sizes — see render_profile_figure.
        ax.set_box_aspect(box_aspect)

    # Trace-axis extent uses the NATIVE trace count, not the (yet-to-be-sized) pixel width.
    x_lo, x_hi = _x_axis_extent(x_axis, ch.dist_km[0], ch.dist_km[-1], n_traces_native)

    if style == "wiggle":
        _draw_wiggle_overlay(ax, d, x_lo, x_hi, t0, t1, vmax=vmax_cb,
                             va_fill=va_fill, show_wiggle_line=show_wiggle_line,
                             wiggle_gain=wiggle_gain,
                             max_traces=max_wiggles, color="#000000")

    # File-seam (chain-join) boundary lines — drawn ONLY when explicitly
    # requested via params["draw_file_boundaries"]. The GUI export dialog
    # defaults this OFF (independent of the interactive viewer's own toggle);
    # the CLI sets it True to preserve its historical always-on behaviour. In the
    # per-trace modes the seam km-value is mapped to its trace-index position.
    if params.get("draw_file_boundaries", False):
        _dk = np.asarray(ch.dist_km, dtype=float)
        for bk in ch.boundaries_km:
            bx = bk if x_axis == "distance" else float(np.searchsorted(_dk, bk))
            ax.axvline(bx, color=C["warn"], lw=1.0, ls="--", alpha=0.7, zorder=5)

    _apply_axes_options(ax, t0, t1, x_lo, x_hi,
                        x_tick_km if x_axis == "distance" else None,
                        t_tick_ms, show_grid, colors=C,
                        axis_font_size=axis_font_size,
                        grid_alpha=grid_alpha, grid_lw=grid_lw)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap_name,
                                      norm=mcolors.Normalize(vmin_cb, vmax_cb))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.015)
    cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=7 * deco_scale)
    cb.ax.yaxis.set_tick_params(labelcolor=C["sub"])
    cb.set_label("Amplitude (global vmax)", color=C["sub"],
                 fontsize=8 * deco_scale)

    title_str = title_override or (
        f"{ch.label}  ·  {ch.n_traces} tr  ·  {ch.dt_us} µs  ·  "
        f"{ch.total_km:.1f} km")
    ax.set_title(title_str, color=C["text"], fontsize=10 * deco_scale,
                 pad=8 * deco_scale)
    _label_x_axis(ax, x_axis, ch.dist_km, n_traces_native, x_tick_km, C,
                  label_font_size=axis_font_size + 2)
    ax.set_ylabel("Time (ms)", color=C["text"], fontsize=axis_font_size + 2)

    if params.get("fix"):
        fixes = compute_fix_positions(
            ch.timestamps, ch.dist_km, ch.track_lons, ch.track_lats,
            int(params.get("fix_iv", 5)))
        _draw_fix_marks(ax, fixes,
                        color=fix_color or C["highlight"],
                        font_size=fix_font_size, bbox_alpha=fix_bbox_alpha,
                        axes_bg=C["entry"])

    if time_tick_min:
        _add_time_axis(ax, ch.timestamps, ch.dist_km, ch.track_lons, ch.track_lats,
                       time_tick_min, time_fmt=time_fmt,
                       time_font_size=time_font_size, time_align=time_align,
                       colors=C, boundaries_km=ch.boundaries_km)

    _draw_picks(ax, picks, x_lo, x_hi, n_traces_native)

    # Layout pass: all decorations settled → measure the real axes pixel box.
    fig.tight_layout(pad=1.2)

    if draw_raster:
        # Pass 2 — colorize segments sized to the ACTUAL axes pixel dimensions.
        ax_pos  = ax.get_position()
        axes_w  = max(1, int(round(ax_pos.width  * figsize[0] * dpi)))
        axes_h  = max(1, int(round(ax_pos.height * figsize[1] * dpi)))
        seg_widths = [round(axes_w * n / total_tr) for n in seg_ns]
        if seg_widths:
            seg_widths[-1] = axes_w - sum(seg_widths[:-1])
        rgba_segs = []
        for i, (seg_d, _vm, _vx) in enumerate(seg_data):
            seg_tgt = (max(1, seg_widths[i]), axes_h)
            # #15: use the unified vmin_cb/vmax_cb (same values driving the
            # colorbar) for every segment, not this segment's own vm/vx.
            rgba_segs.append(_colorize_for_target(seg_d, cmap_name, vmin_cb, vmax_cb, seg_tgt,
                                                  max_abs_pool=max_abs_pool,
                                                  interp=params.get("interp")))
        resized = np.concatenate(rgba_segs, axis=1) if rgba_segs else None
        if resized is not None:
            ax.imshow(resized, aspect="auto",
                      interpolation=params.get("interp") or "nearest",
                      extent=[x_lo, x_hi, t1, t0], rasterized=True)
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
    clip_pct  = params.get("clip", 99.6)
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

    sm2 = matplotlib.cm.ScalarMappable(cmap="inferno", norm=mcolors.Normalize(-50, 0))
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

    # Adjust subplot spacing to prevent label/tick/colorbar clipping without
    # triggering the extra full-DPI Agg rasterization that bbox_inches="tight"
    # requires. tight_layout uses approximate text metrics (not a pixel render).
    fig.tight_layout(pad=0.3)

    kwargs: dict = {"dpi": dpi, "facecolor": fig.get_facecolor()}
    if fmt:
        kwargs["format"] = fmt
    fig.savefig(path, **kwargs)
