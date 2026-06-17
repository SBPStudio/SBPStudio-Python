"""
_render.py — Shared seismic-section view computation (runs on a worker thread).

Given an already-processed data matrix and the control params, computes the
display window, clip vmax, colormap name, and a decimated float32 amplitude
raster. No colorisation happens here — ``SeismicView`` applies the LUT on the
GPU/C++ side via ``pg.ImageItem.setLookupTable``.

Used by both the Visualizer (profile) and Chains tabs so the logic lives in
one place.

Strict separation
-----------------
UI path  : returns a decimated float32 array (MAX_COLS × MAX_ROWS).
           This is the interactive display buffer — strictly temporary.
Export   : ``_base.SubTabbedTab`` calls ``render_profile/chain_figure``
           (core headless renderer) with the FULL-resolution data matrix.
           Zero dependency on the display buffer.
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

# Hard cap on the export render DPI floor — keeps a pathological tiny-figure /
# huge-matrix export from producing an enormous embedded raster.
EXPORT_DPI_CEILING = 2400

# Peak host-RAM the Matplotlib/Agg raster holds per output pixel — mirrors the
# CLI's commands._RASTER_BYTES_PER_PX so the GUI export obeys the SAME memory
# limit. ~4 live RGBA buffers + slack (the WYSIWYG loop draws the canvas a few
# times). Without this cap effective_export_dpi could push a long, deep line to
# ~1.5 Gpx and the export hangs; with it the DPI is reduced to fit the budget.
RASTER_BYTES_PER_PX = 18.0


def pixel_budget(mem_budget_gb: Optional[float]) -> Optional[float]:
    """Max output pixels for a RAM budget in GB (None/≤0 → no limit)."""
    if not mem_budget_gb or mem_budget_gb <= 0:
        return None
    return max(1.0, (float(mem_budget_gb) * 1024 ** 3) / RASTER_BYTES_PER_PX)


def dpi_for_budget(figsize: tuple, mem_budget_gb: Optional[float]) -> Optional[int]:
    """Largest DPI whose raster (figw·dpi × figh·dpi) fits the RAM budget, or
    None when unbounded. Used to CAP the export DPI so a high-DPI / long-line GUI
    export can never balloon past the budget (the cause of the GUI export hang)."""
    budget = pixel_budget(mem_budget_gb)
    w, h = float(figsize[0]), float(figsize[1])
    if budget is None or w <= 0 or h <= 0:
        return None
    return int(max(1, (budget / (w * h)) ** 0.5))

# Interactive-view resolution caps. A screen shows at most a few thousand
# pixels, so colormapping a 100k-trace matrix at full resolution is wasted
# work. We decimate to these caps for the on-screen raster ONLY.
# Export uses the full-resolution matrix via the core renderer (separate path).
MAX_COLS = 8000   # traces (horizontal) in the display buffer
MAX_ROWS = 4000   # samples (vertical)  in the display buffer


def _pool_rows_maxabs(arr: np.ndarray, stride: int) -> np.ndarray:
    """Downsample rows by ``stride`` keeping the max-|amplitude| sample per block.

    Plain striding would drop thin reflectors; max-abs pooling preserves them.
    """
    if stride <= 1:
        return arr
    n = (arr.shape[0] // stride) * stride
    if n == 0:
        return arr
    head = arr[:n].reshape(n // stride, stride, arr.shape[1])
    idx = np.argmax(np.abs(head), axis=1)
    pooled = np.take_along_axis(head, idx[:, None, :], axis=1)[:, 0, :]
    if n < arr.shape[0]:                       # fold any remainder into one row
        tail = arr[n:]
        ti = np.argmax(np.abs(tail), axis=0)
        pooled = np.vstack([pooled, np.take_along_axis(tail, ti[None, :], axis=0)])
    return pooled


def compute_figsize(source: Any, dpi: int, x_scale: Optional[float],
                    ratio: Optional[float], velocity: float,
                    y_scale: Optional[float] = None,
                    px_per_trace: float = 2.0, figheight: float = 7.0,
                    ve: Optional[float] = None,
                    max_aspect: Optional[float] = None) -> tuple:
    """Figure (width_in, height_in) — ports the CLI export-image scale logic,
    including the three finalised scaling modes.

    Width:  --x-scale km/in, else n_traces·px_per_trace/dpi.
    Height priority:
      1. ``ve`` (+ x_scale)  → h = VE·depth_km/x_scale  (fixed vertical exaggeration,
         length-independent). If ``max_aspect`` is set and w/h would exceed it, the
         height is raised so the aspect locks at ``max_aspect`` (the 'noodle' guard).
      2. ``ratio``           → h = w / ratio             (locked aspect, VE floats).
      3. x_scale (no y_scale)→ h = depth_km / x_scale     (VE=1 true scale).
      4. ``y_scale``         → h = record_ms / y_scale.
      5. fallback            → ``figheight``.
    """
    total_km = source.total_km
    record_ms = source.ns * source.dt_us / 1000.0
    depth_km = record_ms * velocity / 2_000_000.0  # TWT → one-way → km

    if x_scale is not None:
        w = max(2.0, total_km / x_scale)
    else:
        w = max(8.0, source.n_traces * px_per_trace / dpi)

    if ve is not None and x_scale is not None:
        h = max(0.5, ve * depth_km / x_scale)
        if max_aspect is not None and max_aspect > 0 and w / h > max_aspect:
            h = w / max_aspect                 # lock aspect, overriding VE
    elif ratio is not None:
        h = max(0.5, w / ratio)
    elif x_scale is not None and y_scale is None:
        h = max(0.5, depth_km / x_scale)
    elif y_scale is not None:
        h = max(0.5, record_ms / y_scale)
    else:
        h = figheight
    return (w, h)


# Internal scale knobs the GUI bakes for the VE/aspect modes (mirror the
# finalised CLI options: Mode 1 uses --px-per-trace 20, Modes 2/3 use --x-scale 2).
# These set only the ABSOLUTE figure size; the exported aspect ratio and VE are
# fixed by the mode's ratio/ve/max_aspect values and are independent of them, and
# the embedded raster resolution is guaranteed separately by effective_export_dpi.
GUI_X_SCALE = 2.0
GUI_PX_PER_TRACE = 20.0


def figsize_for_scale(source: Any, scale_cfg: dict, dpi: int,
                      velocity: float) -> tuple:
    """Resolve (width_in, height_in) for a GUI scale-mode config dict.

    ``scale_cfg`` = {mode: 'aspect'|'ve'|'hybrid', ratio, ve, max_aspect}. This is
    the single place the GUI maps its three radio modes onto ``compute_figsize``.
    """
    mode = scale_cfg.get("mode", "aspect")
    ppt = scale_cfg.get("px_per_trace") or GUI_PX_PER_TRACE
    if mode == "aspect":
        return compute_figsize(source, dpi, None, scale_cfg.get("ratio"),
                               velocity, px_per_trace=ppt)
    # VE / hybrid: anchor width on a physical km/in so VE is well-defined.
    return compute_figsize(
        source, dpi, GUI_X_SCALE, None, velocity,
        ve=scale_cfg.get("ve"),
        max_aspect=(scale_cfg.get("max_aspect") if mode == "hybrid" else None))


def effective_aspect(source: Any, scale_cfg: dict,
                     velocity: float = 1500.0) -> Optional[float]:
    """The W:H aspect the export will actually produce for ``scale_cfg`` on THIS
    line — drives the live PyQtGraph preview so the on-screen proportions match
    the PDF. ``None`` means 'free / fill the panel'.

    aspect mode → the fixed ratio. ve/hybrid → total_km / (VE·depth_km), which
    adapts to the line's length & TWT (and is clamped to max_aspect in hybrid).
    """
    mode = scale_cfg.get("mode", "aspect")
    ratio = scale_cfg.get("ratio") or None
    if mode == "aspect" or source is None:
        return ratio
    record_ms = source.ns * source.dt_us / 1000.0
    depth_km = record_ms * velocity / 2_000_000.0
    denom = (scale_cfg.get("ve") or 0.0) * depth_km
    asp = (source.total_km / denom) if denom > 0 else ratio
    if mode == "hybrid":
        mx = scale_cfg.get("max_aspect") or 0.0
        if mx > 0 and asp is not None:
            asp = min(asp, mx)
    return asp


def effective_export_dpi(figsize: tuple, data_shape: tuple, requested_dpi: int,
                         ceiling: int = EXPORT_DPI_CEILING) -> int:
    """Render DPI that guarantees the embedded raster carries the FULL native
    sample grid — no decimation in the core colouriser.

    The core renders the matrix to ``target = figsize × dpi`` before ``imshow``.
    If ``target`` is smaller than the data's native ``(n_traces, ns)`` the matrix
    is decimated → soft/blurry PDF. We raise the DPI just enough that
    ``figsize × dpi >= (n_traces, ns)`` in BOTH axes, with ``requested_dpi`` as the
    floor (never lower than what the user picked) and ``ceiling`` as the cap.

    This lives in the GUI export wiring only; it does NOT change ``figsize``
    (the aspect/page proportions) nor the core renderer, and it never touches the
    CLI path (which computes its own figsize/DPI).

    Parameters
    ----------
    figsize       : (width_in, height_in) of the export figure
    data_shape    : the FULL native matrix shape ``(ns, n_traces)``
    requested_dpi : the user/dialog-selected DPI (the floor)
    ceiling       : upper bound to keep the embedded raster sane

    Returns
    -------
    int DPI in ``[requested_dpi, ceiling]``.
    """
    src_h, src_w = int(data_shape[0]), int(data_shape[1])     # (ns, n_traces)
    w_in, h_in = float(figsize[0]), float(figsize[1])
    req = int(requested_dpi)
    if w_in <= 0 or h_in <= 0:
        return req
    need = max(src_w / w_in, src_h / h_in)        # dpi to reach native in both axes
    return int(min(max(req, math.ceil(need)), ceiling))


def compute_section(obj: Any, data: np.ndarray, params: dict,
                    boundaries: Sequence[float] = ()) -> dict:
    """Build the dict consumed by ``SeismicView.show_image`` (minus the title).

    Returns a decimated **float32** amplitude array (not RGBA). The
    ``SeismicView`` applies colorisation via a LUT on the C++/GPU side so
    that the expensive Python colourisation step is eliminated from the
    GUI thread entirely.

    The returned ``arr`` is the display buffer; it must NOT be passed to the
    core matplotlib renderer — exports call ``render_profile/chain_figure``
    directly with the full-resolution ``data`` matrix.
    """
    from sbp_studio.core import compute_fix_positions, time_window
    from sbp_studio.core.constants import CMAPS

    _i0, _i1, t0, t1 = time_window(obj, data.shape[0], params["align"])
    d = data[_i0:_i1, :]

    # ── Column decimation: cap at MAX_COLS for the display buffer ──────────
    col_stride = max(1, d.shape[1] // MAX_COLS)
    if col_stride > 1:
        d = d[:, ::col_stride]

    # ── vmax percentile (on the col-decimated matrix, before row pooling) ──
    valid = np.abs(d[~np.isnan(d)])
    vmax = float(np.percentile(valid, params["clip"])) if valid.size else 1.0
    vmax = vmax or 1.0

    # ── Row decimation: max-|amp| pooling to preserve thin reflectors ──────
    row_stride = max(1, d.shape[0] // MAX_ROWS)
    arr = np.nan_to_num(d, nan=0.0).astype(np.float32)
    arr = _pool_rows_maxabs(arr, row_stride)

    cmap_name = CMAPS[params["cmap"]] + ("_r" if params["inv_cmap"] else "")

    fixes = []
    if params.get("fix"):
        fixes = [(f[0], f[1], f[2]) for f in compute_fix_positions(
            obj.timestamps, obj.dist_km, obj.track_lons, obj.track_lats, int(params["fix_iv"]))]

    return dict(
        arr=arr,                                     # (rows_dec, cols_dec) float32
        dist0=float(obj.dist_km[0]), dist1=float(obj.dist_km[-1]),
        t0=t0, t1=t1,
        cmap_name=cmap_name, vmax=vmax,
        fixes=fixes, boundaries=tuple(boundaries),
    )
