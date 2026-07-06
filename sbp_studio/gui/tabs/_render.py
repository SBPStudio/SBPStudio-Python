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

# Standard paper sizes for the export dialog's paper-size dropdown.
# Landscape orientation (width > height) suits seismic profiles.
# Keys must match the QComboBox items in ExportDialog.
PAPER_SIZES: dict = {
    "A4": (11.69,  8.27),
    "A3": (16.54, 11.69),
    "A0": (46.81, 33.11),
}

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


# HARD canvas ceiling for a GUI export, in megapixels — an ABSOLUTE safety cap,
# independent of the (user-tunable) RAM budget. The budget default (6 GB ≈ 358
# Mpx) proved far too permissive in the field: a ~293 Mpx canvas allocation
# (matplotlib's resample index buffer alone is 4 bytes/px, the full draw ~18)
# hard-crashed an 8 GB laptop with MemoryError. 100 Mpx (e.g. ~12500 × 8000 px,
# an A0 page at ~260 DPI) is far beyond any legitimate deliverable while keeping
# the worst-case draw footprint under ~2 GB. When the cap engages, only the DPI
# drops (figsize — the physical proportions/VE — is never touched) and a warning
# is logged so the reduction is visible in app.log.
MAX_EXPORT_MEGAPIXELS = 100.0


def hard_dpi_cap(figsize: tuple,
                 max_megapixels: float = MAX_EXPORT_MEGAPIXELS) -> Optional[int]:
    """Largest DPI whose canvas (figw·dpi × figh·dpi) stays under the HARD
    megapixel ceiling, or None for degenerate figsize. Same coupled down-scale
    contract as :func:`dpi_for_budget`, but non-negotiable (not budget-driven)."""
    w, h = float(figsize[0]), float(figsize[1])
    if w <= 0 or h <= 0 or max_megapixels <= 0:
        return None
    return int(max(1, ((max_megapixels * 1e6) / (w * h)) ** 0.5))


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

# Horizontal-scale control: figure width derives from the physical trace spacing
# (traces per cm), DPI-independent. Default chosen to land near the historical
# px/trace width for a typical line; the slider lets the user stretch/compress.
#
# PERFORMANCE NOTE: a "rigorous" VE formula that derived height from this
# control's actual width (h = VE * w * depth_km / total_km, so VE stays a true
# physical ratio regardless of trace density) was implemented and then
# REVERTED at the user's explicit request — it made every drag tick on this
# slider recompute the full VE/hybrid vertical geometry, which was too slow/
# laggy in practice. VE/hybrid height is therefore intentionally DECOUPLED
# from this control again (see the ``else`` branch in figsize_for_scale,
# which always uses the fixed GUI_X_SCALE constant) so dragging this slider
# stays cheap: only the width changes, never the height.
DEFAULT_TRACES_PER_CM = 40.0
_CM_PER_IN = 2.54


def figsize_for_scale(source: Any, scale_cfg: dict, dpi: int,
                      velocity: float) -> tuple:
    """Resolve (width_in, height_in) for a GUI scale-mode config dict.

    ``scale_cfg`` keys:
      mode          : 'free'|'aspect'|'ve'|'hybrid'  (the height sub-mode)
      ratio/ve/max_aspect : the sub-mode values
      traces_per_cm : horizontal scale — figure WIDTH = n_traces / tpc / 2.54
      layout_mode   : 'aspect'|'decoupled'

    WIDTH always comes ONLY from ``traces_per_cm`` (``w = n_traces/tpc/2.54``) —
    changing it never rescales time by itself, and is CHEAP (no dependency on
    VE/total_km — see the performance note above). HEIGHT then follows the
    active mode's OWN rule, independently of that width:
      * ``ve``            — h = VE·depth_km / GUI_X_SCALE (length-independent;
        traces_per_cm changing w does NOT change h, by design — this keeps
        dragging the horizontal slider instant, at the cost of VE not being
        perfectly anchored to the chosen trace density).
      * ``aspect``         — h = w / ratio (the locked W:H ratio is the
        invariant, so h is RECOMPUTED from the new w to hold it exactly).
      * ``hybrid``         — same VE-based h as 've', but capped so w/h never
        exceeds ``max_aspect`` (recomputed against the new w too).
      * ``free`` (and any unset/legacy mode) — falls back to the VE formula;
        the live preview never reaches this branch for 'free' (see
        :func:`effective_aspect`, which returns ``None`` first), this is only
        the EXPORT figure's fallback size.

    Legacy: if ``traces_per_cm`` is absent (UI not yet upgraded) the old
    px/trace-based path is used so nothing breaks mid-migration.
    """
    tpc = scale_cfg.get("traces_per_cm")
    if tpc is None:
        return _legacy_figsize_for_scale(source, scale_cfg, dpi, velocity)
    tpc = float(tpc) or DEFAULT_TRACES_PER_CM
    n_tr = int(getattr(source, "n_traces", 0) or 0)
    record_ms = source.ns * source.dt_us / 1000.0
    depth_km = record_ms * velocity / 2_000_000.0

    w = max(2.0, n_tr / tpc / _CM_PER_IN)

    mode = scale_cfg.get("mode", "ve")
    if mode == "aspect":
        ratio = scale_cfg.get("ratio") or 1.0
        h = max(0.5, w / ratio)
    elif mode == "free" and scale_cfg.get("pixel_aspect"):
        # WYSIWYG free mode: caller injected the ViewBox's W/H pixel ratio so the
        # exported figure has the same landscape/portrait feel as the live screen.
        h = max(0.5, w / float(scale_cfg["pixel_aspect"]))
    else:
        ve = scale_cfg.get("ve") or 1.0
        h = max(0.5, ve * depth_km / GUI_X_SCALE)
        if mode == "hybrid":
            max_aspect = scale_cfg.get("max_aspect") or 0.0
            if max_aspect > 0 and w / h > max_aspect:
                h = w / max_aspect             # lock aspect, overriding VE
    return (w, h)


def _legacy_figsize_for_scale(source: Any, scale_cfg: dict, dpi: int,
                              velocity: float) -> tuple:
    """Pre-traces-per-cm figure sizing (px/trace based). Retained as the fallback
    until the controls panel supplies ``traces_per_cm`` / ``layout_mode``."""
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

    Mode ``'free'`` ALWAYS returns ``None`` here, before anything else runs —
    it has no ratio/VE rule of its own; it means 'no aspect lock' full stop
    (``SeismicView.set_aspect(None)`` unlocks the ViewBox and auto-fits).

    With ``traces_per_cm`` present this is simply ``width/height`` from
    :func:`figsize_for_scale` (covers both layout modes). Falls back to the legacy
    formula when the new key is absent.
    """
    if scale_cfg.get("mode") == "free":
        return None
    if scale_cfg.get("traces_per_cm") is None:
        return _legacy_effective_aspect(source, scale_cfg, velocity)
    if source is None:
        return scale_cfg.get("ratio") or None
    w, h = figsize_for_scale(source, scale_cfg, 100, velocity)
    return (w / h) if h > 0 else None


def _legacy_effective_aspect(source: Any, scale_cfg: dict,
                             velocity: float = 1500.0) -> Optional[float]:
    """Pre-traces-per-cm aspect resolution (mode-based). Fallback only."""
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


def traces_per_cm_on_screen(visible_traces: float, viewport_px: float,
                            dpi: float) -> float:
    """Physical horizontal density (traces per cm) currently shown on screen.

    ``visible_traces`` traces span ``viewport_px`` logical pixels of the data
    ViewBox, on a display of ``dpi`` logical dots-per-inch. Physical width in cm
    is ``viewport_px / dpi * 2.54``, so the density is ``visible_traces`` over
    that. Pure + DPI-derived so the live SeismicView can convert a mouse-wheel
    X-zoom into the SAME 'traces/cm' unit the export width uses
    (``figsize_for_scale``: ``w = n_traces / tpc / 2.54``), keeping the control
    in sync with the zoom. Returns 0.0 for degenerate input (caller skips)."""
    if visible_traces <= 0 or viewport_px <= 0 or dpi <= 0:
        return 0.0
    width_cm = viewport_px / dpi * _CM_PER_IN
    return (visible_traces / width_cm) if width_cm > 0 else 0.0


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
