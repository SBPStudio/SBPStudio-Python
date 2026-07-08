"""
export_headless.py — The GUI's export-render pipeline, extracted Qt-free.

This module is the UNTANGLING of the batch-export render path: everything a
worker process needs to reproduce a GUI-quality export — the full-resolution
DSP pass, the scale/DPI math, the WYSIWYG aspect fit, and the figure save —
with ZERO PyQt6 anywhere in its import graph. That property (enforced by a
regression test that imports this module in a fresh interpreter and asserts
no ``PyQt6`` module loaded) is what makes a ``ProcessPoolExecutor`` batch
export safe in the frozen Windows exe: a spawned worker re-executes
``SBPstudio.exe``, is intercepted by ``freeze_support()`` (see
applications/SBPStudio_GUI.py), and imports only THIS module's core+viz stack
— never the Qt UI.

Provenance / fidelity contract
--------------------------------
``process_full_array`` and ``render_export_figure`` are the former
``gui.tabs._base._process_full_array`` / ``_render_export_figure`` moved
VERBATIM (same math, same WYSIWYG aspect-fit loop, same RAM-capped DPI) —
``_base`` now delegates here, so the single export, the sequential batch and
the process-pool batch all run literally the same code. The only structural
change: the render dispatch takes a ``kind`` string ("profile" | "chain")
instead of a live SourceHandler object (handlers aren't picklable and aren't
needed — they only dispatched to the two headless viz renderers anyway).

Import-purity rules for future edits
--------------------------------------
Only ``sbp_studio.core``, ``sbp_studio.viz``, numpy/stdlib, and the PURE gui
submodules (``gui.dsp.nodes``, ``gui.tabs._render`` — module-level imports are
numpy/stdlib/core only; their package __init__s are lazy, see
``gui/__init__.py``) may be imported here. Anything Qt-touching breaks the
frozen-worker contract and the purity regression test will fail.
"""
from __future__ import annotations

import logging
from typing import Optional

_LOG = logging.getLogger(__name__)


class _NoOpCancel:
    """Cancel stand-in for worker processes — the PARENT cancels by shutting
    the pool down; an in-flight worker item always runs to completion."""
    def check(self) -> None:
        return None


# ── Full-resolution DSP pass (moved verbatim from gui.tabs._base) ──────────────

def process_full_array(obj, params, node_cfg, align_enabled, cancel):
    """Run static delay-alignment + the dynamic DSP nodes on the FULL native
    matrix. Returns ``(processed, t0_ms)`` where ``t0_ms`` is the time of the
    processed array's row 0 (min_delay when aligned, else delay_ms)."""
    from sbp_studio.core import apply_delay_alignment
    from sbp_studio.gui.dsp.nodes import DSPContext, make_node
    data = obj.data.copy()
    t0 = float(getattr(obj, "delay_ms", 0.0) or 0.0)
    if align_enabled and getattr(obj, "delays", None) is not None:
        data = apply_delay_alignment(
            data, obj.delays, obj.min_delay, obj.dt_us,
            fill_value=params["fill_value"])
        t0 = float(getattr(obj, "min_delay", 0.0) or 0.0)
    ctx = DSPContext.from_source(obj)
    for key, npar in node_cfg:
        cancel.check()
        data = make_node(key, npar).apply(data, ctx)
    cancel.check()
    return data, t0


# ── Viewport crop (WYSIWYG export extent + the HQ overlay) ─────────────────────
# Moved here from gui.tabs._base so the HQ viewport overlay AND the file export's
# WYSIWYG crop share ONE Qt-free implementation. Pure numpy — no PyQt6 (the
# import-purity regression test still passes).

def _view_bounds(obj, shape, t0_full, x_range, y_range):
    """Map the visible ViewBox window to half-open matrix bounds
    ``(c0, c1, r0, r1)``. Columns via the per-trace distance axis (searchsorted),
    rows via the processed array's time origin ``t0_full``. Single source of
    truth for both the crop and the 'is the whole line visible?' test."""
    import numpy as np
    dist = np.asarray(obj.dist_km, dtype=float)
    n_rows, n_cols = shape
    xa, xb = sorted((float(x_range[0]), float(x_range[1])))
    c0 = int(np.clip(np.searchsorted(dist, xa, side="left"), 0, n_cols - 1))
    c1 = int(np.clip(np.searchsorted(dist, xb, side="right"), c0 + 1, n_cols))
    dt_ms = obj.dt_us / 1000.0
    ya, yb = sorted((float(y_range[0]), float(y_range[1])))
    r0 = int(np.clip(round((ya - t0_full) / dt_ms), 0, n_rows - 1))
    r1 = int(np.clip(round((yb - t0_full) / dt_ms), r0 + 1, n_rows))
    return c0, c1, r0, r1


def _crop_at_bounds(obj, proc, t0_full, c0, c1, r0, r1):
    """Slice ``proc`` to ``[r0:r1, c0:c1]`` and return a lightweight source clone
    carrying only the attributes the headless renderer + figsize_for_scale read,
    with the per-trace arrays sliced so nothing is misaligned."""
    import numpy as np
    from types import SimpleNamespace
    dist = np.asarray(obj.dist_km, dtype=float)
    dt_ms = obj.dt_us / 1000.0
    crop = np.ascontiguousarray(proc[r0:r1, c0:c1])

    def _sl(name):
        arr = getattr(obj, name, None)
        return None if arr is None else np.asarray(arr)[c0:c1]

    dk = dist[c0:c1]
    t0_crop = t0_full + r0 * dt_ms
    base_name = getattr(obj, "name", None) or getattr(obj, "label", "") or "viewport"
    # File-seam boundaries (chain export) that fall inside the cropped km window —
    # absolute km positions, so keep only those strictly between the crop edges.
    raw_b = getattr(obj, "boundaries_km", None)
    if raw_b is not None and dk.size:
        lo_k, hi_k = float(dk[0]), float(dk[-1])
        bnds = [float(b) for b in raw_b if lo_k < float(b) < hi_k]
    else:
        bnds = []
    clone = SimpleNamespace(
        # Keep the real name/label (no suffix) so the cropped export's title
        # matches the full-line export exactly — only the extent changes.
        name=base_name, label=base_name,
        n_traces=int(c1 - c0), ns=int(r1 - r0), dt_us=int(obj.dt_us),
        dist_km=dk, total_km=float(dk[-1] - dk[0]) if dk.size else 0.0,
        timestamps=list(getattr(obj, "timestamps", []) or [])[c0:c1],
        lons=_sl("lons"), lats=_sl("lats"), delays=_sl("delays"),
        water_depth=_sl("water_depth"),
        # Display track (the renderer's top time-axis reads these — profile AND
        # chain); sliced to the crop so position labels stay aligned.
        track_lons=_sl("track_lons"), track_lats=_sl("track_lats"),
        boundaries_km=bnds,
        # Already-processed data → render with align=False; the time origin is the
        # crop's first row, so both align branches of time_window yield t0_crop.
        min_delay=t0_crop, delay_ms=t0_crop, max_delay=t0_crop,
    )
    return clone, crop


def _crop_for_viewport(obj, proc, t0_full, x_range, y_range):
    """Crop the processed matrix to the visible ViewBox window and return a
    lightweight source clone the headless renderer can consume (used by the HQ
    viewport overlay). ``x_range`` is (km, km), ``y_range`` is (ms, ms)."""
    c0, c1, r0, r1 = _view_bounds(obj, proc.shape, t0_full, x_range, y_range)
    return _crop_at_bounds(obj, proc, t0_full, c0, c1, r0, r1)


# NOTE: the former ``crop_export_to_view`` (WYSIWYG *extent* cropping) was
# REMOVED by field directive: zooming is a SCALE-authoring gesture, not an
# extent selection - the export always covers the FULL line, at the
# on-screen physical scale (see ``_page_from_scale``). The crop engine above
# survives solely for the in-viewer HQ overlay (``_crop_for_viewport``).


# HARD physical page ceiling, inches per axis. The PDF format caps a page at
# 14 400 × 14 400 user units = 200 × 200 in — Acrobat refuses to open anything
# larger ('Page dimensions exceed limits', field-confirmed), and Matplotlib's
# Agg canvas separately caps each axis at 2^16 px. An unbounded 'Auto' page
# (long line × generous in/km) can exceed both. When the finished page is
# over the ceiling it is scaled UNIFORMLY (both axes by the same factor —
# aspect and VE untouched) and the DPI is raised by the inverse factor, so
# the PIXEL COUNT IS PRESERVED exactly: this is a print-scale reduction, not
# a quality cap (uncompromising-quality directive intact). 199 leaves margin
# under the 200 limit.
MAX_PAGE_IN = 199.0


# Reference page for decoration sizing: the renderers' own default figsize
# (12 × 7 in), where every historical point size (7 pt axes, 10 pt title, 8 pt
# colorbar label…) is proportionate by construction. deco_scale == 1.0 there.
_DECO_REF_AREA = 12.0 * 7.0


def deco_scale_for(figsize) -> float:
    """Decoration scale factor for a dynamically-sized page.

    Matplotlib font sizes are POINTS — an absolute physical unit (1/72 in) —
    so on the WYSIWYG 'Auto' page (which can legitimately be a few inches
    wide, or metres for a long line) fixed point sizes become disproportionate:
    massive text crowding a small page, or invisible text on a huge one. Scale
    every decoration by √(page_area / reference_area) — the geometric mean
    treats narrow, wide and tall pages symmetrically — clamped to [0.5, 3.0]
    so text never becomes unreadably small nor comically large."""
    w, h = float(figsize[0]), float(figsize[1])
    if w <= 0 or h <= 0:
        return 1.0
    return max(0.5, min(3.0, ((w * h) / _DECO_REF_AREA) ** 0.5))


def _page_from_scale(obj, data, cfg, view_scale):
    """The 'Auto' page: the FULL data extent at the on-screen physical scale.

    ``view_scale`` is the GUI-thread snapshot ``(in_per_km, in_per_ms)`` — the
    viewport's physical scale per axis (SeismicView.current_view_scale). The
    viewport is a SCALE AUTHOR, never an extent selector: zooming tunes the
    proportions, and the export applies them to the WHOLE line —

        page_w = full_line_km   · in_per_km
        page_h = (full_record_ms + export margins) · in_per_ms

    The two axes are mathematically INDEPENDENT: compressing X on screen
    narrows the page but cannot touch its height, and vice versa. Margins ride
    along in height at the same ms-per-inch so the data itself stays at true
    scale. Returns ``None`` for degenerate input (caller falls back to the
    formula figsize)."""
    import numpy as np
    ipk, ipm = float(view_scale[0]), float(view_scale[1])
    dist = np.asarray(getattr(obj, "dist_km", ()), dtype=float)
    if ipk <= 0 or ipm <= 0 or dist.size < 2 or data.size == 0:
        return None
    data_km = float(dist[-1] - dist[0])
    data_ms = data.shape[0] * float(obj.dt_us) / 1000.0
    data_ms += (float(cfg.get("margin_top") or 0.0)
                + float(cfg.get("margin_bottom") or 0.0))
    if data_km <= 0 or data_ms <= 0:
        return None
    return (max(0.5, data_km * ipk), max(0.5, data_ms * ipm))


# ── Export figure render (moved verbatim from gui.tabs._base) ──────────────────

def render_export_figure(obj, cfg, params, node_cfg, scale_cfg, align_enabled,
                         kind, cancel, picks=None, view_scale=None):
    """Shared per-item export render — used by the single export, the
    sequential batch AND the process-pool batch, so their quality can never
    drift.

    Runs the FULL-resolution DSP pipeline (static delay-alignment + dynamic
    nodes) on ``obj.data`` (the 100 % native matrix — never the live view's
    decimated ``_arr``), then renders to a Matplotlib Figure at a DPI floored
    by ``effective_export_dpi`` so the embedded raster is never decimated,
    then applies the WYSIWYG aspect fit. Returns ``(fig, render_dpi)``; the
    caller saves and closes the figure.

    ``kind`` — "profile" → :func:`viz.render.render_profile_figure`,
    "chain" → :func:`viz.render.render_chain_figure` (the same two renderers
    the SourceHandler strategy objects dispatched to).

    ``picks`` — interpretation markers to burn into the raster at full export
    resolution; ``None``/empty draws nothing (batch exports never pass it).

    ``view_range`` — ``((x0_km, x1_km), (y0_ms, y1_ms))`` of the live ViewBox, or
    ``None``. When given AND the user has zoomed into a sub-window, the processed
    matrix + object + picks are cropped to it so the exported file matches the
    screen (WYSIWYG extent). A full-line view (or ``None`` — the batch/pool path)
    leaves everything untouched, so the historical full-line export is unchanged.

    ``view_scale`` — the GUI-thread scale snapshot ``(in_per_km, in_per_ms)``,
    or ``None``. The viewport is a SCALE AUTHOR only — the export ALWAYS
    covers the full line (extent cropping was removed by field directive).
    Figsize priority, in order:
      1. an EXPLICIT paper size the user selected in the dialog (A4/A3/A0);
      2. ``view_scale`` — the 'Auto' dynamic page: the FULL data extent at the
         on-screen physical scale, each axis independent (_page_from_scale);
      3. the ``figsize_for_scale`` formula (no live view existed).
    """
    from sbp_studio.viz.render import (build_theme, render_chain_figure,
                                       render_profile_figure)
    from sbp_studio.gui.tabs._render import (PAPER_SIZES, dpi_for_budget,
                                             effective_export_dpi,
                                             figsize_for_scale)
    # Full-resolution DSP pipeline (alignment + nodes) on the FULL matrix.
    data, _t0_full = process_full_array(
        obj, params, node_cfg, align_enabled, cancel)
    # figsize priority (see docstring): explicit paper > full line at the
    # viewport scale (Auto) > formula. The decision is ALWAYS logged — one line
    # in app.log per export states which sizing ran, for field triage.
    paper_key = cfg.get("paper_size", "")
    scaled = (_page_from_scale(obj, data, cfg, view_scale)
              if view_scale is not None else None)
    if paper_key in PAPER_SIZES:
        figsize = PAPER_SIZES[paper_key]
        _LOG.info("export sizing: fixed paper %s.", paper_key)
    elif scaled is not None:
        figsize = scaled
        _LOG.info("export sizing: FULL line at the on-screen scale "
                  "(%.4f in/km, %.5f in/ms).", view_scale[0], view_scale[1])
    else:
        figsize = figsize_for_scale(obj, scale_cfg, int(cfg["dpi"]), cfg["velocity"])
        _LOG.info("export sizing: formula figsize (no live view scale — "
                  "batch default, or nothing displayed).")
    render_dpi = effective_export_dpi(figsize, data.shape, int(cfg["dpi"]))
    # RAM cap (parity with the CLI): never let the raster exceed the memory
    # budget. effective_export_dpi can raise the DPI toward 2400 to hit the native
    # grid; for a long/deep line that is gigapixels → the GUI export hangs. Cap the
    # DPI to what the budget allows. figsize (the aspect/VE proportions) is kept
    # exact — only pixel density drops, exactly like the CLI's coupled down-scale.
    # UNCOMPROMISING QUALITY (user directive): there is NO hidden hard pixel
    # ceiling. The ONLY limit on the canvas is the USER-OWNED memory budget
    # (the export dialog's 'Memory budget (GB)' — cfg["mem_budget_gb"]): its
    # DPI clamp is announced in the dialog BEFORE exporting, and the dialog
    # separately warns when the budget exceeds the machine's free RAM. A
    # 64 GB workstation with a raised budget gets its full-resolution export;
    # driving the budget past physical RAM is an informed user decision.
    dpi_cap = dpi_for_budget(figsize, cfg.get("mem_budget_gb"))
    if dpi_cap is not None and render_dpi > dpi_cap:
        render_dpi = max(50, dpi_cap)
    _LOG.info("export render: figsize=%.2f×%.2f in, dpi=%d (~%.0f Mpx), "
              "budget=%s GB.", figsize[0], figsize[1], render_dpi,
              (figsize[0] * render_dpi) * (figsize[1] * render_dpi) / 1e6,
              cfg.get("mem_budget_gb"))
    # Proportionate decorations on the dynamic page: every font (and the
    # renderer-internal title/colorbar text, via the deco_scale kwarg) is
    # multiplied by the page-area factor, so the layout reads identically
    # whether the WYSIWYG page came out 4 inches wide or 40. The user's
    # dialog font choices scale RELATIVELY (their ratios are preserved).
    _deco = deco_scale_for(figsize)
    render_opts = dict(
        x_tick_km=cfg["x_tick"], t_tick_ms=cfg["t_tick"], show_grid=cfg["grid"],
        title_override=None, clip_lo=0.0, time_tick_min=cfg["time_ticks"],
        margin_top_ms=cfg["margin_top"], margin_bottom_ms=cfg["margin_bottom"],
        time_fmt=cfg["time_fmt"],
        time_font_size=cfg["time_font_size"] * _deco,
        time_align=cfg["time_align"], fix_font_size=5.0 * _deco,
        fix_bbox_alpha=cfg["fix_bbox_alpha"], fix_color=cfg["fix_color"],
        axis_font_size=cfg.get("axis_font_size", 7.0) * _deco,
        deco_scale=_deco,
        grid_alpha=cfg.get("grid_alpha", 0.18), grid_lw=cfg.get("grid_lw", 0.5),
        colors=build_theme(theme=cfg["theme"]),
        # Layered render style from the live controls (display_params / scale_cfg),
        # so the export honours Density vs Wiggle + raster underlay regardless of
        # whether wiggles are currently on screen. These bind to render_figure's
        # explicit show_raster / style / layout_mode kwargs (not **kwargs).
        show_raster=bool(params.get("show_raster", True)),
        style=params.get("style", "density"),
        layout_mode=scale_cfg.get("layout_mode", "aspect"),
        # Variable-area fill / wiggle-line visibility, mirrored from the live
        # PyQtGraph view's checkboxes (display_params) so exports never silently
        # diverge from what's on screen.
        va_fill=bool(params.get("va_fill", True)),
        show_wiggle_line=bool(params.get("show_wiggle_line", True)),
        # Reflector-safe downscale: when the export must shrink below native
        # (RAM-capped DPI), pool by max-|amplitude| instead of bilinear so thin
        # high-amplitude reflectors are preserved. Default on.
        max_abs_pool=bool(cfg.get("max_abs_pool", True)),
        picks=picks)
    render_fn = render_chain_figure if kind == "chain" else render_profile_figure
    fig = render_fn(obj, data, params, figsize=figsize, dpi=render_dpi,
                    **render_opts)
    # WYSIWYG data-box fit: ``figsize`` is the target size OF THE DATA BOX
    # ITSELF (full extents × the on-screen scale, or the formula size); the
    # page grows beyond it by exactly the MEASURED decoration margins, PER
    # AXIS. This makes the two axes mathematically independent — compressing
    # X can never change the page height — and lands in/km + in/ms on the
    # page EXACTLY. (The former loop was width-anchored: it squeezed the data
    # box into (page − decorations), shrinking BOTH axes by the decoration
    # fraction — the 'narrowing X also shortened the page' field bug.)
    # Skipped when a paper size is set: the page dimensions are fixed by the
    # chosen standard size.
    if paper_key not in PAPER_SIZES:
        target_w, target_h = float(figsize[0]), float(figsize[1])
        seis = next((a for a in fig.axes if a.get_images()), None)
        if seis is not None and target_w > 0 and target_h > 0:
            for _ in range(4):
                fig.canvas.draw()
                pos = seis.get_position()
                fw, fh = fig.get_size_inches()
                if pos.width <= 0 or pos.height <= 0:
                    break
                margin_w = fw - pos.width * fw    # decoration inches, X axis
                margin_h = fh - pos.height * fh   # decoration inches, Y axis
                new_w = target_w + margin_w
                new_h = target_h + margin_h
                if abs(new_w - fw) < 1e-3 and abs(new_h - fh) < 1e-3:
                    break                          # converged
                fig.set_size_inches(new_w, new_h)
                try:
                    fig.tight_layout(pad=1.2)
                except Exception:
                    pass
    # Physical page ceiling (PDF 200 in limit / Agg 2^16 px — see MAX_PAGE_IN).
    # Applied to the FINISHED page (after the data-box growth), scaled
    # uniformly with inverse-DPI compensation: proportions, VE and the total
    # pixel count are all preserved exactly.
    fw, fh = fig.get_size_inches()
    over = max(float(fw), float(fh)) / MAX_PAGE_IN
    if over > 1.0:
        s = 1.0 / over
        fig.set_size_inches(fw * s, fh * s)
        new_dpi = max(1, int(round(render_dpi / s)))
        _LOG.warning(
            "export page %.0f×%.0f in exceeds the %.0f in PDF/renderer ceiling"
            " — uniformly print-scaled to %.1f×%.1f in; DPI %d→%d (pixel "
            "count preserved).", fw, fh, MAX_PAGE_IN, fw * s, fh * s,
            render_dpi, new_dpi)
        render_dpi = new_dpi
    return fig, render_dpi


# ── Process-pool worker + orchestrator (GUI batch export, roadmap #1-GUI) ──────

def render_batch_item(payload: dict) -> tuple:
    """Top-level, picklable ProcessPoolExecutor target: load ONE profile from
    disk and export it with the exact GUI render pipeline above. Returns
    ``(out_path, ok, error)``; never raises (a worker exception would poison
    the pool). Profile kind only — chains assemble multi-file state in place
    and are deliberately excluded from the pool (see ChainHandler).

    ``payload`` (all plain picklable values, snapshotted on the GUI thread):
    path, out, fmt, pdf_page, cfg, params, node_cfg, scale_cfg, align_enabled,
    name (for progress/failure reporting), and optionally ``view_scale`` — the
    ``(in_per_km, in_per_ms)`` snapshot of the live viewport, so EVERY item in
    the batch exports its FULL line at the SAME on-screen visual scale (the
    'set a proportion, export everything' field workflow).
    """
    out = str(payload.get("out", ""))
    fig = None
    try:
        from sbp_studio.core import load_profile
        from sbp_studio.viz.render import save_figure

        prof = load_profile(payload["path"], load_traces=True)
        if getattr(prof, "error", None):
            raise RuntimeError(f"Cannot load {prof.name}: {prof.error}")
        fig, render_dpi = render_export_figure(
            prof, payload["cfg"], payload["params"], payload["node_cfg"],
            payload["scale_cfg"], payload["align_enabled"], "profile",
            _NoOpCancel(), view_scale=payload.get("view_scale"))
        save_figure(fig, out, dpi=render_dpi, fmt=payload["fmt"],
                    pdf_page=payload["pdf_page"])
        return (out, True, "")
    except BaseException as exc:              # noqa: BLE001 — never poison the pool
        return (out, False, f"{type(exc).__name__}: {exc}")
    finally:
        if fig is not None:
            fig.clear()


def divide_mem_budget(cfg: dict, n_workers: int) -> dict:
    """Per-worker copy of ``cfg`` with the rasteriser RAM budget divided
    across the pool — the same anti-oversubscription rule as the CLI batch
    (see cli.commands.cmd_batch_export): the dialog's budget (or the
    fraction-of-available default) is a MACHINE-wide budget, so each of N
    workers gets 1/N of it, floored at 0.25 GB so an export stays usable."""
    from sbp_studio.core._backends import available_ram_bytes

    out = dict(cfg)
    user = cfg.get("mem_budget_gb")
    if user and user > 0:
        total_gb = float(user)
    else:
        total_gb = (available_ram_bytes() * 0.55) / 1024 ** 3
    out["mem_budget_gb"] = max(0.25, total_gb / max(1, int(n_workers)))
    return out


def run_batch_export_pool(payloads: list, n_workers: int, *,
                          progress, cancel=None) -> tuple:
    """Drive ``render_batch_item`` over ``payloads`` in a ProcessPoolExecutor,
    mapping completions onto ``progress(fraction, name)`` via as_completed.
    Returns ``(saved, failed)`` with the same shapes as the sequential loop
    (``_run_batch_export_loop``). One bad item is recorded, not fatal.

    Cancellation granularity: checked between COMPLETIONS — items already
    running in workers finish (a process can't be interrupted cooperatively);
    everything still queued is dropped via shutdown(cancel_futures=True).
    """
    import concurrent.futures as cf

    saved: list = []
    failed: list = []
    n = len(payloads)
    with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(render_batch_item, p): p for p in payloads}
        try:
            for done, fut in enumerate(cf.as_completed(futs), 1):
                if cancel is not None:
                    cancel.check()            # raises → except below cleans up
                out, ok, err = fut.result()
                name = futs[fut].get("name", out)
                if ok:
                    saved.append(out)
                else:
                    failed.append((name, err))
                progress(done / n, name)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    return saved, failed
