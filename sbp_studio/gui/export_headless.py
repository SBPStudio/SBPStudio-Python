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


def _shift_picks(picks, c0, c1, t_lo, t_hi):
    """Re-index interpretation picks into a crop: shift ``trace_index`` by the
    crop's left column and drop any pick outside the crop's column/time window.
    Returns duck-typed stand-ins (``_draw_picks`` reads only trace_index/time_ms/
    id), so the caller need not import PickPoint."""
    if not picks:
        return picks
    from types import SimpleNamespace
    ncrop = c1 - c0
    lo, hi = (t_lo, t_hi) if t_lo <= t_hi else (t_hi, t_lo)
    out = []
    for p in picks:
        j = int(p.trace_index) - c0
        if 0 <= j < ncrop and lo <= float(p.time_ms) <= hi:
            out.append(SimpleNamespace(
                trace_index=j, time_ms=float(p.time_ms), id=p.id))
    return out


def crop_export_to_view(obj, data, t0_full, x_range, y_range, picks=None):
    """WYSIWYG export extent. If the visible ViewBox window (``x_range`` km,
    ``y_range`` ms) is a STRICT sub-window of ``obj``'s full extent, return a
    cropped ``(clone, data, picks)`` so the exported file matches the on-screen
    zoom. When the view already covers the whole line (nothing zoomed away), the
    inputs are returned UNCHANGED — the historical full-line export, byte for
    byte (this is what keeps the batch/pool paths, which pass no view, identical).
    ``picks`` are re-indexed into the crop; those outside it are dropped."""
    import numpy as np
    dist = getattr(obj, "dist_km", None)
    if dist is None:
        _LOG.info("WYSIWYG export: source has no distance axis — full line.")
        return obj, data, picks
    dist = np.asarray(dist, dtype=float)
    if dist.size < 2 or data.size == 0:
        _LOG.info("WYSIWYG export: degenerate geometry (%d fixes, %d samples)"
                  " — full line.", dist.size, data.size)
        return obj, data, picks
    n_rows, n_cols = data.shape
    c0, c1, r0, r1 = _view_bounds(obj, data.shape, t0_full, x_range, y_range)
    if c0 <= 0 and c1 >= n_cols and r0 <= 0 and r1 >= n_rows:
        # Field-diagnosable: app.log states explicitly that the whole line was
        # visible, so "the export ignored my zoom" reports can be triaged from
        # the log alone (crop applied vs. view genuinely covered everything).
        _LOG.info("WYSIWYG export: view covers the full line "
                  "(%d traces × %d samples) — no crop.", n_cols, n_rows)
        return obj, data, picks              # whole line visible → no crop
    clone, crop = _crop_at_bounds(obj, data, t0_full, c0, c1, r0, r1)
    _LOG.info("WYSIWYG export: cropped to the on-screen window — traces "
              "[%d:%d) of %d, samples [%d:%d) of %d.",
              c0, c1, n_cols, r0, r1, n_rows)
    t_hi = clone.min_delay + crop.shape[0] * (obj.dt_us / 1000.0)
    picks_out = _shift_picks(picks, c0, c1, clone.min_delay, t_hi)
    return clone, crop, picks_out


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


def _wysiwyg_figsize(obj, data, cfg, view_range, view_figsize):
    """The 'Auto' page: the DATA region's physical on-screen size, in inches.

    The raw ``view_figsize`` is the WHOLE ViewBox — but when the user zooms
    OUT past the data on an axis, the data occupies only part of the screen,
    while the export crop is clamped to the data bounds. Sizing the page to
    the full viewport would stretch the clamped crop across it (the reported
    'proportions destroyed' failure: 26 km of view over a 17.4 km line →
    ~49 % horizontal stretch). WYSIWYG means the PHYSICAL SCALE must carry
    over, per axis: inches-per-km and inches-per-ms on the page must equal
    the screen's. So convert the on-screen scale (view extent over viewport
    inches) to the POST-CROP data extents:

        page_w = data_km · (viewport_w_in / view_km)
        page_h = (data_ms + margins) · (viewport_h_in / view_ms)

    Zoomed IN, crop extent ≈ view extent → page ≈ viewport (unchanged
    behaviour). Zoomed OUT, the page shrinks to the data's true on-screen
    size instead of stretching the data. The export margins ride along in
    height so the data itself stays at true scale. Falls back to the raw
    viewport inches when the scale cannot be derived (no view_range or
    degenerate extents)."""
    import numpy as np
    w_in, h_in = float(view_figsize[0]), float(view_figsize[1])
    if view_range is None:
        return (w_in, h_in)
    xa, xb = sorted((float(view_range[0][0]), float(view_range[0][1])))
    ya, yb = sorted((float(view_range[1][0]), float(view_range[1][1])))
    view_km, view_ms = xb - xa, yb - ya
    dist = np.asarray(getattr(obj, "dist_km", ()), dtype=float)
    if view_km <= 0 or view_ms <= 0 or dist.size < 2 or data.size == 0:
        return (w_in, h_in)
    data_km = float(dist[-1] - dist[0])
    data_ms = data.shape[0] * float(obj.dt_us) / 1000.0
    data_ms += (float(cfg.get("margin_top") or 0.0)
                + float(cfg.get("margin_bottom") or 0.0))
    if data_km <= 0 or data_ms <= 0:
        return (w_in, h_in)
    return (max(0.5, data_km * w_in / view_km),
            max(0.5, data_ms * h_in / view_ms))


# ── Export figure render (moved verbatim from gui.tabs._base) ──────────────────

def render_export_figure(obj, cfg, params, node_cfg, scale_cfg, align_enabled,
                         kind, cancel, picks=None, view_range=None,
                         view_figsize=None):
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

    ``view_figsize`` — the live data-ViewBox's PHYSICAL on-screen size in inches
    (``(w_in, h_in)``), or ``None``. Figsize priority, in order:
      1. an EXPLICIT paper size the user selected in the dialog (A4/A3/A0);
      2. ``view_figsize`` — the 'Auto (match view)' dynamic page: the page takes
         the viewport's own proportions/size and GROWS with the zoomed extent at
         the on-screen scale, never squeezing the section onto a fixed sheet;
      3. the ``figsize_for_scale`` formula (batch/pool — no live view exists).
    """
    from sbp_studio.viz.render import (build_theme, render_chain_figure,
                                       render_profile_figure)
    from sbp_studio.gui.tabs._render import (PAPER_SIZES, dpi_for_budget,
                                             effective_export_dpi,
                                             figsize_for_scale)
    # Full-resolution DSP pipeline (alignment + nodes) — shared with the viewport
    # HQ export so the crop is processed identically.
    data, t0_full = process_full_array(
        obj, params, node_cfg, align_enabled, cancel)
    # WYSIWYG export extent: honour the live viewport crop when the caller passed
    # the ViewBox window and the user has zoomed in (a full-line view is a no-op).
    # The decision is ALWAYS logged — one line in app.log per export states which
    # path ran, so a field report of a wrong extent is triaged from the log alone.
    if view_range is not None:
        obj, data, picks = crop_export_to_view(
            obj, data, t0_full, view_range[0], view_range[1], picks)
    else:
        _LOG.info("WYSIWYG export: no view_range supplied — full-line export "
                  "(batch/pool path, or no live view).")
    # figsize priority (see docstring): explicit paper > live viewport (Auto) >
    # formula. Paper-size exports skip the post-render WYSIWYG aspect loop since
    # the page dimensions are fixed by the chosen standard size.
    paper_key = cfg.get("paper_size", "")
    if paper_key in PAPER_SIZES:
        figsize = PAPER_SIZES[paper_key]
    elif (view_figsize is not None and float(view_figsize[0]) > 0
          and float(view_figsize[1]) > 0):
        # Auto (match view): dynamic paper sized so the PHYSICAL on-screen
        # scale (km/in, ms/in) carries onto the page exactly — the cropped
        # data extents at the viewport's scale, NOT the raw viewport rect
        # (which would stretch a data-clamped crop when the user has zoomed
        # OUT past the data). See _wysiwyg_figsize. Computed AFTER the crop,
        # from the same obj/data the renderer will draw.
        figsize = _wysiwyg_figsize(obj, data, cfg, view_range, view_figsize)
    else:
        figsize = figsize_for_scale(obj, scale_cfg, int(cfg["dpi"]), cfg["velocity"])
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
    eff_aspect = figsize[0] / figsize[1] if figsize[1] > 0 else None
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
    # WYSIWYG aspect fit: grow the figure so the DATA box hits the mode's effective
    # aspect at full size (decorations take a fixed inch margin) — restores pixels.
    # Skipped when a paper size is set: the page dimensions are fixed by the chosen
    # standard size; the two-pass RGBA sizing inside render_*_figure already ensures
    # the raster fills the exact axes box without a secondary Matplotlib resample.
    aspect = None if paper_key in PAPER_SIZES else eff_aspect
    if aspect:
        seis = next((a for a in fig.axes if a.get_images()), None)
        if seis is not None:
            for _ in range(4):
                fig.canvas.draw()
                pos = seis.get_position()
                fw, fh = fig.get_size_inches()
                if pos.width <= 0 or pos.height <= 0:
                    break
                data_w = pos.width * fw
                margin_v = fh - pos.height * fh   # absolute non-data height
                fig.set_size_inches(fw, data_w / aspect + margin_v)
                try:
                    fig.tight_layout(pad=1.2)
                except Exception:
                    pass
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
    name (for progress/failure reporting).
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
            _NoOpCancel())
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
