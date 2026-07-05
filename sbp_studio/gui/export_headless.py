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

from typing import Optional


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


# ── Export figure render (moved verbatim from gui.tabs._base) ──────────────────

def render_export_figure(obj, cfg, params, node_cfg, scale_cfg, align_enabled,
                         kind, cancel, picks=None):
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
    """
    from sbp_studio.viz.render import (build_theme, render_chain_figure,
                                       render_profile_figure)
    from sbp_studio.gui.tabs._render import (PAPER_SIZES, dpi_for_budget,
                                             effective_export_dpi,
                                             figsize_for_scale)
    # Full-resolution DSP pipeline (alignment + nodes) — shared with the viewport
    # HQ export so the crop is processed identically.
    data, _t0_full = process_full_array(
        obj, params, node_cfg, align_enabled, cancel)
    # figsize: paper size (fixed landscape inches) when the user selected one,
    # otherwise the view-scale-derived figsize (aspect / VE / hybrid mode).
    # Paper size exports skip the post-render WYSIWYG aspect loop since the
    # page dimensions are fixed by the chosen standard size.
    paper_key = cfg.get("paper_size", "")
    if paper_key in PAPER_SIZES:
        figsize = PAPER_SIZES[paper_key]
    else:
        figsize = figsize_for_scale(obj, scale_cfg, int(cfg["dpi"]), cfg["velocity"])
    render_dpi = effective_export_dpi(figsize, data.shape, int(cfg["dpi"]))
    # RAM cap (parity with the CLI): never let the raster exceed the memory
    # budget. effective_export_dpi can raise the DPI toward 2400 to hit the native
    # grid; for a long/deep line that is gigapixels → the GUI export hangs. Cap the
    # DPI to what the budget allows. figsize (the aspect/VE proportions) is kept
    # exact — only pixel density drops, exactly like the CLI's coupled down-scale.
    dpi_cap = dpi_for_budget(figsize, cfg.get("mem_budget_gb"))
    if dpi_cap is not None and render_dpi > dpi_cap:
        render_dpi = max(50, dpi_cap)
    eff_aspect = figsize[0] / figsize[1] if figsize[1] > 0 else None
    render_opts = dict(
        x_tick_km=cfg["x_tick"], t_tick_ms=cfg["t_tick"], show_grid=cfg["grid"],
        title_override=None, clip_lo=0.0, time_tick_min=cfg["time_ticks"],
        margin_top_ms=cfg["margin_top"], margin_bottom_ms=cfg["margin_bottom"],
        time_fmt=cfg["time_fmt"], time_font_size=cfg["time_font_size"],
        time_align=cfg["time_align"], fix_font_size=5.0,
        fix_bbox_alpha=cfg["fix_bbox_alpha"], fix_color=cfg["fix_color"],
        axis_font_size=cfg.get("axis_font_size", 7.0),
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
