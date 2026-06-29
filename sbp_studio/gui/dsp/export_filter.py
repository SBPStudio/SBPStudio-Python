"""
export_filter.py — Apply the active DSP filter chain to a FULL trace matrix
for filtered SEG-Y export (Reprojector tab's "Aplicar filtros" checkbox).

Memory strategy
---------------
If the matrix fits comfortably in a fraction of available RAM, it is
processed in ONE shot — every node sees the complete, continuous trace set,
required for any 2-D / cross-trace filter (e.g. the F-K dip filter) to be
mathematically correct. If it doesn't fit, the matrix is processed in
column BLOCKS with a small overlap halo (derived from the active nodes' own
``trace_halo()``), reusing ``extract_visible_window(..., full_depth=True)``
— the SAME window+halo+full_depth machinery the live preview uses, with a
synthetic "distance" axis (1 unit per trace) so a column-index range can be
expressed directly as an ``x_range``. ``full_depth=True`` guarantees every
block still keeps every time sample (the same "no Y-cropping" guarantee
that fixed the live-preview's color-pumping/washout bug — see
``PreviewController._refresh``), so a node's numerical output is identical
regardless of which strategy ran; only the block boundary's edge handling
differs from the in-memory chain (mitigated by ``trace_halo``).

Pure NumPy/DSP-node logic — no Qt, no file I/O. The Reprojector tab wraps
this into a plain ``Callable[[np.ndarray], np.ndarray]`` that
``core.io_segy.reproject_one``/``reproject_chain`` calls when the user
checks "Aplicar filtros"; the core layer never imports anything from here
(core stays GUI-free — see CLAUDE.md/this repo's established contract).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .nodes import DSPContext, make_node
from .pipeline import extract_visible_window

# Fraction of currently-available RAM a single full-matrix DSP pass may use.
# Mirrors cli.commands._RAM_SAFE_FRACTION's philosophy: leave headroom for
# the OS, the already-loaded source matrix, and the pipeline's own working
# copies (a node's _apply may transiently hold 2-3x the input size).
_RAM_SAFE_FRACTION = 0.30
_BYTES_PER_SAMPLE_WORKING_SET = 4 * 3   # float32 in + ~2 working copies, worst case

# Column block size (traces) for the chunked fallback. Overlap is added on
# top of this per-block, sized to whatever spatial halo the active nodes
# actually report (see trace_halo below) — never a fixed guess.
_BLOCK_TRACES = 2000


def _available_ram_bytes() -> float:
    """Best-effort free-RAM probe — see cli.commands._available_ram_bytes."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        return 4.0 * 1024 ** 3


def fits_in_memory(ns: int, n_traces: int,
                   mem_budget_gb: Optional[float] = None) -> bool:
    """Whether a full-matrix DSP pass over ``(ns, n_traces)`` float32 samples
    is safe given the available (or overridden) RAM budget."""
    needed = ns * n_traces * _BYTES_PER_SAMPLE_WORKING_SET
    budget = (float(mem_budget_gb) * 1024 ** 3 if mem_budget_gb
             else _available_ram_bytes() * _RAM_SAFE_FRACTION)
    return needed <= budget


def apply_pipeline_to_matrix(data: np.ndarray,
                             node_cfg: List[Tuple[str, Dict]],
                             dt_us: int, cancel=None, *,
                             mem_budget_gb: Optional[float] = None) -> np.ndarray:
    """Run ``node_cfg`` (the ``[(KEY, params), …]`` node snapshot — see
    ``PipelinePanel.active_nodes()``) over the FULL ``data`` matrix.

    Returns a NEW array of the SAME shape, so the caller can write it
    straight back into a SEG-Y file with the original trace/sample geometry
    unchanged (no delay-alignment or row-count change is applied here —
    that would break a 1:1 per-trace write into the source file's spec).

    ``cancel`` — optional ``CancelToken``-like object with a ``.check()``
    method (raises to abort); checked between nodes/blocks so a long export
    can still be cancelled promptly.
    """
    if not node_cfg:
        return data
    ns, n_traces = data.shape
    nodes = [make_node(key, params) for key, params in node_cfg]
    base_ctx = DSPContext(dt_us=dt_us, ns=ns, n_traces=n_traces)
    trace_halo = max((n.trace_halo(base_ctx) for n in nodes), default=0)

    if fits_in_memory(ns, n_traces, mem_budget_gb):
        out = data
        for node in nodes:
            if cancel is not None:
                cancel.check()
            out = node.apply(out, base_ctx)
        return out

    # ── Block-based fallback (matrix too large to process in one shot) ────
    idx_axis = np.arange(n_traces, dtype=np.float64)   # 1 unit == 1 trace
    t_span_ms = ns * (dt_us / 1000.0)
    out = np.empty_like(data)
    c = 0
    while c < n_traces:
        if cancel is not None:
            cancel.check()
        c_end = min(c + _BLOCK_TRACES, n_traces)
        win = extract_visible_window(
            data, idx_axis, t0_ms=0.0, dt_us=dt_us,
            x_range=(float(c), float(c_end - 1)), y_range=(0.0, t_span_ms),
            trace_halo=trace_halo, full_depth=True)
        ctx = DSPContext(dt_us=win.effective_dt_us, ns=win.sub.shape[0],
                         n_traces=win.sub.shape[1])
        chunk = win.sub
        for node in nodes:
            if cancel is not None:
                cancel.check()
            chunk = node.apply(chunk, ctx)
        out[:, win.c_vis0:win.c_vis1] = win.crop_visible(chunk)
        c = c_end
    return out
