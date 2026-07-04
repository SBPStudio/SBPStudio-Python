"""
export_filter.py — Apply the active DSP filter chain to a FULL trace matrix
for filtered SEG-Y export (Reprojector tab's "Aplicar filtros" checkbox), and
the canonical chunked-block DSP runner reused by anything that needs to
process a (possibly segmented, never-fully-materialised) source.

Memory strategy
---------------
If the matrix fits comfortably in a fraction of available RAM, it is
processed in ONE shot — every node sees the complete, continuous trace set,
required for any 2-D / cross-trace filter (e.g. the F-K dip filter) to be
mathematically correct. If it doesn't fit, the matrix is processed in
column BLOCKS with a small overlap halo (derived from the active nodes' own
``trace_halo()``).

``apply_pipeline_to_source`` (Phase 7) is the generalised engine: it reads
each block via ``source.read_columns(c0, c1)`` instead of slicing a
pre-built ndarray — so it works identically over a plain in-memory matrix
(via the ``_NdarraySource`` adapter, what ``apply_pipeline_to_matrix`` below
still presents to existing callers) AND over a segmented
``ProfileChain``/``SegyProfile`` (Phase 7's ``read_columns`` — see
model.py), which NEVER builds the whole-chain monolithic array just to
export-process it. Block reads are sequential, integer column ranges — no
ViewBox/km mapping needed here (that machinery, ``extract_visible_window``,
is for the live preview's distance-axis-driven viewport; this is plain
block iteration over a known trace count), so this module no longer depends
on it. ``full_depth`` is implicit: every block is read at the source's full
row count, so a node's numerical output is identical regardless of which
strategy ran; only the block boundary's edge handling differs from a single
in-memory pass (mitigated by ``trace_halo``).

Pure NumPy/DSP-node logic — no Qt, no file I/O beyond what ``read_columns``
itself does. The Reprojector tab wraps ``apply_pipeline_to_matrix`` into a
plain ``Callable[[np.ndarray], np.ndarray]`` that
``core.io_segy.reproject_one``/``reproject_chain`` calls when the user
checks "Aplicar filtros"; the core layer never imports anything from here
(core stays GUI-free — see CLAUDE.md/this repo's established contract).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .nodes import DSPContext, make_node

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


@runtime_checkable
class MatrixSource(Protocol):
    """Duck-typed contract for anything ``apply_pipeline_to_source`` can read
    column-bounded, full-depth blocks from — a plain ndarray (wrapped by
    ``_NdarraySource``), a ``SegyProfile``, or a ``ProfileChain`` (model.py)
    all satisfy this without any inheritance."""
    def read_columns(self, c0: int, c1: int) -> np.ndarray: ...


class _NdarraySource:
    """Adapts a plain in-memory ``(ns, n_traces)`` ndarray to the
    ``MatrixSource`` contract — lets ``apply_pipeline_to_matrix`` keep its
    existing plain-ndarray signature while sharing ``apply_pipeline_to_source``
    with segmented (``ProfileChain``) callers."""
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def read_columns(self, c0: int, c1: int) -> np.ndarray:
        return self._data[:, c0:c1]


def _available_ram_bytes() -> float:
    """Live free-RAM probe — delegates to the shared capacity-planning
    authority (core._backends.available_ram_bytes), same psutil-else-4GB
    semantics this module always had."""
    from ...core._backends import available_ram_bytes
    return available_ram_bytes()


def fits_in_memory(ns: int, n_traces: int,
                   mem_budget_gb: Optional[float] = None) -> bool:
    """Whether a full-matrix DSP pass over ``(ns, n_traces)`` float32 samples
    is safe given the available (or overridden) RAM budget."""
    needed = ns * n_traces * _BYTES_PER_SAMPLE_WORKING_SET
    budget = (float(mem_budget_gb) * 1024 ** 3 if mem_budget_gb
             else _available_ram_bytes() * _RAM_SAFE_FRACTION)
    return needed <= budget


def apply_pipeline_to_source(source: MatrixSource, ns: int, n_traces: int,
                             node_cfg: List[Tuple[str, Dict]],
                             dt_us: int, cancel=None, *,
                             mem_budget_gb: Optional[float] = None) -> np.ndarray:
    """Run ``node_cfg`` over ``source`` (anything satisfying ``MatrixSource``
    — see module docstring), reading only the column block(s) each pass
    actually needs. Returns a NEW, fully-materialised ``(ns, n_traces)``
    array (the DSP output always needs to exist somewhere; only the SOURCE
    read is block-wise/never-monolithic).

    ``cancel`` — optional ``CancelToken``-like object with a ``.check()``
    method (raises to abort); checked between nodes/blocks so a long export
    can still be cancelled promptly.

    GLOBAL_STATS safety: when any active node sets ``DSPNode.GLOBAL_STATS``
    (LogCompression, CLAHE), the single-shot full-matrix path is forced
    regardless of ``mem_budget_gb``/``fits_in_memory`` — chunking would give
    each block its own normalisation scale, writing a permanent amplitude
    seam into the output. This trades RAM safety for correctness on purpose;
    see the guard's own comment below.
    """
    if not node_cfg:
        return source.read_columns(0, n_traces)
    nodes = [make_node(key, params) for key, params in node_cfg]
    base_ctx = DSPContext(dt_us=dt_us, ns=ns, n_traces=n_traces)
    trace_halo = max((n.trace_halo(base_ctx) for n in nodes), default=0)

    # GLOBAL_STATS guard (see DSPNode.GLOBAL_STATS's docstring): LogCompression/
    # CLAHE normalise by max(abs(data)) over WHATEVER window they're given — a
    # column-block pass would give each block its OWN normalisation scale,
    # writing a permanent amplitude discontinuity into the export at every
    # block boundary. No halo can fix a normalisation mismatch (halos only
    # help spatial/windowed filters, not global ones). Correctness must win
    # over the RAM budget here — exactly like the live preview's own
    # full-Y-band override for these nodes (see preview.py's Strategy B) —
    # so a GLOBAL_STATS node forces the single-shot path unconditionally,
    # regardless of what fits_in_memory() says.
    has_global_stats = any(getattr(n, "GLOBAL_STATS", False) for n in nodes)

    if has_global_stats or fits_in_memory(ns, n_traces, mem_budget_gb):
        out = source.read_columns(0, n_traces)
        for node in nodes:
            if cancel is not None:
                cancel.check()
            out = node.apply(out, base_ctx)
        return out

    # ── Block-based fallback (matrix too large to process in one shot) ────
    # Plain integer column-block iteration — no ViewBox/km mapping needed
    # (that's extract_visible_window's job for the live preview's distance
    # axis; here the "viewport" already IS a trace-index range).
    out = np.empty((ns, n_traces), dtype=np.float32)
    c_vis = 0
    while c_vis < n_traces:
        if cancel is not None:
            cancel.check()
        c_vis_end = min(c_vis + _BLOCK_TRACES, n_traces)
        c0 = max(0, c_vis - trace_halo)
        c1 = min(n_traces, c_vis_end + trace_halo)
        block = source.read_columns(c0, c1)
        ctx = DSPContext(dt_us=dt_us, ns=ns, n_traces=block.shape[1])
        chunk = block
        for node in nodes:
            if cancel is not None:
                cancel.check()
            chunk = node.apply(chunk, ctx)
        out[:, c_vis:c_vis_end] = chunk[:, (c_vis - c0):(c_vis_end - c0)]
        c_vis = c_vis_end
    return out


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

    Unchanged public contract — a thin wrapper over
    ``apply_pipeline_to_source`` (Phase 7's generalised engine) via
    ``_NdarraySource``, so every existing caller (the Reprojector tab) needs
    no changes at all.
    """
    if not node_cfg:
        return data
    ns, n_traces = data.shape
    return apply_pipeline_to_source(
        _NdarraySource(data), ns, n_traces, node_cfg, dt_us, cancel,
        mem_budget_gb=mem_budget_gb)
