"""
pipeline.py — Ordered DSP pipeline with prefix memoization + ViewBox extraction.

The :class:`Pipeline` runs an ordered list of :class:`DSPNode` objects over an
input array, caching the output of every prefix so that editing node *i* only
recomputes nodes *i…N* and reuses the cached output of nodes *0…i-1*.

It is GUI-thread-agnostic pure logic (no Qt) so it is unit-testable in
isolation (see ``tests/test_dsp.py``) and can run inside a worker thread.

ViewBox-limited preview
-----------------------
:func:`extract_visible_window` maps a PyQtGraph ViewBox range to a sub-array of
the FULL matrix, expanded by the pipeline's largest time/trace halo so window
filters keep correct edges. The preview processes only that bounding box;
exporting still uses the full array through the core ``process_*_data`` path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nodes import DSPContext, DSPNode


# ── Pipeline ────────────────────────────────────────────────────────────────────

class Pipeline:
    """Ordered DSP nodes with per-prefix output memoization.

    Cache key for the output after node *i* is the cumulative tuple
    ``(input_token, sig0, sig1, …, sigi)``. Because it is cumulative, an
    upstream edit changes every downstream key (miss → recompute) while
    untouched upstream prefixes keep identical keys (hit → reuse). After each
    run the cache is pruned to the current chain, bounding it to N entries —
    exactly the "state cache between nodes" the design calls for.
    """

    def __init__(self, nodes: Optional[Sequence[DSPNode]] = None) -> None:
        self.nodes: List[DSPNode] = list(nodes) if nodes else []
        self._cache: Dict[tuple, np.ndarray] = {}
        # Instrumentation: number of node.apply() calls in the last process().
        self.last_compute_count: int = 0

    # ── Mutation ────────────────────────────────────────────────────────────
    def set_nodes(self, nodes: Sequence[DSPNode]) -> None:
        self.nodes = list(nodes)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── Execution ───────────────────────────────────────────────────────────
    def process(self, data: np.ndarray, ctx: DSPContext,
                input_token: object = "") -> np.ndarray:
        """Run the chain over ``data``, reusing cached prefixes where possible.

        ``input_token`` identifies the input array (e.g. the ViewBox bounds +
        data version). Keep it stable across re-runs of the same input so the
        cache can hit; change it when the underlying data or window changes.
        """
        cur = data
        cum: tuple = (input_token,)
        valid_keys: set = set()
        self.last_compute_count = 0

        for node in self.nodes:
            cum = cum + (node.signature(),)
            valid_keys.add(cum)
            cached = self._cache.get(cum)
            if cached is not None:
                cur = cached
            else:
                cur = node.apply(cur, ctx)
                self._cache[cum] = cur
                self.last_compute_count += 1

        # Prune anything not on the current chain → bounded to len(nodes).
        if self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k in valid_keys}
        return cur

    # ── Halos ───────────────────────────────────────────────────────────────
    def max_time_halo(self, ctx: DSPContext) -> int:
        return max((n.time_halo_samples(ctx) for n in self.nodes), default=0)

    def max_trace_halo(self, ctx: DSPContext) -> int:
        return max((n.trace_halo(ctx) for n in self.nodes), default=0)


# ── ViewBox-limited extraction ──────────────────────────────────────────────────

@dataclass(frozen=True)
class VisibleWindow:
    """A (possibly decimated) bbox of the full matrix plus the metadata to place
    it back. Crop indices ``r0``/``r1`` are expressed in the DECIMATED ``sub``
    row space; the FULL-array indices ``s0``/``s1``/``c0``/``c1`` and the
    original ``dt`` still drive the on-screen extent (km × ms)."""
    sub:    np.ndarray          # (rows, cols) float32 — the decimated haloed bbox
    r0:     int                 # visible-region row start WITHIN ``sub`` (decimated)
    r1:     int                 # visible-region row stop  WITHIN ``sub`` (decimated)
    c0:     int                 # first trace index in the FULL array
    c1:     int                 # last+1 trace index in the FULL array
    s0:     int                 # first sample index in the FULL array (visible, full-res)
    s1:     int                 # last+1 sample index in the FULL array (visible, full-res)
    row_stride: int             # rows decimation factor (1 = none)
    col_stride: int             # cols decimation factor (1 = none)
    effective_dt_us: int        # dt_us × row_stride — the preview's sampling interval
    token:  tuple               # stable identity of this window (for the cache)

    def crop_visible(self, processed: np.ndarray) -> np.ndarray:
        """Drop the halo rows, returning just the visible region of a processed sub."""
        return processed[self.r0:self.r1, :]


def extract_visible_window(
    data: np.ndarray,
    dist_km: np.ndarray,
    t0_ms: float,
    dt_us: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    *,
    time_halo: int = 0,
    trace_halo: int = 0,
    data_version: int = 0,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
) -> VisibleWindow:
    """Map a PyQtGraph ViewBox (km × ms) to a (decimated) haloed sub-array.

    PREVIEW-ONLY decimation: when ``max_rows``/``max_cols`` are given, the
    haloed window is strided down so the preview never processes more than that
    many samples/traces. Row striding increases the effective sample interval,
    returned as ``effective_dt_us = dt_us × row_stride`` — the caller MUST feed
    this into the node context so time/frequency maths use the right rate.

    The export path passes neither cap (full resolution, original ``dt``).

    Parameters
    ----------
    data        : (ns, n_traces) float32 — the FULL matrix
    dist_km     : (n_traces,) — along-track distance of each trace
    t0_ms       : time of sample row 0 (ms)
    dt_us       : sample interval (µs)
    x_range     : (xmin_km, xmax_km) visible distance range
    y_range     : (ymin_ms, ymax_ms) visible time range
    time_halo   : extra FULL-RES samples kept above/below (edge correctness)
    trace_halo  : extra traces kept left/right (spatial filters)
    data_version: bump when the underlying array changes (cache invalidation)
    max_rows    : cap on processed samples (rows) — decimate if exceeded
    max_cols    : cap on processed traces (cols) — decimate if exceeded

    Returns
    -------
    VisibleWindow with the decimated ``sub``, decimated crop indices,
    ``effective_dt_us`` and a stable token.
    """
    ns, n_traces = data.shape
    dt_ms = dt_us / 1000.0

    # ── Distance → trace columns (dist_km is monotonic non-decreasing) ──────
    xmin, xmax = sorted(x_range)
    c_vis0 = int(np.searchsorted(dist_km, xmin, side="left"))
    c_vis1 = int(np.searchsorted(dist_km, xmax, side="right"))
    c_vis0 = max(0, min(c_vis0, n_traces - 1))
    c_vis1 = max(c_vis0 + 1, min(c_vis1, n_traces))
    c0 = max(0, c_vis0 - trace_halo)
    c1 = min(n_traces, c_vis1 + trace_halo)

    # ── Time → sample rows (full-res, with halo) ────────────────────────────
    ymin, ymax = sorted(y_range)
    s_vis0 = int(np.floor((ymin - t0_ms) / dt_ms))
    s_vis1 = int(np.ceil((ymax - t0_ms) / dt_ms)) + 1
    s_vis0 = max(0, min(s_vis0, ns - 1))
    s_vis1 = max(s_vis0 + 1, min(s_vis1, ns))
    sh0 = max(0, s_vis0 - time_halo)
    sh1 = min(ns, s_vis1 + time_halo)

    full_sub = data[sh0:sh1, c0:c1]
    full_rows = sh1 - sh0

    # ── Decimation (preview only) ───────────────────────────────────────────
    # Ceil-division so the decimated size is GUARANTEED ≤ the cap (floor would
    # let e.g. 50 000 // 8 000 = 6 leave 8 334 > 8 000 columns).
    row_stride = max(1, -(-full_rows // max_rows)) if max_rows else 1
    col_stride = max(1, -(-(c1 - c0) // max_cols)) if max_cols else 1
    if row_stride > 1 or col_stride > 1:
        sub = full_sub[::row_stride, ::col_stride]
    else:
        sub = full_sub
    effective_dt_us = int(dt_us * row_stride)

    # Crop indices in the DECIMATED row space: the visible band [s_vis0, s_vis1)
    # sits at offset (s_vis0 - sh0) full-res rows into the window.
    r0_full = s_vis0 - sh0
    visible_rows_full = s_vis1 - s_vis0
    r0 = r0_full // row_stride
    r1 = min(sub.shape[0], r0 + max(1, -(-visible_rows_full // row_stride)))  # ceil-div

    token = (data_version, c0, c1, sh0, sh1, row_stride, col_stride)
    return VisibleWindow(
        sub=sub, r0=r0, r1=r1, c0=c0, c1=c1, s0=s_vis0, s1=s_vis1,
        row_stride=row_stride, col_stride=col_stride,
        effective_dt_us=effective_dt_us, token=token)
