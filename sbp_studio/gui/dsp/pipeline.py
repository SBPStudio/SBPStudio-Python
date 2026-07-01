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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sbp_studio.core.tasks import CancelToken

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
                input_token: object = "",
                cancel: Optional[CancelToken] = None) -> np.ndarray:
        """Run the chain over ``data``, reusing cached prefixes where possible.

        ``input_token`` identifies the input array (e.g. the ViewBox bounds +
        data version). Keep it stable across re-runs of the same input so the
        cache can hit; change it when the underlying data or window changes.

        ``cancel`` — optional cooperative cancellation, checked BETWEEN each
        node (raises ``Cancelled`` immediately, skipping every remaining
        node in the chain) — the live preview's worker uses this so a
        request superseded by a fresher one (e.g. the user kept dragging a
        slider) aborts at the next node boundary instead of finishing every
        downstream node uselessly. Individual nodes may ALSO check
        ``ctx.cancel`` themselves for finer-grained interruption inside
        their own loop (see PredictiveDeconNode). ``None`` (every caller
        outside the live preview — export, CLI, analysis) never checks."""
        cur = data
        cum: tuple = (input_token,)
        valid_keys: set = set()
        self.last_compute_count = 0

        for node in self.nodes:
            if cancel is not None:
                cancel.check()
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
    cv0:    int                 # visible-region col start WITHIN ``sub`` (decimated)
    cv1:    int                 # visible-region col stop  WITHIN ``sub`` (decimated)
    c0:     int                 # first trace index in the FULL array (incl. halo)
    c1:     int                 # last+1 trace index in the FULL array (incl. halo)
    c_vis0: int                 # first VISIBLE trace index in the FULL array
    c_vis1: int                 # last+1 VISIBLE trace index in the FULL array
    s0:     int                 # first sample index in the FULL array (visible, full-res)
    s1:     int                 # last+1 sample index in the FULL array (visible, full-res)
    row_stride: int             # rows decimation factor (1 = none)
    col_stride: int             # cols decimation factor (1 = none)
    effective_dt_us: int        # dt_us × row_stride — the preview's sampling interval
    token:  tuple               # stable identity of this window (for the cache)

    def crop_visible(self, processed: np.ndarray) -> np.ndarray:
        """Drop the halo rows AND columns, returning just the visible region of a
        processed sub. Cropping columns matters whenever a spatial filter (e.g.
        the F-K dip filter's ``trace_halo``) widened the window: without this,
        the extra halo traces — kept only so the filter sees correct edges —
        would otherwise render on screen past the true ViewBox edge."""
        return processed[self.r0:self.r1, self.cv0:self.cv1]


def _pool_rows_maxabs(arr: np.ndarray, stride: int) -> np.ndarray:
    """Downsample rows by ``stride`` keeping the max-|amplitude| sample per block.

    Retained for reference / backward compatibility.  The active Overview-LOD
    reducer is :func:`_pool_rows_rms` — see its docstring for rationale."""
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


def _pool_rows_rms(arr: np.ndarray, stride: int) -> np.ndarray:
    """Downsample rows by ``stride`` using signed-RMS pooling.

    Each output row = sign(peak_abs_sample) × RMS(block), where the sign comes
    from the sample with the largest magnitude in the block (identical to what
    ``_pool_rows_maxabs`` would keep, so polarity is always correct for wiggles
    and diverging colormaps).

    Why RMS instead of max-abs:
      * **Noise floor**: Gaussian noise of variance σ² produces E[RMS] ≈ σ per
        output row, versus E[max|x|] ≈ 2.3σ for max-abs at stride = 15.  The
        water column and acoustic background therefore look quiet and honest
        rather than artificially "hot".
      * **Reflectors**: multi-sample bandlimited SBP pulses carry their energy
        across several consecutive samples; RMS of the block captures that
        energy faithfully.  A genuinely thin (1-sample) reflector is attenuated
        by √stride, but at overview zoom levels such a feature is sub-pixel and
        the energy still stands visibly above the noise floor.
      * **Sign**: preserved exactly as in max-abs, so no polarity artefacts.

    Output rows = ceil(arr.shape[0] / stride); dtype preserved (float32 in/out).
    This is the active Overview-LOD row reducer called by
    ``extract_visible_window`` when ``overview_max_rows`` is set."""
    if stride <= 1:
        return arr
    n = (arr.shape[0] // stride) * stride
    if n == 0:
        return arr
    head = arr[:n].reshape(n // stride, stride, arr.shape[1])   # (blocks, stride, cols)
    rms  = np.sqrt(np.mean(head ** 2, axis=1))                  # (blocks, cols) float64
    idx  = np.argmax(np.abs(head), axis=1)                      # (blocks, cols)
    peak = np.take_along_axis(head, idx[:, None, :], axis=1)[:, 0, :]  # (blocks, cols)
    pooled = np.sign(peak) * rms
    if n < arr.shape[0]:
        tail      = arr[n:]                                      # (rem, cols)
        tail_rms  = np.sqrt(np.mean(tail ** 2, axis=0))         # (cols,)
        ti        = np.argmax(np.abs(tail), axis=0)             # (cols,)
        tail_sign = np.sign(tail[ti, np.arange(tail.shape[1])]) # (cols,)
        pooled    = np.vstack([pooled, (tail_sign * tail_rms)[None, :]])
    return pooled.astype(arr.dtype)


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
    full_depth: bool = False,
    active_ns_lookup: Optional[Callable[[int, int], Optional[int]]] = None,
    active_band_lookup: Optional[Callable[[int, int], Optional[Tuple[int, int]]]] = None,
    col_margin: int = 0,
    y_halo: Optional[int] = None,
    overview_max_rows: Optional[int] = None,
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
    y_range     : (ymin_ms, ymax_ms) visible time range — IGNORED for the row
                  extent when ``full_depth`` is True (still used to compute
                  the visible-row crop-back indices ``r0``/``r1``).
    time_halo   : extra FULL-RES samples kept above/below (edge correctness).
                  Ignored when ``full_depth`` is True (the whole trace is
                  already kept, so no edge margin is needed).
    trace_halo  : extra traces kept left/right (spatial filters)
    data_version: bump when the underlying array changes (cache invalidation)
    max_rows    : cap on processed samples (rows) — decimate if exceeded.
                  Ignored when ``full_depth`` is True.
    max_cols    : cap on processed traces (cols) — decimate if exceeded
    full_depth  : keep EVERY sample of every selected trace (rows 0..ns,
                  stride 1) instead of cropping/decimating to the visible
                  Y-range — see PreviewController._refresh's docstring for
                  why: a time-series filter (AGC, Deconvolution, Envelope)
                  computed on a vertically truncated trace yields a
                  different result than the same filter run on the full
                  trace, so the live preview must process full depth and
                  crop the Y-range only AFTER the DSP pipeline runs.
    active_ns_lookup: optional ``(c0, c1) -> Optional[int]`` resolving the
                  REAL listening-window DEPTH (bottom only) of the file(s)
                  spanning the resolved column range [c0, c1) — see
                  ``SegyProfile.active_ns`` / ``ProfileChain.
                  active_ns_for_traces``. Superseded by ``active_band_lookup``
                  when both are given; kept for callers that only have the
                  legacy bottom-only value.
    active_band_lookup: optional ``(c0, c1) -> Optional[(lo, hi)]`` resolving
                  the REAL signal band — top AND bottom — of the file(s)
                  spanning [c0, c1); see ``SegyProfile.active_lo``/
                  ``active_ns`` and ``active_band_for_traces``. Some surveys
                  record every shot with a FIXED window sized for the
                  deepest expected water depth (dead trailing rows below the
                  signal — ``hi``) and/or start recording before the
                  sub-bottom reflectors of interest arrive (dead leading
                  rows above it — ``lo``, e.g. a deep-water travel-time
                  delay on a high-res SBP system). When ``full_depth`` is
                  True, the processed window is clamped to ``[lo, hi]`` —
                  extended if needed so it NEVER excludes the Y-range the
                  user is actually viewing. ``None`` (no lookup, or an
                  unloaded constituent) falls back to ``active_ns_lookup``
                  (bottom only, ``lo`` stays 0) or, if that's also absent,
                  the unmodified ``[0, ns]`` — identical to omitting both.
    y_halo      : optional sample count. When given (together with
                  ``full_depth``), the processed window becomes the VISIBLE
                  Y-range plus this halo on each side — clamped to the
                  active band, never narrower than what's visible — instead
                  of the WHOLE active band. Safe ONLY for nodes whose result
                  depends on local context (a halo) rather than the entire
                  window's statistics — see DSPNode.GLOBAL_STATS; the caller
                  (PreviewController._refresh) is responsible for passing
                  ``None`` here whenever a GLOBAL_STATS node is active, in
                  which case the whole active band is processed exactly as
                  before this parameter existed.
    overview_max_rows: optional Overview-LOD row cap. ONLY consulted when
                  ``full_depth`` is True. When given, the extracted rows are
                  RMS-POOLED (see ``_pool_rows_rms``) down to at most this
                  many — the zoom-out fix: a deep cadena whose visible band is
                  15-30k rows tall is pooled before the DSP runs, instead of
                  pushing every native sample through the chain only to be
                  downsampled for a ~1.5k-px screen. ``None``
                  keeps the exact full-resolution behaviour (row_stride == 1).
                  Pooling (not striding) preserves thin reflectors; the caller
                  (PreviewController._refresh) only sets this when the visible
                  Y-span genuinely oversamples the screen, where the per-sample
                  DSP difference is sub-pixel and invisible.

    Returns
    -------
    VisibleWindow with the decimated ``sub``, decimated crop indices,
    ``effective_dt_us`` and a stable token.
    """
    ns, n_traces = data.shape
    dt_ms = (dt_us or 1) / 1000.0  # guard: dt_us=0 (corrupt SEG-Y) must not produce ÷0

    # ── Distance → trace columns (dist_km is monotonic non-decreasing) ──────
    xmin, xmax = sorted(x_range)
    c_vis0 = int(np.searchsorted(dist_km, xmin, side="left"))
    c_vis1 = int(np.searchsorted(dist_km, xmax, side="right"))
    c_vis0 = max(0, min(c_vis0, n_traces - 1))
    c_vis1 = max(c_vis0 + 1, min(c_vis1, n_traces))
    # ``col_margin`` widens the processed band BEYOND the spatial-filter halo so
    # the live preview can pan laterally WITHIN the extra columns without
    # re-running DSP (see preview.reslice_band / the pan-margin cache). It never
    # shrinks below ``trace_halo`` — F-K et al. always get their edge context.
    side = max(trace_halo, col_margin)
    c0 = max(0, c_vis0 - side)
    c1 = min(n_traces, c_vis1 + side)

    # ── Time → sample rows ───────────────────────────────────────────────────
    ymin, ymax = sorted(y_range)
    s_vis0 = int(np.floor((ymin - t0_ms) / dt_ms))
    s_vis1 = int(np.ceil((ymax - t0_ms) / dt_ms)) + 1
    s_vis0 = max(0, min(s_vis0, ns - 1))
    s_vis1 = max(s_vis0 + 1, min(s_vis1, ns))
    if full_depth:
        # Every sample of every selected trace, full resolution — no halo
        # needed against the BAND edge (there's no crop edge there to
        # protect against pre-DSP); a separate, NODE-DRIVEN halo (y_halo)
        # may still apply against the VISIBLE-Y edge — see below.
        band = active_band_lookup(c0, c1) if active_band_lookup is not None else None
        if band is not None:
            lo_band, hi_band = band
        else:
            lo_band, hi_band = 0, ns
            if active_ns_lookup is not None:
                active_max_row = active_ns_lookup(c0, c1)
                if active_max_row is not None:
                    hi_band = min(ns, int(active_max_row))
        lo_band = max(0, min(lo_band, ns))
        hi_band = max(lo_band, min(hi_band, ns))

        if y_halo is not None:
            # Strategy B: visible Y-range + a node-driven halo, clamped to
            # the active band — far cheaper than the whole band when zoomed
            # in on a deep file, safe only for local (non-GLOBAL_STATS)
            # nodes (see the parameter docstring).
            sh0 = max(lo_band, s_vis0 - y_halo)
            sh1 = min(hi_band, s_vis1 + y_halo)
        else:
            # Strategy A only (or no band info at all): the WHOLE active
            # band, extended if needed for the visible range.
            sh0 = lo_band
            sh1 = max(hi_band, s_vis1)

        # HARD safety floor, regardless of strategy: never let either bound
        # cut into the visible Y-range the user is actually trying to see —
        # a mis-detected active band, or a deliberate zoom into what looks
        # like a dead zone, must still show real data, never silently crop it.
        sh0 = min(sh0, s_vis0)
        sh1 = max(sh1, s_vis1)
        sh0 = max(0, min(sh0, ns - 1))
        sh1 = max(sh0 + 1, min(sh1, ns))
    else:
        sh0 = max(0, s_vis0 - time_halo)
        sh1 = min(ns, s_vis1 + time_halo)

    full_sub = data[sh0:sh1, c0:c1]
    full_rows = sh1 - sh0

    # ── Decimation (preview only) ───────────────────────────────────────────
    # Ceil-division so the decimated size is GUARANTEED ≤ the cap (floor would
    # let e.g. 50 000 // 8 000 = 6 leave 8 334 > 8 000 columns).
    #
    # Row decimation has three regimes:
    #   1. full_depth + overview_max_rows → Overview LOD: RMS POOLING down to
    #      overview_max_rows (zoom-out fix — honest energy, see _pool_rows_rms).
    #      The pipeline then runs on ~overview_max_rows rows instead of the
    #      full 15-30k-deep band.
    #   2. full_depth (no overview)       → exact path, NO row decimation.
    #   3. not full_depth                 → legacy plain striding to max_rows.
    if full_depth:
        row_stride = (max(1, -(-full_rows // overview_max_rows))
                      if overview_max_rows else 1)
    else:
        row_stride = max(1, -(-full_rows // max_rows)) if max_rows else 1
    col_stride = max(1, -(-(c1 - c0) // max_cols)) if max_cols else 1

    # Columns first (cheap stride), then rows. In Overview mode the rows are
    # max-abs POOLED (peak-preserving); every other path uses plain striding.
    sub = full_sub
    if col_stride > 1:
        sub = sub[:, ::col_stride]
    if row_stride > 1:
        if full_depth and overview_max_rows:
            sub = _pool_rows_rms(sub, row_stride)
        else:
            sub = sub[::row_stride, :]
    # Materialise ONCE, here, as a contiguous array. ``data[sh0:sh1, c0:c1]`` is
    # a COLUMN-band slice of a C-contiguous matrix, hence strided (each row's
    # band is contiguous but rows sit n_traces apart); ``[::stride]`` only adds
    # more striding. Every downstream node would otherwise walk that strided
    # view — and scipy's fft2 / many core ops copy a non-contiguous input
    # internally anyway, so they'd each pay the gather. One ascontiguousarray
    # here amortises that into a single cache-friendly copy the whole chain
    # then reuses. np.ascontiguousarray is a no-op when sub already happens to
    # be contiguous (e.g. a full-width window), so this never copies needlessly.
    sub = np.ascontiguousarray(sub)
    effective_dt_us = int(dt_us * row_stride)

    # Crop indices in the DECIMATED row space: the visible band [s_vis0, s_vis1)
    # sits at offset (s_vis0 - sh0) full-res rows into the window.
    r0_full = s_vis0 - sh0
    visible_rows_full = s_vis1 - s_vis0
    r0 = r0_full // row_stride
    r1 = min(sub.shape[0], r0 + max(1, -(-visible_rows_full // row_stride)))  # ceil-div

    # Same ceil-div mapping for the trace-halo (spatial filters, e.g. F-K) so the
    # halo columns are cropped away exactly like the halo rows above.
    cv0_full = c_vis0 - c0
    visible_cols_full = c_vis1 - c_vis0
    cv0 = cv0_full // col_stride
    cv1 = min(sub.shape[1], cv0 + max(1, -(-visible_cols_full // col_stride)))  # ceil-div

    token = (data_version, c0, c1, sh0, sh1, row_stride, col_stride)
    return VisibleWindow(
        sub=sub, r0=r0, r1=r1, cv0=cv0, cv1=cv1,
        c0=c0, c1=c1, c_vis0=c_vis0, c_vis1=c_vis1, s0=s_vis0, s1=s_vis1,
        row_stride=row_stride, col_stride=col_stride,
        effective_dt_us=effective_dt_us, token=token)


# ── Pan-margin cache (processed-band re-slice) ───────────────────────────────────

@dataclass(frozen=True)
class ProcessedBand:
    """A pipeline-PROCESSED, full-depth column band kept so the live preview can
    pan laterally WITHIN it on the GUI thread — re-slicing the already-computed
    result instead of dispatching the DSP worker for every new column window.

    The band is the un-decimated processed output of an ``extract_visible_window``
    call made with a generous ``col_margin`` (see PreviewController._refresh). It
    is only built for the zoomed-in regime where ``col_stride == 1`` and
    ``row_stride == 1`` (full_depth), so column ``j`` of ``processed`` is exactly
    FULL-array trace ``c0 + j`` and row ``i`` is sample ``i`` — making the
    re-slice pure integer indexing with no decimation bookkeeping.

    Re-slicing is valid for EVERY filter (even the column-coupled F-K dip filter):
    the band's interior columns were computed with the full band as 2-D context,
    which is at least as correct as a tight per-window halo. Only the outermost
    ``trace_halo`` columns of the band have degraded context, so the trusted
    interior (where a re-slice is served) excludes them — see ``reslice_band``."""
    processed:    np.ndarray   # (rows, c1-c0) float32 — col_stride=1, row_stride=1
    c0:           int          # FULL-array first trace of the band (inclusive)
    c1:           int          # FULL-array last+1 trace of the band
    n_traces:     int          # FULL-array trace count (for band-edge detection)
    trace_halo:   int          # spatial-filter halo → trusted-interior inset
    t0_ms:        float        # time of sample row 0 (ms)
    dt_us:        int          # sample interval (µs)
    data_ns:      int          # FULL data depth (rows in the source matrix) — the
                               # true sample bottom, so a full-depth-Y view that
                               # overshoots by the usual +1 sample is CLAMPED here
                               # rather than mis-flagged as "deeper than the band"
    data_version: int          # must match the request's (else the band is stale)
    pipeline_sig: tuple        # node-chain signature this band was processed with
    row0:         int = 0      # FULL-array row index of processed[0, :]. Always 0
                               # when the band covers the whole active depth (the
                               # original, pre-Strategy-B contract); nonzero when
                               # Strategy B's visible-Y+halo windowing built this
                               # band starting partway down the trace — see
                               # extract_visible_window's ``y_halo``. Default 0
                               # keeps every existing call site (which never built
                               # a row-offset band) unchanged.


def reslice_band(
    band: ProcessedBand,
    dist_km: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    *,
    max_cols: int,
    data_version: int,
    pipeline_sig: tuple,
) -> Optional[VisibleWindow]:
    """Map a NEW viewport onto a cached :class:`ProcessedBand`, returning a
    :class:`VisibleWindow` that points into ``band.processed`` (so the existing
    ``win.crop_visible`` / render path works unchanged) — or ``None`` if the
    band cannot faithfully serve this viewport, in which case the caller MUST
    fall back to a fresh worker dispatch.

    Returns ``None`` (a "miss") when ANY of:
      * the band is stale (``data_version`` or ``pipeline_sig`` differ);
      * the viewport is zoomed OUT past the column cap (``max_cols``) — the band
        is un-decimated, so serving it would blow the render budget;
      * the visible columns reach into the band's outer ``trace_halo`` (degraded
        spatial-filter context) — unless that side is the true array edge;
      * the visible time range extends DEEPER than the band was processed.

    All index math mirrors ``extract_visible_window`` exactly (verified by an
    equivalence test) so a re-sliced crop is byte-identical to what a direct
    full-window extraction of the same viewport would produce."""
    if band.data_version != data_version or band.pipeline_sig != pipeline_sig:
        return None

    dt_ms = (band.dt_us or 1) / 1000.0
    xmin, xmax = sorted(x_range)
    n_traces = band.n_traces
    c_vis0 = int(np.searchsorted(dist_km, xmin, side="left"))
    c_vis1 = int(np.searchsorted(dist_km, xmax, side="right"))
    c_vis0 = max(0, min(c_vis0, n_traces - 1))
    c_vis1 = max(c_vis0 + 1, min(c_vis1, n_traces))

    # Zoomed-out past the cap → would need decimation the band doesn't have.
    if (c_vis1 - c_vis0) > max_cols:
        return None

    # Trusted interior: exclude the band's outer halo columns (degraded F-K
    # context), except where the band already sits at the true array edge.
    left_limit  = band.c0 if band.c0 == 0 else band.c0 + band.trace_halo
    right_limit = band.c1 if band.c1 == n_traces else band.c1 - band.trace_halo
    if c_vis0 < left_limit or c_vis1 > right_limit:
        return None

    rows = band.processed.shape[0]
    ymin, ymax = sorted(y_range)
    s_vis0 = int(np.floor((ymin - band.t0_ms) / dt_ms))
    s_vis1 = int(np.ceil((ymax - band.t0_ms) / dt_ms)) + 1
    # Clamp to the true data bottom FIRST: a full-depth-Y view legitimately
    # asks for ``ns + 1`` (the ceil + 1 overshoot) — that is the data bottom,
    # not a request for rows the band lacks.
    s_vis1 = min(s_vis1, band.data_ns)
    s_vis0 = max(0, min(s_vis0, max(0, band.data_ns - 1)))

    # Band-relative row range: band.processed[0, :] is FULL-array row
    # band.row0 (0 unless Strategy B's windowed-with-halo mode built this
    # band — see ProcessedBand.row0). Either bound falling outside the
    # band's own row coverage (above OR below — a Strategy-B band can miss
    # on EITHER side, unlike the historical row0=0 case which could only
    # miss on the bottom) → miss, re-dispatch for a band that covers it.
    r_vis0 = s_vis0 - band.row0
    r_vis1 = s_vis1 - band.row0
    if r_vis0 < 0 or r_vis1 > rows:
        return None
    r_vis0 = max(0, min(r_vis0, rows - 1))
    r_vis1 = max(r_vis0 + 1, min(r_vis1, rows))

    cv0 = c_vis0 - band.c0
    cv1 = c_vis1 - band.c0
    # Per-viewport token (distinct for each visible range) so downstream
    # geometry caches keyed on it — e.g. the wiggle path — still rebuild on pan.
    token = (data_version, c_vis0, c_vis1, s_vis0, s_vis1, 1, 1)
    return VisibleWindow(
        sub=band.processed, r0=r_vis0, r1=r_vis1, cv0=cv0, cv1=cv1,
        c0=band.c0, c1=band.c1, c_vis0=c_vis0, c_vis1=c_vis1,
        s0=s_vis0, s1=s_vis1, row_stride=1, col_stride=1,
        effective_dt_us=band.dt_us, token=token)
