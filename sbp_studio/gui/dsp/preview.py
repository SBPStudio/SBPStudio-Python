"""
preview.py — Live, ViewBox-limited DSP preview controller.

Closes the interactive loop:

    user pans/zooms  ──┐
                       ├─▶  _refresh()  ──▶  extract visible window (+halo)
    user edits node ───┘                    ──▶  Pipeline.process (memoized)
                                            ──▶  crop halo ──▶  colorize+show

The controller NEVER processes the full array — only the visible bounding box
(columns decimated to a cap; rows kept full so AGC stays exact at every zoom).
The export path is a completely separate routine that runs the same node
configuration over the 100 % full array (see ``tabs/_base._on_export_image``).

Pure-Qt orchestration; all DSP is delegated to the nodes, which delegate to the
core. No DSP math lives here.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from PyQt6.QtCore import QObject

from sbp_studio.core.logger import get_logger

from .nodes import DSPContext
from .pipeline import Pipeline, extract_visible_window
from .preview_worker import PipelineWorker

_LOG = get_logger("dsp.preview")

# Preview raster caps. Rows AND columns are decimated so the pipeline never
# processes more than this on screen; row decimation raises the effective dt
# (passed into the node context) so the DSP maths stay physically correct.
# Zoomed in deeply, strides fall to 1 → full-resolution, exact processing.
# Columns capped at 4000: a 4K monitor is only 3840 px wide, so processing
# more traces than that is pure overdraw the screen can't even show.
MAX_PREVIEW_COLS = 4000
MAX_PREVIEW_ROWS = 4000

# Higher caps used ONLY while a NEEDS_FULL_RES node (e.g. the F-K dip filter) is
# active, so the 2-D filter sees the true viewport trace spacing. Bounded so a
# zoomed-out viewport can't request a multi-thousand² 2-D FFT that would freeze
# the settle; at ≤ these the window is taken at stride 1 (exact dt + spacing).
FK_MAX_COLS = 4096
FK_MAX_ROWS = 4096

# Live wiggle tuning. Deflection of a full-scale sample ≈ one trace slot × gain.
# Rows are capped well below the raster's so the single batched curve stays fluid
# (a wiggle needs far fewer samples than a density raster to read cleanly).
# Gain < 1.0 (not 1.0): at gain=1.0 a fully-saturated sample (amp clipped to
# ±vmax) deflects by EXACTLY one full inter-trace spacing, landing precisely on
# the neighbouring trace's own anchor — zero margin. Near a strong reflector
# (several consecutive near-saturated samples), that fill visibly bleeds past
# where the neighbour's own data would justify any fill, reading as a "blob"
# with no local VA data. 0.9 keeps deflection strong/legible while guaranteeing
# a gap between adjacent traces' maximum excursions, so fills never cross.
WIGGLE_GAIN     = 0.9
WIGGLE_MAX_ROWS = 2000   # row cap for the wiggle LINE (polyline only, light)

# Variable-area fill has its own tighter caps to prevent arrayToQPath from
# blocking the GUI thread on dense/deep full-fits:
#   VA_MAX_ROWS       — VA polygon uses at most this many time samples per trace
#                       (independently of WIGGLE_MAX_ROWS so the line stays sharp).
#   VA_TRACE_THRESHOLD — above this many drawn traces the fill lobes are too thin
#                       to read and the QPainterPath cost dominates; suppress VA.
VA_MAX_ROWS        = 600
VA_TRACE_THRESHOLD = 600


class PreviewController(QObject):
    """Drives a :class:`SeismicView` from a :class:`PipelinePanel`, live.

    Parameters
    ----------
    view          : the SeismicView to update (put into preview mode here).
    panel         : the PipelinePanel supplying the ordered DSP nodes.
    get_source    : callable → the active SegyProfile/ProfileChain (or None).
    get_display   : callable → presentation dict (cmap, inv_cmap, clip, fix,
                    fix_iv, boundaries). Drives colour/clip/overlays only.
    """

    def __init__(self, view, panel, *,
                 get_source: Callable[[], object],
                 get_display: Callable[[], dict],
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.view = view
        self.panel = panel
        self._get_source = get_source
        self._get_display = get_display

        self.pipeline = Pipeline()       # the dynamic per-WINDOW filter nodes
        self._precrop_nodes: list = []   # PRECROP nodes (e.g. Water Mute) — full-array
        self._precrop_sig: tuple = ()
        self._data_version = 0           # bumped when source / align / pre-crop changes
        self._has_source = False

        # Cached PREPARED base = full array with (1) static delay alignment and
        # (2) PRE-CROP nodes (water mute) applied. Rebuilt only when the source,
        # the align flag, or the pre-crop node params change — NOT on pan/zoom
        # and NOT on per-window filter edits.
        self._pc_array = None
        self._pc_t0 = 0.0
        self._pc_src = None
        self._pc_key = None              # (align_flag, precrop_signature)

        # Display-param snapshot: re-read from Qt widgets only when a display
        # setting actually changed (not on every pan/zoom frame).
        self._disp_cache: Optional[dict] = None
        self._disp_dirty: bool = True

        # Halo cache: max_time_halo + max_trace_halo only change when the source
        # object or the pipeline node set changes — not on pan/zoom.
        self._halo_cache: Optional[tuple] = None  # (data_ver, pipeline_ver, th, trh)
        self._pipeline_version: int = 0

        # Cached wiggle geometry key. When window bounds, vmax, and va_fill are
        # unchanged, _build_wiggle + arrayToQPath are skipped entirely — colormap,
        # FIX-mark, and show_raster changes pay zero geometry cost.
        self._wiggle_state_key = None

        # A/B Compare wiper state. ``_ab_frac`` is the user's chosen split as a
        # fraction of the visible width (persists across pan/zoom — a screen-
        # space wipe). ``_ab_cache`` holds the last raw + processed visible
        # arrays + their extent so dragging the wiper recomposes the split
        # WITHOUT re-running the DSP pipeline.
        self._ab_frac: float = 0.5
        self._ab_cache: Optional[dict] = None

        # Off-GUI-thread pipeline execution (see preview_worker.PipelineWorker).
        # Only one worker runs at a time; further triggers received while busy
        # are coalesced here and replayed once it completes (_dispatch_pending).
        self._worker: Optional[PipelineWorker] = None
        self._refresh_ctx: Optional[dict] = None
        self._pending: Optional[tuple] = None   # (fit, overlays) to re-run
        self._pending_sync: bool = False         # a pipeline_changed was deferred

        view.enable_preview(True)
        view.view_range_changed.connect(self._on_view_changed)
        panel.pipeline_changed.connect(self._on_pipeline_changed)
        view.ab_split_changed.connect(self._on_ab_drag)

    # ── External triggers ───────────────────────────────────────────────────

    def set_source(self, obj: object) -> None:
        """New profile/chain selected (or its traces just finished loading)."""
        self._ensure_idle()
        self._data_version += 1
        self._disp_dirty = True
        self.pipeline.clear_cache()
        self._sync_nodes()
        self._pc_array = None
        self._pc_src = None
        self._has_source = obj is not None and getattr(obj, "data", None) is not None
        if self._has_source:
            self._refresh(fit=True, overlays=True)   # initial full-view render

    def display_changed(self) -> None:
        """Presentation (cmap / clip / FIX / boundaries) changed — recolour."""
        self._disp_dirty = True
        if self._has_source:
            self._refresh(fit=False, overlays=True)

    def alignment_changed(self) -> None:
        """Static delay-alignment toggled — rebuild the prepared base, invalidate
        caches, and refit (the time origin and matrix height change)."""
        self._ensure_idle()
        self._pc_array = None
        self._data_version += 1          # window cache keys include data_version
        self._disp_dirty = True
        if self._has_source:
            self._refresh(fit=True, overlays=True)

    def fit(self) -> None:
        """Re-fit the whole section to the panel (the 'Fit view' action)."""
        if self._has_source:
            self._refresh(fit=True, overlays=True)

    # ── Internal trigger slots ──────────────────────────────────────────────

    def _sync_nodes(self) -> bool:
        """Split panel nodes into PRE-CROP (full-array) vs per-window filters.
        Returns True if the pre-crop set changed (→ prepared base must rebuild).

        Reads ``active_nodes()`` (enabled-only), so a MUTED node is bypassed in
        the live preview exactly as it is in the export."""
        nodes = self.panel.active_nodes()
        precrop = [n for n in nodes if getattr(n, "PRECROP", False)]
        window  = [n for n in nodes if not getattr(n, "PRECROP", False)]
        self.pipeline.set_nodes(window)
        sig = tuple(n.signature() for n in precrop)
        self._precrop_nodes = precrop
        if sig != self._precrop_sig:
            self._precrop_sig = sig
            return True
        return False

    def _on_pipeline_changed(self) -> None:
        # _sync_nodes() mutates self.pipeline.nodes, which the worker thread may
        # be iterating right now — defer the whole handler (not just _refresh)
        # until it's done, rather than racing it.
        if self._worker is not None and self._worker.isRunning():
            self._pending_sync = True
            return
        if self._sync_nodes():           # pre-crop (e.g. Water Mute) changed
            self._pc_array = None         # rebuild the prepared base
            self._data_version += 1       # invalidate per-window cache
        self._pipeline_version += 1      # invalidate halo cache (node set changed)
        if self._has_source:
            self._refresh(fit=False, overlays=False)

    def _on_view_changed(self) -> None:
        if self._has_source:
            self._refresh(fit=False, overlays=False)

    # ── Prepared base (alignment + PRE-CROP nodes on the FULL array, cached) ──

    def _prepared_base(self, obj: object):
        """Return (base_array, t0_ms): the FULL array with static delay alignment
        AND any PRE-CROP nodes (water mute) applied — cached so pan/zoom and
        per-window filter edits never recompute it.

        Water-column mute MUST run here (not per ViewBox window): its seabed pick
        needs the whole trace, and the window's row 0 is not true t=0. Running it
        full-array first guarantees a flawlessly muted preview."""
        align = bool(self._get_display().get("align", False))
        key = (align, self._precrop_sig)
        if (self._pc_array is not None and self._pc_src is obj
                and self._pc_key == key):
            return self._pc_array, self._pc_t0

        base = obj.data
        t0 = float(getattr(obj, "delay_ms", 0.0))
        full_ctx = DSPContext.from_source(obj)
        # 1) Static geometry: delay alignment (taller array, t0 = min_delay).
        if align and getattr(obj, "delays", None) is not None:
            from sbp_studio.core import apply_delay_alignment
            base = apply_delay_alignment(
                base, obj.delays, obj.min_delay, obj.dt_us, fill_value=0.0)
            t0 = float(getattr(obj, "min_delay", 0.0))
        # 2) PRE-CROP nodes (water mute) on the full aligned array.
        for node in self._precrop_nodes:
            base = node.apply(base, full_ctx)

        self._pc_array, self._pc_t0, self._pc_src, self._pc_key = base, t0, obj, key
        return base, t0

    # ── The live loop ───────────────────────────────────────────────────────

    def _refresh(self, *, fit: bool, overlays: bool = False) -> None:
        # Only one pipeline worker runs at a time (see preview_worker). A
        # trigger that arrives while one is in flight is coalesced into
        # ``_pending`` and replayed once it finishes — never overlapped.
        if self._worker is not None and self._worker.isRunning():
            self._pending = (fit, overlays)
            return
        obj = self._get_source()
        if obj is None or getattr(obj, "data", None) is None:
            return
        # Prepared base = full array with alignment + pre-crop (water mute). The
        # remaining per-window filters run on the cropped ViewBox of this base;
        # the export path runs the full pipeline in list order separately.
        data, t0_full = self._prepared_base(obj)
        dist_km = obj.dist_km
        n_traces = data.shape[1]
        dt_us = int(obj.dt_us)
        dt_ms = dt_us / 1000.0

        # Re-read display params only when a setting actually changed (not on
        # every pan/zoom frame — avoids ~10 Qt widget reads per settle).
        if self._disp_dirty or self._disp_cache is None:
            self._disp_cache = self._get_display()
            self._disp_dirty = False
        disp = self._disp_cache

        # Halo sizes only change when the source or pipeline node set changes;
        # cache them so pan/zoom skips DSPContext construction + node iteration.
        halo_key = (self._data_version, self._pipeline_version)
        if self._halo_cache is None or self._halo_cache[0] != halo_key:
            ctx = DSPContext.from_source(obj)
            self._halo_cache = (halo_key,
                                self.pipeline.max_time_halo(ctx),
                                self.pipeline.max_trace_halo(ctx))
        _, time_halo, trace_halo = self._halo_cache

        # ── Determine the visible window ────────────────────────────────────
        if fit:
            x_range = (float(dist_km[0]), float(dist_km[-1]))
            y_range = (t0_full, t0_full + data.shape[0] * dt_ms)
        else:
            x_range, y_range = self.view.current_view_range()

        # ViewBox-limited extraction with row+column decimation (preview only).
        # Live horizontal detail: the "Pixels / trace" control raises the column
        # cap so zooming in stays sharp (default 20 → the historical 4000-col cap;
        # higher keeps more native traces, lower decimates more aggressively).
        ppt = float(disp.get("px_per_trace", 20.0)) or 20.0
        eff_max_cols = int(np.clip(ppt / 20.0 * MAX_PREVIEW_COLS, 1000, 16000))
        # A NEEDS_FULL_RES node (e.g. the F-K dip filter) must see the true,
        # un-decimated trace spacing of the viewport — column decimation would
        # corrupt its wavenumber axis. When one is active, extract at full
        # resolution up to a high safety cap (exact at any practical working
        # zoom; only a viewport wider/deeper than the cap decimates gently, and
        # there the result is handed unchanged to the decimation/render layer).
        if any(getattr(n, "NEEDS_FULL_RES", False) for n in self.pipeline.nodes):
            max_rows_use, max_cols_use = FK_MAX_ROWS, FK_MAX_COLS
        else:
            max_rows_use, max_cols_use = MAX_PREVIEW_ROWS, eff_max_cols
        win = extract_visible_window(
            data, dist_km, t0_ms=t0_full, dt_us=dt_us,
            x_range=x_range, y_range=y_range,
            time_halo=time_halo, trace_halo=trace_halo,
            data_version=self._data_version,
            max_rows=max_rows_use, max_cols=max_cols_use)

        # ── Run the (memoized) pipeline on the small bbox at EFFECTIVE dt ────
        # Row decimation raised the sample interval; the nodes must see it so
        # AGC (and future frequency filters) use the correct sampling rate.
        # This is the heavy step (F-K's 2-D FFT, Deconvolution, …) — it runs on
        # a worker thread (see preview_worker.PipelineWorker) so the GUI thread
        # never blocks while it computes; _finish_refresh resumes on the result.
        sub_ctx = DSPContext(dt_us=win.effective_dt_us, ns=win.sub.shape[0],
                             n_traces=win.sub.shape[1])
        self._refresh_ctx = dict(obj=obj, win=win, disp=disp, dist_km=dist_km,
                                 n_traces=n_traces, t0_full=t0_full, dt_ms=dt_ms,
                                 fit=fit, overlays=overlays)
        worker = PipelineWorker(self.pipeline, win.sub, sub_ctx, win.token, parent=self)
        worker.succeeded.connect(self._on_pipeline_result)
        worker.failed.connect(self._on_pipeline_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_pipeline_result(self, token: object, processed: np.ndarray) -> None:
        """GUI-thread continuation of _refresh once the worker's pipeline.process
        finishes — runs the cheap presentation logic and touches Qt widgets."""
        ctx = self._refresh_ctx
        self._worker = None
        self._finish_refresh(processed, **ctx)
        self._dispatch_pending()

    def _on_pipeline_failed(self, token: object, message: str) -> None:
        self._worker = None
        _LOG.error("DSP preview pipeline failed: %s", message)
        self._dispatch_pending()

    def _dispatch_pending(self) -> None:
        """Replay the most recent trigger that arrived while the worker was busy.
        A deferred ``_on_pipeline_changed`` (node list mutation) takes priority
        over a plain refresh, since it must run before any new window is built."""
        if self._pending_sync:
            self._pending_sync = False
            self._pending = None
            self._on_pipeline_changed()
        elif self._pending is not None:
            fit, overlays = self._pending
            self._pending = None
            self._refresh(fit=fit, overlays=overlays)

    def _ensure_idle(self) -> None:
        """Block until any in-flight preview worker finishes. Used by callers
        that mutate pipeline/cache state directly (source switch, alignment
        toggle) — rare events where a brief wait is preferable to racing the
        worker thread over shared state. The common pan/zoom-while-computing
        case never calls this — it stays fully non-blocking via _pending."""
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

    def _finish_refresh(self, processed: np.ndarray, *, obj: object, win,
                        disp: dict, dist_km: np.ndarray, n_traces: int,
                        t0_full: float, dt_ms: float, fit: bool,
                        overlays: bool) -> None:
        visible = win.crop_visible(processed)   # drop the halo rows+cols (decimated)

        if visible.size == 0:
            return

        # ── Presentation: clip percentile (needed below for a fair A/B vmax) ──
        from sbp_studio.core.constants import CMAPS, DEFAULT_CLIP_PCT
        clip = float(disp.get("clip", DEFAULT_CLIP_PCT))

        # ── A/B Compare: splice RAW (left) | PROCESSED (right) ───────────────
        # The raw window is the SAME prepared base (alignment + pre-crop mute)
        # WITHOUT the per-window filter pipeline — so both halves share identical
        # geometry/decimation and composite cleanly into one image. The split
        # follows the user's draggable wiper (``_ab_frac``, a fraction of the
        # visible width that persists across pan/zoom). The raw + processed
        # visible arrays are cached so dragging the wiper recomposes without
        # re-running the pipeline (see _on_ab_drag).
        ab_on = bool(disp.get("ab_compare"))
        ab_split_frac = self._ab_frac
        ab_vmax: Optional[float] = None
        if ab_on:
            raw_visible = np.ascontiguousarray(win.crop_visible(win.sub),
                                               dtype=np.float32)
            # Explicit COPY (not a view): ``visible`` is a row/col slice of the
            # pipeline's cached prefix output (self.pipeline._cache). Caching a
            # view here would tie the wiper's lifetime to that cache entry —
            # the next pipeline.process() call can prune/overwrite it from
            # under us (and, with the preview worker, from another thread).
            proc_visible = np.array(visible, dtype=np.float32, copy=True)
            n_ab = proc_visible.shape[1]
            split = int(round(min(max(self._ab_frac, 0.0), 1.0) * n_ab))
            ab_split_frac = (split / n_ab) if n_ab else 0.5
            # Estimate vmax from EACH full half independently (not the spliced
            # composite, whose raw/processed proportion tracks the wiper
            # position) and take the larger: a fair, split-position-independent
            # ceiling that never washes out or saturates whichever side ends up
            # the minority of on-screen pixels. Locked into _ab_cache below so
            # dragging the wiper never flickers the brightness.
            ab_vmax = max(self._estimate_vmax(raw_visible, clip),
                          self._estimate_vmax(proc_visible, clip))
            visible = self._compose_ab(raw_visible, proc_visible, split)

        vmax = ab_vmax if ab_vmax is not None else self._estimate_vmax(visible, clip)
        # Amplitude range: diverging maps [−vmax, vmax] (signed, colormap centred
        # on zero); sequential maps [0, vmax] (|amp|, historical default).
        vmin = -vmax if disp.get("amp_range") == "diverging" else 0.0

        cmap_name = CMAPS.get(disp.get("cmap", "Viridis"), "viridis")
        if disp.get("inv_cmap"):
            cmap_name += "_r"

        c1_idx = min(win.c_vis1 - 1, n_traces - 1)
        dist0 = float(dist_km[win.c_vis0])
        dist1 = float(dist_km[c1_idx])
        t_top = t0_full + win.s0 * dt_ms
        t_bot = t0_full + win.s1 * dt_ms

        # The decimated visible band is a row-slice of the pipeline's contiguous
        # output → already C-contiguous float32, so this is a no-op (no full
        # copy). Phase 2 introduces no NaNs (align is off), so we skip the
        # expensive nan_to_num scan entirely.
        arr = np.ascontiguousarray(visible, dtype=np.float32)
        self.view.set_colormap(cmap_name, vmax, vmin=vmin)
        # c0/c1: the FULL-RESOLUTION trace-index bounds this window's arr
        # actually represents — lets the view keep interpretation picks
        # glued to the image's own linear pixel-grid approximation across
        # pan/zoom (see SeismicView._pick_x_km's docstring).
        self.view.show_preview(arr, dist0, dist1, t_top, t_bot,
                               vmax=vmax, vmin=vmin, fit=fit,
                               c0=win.c_vis0, c1=c1_idx + 1)

        # Emit after show_preview so the ViewBox is already fitted (fit=True):
        # the autoRange() inside show_preview fires sigRangeChanged, but
        # _emit_visible_traces bails early while _dist_km is None.  Setting
        # _dist_km here guarantees the first emission uses the correct,
        # already-fitted range — fixing the map regression on initial load.
        self.view.set_distance_axis(dist_km)

        # A/B Compare draws a labeled, draggable raw|processed wiper over the
        # composite raster. The wiggle/VA overlay is always suppressed in this
        # mode (density raster only) — the wiper compares pixel intensity, not
        # waveform shape. The raw + processed visible arrays and the extent
        # are cached so dragging the wiper recomposes the split without
        # re-running the pipeline.
        if ab_on:
            self._ab_cache = dict(
                raw=raw_visible, proc=proc_visible,
                dist0=dist0, dist1=dist1, t_top=t_top, t_bot=t_bot,
                vmax=vmax, vmin=vmin, cmap=cmap_name,
                c0=win.c_vis0, c1=c1_idx + 1)
            split_km = dist0 + ab_split_frac * (dist1 - dist0)
            self.view.update_ab(True, split_km, t_top, t_bot, x0=dist0, x1=dist1)
            self.view.disable_wiggle()
            self._wiggle_state_key = None
            if fit or overlays:
                self._draw_overlays(obj, disp)
            return
        self._ab_cache = None
        self.view.update_ab(False)

        # ── Section style: density raster (default) vs wiggle / variable-area ─
        # The raster is ALWAYS pushed above (kept ready underneath), so the
        # wiggle⇄raster transition is flicker-free. When 'wiggle', a budget-limited
        # set of vector traces (+ optional VA fill) is drawn OVER the raster at ANY
        # zoom: the visible traces are decimated to WIGGLE_BUDGET so it stays fluid
        # and legible (zoom in → fewer real traces per drawn wiggle → more detail).
        # Selecting Wiggle is therefore always visible, never a silent no-op. The
        # raster underlay can be hidden ('Wiggle Only') via show_raster.
        if disp.get("style", "density") == "wiggle":
            budget = int(self.view.WIGGLE_BUDGET)
            n_vis = visible.shape[1]
            # The raster (img.setRect) stretches its n_vis columns UNIFORMLY
            # across [dist0, dist1] — it has no notion of dist_km's true,
            # non-uniform per-trace spacing. x_centers MUST use that same
            # uniform mapping (not dist_km[col_idx] directly), or the wiggle/VA
            # drift away from the raster pixels wherever spacing isn't constant
            # within the window (worse toward whichever edge is denser).
            x_centers = dist0 + (np.arange(n_vis, dtype=float) + 0.5) / n_vis * (dist1 - dist0)
            amps = visible
            if n_vis > budget:                       # decimate to the budget
                sel = np.linspace(0, n_vis - 1, budget).astype(int)
                amps = np.ascontiguousarray(visible[:, sel])
                x_centers = x_centers[sel]

            show_raster      = bool(disp.get("show_raster", True))
            va_fill          = bool(disp.get("va_fill", True))
            show_wiggle_line = bool(disp.get("show_wiggle_line", True))

            # Auto-suppress VA when the drawn trace count exceeds the threshold:
            # lobes are too thin to read and arrayToQPath on millions of polygon
            # points would block the GUI thread for tens of ms per settle.
            effective_va = va_fill and (amps.shape[1] <= VA_TRACE_THRESHOLD)

            # Geometry key: window bounds + vmax + effective_va determine the
            # shape of the wiggle path. Colormap, FIX-mark, show_raster, and
            # show_wiggle_line changes pay zero geometry cost (visibility only).
            wiggle_key = (win.token, vmax, effective_va)
            if wiggle_key == self._wiggle_state_key:
                self.view.set_raster_visible(show_raster)
                self.view.set_wiggle_line_visible(show_wiggle_line)
            else:
                # Sanitize NaN/Inf before geometry: NaN survives clip and
                # fragments traces; Inf breaks deflection math. The raster (arr)
                # was already submitted above, so mutating amps is safe.
                np.nan_to_num(amps, copy=False, posinf=0.0, neginf=0.0)
                res = self._build_wiggle(amps, x_centers, t_top, t_bot, vmax,
                                         va_max_rows=VA_MAX_ROWS,
                                         build_va=effective_va)
                if res is not None:
                    xs, ys, xs_va, ys_va = res
                    self.view.update_wiggle(
                        xs, ys, xs_va, ys_va,
                        show_raster=show_raster,
                        va_fill=effective_va,
                        show_line=show_wiggle_line)
                    self._wiggle_state_key = wiggle_key
                else:
                    self.view.disable_wiggle()
                    self._wiggle_state_key = None
        else:
            self.view.disable_wiggle()
            self._wiggle_state_key = None

        # ── Overlays (FIX marks / chain boundaries) ─────────────────────────
        # Redrawn on fit and on presentation changes (e.g. toggling FIX), but
        # left untouched during pure pan/zoom (the InfiniteLines are at fixed
        # positions, so they persist correctly without a rebuild).
        if fit or overlays:
            self._draw_overlays(obj, disp)

    # ── A/B Compare wiper ─────────────────────────────────────────────────────

    @staticmethod
    def _compose_ab(raw: np.ndarray, proc: np.ndarray, split: int) -> np.ndarray:
        """Splice RAW (columns left of ``split``) | PROCESSED (the rest) into one
        contiguous float32 image. Both arrays share the window's geometry, so
        the result is spatially continuous — a clean vertical wipe."""
        n = proc.shape[1]
        split = max(0, min(int(split), n))
        combined = np.array(proc, dtype=np.float32, copy=True)
        if split > 0:
            combined[:, :split] = raw[:, :min(split, raw.shape[1])]
        return combined

    def _on_ab_drag(self, split_km: float) -> None:
        """User dragged the A/B wiper → recompose the split at the new position
        from the CACHED raw/processed arrays (no DSP rerun) and swap the image.
        Also remembers the fraction so pan/zoom keeps the same wipe position."""
        c = self._ab_cache
        if c is None:
            return
        span = c["dist1"] - c["dist0"]
        frac = (split_km - c["dist0"]) / span if span else 0.5
        frac = min(max(frac, 0.0), 1.0)
        self._ab_frac = frac
        n = c["proc"].shape[1]
        combined = self._compose_ab(c["raw"], c["proc"], int(round(frac * n)))
        self.view.set_colormap(c["cmap"], c["vmax"], vmin=c["vmin"])
        self.view.show_preview(combined, c["dist0"], c["dist1"],
                               c["t_top"], c["t_bot"],
                               vmax=c["vmax"], vmin=c["vmin"], fit=False,
                               c0=c["c0"], c1=c["c1"])

    # ── On-demand advanced spectrum inputs (heavy Welch runs in a worker) ────

    def analysis_inputs(self, scope: str):
        """Snapshot the data + node config for an advanced spectrum analysis.

        Cheap (GUI thread): returns the prepared-base slice for the scope plus a
        thread-safe node snapshot; the caller's worker applies the dynamic nodes
        (full-res) and runs ``core.compute_spectrum``. ``scope`` is "viewbox"
        (current window) or "full" (whole profile). Returns None if no source.
        """
        obj = self._get_source()
        if obj is None or getattr(obj, "data", None) is None:
            return None
        base, t0 = self._prepared_base(obj)        # full array: aligned + pre-crop mute
        dt_us = int(obj.dt_us)
        node_cfgs = [(n.KEY, dict(n.params)) for n in self.pipeline.nodes]

        if scope == "viewbox":
            (xv0, xv1), (yv0, yv1) = self.view.current_view_range()
            # Clamp the (possibly padded / aspect-expanded) view range to the
            # data extent so the crop is exact and never silently falls back to
            # the full profile. The seismic t-axis spans [t0, t0 + ns·dt_ms].
            dt_ms = dt_us / 1000.0
            dmin, dmax = float(obj.dist_km[0]), float(obj.dist_km[-1])
            tmin, tmax = t0, t0 + base.shape[0] * dt_ms
            x0 = max(dmin, min(xv0, xv1)); x1 = min(dmax, max(xv0, xv1))
            y0 = max(tmin, min(yv0, yv1)); y1 = min(tmax, max(yv0, yv1))
            # Full-RESOLUTION window (no row/col caps) for an accurate spectrum.
            win = extract_visible_window(
                base, obj.dist_km, t0_ms=t0, dt_us=dt_us,
                x_range=(x0, x1), y_range=(y0, y1),
                data_version=self._data_version)
            arr = np.ascontiguousarray(win.sub)
            dist = obj.dist_km[win.c0:win.c1]
            x0, x1 = float(dist[0]), float(dist[-1])
        else:  # full profile
            arr = base
            dist = obj.dist_km
            x0, x1 = float(dist[0]), float(dist[-1])

        boundaries = tuple(b for b in (getattr(obj, "boundaries_km", ()) or ())
                           if x0 <= float(b) <= x1)
        return dict(arr=arr, node_cfgs=node_cfgs, dt_us=dt_us,
                    dist_km=dist, boundaries=boundaries)

    @staticmethod
    def _build_wiggle(visible: np.ndarray, x_centers: np.ndarray,
                      t_top: float, t_bot: float, vmax: float,
                      gain: float = WIGGLE_GAIN, *,
                      va_max_rows: int = WIGGLE_MAX_ROWS,
                      build_va: bool = True):
        """Build NaN-separated geometry for ALL visible traces in one shot:
        ``(xs, ys, xs_va, ys_va)``.

        ``xs/ys``       — the wiggle polyline (one ``setData`` → one draw call).
        ``xs_va/ys_va`` — a per-trace closed polygon hugging the centre line where
                          amplitude ≤ 0 and bulging along the positive envelope,
                          so a single filled path renders the variable-area lobes.
                          Sign changes between adjacent rows get an exact
                          interpolated zero-crossing vertex (rather than a
                          straight line between clamped samples), so the fill
                          boundary lines up with the unclamped wiggle line —
                          no "staircase" artifact at low row counts. Each
                          trace's polygon explicitly closes back to its own
                          (x_centers, t_va[0]) start vertex in data
                          coordinates (not relying on Qt's implicit close-on-
                          fill), so the baseline can never drift from the
                          wiggle line under non-uniform (asymmetric) zoom.

        Each trace deflects horizontally by ``(amp/vmax)·spacing·gain`` about its
        km position; traces are joined by NaN so ``connect='finite'`` breaks both
        the line and the fill between them. Rows are capped at ``WIGGLE_MAX_ROWS``
        for fluidity. Returns ``None`` if degenerate."""
        rows, cols = visible.shape
        n = int(min(cols, x_centers.size))
        if n < 1 or rows < 2:
            return None
        # float32 throughout: halves Qt upload bandwidth; km/ms coordinates need
        # only ~7 significant digits (sub-cm / sub-µs resolution at display scales).
        visible = np.ascontiguousarray(visible[:, :n], dtype=np.float32)
        x_centers = np.asarray(x_centers[:n], dtype=np.float32)
        if rows > WIGGLE_MAX_ROWS:                       # decimate rows for fluidity
            step = int(np.ceil(rows / WIGGLE_MAX_ROWS))
            visible = visible[::step, :]
            rows = visible.shape[0]
        vmax = float(vmax) or 1.0
        t = np.linspace(t_top, t_bot, rows, dtype=np.float32)
        # Median absolute inter-trace gap: robust against a stationary vessel
        # (all x_centers identical → global-average collapses to zero → wiggle
        # disappears). Falls back to 1.0 when all traces share a position.
        if n > 1:
            med = float(np.median(np.abs(np.diff(x_centers))))
            spacing = med if med > 0.0 else 1.0
        else:
            spacing = 1.0
        deflect = np.float32(spacing * float(gain))  # keep float32; scalar upcast avoids it
        xc   = x_centers[None, :]                    # (1, n) broadcast anchor
        norm = np.clip(visible / vmax, -1.0, 1.0)   # (rows, n) float32

        # ── Wiggle line ──
        # Build (n, rows+1) so C-order .ravel() interleaves per-trace without a
        # Fortran-reshape copy (the old _flat() needed an F-order reshape → copy).
        xs_line_T = np.empty((n, rows + 1), dtype=np.float32)
        xs_line_T[:, :rows] = (xc + norm * deflect).T   # (n, rows) from F-view
        xs_line_T[:, rows]  = np.nan
        xs_flat = xs_line_T.ravel()                      # zero-copy C-order view

        # Y is identical for every trace: tile the 1D vector, skip the 2D repeat.
        t_nan = np.empty(rows + 1, dtype=np.float32)
        t_nan[:rows] = t
        t_nan[rows]  = np.nan
        ys_flat = np.tile(t_nan, n)

        if not build_va:
            return xs_flat, ys_flat, None, None

        # ── Variable-area polygon: centre@top → positive envelope → centre@bot ──
        # VA uses its own independent row decimation (va_max_rows) so arrayToQPath
        # stays fast on dense/deep fits where WIGGLE_MAX_ROWS is much larger.
        va_step = max(1, -(-rows // va_max_rows))
        if va_step > 1:
            norm_va = norm[::va_step, :]
            rows_va = norm_va.shape[0]
            t_va    = np.linspace(t_top, t_bot, rows_va, dtype=np.float32)
        else:
            norm_va, rows_va, t_va = norm, rows, t

        env_va = xc + np.maximum(norm_va, 0.0) * deflect   # (rows_va, n)

        # ── Exact zero-crossing interpolation between adjacent VA rows ──
        # Naively connecting two CLAMPED envelope points with a straight line
        # is wrong whenever the raw (unclamped) signal changes sign between
        # them: e.g. norm[i]=-0.5, norm[i+1]=+0.3 clamp to (0, 0.3), and a
        # direct line from 0 to 0.3 starts rising immediately at t[i] — but
        # the true signal is still negative for the first part of that gap,
        # so max(signal, 0) should stay flat at the centre line until the
        # EXACT time the raw amplitude crosses zero, then rise. That flat
        # plateau is exactly what produced the "staircase" artifact. Fix:
        # insert one extra vertex per inter-sample gap, placed at the true
        # interpolated crossing (centre line, fractional time) whenever the
        # two samples have opposite sign; where there is no sign change the
        # inserted vertex is a harmless duplicate of the left envelope point
        # (already colinear, so it changes nothing).
        if rows_va > 1:
            left, right = norm_va[:-1, :], norm_va[1:, :]        # (rows_va-1, n)
            t_left  = t_va[:-1, None]
            t_right = t_va[1:, None]
            crosses = (left * right) < 0.0                       # true sign change only
            denom = np.where(crosses, left - right, 1.0)          # dodge 0/0 where unused
            frac = np.clip(left / denom, 0.0, 1.0)
            t_cross = t_left + frac * (t_right - t_left)
            mid_x = np.where(crosses, xc, env_va[:-1, :])
            mid_t = np.where(crosses, t_cross,
                             np.broadcast_to(t_left, t_cross.shape))

            rows_up = 2 * rows_va - 1
            upper_x = np.empty((rows_up, n), dtype=np.float32)
            upper_t = np.empty((rows_up, n), dtype=np.float32)
            upper_x[0::2, :] = env_va
            upper_x[1::2, :] = mid_x
            upper_t[0::2, :] = np.broadcast_to(t_va[:, None], env_va.shape)
            upper_t[1::2, :] = mid_t
        else:
            rows_up = rows_va
            upper_x = env_va
            upper_t = np.broadcast_to(t_va[:, None], env_va.shape)

        # ── Explicit baseline closure ──
        # Qt's fillPath() implicitly closes any open subpath with a straight
        # line back to its start before filling, so leaving the polygon "open"
        # (top anchor → envelope → bottom anchor → NaN) renders correctly in
        # isolation. But that implicit edge is a Qt fill-time behaviour, not
        # data — anything that later inspects/strokes this path (or a Qt
        # version/backend with different implicit-close semantics) would not
        # see it. Writing the return-to-(x_centers, t_va[0]) vertex explicitly
        # makes every trace's polygon a literal closed loop anchored to its
        # own x_centers value in pure data coordinates, independent of any
        # transform applied later — so asymmetric (non-uniform X/Y) zoom can
        # never pull the fill's baseline away from the wiggle line's own
        # zero-crossing, which is built from the exact same x_centers array.
        xs_va_T = np.empty((n, rows_up + 4), dtype=np.float32)
        xs_va_T[:, 0]              = x_centers              # centre-line anchor at top
        xs_va_T[:, 1:rows_up + 1]  = upper_x.T             # envelope + exact crossings
        xs_va_T[:, rows_up + 1]    = x_centers             # centre-line anchor at bottom
        xs_va_T[:, rows_up + 2]    = x_centers             # explicit close: back to top anchor
        xs_va_T[:, rows_up + 3]    = np.nan                # trace separator
        xs_va_flat = xs_va_T.ravel()                       # zero-copy C-order view

        # Unlike xs, the time coordinate is no longer identical across traces
        # — each trace's inserted crossings land at its own fractional time —
        # so ys_va is built per-trace (ys_va_T) rather than tiled from one
        # shared vector.
        ys_va_T = np.empty((n, rows_up + 4), dtype=np.float32)
        ys_va_T[:, 0]              = t_va[0]
        ys_va_T[:, 1:rows_up + 1]  = upper_t.T
        ys_va_T[:, rows_up + 1]    = t_va[-1]
        ys_va_T[:, rows_up + 2]    = t_va[0]                # same t as the start point → exact close
        ys_va_T[:, rows_up + 3]    = np.nan
        ys_va_flat = ys_va_T.ravel()

        return xs_flat, ys_flat, xs_va_flat, ys_va_flat

    @staticmethod
    def _estimate_vmax(visible: np.ndarray, clip: float) -> float:
        """Clip percentile for the display level — estimated on a bounded
        subsample so the cost stays flat regardless of preview size (the value
        is a display ceiling, exact precision is unnecessary)."""
        samp = visible
        n = samp.size
        cap = 200_000
        if n > cap:
            step = int((n / cap) ** 0.5) + 1
            samp = samp[::step, ::step]
        samp = samp[np.isfinite(samp)]
        if samp.size == 0:
            return 1.0
        return float(np.percentile(np.abs(samp), clip)) or 1.0

    def _draw_overlays(self, obj: object, disp: dict) -> None:
        boundaries = tuple(getattr(obj, "boundaries_km", ()) or ())
        fixes = ()
        if disp.get("fix"):
            from sbp_studio.core import compute_fix_positions
            fixes = [(f[0], f[1], f[2]) for f in compute_fix_positions(
                obj.timestamps, obj.dist_km, obj.lons, obj.lats,
                int(disp.get("fix_iv", 5)))]
        self.view.set_overlays(boundaries=boundaries, fixes=fixes)
