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
from PyQt6.QtCore import QObject, QTimer

import os
from time import perf_counter

from sbp_studio.core.constants import DEFAULT_CLIP_PCT
from sbp_studio.core.logger import get_logger
from sbp_studio.core.tasks import CancelToken

from .nodes import DSPContext
from .pipeline import Pipeline, ProcessedBand, extract_visible_window, reslice_band
from .preview_worker import CANCELLED_MESSAGE, BasePrepWorker, PipelineWorker

_LOG = get_logger("dsp.preview")

# Lightweight perf telemetry, off by default (zero overhead — a single bool
# check per hot call). Run with the env var ``TOPAS_PERF=1`` to emit per-stage
# timing for _prepared_base, the _refresh→result round-trip (here) and
# MapView.set_visible_range (seismic side) at INFO so they show on the console
# while dragging a real cadena. See the Phase-2 telemetry note.
_PERF = os.environ.get("TOPAS_PERF") == "1"

# Preview raster column cap. Rows are NEVER decimated/cropped for the live
# preview — see _refresh's full_depth note: a time-series filter (AGC,
# Deconvolution, Envelope) must see every sample of the FULL trace to compute
# the same result it would at full zoom, so only the COLUMN (trace) extent is
# capped/decimated here. Zoomed in deeply, the column stride falls to 1 →
# full-resolution, exact processing. Capped at 4000: a 4K monitor is only
# 3840 px wide, so processing more traces than that is pure overdraw the
# screen can't even show.
MAX_PREVIEW_COLS = 4000

# Interactive LOD (level-of-detail): while the user is actively DRAGGING a
# DSP node's parameter slider (see PipelinePanel.interactionStarted/Ended),
# the column extent handed to the worker is slashed by this fraction —
# masking a heavy node's (Deconvolution, F-K) per-call compute cost during
# the drag, when many intermediate frames are inherently throwaway anyway.
# Deliberately HORIZONTAL-only: rows stay full_depth=True always (the very
# physics fix this must never regress — see the column-cap note above), so
# this never trades away the math correctness, only on-screen trace density
# for the duration of the drag. The slider's OWN debounce (PipelinePanel.
# DEBOUNCE_MS) already limits how often a frame is even requested; LOD
# additionally shrinks the COST of each of those frames specifically while
# the mouse button is still down. interactionEnded triggers one final,
# full-resolution _refresh() once the drag stops.
LOD_SCALE = 0.25       # keep ~25% of the normal column cap while dragging
LOD_MIN_COLS = 250     # floor so a narrow viewport never degenerates further

# Overview LOD (the zoom-out fix). The exact full_depth path processes EVERY
# native sample of the visible Y-span — correct when zoomed in, but ruinous at
# zoom-out (15-30k rows through DSP for a ~1.5k-px screen). When visible rows
# exceed OVERVIEW_TRIGGER_FACTOR × adaptive_target, rows are RMS-pooled down to
# adaptive_target *before* the pipeline runs (see pipeline._pool_rows_rms).
#
# The target is ADAPTIVE: OVERVIEW_OVERSAMPLE × the ViewBox pixel height, so the
# overview carries ~4 samples per display pixel — much gentler compression than
# a fixed cap, and the larger stride means the RMS estimate is more accurate
# (honest energy ≈ σ, vs max-abs's ≈ 2.3σ). Never drops below OVERVIEW_MIN_ROWS
# (safety floor for tiny/not-yet-laid-out panels).
#
# Gated on the VISIBLE span (not the band depth) so a thin Y-zoom keeps the
# exact Strategy-B window; only genuine oversampling pools.
OVERVIEW_MIN_ROWS      = 1000  # floor: never pool to fewer than this many rows
OVERVIEW_OVERSAMPLE    = 4     # target = OVERSAMPLE × ViewBox pixel height
OVERVIEW_TRIGGER_FACTOR = 2.0  # activate when visible_rows > TRIGGER × target

# Deferred global-levels recompute after a slider release. interactionEnded
# shows the final full-res raster immediately reusing the LAST locked ceiling
# (imperceptible: the levels are zoom-independent and barely move per tweak),
# then this long settle delay elapses before the one real 5-block levels pass
# runs — so rapid edit→release→edit→release cycles never pay it per release,
# only once the user truly stops. Longer than the param debounce so a quick
# follow-up edit pre-empts it.
LEVELS_SETTLE_MS = 400

# ── Vertical windowing (Strategy A + B) ──────────────────────────────────────────
# Strategy B: when no active node needs the WHOLE active band (see DSPNode.
# GLOBAL_STATS), the full_depth extraction becomes visible-Y + a halo instead
# of the whole band — collapsing the row count from "the file's entire real
# signal depth" (tens of thousands of samples on a deep cadena) down to a few
# thousand at most, which is what actually starves the pan-margin cache's
# column budget (see PAN_BAND_BUDGET_BYTES below). The halo is the LARGER of
# this floor and whatever the active nodes themselves declare via
# time_halo_samples (e.g. AGC's is exactly half its sliding-RMS window) — see
# _refresh's any_global_stats/y_halo. 500 samples is generous slack for any
# node whose declared halo is small or zero, while staying tiny next to a
# typical real-signal band.
Y_HALO_MIN_SAMPLES = 500

# ── Pan-margin cache (the decisive lateral-pan fix) ──────────────────────────────
# When zoomed in, the visible window is a thin COLUMN band of the full-depth
# matrix. Lateral panning shifts that band, and because the pipeline cache keys
# on the exact window, every pan re-runs the whole DSP chain (F-K's 2-D FFT over
# the full real depth — seconds on a deep cadena). Instead, on an eligible
# refresh we process a band WIDER than the viewport (``PAN_MARGIN_FRAC`` extra on
# each side) and keep the processed result (see pipeline.ProcessedBand). A
# subsequent pan whose viewport still falls inside that band's trusted interior
# is served by re-slicing the cache on the GUI thread — NO worker, the same way
# PyQtGraph already pans the raster itself. A fresh band is dispatched only when
# the user pans out toward the margin edge, zooms, or edits the chain.
PAN_MARGIN_FRAC = 0.6
# Hard ceiling on the cached band's bytes so a genuinely deep file (active_ns in
# the tens of thousands) can't blow RAM: the margin auto-shrinks to fit, down to
# zero (then this gracefully degrades to the per-pan worker path). float32.
PAN_BAND_BUDGET_BYTES = 512 * 1024 * 1024

# Global amplitude-level estimation — FULL-RESOLUTION representative blocks,
# NOT a globally decimated grid. Row/column striding would corrupt the very
# physics the DSP chain depends on: a temporal filter (AGC, Deconvolution,
# Envelope) is defined in terms of the TRUE sample interval, so processing a
# coarsened dt yields a numerically different — "fake" — amplitude that drifts
# from whatever the user actually sees once they zoom in to full resolution.
# Instead, GLOBAL_LEVELS_BLOCKS CONTIGUOUS spans of GLOBAL_LEVELS_BLOCK_TRACES
# traces each, evenly spaced across the profile, are extracted with EVERY time
# sample kept (no row skipping at all) — each block is processed by the
# CURRENT per-window filter chain exactly as a real ViewBox window would be,
# so the result is mathematically identical to full-resolution zoomed-in DSP,
# just sampled at a few representative locations instead of everywhere. Kept
# narrow (a few hundred traces total) so the WORKER-thread recompute (see
# PipelineWorker's global_levels_job — never inline on the GUI thread) stays
# fast even with a heavier node in the chain — it only runs on a cache miss,
# when the source, the alignment flag, the pre-crop/window node chain's
# CONTENT, or the clip percentile actually changes, never on pan/zoom (see
# _refresh's levels_key/_global_levels_cache check).
GLOBAL_LEVELS_BLOCKS = 5
GLOBAL_LEVELS_BLOCK_TRACES = 150

# Higher column cap used ONLY while a NEEDS_FULL_RES node (e.g. the F-K dip
# filter) is active, so the 2-D filter sees the true viewport trace spacing
# (rows are already always full-resolution — see MAX_PREVIEW_COLS above).
# Bounded so a zoomed-out viewport can't request a many-thousand-trace 2-D
# FFT that would freeze the settle; at or below this the column window is
# taken at stride 1 (exact trace spacing).
FK_MAX_COLS = 4096

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

        self.pipeline = Pipeline()         # all per-WINDOW filter nodes (full node list)
        self.pipeline_global = Pipeline()  # GLOBAL_STATS nodes + their predecessors
        self.pipeline_local = Pipeline()   # nodes after the last GLOBAL_STATS node
        self._precrop_nodes: list = []   # PRECROP nodes (e.g. Water Mute) — full-array
        self._precrop_sig: tuple = ()
        self._data_version = 0           # bumped when source / align / pre-crop changes
        self._has_source = False

        # Cached PREPARED base: static delay alignment + PRE-CROP nodes (water
        # mute) applied. Phase 7: this is now COLUMN-BOUNDED, not the whole
        # chain — see _prepared_base's docstring. ``_pc_c0``/``_pc_c1`` are the
        # ABSOLUTE column range ``_pc_array`` actually covers (0/n_traces for
        # the alignment-active or fit case, which still need the whole chain).
        # Rebuilt when the source/align/pre-crop signature changes, OR when a
        # pan reaches outside the cached range — NOT on every pan/zoom frame.
        self._pc_array = None
        self._pc_t0 = 0.0
        self._pc_src = None
        self._pc_key = None              # (align_flag, precrop_signature)
        self._pc_c0 = 0
        self._pc_c1 = 0

        # Display-param snapshot: re-read from Qt widgets only when a display
        # setting actually changed (not on every pan/zoom frame).
        self._disp_cache: Optional[dict] = None
        self._disp_dirty: bool = True

        # Halo cache: max_trace_halo (spatial filters' lateral padding) and
        # max_time_halo (per-node local-context requirement — used by
        # Strategy B's vertical windowing, see _refresh) only change when
        # the source object or the pipeline node set changes — not on
        # pan/zoom.
        self._halo_cache: Optional[tuple] = None  # (halo_key, trace_halo, time_halo)
        self._pipeline_version: int = 0

        # Cached wiggle geometry key. When window bounds, vmax, and va_fill are
        # unchanged, _build_wiggle + arrayToQPath are skipped entirely — colormap,
        # FIX-mark, and show_raster changes pay zero geometry cost.
        self._wiggle_state_key = None

        # Locked GLOBAL amplitude levels (fixes "color pumping" on pan/zoom):
        # computed once from a few FULL-RESOLUTION representative blocks of
        # the profile (see GLOBAL_LEVELS_BLOCKS/BLOCK_TRACES — never row/col
        # decimated, so the DSP math matches full-zoom exactly) whenever the
        # source, alignment, PRE-CROP nodes, per-window filter chain, or clip
        # percentile actually changes — see the levels_key check in _refresh.
        # NEVER recomputed from the current ViewBox, so the palette stays
        # stable across pan/zoom and only moves when the data or DSP
        # settings actually do.
        self._global_vmax: float = 1.0
        self._global_vmax_raw: float = 1.0   # A/B Compare's "raw" half ceiling
        self._global_levels_key: Optional[tuple] = None
        # Cache of every (vmax_proc, vmax_raw) result ever computed, keyed by
        # the pipeline's own CONTENT signature (not the ever-incrementing
        # _pipeline_version counter) — so toggling a node off and back on,
        # or undoing/redoing an edit, reuses a previously-computed result
        # instantly instead of recomputing. Unbounded but cheap (two floats
        # per entry); see _refresh for the key shape and _on_pipeline_result
        # for where a miss gets stored. Never computed on the GUI thread —
        # see _pending_levels_key/PipelineWorker's global_levels_job.
        self._global_levels_cache: dict = {}
        self._pending_levels_key: Optional[tuple] = None
        # Cache of the RAW SAMPLE ARRAYS (raw_sample, processed_sample) the
        # 5-block computation produced, keyed by (data_version, pipeline_sig)
        # — deliberately WITHOUT clip, since the samples themselves don't
        # depend on it (only the percentile derived FROM them does). This is
        # what makes the Clip slider instant (see clip_changed): moving it
        # alone never needs a new PipelineWorker round-trip — it just
        # re-percentiles the SAME already-computed samples on the GUI
        # thread, which is cheap, then pushes [vmin, vmax] straight to the
        # ImageItem via SeismicView.set_levels_only (no array resubmission).
        self._global_samples_cache: dict = {}

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
        self._pending_params: bool = False       # a params_changed was deferred
        # Off-GUI-thread disk I/O for evicted chain segments (Phase 10).
        # Runs BEFORE PipelineWorker; the two are never in flight simultaneously.
        # _pending_base_ctx stores the (fit, overlays) to replay once warm.
        self._base_worker: Optional[BasePrepWorker] = None
        self._pending_base_ctx: Optional[tuple] = None
        # Cooperative cancellation: the token handed to the CURRENTLY in-
        # flight worker (None when idle). A fresher request arriving while
        # busy cancels this SAME token (see _refresh) so the stale worker
        # aborts at its next checkpoint instead of finishing uselessly.
        self._cancel_token: Optional[CancelToken] = None
        # Interactive LOD: True for the whole span between a param slider's
        # press and release — see _on_interaction_started/_ended.
        self._lod_active: bool = False
        # Deferred global levels: True from a slider release until the
        # LEVELS_SETTLE_MS timer fires, during which the locked ceiling is held
        # and no 5-block recompute is armed (see _refresh's levels block).
        self._levels_frozen: bool = False
        self._levels_timer = QTimer(self)
        self._levels_timer.setSingleShot(True)
        self._levels_timer.setInterval(LEVELS_SETTLE_MS)
        self._levels_timer.timeout.connect(self._on_levels_settle)
        # Settle gating: the render key (window token + pipeline + display
        # signature) of the image CURRENTLY on screen. A pure pan/zoom that
        # resolves to the SAME column window produces a byte-identical image,
        # so _refresh skips the whole worker round-trip when this is unchanged.
        # Set only after a render actually completes (see _refresh's dispatch /
        # _on_pipeline_failed reset) so a cancelled run never leaves a stale
        # key that would wrongly gate out the next request.
        self._last_render_key: Optional[tuple] = None
        # Telemetry: perf_counter at worker dispatch, read once in
        # _on_pipeline_result for the round-trip timing (only when _PERF).
        self._refresh_dispatch_t: Optional[float] = None
        # Pan-margin cache: the last processed-WIDER-than-viewport band (see
        # PAN_MARGIN_FRAC / pipeline.ProcessedBand). A lateral pan that stays
        # inside its trusted interior is re-sliced from here on the GUI thread
        # instead of dispatching the DSP worker. Invalidated whenever the data
        # or pipeline changes (set_source / _on_pipeline_changed / alignment).
        self._band_cache: Optional[ProcessedBand] = None

        view.enable_preview(True)
        view.view_range_changed.connect(self._on_view_changed)
        panel.pipeline_changed.connect(self._on_pipeline_changed)
        panel.params_changed.connect(self._on_params_changed)
        panel.interactionStarted.connect(self._on_interaction_started)
        panel.interactionEnded.connect(self._on_interaction_ended)
        panel.node_selection_changed.connect(self._on_node_selected)
        view.ab_split_changed.connect(self._on_ab_drag)
        view.mute_horizon_edited.connect(self._on_mute_horizon_edited)

    # ── External triggers ───────────────────────────────────────────────────

    def set_source(self, obj: object) -> None:
        """New profile/chain selected (or its traces just finished loading)."""
        self._ensure_idle()
        self._data_version += 1
        self._disp_dirty = True
        self._last_render_key = None     # new data → no prior frame to gate against
        self._band_cache = None          # new data → stale pan-margin band
        self.pipeline.clear_cache()
        self.pipeline_global.clear_cache()
        self.pipeline_local.clear_cache()
        self._sync_nodes()
        self._pc_array = None
        self._pc_src = None
        self._pending_base_ctx = None
        self._has_source = obj is not None and getattr(obj, "data", None) is not None
        # Remove any stale horizon from the previous source; it will be rebuilt
        # by _on_node_selected once the new data is available.
        self.view.hide_mute_horizon()
        if self._has_source:
            self._refresh(fit=True, overlays=True)   # initial full-view render
            # If a WaterMuteNode is currently selected, rebuild its overlay now.
            from sbp_studio.gui.dsp.nodes import WaterMuteNode
            node = self.panel.selected_node()
            if isinstance(node, WaterMuteNode):
                self._on_node_selected(node)

    def display_changed(self) -> None:
        """Presentation (cmap / FIX / boundaries / etc.) changed — recolour.

        NOT used by the Clip slider any more — see clip_changed, which
        updates [vmin, vmax] instantly without a PipelineWorker round trip."""
        self._disp_dirty = True
        if self._has_source:
            self._refresh(fit=False, overlays=True)

    def clip_changed(self) -> None:
        """Clip slider moved — recompute ``[vmin, vmax]`` INSTANTLY from the
        cached global-levels SAMPLE arrays (see _global_samples_cache) and
        push the result straight to the view via
        ``SeismicView.set_levels_only`` — NO ``_refresh()``, NO
        ``PipelineWorker`` dispatch, no array resubmission. Must track a
        slider drag at 60 fps, which a worker round-trip per tick cannot
        guarantee.

        ``_disp_dirty`` is still set so the NEXT real refresh (a pan/zoom,
        a different control) re-reads the display dict and picks up this
        new clip value rather than a stale cached one. If no samples are
        cached yet for the current (data_version, pipeline_sig) — e.g. the
        very first frame hasn't finished rendering — this is a no-op; the
        upcoming real refresh will compute everything correctly anyway."""
        self._disp_dirty = True
        if not self._has_source:
            return
        disp = self._get_display()
        clip = float(disp.get("clip", DEFAULT_CLIP_PCT))
        pipeline_sig = tuple(n.signature() for n in self.pipeline.nodes)
        samples = self._global_samples_cache.get((self._data_version, pipeline_sig))
        if samples is None:
            return
        raw_sample, processed_sample = samples
        vmax_raw = self._estimate_vmax(raw_sample, clip)
        vmax_proc = (self._estimate_vmax(processed_sample, clip)
                    if processed_sample is not None else vmax_raw)
        self._global_vmax, self._global_vmax_raw = vmax_proc, vmax_raw
        levels_key = (self._data_version, pipeline_sig, clip)
        self._global_levels_cache[levels_key] = (vmax_proc, vmax_raw)
        self._global_levels_key = levels_key

        ab_on = self._ab_anchor_active()
        vmax = max(vmax_raw, vmax_proc) if ab_on else vmax_proc
        vmin = -vmax if disp.get("amp_range") == "diverging" else 0.0
        self.view.set_levels_only(vmax, vmin)

    def alignment_changed(self) -> None:
        """Static delay-alignment toggled — rebuild the prepared base, invalidate
        caches, and refit (the time origin and matrix height change)."""
        self._ensure_idle()
        self._pc_array = None
        self._data_version += 1          # window cache keys include data_version
        self._disp_dirty = True
        self._band_cache = None          # geometry changed → stale pan-margin band
        if self._has_source:
            self._refresh(fit=True, overlays=True)

    def fit(self) -> None:
        """Re-fit the whole section to the panel (the 'Fit view' action)."""
        if self._has_source:
            self._refresh(fit=True, overlays=True)

    # ── Interactive LOD (param slider press/release) ────────────────────────

    def _on_interaction_started(self) -> None:
        """A DSP node param slider was just pressed — every refresh until
        release uses a slashed column extent (see LOD_SCALE/_refresh). Also
        cancels any pending deferred-levels timer from a PREVIOUS release: a
        new drag has started, so that stale recompute must not fire mid-drag."""
        self._lod_active = True
        self._levels_timer.stop()

    def _on_interaction_ended(self) -> None:
        """Slider released — restore full COLUMN resolution and render one final
        high-quality raster immediately, but keep the locked color ceiling
        FROZEN (``_levels_frozen``) and defer the real 5-block levels recompute
        to LEVELS_SETTLE_MS later (see _on_levels_settle). So the released
        frame appears instantly reusing the last ceiling instead of blocking on
        a full global-levels pass, and a rapid next edit pre-empts that pass
        entirely."""
        self._lod_active = False
        self._levels_frozen = True
        if self._has_source:
            self._refresh(fit=False, overlays=False)
        self._levels_timer.start()           # restarts if already pending

    def _on_levels_settle(self) -> None:
        """The post-release settle elapsed with no new drag — unfreeze and do
        the one real global-levels recompute now (a normal _refresh; the
        levels block re-arms because the freeze is cleared). Skipped while a
        worker is mid-flight or another drag is active — _dispatch_pending /
        the next release will re-trigger it."""
        self._levels_frozen = False
        if self._lod_active or not self._has_source:
            return
        # overlays=True so settle gating (which would otherwise skip this as a
        # no-op — same window + pipeline as the already-rendered release frame)
        # is bypassed and the now-unfrozen levels block actually re-arms.
        self._refresh(fit=False, overlays=True)

    # ── Internal trigger slots ──────────────────────────────────────────────

    def _sync_nodes(self) -> bool:
        """Split panel nodes into PRE-CROP (full-array) vs per-window filters.
        Returns True if the pre-crop set changed (→ prepared base must rebuild).

        Reads ``active_nodes()`` (enabled-only), so a MUTED node is bypassed in
        the live preview exactly as it is in the export."""
        from .nodes import AB_AnchorNode
        nodes = self.panel.active_nodes()
        precrop = [n for n in nodes if getattr(n, "PRECROP", False)]
        window  = [n for n in nodes if not getattr(n, "PRECROP", False)]
        self.pipeline.set_nodes(window)
        # Split at the last GLOBAL_STATS node boundary for two-pass execution.
        # pipeline_global processes all nodes up to and including the last
        # GLOBAL_STATS node on the full active band; pipeline_local processes
        # the remaining (purely local) nodes on the visible+halo window.
        # The anchor must always live in pipeline_local so its _snapshot
        # shape matches win.crop_visible() — cap the global split point
        # before the anchor's position to enforce this invariant.
        anchor_pos = next(
            (i for i, n in enumerate(window) if isinstance(n, AB_AnchorNode)),
            len(window))
        last_gs = max((i for i, n in enumerate(window[:anchor_pos])
                       if getattr(n, "GLOBAL_STATS", False)), default=-1)
        self.pipeline_global.set_nodes(window[:last_gs + 1])
        self.pipeline_local.set_nodes(window[last_gs + 1:])
        sig = tuple(n.signature() for n in precrop)
        self._precrop_nodes = precrop
        if sig != self._precrop_sig:
            self._precrop_sig = sig
            return True
        return False

    def _on_pipeline_changed(self) -> None:
        # _sync_nodes() mutates self.pipeline.nodes, which the worker thread may
        # be iterating right now — defer the whole handler (not just _refresh)
        # until it's done, rather than racing it.  Also defer while a
        # BasePrepWorker is warming the cache (_pc_array is being written).
        if ((self._worker is not None and self._worker.isRunning())
                or (self._base_worker is not None and self._base_worker.isRunning())):
            self._pending_sync = True
            return
        if self._sync_nodes():           # pre-crop (e.g. Water Mute) changed
            self._pc_array = None         # rebuild the prepared base
            self._data_version += 1       # invalidate per-window cache
        self._pipeline_version += 1      # invalidate halo cache (node set changed)
        self._band_cache = None          # node set changed → stale pan-margin band
        self._ab_cache = None            # anchor may have moved → stale A/B arrays
        if self._has_source:
            self._refresh(fit=False, overlays=False)

    def _on_params_changed(self) -> None:
        """A node PARAMETER value changed (slider / spinbox) — the node set is
        unchanged, only amplitude/filter values differ.

        Unlike ``_on_pipeline_changed`` (structural), this does NOT clear
        ``_band_cache``: the cached band's spatial footprint (column range,
        row range, strides) is identical to what the fresh run will produce,
        so the ``pipeline_sig`` guard in ``reslice_band`` is sufficient to
        prevent serving stale data — the old band can't be matched once
        pipeline_sig changes, and the new worker run overwrites the cache.

        ``_sync_nodes`` is also skipped: node objects update their own
        ``params`` dict in-place, so ``self.pipeline.nodes`` already reflects
        the current values without a re-set.

        ``_pipeline_version`` IS bumped so the halo cache (``_halo_cache``)
        is rebuilt on the next refresh — a param change (e.g. AGC window
        size) can legitimately alter a node's ``time_halo_samples``."""
        if ((self._worker is not None and self._worker.isRunning())
                or (self._base_worker is not None and self._base_worker.isRunning())):
            self._pending_params = True
            return
        self._pipeline_version += 1
        if self._has_source:
            self._refresh(fit=False, overlays=False)

    def _on_view_changed(self) -> None:
        if self._has_source:
            self._refresh(fit=False, overlays=False)

    # ── Mute-horizon overlay lifecycle ──────────────────────────────────────

    def _on_node_selected(self, node: object) -> None:
        """Show the interactive horizon when a WaterMuteNode is selected."""
        from sbp_studio.gui.dsp.nodes import WaterMuteNode
        if not isinstance(node, WaterMuteNode):
            self.view.hide_mute_horizon()
            return
        obj = self._get_source()
        if obj is None or getattr(obj, "data", None) is None:
            self.view.hide_mute_horizon()
            return
        data = obj.data
        dt_us = getattr(obj, "dt_us", 1000)
        n_traces = data.shape[1]
        dist_km = np.asarray(getattr(obj, "dist_km",
                                      np.linspace(0.0, 1.0, n_traces)))
        # t0_ms: absolute display time (ms) of raw sample-0. When delay
        # alignment is inactive this is 0. Auto-picks are computed on obj.data
        # (raw, 0-relative) so they must be shifted by t0_ms before they reach
        # the ViewBox. Manual picks are already stored in absolute display ms
        # (written by _on_mute_horizon_edited from ViewBox Y coords) — no shift.
        try:
            _, t0_ms, _ = self._prepared_base(obj)
        except Exception:
            t0_ms = 0.0
        picks_ms = node.get_manual_picks_ms(n_traces)
        if picks_ms is None:
            picks_ms = WaterMuteNode.compute_auto_picks_ms(
                data, dt_us,
                node.params.get("threshold_pct", 30.0),
                node.params.get("smoothing_ms", 0.5))
            picks_ms = picks_ms + t0_ms  # raw-relative → absolute display ms
        self.view.show_mute_horizon(picks_ms, dist_km)

    def _on_mute_horizon_edited(self, km_time_pts: list) -> None:
        """Convert viewport (km, ms) control points → trace fractions, store in node."""
        from sbp_studio.gui.dsp.nodes import WaterMuteNode
        node = self.panel.selected_node()
        if not isinstance(node, WaterMuteNode):
            return
        obj = self._get_source()
        if obj is None:
            return
        n_traces = getattr(obj, "n_traces", None) or (
            obj.data.shape[1] if getattr(obj, "data", None) is not None else 0)
        if n_traces == 0:
            return
        dist_km = np.asarray(getattr(obj, "dist_km",
                                      np.linspace(0.0, 1.0, n_traces)), dtype=float)
        ctrl_pts = []
        for x_km, time_ms in km_time_pts:
            idx = int(np.clip(np.searchsorted(dist_km, x_km), 0, n_traces - 1))
            frac = float(idx) / max(1, n_traces - 1)
            ctrl_pts.append([frac, float(time_ms)])
        node.set_manual_picks(ctrl_pts)
        # manual_picks changed → pre-crop sig changes → _on_pipeline_changed
        # will detect it and rebuild _pc_array.
        self._on_pipeline_changed()

    # ── Prepared base (alignment + PRE-CROP nodes, column-bounded, cached) ───

    def _prepared_base(self, obj: object, c0_req: Optional[int] = None,
                       c1_req: Optional[int] = None, *,
                       _align: Optional[bool] = None):
        """Return ``(base_array, t0_ms, base_c0)``: static delay alignment +
        any PRE-CROP nodes (water mute) applied to a column-bounded slice of
        ``obj`` covering AT LEAST ``[c0_req, c1_req)`` — ``base_c0`` is the
        ABSOLUTE column ``base_array[:, 0]`` corresponds to (0 for the
        whole-chain cases below). Cached so a pan WITHIN the cached range
        never re-reads/re-precrops anything.

        Water-column mute MUST see the whole TIME extent (full depth) of each
        trace it touches — its seabed pick needs the whole trace, and the
        window's row 0 is not true t=0 — but it does NOT need every trace in
        the chain simultaneously: it's per-trace (at most a small
        ``trace_halo`` for spatial PRECROP nodes), so a column-bounded read
        (via ``read_columns`` — see model.py) is exactly as correct as the
        historical whole-chain pass, while never materialising the rest of a
        deep cadena just to view one segment.

        Falls back to the WHOLE chain (``c0_req``/``c1_req`` ignored) when:
          * no bounds are given (e.g. ``analysis_inputs``'s "full" scope,
            which legitimately wants everything), or
          * static delay alignment is ON — ``apply_delay_alignment``'s
            row-growth (``offsets.max()``) depends on the delay SPREAD of
            whichever traces it's given; computing it from a column-bounded
            slice would make the array's row count (and thus ``t0``/every
            row index downstream) drift depending on which columns happen to
            be in view — a real correctness hazard, not just a missed
            optimisation. Fixing that needs an explicit global-offset
            parameter on ``apply_delay_alignment`` (core/processing.py) —
            future work, out of this pass's scope. Alignment is an opt-in
            toggle, so the common (alignment-off) case still gets the win.
        """
        align = (_align if _align is not None
                 else bool(self._get_display().get("align", False)))
        key = (align, self._precrop_sig)
        whole_chain = align or c0_req is None or c1_req is None
        n_traces_total = int(getattr(obj, "n_traces", 0))
        if whole_chain:
            c0_req, c1_req = 0, n_traces_total

        if (self._pc_array is not None and self._pc_src is obj
                and self._pc_key == key
                and self._pc_c0 <= c0_req and c1_req <= self._pc_c1):
            return self._pc_array, self._pc_t0, self._pc_c0

        # Cache MISS. Whole-chain case: read everything (the historical
        # behaviour, unavoidable here — see the docstring). Bounded case:
        # widen the request by a margin (same RAM-budget philosophy as the
        # pan-margin cache — PAN_BAND_BUDGET_BYTES — sized against the
        # FULL row depth, since this read is always full_depth) so a
        # following pan within that margin hits the cache too.
        t_start = perf_counter() if _PERF else 0.0
        if whole_chain:
            base_c0, base_c1 = 0, n_traces_total
        else:
            ns_full = int(getattr(obj, "ns", 1)) or 1
            budget_cols = max(1, PAN_BAND_BUDGET_BYTES // (ns_full * 4))
            req_width = max(1, c1_req - c0_req)
            margin = max(0, min(int(PAN_MARGIN_FRAC * req_width),
                                (budget_cols - req_width) // 2))
            base_c0 = max(0, c0_req - margin)
            base_c1 = min(n_traces_total, c1_req + margin)

        read_columns = getattr(obj, "read_columns", None)
        base = (read_columns(base_c0, base_c1) if read_columns is not None
                else obj.data[:, base_c0:base_c1])
        t0 = float(getattr(obj, "delay_ms", 0.0))
        full_ctx = DSPContext.from_source(obj)
        # 1) Static geometry: delay alignment (taller array, t0 = min_delay).
        #    Only reachable when whole_chain is True — see the docstring.
        if align and getattr(obj, "delays", None) is not None:
            from sbp_studio.core import apply_delay_alignment
            base = apply_delay_alignment(
                base, obj.delays, obj.min_delay, obj.dt_us, fill_value=0.0)
            t0 = float(getattr(obj, "min_delay", 0.0))
        # 2) PRE-CROP nodes (water mute) on the (bounded or whole) aligned array.
        for node in self._precrop_nodes:
            base = node.apply(base, full_ctx)

        self._pc_array, self._pc_t0, self._pc_src, self._pc_key = base, t0, obj, key
        self._pc_c0, self._pc_c1 = base_c0, base_c1
        if _PERF:
            _LOG.info("perf _prepared_base REBUILD: %.1f ms "
                      "(precrop_nodes=%d, align=%s, cols=[%d,%d) of %d, shape=%s)",
                      (perf_counter() - t_start) * 1e3, len(self._precrop_nodes),
                      align, base_c0, base_c1, n_traces_total,
                      getattr(base, "shape", None))
        return base, t0, base_c0

    def _compute_global_levels(self, data: np.ndarray, dt_us: int,
                               clip: float,
                               cancel: Optional[CancelToken] = None,
                               active_ns: Optional[int] = None) -> tuple:
        """Zoom-independent ``(vmax_processed, vmax_raw, raw_sample,
        processed_sample)`` for the WHOLE profile — the fix for "color
        pumping": the levels never depend on the current ViewBox, only on
        the data itself.

        ``active_ns`` — when given (the profile/chain's real listening-window
        depth, see SegyProfile.active_ns), the sampled blocks are capped to
        that many rows BEFORE the per-window DSP runs. This is the same
        true-depth saving the viewport already takes (Phase 4): the heavy
        node (F-K's 2-D FFT over ~32 k rows, Decon) no longer chews through a
        file's dead/padded tail just to estimate a ceiling. NOTE it also makes
        the ceiling MORE representative — a gain stage (AGC) amplifies the
        near-silent dead zone to full scale, and including those rows in the
        percentile biased ``vmax_proc`` toward that amplified noise; capping
        them out ties the ceiling to the real-signal zone instead.

        ``cancel`` — same cooperative-cancellation token as the viewport's
        own pipeline run (see _refresh); checked once per block below so a
        superseded request aborts here too instead of finishing all 5
        blocks uselessly.

        ``data`` is the prepared base (alignment + pre-crop already applied,
        see _prepared_base). Sampled as a few FULL-RESOLUTION, CONTIGUOUS
        blocks (GLOBAL_LEVELS_BLOCKS × GLOBAL_LEVELS_BLOCK_TRACES traces,
        every time sample kept — see the module-level comment) rather than a
        globally decimated grid, so each block is run through the CURRENT
        per-window filter chain at the EXACT same sample interval (``dt_us``,
        unaltered) the real ViewBox window uses — the ceiling this produces
        is mathematically identical to what full-resolution zoomed-in DSP
        would show, not an artifact of a coarsened/distorted dt. ``vmax_raw``
        (no per-window filters) is kept separately for A/B Compare, which
        needs a fair ceiling for BOTH its raw and processed halves.

        The returned ``raw_sample``/``processed_sample`` arrays (the SAME
        ones the percentile was estimated from) are cached by the caller
        (see _on_pipeline_result → _global_samples_cache) so a LATER
        clip-only change can re-percentile them directly on the GUI thread
        — see clip_changed — instead of recomputing this whole block pass.

        Called from the WORKER thread (see PipelineWorker's
        global_levels_job) on a cache miss, never inline on the GUI thread
        — even though the sampled trace count stays small, a heavy node
        (Deconvolution, F-K) running 5 extra blocks synchronously was
        measurably enough to freeze the UI on every filter add/edit. Only
        triggered when the source/alignment/pipeline-content/clip actually
        changes (see _refresh's levels_key/_global_levels_cache check),
        never on every pan/zoom frame."""
        ns, n_traces = data.shape
        ns_eff = min(ns, int(active_ns)) if active_ns else ns   # true-depth cap
        block_w = min(GLOBAL_LEVELS_BLOCK_TRACES, n_traces)
        n_blocks = min(GLOBAL_LEVELS_BLOCKS, max(1, n_traces // block_w))
        if n_blocks <= 1:
            starts = [0]
        else:
            starts = np.linspace(0, n_traces - block_w, n_blocks).astype(int)
        raw_blocks = [data[:ns_eff, s:s + block_w] for s in starts]

        raw_sample = np.concatenate(raw_blocks, axis=1)
        vmax_raw = self._estimate_vmax(raw_sample, clip)
        if not self.pipeline.nodes:
            return vmax_raw, vmax_raw, raw_sample, None

        ctx = DSPContext(dt_us=int(dt_us), ns=ns_eff, n_traces=block_w,
                         cancel=cancel, preview=True)
        processed_blocks = []
        for i, block in enumerate(raw_blocks):
            if cancel is not None:
                cancel.check()
            processed_blocks.append(self.pipeline.process(
                np.ascontiguousarray(block, dtype=np.float32), ctx,
                input_token=("__global_levels__", self._data_version, i),
                cancel=cancel))
        processed_sample = np.concatenate(processed_blocks, axis=1)
        vmax_proc = self._estimate_vmax(processed_sample, clip)
        return vmax_proc, vmax_raw, raw_sample, processed_sample

    # ── The live loop ───────────────────────────────────────────────────────

    def _render_key(self, win, pipeline_sig: tuple, clip: float, disp: dict,
                    col_offset: int = 0) -> tuple:
        """Stable identity of the IMAGE a refresh would put on screen — drives
        settle gating (skip a redundant re-render) for BOTH the worker-dispatch
        and the pan-margin re-slice paths, so they agree on what "already
        shown" means. Keyed on the VISIBLE trace/sample bounds (not the
        processed band's full extent) so it discriminates a Y-zoom — which
        leaves full_depth's processed rows unchanged but does change what the
        viewer sees — and so a dispatch render and a later re-slice of the same
        viewport produce the SAME key.

        ``col_offset`` — the ABSOLUTE column ``win``'s indices are relative
        to (0 for the pan-margin reslice path, which already resolves against
        the full chain's dist_km; the prepared-base's own ``base_c0`` for the
        worker-dispatch path, whose ``win`` is local to a column-bounded
        ``_prepared_base`` read — see ``_refresh``). Without this, two
        DIFFERENT bounded reads that happen to produce the same LOCAL indices
        would alias to the same key despite being different viewports."""
        vis_token = (self._data_version, win.c_vis0 + col_offset,
                     win.c_vis1 + col_offset, win.s0, win.s1,
                     win.col_stride, win.row_stride)
        return (
            vis_token, pipeline_sig, clip,
            disp.get("cmap"), disp.get("inv_cmap"), disp.get("amp_range"),
            disp.get("style", "density"), self._ab_anchor_active(),
            bool(disp.get("va_fill", True)), bool(disp.get("show_raster", True)),
            bool(disp.get("show_wiggle_line", True)),
            self._ab_frac, self._lod_active)

    def _refresh(self, *, fit: bool, overlays: bool = False) -> None:
        # Only one pipeline worker runs at a time (see preview_worker). A
        # trigger that arrives while one is in flight is coalesced into
        # ``_pending`` and replayed once it finishes — never overlapped.
        # This NEW request supersedes whatever the busy worker is doing, so
        # cancel its token: the worker aborts at its next checkpoint
        # (between nodes, or inside a node's own loop — see DSPContext.cancel)
        # instead of wastefully finishing a now-stale computation, freeing
        # the thread for THIS request as soon as it notices and exits.
        if self._worker is not None and self._worker.isRunning():
            if self._cancel_token is not None:
                self._cancel_token.cancel()
            self._pending = (fit, overlays)
            return
        if self._base_worker is not None and self._base_worker.isRunning():
            self._pending_base_ctx = (fit, overlays)
            return
        obj = self._get_source()
        if obj is None or getattr(obj, "data", None) is None:
            return
        dist_km = obj.dist_km
        n_traces_total = int(getattr(obj, "n_traces", dist_km.size))
        dt_us = int(obj.dt_us)
        dt_ms = dt_us / 1000.0

        # Resolve the visible COLUMN range EARLY for the common fit=False
        # case — pure header/ViewBox data (dist_km, n_traces_total), no
        # _prepared_base result needed yet — so _prepared_base (below) can be
        # told which columns this refresh actually needs and read a BOUNDED
        # slice of a deep cadena instead of the whole chain. fit=True needs
        # the WHOLE chain regardless (the user asked to see everything), so
        # it skips this and _prepared_base falls back to its whole-chain path.
        if fit:
            cvis0, cvis1 = 0, n_traces_total
        else:
            x_range, y_range = self.view.current_view_range()
            xmin_v, xmax_v = sorted(x_range)
            cvis0 = int(np.clip(np.searchsorted(dist_km, xmin_v, side="left"),
                                0, n_traces_total - 1))
            cvis1 = int(np.clip(np.searchsorted(dist_km, xmax_v, side="right"),
                                cvis0 + 1, n_traces_total))

        # Re-read display params only when a setting actually changed (not on
        # every pan/zoom frame — avoids ~10 Qt widget reads per settle).
        if self._disp_dirty or self._disp_cache is None:
            self._disp_cache = self._get_display()
            self._disp_dirty = False
        disp = self._disp_cache

        # Prepared base = static delay alignment + pre-crop (water mute)
        # applied to a column-bounded slice covering at least [cvis0, cvis1)
        # — see _prepared_base's docstring (Phase 7: no longer the whole
        # chain, except when alignment is active or fit=True, both of which
        # genuinely need it). The remaining per-window filters run on the
        # cropped ViewBox of this base; the export path runs the full
        # pipeline in list order separately.
        #
        # Phase 10: if the viewport spans an evicted chain segment the
        # read_columns call inside _prepared_base would block the GUI thread
        # on disk I/O.  Detect this BEFORE calling it: capture align on the
        # GUI thread (safe), check the _pc_array cache directly, then ask the
        # ProfileChain whether the requested columns are already in memory.
        # On a MISS dispatch a BasePrepWorker which runs the read off-thread;
        # on SUCCESS _on_base_prep_ready replays _refresh and hits the cache.
        align_for_base = bool(disp.get("align", False))
        _pc_hit = (
            self._pc_array is not None
            and self._pc_src is obj
            and self._pc_key == (align_for_base, self._precrop_sig)
            and self._pc_c0 <= cvis0 and cvis1 <= self._pc_c1)
        if not _pc_hit and not align_for_base:
            _cols_in_cache = getattr(obj, "columns_in_cache", None)
            if _cols_in_cache is not None and not _cols_in_cache(cvis0, cvis1):
                self._start_base_prep(obj, cvis0, cvis1, align_for_base,
                                      fit, overlays)
                return

        data, t0_full, base_c0 = self._prepared_base(obj, cvis0, cvis1,
                                                      _align=align_for_base)
        n_traces = data.shape[1]                      # the (possibly bounded) width
        dist_km_window = dist_km[base_c0:base_c0 + n_traces]   # paired LOCAL slice
        # True listening-window depth/band (SegyProfile.active_ns/active_lo /
        # ProfileChain). Duck-typed: any source/test double without these →
        # None → full ns / [0, ns]. active_ns_for_traces (bottom only) feeds
        # the viewport window cap's legacy path + the global-levels cap;
        # active_band_for_traces (top+bottom — Strategy A) feeds the richer
        # full_depth clamp and the margin-budget sizing below. Both are
        # ALWAYS queried against the CHAIN-WIDE range — intentionally
        # unaffected by _prepared_base's own column bound.
        active_ns_for_traces = getattr(obj, "active_ns_for_traces", None)
        active_band_for_traces = getattr(obj, "active_band_for_traces", None)
        levels_active_ns = (active_ns_for_traces(0, n_traces_total)
                            if active_ns_for_traces is not None else None)

        clip = float(disp.get("clip", DEFAULT_CLIP_PCT))

        # Recompute the LOCKED global levels only when something that
        # actually changes the amplitude distribution has changed — the
        # source/alignment/pre-crop chain (data_version), the per-window
        # filter chain's CONTENT (node signatures — not the ever-incrementing
        # _pipeline_version counter, so a config seen before hits the cache
        # below), or the clip percentile itself. A pure pan/zoom touches
        # none of these, so the key is unchanged and this whole block is a
        # no-op — the whole point of the fix (see _compute_global_levels).
        #
        # NEVER computed synchronously here on the GUI thread (that froze
        # the UI on every filter add/edit — see GLOBAL_LEVELS_BLOCKS' note):
        # on a cache MISS, ``global_levels_job`` below is bundled into the
        # SAME background PipelineWorker run that already processes this
        # refresh's viewport window, and the result is applied later in
        # _on_pipeline_result once the worker reports back.
        pipeline_sig = tuple(n.signature() for n in self.pipeline.nodes)
        levels_key = (self._data_version, pipeline_sig, clip)
        global_levels_job = None
        # While a param slider is actively dragged (_lod_active), FREEZE the
        # locked levels: keep the last committed _global_vmax and skip the
        # 5-block recompute for every throwaway intermediate value. The levels
        # are zoom-independent anyway, so a brief mid-drag staleness is
        # invisible; interactionEnded fires one final full-res refresh (lod
        # inactive) that recomputes them correctly for the committed value.
        # ``_levels_frozen`` extends the freeze briefly PAST a slider release:
        # interactionEnded shows the final full-res raster immediately reusing
        # the last ceiling, and a longer settle timer (LEVELS_SETTLE_MS) does
        # the one real levels recompute afterwards — see _on_interaction_ended /
        # _on_levels_settle. So a flurry of edit→release→edit never pays the
        # 5-block pass per release.
        if (levels_key != self._global_levels_key
                and not self._lod_active and not self._levels_frozen):
            cached = self._global_levels_cache.get(levels_key)
            if cached is not None:
                self._global_vmax, self._global_vmax_raw = cached
                self._global_levels_key = levels_key
            else:
                self._pending_levels_key = levels_key
                global_levels_job = (
                    lambda d=data, dt=dt_us, c=clip, a=levels_active_ns:
                    self._compute_global_levels(d, dt, c, active_ns=a))

        # Trace halo + time halo only change when the source or pipeline node
        # set changes; cache both so pan/zoom skips DSPContext construction +
        # node iteration. Time halo was historically dropped here once
        # full_depth=True made it unnecessary for the OLD "always the whole
        # band" strategy — revived for Strategy B (see ``any_global_stats``/
        # ``y_halo`` below), which needs each active node's own declared
        # local-context requirement (``time_halo_samples`` — e.g. AGC's is
        # exactly half its sliding-RMS window) to size a SAFE windowed
        # extraction instead of always processing the whole active band.
        halo_key = (self._data_version, self._pipeline_version)
        if self._halo_cache is None or self._halo_cache[0] != halo_key:
            ctx = DSPContext.from_source(obj)
            self._halo_cache = (halo_key, self.pipeline.max_trace_halo(ctx),
                               self.pipeline.max_time_halo(ctx))
        _, trace_halo, node_time_halo = self._halo_cache

        # Strategy B eligibility (see DSPNode.GLOBAL_STATS's docstring —
        # PredictiveDeconNode/LogCompressionNode/CLAHENode).
        #
        # Two-pass mode: when GLOBAL_STATS nodes are present BUT local nodes
        # follow the last one, we split execution so only the global phase
        # (pipeline_global: all nodes up to and including the last GLOBAL_STATS
        # node) runs on the full active band, while the local phase
        # (pipeline_local: all remaining nodes) sees only visible+halo. This
        # breaks the stacking multiplier: AGC, Bandpass, Notch etc. go back to
        # processing ~1500 rows even when Decon is in the chain.
        #
        # Alignment-shifted data: _prepared_base row-inserts delay offsets,
        # making window_band sample indices disagree with data row indices —
        # skip two-pass when align is on to avoid extracting the wrong rows.
        any_global_stats = any(getattr(n, "GLOBAL_STATS", False)
                               for n in self.pipeline.nodes)
        has_local_after_gs = (any_global_stats and bool(self.pipeline_local.nodes)
                              and not align_for_base)
        # y_halo drives extract_visible_window's vertical windowing:
        #  - None  → full active band (Strategy A only)
        #  - value → visible+halo window (Strategy B, local phase)
        # In two-pass mode y_halo is set so win.sub covers the local phase
        # dimensions; the global phase builds its own full-band array inline.
        y_halo = (None if any_global_stats and not has_local_after_gs
                  else max(Y_HALO_MIN_SAMPLES, node_time_halo))

        # ── Determine the visible window ────────────────────────────────────
        # fit=True needs the whole chain's extent (resolved only now that
        # _prepared_base's whole-chain result is available); fit=False's
        # x_range/y_range were already resolved early, above, to bound
        # _prepared_base's read.
        if fit:
            x_range = (float(dist_km[0]), float(dist_km[-1]))
            y_range = (t0_full, t0_full + data.shape[0] * dt_ms)

        # ── Overview LOD gate (zoom-out fix) ────────────────────────────────
        # How many NATIVE rows does the visible Y-span cover? (Same floor/ceil
        # mapping extract_visible_window uses for s_vis0/s_vis1.) When that far
        # exceeds TRIGGER × adaptive_target, switch to Overview mode: rows are
        # RMS-pooled down to adaptive_target before DSP runs (quieter noise
        # floor vs. old max-abs, and gentler compression via viewport-adaptive
        # sizing). Gated on the VISIBLE span (not the band depth) so a thin
        # Y-zoom keeps the exact Strategy-B window.
        ymin_ov, ymax_ov = sorted(y_range)
        s0_ov = max(0, int(np.floor((ymin_ov - t0_full) / dt_ms)))
        s1_ov = int(np.ceil((ymax_ov - t0_full) / dt_ms)) + 1
        visible_rows_ov = max(1, s1_ov - s0_ov)
        # Adaptive cap: 4× the live ViewBox pixel height so the overview carries
        # ~4 rows per display pixel — gentler strides mean better RMS estimates.
        _vb_h = int(self.view.plot.getViewBox().height())
        _overview_target = (max(OVERVIEW_MIN_ROWS, _vb_h * OVERVIEW_OVERSAMPLE)
                            if _vb_h > 0 else OVERVIEW_MIN_ROWS)
        overview_max_rows = None
        if visible_rows_ov > OVERVIEW_TRIGGER_FACTOR * _overview_target:
            overview_max_rows = _overview_target
            # Overview RMS-pools the whole visible band uniformly. The two-pass
            # split and Strategy-B windowing both assume UN-pooled row indices
            # (their sh0/sh1/halo math is in native samples), so disable them
            # and let one pooled array carry every node — GLOBAL_STATS included,
            # whose statistics on the pooled overview are the accepted zoom-out
            # approximation. y_halo=None ⇒ extract takes the whole active band
            # (clamped to visible), then RMS-pools it down to overview_max_rows.
            has_local_after_gs = False
            y_halo = None

        # ViewBox-limited extraction (preview only). Live horizontal detail:
        # the "Pixels / trace" control raises the column cap so zooming in
        # stays sharp (default 20 → the historical 4000-col cap; higher
        # keeps more native traces, lower decimates more aggressively).
        ppt = float(disp.get("px_per_trace", 20.0)) or 20.0
        eff_max_cols = int(np.clip(ppt / 20.0 * MAX_PREVIEW_COLS, 1000, 16000))
        # A NEEDS_FULL_RES node (e.g. the F-K dip filter) must see the true,
        # un-decimated trace spacing of the viewport — column decimation would
        # corrupt its wavenumber axis. When one is active, extract at full
        # resolution up to a high safety cap (exact at any practical working
        # zoom; only a viewport wider than the cap decimates gently).
        if any(getattr(n, "NEEDS_FULL_RES", False) for n in self.pipeline.nodes):
            max_cols_use = FK_MAX_COLS
        else:
            max_cols_use = eff_max_cols
        # Interactive LOD: while a param slider is actively being dragged,
        # slash the column extent (see LOD_SCALE's docstring) — masks a
        # heavy node's per-call cost for the (inherently throwaway)
        # intermediate frames; interactionEnded restores full resolution.
        if self._lod_active:
            max_cols_use = max(LOD_MIN_COLS, int(max_cols_use * LOD_SCALE))

        # ── Pan-margin cache: GUI-thread re-slice fast path ─────────────────
        # cvis0/cvis1 (the visible CHAIN-WIDE column bounds) were already
        # resolved early, above (needed to bound _prepared_base's read) —
        # reused here unchanged. The fast path is the zoomed-in raster
        # lateral-pan scenario only: not A/B (its own wiper cache), not
        # mid-drag (LOD owns that), not a fit/overlay pass, and not zoomed
        # out past the render cap (the un-decimated band can't serve that).
        # On a miss we fall straight through to a worker dispatch.
        ab_on = self._ab_anchor_active()
        visible_cols = cvis1 - cvis0
        band_eligible = (not fit and not ab_on and not self._lod_active
                         and visible_cols <= max_cols_use)
        # LOCAL active band for the columns about to be touched — NOT the
        # chain-wide ``levels_active_ns`` (that one intentionally stays
        # worst-cased across the WHOLE chain for the separate global-levels
        # job). Sizing the margin budget from the chain-wide value was the
        # bug: panning inside a shallow segment of a deep cadena had its
        # margin needlessly strangled by some unrelated deep segment
        # elsewhere in the same chain — see the margin-budget fix below.
        window_band = (active_band_for_traces(cvis0, cvis1)
                       if active_band_for_traces is not None else None)
        if band_eligible and not overlays and self._band_cache is not None:
            # reslice_band operates on the cached ProcessedBand + the FULL
            # chain's dist_km — its win.c_vis0/c_vis1 are ALREADY absolute,
            # entirely independent of _prepared_base's own bounded read, so
            # col_offset=0 here (vs base_c0 for the worker-dispatch path
            # below, whose win is local to that bounded read).
            rwin = reslice_band(
                self._band_cache, dist_km, x_range, y_range,
                max_cols=max_cols_use, data_version=self._data_version,
                pipeline_sig=pipeline_sig)
            if rwin is not None:
                rkey = self._render_key(rwin, pipeline_sig, clip, disp)
                if rkey != self._last_render_key:
                    self._finish_refresh(
                        self._band_cache.processed, obj=obj, win=rwin, disp=disp,
                        dist_km=dist_km, n_traces=n_traces_total, t0_full=t0_full,
                        dt_ms=dt_ms, fit=False, overlays=False, band_meta=None,
                        col_offset=0)
                    self._last_render_key = rkey
                self._pending_levels_key = None      # no worker → no levels job
                if _PERF:
                    _LOG.info("perf pan-margin RESLICE hit (GUI thread, no worker)")
                return

        # Size a fresh margin band (only when eligible). The extra columns are
        # bounded by PAN_BAND_BUDGET_BYTES so a deep file can't blow RAM — the
        # margin shrinks to 0 (then this is exactly the pre-existing per-pan
        # dispatch). When a margin IS granted, widen the extraction cap so the
        # band stays un-decimated (col_stride == 1) for a clean 1:1 re-slice.
        col_margin = 0
        extract_max_cols = max_cols_use
        if band_eligible:
            # Margin-budget sizing: use the LOCAL window's depth, not the
            # chain-wide worst case (the fix — see window_band's comment
            # above). When Strategy B applies (y_halo is not None), the row
            # count that will ACTUALLY be extracted is visible+halo, often
            # far shallower than the whole local band — use whichever is
            # smaller so a deep file's UNTOUCHED depth never strangles the
            # margin for a window that won't even process it.
            if window_band is not None:
                local_depth = max(1, window_band[1] - window_band[0])
            elif levels_active_ns:
                local_depth = min(int(levels_active_ns), data.shape[0])
            else:
                local_depth = data.shape[0]
            if y_halo is not None:
                ymin_v, ymax_v = sorted(y_range)
                s_vis0_est = max(0, int(np.floor((ymin_v - t0_full) / dt_ms)))
                s_vis1_est = int(np.ceil((ymax_v - t0_full) / dt_ms)) + 1
                windowed_depth = max(1, (s_vis1_est - s_vis0_est) + 2 * y_halo)
                rows_est = min(local_depth, windowed_depth)
            else:
                rows_est = local_depth
            budget_cols = PAN_BAND_BUDGET_BYTES // (max(1, rows_est) * 4)
            half = max(0, (budget_cols - visible_cols) // 2)
            col_margin = min(int(PAN_MARGIN_FRAC * visible_cols), half)
            if col_margin > 0:
                extract_max_cols = visible_cols + 2 * max(trace_halo, col_margin) + 8
        # full_depth=True: every selected trace is fetched at its FULL
        # vertical depth (all ns samples, no Y-cropping or row decimation),
        # NOT just the visible time window — a time-series filter (AGC,
        # Deconvolution, Envelope) computed on a vertically-truncated trace
        # produces a numerically different (typically suppressed) amplitude
        # than the same filter run on the full trace, which is exactly what
        # the GLOBAL levels were estimated from (_compute_global_levels) —
        # so without this, zooming in would "wash out" against the locked
        # palette. The DSP pipeline below runs on the full-depth array; the
        # Y-range crop happens AFTER, in win.crop_visible (_finish_refresh).
        # Vertical windowing: Strategy A (active_band_for_traces — top+bottom
        # real-signal trim) always applies when available; Strategy B
        # (y_halo, set above) additionally narrows to visible+halo whenever
        # no active node needs the WHOLE band. See extract_visible_window's
        # active_band_lookup/y_halo docstrings.
        # dist_km_window is the LOCAL slice of dist_km paired with `data`
        # (identical to dist_km itself when _prepared_base returned the
        # whole chain — fit=True or alignment-active) — win's c_vis0/c_vis1
        # below come out LOCAL to base_c0, converted back to absolute at the
        # 3 specific points that need it (_render_key/ProcessedBand/
        # show_preview — see _finish_refresh's col_offset parameter).
        win = extract_visible_window(
            data, dist_km_window, t0_ms=t0_full, dt_us=dt_us,
            x_range=x_range, y_range=y_range,
            trace_halo=trace_halo,
            data_version=self._data_version,
            max_cols=extract_max_cols, full_depth=True,
            active_ns_lookup=active_ns_for_traces,
            active_band_lookup=active_band_for_traces,
            col_margin=col_margin, y_halo=y_halo,
            overview_max_rows=overview_max_rows)
        # Eligible to populate the pan-margin cache when a real margin band was
        # built un-decimated (col_stride == 1 ⇒ 1:1 re-slice; row_stride is
        # always 1 under full_depth). Stored after the render in _finish_refresh.
        # ``row0`` (the FULL-array row of win.sub's row 0) is recovered from
        # win.s0 - win.r0 — see VisibleWindow's r0/s0 contract — rather than
        # adding a new field: r0 = (s_vis0 - sh0) // row_stride, row_stride is
        # always 1 under full_depth, so sh0 = s0 - r0 exactly.
        band_meta = None
        if (col_margin > 0 and win.col_stride == 1 and win.row_stride == 1):
            # n_traces_total (CHAIN-WIDE, not the bounded `n_traces`) is what
            # ProcessedBand.n_traces needs — reslice_band always compares
            # against the full chain's dist_km, so band-edge detection
            # (band.c1 == n_traces) must use the same chain-wide count even
            # though this dispatch only processed a bounded slice.
            band_meta = dict(trace_halo=trace_halo, dt_us=dt_us,
                             data_ns=data.shape[0], pipeline_sig=pipeline_sig,
                             row0=win.s0 - win.r0, n_traces_total=n_traces_total)
        if _PERF and active_ns_for_traces is not None:
            _LOG.info("perf active-depth cap: rows=%d of ns=%d (%.0f%% skipped)",
                      win.sub.shape[0], data.shape[0],
                      100.0 * (1.0 - win.sub.shape[0] / max(1, data.shape[0])))

        # ── Settle gating ───────────────────────────────────────────────────
        # A pure pan/zoom (overlays=False, not a fit) that resolved to the SAME
        # VISIBLE viewport AND identical pipeline/display state yields a
        # byte-identical image — skip the entire worker round-trip. The key is
        # built from the VISIBLE trace/sample bounds (_render_key), so a 2-px
        # nudge mapping to the same integer indices is gated, while a Y-zoom
        # (which leaves the full_depth processed extent unchanged) correctly is
        # NOT. A filter edit changes pipeline_sig, a clip/cmap change changes
        # disp — both alter the key; an overlay-only refresh / a fit bypass it.
        render_key = self._render_key(win, pipeline_sig, clip, disp,
                                      col_offset=base_c0)
        if (not fit and not overlays
                and render_key == self._last_render_key):
            # Nothing on screen would change — roll back the levels job we may
            # have armed above (it must not run) and bail before dispatch.
            self._pending_levels_key = None
            return

        # ── Run the (memoized) pipeline on the full-depth bbox at the TRUE
        # dt (full_depth=True ⇒ effective_dt_us == dt_us always) ───────────
        # This is the heavy step (F-K's 2-D FFT, Deconvolution, …) — it runs on
        # a worker thread (see preview_worker.PipelineWorker) so the GUI thread
        # never blocks while it computes; _finish_refresh resumes on the result.
        # A fresh CancelToken per dispatch: cancelling THIS one (see the
        # worker-busy branch above) can never affect a future, unrelated run.
        cancel_token = CancelToken()
        self._cancel_token = cancel_token
        sub_ctx = DSPContext(dt_us=win.effective_dt_us, ns=win.sub.shape[0],
                             n_traces=win.sub.shape[1], cancel=cancel_token,
                             preview=True)
        if global_levels_job is not None:
            global_levels_job = (
                lambda d=data, dt=dt_us, c=clip, tok=cancel_token, a=levels_active_ns:
                self._compute_global_levels(d, dt, c, cancel=tok, active_ns=a))
        # NOTE: dist_km here is the FULL chain-wide array (NOT dist_km_window)
        # — _finish_refresh's set_distance_axis call needs the whole chain's
        # geometry for map-sync/pick lookups; win's LOCAL indices are
        # converted back to absolute via col_offset at the 3 specific points
        # that need it (see _finish_refresh's docstring).
        self._refresh_ctx = dict(obj=obj, win=win, disp=disp, dist_km=dist_km,
                                 n_traces=n_traces, t0_full=t0_full, dt_ms=dt_ms,
                                 fit=fit, overlays=overlays, band_meta=band_meta,
                                 col_offset=base_c0)
        # Telemetry: start of the round-trip clock, popped (not forwarded to
        # _finish_refresh) in _on_pipeline_result. Kept out of render_key so it
        # never affects gating.
        self._refresh_dispatch_t = perf_counter() if _PERF else None
        # Record the key for THIS dispatch so a later identical pan is gated.
        # Reset to None on failure/cancel (see _on_pipeline_failed) so a
        # never-rendered key can't wrongly suppress the next request.
        self._last_render_key = render_key
        # Two-pass split: Phase 1 runs pipeline_global on the full active band
        # (same columns as win.sub but all active rows); Phase 2 extracts the
        # visible+halo rows from Phase 1's output and runs pipeline_local on
        # that narrower array.  win.sub was already sized with y_halo (the local
        # window), so win.r0/r1/s0/s1 are local-window-relative and
        # _finish_refresh needs no changes.  Default-argument capture in the
        # closure prevents late-binding surprises.
        split_fn = None
        if has_local_after_gs:
            full_lo = window_band[0] if window_band is not None else 0
            full_hi = window_band[1] if window_band is not None else data.shape[0]
            # Safety: full band must at least cover the local window (win.sub).
            local_top = win.s0 - win.r0          # = sh0 from extract_visible_window
            local_bot = local_top + win.sub.shape[0]   # = sh1 (exclusive)
            full_hi = max(full_hi, local_bot)
            win_full_sub = np.ascontiguousarray(
                data[full_lo:full_hi, win.c0:win.c1])
            full_token = (self._data_version, "gs",
                          win.c0, win.c1, full_lo, full_hi)
            lr0 = local_top - full_lo   # start of local window in global result
            lr1 = lr0 + win.sub.shape[0]
            _dt = dt_us
            _ns_loc = win.sub.shape[0]
            _nc_loc = win.sub.shape[1]
            _pg = self.pipeline_global
            _pl = self.pipeline_local
            _ltok = win.token

            def split_fn(cancel, _fw=win_full_sub, _ft=full_token,
                         _lr0=lr0, _lr1=lr1, _dt=_dt,
                         _ns=_ns_loc, _nc=_nc_loc,
                         _pg=_pg, _pl=_pl, _ltok=_ltok):
                fc = DSPContext(dt_us=_dt, ns=_fw.shape[0],
                                n_traces=_fw.shape[1], cancel=cancel, preview=True)
                g_out = _pg.process(_fw, fc, input_token=_ft, cancel=cancel)
                loc = np.ascontiguousarray(g_out[_lr0:_lr1, :])
                lc = DSPContext(dt_us=_dt, ns=_ns, n_traces=_nc,
                                cancel=cancel, preview=True)
                return _pl.process(loc, lc, input_token=_ltok, cancel=cancel)

        worker = PipelineWorker(
            self.pipeline, win.sub, sub_ctx, win.token,
            split_fn=split_fn,
            global_levels_job=global_levels_job,
            cancel=cancel_token, parent=self)
        worker.succeeded.connect(self._on_pipeline_result)
        worker.failed.connect(self._on_pipeline_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_pipeline_result(self, token: object, processed: np.ndarray,
                            levels: Optional[tuple]) -> None:
        """GUI-thread continuation of _refresh once the worker's pipeline.process
        (and, on a cache miss, _compute_global_levels) finishes — runs the
        cheap presentation logic and touches Qt widgets.

        ``levels`` is the ``(vmax_proc, vmax_raw)`` the worker computed in
        the background when _refresh found no cached entry for
        ``self._pending_levels_key`` — apply it now and cache it (both the
        derived levels AND the raw sample arrays they came from, the latter
        keyed without clip — see clip_changed) so the SAME pipeline config
        never needs recomputing again, and a later clip-only change can
        re-percentile these exact samples instantly instead."""
        ctx = self._refresh_ctx
        if levels is not None and self._pending_levels_key is not None:
            vmax_proc, vmax_raw, raw_sample, processed_sample = levels
            self._global_vmax, self._global_vmax_raw = vmax_proc, vmax_raw
            self._global_levels_cache[self._pending_levels_key] = (vmax_proc, vmax_raw)
            self._global_levels_key = self._pending_levels_key
            samples_key = self._pending_levels_key[:2]   # (data_version, pipeline_sig) — no clip
            self._global_samples_cache[samples_key] = (raw_sample, processed_sample)
        self._pending_levels_key = None
        self._worker = None
        self._finish_refresh(processed, **ctx)
        if _PERF and self._refresh_dispatch_t is not None:
            _LOG.info("perf _refresh->result round-trip: %.1f ms (cols=%d, rows=%d)",
                      (perf_counter() - self._refresh_dispatch_t) * 1e3,
                      processed.shape[1], processed.shape[0])
            self._refresh_dispatch_t = None
        self._dispatch_pending()

    def _on_pipeline_failed(self, token: object, message: str) -> None:
        self._worker = None
        self._pending_levels_key = None   # the job that would have filled it never ran
        # This dispatch never rendered, so its render_key does NOT describe the
        # on-screen image — drop it so the gate can't wrongly skip the retry.
        self._last_render_key = None
        if message != CANCELLED_MESSAGE:
            _LOG.error("DSP preview pipeline failed: %s", message)
        # else: an intentional cooperative-cancellation abort (see _refresh's
        # worker-busy branch) — expected, not an error, nothing to log.
        self._dispatch_pending()

    # ── BasePrepWorker helpers (Phase 10: off-thread chain segment reload) ──

    def _start_base_prep(self, obj: object, c0: int, c1: int, align: bool,
                         fit: bool, overlays: bool) -> None:
        """Dispatch a BasePrepWorker to read evicted chain columns off-thread.

        ``align`` is captured HERE on the GUI thread (safe) and threaded into
        ``_prepared_base`` via the ``_align`` kwarg so the worker never touches
        Qt widget state.  On success, ``_on_base_prep_ready`` replays _refresh
        which now finds ``_pc_array`` warm and dispatches PipelineWorker."""
        self._pending_base_ctx = (fit, overlays)
        token = (self._data_version, "base_prep", c0, c1)

        def prep_fn(_obj=obj, _c0=c0, _c1=c1, _align=align):
            self._prepared_base(_obj, _c0, _c1, _align=_align)

        w = BasePrepWorker(prep_fn, token, parent=self)
        w.succeeded.connect(self._on_base_prep_ready)
        w.failed.connect(self._on_base_prep_failed)
        self._base_worker = w
        w.start()

    def _on_base_prep_ready(self, token: object) -> None:
        """_pc_array is warm — clear the worker slot, then replay pending work.

        If a structural or param change arrived while the disk read was in
        flight, ``_dispatch_pending`` handles it at the correct priority.
        Otherwise replay the original (fit, overlays) context, which now
        goes through _prepared_base as an instant cache HIT."""
        self._base_worker = None
        ctx = self._pending_base_ctx
        self._pending_base_ctx = None
        if self._pending_sync or self._pending_params:
            self._dispatch_pending()
            return
        if ctx is not None and self._has_source:
            fit, overlays = ctx
            self._refresh(fit=fit, overlays=overlays)

    def _on_base_prep_failed(self, token: object, msg: str) -> None:
        """Segment reload failed — log and clear state.  The ViewBox stays
        frozen at its last frame; the user can try panning back or reopening
        the file."""
        self._base_worker = None
        self._pending_base_ctx = None
        _LOG.error("BasePrepWorker: could not reload chain segment: %s", msg)

    def _dispatch_pending(self) -> None:
        """Replay the most recent trigger that arrived while the worker was busy.
        Priority: structural (pipeline_changed) > param-only (params_changed) >
        plain refresh (pan/zoom). A structural change subsumes any pending param
        change (the full _on_pipeline_changed rebuild handles both)."""
        if self._pending_sync:
            self._pending_sync = False
            self._pending_params = False  # subsumed by structural rebuild
            self._pending = None
            self._on_pipeline_changed()
        elif self._pending_params:
            self._pending_params = False
            self._pending = None
            self._on_params_changed()
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
        if self._base_worker is not None:
            self._base_worker.wait()
            self._base_worker = None
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

    def _finish_refresh(self, processed: np.ndarray, *, obj: object, win,
                        disp: dict, dist_km: np.ndarray, n_traces: int,
                        t0_full: float, dt_ms: float, fit: bool,
                        overlays: bool, band_meta: Optional[dict] = None,
                        col_offset: int = 0) -> None:
        """``dist_km`` is always the FULL, chain-wide array (map-sync/pick
        lookups in ``set_distance_axis`` need the whole chain's geometry).
        ``win``, however, may be LOCAL to a column-bounded ``_prepared_base``
        read (Phase 7: the worker-dispatch path's ``win`` is local to
        ``base_c0``; the pan-margin reslice path's ``rwin`` is already
        absolute). ``col_offset`` is the ABSOLUTE column ``win``'s LOCAL
        indices are relative to (0 / ``base_c0`` respectively) — added at
        every point that indexes ``dist_km`` with a ``win`` index, or stores/
        emits an absolute trace index for something OUTSIDE this call
        (interpretation picks, the pan-margin cache)."""
        visible = win.crop_visible(processed)   # drop the halo rows+cols (decimated)

        if visible.size == 0:
            return

        # Populate the pan-margin cache from this (margin-widened) dispatch so
        # subsequent lateral pans within the band re-slice on the GUI thread
        # (see PreviewController._refresh's fast path). ``processed`` is the
        # full processed band; holding a reference is safe — DSP nodes never
        # mutate their output in place, so this array is immutable once made,
        # even after the pipeline prunes its own prefix cache. ``band_meta`` is
        # None for the re-slice render itself (no re-store) and for any
        # non-eligible dispatch (decimated / A-B / zoomed-out).
        if band_meta is not None:
            self._band_cache = ProcessedBand(
                processed=processed, c0=win.c0 + col_offset,
                c1=win.c1 + col_offset, n_traces=band_meta["n_traces_total"],
                trace_halo=band_meta["trace_halo"], t0_ms=t0_full,
                dt_us=band_meta["dt_us"], data_ns=band_meta["data_ns"],
                data_version=self._data_version,
                pipeline_sig=band_meta["pipeline_sig"], row0=band_meta["row0"])

        # ── Presentation ────────────────────────────────────────────────────
        from sbp_studio.core.constants import CMAPS

        # ── A/B Compare: splice ANCHOR (left) | PROCESSED (right) ───────────
        # State A is the anchor node's snapshot (data at the insertion point in
        # the pipeline), cropped to the same halo window as the final output so
        # the two halves have identical geometry and composite cleanly. State B
        # is the final pipeline output. The split follows the user's draggable
        # wiper (``_ab_frac``, a fraction of the visible width that persists
        # across pan/zoom). Arrays are cached so dragging the wiper recomposes
        # without re-running the pipeline (see _on_ab_drag).
        ab_on = self._ab_anchor_active()
        ab_split_frac = self._ab_frac
        ab_vmax: Optional[float] = None
        if ab_on:
            anchor = self._get_anchor_node()
            if anchor is not None and anchor._snapshot is not None:
                a_visible = np.ascontiguousarray(
                    win.crop_visible(anchor._snapshot), dtype=np.float32)
            else:
                # Anchor not yet run (first frame) — fall back to prepared base
                a_visible = np.ascontiguousarray(win.crop_visible(win.sub),
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
            # Fair ceiling for BOTH halves, locked to the GLOBAL (whole-
            # profile) levels rather than whatever happens to be visible —
            # otherwise A/B Compare would "pump" on pan/zoom exactly like the
            # main view used to. _global_vmax_raw/_global_vmax are computed
            # once in _refresh from the full prepared base (see
            # _compute_global_levels), never from this call's visible crop.
            ab_vmax = max(self._global_vmax_raw, self._global_vmax)
            visible = self._compose_ab(a_visible, proc_visible, split)

        # Locked to the GLOBAL (whole-profile) levels — see _refresh's
        # levels_key check / _compute_global_levels. This is the core color-
        # pumping fix: vmax/vmin no longer depend on the current ViewBox.
        vmax = ab_vmax if ab_vmax is not None else self._global_vmax
        # Amplitude range: diverging maps [−vmax, vmax] (signed, colormap centred
        # on zero); sequential maps [0, vmax] (|amp|, historical default).
        vmin = -vmax if disp.get("amp_range") == "diverging" else 0.0

        cmap_name = CMAPS.get(disp.get("cmap", "Viridis"), "viridis")
        if disp.get("inv_cmap"):
            cmap_name += "_r"

        c1_idx = min(win.c_vis1 - 1, n_traces - 1)
        # dist_km here is the FULL chain-wide array (see _refresh_ctx's
        # comment) — win.c_vis0/c1_idx are LOCAL to a column-bounded
        # _prepared_base read when col_offset != 0, so the absolute index
        # into dist_km is the LOCAL index plus that offset.
        dist0 = float(dist_km[win.c_vis0 + col_offset])
        dist1 = float(dist_km[c1_idx + col_offset])
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
                               c0=win.c_vis0 + col_offset, c1=c1_idx + 1 + col_offset)

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
                raw=a_visible, proc=proc_visible,
                dist0=dist0, dist1=dist1, t_top=t_top, t_bot=t_bot,
                vmax=vmax, vmin=vmin, cmap=cmap_name,
                c0=win.c_vis0 + col_offset, c1=c1_idx + 1 + col_offset)
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
            s = min(split, raw.shape[1])
            combined[:, :s] = raw[:, :s]
        return combined

    def _ab_anchor_active(self) -> bool:
        """True when the pipeline contains at least one enabled AB_AnchorNode."""
        from .nodes import AB_AnchorNode
        return any(isinstance(n, AB_AnchorNode) for n in self.panel.active_nodes())

    def _get_anchor_node(self):
        """Return the first enabled AB_AnchorNode in the pipeline, or None."""
        from .nodes import AB_AnchorNode
        return next(
            (n for n in self.panel.active_nodes() if isinstance(n, AB_AnchorNode)),
            None)

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
        # No bounds passed → _prepared_base's whole-chain fallback (this path
        # wants the full array regardless of scope; base_c0 is always 0 here).
        base, t0, _base_c0 = self._prepared_base(obj)
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
