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

from .nodes import DSPContext
from .pipeline import Pipeline, extract_visible_window

# Preview raster caps. Rows AND columns are decimated so the pipeline never
# processes more than this on screen; row decimation raises the effective dt
# (passed into the node context) so the DSP maths stay physically correct.
# Zoomed in deeply, strides fall to 1 → full-resolution, exact processing.
# Columns capped at 4000: a 4K monitor is only 3840 px wide, so processing
# more traces than that is pure overdraw the screen can't even show.
MAX_PREVIEW_COLS = 4000
MAX_PREVIEW_ROWS = 4000

# Live wiggle tuning. Deflection of a full-scale sample ≈ one trace slot × gain.
# Rows are capped well below the raster's so the single batched curve stays fluid
# (a wiggle needs far fewer samples than a density raster to read cleanly).
WIGGLE_GAIN     = 1.0
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

        view.enable_preview(True)
        view.view_range_changed.connect(self._on_view_changed)
        panel.pipeline_changed.connect(self._on_pipeline_changed)

    # ── External triggers ───────────────────────────────────────────────────

    def set_source(self, obj: object) -> None:
        """New profile/chain selected (or its traces just finished loading)."""
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
        Returns True if the pre-crop set changed (→ prepared base must rebuild)."""
        nodes = self.panel.nodes()
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
        win = extract_visible_window(
            data, dist_km, t0_ms=t0_full, dt_us=dt_us,
            x_range=x_range, y_range=y_range,
            time_halo=time_halo, trace_halo=trace_halo,
            data_version=self._data_version,
            max_rows=MAX_PREVIEW_ROWS, max_cols=eff_max_cols)

        # ── Run the (memoized) pipeline on the small bbox at EFFECTIVE dt ────
        # Row decimation raised the sample interval; the nodes must see it so
        # AGC (and future frequency filters) use the correct sampling rate.
        sub_ctx = DSPContext(dt_us=win.effective_dt_us, ns=win.sub.shape[0],
                             n_traces=win.sub.shape[1])
        processed = self.pipeline.process(win.sub, sub_ctx, input_token=win.token)
        visible = win.crop_visible(processed)   # drop the halo rows (decimated)

        if visible.size == 0:
            return

        # ── Presentation: clip vmax, colormap, extent ───────────────────────
        from sbp_studio.core.constants import CMAPS, DEFAULT_CLIP_PCT
        clip = float(disp.get("clip", DEFAULT_CLIP_PCT))
        vmax = self._estimate_vmax(visible, clip)
        # Amplitude range: diverging maps [−vmax, vmax] (signed, colormap centred
        # on zero); sequential maps [0, vmax] (|amp|, historical default).
        vmin = -vmax if disp.get("amp_range") == "diverging" else 0.0

        cmap_name = CMAPS.get(disp.get("cmap", "Viridis"), "viridis")
        if disp.get("inv_cmap"):
            cmap_name += "_r"

        c1_idx = min(win.c1 - 1, n_traces - 1)
        dist0 = float(dist_km[win.c0])
        dist1 = float(dist_km[c1_idx])
        t_top = t0_full + win.s0 * dt_ms
        t_bot = t0_full + win.s1 * dt_ms

        # The decimated visible band is a row-slice of the pipeline's contiguous
        # output → already C-contiguous float32, so this is a no-op (no full
        # copy). Phase 2 introduces no NaNs (align is off), so we skip the
        # expensive nan_to_num scan entirely.
        arr = np.ascontiguousarray(visible, dtype=np.float32)
        self.view.set_colormap(cmap_name, vmax, vmin=vmin)
        self.view.show_preview(arr, dist0, dist1, t_top, t_bot,
                               vmax=vmax, vmin=vmin, fit=fit)

        # Emit after show_preview so the ViewBox is already fitted (fit=True):
        # the autoRange() inside show_preview fires sigRangeChanged, but
        # _emit_visible_traces bails early while _dist_km is None.  Setting
        # _dist_km here guarantees the first emission uses the correct,
        # already-fitted range — fixing the map regression on initial load.
        self.view.set_distance_axis(dist_km)

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
            # Full-array trace index of each visible (possibly col-decimated) column
            # → exact km position from dist_km (traces are not evenly spaced).
            col_idx = win.c0 + np.arange(n_vis) * max(1, win.col_stride)
            col_idx = np.clip(col_idx, 0, n_traces - 1)
            x_centers = np.asarray(dist_km[col_idx], dtype=float)
            amps = visible
            if n_vis > budget:                       # decimate to the budget
                sel = np.linspace(0, n_vis - 1, budget).astype(int)
                amps = np.ascontiguousarray(visible[:, sel])
                x_centers = x_centers[sel]

            show_raster = bool(disp.get("show_raster", True))
            va_fill     = bool(disp.get("va_fill", True))

            # Auto-suppress VA when the drawn trace count exceeds the threshold:
            # lobes are too thin to read and arrayToQPath on millions of polygon
            # points would block the GUI thread for tens of ms per settle.
            effective_va = va_fill and (amps.shape[1] <= VA_TRACE_THRESHOLD)

            # Geometry key: window bounds + vmax + effective_va determine the
            # shape of the wiggle path. Colormap, FIX-mark, and show_raster
            # changes pay zero geometry cost (only img visibility is updated).
            wiggle_key = (win.token, vmax, effective_va)
            if wiggle_key == self._wiggle_state_key:
                self.view.img.setVisible(show_raster)
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
                        va_fill=effective_va)
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

        xs_va_T = np.empty((n, rows_va + 3), dtype=np.float32)
        xs_va_T[:, 0]              = x_centers              # centre-line anchor at top
        xs_va_T[:, 1:rows_va + 1]  = env_va.T              # positive envelope
        xs_va_T[:, rows_va + 1]    = x_centers             # centre-line anchor at bottom
        xs_va_T[:, rows_va + 2]    = np.nan                # trace separator
        xs_va_flat = xs_va_T.ravel()                       # zero-copy C-order view

        yy = np.empty(rows_va + 3, dtype=np.float32)
        yy[0]              = t_va[0]
        yy[1:rows_va + 1]  = t_va
        yy[rows_va + 1]    = t_va[-1]
        yy[rows_va + 2]    = np.nan
        ys_va_flat = np.tile(yy, n)

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
