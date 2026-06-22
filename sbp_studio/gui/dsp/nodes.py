"""
nodes.py — DSP pipeline node base class and concrete nodes.

A *node* is one composable DSP stage in a user-ordered pipeline. Nodes are the
GUI-side wrappers around the CORE math: a node NEVER implements DSP itself, it
only holds parameters and calls a ``sbp_studio.core`` function. This keeps the
strict GUI/core separation — the core stays headless and authoritative.

Design
------
``ParamSpec``  — declarative descriptor for one NUMERIC parameter (slider+spin).
``ChoiceSpec`` — declarative descriptor for one CATEGORICAL parameter (combo).
``DSPContext`` — read-only acquisition metadata a node needs (dt_us, delays …).
``DSPNode``    — abstract base. Subclasses set ``KEY``, ``DISPLAY``, ``SPECS``
                 and implement :meth:`apply`. ``signature()`` (identity + params)
                 drives the pipeline's memoization cache, and
                 ``time_halo_samples()``/``trace_halo()`` let the ViewBox preview
                 extract a slightly larger window so window-based filters have
                 correct edges.

Migrated stages (Phase 3): Predictive Deconvolution, Bandpass, Preset/Attribute
(incl. Envelope), TVG, AGC, Delay Alignment — each wraps the matching core
``apply_*`` function (the single source of truth).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ── Parameter descriptors ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParamSpec:
    """Declarative spec for one NUMERIC parameter (drives a slider+spinbox)."""
    name:     str                 # programmatic key (matches DSPNode.params key)
    label:    str                 # English source label (localised via node_i18n)
    lo:       float
    hi:       float
    default:  float
    step:     float = 1.0
    decimals: int   = 0           # 0 → integer spinbox/slider
    unit:     str   = ""          # e.g. "ms", "Hz" (display only)


@dataclass(frozen=True)
class ChoiceSpec:
    """Declarative spec for one CATEGORICAL parameter (drives a combo box)."""
    name:    str
    label:   str
    choices: Tuple[Tuple[str, str], ...]   # ((value, display), …)
    default: str


Spec = Union[ParamSpec, ChoiceSpec]


# ── Acquisition context ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DSPContext:
    """Read-only metadata a node may need. Built from a SegyProfile/ProfileChain.

    ``delays``/``min_delay`` are carried for the geometric Delay-Alignment node;
    the filter nodes use only ``dt_us``.
    """
    dt_us:     int
    ns:        int
    n_traces:  int
    delays:    Optional[np.ndarray] = None
    min_delay: float = 0.0
    delay_ms:  float = 0.0

    @classmethod
    def from_source(cls, obj: Any) -> "DSPContext":
        return cls(
            dt_us=int(obj.dt_us), ns=int(obj.ns), n_traces=int(obj.n_traces),
            delays=getattr(obj, "delays", None),
            min_delay=float(getattr(obj, "min_delay", 0.0)),
            delay_ms=float(getattr(obj, "delay_ms", 0.0)))


# ── Node base class ─────────────────────────────────────────────────────────────

class DSPNode(ABC):
    """One DSP stage. Holds params, wraps a core function. Never does DSP itself."""

    KEY:     str = ""             # stable identifier (used in cache signature)
    DISPLAY: str = ""             # English source name (localised via node_i18n)
    SPECS:   Tuple[Spec, ...] = ()

    # PRE-CROP nodes need the FULL trace to be correct (e.g. water-column mute
    # picks the seabed from the whole trace). In the live preview the controller
    # applies them to the full base BEFORE ViewBox cropping (cached), then runs
    # the remaining nodes on the cropped window. They remain ordinary reorderable
    # nodes; the EXPORT runs the full pipeline in list order over the full array.
    PRECROP: bool = False

    # FULL-RESOLUTION nodes must see the TRUE, un-decimated trace spacing of the
    # viewport (e.g. the F-K dip filter, whose wavenumber axis is meaningless on
    # column-decimated data). When any active window node sets this, the preview
    # controller extracts the viewport at full resolution (up to a high safety
    # cap) and runs the pipeline there, THEN hands the result to the
    # decimation/render layer — see PreviewController._refresh.
    NEEDS_FULL_RES: bool = False

    def __init__(self, params: Dict[str, Any] | None = None,
                 enabled: bool = True) -> None:
        self.params: Dict[str, Any] = {s.name: s.default for s in self.SPECS}
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.params})
        # Mute / bypass flag (UI checkbox per node). A disabled node stays in the
        # list (so the user can audition a filter without deleting it) but is
        # filtered OUT of the executed chain — see PipelinePanel.active_nodes(),
        # which both the live preview and the export read. NOT part of
        # signature(): muting changes the active node SET, which already forces
        # a recompute, so the cache key needs no enabled bit.
        self.enabled: bool = bool(enabled)

    # ── Identity / memoization ──────────────────────────────────────────────
    def signature(self) -> tuple:
        """Hashable identity: node key + sorted params. Drives the cache key."""
        return (self.KEY, tuple(sorted(self.params.items())))

    # ── Halos for ViewBox-limited preview ───────────────────────────────────
    def time_halo_samples(self, ctx: DSPContext) -> int:
        """Extra samples above/below the visible window so window/transient
        filters have correct edges (cropped after apply). Default 0."""
        return 0

    def trace_halo(self, ctx: DSPContext) -> int:
        """Extra traces left/right for spatial (cross-trace) filters. Default 0."""
        return 0

    # ── The actual DSP (delegates to core) ──────────────────────────────────
    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        """Sanitize non-finite input, then delegate to the node's DSP.

        A single NaN/Inf (corrupt SEG-Y, or the export's NaN ``fill_value``
        on exposed alignment gaps) would otherwise poison every node — most
        catastrophically the F-K filter, whose global 2-D FFT spreads one bad
        sample across the entire output. Every node funnels through here
        (Pipeline.process and the full-array export both call ``node.apply``),
        so this is the one chokepoint that shields the whole chain.
        """
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return self._apply(data, ctx)

    @abstractmethod
    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        """Return a NEW array; never mutate ``data``. Must call core DSP."""
        raise NotImplementedError


# ── Concrete nodes ──────────────────────────────────────────────────────────────

class PredictiveDeconNode(DSPNode):
    """Predictive (Wiener-Levinson) deconvolution — wraps ``core.apply_predictive_decon``."""

    KEY     = "decon"
    DISPLAY = "Predictive Deconvolution"
    SPECS   = (
        ParamSpec("op_ms",     "Operator length",   1.0, 50.0, 10.0, 1.0, 0, "ms"),
        ParamSpec("gap_ms",    "Prediction gap",    0.1, 20.0,  2.0, 0.1, 1, "ms"),
        ParamSpec("white_pct", "Pre-whitening",     0.1, 10.0,  1.0, 0.1, 1, "%"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        dt_ms = ctx.dt_us / 1000.0
        return int((self.params["op_ms"] + self.params["gap_ms"]) / dt_ms) + 1

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_predictive_decon
        return apply_predictive_decon(
            data, ctx.dt_us, self.params["op_ms"],
            self.params["gap_ms"], self.params["white_pct"])


class BandpassNode(DSPNode):
    """Butterworth bandpass (4th-order SOS) — wraps ``core.apply_bandpass``."""

    KEY     = "bandpass"
    DISPLAY = "Bandpass Filter"
    SPECS   = (
        ParamSpec("flo", "F low",  500.0, 15000.0, 2000.0, 100.0, 0, "Hz"),
        ParamSpec("fhi", "F high", 500.0, 15000.0, 7000.0, 100.0, 0, "Hz"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        # A few transient lengths of the lowest passband frequency.
        fs = 1e6 / ctx.dt_us
        flo = max(10.0, self.params["flo"])
        return int(min(1024, max(64, 3.0 * fs / flo)))

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_bandpass
        return apply_bandpass(data, self.params["flo"], self.params["fhi"], ctx.dt_us)


def _preset_choices() -> Tuple[Tuple[str, str], ...]:
    """(key, display) pairs for every preset except 'none' (domain data, not tr'd)."""
    from sbp_studio.core.constants import FILTER_PRESETS
    return tuple((key, disp) for disp, key in FILTER_PRESETS.items() if key != "none")


class PresetNode(DSPNode):
    """Filter preset / seismic attribute (incl. Envelope) — wraps ``core.apply_filter_preset``."""

    KEY     = "preset"
    DISPLAY = "Filter Preset / Attribute"
    SPECS   = (
        ChoiceSpec("preset", "Type", _preset_choices(), default="envelope"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        # Small-kernel presets need a few samples; Hilbert attributes (envelope,
        # phase, freq) are FFT-global so a finite halo only reduces edge ringing
        # (exact when the whole trace is visible / zoomed out).
        return 128

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_filter_preset
        return apply_filter_preset(data, self.params["preset"], ctx.dt_us)


class TVGNode(DSPNode):
    """Topography-aware ("Smart") time-variant exponential gain — wraps
    ``core.apply_tvg``. The gain ramp starts at each trace's OWN picked
    water-bottom time (same energy-threshold pick as WaterMuteNode), not a
    global t=0; noisy/dead traces where the pick fails fall back to the
    legacy global ramp automatically (see apply_tvg's docstring)."""

    KEY     = "tvg"
    DISPLAY = "TVG (Time-Variant Gain)"
    PRECROP = True                # seabed pick needs the full trace → run pre-crop
    SPECS   = (
        ParamSpec("alpha", "Attenuation coef. alpha", 0.0, 150.0, 15.0, 1.0, 0),
        ParamSpec("threshold_pct", "Seabed threshold", 0.0, 100.0, 30.0, 1.0, 0, "%"),
    )

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_tvg
        return apply_tvg(data, self.params["alpha"], ctx.dt_us, self.params["threshold_pct"])


class AGCNode(DSPNode):
    """Automatic Gain Control — wraps ``core.apply_agc``."""

    KEY     = "agc"
    DISPLAY = "AGC (Automatic Gain Control)"
    SPECS   = (
        ParamSpec("win_ms", "Window", 5.0, 200.0, 20.0, 1.0, 0, "ms"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        win_s = max(3, int(self.params["win_ms"] / (ctx.dt_us / 1000.0)))
        return win_s // 2 + 1

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_agc
        return apply_agc(data, self.params["win_ms"], ctx.dt_us)


class LogCompressionNode(DSPNode):
    """Seismic HDR — phase-preserving log compression — wraps ``core.apply_log_compression``."""

    KEY     = "log_compress"
    DISPLAY = "Log Compression (Seismic HDR)"
    SPECS   = (
        ParamSpec("k", "Strength (k)", 1.0, 100.0, 10.0, 1.0, 0),
    )

    # Pointwise rescale → no halo needed.
    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_log_compression
        return apply_log_compression(data, self.params["k"])


class CLAHENode(DSPNode):
    """CLAHE (Adaptive Local Contrast) — 2-D, tile-based "Seismic HDR" —
    wraps ``core.apply_clahe``. Unlike LogCompressionNode's single global
    curve, this equalises contrast independently in local tiles. Requires
    ``opencv-python-headless`` (cv2), lazily imported by the core function.
    """

    KEY     = "clahe"
    DISPLAY = "CLAHE (Adaptive Local Contrast)"
    SPECS   = (
        ParamSpec("clip_limit", "Clip Limit",    1.0, 40.0, 2.0, 0.5, 1),
        ParamSpec("tile_grid",  "Tile Grid Size", 2.0, 64.0, 8.0, 1.0, 0),
    )

    # Tile-relative normalisation (own peak amplitude, own tile grid over
    # whatever window is fed in) → no halo, same contract as LogCompression.
    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_clahe
        return apply_clahe(data, self.params["clip_limit"], int(self.params["tile_grid"]))


class DespikeNode(DSPNode):
    """Impulsive-noise (spike) removal — rolling-median/MAD — wraps
    ``core.apply_despike``. Replaces samples that exceed a robust local
    threshold with the local median; everything else passes through.

    Window stays NARROW by design: true impulsive noise is 1-2 samples wide,
    while a real reflector wavelet spans many samples with smooth flanks. A
    window much wider than the noise (but still narrow relative to a real
    wavelet) keeps a genuine peak from looking like an outlier within its
    own window — too wide a window re-introduces false positives on real
    reflectors. Default threshold (6x the local robust std) sits comfortably
    above a clean wavelet's own peak-vs-window ratio (~4-5x) while still
    catching genuine spikes (typically 1-2 orders of magnitude above that)."""

    KEY     = "despike"
    DISPLAY = "Despike (Impulsive Noise Removal)"
    SPECS   = (
        ParamSpec("window_ms", "Window Size", 0.5, 20.0, 2.0, 0.5, 1, "ms"),
        ParamSpec("threshold", "Threshold",    2.0, 15.0, 6.0, 0.5, 1),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        win_s = max(3, int(self.params["window_ms"] / (ctx.dt_us / 1000.0)))
        return win_s // 2 + 1

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_despike
        win_s = max(3, int(round(self.params["window_ms"] / (ctx.dt_us / 1000.0))))
        return apply_despike(data, win_s, self.params["threshold"])


class SpectralWhiteningNode(DSPNode):
    """Spectral whitening (resolution enhancement) — wraps ``core.apply_spectral_whitening``.

    Flattens the amplitude spectrum within [flo, fhi] while preserving phase
    exactly (zero-phase operation).  Typical SBP workflow: place after Bandpass
    to sharpen reflectors; use the Spectrum tab to judge before/after.
    """

    KEY     = "whiten"
    DISPLAY = "Spectral Whitening"
    SPECS   = (
        ParamSpec("flo",       "F low",         10.0, 10000.0, 1000.0, 100.0, 0, "Hz"),
        ParamSpec("fhi",       "F high",        100.0, 15000.0, 8000.0, 100.0, 0, "Hz"),
        ParamSpec("smooth_hz", "Smooth window",  10.0,  2000.0,  300.0,  10.0, 0, "Hz"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        # The rfft spans the entire (windowed) trace; a halo reduces spectral
        # ringing at the window edges — same rationale as PresetNode (Hilbert).
        return 128

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_spectral_whitening
        return apply_spectral_whitening(
            data, ctx.dt_us,
            self.params["flo"],
            self.params["fhi"],
            self.params["smooth_hz"])


class WaterMuteNode(DSPNode):
    """Water-column mute (mute above the seabed) — wraps ``core.apply_water_mute``."""

    KEY     = "water_mute"
    DISPLAY = "Water Column Mute"
    PRECROP = True                # seabed pick needs the full trace → run pre-crop
    SPECS   = (
        ParamSpec("threshold_pct", "Threshold", 1.0, 100.0, 30.0, 1.0, 0, "%"),
        ParamSpec("margin_ms",     "Margin",    0.0, 100.0,  5.0, 1.0, 0, "ms"),
    )

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_water_mute
        return apply_water_mute(
            data, self.params["threshold_pct"], self.params["margin_ms"], ctx.dt_us)


class SwellFilterNode(DSPNode):
    """Swell / heave correction (cross-correlation statics) — wraps ``core.apply_swell_filter``."""

    KEY     = "swell"
    DISPLAY = "Swell Filter / Heave Correction"
    SPECS   = (
        ParamSpec("window_traces", "Trace window", 3.0, 201.0, 21.0, 2.0, 0, "tr"),
        ParamSpec("max_shift_ms",  "Max shift",    1.0,  50.0, 10.0, 1.0, 0, "ms"),
    )

    def trace_halo(self, ctx: DSPContext) -> int:
        # Neighbours needed for the smooth rolling-mean reference at the edges.
        return int(self.params["window_traces"]) // 2

    def time_halo_samples(self, ctx: DSPContext) -> int:
        # Margin for the static roll at the window's top/bottom.
        return int(self.params["max_shift_ms"] / (ctx.dt_us / 1000.0)) + 2

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_swell_filter
        return apply_swell_filter(
            data, int(self.params["window_traces"]),
            self.params["max_shift_ms"], ctx.dt_us)


class FKFilterNode(DSPNode):
    """F-K (frequency–wavenumber) 2-D dip filter — wraps ``core.apply_fk_filter``.

    Rejects a fan of coherently DIPPING events (side-echoes, diffractions,
    towfish/cable noise) by apparent slope, leaving flat reflectors intact.
    NEEDS_FULL_RES: the math runs on the TRUE un-decimated viewport trace
    spacing (decimating columns would corrupt the wavenumber axis)."""

    KEY            = "fk"
    DISPLAY        = "F-K Dip Filter"
    NEEDS_FULL_RES = True
    SPECS = (
        ParamSpec("dip",   "Reject dip",     -5.0, 5.0, 1.0, 0.1, 1, "ms/tr"),
        ParamSpec("width", "Fan half-width",  0.1, 5.0, 0.5, 0.1, 1, "ms/tr"),
        ChoiceSpec("mode", "Mode",
                   (("reject_both", "Reject ± dips"),
                    ("reject_one",  "Reject one sign"),
                    ("pass",        "Keep fan only")),
                   default="reject_both"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        return 64        # soften the 2-D FFT wrap-around at the window's top/bottom

    def trace_halo(self, ctx: DSPContext) -> int:
        return 16        # extra traces left/right reduce the F-K horizontal wrap

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_fk_filter
        return apply_fk_filter(data, ctx.dt_us, self.params["dip"],
                               self.params["width"], self.params["mode"])


class MultipleSuppressionNode(DSPNode):
    """Seabed (water-bottom) multiple suppression — wraps
    ``core.apply_multiple_suppression``.

    Predictive adaptive subtraction at the seabed period; critical for shallow
    SBP where the first water-bottom multiple masks the sub-bottom. PRECROP:
    the per-trace period pick + gain estimate need the whole trace (the multiple
    sits at ~2× the seabed two-way time, often below the visible window)."""

    KEY     = "demultiple"
    DISPLAY = "Seabed Multiple Suppression"
    PRECROP = True
    SPECS = (
        ParamSpec("threshold_pct", "Seabed threshold", 1.0, 100.0, 30.0, 1.0, 0, "%"),
        ParamSpec("period_ms",     "Period (0=auto)",   0.0, 500.0,  0.0, 1.0, 0, "ms"),
        ParamSpec("max_gain",      "Max gain",          0.0,   2.0,  1.0, 0.05, 2),
    )

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_multiple_suppression
        return apply_multiple_suppression(
            data, ctx.dt_us, self.params["threshold_pct"],
            self.params["period_ms"], self.params["max_gain"])


class NotchNode(DSPNode):
    """Surgical band-stop (IIR notch) filter — wraps ``core.apply_notch``.

    Removes a single narrow interference frequency (electrical resonance,
    tow-cable strum) with a zero-phase ``scipy.signal.iirnotch`` + filtfilt."""

    KEY     = "notch"
    DISPLAY = "Notch Filter"
    SPECS = (
        ParamSpec("freq", "Notch frequency", 50.0, 15000.0, 1000.0, 10.0, 0, "Hz"),
        ParamSpec("q",    "Q factor",         1.0,   100.0,   30.0,  1.0, 0),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        # filtfilt transient ~ a few cycles of the notch frequency.
        fs = 1e6 / ctx.dt_us
        freq = max(1.0, self.params["freq"])
        return int(min(1024, max(64, 3.0 * fs / freq)))

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_notch
        return apply_notch(data, ctx.dt_us, self.params["freq"], self.params["q"])


# NOTE: Delay alignment is NOT a pipeline node — it is a static geometry
# correction (the ``align`` checkbox in the Geometry & Presentation panel). The
# preview controller and the export job apply ``core.apply_delay_alignment`` to
# the base array BEFORE the dynamic node pipeline. A vertical per-trace shift is
# a one-off geometry fix, not a reorderable DSP filter.


# ── Registry (the Add menu reads this; order = a sensible default DSP order) ─────

NODE_REGISTRY: List[type[DSPNode]] = [
    DespikeNode,             # impulsive-noise cleanup first, before anything else
    SwellFilterNode,
    FKFilterNode,            # 2-D dip reject (spatial, full-res viewport)
    WaterMuteNode,
    MultipleSuppressionNode,  # seabed de-multiple (PRECROP, needs full trace)
    PredictiveDeconNode,
    BandpassNode,
    NotchNode,               # surgical band-stop alongside the bandpass
    SpectralWhiteningNode,   # sharpen after filtering; before attribute/gain
    PresetNode,
    TVGNode,
    AGCNode,
    LogCompressionNode,
    CLAHENode,
]


def make_node(key: str, params: Dict[str, Any] | None = None,
              enabled: bool = True) -> DSPNode:
    """Instantiate a node by its KEY (used when restoring a saved pipeline)."""
    for cls in NODE_REGISTRY:
        if cls.KEY == key:
            return cls(params, enabled=enabled)
    raise KeyError(f"Unknown DSP node key: {key!r}")
