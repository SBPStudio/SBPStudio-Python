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
    # Optional per-choice hover tooltip (value -> description), e.g.
    # PresetNode's "Type" combo reusing core.constants.FILTER_DESCRIPTIONS.
    # None for ChoiceSpecs with no such domain-data description (e.g.
    # FKFilterNode's "mode") — the UI simply skips setting an item tooltip.
    tooltips: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class BoolSpec:
    """Declarative spec for one BOOLEAN parameter (drives a checkbox)."""
    name:    str
    label:   str
    default: bool = True


Spec = Union[ParamSpec, ChoiceSpec, BoolSpec]


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
    TOOLTIP: str = ""             # English source 1-2 sentence geophysical
                                   # explanation (localised via node_i18n.tr_tooltip),
                                   # shown on hover in the "Add module" dropdown.
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

class TraceMixingNode(DSPNode):
    """Trace Mixing (Horizontal Spatial Smoothing) — wraps
    ``core.apply_trace_mixing``.

    Rolling average ACROSS traces (not down a trace): a continuous reflector
    has coherent amplitude/phase between neighbours and survives via
    constructive interference, while incoherent salt-and-pepper noise has no
    such correlation and is attenuated via destructive interference —
    countering the noise a downstream gain stage (AGC, TVG) would otherwise
    amplify into visible speckle in deep, low-SNR sections.

    ``window_size`` MUST be odd (centred average, no lateral event shift);
    enforced again at the core-function level even if the UI slider lets an
    even value slip through.
    """

    KEY     = "trace_mix"
    DISPLAY = "Trace Mixing (Horizontal Smoothing)"
    TOOLTIP = ("Averages each sample with its horizontal neighbour traces. "
              "Coherent reflectors survive via constructive interference; "
              "incoherent random noise is attenuated via destructive interference.")
    SPECS   = (
        ParamSpec("window_size", "Traces to mix", 3.0, 51.0, 3.0, 2.0, 0, "tr"),
    )

    def trace_halo(self, ctx: DSPContext) -> int:
        # Neighbours needed on each side so the rolling average is correct
        # right up to the visible window's edges (same rationale as
        # SwellFilterNode's trace_halo).
        return int(self.params["window_size"]) // 2

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_trace_mixing
        return apply_trace_mixing(data, int(self.params["window_size"]))


class MedianFilterNode(DSPNode):
    """Median Filter (Edge-Preserving) — wraps ``core.apply_median_filter``.

    The edge-preserving alternative to TraceMixingNode: each sample becomes
    the MEDIAN of itself and its horizontal neighbours instead of their MEAN.
    A median is always one of the actual input values, never an interpolated
    blend, so an isolated noise spike is rejected outright rather than
    smeared across neighbours — and unlike the mean, it does not "watercolor"
    blur the sharp lateral edge of a fault or a steeply-dipping reflector.

    ``window_size`` MUST be odd (centred window, no lateral event shift);
    enforced again at the core-function level even if the UI slider lets an
    even value slip through.
    """

    KEY     = "median_filter"
    DISPLAY = "Median Filter (Edge-Preserving)"
    TOOLTIP = ("Replaces each sample with the median of its horizontal neighbours, "
              "rejecting isolated noise spikes outright instead of blending them — "
              "preserves sharp fault/reflector edges that a mean filter would blur.")
    SPECS   = (
        ParamSpec("window_size", "Traces to evaluate", 3.0, 51.0, 3.0, 2.0, 0, "tr"),
    )

    def trace_halo(self, ctx: DSPContext) -> int:
        # Same rationale as TraceMixingNode.trace_halo: neighbours needed on
        # each side so the median is correct right up to the visible
        # window's edges.
        return int(self.params["window_size"]) // 2

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_median_filter
        return apply_median_filter(data, int(self.params["window_size"]))


class SVDFilterNode(DSPNode):
    """SVD Filter / Karhunen-Loeve Transform — wraps ``core.apply_svd_filter``.

    A fundamentally different (and stronger) noise-attenuation mechanism
    than TraceMixingNode/MedianFilterNode's local neighbour-window: rebuilds
    the matrix from only its top ``num_components`` singular values/vectors.
    A coherent reflector concentrates its energy into a handful of dominant
    components; dense, spatially incoherent thermal/random noise spreads
    thinly across all of them, so truncation keeps the former and discards
    the latter — useful when Trace Mixing/Median Filter are too conservative
    against widespread noise in deep, low-SNR sections.

    trace_halo is deliberately GENEROUS (not tied to num_components): SVD
    decomposes the WHOLE matrix chunk it is given, so the live preview's
    ViewBox needs enough surrounding structural context on each side for the
    decomposition (and thus the visible result) to be stable — too small a
    halo would make the filtered output near the window's edges look
    different every time the user scrolls, even though nothing in the
    underlying data changed.
    """

    KEY     = "svd_filter"
    DISPLAY = "SVD Filter (Eigenvalues)"
    TOOLTIP = ("Rebuilds the section from only its top singular components. Coherent "
              "reflectors concentrate their energy into a few components; dense "
              "random noise spreads thinly across all of them and is discarded.")
    SPECS   = (
        ParamSpec("num_components", "Principal Components", 2.0, 100.0, 10.0, 1.0, 0),
    )

    def trace_halo(self, ctx: DSPContext) -> int:
        return 50

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_svd_filter
        return apply_svd_filter(data, int(self.params["num_components"]))


class BilateralFilterNode(DSPNode):
    """Bilateral Filter (Edge-Preserving Spatial Smoothing) — wraps
    ``core.apply_bilateral_filter``.

    A tunable middle ground between TraceMixingNode (smooths everywhere,
    blurs edges) and MedianFilterNode (preserves edges, but a hard on/off
    decision with no weighted blend): each neighbour trace's contribution is
    weighted by BOTH spatial distance and amplitude similarity, so noise in
    an otherwise-flat region is averaged down same as Trace Mixing, while a
    neighbour on the far side of a fault/sharp reflector edge — whose
    amplitude differs sharply — is automatically excluded from the average.

    ``sigma_color`` ("Tolerancia de Amplitud") is unitless/relative (a
    multiple of the chunk's own std-dev, not a raw amplitude), so the user
    tunes "how different is too different" without knowing the data's
    absolute amplitude scale.

    ``window_size`` MUST be odd (centred window, no lateral event shift);
    enforced again at the core-function level even if the UI slider lets an
    even value slip through.
    """

    KEY     = "bilateral_filter"
    DISPLAY = "Bilateral Filter (Smart Smoothing)"
    TOOLTIP = ("Averages horizontal neighbour traces only where their amplitude is "
              "similar — smooths random noise in flat zones while excluding "
              "neighbours across a fault or steep edge, so the edge stays sharp.")
    SPECS   = (
        ParamSpec("window_size", "Traces to evaluate", 3.0, 51.0, 5.0, 2.0, 0, "tr"),
        ParamSpec("sigma_color", "Amplitude Tolerance", 0.05, 3.0, 0.5, 0.05, 2),
    )

    def trace_halo(self, ctx: DSPContext) -> int:
        # Same rationale as TraceMixingNode/MedianFilterNode.trace_halo:
        # neighbours needed on each side so the weighted average is correct
        # right up to the visible window's edges.
        return int(self.params["window_size"]) // 2

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_bilateral_filter
        return apply_bilateral_filter(
            data, int(self.params["window_size"]),
            sigma_color=self.params["sigma_color"])


class PredictiveDeconNode(DSPNode):
    """Predictive (Wiener-Levinson) deconvolution — wraps ``core.apply_predictive_decon``."""

    KEY     = "decon"
    DISPLAY = "Predictive Deconvolution"
    TOOLTIP = ("Estimates and removes the predictable (repetitive) part of each trace's "
              "waveform, such as reverberation, to sharpen the source pulse and "
              "improve vertical resolution.")
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
    TOOLTIP = ("Passes only frequencies between F-low and F-high, rejecting low-frequency "
              "swell/heave noise and high-frequency electrical/thermal noise outside "
              "the source's useful bandwidth.")
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


# Sentinel ``value`` marking a category-header entry inside
# PresetNode.SPECS' "Type" ChoiceSpec.choices (see _preset_choices). Never a
# real FILTER_PRESETS key, so it can never be matched by combo.findData(cur)
# / become the current selection — _ChoiceRow (pipeline_panel.py) checks for
# this exact value to render the row as a disabled, styled header instead of
# a selectable preset.
PRESET_HEADER_VALUE = "__preset_category_header__"

# Canonical grouping for BOTH PresetNode's dynamic "Type" combo (here) and
# ProcessingControls' static preset combo (gui/components/processing_controls.py,
# which imports this same tuple) — a standard marine-seismic workflow order:
# complex-trace attributes -> structural attributes -> 2D image filters ->
# frequency/smoothing. Each tuple is (English source category header, preset
# KEYs in the order they should appear within it) — the header is translated
# via node_i18n.tr_preset_category at combo-build time, not baked in here,
# since this module-level tuple (like _preset_choices()'s output) is only
# ever evaluated once, at PresetNode's class-definition/import time.
PRESET_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Complex Trace Attributes",
     ("envelope", "inst_phase", "inst_freq", "cos_phase")),
    ("Structural Attributes",
     ("similarity", "sobel_v", "laplacian")),
    ("2D Image Filters",
     ("highboost", "median5", "wiener7")),
    ("Frequency & Smoothing Filters",
     ("gauss1", "topas_narrow", "topas_wide", "topas_hires", "derivative", "integral")),
)


def _preset_choices() -> Tuple[Tuple[str, str], ...]:
    """(value, display) pairs for the "Type" combo: every preset except
    'none' (domain data, not tr'd), grouped under PRESET_CATEGORIES headers
    in the same logical order as ProcessingControls' static preset combo —
    see PRESET_HEADER_VALUE for how a header entry is encoded.
    """
    from sbp_studio.core.constants import FILTER_PRESETS
    key_to_display = {key: disp for disp, key in FILTER_PRESETS.items()}
    pairs: List[Tuple[str, str]] = []
    categorized: set = set()
    for header_en, keys in PRESET_CATEGORIES:
        pairs.append((PRESET_HEADER_VALUE, header_en))
        for key in keys:
            categorized.add(key)
            disp = key_to_display.get(key)
            if disp is not None:
                pairs.append((key, disp))

    # Defensive catch-all: a preset NOT listed in PRESET_CATEGORIES (e.g. a
    # future addition nobody re-categorised yet) still appears here rather
    # than silently vanishing from the combo — FILTER_PRESETS stays the
    # single source of truth for "what's selectable" (same pattern as the
    # "Add module" menu's and static preset combo's own leftover handling).
    leftover = [k for k in key_to_display if k != "none" and k not in categorized]
    if leftover:
        pairs.append((PRESET_HEADER_VALUE, "Other"))
        for key in leftover:
            pairs.append((key, key_to_display[key]))
    return tuple(pairs)


def _preset_tooltips() -> Dict[str, str]:
    """value -> hover description for every preset choice, reusing the SAME
    domain-data dict the legacy ProcessingControls "FILTROS PREESTABLECIDOS"
    combo already shows (FILTER_DESCRIPTIONS) — one description, two combos."""
    from sbp_studio.core.constants import FILTER_DESCRIPTIONS
    return dict(FILTER_DESCRIPTIONS)


class PresetNode(DSPNode):
    """Filter preset / seismic attribute (incl. Envelope) — wraps ``core.apply_filter_preset``."""

    KEY     = "preset"
    DISPLAY = "Filter Preset / Attribute"
    TOOLTIP = ("Applies a named seismic attribute transform (e.g. envelope, instantaneous "
              "phase/frequency via the Hilbert transform) used to highlight specific "
              "geological features instead of the raw amplitude.")
    SPECS   = (
        ChoiceSpec("preset", "Type", _preset_choices(), default="envelope",
                  tooltips=_preset_tooltips()),
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
    TOOLTIP = ("Exponentially boosts amplitude with time-since-seabed to compensate "
              "for signal attenuation with depth, so weak deep reflectors become as "
              "visible as the strong shallow seabed return.")
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
    TOOLTIP = ("Statistically equalises amplitude using a sliding RMS window per trace, "
              "boosting weak zones and damping strong ones, regardless of where they "
              "occur — a data-driven gain, unlike TVG's fixed geometric curve.")
    SPECS   = (
        ParamSpec("win_ms", "Window", 5.0, 200.0, 20.0, 1.0, 0, "ms"),
    )

    def time_halo_samples(self, ctx: DSPContext) -> int:
        win_s = max(3, int(self.params["win_ms"] / (ctx.dt_us / 1000.0)))
        return win_s // 2 + 1

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_agc
        return apply_agc(data, self.params["win_ms"], ctx.dt_us)


class SphericalDivergenceNode(DSPNode):
    """Spherical Divergence Correction (True Amplitude Recovery) — wraps
    ``core.apply_spherical_divergence``.

    A DETERMINISTIC, physics-based gain curve (``t**exponent``) — the
    deterministic counterpart to AGC's purely STATISTICAL windowed-RMS gain
    just above. ``exponent=1.0`` is the theoretical pure geometric-spreading
    loss; values above 1.0 let the user empirically push further to also
    compensate for inelastic (absorption) attenuation, which spherical
    spreading alone does not model.

    ``reference_seabed`` (default True, the geophysically correct choice):
    references the gain to each trace's OWN picked seafloor (via
    ``core.pick_seabed``) instead of a single global clock shared by every
    trace — a deep-water trace's water column is no longer blown out just
    because a shallow-water trace's seafloor arrived early.

    PRECROP: the seabed pick (and, even in the ``reference_seabed=False``
    legacy mode, the meaning of "sample index from t=0 of the recording")
    both require the FULL trace, not whatever fragment happens to be in the
    live preview's ViewBox — same rationale as TVGNode/WaterMuteNode.
    """

    KEY     = "spherical_divergence"
    DISPLAY = "Spherical Divergence (True Amplitude)"
    TOOLTIP = ("Deterministic, physics-based gain (t^exponent) referenced to each "
              "trace's own picked seafloor, compensating for wavefront spreading "
              "loss without the artificial water-column boost of a global t=0 curve.")
    PRECROP = True
    SPECS   = (
        ParamSpec("exponent", "Falloff Exponent", 0.0, 5.0, 1.0, 0.1, 1),
        BoolSpec("reference_seabed", "Reference Seabed", True),
    )

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_spherical_divergence
        return apply_spherical_divergence(
            data, self.params["exponent"], self.params["reference_seabed"])


class TraceEqualizationNode(DSPNode):
    """Trace Equalization (RMS Balance) — wraps ``core.apply_trace_equalization``.

    Divides each trace by its own RMS amplitude so every trace carries
    comparable energy along the line. A SELECTABLE, order-sensitive
    counterpart to the mandatory load-time ``apply_dc_removal`` (which only
    zero-centres each trace — a precondition for a meaningful RMS, not an
    energy balance): a downstream gain (AGC) or local-contrast (CLAHE) stage
    amplifies whatever trace-to-trace imbalance still survives at that
    point in the chain, which is what produces vertical 'striping' in the
    water column. Placing this node before such a stage fixes the cause;
    after it re-balances whatever that stage produced — the user's choice,
    via where they drop it in their chain.

    PRECROP: RMS must reflect each trace's WHOLE energy, not just whatever
    happens to be visible in the current viewport — otherwise zooming/
    panning would change the normalization factor. No params, no halo:
    a single scalar-per-trace operation.
    """

    KEY     = "trace_eq"
    DISPLAY = "Trace Equalization (RMS Balance)"
    TOOLTIP = ("Divides each trace by its own RMS amplitude so every trace carries "
              "comparable energy along the line, correcting for source/receiver "
              "coupling variation before a downstream gain stage amplifies it.")
    PRECROP = True
    SPECS   = ()

    def _apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_trace_equalization
        return apply_trace_equalization(data)


class LogCompressionNode(DSPNode):
    """Seismic HDR — phase-preserving log compression — wraps ``core.apply_log_compression``."""

    KEY     = "log_compress"
    DISPLAY = "Log Compression (Seismic HDR)"
    TOOLTIP = ("Phase-preserving logarithmic rescale that compresses dynamic range, "
              "making weak reflectors visible alongside strong ones in the same "
              "display without clipping — like HDR tone-mapping for a photograph.")
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
    TOOLTIP = ("Equalises contrast independently within small local tiles instead of "
              "globally, revealing subtle structure in both low- and high-amplitude "
              "regions of the same section simultaneously.")
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
    TOOLTIP = ("Replaces samples that exceed a robust local amplitude threshold with "
              "the local median, removing 1-2 sample impulsive spikes (electrical "
              "transients, bad bits) while leaving genuine wavelet peaks untouched.")
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
    TOOLTIP = ("Flattens the amplitude spectrum within a chosen band while preserving "
              "phase exactly, sharpening reflectors and improving vertical resolution "
              "by recovering frequencies the source/medium attenuated unevenly.")
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
    TOOLTIP = ("Zeroes everything above each trace's own picked seafloor, removing the "
              "water column entirely so reverberation and direct-wave energy don't "
              "interfere with sub-bottom interpretation.")
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
    TOOLTIP = ("Removes vessel heave from sea-surface swell by aligning each trace to a "
              "smooth spatial reference via cross-correlation, flattening the wavy "
              "seafloor distortion that heave introduces into the section.")
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
    TOOLTIP        = ("Rejects coherently DIPPING events (side-echoes, diffractions, "
                      "cable/towfish noise) by their apparent slope in the "
                      "frequency-wavenumber domain, leaving flat reflectors untouched.")
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
    TOOLTIP = ("Predicts and adaptively subtracts the seabed (water-bottom) multiple "
              "reflection at roughly twice the seafloor's two-way time, which "
              "otherwise masks weaker, genuine sub-bottom reflectors beneath it.")
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
    TOOLTIP = ("Surgically removes a single narrow interference frequency (electrical "
              "resonance, tow-cable strum) with a zero-phase band-stop, leaving the "
              "rest of the spectrum untouched.")
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
    TraceMixingNode,         # spatial denoise before any frequency-domain filter
    MedianFilterNode,        # edge-preserving alternative to Trace Mixing
    BilateralFilterNode,     # weighted middle ground between mixing and median
    SVDFilterNode,           # stronger denoise for widespread/dense random noise
    PredictiveDeconNode,
    BandpassNode,
    NotchNode,               # surgical band-stop alongside the bandpass
    SpectralWhiteningNode,   # sharpen after filtering; before attribute/gain
    PresetNode,
    TraceEqualizationNode,   # balance per-trace energy before gain/contrast amplify it
    TVGNode,
    AGCNode,
    SphericalDivergenceNode,  # deterministic, physics-based gain alongside TVG/AGC
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
