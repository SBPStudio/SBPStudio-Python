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

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params: Dict[str, Any] = {s.name: s.default for s in self.SPECS}
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.params})

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
    @abstractmethod
    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
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

    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
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

    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
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

    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_filter_preset
        return apply_filter_preset(data, self.params["preset"], ctx.dt_us)


class TVGNode(DSPNode):
    """Time-variant exponential gain — wraps ``core.apply_tvg``."""

    KEY     = "tvg"
    DISPLAY = "TVG (Time-Variant Gain)"
    SPECS   = (
        ParamSpec("alpha", "Attenuation coef. alpha", 0.0, 150.0, 15.0, 1.0, 0),
    )

    # Pointwise multiply → no halo needed.
    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_tvg
        return apply_tvg(data, self.params["alpha"], ctx.dt_us)


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

    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_agc
        return apply_agc(data, self.params["win_ms"], ctx.dt_us)


class WaterMuteNode(DSPNode):
    """Water-column mute (mute above the seabed) — wraps ``core.apply_water_mute``."""

    KEY     = "water_mute"
    DISPLAY = "Water Column Mute"
    PRECROP = True                # seabed pick needs the full trace → run pre-crop
    SPECS   = (
        ParamSpec("threshold_pct", "Threshold", 1.0, 100.0, 30.0, 1.0, 0, "%"),
        ParamSpec("margin_ms",     "Margin",    0.0, 100.0,  5.0, 1.0, 0, "ms"),
    )

    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
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

    def apply(self, data: np.ndarray, ctx: DSPContext) -> np.ndarray:
        from sbp_studio.core import apply_swell_filter
        return apply_swell_filter(
            data, int(self.params["window_traces"]),
            self.params["max_shift_ms"], ctx.dt_us)


# NOTE: Delay alignment is NOT a pipeline node — it is a static geometry
# correction (the ``align`` checkbox in the Geometry & Presentation panel). The
# preview controller and the export job apply ``core.apply_delay_alignment`` to
# the base array BEFORE the dynamic node pipeline. A vertical per-trace shift is
# a one-off geometry fix, not a reorderable DSP filter.


# ── Registry (the Add menu reads this; order = a sensible default DSP order) ─────

NODE_REGISTRY: List[type[DSPNode]] = [
    SwellFilterNode,
    WaterMuteNode,
    PredictiveDeconNode,
    BandpassNode,
    PresetNode,
    TVGNode,
    AGCNode,
]


def make_node(key: str, params: Dict[str, Any] | None = None) -> DSPNode:
    """Instantiate a node by its KEY (used when restoring a saved pipeline)."""
    for cls in NODE_REGISTRY:
        if cls.KEY == key:
            return cls(params)
    raise KeyError(f"Unknown DSP node key: {key!r}")
