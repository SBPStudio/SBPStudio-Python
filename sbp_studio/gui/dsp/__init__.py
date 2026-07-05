"""
dsp — Dynamic, reorderable DSP node pipeline for the SBP Studio GUI.

GUI-side orchestration only: nodes wrap CORE math (``sbp_studio.core``) and the
:class:`Pipeline` memoizes per-node outputs for real-time, ViewBox-limited
preview. The core stays the single source of DSP truth.

Import split (headless-worker contract)
----------------------------------------
The EAGER imports below (nodes / pipeline / export_filter) are PyQt6-free by
construction — module-level imports are stdlib + numpy + ``core`` only — so
``import sbp_studio.gui.dsp`` (and any submodule) stays Qt-free. The two
Qt-touching names — :class:`PreviewController` (QObject machinery) and the
``tr_*`` helpers (QCoreApplication.translate) — resolve LAZILY on first access
(PEP 562), which is what lets a spawned batch-export worker run the full DSP
node chain without importing Qt. GUI callers see no difference.
"""
from __future__ import annotations

from .nodes import (
    AB_AnchorNode,
    AGCNode, BandpassNode, BilateralFilterNode, BoolSpec, ChoiceSpec, CLAHENode,
    DespikeNode, DSPContext, DSPNode, FKFilterNode, LogCompressionNode, MedianFilterNode,
    MultipleSuppressionNode, NotchNode, NODE_REGISTRY, ParamSpec, PRESET_CATEGORIES,
    PRESET_HEADER_VALUE, PredictiveDeconNode, PresetNode, SphericalDivergenceNode,
    SVDFilterNode, SwellFilterNode, TraceEqualizationNode, TraceMixingNode, TVGNode,
    WaterMuteNode, make_node,
)
from .export_filter import apply_pipeline_to_matrix, fits_in_memory
from .pipeline import Pipeline, VisibleWindow, extract_visible_window

__all__ = [
    "DSPNode", "DSPContext", "ParamSpec", "ChoiceSpec", "BoolSpec", "NODE_REGISTRY", "make_node",
    "AB_AnchorNode",
    "AGCNode", "BandpassNode", "TVGNode", "PredictiveDeconNode", "PresetNode",
    "SwellFilterNode", "WaterMuteNode", "LogCompressionNode", "CLAHENode", "DespikeNode",
    "FKFilterNode", "MultipleSuppressionNode", "NotchNode", "TraceEqualizationNode",
    "TraceMixingNode", "MedianFilterNode", "SVDFilterNode", "BilateralFilterNode",
    "SphericalDivergenceNode", "PRESET_CATEGORIES", "PRESET_HEADER_VALUE",
    "tr_node", "tr_param", "tr_tooltip", "tr_preset_category",
    "Pipeline", "VisibleWindow", "extract_visible_window",
    "PreviewController", "apply_pipeline_to_matrix", "fits_in_memory",
]

_LAZY_I18N = ("tr_node", "tr_param", "tr_tooltip", "tr_preset_category")


def __getattr__(name: str):
    if name == "PreviewController":
        from .preview import PreviewController
        return PreviewController
    if name in _LAZY_I18N:
        from . import node_i18n
        return getattr(node_i18n, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
