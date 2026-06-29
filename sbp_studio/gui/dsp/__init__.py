"""
dsp — Dynamic, reorderable DSP node pipeline for the SBP Studio GUI.

GUI-side orchestration only: nodes wrap CORE math (``sbp_studio.core``) and the
:class:`Pipeline` memoizes per-node outputs for real-time, ViewBox-limited
preview. The core stays the single source of DSP truth.
"""
from .nodes import (
    AGCNode, BandpassNode, BilateralFilterNode, BoolSpec, ChoiceSpec, CLAHENode,
    DespikeNode, DSPContext, DSPNode, FKFilterNode, LogCompressionNode, MedianFilterNode,
    MultipleSuppressionNode, NotchNode, NODE_REGISTRY, ParamSpec, PRESET_CATEGORIES,
    PRESET_HEADER_VALUE, PredictiveDeconNode, PresetNode, SphericalDivergenceNode,
    SVDFilterNode, SwellFilterNode, TraceEqualizationNode, TraceMixingNode, TVGNode,
    WaterMuteNode, make_node,
)
from .export_filter import apply_pipeline_to_matrix, fits_in_memory
from .node_i18n import tr_node, tr_param, tr_preset_category, tr_tooltip
from .pipeline import Pipeline, VisibleWindow, extract_visible_window
from .preview import PreviewController

__all__ = [
    "DSPNode", "DSPContext", "ParamSpec", "ChoiceSpec", "BoolSpec", "NODE_REGISTRY", "make_node",
    "AGCNode", "BandpassNode", "TVGNode", "PredictiveDeconNode", "PresetNode",
    "SwellFilterNode", "WaterMuteNode", "LogCompressionNode", "CLAHENode", "DespikeNode",
    "FKFilterNode", "MultipleSuppressionNode", "NotchNode", "TraceEqualizationNode",
    "TraceMixingNode", "MedianFilterNode", "SVDFilterNode", "BilateralFilterNode",
    "SphericalDivergenceNode", "PRESET_CATEGORIES", "PRESET_HEADER_VALUE",
    "tr_node", "tr_param", "tr_tooltip", "tr_preset_category",
    "Pipeline", "VisibleWindow", "extract_visible_window",
    "PreviewController", "apply_pipeline_to_matrix", "fits_in_memory",
]
