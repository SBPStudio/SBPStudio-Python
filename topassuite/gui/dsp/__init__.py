"""
dsp — Dynamic, reorderable DSP node pipeline for the TOPAS Suite GUI.

GUI-side orchestration only: nodes wrap CORE math (``topassuite.core``) and the
:class:`Pipeline` memoizes per-node outputs for real-time, ViewBox-limited
preview. The core stays the single source of DSP truth.
"""
from .nodes import (
    AGCNode, BandpassNode, ChoiceSpec, DSPContext, DSPNode,
    NODE_REGISTRY, ParamSpec, PredictiveDeconNode, PresetNode, SwellFilterNode,
    TVGNode, WaterMuteNode, make_node,
)
from .node_i18n import tr_node, tr_param
from .pipeline import Pipeline, VisibleWindow, extract_visible_window
from .preview import PreviewController

__all__ = [
    "DSPNode", "DSPContext", "ParamSpec", "ChoiceSpec", "NODE_REGISTRY", "make_node",
    "AGCNode", "BandpassNode", "TVGNode", "PredictiveDeconNode", "PresetNode",
    "SwellFilterNode", "WaterMuteNode",
    "tr_node", "tr_param",
    "Pipeline", "VisibleWindow", "extract_visible_window",
    "PreviewController",
]
