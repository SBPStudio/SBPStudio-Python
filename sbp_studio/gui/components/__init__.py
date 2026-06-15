"""
Specialised visual components for the SBP Studio GUI.

Heavy/recurring graphical widgets, each isolated in its own class so drawing
logic stays out of the tab modules.
"""
from .export_dialog import ExportDialog
from .header_view import HeaderView
from .map_view import MapView
from .pipeline_panel import PipelinePanel
from .placeholder import PlaceholderView
from .processing_controls import ProcessingControls
from .seismic_view import SeismicView
from .spectrum_view import SpectrumView

__all__ = ["ExportDialog", "HeaderView", "MapView", "PipelinePanel",
           "PlaceholderView", "ProcessingControls", "SeismicView", "SpectrumView"]
