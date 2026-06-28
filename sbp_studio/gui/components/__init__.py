"""
Specialised visual components for the SBP Studio GUI.

Heavy/recurring graphical widgets, each isolated in its own class so drawing
logic stays out of the tab modules.
"""
from .crs_selector import CRSAdvancedSearchDialog, CRSSelectorDialog
from .export_dialog import ExportDialog
from .header_view import HeaderView
from .map_view import MapView
from .metadata_inspector import MetadataInspectorDialog
from .picking_export_dialog import PickingExportImportDialog
from .pipeline_panel import PipelinePanel
from .placeholder import PlaceholderView
from .processing_controls import ProcessingControls
from .seismic_view import SeismicView
from .spectrum_view import SpectrumView

__all__ = ["CRSAdvancedSearchDialog", "CRSSelectorDialog", "ExportDialog",
           "HeaderView", "MapView", "MetadataInspectorDialog",
           "PickingExportImportDialog", "PipelinePanel",
           "PlaceholderView", "ProcessingControls", "SeismicView", "SpectrumView"]
