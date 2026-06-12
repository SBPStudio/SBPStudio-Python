"""
topassuite.core — headless computational core.

GUI-independence contract: this package MUST NOT import tkinter, PyQt/PySide,
matplotlib.pyplot/Figure, or PyQtGraph at module level. matplotlib colormaps
are accessed lazily inside coloring.py only.
"""
from .constants import CMAPS, COORD_UNITS, PRESETS_CRS, FILTER_PRESETS, FILTER_DESCRIPTIONS
from ._backends import gpu_available, worker_count
from .tasks import (
    TopasCoreError, SegyLoadError, CRSError, ReprojectionError, Cancelled,
    ProgressCallback, CancelToken,
)
from .model import SegyMetadata, SegyProfile, ProfileChain
from .io_segy import (load_metadata, load_profile, smooth_track,
                      reproject_one, reproject_chain, join_profiles)
from .processing import (
    apply_predictive_decon, apply_filter_preset, apply_agc,
    apply_bandpass, apply_tvg, apply_delay_alignment,
    apply_water_mute, apply_swell_filter, compute_amplitude_spectrum,
    process_profile_data, process_chain_data, time_window,
)
from .spectrum import compute_spectrum, SpectrumResult
from .coordinates import resolve_crs, validate_crs
from .spatial import reproject_points, to_geographic, CRS_CATALOG, CRS_PRESETS
from .gis_io import (read_gis_layer, read_vector, read_geotiff,
                     VectorLayer, RasterLayer)
from .chaining import detect_chains
from .coloring import colormapped_rgba
from .geometry_export import (
    parse_timestamp, compute_fix_positions,
    write_fix_points_shp, write_fix_points_geojson, write_fix_points_csv,
    write_navline_shp, write_navline_geojson, write_navline_csv,
)

__all__ = [
    # Constants
    "CMAPS", "COORD_UNITS", "PRESETS_CRS", "FILTER_PRESETS", "FILTER_DESCRIPTIONS",
    # Backends
    "gpu_available", "worker_count",
    # Exceptions / tasks
    "TopasCoreError", "SegyLoadError", "CRSError", "ReprojectionError", "Cancelled",
    "ProgressCallback", "CancelToken",
    # Model
    "SegyMetadata", "SegyProfile", "ProfileChain",
    # I/O
    "load_metadata", "load_profile", "smooth_track", "reproject_one", "reproject_chain",
    "join_profiles",
    # Processing
    "apply_predictive_decon", "apply_filter_preset", "apply_agc",
    "apply_bandpass", "apply_tvg", "apply_delay_alignment",
    "apply_water_mute", "apply_swell_filter", "compute_amplitude_spectrum",
    "process_profile_data", "process_chain_data", "time_window",
    # Spectrum
    "compute_spectrum", "SpectrumResult",
    # Coordinates
    "resolve_crs", "validate_crs", "reproject_points", "to_geographic",
    "CRS_CATALOG", "CRS_PRESETS",
    "read_gis_layer", "read_vector", "read_geotiff", "VectorLayer", "RasterLayer",
    # Chaining
    "detect_chains",
    # Coloring
    "colormapped_rgba",
    # Geometry export
    "parse_timestamp", "compute_fix_positions",
    "write_fix_points_shp", "write_fix_points_geojson", "write_fix_points_csv",
    "write_navline_shp", "write_navline_geojson", "write_navline_csv",
]
