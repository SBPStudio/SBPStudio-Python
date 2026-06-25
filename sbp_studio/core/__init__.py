"""
sbp_studio.core — headless computational core.

GUI-independence contract: this package MUST NOT import tkinter, PyQt/PySide,
matplotlib.pyplot/Figure, or PyQtGraph at module level. matplotlib colormaps
are accessed lazily inside coloring.py only.
"""
from .constants import CMAPS, COORD_UNITS, PRESETS_CRS, FILTER_PRESETS, FILTER_DESCRIPTIONS
from .logger import configure_logging, get_logger
from ._backends import gpu_available, worker_count
from .tasks import (
    TopasCoreError, SegyLoadError, CRSError, ReprojectionError, Cancelled,
    ProgressCallback, CancelToken,
)
from .model import SegyMetadata, SegyProfile, ProfileChain
from .io_segy import (load_metadata, load_profile, smooth_track,
                      reproject_one, reproject_chain, join_profiles,
                      patch_segy_headers, decode_text_header,
                      detect_text_header_encoding, read_raw_text_header,
                      trace_field_names, trace_field_int_range,
                      patch_trace_header_field, set_crs_override)
from .header_calc import (evaluate_header_expr, parse_assignment,
                          validate_header_result, available_functions,
                          HeaderExprError)
from .processing import (
    apply_dc_removal, apply_trace_equalization, apply_trace_mixing, apply_median_filter,
    apply_predictive_decon, apply_filter_preset, apply_agc,
    apply_bandpass, apply_tvg, apply_delay_alignment, apply_log_compression,
    apply_clahe, apply_despike,
    apply_water_mute, apply_swell_filter, apply_spectral_whitening,
    apply_fk_filter, apply_multiple_suppression, apply_notch,
    compute_amplitude_spectrum,
    process_profile_data, process_chain_data, time_window,
)
from .spectrum import compute_spectrum, SpectrumResult
from .coordinates import resolve_crs, validate_crs
from .spatial import (reproject_points, to_geographic, safe_map_coords,
                      crs_produces_geographic, CRS_CATALOG, CRS_PRESETS)
from .gis_io import (read_gis_layer, read_vector, read_geotiff,
                     VectorLayer, RasterLayer)
from .chaining import detect_chains, import_chains_from_directory
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
    # Logging
    "configure_logging", "get_logger",
    # Exceptions / tasks
    "TopasCoreError", "SegyLoadError", "CRSError", "ReprojectionError", "Cancelled",
    "ProgressCallback", "CancelToken",
    # Model
    "SegyMetadata", "SegyProfile", "ProfileChain",
    # I/O
    "load_metadata", "load_profile", "smooth_track", "reproject_one", "reproject_chain",
    "join_profiles", "patch_segy_headers", "decode_text_header",
    "detect_text_header_encoding", "read_raw_text_header",
    "trace_field_names", "trace_field_int_range", "patch_trace_header_field",
    "set_crs_override",
    # Header calculator (safe sandboxed expression evaluator)
    "evaluate_header_expr", "parse_assignment", "validate_header_result",
    "available_functions", "HeaderExprError",
    # Processing
    "apply_dc_removal", "apply_trace_equalization", "apply_trace_mixing", "apply_median_filter",
    "apply_predictive_decon", "apply_filter_preset", "apply_agc",
    "apply_bandpass", "apply_tvg", "apply_delay_alignment", "apply_log_compression",
    "apply_clahe", "apply_despike",
    "apply_water_mute", "apply_swell_filter", "apply_spectral_whitening",
    "apply_fk_filter", "apply_multiple_suppression", "apply_notch",
    "compute_amplitude_spectrum",
    "process_profile_data", "process_chain_data", "time_window",
    # Spectrum
    "compute_spectrum", "SpectrumResult",
    # Coordinates
    "resolve_crs", "validate_crs", "reproject_points", "to_geographic",
    "safe_map_coords", "crs_produces_geographic", "CRS_CATALOG", "CRS_PRESETS",
    "read_gis_layer", "read_vector", "read_geotiff", "VectorLayer", "RasterLayer",
    # Chaining
    "detect_chains", "import_chains_from_directory",
    # Coloring
    "colormapped_rgba",
    # Geometry export
    "parse_timestamp", "compute_fix_positions",
    "write_fix_points_shp", "write_fix_points_geojson", "write_fix_points_csv",
    "write_navline_shp", "write_navline_geojson", "write_navline_csv",
]
