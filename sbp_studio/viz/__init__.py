"""
sbp_studio.viz — Headless visualisation layer (Agg/Pillow).

Imports matplotlib with Agg backend. MUST NOT be imported by core/.
"""
from .render import (
    render_profile_figure,
    render_chain_figure,
    render_spectrum_figure,
    save_figure,
    save_raw_rgba,
    build_theme,
)

__all__ = [
    "render_profile_figure",
    "render_chain_figure",
    "render_spectrum_figure",
    "save_figure",
    "save_raw_rgba",
]
