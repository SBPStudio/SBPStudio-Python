"""
test_export_figsize.py — export pipeline AFTER the selective rollback.

The figsize experiments (build/clamp/dynamic_export_figsize, _imshow_interpolation)
are gone: the export is back to the ORIGINAL compute_figsize scaling +
interpolation='none'. The KEPT features are verified: the effective_export_dpi
anti-downsampling floor, the 1200 DPI default, the clean top axis, and batch
export.
"""
from __future__ import annotations


def test_figsize_experiments_are_removed():
    """The data-density / clamp / dynamic experiments were rolled back."""
    import sbp_studio.gui.tabs._render as r
    for name in ("build_export_figsize", "clamp_export_figsize",
                 "dynamic_export_figsize"):
        assert not hasattr(r, name), name


def test_export_helper_uses_original_compute_figsize():
    """_render_export_figure routes through figsize_for_scale (which wraps the
    original compute_figsize to add the aspect/VE/hybrid modes), not the reverted
    experiments; the anti-downsampling floor is KEPT."""
    import inspect
    from sbp_studio.gui.tabs import _base, _render
    src = inspect.getsource(_base._render_export_figure)
    assert "figsize_for_scale(" in src
    # figsize_for_scale must still be a thin wrapper over the original sizing.
    assert "compute_figsize(" in inspect.getsource(_render.figsize_for_scale)
    for gone in ("build_export_figsize", "clamp_export_figsize",
                 "dynamic_export_figsize"):
        assert gone not in src
    assert "effective_export_dpi(" in src


def test_core_renderer_strict_none_no_corruption_cap():
    """No dynamic blurring, and the PDF-corrupting array cap is NOT restored."""
    import inspect
    from sbp_studio.viz import render
    src = inspect.getsource(render)
    assert 'interpolation="none"' in src
    assert "_imshow_interpolation" not in src
    assert "12000" not in src


def test_default_export_dpi_is_1200():
    from sbp_studio.gui.components.export_dialog import DEFAULT_DPI
    assert DEFAULT_DPI == "1200"


def test_batch_export_feature_intact():
    from sbp_studio.gui.tabs._base import SubTabbedTab
    assert hasattr(SubTabbedTab, "export_batch")
    assert hasattr(SubTabbedTab, "_on_batch_exported")


def test_export_top_axis_is_clean(tmp_path):
    """time_tick_min=None (the GUI export default) → no top UTC time markers."""
    from tests.make_synthetic_segy import make_synthetic_segy
    from sbp_studio.core import load_profile
    from sbp_studio.viz.render import render_profile_figure

    p = make_synthetic_segy(str(tmp_path / "t.sgy"), n_traces=200, ns=512)
    sd = load_profile(p, load_traces=True)
    fig = render_profile_figure(sd, sd.data, {}, figsize=(8, 3), dpi=300,
                                time_tick_min=None)
    for ax in fig.axes:
        for lbl in ax.get_xticklabels():
            assert "#" not in lbl.get_text() and "Z" not in lbl.get_text()
    fig.clear()


def test_gui_export_config_disables_time_axis():
    import inspect
    from sbp_studio.gui.components import export_dialog
    assert "time_ticks=None" in inspect.getsource(export_dialog.ExportDialog.config)
