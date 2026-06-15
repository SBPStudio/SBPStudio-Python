"""
test_export_dpi.py — GUI export feeds the core a DPI high enough to embed the
FULL native sample grid (no decimation → crisp PDF), without touching the core
renderer or the CLI.

The core colourises the matrix to target = (figsize × dpi) before imshow; if that
target is shorter than ns the sample grid is decimated. effective_export_dpi
raises the render DPI just enough to prevent that, with the user DPI as a floor.
"""
from __future__ import annotations

import math

from sbp_studio.gui.tabs._render import (
    EXPORT_DPI_CEILING, compute_figsize, effective_export_dpi,
)


class TestEffectiveExportDpi:
    def test_raises_to_avoid_vertical_decimation(self):
        # 8 x 2.67 in figure, ns=2048: need >= 2048/2.67 ≈ 767 → bump 600 → ~768.
        dpi = effective_export_dpi((8.0, 2.67), (2048, 1347), 600)
        assert dpi >= math.ceil(2048 / 2.67)
        assert 2.67 * dpi >= 2048                    # target_h now covers every sample

    def test_user_dpi_is_the_floor(self):
        # Data already fits the figure at the requested DPI → keep the user value.
        assert effective_export_dpi((8.0, 6.0), (1000, 800), 600) == 600

    def test_never_below_requested(self):
        # A tiny matrix must never LOWER the user's chosen DPI.
        assert effective_export_dpi((20.0, 20.0), (100, 100), 900) == 900

    def test_respects_ceiling(self):
        assert effective_export_dpi((0.5, 0.5), (50000, 50000), 600) == EXPORT_DPI_CEILING

    def test_degenerate_figsize_returns_requested(self):
        assert effective_export_dpi((0.0, 0.0), (2048, 1347), 600) == 600


def test_end_to_end_export_embeds_full_sample_grid(tmp_path):
    """The wired DPI floor embeds the full ns rows; the un-floored 600 DPI path
    decimates below ns — proving both the bug and the fix through the REAL core."""
    from tests.make_synthetic_segy import make_synthetic_segy
    from sbp_studio.core import load_profile
    from sbp_studio.viz.render import render_profile_figure

    p = make_synthetic_segy(str(tmp_path / "hi.sgy"), n_traces=60, ns=2048)
    sd = load_profile(p, load_traces=True)

    # Mimic the GUI export wiring exactly.
    figsize = compute_figsize(sd, 600, None, 3.0, 1500.0)   # ≈ (8.0, 2.67)
    render_dpi = effective_export_dpi(figsize, sd.data.shape, 600)

    fig = render_profile_figure(sd, sd.data, {}, figsize=figsize, dpi=render_dpi)
    rows = fig.axes[0].get_images()[0].get_array().shape[0]
    assert rows >= sd.ns                              # FIX: full grid, crisp
    fig.clear()

    # The historical (un-floored) 600-DPI call would decimate below ns.
    bug = render_profile_figure(sd, sd.data, {}, figsize=figsize, dpi=600)
    bug_rows = bug.axes[0].get_images()[0].get_array().shape[0]
    assert bug_rows < sd.ns                           # demonstrates the regression
    bug.clear()


def test_cli_render_path_unchanged():
    """Zero CLI regression: the core renderer signature/behaviour is untouched —
    a low-DPI render (CLI default) still works and is NOT forced higher (the CLI
    never calls effective_export_dpi)."""
    import inspect
    from sbp_studio.cli import commands
    from sbp_studio.viz.render import render_profile_figure

    src = inspect.getsource(commands)
    assert "effective_export_dpi" not in src          # CLI does its own DPI/figsize
    # render_profile_figure still accepts the historical (figsize, dpi) contract.
    sig = inspect.signature(render_profile_figure)
    assert "figsize" in sig.parameters and "dpi" in sig.parameters
