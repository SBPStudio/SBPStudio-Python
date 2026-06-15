"""
test_render_x_axis.py — bottom X-axis modes in the core renderer (Tasks 6 & 7).

  distance (default) : km extent  → image stretched uniformly across [d0, d1].
  trace              : trace-index extent [0, n], axis labelled "Trace number".
  km                 : trace-index extent [0, n] (no stretch) but km labels at the
                       real trace positions.
"""
from __future__ import annotations

import numpy as np


def _profile(tmp_path):
    from tests.make_synthetic_segy import make_synthetic_segy
    from sbp_studio.core import load_profile
    p = make_synthetic_segy(str(tmp_path / "x.sgy"), n_traces=200, ns=600,
                            base_lon=-3.0, base_lat=43.0)
    return load_profile(p, load_traces=True)


def _render(sd, x_axis):
    from sbp_studio.viz.render import render_profile_figure
    return render_profile_figure(sd, sd.data, {}, figsize=(8, 4), dpi=150,
                                 x_axis=x_axis)


def test_distance_mode_uses_km_extent(tmp_path):
    sd = _profile(tmp_path)
    fig = _render(sd, "distance")
    ax = fig.axes[0]
    x0, x1, *_ = ax.get_images()[0].get_extent()
    assert abs(x0 - sd.dist_km[0]) < 1e-6 and abs(x1 - sd.dist_km[-1]) < 1e-6
    assert ax.get_xlabel() == "Distance (km)"
    fig.clear()


def test_trace_mode_uses_trace_extent_and_label(tmp_path):
    sd = _profile(tmp_path)
    fig = _render(sd, "trace")
    ax = fig.axes[0]
    x0, x1, *_ = ax.get_images()[0].get_extent()
    assert x0 == 0.0 and abs(x1 - sd.n_traces) < 1e-6     # one column per trace
    assert ax.get_xlabel() == "Trace number"
    fig.clear()


def test_km_no_stretch_mode_trace_extent_km_labels(tmp_path):
    sd = _profile(tmp_path)
    fig = _render(sd, "km")
    ax = fig.axes[0]
    x0, x1, *_ = ax.get_images()[0].get_extent()
    assert x0 == 0.0 and abs(x1 - sd.n_traces) < 1e-6     # NOT stretched to km
    assert ax.get_xlabel() == "Distance (km)"             # but axis reads in km
    # tick positions are trace indices (0..n), labels are km values
    ticks = ax.get_xticks()
    assert np.all(ticks >= -1) and np.all(ticks <= sd.n_traces + 1)
    fig.clear()


def test_default_x_axis_is_distance(tmp_path):
    """Zero regression: default render is the historical distance mode."""
    sd = _profile(tmp_path)
    from sbp_studio.viz.render import render_profile_figure
    fig = render_profile_figure(sd, sd.data, {}, figsize=(8, 4), dpi=150)
    assert fig.axes[0].get_xlabel() == "Distance (km)"
    fig.clear()


def _chain(tmp_path):
    from tests.make_synthetic_segy import make_chain_pair
    from sbp_studio.core import load_profile, detect_chains
    p1, p2 = make_chain_pair(str(tmp_path), prefix="c", gap_km=0.05)
    ch = detect_chains([load_profile(p1), load_profile(p2)], gap_km=1.0)[0]
    ch.load_chain_traces()
    return ch


def test_chain_image_resized_to_output_width_no_nearest_block(tmp_path):
    """The chain renderer must scale the stitched native-width image to the OUTPUT
    pixel width (so imshow is 1:1 / smooth), NOT leave it at native width for
    imshow to nearest-upscale into px_per_trace blocks (the horizontal smearing)."""
    from sbp_studio.viz.render import render_chain_figure
    ch = _chain(tmp_path)
    n, dpi, ppt = ch.n_traces, 150, 4
    figw = n * ppt / dpi                                  # trace-based width
    fig = render_chain_figure(ch, ch.data, {}, figsize=(figw, 4), dpi=dpi,
                              x_axis="trace")
    im = fig.axes[0].get_images()[0]
    assert im.get_array().shape[1] == int(figw * dpi)    # = output width (resized)
    assert im.get_array().shape[1] != n                  # NOT native (would block)
    assert abs(im.get_extent()[1] - n) < 1e-6            # trace axis still 0..n
    fig.clear()


def test_chain_distance_mode_also_resized(tmp_path):
    from sbp_studio.viz.render import render_chain_figure
    ch = _chain(tmp_path)
    fig = render_chain_figure(ch, ch.data, {}, figsize=(8, 4), dpi=150,
                              x_axis="distance")
    im = fig.axes[0].get_images()[0]
    assert im.get_array().shape[1] == int(8 * 150)       # output width, smooth
    fig.clear()
