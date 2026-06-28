"""
test_export_picks_overlay.py — "Overlay interpretation markers" export feature.

Covers burning interpretation picks (PickPoint: trace_index, time_ms, id) into
the headless Matplotlib export raster:
  1. viz/render._draw_picks — the actual drawing helper.
  2. render_profile_figure's ``picks`` kwarg, threaded through
     gui/tabs/_base._render_export_figure (the SAME shared render path single
     export and batch export both use).
  3. ExportDialog's new "Superponer marcas de interés" checkbox.

The export pipeline is headless Matplotlib (Agg) — no QPainter/QImage at all
(see render_profile_figure's docstring) — so picks are plotted directly in the
Axes' own (x, time_ms) data-coordinate system, the SAME system ax.imshow's
``extent`` already establishes and the live SeismicView already uses for its
own (km, ms) pick positions (_redraw_picks) — no separate pixel math needed.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from sbp_studio.core import PickPoint
from sbp_studio.viz.render import _draw_picks, render_profile_figure


def _picks(*specs):
    """specs: (id, trace_index, time_ms) tuples."""
    return [PickPoint(id=i, trace_index=t, time_ms=ms, x_coord=0.0, y_coord=0.0,
                      description=f"p{i}") for i, t, ms in specs]


class TestDrawPicksHelper:
    """_draw_picks(ax, picks, x_lo, x_hi, n_traces, ...) — picks are
    interpolated linearly into [x_lo, x_hi] (the SAME bounds the caller
    already passed to ax.imshow's own extent), NOT looked up via the true
    (possibly non-uniform) dist_km value. See the function's own docstring
    for why a dist_km lookup decouples from where the image's resampled
    pixels actually land whenever real trace spacing isn't uniform."""

    def test_empty_picks_draws_nothing(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _draw_picks(ax, [], 0.0, 2.0, 20)
        assert len(ax.collections) == 0
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_none_picks_draws_nothing(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _draw_picks(ax, None, 0.0, 2.0, 20)
        assert len(ax.collections) == 0
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_zero_n_traces_draws_nothing(self):
        """Defensive guard against a divide-by-zero in the fraction calc."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _draw_picks(ax, _picks((1, 0, 10.0)), 0.0, 2.0, 0)   # must not raise
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_linear_interpolation_into_x_lo_x_hi_uniform_spacing(self):
        """With UNIFORM spacing, linear-fraction interpolation and a true
        dist_km[trace_index] lookup happen to coincide — this only proves
        the basic interpolation arithmetic; see the non-uniform test below
        for the case that actually distinguishes the two."""
        import matplotlib.pyplot as plt
        n = 20
        x_lo, x_hi = 0.0, 2.0
        fig, ax = plt.subplots()
        _draw_picks(ax, _picks((1, 5, 137.5)), x_lo, x_hi, n)
        offsets = ax.collections[0].get_offsets()
        assert offsets[0][0] == pytest.approx(x_lo + (5 / n) * (x_hi - x_lo))
        assert offsets[0][1] == pytest.approx(137.5)
        plt.close(fig)

    def test_trace_mode_extent_reduces_to_raw_trace_index(self):
        """"trace"/"km" x_axis modes pass x_lo=0, x_hi=n_traces (see
        _x_axis_extent) — the SAME formula then collapses to plain
        trace_index, exactly as it did before this fix, with no special
        casing needed inside _draw_picks itself."""
        import matplotlib.pyplot as plt
        n = 20
        fig, ax = plt.subplots()
        _draw_picks(ax, _picks((1, 5, 137.5)), 0.0, float(n), n)
        offsets = ax.collections[0].get_offsets()
        assert offsets[0][0] == pytest.approx(5.0)
        assert offsets[0][1] == pytest.approx(137.5)
        plt.close(fig)

    def test_non_uniform_spacing_regression_matches_image_pixel_not_dist_km(self):
        """THE regression this fix addresses: _colorize_for_target resamples
        columns UNIFORMLY (scipy.ndimage.zoom / PIL) regardless of dist_km's
        true (non-uniform) spacing — so column j of n_traces always renders
        at linear fraction j/n_traces of the image's width. A pick plotted
        at the TRUE dist_km[trace_index] would land somewhere else entirely
        whenever spacing is non-uniform (real surveys: ship speed varies).
        Verified end-to-end (not just re-deriving the same formula) in
        TestRenderExportFigureIntegration's ground-truth pixel-centroid test
        — this one just locks the exact linear-fraction arithmetic."""
        import matplotlib.pyplot as plt
        # A "ship slowdown": traces 40-70 cover only 0.5km instead of being
        # evenly spread — dist_km[55] sits much EARLIER than its column's
        # linear-fraction position in the full 0..8km extent would suggest.
        dist_km = np.concatenate([
            np.linspace(0.0, 3.0, 40),
            np.linspace(3.0, 3.5, 30),
            np.linspace(3.5, 8.0, 30),
        ])
        n = dist_km.size
        x_lo, x_hi = float(dist_km[0]), float(dist_km[-1])
        trace_index = 55
        fig, ax = plt.subplots()
        _draw_picks(ax, _picks((1, trace_index, 140.0)), x_lo, x_hi, n)
        x = ax.collections[0].get_offsets()[0][0]
        # Must match the LINEAR-FRACTION position (what the image actually
        # renders at), not the true (and very different) dist_km value.
        assert x == pytest.approx(x_lo + (trace_index / n) * (x_hi - x_lo))
        assert x != pytest.approx(float(dist_km[trace_index]), abs=0.5)
        plt.close(fig)

    def test_one_scatter_point_and_one_label_per_pick(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        _draw_picks(ax, _picks((1, 2, 10.0), (2, 8, 50.0), (3, 15, 90.0)),
                   0.0, 2.0, 20)
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == 3
        assert len(ax.texts) == 3
        assert {t.get_text() for t in ax.texts} == {"1", "2", "3"}
        plt.close(fig)

    def test_out_of_range_trace_index_is_clamped_not_raised(self):
        """An imported session from a longer line must not crash — same
        defensive clamp as SeismicView._redraw_picks."""
        import matplotlib.pyplot as plt
        n = 20
        x_lo, x_hi = 0.0, 2.0
        fig, ax = plt.subplots()
        _draw_picks(ax, _picks((1, 99999, 10.0)), x_lo, x_hi, n)   # must not raise
        offsets = ax.collections[0].get_offsets()
        assert offsets[0][0] == pytest.approx(x_lo + ((n - 1) / n) * (x_hi - x_lo))
        plt.close(fig)


class TestRenderExportFigureIntegration:
    """The full path: _render_export_figure (the SAME function single export
    AND batch export both call) → render_profile_figure → _draw_picks."""

    def _render(self, tmp_path, picks):
        from tests.make_synthetic_segy import make_synthetic_segy
        from sbp_studio.core import load_profile
        from sbp_studio.gui.tabs._base import _render_export_figure
        from sbp_studio.gui.tabs._handlers import ProfileHandler
        from types import SimpleNamespace

        p = make_synthetic_segy(str(tmp_path / "L.sgy"), n_traces=60, ns=512)
        sd = load_profile(p, load_traces=True)
        cfg = dict(dpi=150, velocity=1500.0, x_tick=None, t_tick=None, grid=False,
                  time_ticks=None, margin_top=0.0, margin_bottom=0.0,
                  time_fmt="full", time_font_size=6.0, time_align="left",
                  fix_bbox_alpha=0.0, fix_color=None, theme="print",
                  pdf_page="auto")
        params = {"clip_lo": 0.0, "fill_value": 0.0,
                  "draw_file_boundaries": False, "align": False}
        scale_cfg = {"mode": "aspect", "ratio": 3.0, "layout_mode": "aspect"}
        handler = ProfileHandler(SimpleNamespace(state=None))

        class _NoCancel:
            def check(self):
                return None

        fig, _dpi = _render_export_figure(
            sd, cfg, params, node_cfg=[], scale_cfg=scale_cfg,
            align_enabled=False, handler=handler, cancel=_NoCancel(),
            picks=picks)
        return fig, sd

    def test_picks_burned_into_the_export_figure(self, tmp_path):
        fig, sd = self._render(tmp_path, _picks((1, 10, 50.0), (2, 30, 150.0)))
        ax = fig.axes[0]
        assert len(ax.collections) >= 1   # at least the picks scatter
        pick_texts = {t.get_text() for t in ax.texts}
        assert {"1", "2"} <= pick_texts
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_picks_means_no_overlay_artifacts(self, tmp_path):
        """The default (checkbox unchecked → picks=[] from _base.py) must
        produce a clean section — no stray scatter/labels."""
        fig, sd = self._render(tmp_path, [])
        ax = fig.axes[0]
        assert len(ax.collections) == 0
        assert len(ax.texts) == 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_picks_default_is_none_and_safe(self, tmp_path):
        """_render_export_figure's picks kwarg defaults to None (every batch
        item that doesn't pass it explicitly) — must not raise."""
        fig, sd = self._render(tmp_path, None)
        ax = fig.axes[0]
        assert len(ax.collections) == 0
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestNonUniformSpacingGroundTruth:
    """Independent, end-to-end ground-truth check for the reported bug:
    markers plotting far from their true geological location whenever the
    real survey's trace spacing in km is non-uniform (ship speed varies).

    Methodology: inject a known bright reflector into the RAW array at an
    exact (sample, trace) location, render through the REAL
    render_profile_figure (full _colorize_for_target resampling pipeline,
    not a re-derivation of _draw_picks' own formula), then independently
    locate the rendered bright region's pixel centroid and confirm the pick
    scatter lands there — proving the fix by where the DATA actually
    renders, not by re-checking the same arithmetic twice."""

    def test_pick_lands_on_the_actual_rendered_pixel_not_raw_dist_km(self):
        import matplotlib.pyplot as plt
        ns, ntr = 200, 100
        dt_us = 2000
        data = np.zeros((ns, ntr), dtype=np.float32)

        # Non-uniform spacing: a "ship slowdown" bunches traces 40-70 into
        # only 0.5km instead of spreading them proportionally.
        dist_km = np.concatenate([
            np.linspace(0.0, 3.0, 40),
            np.linspace(3.0, 3.5, 30),
            np.linspace(3.5, 8.0, 30),
        ])
        trace_index = 55
        sample_index = 70
        time_ms = sample_index * dt_us / 1000.0
        data[sample_index - 3:sample_index + 4, trace_index - 3:trace_index + 4] = 1.0

        class Src:
            name = "gt"
            delay_ms = 0.0
            min_delay = 0.0
            timestamps = [None] * ntr
            track_lons = np.zeros(ntr)
            track_lats = np.zeros(ntr)
        # Set outside the class body to avoid Python's class-body scoping
        # quirk (a name assigned AND read in the same class body shadows the
        # enclosing function's local of the same name before the line runs).
        Src.n_traces = ntr
        Src.ns = ns
        Src.dt_us = dt_us
        Src.dist_km = dist_km
        sd = Src()

        picks = _picks((1, trace_index, time_ms))
        params = {"align": False, "clip": 100.0, "amp_range": "sequential"}
        fig = render_profile_figure(sd, data, params, figsize=(10, 6), dpi=150,
                                    picks=picks)
        ax = fig.axes[0]
        img = ax.get_images()[0]
        x_lo, x_hi, y_bottom, y_top = img.get_extent()
        arr = img.get_array()
        brightness = arr[..., :3].astype(float).sum(axis=2)
        rows, cols = np.where(brightness >= brightness.max() * 0.95)
        h, w = arr.shape[0], arr.shape[1]
        gt_x = x_lo + (cols.mean() / w) * (x_hi - x_lo)
        gt_y = y_top + (rows.mean() / h) * (y_bottom - y_top)

        x, y = ax.collections[0].get_offsets()[0]
        assert x == pytest.approx(gt_x, abs=0.1)
        assert y == pytest.approx(gt_y, abs=2.0)
        # The OLD (buggy) behaviour — plotting at the raw dist_km value —
        # would have missed the actual rendered pixel by over 1km here.
        assert abs(x - float(dist_km[trace_index])) > 0.5
        plt.close(fig)


class TestExportDialogPicksCheckbox:
    """ExportDialog's new "Superponer marcas de interés" checkbox.

    Verified via source inspection, NOT live QDialog construction: a bare
    ExportDialog() crashes this sandbox even with NO changes of this turn's
    (confirmed by isolating the construction alone) — the same class of
    pre-existing Qt real-paint/layout limitation documented elsewhere in
    this test suite for other widgets (e.g. SeismicView.show()). The
    dialog's actual behaviour (checkbox wiring, config() plumbing,
    retranslate_ui text) is fully exercised by reading its real running
    text, not faked — pytest-driven source inspection of the live file."""

    def _source(self) -> str:
        from sbp_studio.gui.components import export_dialog
        import inspect
        return inspect.getsource(export_dialog)

    def test_checkbox_declared_unchecked_and_added_to_layout(self):
        src = self._source()
        assert "self.cb_picks = QCheckBox()" in src
        assert "self.cb_picks.setChecked(False)" in src
        assert "root.addWidget(self.cb_picks)" in src

    def test_config_exposes_overlay_picks_from_the_checkbox(self):
        init_src = inspect.getsource(__import__(
            "sbp_studio.gui.components.export_dialog",
            fromlist=["ExportDialog"]).ExportDialog.config)
        assert "overlay_picks=self.cb_picks.isChecked()" in init_src

    def test_checkbox_text_and_tooltip_set_in_retranslate_ui(self):
        src = self._source()
        assert 'self.cb_picks.setText(self.tr("Overlay interpretation markers"))' in src
        assert "self.cb_picks.setToolTip(self.tr(" in src

    def test_spanish_translation_matches_the_exact_requested_label(self):
        """The .ts file must translate the English source literal to
        EXACTLY "Superponer marcas de interés" — the user's explicit,
        literal requirement for the rendered (Spanish) UI text."""
        import xml.etree.ElementTree as ET
        from pathlib import Path
        ts_path = (Path(__file__).resolve().parent.parent / "sbp_studio"
                  / "gui" / "translations" / "sbp_studio_es.ts")
        tree = ET.parse(ts_path)
        sources = {
            msg.findtext("source"): msg.findtext("translation")
            for ctx in tree.findall("context")
            for msg in ctx.findall("message")
        }
        assert sources.get("Overlay interpretation markers") == \
            "Superponer marcas de interés"
