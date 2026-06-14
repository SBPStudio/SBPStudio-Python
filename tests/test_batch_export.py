"""
test_batch_export.py — Batch export file-routing + shared render pipeline.

The GUI method SubTabbedTab.export_batch builds a CoreWorker; its two pure,
core-facing pieces are tested headlessly here (the Qt method itself can't run in
the sandbox):

  * _batch_output_path — folder-named output in the SOURCE directory, de-duped.
  * _render_export_figure — the SAME render path the single export uses, proving
    batch output is pristine (decimation-free DPI floor) and that it does NOT
    mutate / cache the input matrix (RAM-safe transient use).
"""
from __future__ import annotations

from pathlib import Path

from topassuite.gui.tabs._base import _batch_output_path, _render_export_figure
from topassuite.gui.tabs._render import compute_figsize, effective_export_dpi


# ── File routing / naming ───────────────────────────────────────────────────────

class TestBatchOutputPath:
    def test_named_after_parent_folder_in_source_dir(self):
        used: set = set()
        out = _batch_output_path(r"Z:/data/Line_01/1.sgy", "pdf", used)
        assert out == Path(r"Z:/data/Line_01/Line_01.pdf")

    def test_dedup_same_folder_does_not_overwrite(self):
        used: set = set()
        a = _batch_output_path("/d/Line_01/1.sgy", "pdf", used); used.add(str(a))
        b = _batch_output_path("/d/Line_01/2.sgy", "pdf", used); used.add(str(b))
        assert a == Path("/d/Line_01/Line_01.pdf")
        assert b == Path("/d/Line_01/Line_01_2.pdf")     # disambiguated, not clobbered
        assert a != b

    def test_format_extension_follows_cfg(self):
        out = _batch_output_path("/d/Survey_A/x.seg", "png", set())
        assert out == Path("/d/Survey_A/Survey_A.png")


# ── Shared render pipeline (single == batch) ────────────────────────────────────

class _NoCancel:
    def check(self):
        return None


def _cfg(dpi=600):
    return dict(dpi=dpi, velocity=1500.0, x_tick=None, t_tick=None, grid=False,
                time_ticks=5, margin_top=20.0, margin_bottom=20.0, time_fmt="full",
                time_font_size=6.0, time_align="left", fix_bbox_alpha=0.0,
                fix_color=None, theme="print", pdf_page="auto")


def test_render_export_figure_is_pristine_and_non_mutating(tmp_path):
    from tests.make_synthetic_segy import make_synthetic_segy
    from topassuite.core import load_profile

    p = make_synthetic_segy(str(tmp_path / "L.sgy"), n_traces=60, ns=2048)
    sd = load_profile(p, load_traces=True)
    before = sd.data.copy()
    cfg = _cfg(600)
    params = {"clip_lo": 0.0, "fill_value": 0.0, "draw_file_boundaries": False,
              "align": False}

    fig, render_dpi = _render_export_figure(
        sd, cfg, params, node_cfg=[], aspect=3.0, align_enabled=False,
        is_chain=False, cancel=_NoCancel())

    # DPI floored so the embedded raster carries the full native sample grid.
    figsize = compute_figsize(sd, 600, None, 3.0, 1500.0)
    assert render_dpi == effective_export_dpi(figsize, sd.data.shape, 600)
    rows = fig.axes[0].get_images()[0].get_array().shape[0]
    assert rows >= sd.ns                                  # crisp, no decimation

    # The pipeline must NOT mutate the source matrix (batch reuses obj across the
    # loop / the live view shares it) — it works on a copy.
    import numpy as np
    assert np.array_equal(sd.data, before)
    import matplotlib.pyplot as plt
    plt.close(fig)
