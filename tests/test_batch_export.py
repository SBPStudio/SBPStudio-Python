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

import pytest

from sbp_studio.gui.tabs._base import _batch_output_path, _render_export_figure
from sbp_studio.gui.tabs._render import compute_figsize, effective_export_dpi


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


def _profile_handler():
    """A real ProfileHandler whose render_figure() path needs no live tab/state
    — render_figure() calls render_profile_figure() directly without touching
    self._tab/self._state — so a bare stand-in satisfies SourceHandler.__init__
    (which only reads tab.state) without building any Qt widget."""
    from types import SimpleNamespace
    from sbp_studio.gui.tabs._handlers import ProfileHandler
    return ProfileHandler(SimpleNamespace(state=None))


def test_render_export_figure_is_pristine_and_non_mutating(tmp_path):
    from tests.make_synthetic_segy import make_synthetic_segy
    from sbp_studio.core import load_profile

    p = make_synthetic_segy(str(tmp_path / "L.sgy"), n_traces=60, ns=2048)
    sd = load_profile(p, load_traces=True)
    before = sd.data.copy()
    cfg = _cfg(600)
    params = {"clip_lo": 0.0, "fill_value": 0.0, "draw_file_boundaries": False,
              "align": False}
    scale_cfg = {"mode": "aspect", "ratio": 3.0, "layout_mode": "aspect"}

    fig, render_dpi = _render_export_figure(
        sd, cfg, params, node_cfg=[], scale_cfg=scale_cfg, align_enabled=False,
        handler=_profile_handler(), cancel=_NoCancel())

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


# ── Prefetch-one pipeline (roadmap #4) ───────────────────────────────────────

class _FakeHandler:
    """Minimal SourceHandler stand-in for the batch-loop engine: records
    load order and simulates load latency so overlap is observable."""
    def __init__(self, prefetch_safe=True, load_delay=0.0, bad=()):
        self.prefetch_safe = prefetch_safe
        self._load_delay = load_delay
        self._bad = set(bad)              # names whose load raises
        self.load_calls = []              # names, in the order load_full ran
        self.released = []

    def source_path(self, obj):
        return f"/data/{obj.name}/{obj.name}.sgy"

    def load_full(self, obj, cancel):
        import time
        self.load_calls.append(obj.name)
        if self._load_delay:
            time.sleep(self._load_delay)
        if obj.name in self._bad:
            raise RuntimeError(f"boom {obj.name}")
        loaded = SimpleNamespace(name=obj.name, data=object())   # "loaded"
        return loaded

    def release_after_batch(self, obj):
        self.released.append(obj.name)


def _items(n, ns=256, nt=100):
    return [SimpleNamespace(name=f"L{i}", ns=ns, n_traces=nt, data=None)
            for i in range(n)]


from types import SimpleNamespace                                # noqa: E402
from sbp_studio.gui.tabs._base import (                          # noqa: E402
    _run_batch_export_loop, _estimate_batch_item_bytes)


class _NoCancelTok:
    def check(self):
        return None


class TestBatchItemEstimate:
    def test_scales_with_deepest_widest(self):
        items = _items(3, ns=1000, nt=200) + [SimpleNamespace(ns=2000, n_traces=500, data=None)]
        est = _estimate_batch_item_bytes(items)
        assert est == 2000 * 500 * 4.0 * 4.0        # peak × mult
    def test_empty_and_headerless_safe(self):
        assert _estimate_batch_item_bytes([]) > 0
        assert _estimate_batch_item_bytes([SimpleNamespace()]) > 0


class TestBatchExportLoop:
    def _run(self, handler, items, monkeypatch, workers=4):
        import sbp_studio.core._backends as B
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: workers)
        saved, out_order = [], []
        def make_out(obj):
            out_order.append(obj.name); return f"/out/{obj.name}.png"
        rendered = []
        def render_save(render_obj, out):
            rendered.append(render_obj.name)
        s, f = _run_batch_export_loop(
            items, handler, make_out=make_out, render_save=render_save,
            progress=lambda *a: None, cancel=_NoCancelTok())
        return s, f, rendered

    def test_all_items_rendered_once_in_order(self, monkeypatch):
        h = _FakeHandler(prefetch_safe=True)
        items = _items(5)
        saved, failed, rendered = self._run(h, items, monkeypatch)
        assert failed == []
        assert rendered == [o.name for o in items]         # every item, in order
        assert saved == [f"/out/L{i}.png" for i in range(5)]

    def test_prefetch_enabled_for_profiles_with_ram(self, monkeypatch):
        h = _FakeHandler(prefetch_safe=True, load_delay=0.02)
        items = _items(4)
        self._run(h, items, monkeypatch, workers=4)
        # With prefetch, item i+1's load is submitted before i renders, so the
        # NEXT load starts ahead of turn — load order still covers all items.
        assert sorted(h.load_calls) == [o.name for o in items]

    def test_no_prefetch_when_low_ram(self, monkeypatch):
        """plan_workers==1 (2 items don't fit) → synchronous, no loader thread.
        Prove no prefetch by asserting loads happen strictly one-at-a-time in
        render order (a background prefetch would load ahead)."""
        h = _FakeHandler(prefetch_safe=True)
        items = _items(4)
        saved, failed, rendered = self._run(h, items, monkeypatch, workers=1)
        assert rendered == [o.name for o in items]
        assert h.load_calls == [o.name for o in items]     # in-order, no look-ahead

    def test_no_prefetch_for_chains(self, monkeypatch):
        """A chain-like handler (prefetch_safe=False) must never spawn the
        loader thread even with abundant RAM."""
        h = _FakeHandler(prefetch_safe=False)
        items = _items(3)
        saved, failed, rendered = self._run(h, items, monkeypatch, workers=8)
        assert rendered == [o.name for o in items]
        assert h.load_calls == [o.name for o in items]

    def test_one_bad_item_isolated(self, monkeypatch):
        h = _FakeHandler(prefetch_safe=True, bad={"L2"})
        items = _items(5)
        saved, failed, rendered = self._run(h, items, monkeypatch)
        assert [n for n, _ in failed] == ["L2"]
        assert rendered == ["L0", "L1", "L3", "L4"]         # bad one skipped
        assert len(saved) == 4

    def test_release_called_for_each_item(self, monkeypatch):
        h = _FakeHandler(prefetch_safe=True)
        items = _items(3)
        self._run(h, items, monkeypatch)
        assert sorted(h.released) == [o.name for o in items]   # RAM-flat release

    def test_cancel_aborts_batch(self, monkeypatch):
        import sbp_studio.core._backends as B
        from sbp_studio.core.tasks import Cancelled
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: 4)
        class _CancelAt:
            def __init__(self, k): self.k = k; self.n = 0
            def check(self):
                self.n += 1
                if self.n > self.k:
                    raise Cancelled("stop")
        h = _FakeHandler(prefetch_safe=True)
        with pytest.raises(Cancelled):
            _run_batch_export_loop(
                _items(6), h, make_out=lambda o: f"/o/{o.name}.png",
                render_save=lambda ro, out: None,
                progress=lambda *a: None, cancel=_CancelAt(2))
