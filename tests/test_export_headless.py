"""
test_export_headless.py — the untangled, Qt-free GUI export-render pipeline
(gui.export_headless) and the process-pool GUI batch export built on it.

The two properties that make the GUI pool safe are pinned here:
  1. PURITY — importing and fully exercising the worker render path must load
     ZERO PyQt6 modules (checked in a fresh subprocess, the only airtight way).
  2. FIDELITY — the pool worker's output must be byte-identical to the
     in-process path (_base._render_export_figure shim + save_figure), because
     both are literally the same moved-verbatim code.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import subprocess
import sys

import pytest

from tests.make_synthetic_segy import make_synthetic_segy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg():
    return dict(dpi=300, velocity=1500.0, x_tick=None, t_tick=None, grid=False,
                time_ticks=5, margin_top=10.0, margin_bottom=10.0,
                time_fmt="full", time_font_size=6.0, time_align="left",
                fix_bbox_alpha=0.0, fix_color=None, theme="print",
                pdf_page="auto")


def _params():
    return dict(cmap="Viridis", clip=99.6, clip_lo=0.0, fill_value=0.0,
                draw_file_boundaries=False, align=False, px_per_trace=2.0)


def _scale_cfg():
    return dict(layout_mode="aspect", mode="ratio", ratio=1.0)


def _payload(path, out, node_cfg=()):
    return dict(path=path, out=out, name=os.path.basename(path),
                fmt="png", pdf_page="auto", cfg=_cfg(), params=_params(),
                node_cfg=list(node_cfg), scale_cfg=_scale_cfg(),
                align_enabled=False)


def _expected_page(prof, cfg, view_scale):
    """Independent derivation of the 'Auto' page: FULL extents x scale."""
    import numpy as np
    d = np.asarray(prof.dist_km, dtype=float)
    data_km = float(d[-1] - d[0])
    rec_ms = prof.ns * prof.dt_us / 1000.0         + float(cfg.get("margin_top") or 0.0)         + float(cfg.get("margin_bottom") or 0.0)
    return (max(0.5, data_km * view_scale[0]), max(0.5, rec_ms * view_scale[1]))


def _scale_for_page(prof, cfg, w_in, h_in):
    """The (in/km, in/ms) view scale that yields a w_in x h_in page for prof."""
    import numpy as np
    d = np.asarray(prof.dist_km, dtype=float)
    data_km = float(d[-1] - d[0])
    rec_ms = prof.ns * prof.dt_us / 1000.0         + float(cfg.get("margin_top") or 0.0)         + float(cfg.get("margin_bottom") or 0.0)
    return (w_in / data_km, h_in / rec_ms)


@pytest.fixture
def profile_file(tmp_path):
    p = str(tmp_path / "hp.sgy")
    make_synthetic_segy(p, n_traces=50, ns=192)
    return p


class TestHeadlessPurity:
    def test_full_worker_render_loads_no_qt(self, profile_file, tmp_path):
        """Fresh interpreter: import export_headless, run a COMPLETE worker
        render (load → DSP → figsize/DPI → render → WYSIWYG fit → save), then
        assert not a single PyQt6 module was imported. This is the contract
        that makes the frozen-exe process pool RAM-safe."""
        out = str(tmp_path / "pure.png")
        code = (
            "import sys\n"
            f"sys.path.insert(0, {REPO!r})\n"
            "import sbp_studio.gui.export_headless as EH\n"
            "payload = dict(path=sys.argv[1], out=sys.argv[2], name='p',\n"
            "    fmt='png', pdf_page='auto',\n"
            "    cfg=dict(dpi=300, velocity=1500.0, x_tick=None, t_tick=None,\n"
            "             grid=False, time_ticks=5, margin_top=10.0,\n"
            "             margin_bottom=10.0, time_fmt='full',\n"
            "             time_font_size=6.0, time_align='left',\n"
            "             fix_bbox_alpha=0.0, fix_color=None, theme='print',\n"
            "             pdf_page='auto'),\n"
            "    params=dict(cmap='Viridis', clip=99.6, clip_lo=0.0,\n"
            "                fill_value=0.0, draw_file_boundaries=False,\n"
            "                align=False, px_per_trace=2.0),\n"
            "    node_cfg=[('agc', {'win_ms': 20.0})],\n"
            "    scale_cfg=dict(layout_mode='aspect', mode='ratio', ratio=1.0),\n"
            "    align_enabled=False)\n"
            "out, ok, err = EH.render_batch_item(payload)\n"
            "assert ok, err\n"
            "qt = [m for m in sys.modules if m.startswith('PyQt6')]\n"
            "sys.exit(1 if qt else 0)\n"
        )
        res = subprocess.run([sys.executable, "-c", code, profile_file, out],
                             capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, f"Qt leaked into the worker path:\n{res.stderr}"
        assert os.path.exists(out)

    def test_lazy_package_inits_stay_pure(self):
        """The enabling refactor: gui / gui.tabs / gui.dsp package __init__s
        must be importable (plus the pure submodules the pipeline needs)
        without PyQt6."""
        code = (
            "import sys\n"
            f"sys.path.insert(0, {REPO!r})\n"
            "import sbp_studio.gui.dsp\n"
            "import sbp_studio.gui.tabs._render\n"
            "from sbp_studio.gui.dsp import DSPContext, make_node\n"
            "qt = [m for m in sys.modules if m.startswith('PyQt6')]\n"
            "sys.exit(1 if qt else 0)\n"
        )
        res = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, f"package inits leak Qt:\n{res.stderr}"


class TestWorkerFidelity:
    def test_worker_matches_inprocess_shim_bytes(self, profile_file, tmp_path):
        """The heart of the 'purely structural' constraint: the pool worker
        and the GUI's in-process path (_base shim + real ProfileHandler) must
        produce BYTE-identical files — they are the same moved code."""
        from types import SimpleNamespace
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_batch_item
        from sbp_studio.gui.tabs._base import _render_export_figure
        from sbp_studio.gui.tabs._handlers import ProfileHandler
        from sbp_studio.viz.render import save_figure

        node_cfg = [("agc", {"win_ms": 20.0})]
        out_pool = str(tmp_path / "pool.png")
        out_gui = str(tmp_path / "gui.png")

        # Worker path.
        _o, ok, err = render_batch_item(_payload(profile_file, out_pool, node_cfg))
        assert ok, err

        # In-process GUI path (the shim, driven exactly like export_batch).
        class _NoCancel:
            def check(self):
                return None

        handler = ProfileHandler(SimpleNamespace(state=None))
        prof = load_profile(profile_file, load_traces=True)
        fig, dpi = _render_export_figure(prof, _cfg(), _params(), node_cfg,
                                         _scale_cfg(), False, handler,
                                         _NoCancel())
        save_figure(fig, out_gui, dpi=dpi, fmt="png", pdf_page="auto")
        fig.clear()

        a = open(out_pool, "rb").read()
        b = open(out_gui, "rb").read()
        assert a == b, "pool worker output differs from the in-process render!"

    def test_worker_failure_isolated(self, tmp_path):
        from sbp_studio.gui.export_headless import render_batch_item
        bad = str(tmp_path / "corrupt.sgy")
        with open(bad, "wb") as fh:
            fh.write(b"junk" * 200)
        out, ok, err = render_batch_item(_payload(bad, str(tmp_path / "x.png")))
        assert ok is False and err          # message returned, nothing raised


class TestPoolOrchestration:
    def test_all_items_saved_and_progress_complete(self, tmp_path, monkeypatch):
        """run_batch_export_pool over a ThreadPoolExecutor stand-in (same
        submit/as_completed API): every payload saved, progress reaches 1.0."""
        import sbp_studio.gui.export_headless as EH
        monkeypatch.setattr(cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)
        paths, payloads = [], []
        for i in range(3):
            p = str(tmp_path / f"f{i}.sgy")
            make_synthetic_segy(p, n_traces=40 + 6 * i, ns=160)
            payloads.append(_payload(p, str(tmp_path / f"f{i}.png")))
        fractions = []
        saved, failed = EH.run_batch_export_pool(
            payloads, 2, progress=lambda f, n: fractions.append(f))
        assert failed == []
        assert sorted(os.path.basename(s) for s in saved) == \
               ["f0.png", "f1.png", "f2.png"]
        assert fractions and fractions[-1] == pytest.approx(1.0)

    def test_bad_item_isolated_in_pool(self, tmp_path, monkeypatch):
        import sbp_studio.gui.export_headless as EH
        monkeypatch.setattr(cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)
        good = str(tmp_path / "good.sgy")
        make_synthetic_segy(good, n_traces=40, ns=160)
        bad = str(tmp_path / "bad.sgy")
        with open(bad, "wb") as fh:
            fh.write(b"junk" * 100)
        payloads = [_payload(good, str(tmp_path / "good.png")),
                    _payload(bad, str(tmp_path / "bad.png"))]
        saved, failed = EH.run_batch_export_pool(
            payloads, 2, progress=lambda *a: None)
        assert len(saved) == 1 and len(failed) == 1

    def test_real_process_pool_smoke(self, tmp_path):
        """One true spawn: proves the picklable worker renders in a real
        subprocess. Skip (not hang) if this environment can't spawn."""
        import sbp_studio.gui.export_headless as EH
        p = str(tmp_path / "rp.sgy")
        make_synthetic_segy(p, n_traces=40, ns=160)
        payloads = [_payload(p, str(tmp_path / "rp.png"))]
        try:
            saved, failed = EH.run_batch_export_pool(
                payloads, 1, progress=lambda *a: None)
        except (cf.process.BrokenProcessPool, OSError, RuntimeError) as exc:
            pytest.skip(f"real pool unavailable here: {exc}")
        assert failed == [] and len(saved) == 1
        assert os.path.exists(saved[0])


class TestDivideMemBudget:
    def test_explicit_budget_split(self):
        from sbp_studio.gui.export_headless import divide_mem_budget
        out = divide_mem_budget({"mem_budget_gb": 8.0}, 4)
        assert out["mem_budget_gb"] == pytest.approx(2.0)

    def test_floor(self):
        from sbp_studio.gui.export_headless import divide_mem_budget
        out = divide_mem_budget({"mem_budget_gb": 0.4}, 8)
        assert out["mem_budget_gb"] == pytest.approx(0.25)

    def test_default_positive_and_original_untouched(self):
        from sbp_studio.gui.export_headless import divide_mem_budget
        cfg = {"dpi": 300}
        out = divide_mem_budget(cfg, 4)
        assert out["mem_budget_gb"] > 0
        assert "mem_budget_gb" not in cfg          # copy, not mutation


# ── Viewport crop engine: HQ-overlay ONLY (export never crops — see below) ────

def _synthetic_obj():
    """A duck-typed profile stand-in with a monotonic distance axis — enough for
    the pure crop math without loading a SEG-Y file."""
    import numpy as np
    from types import SimpleNamespace
    ns, nt = 100, 200
    data = np.arange(ns * nt, dtype=float).reshape(ns, nt)
    dist = np.linspace(0.0, 10.0, nt)          # 0..10 km, uniform
    obj = SimpleNamespace(
        name="L", dist_km=dist, dt_us=1000, ns=ns, n_traces=nt, total_km=10.0,
        timestamps=list(range(nt)), lons=dist.copy(), lats=dist.copy(),
        delays=None, water_depth=None)
    return obj, data


class TestViewportCropForHQ:
    """_crop_for_viewport backs ONLY the in-viewer HQ overlay. The former
    export-extent cropping (crop_export_to_view) was REMOVED by field
    directive — the viewport authors the SCALE, never the extent."""

    def test_crop_engine_slices_both_axes(self):
        from sbp_studio.gui.export_headless import _crop_for_viewport
        obj, data = _synthetic_obj()
        clone, crop = _crop_for_viewport(obj, data, 0.0, (2.0, 4.0), (10.0, 40.0))
        assert crop.shape[0] < data.shape[0] and crop.shape[1] < data.shape[1]
        assert clone.n_traces == crop.shape[1] and clone.ns == crop.shape[0]
        assert clone.dist_km[0] >= 2.0 - 1e-6 and clone.dist_km[-1] <= 4.0 + 1e-6

    def test_extent_cropping_is_gone_from_the_export(self):
        """Regression pin for the field workflow: zooming must never crop the
        exported data. The extent-crop entry point no longer exists."""
        import sbp_studio.gui.export_headless as EH
        assert not hasattr(EH, "crop_export_to_view")
        assert not hasattr(EH, "_shift_picks")


# ── Uncompromising quality: NO hidden pixel ceiling; the budget is the law ────

class TestUncompromisingQuality:
    """User directive: maximum export resolution always. The 100 Mpx hard cap
    (a silent DPI downgrader) was REMOVED — the ONLY canvas limit is the
    USER-OWNED memory budget (the dialog's 'Memory budget (GB)'), whose clamp
    is announced in the dialog before exporting. These tests pin (a) the cap
    stays gone and (b) the budget contract is exact."""

    def test_hard_cap_is_gone(self):
        import inspect
        import sbp_studio.gui.tabs._render as R
        import sbp_studio.gui.export_headless as EH
        assert not hasattr(R, "hard_dpi_cap")
        assert not hasattr(R, "MAX_EXPORT_MEGAPIXELS")
        src = inspect.getsource(EH)
        assert "hard_dpi_cap" not in src
        assert "MAX_EXPORT_MEGAPIXELS" not in src

    def test_dpi_is_exactly_effective_or_budget(self, profile_file):
        """render_dpi == the native-grid floor, clamped ONLY by the user's
        budget — no other reducer exists in the path."""
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        from sbp_studio.gui.tabs._render import (dpi_for_budget,
                                                 effective_export_dpi)
        prof = load_profile(profile_file, load_traces=True)
        cfg = dict(_cfg(), paper_size="Auto", mem_budget_gb=6.0)
        view_scale = (2.0, 0.5)                  # in/km, in/ms
        fig, render_dpi = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_scale=view_scale)
        fig.clear()
        figsize = _expected_page(prof, cfg, view_scale)
        expect = effective_export_dpi(figsize, (prof.ns, prof.n_traces),
                                      int(cfg["dpi"]))
        cap = dpi_for_budget(figsize, 6.0)
        if cap is not None and expect > cap:
            expect = max(50, cap)
        assert render_dpi == expect

    def test_raising_the_budget_raises_the_dpi(self, profile_file):
        """The user OWNS the ceiling: a bigger budget must yield a bigger (or
        equal, once the native floor is reached) DPI — never a hidden clamp."""
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        dpis = {}
        for budget in (0.5, 64.0):
            prof = load_profile(profile_file, load_traces=True)
            cfg = dict(_cfg(), dpi=2400, paper_size="Auto",
                       mem_budget_gb=budget)
            fig, render_dpi = render_export_figure(
                prof, cfg, _params(), [], _scale_cfg(), False, "profile",
                _NoOpCancel(), view_scale=_scale_for_page(prof, cfg, 3.0, 2.0))
            fig.clear()
            dpis[budget] = render_dpi
        assert dpis[64.0] > dpis[0.5]    # 0.5 GB clamps below 2400; 64 GB doesn't
        assert dpis[64.0] == 2400        # tiny file: the full requested DPI


# The traces/cm ↔ zoom sync (traces_per_cm_on_screen + its tests) was removed
# with the manual 'Traces / cm' control: the viewport is the scale authority
# now (universal axis-wheel zoom), so there is nothing to sync a widget to.


# ── Field triage: every export logs its view decision, unconditionally ─────────

class TestViewDecisionLogging:
    """One line in app.log per export states which SIZING path ran (fixed
    paper / full line at the on-screen scale / formula) — the field-triage
    instrument. core.logger sets the sbp_studio root logger propagate=False
    (records go only to app.log), so the tests re-enable propagation."""

    def _messages(self, caplog):
        return [r.getMessage() for r in caplog.records]

    def _render(self, path, cfg_extra=None, view_scale=None):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        prof = load_profile(path, load_traces=True)
        cfg = dict(_cfg(), **(cfg_extra or {}))
        fig, _ = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_scale=view_scale)
        fig.clear()

    @pytest.fixture(autouse=True)
    def _propagate(self, monkeypatch):
        import logging
        monkeypatch.setattr(logging.getLogger("sbp_studio"), "propagate", True)

    def test_formula_path_logged(self, profile_file, caplog):
        import logging
        with caplog.at_level(logging.INFO, "sbp_studio.gui.export_headless"):
            self._render(profile_file)
        assert any("formula figsize" in m for m in self._messages(caplog))

    def test_view_scale_path_logged(self, profile_file, caplog):
        import logging
        with caplog.at_level(logging.INFO, "sbp_studio.gui.export_headless"):
            self._render(profile_file, view_scale=(2.0, 0.5))
        assert any("FULL line at the on-screen scale" in m
                   for m in self._messages(caplog))

    def test_paper_path_logged(self, profile_file, caplog):
        import logging
        with caplog.at_level(logging.INFO, "sbp_studio.gui.export_headless"):
            self._render(profile_file, cfg_extra={"paper_size": "A4"},
                         view_scale=(2.0, 0.5))
        assert any("fixed paper A4" in m for m in self._messages(caplog))


class TestDynamicPaper:
    """Figsize priority in render_export_figure: explicit paper (A4/A3/A0) >
    FULL line at the viewport scale (Auto) > figsize_for_scale formula."""

    def _fig_size(self, path, cfg_extra, view_scale=None):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        prof = load_profile(path, load_traces=True)
        cfg = dict(_cfg(), **cfg_extra)
        fig, _dpi = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_scale=view_scale)
        size = tuple(fig.get_size_inches())
        fig.clear()
        return size

    def test_auto_page_is_full_extent_times_scale(self, profile_file):
        from sbp_studio.core import load_profile
        prof = load_profile(profile_file)
        cfg = dict(_cfg(), paper_size="Auto")
        vs = (2.0, 0.5)
        w, _h = self._fig_size(profile_file, {"paper_size": "Auto"}, vs)
        assert w == pytest.approx(_expected_page(prof, cfg, vs)[0])

    def test_explicit_paper_still_wins(self, profile_file):
        size = self._fig_size(profile_file, {"paper_size": "A4"},
                              view_scale=(2.0, 0.5))
        assert size == (pytest.approx(11.69), pytest.approx(8.27))

    def test_auto_without_view_falls_back_to_formula(self, profile_file):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.tabs._render import figsize_for_scale
        prof = load_profile(profile_file)
        expect_w = figsize_for_scale(prof, _scale_cfg(), 300, 1500.0)[0]
        w, _h = self._fig_size(profile_file, {"paper_size": "Auto"})
        assert w == pytest.approx(expect_w)

    def test_degenerate_view_scale_ignored(self, profile_file):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.tabs._render import figsize_for_scale
        prof = load_profile(profile_file)
        expect_w = figsize_for_scale(prof, _scale_cfg(), 300, 1500.0)[0]
        w, _h = self._fig_size(profile_file, {"paper_size": "Auto"},
                               view_scale=(0.0, 0.5))
        assert w == pytest.approx(expect_w)


class TestProportionateDecorations:
    """Fonts are POINTS (absolute 1/72 in): on the dynamic WYSIWYG page they
    must scale with the page or a small page drowns in text (the field
    'massive fonts / cramped colorbar' report). deco_scale = √(area/ref_area),
    ref = the renderer's default 12×7 page (scale 1.0 there), clamped
    [0.5, 3.0]."""

    def test_formula(self):
        from sbp_studio.gui.export_headless import deco_scale_for
        assert deco_scale_for((12.0, 7.0)) == pytest.approx(1.0)
        assert deco_scale_for((6.0, 3.5)) == pytest.approx(0.5)   # quarter area
        assert deco_scale_for((1.0, 1.0)) == 0.5                  # clamp floor
        assert deco_scale_for((120.0, 70.0)) == 3.0               # clamp ceil
        assert deco_scale_for((0.0, 7.0)) == 1.0                  # degenerate

    def test_small_page_scales_title_and_colorbar(self, profile_file):
        """A small Auto page: the renderer-internal title (historical 10 pt)
        and colorbar label (8 pt) shrink by exactly deco_scale."""
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import (render_export_figure,
                                                    deco_scale_for, _NoOpCancel)
        prof = load_profile(profile_file, load_traces=True)
        cfg = dict(_cfg(), paper_size="Auto")
        vs = _scale_for_page(prof, cfg, 4.0, 4.0)
        fig, _ = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_scale=vs)
        s = deco_scale_for(_expected_page(prof, cfg, vs))
        assert s == 0.5                                   # clamped floor here
        seis = next(a for a in fig.axes if a.get_images())
        cbar = next(a for a in fig.axes if not a.get_images())
        assert seis.title.get_fontsize() == pytest.approx(10 * s)
        assert cbar.yaxis.label.get_size() == pytest.approx(8 * s)
        fig.clear()

    def test_direct_renderer_default_sizes_unchanged(self, profile_file):
        """CLI zero-regression pin: calling the core renderer WITHOUT
        deco_scale keeps the historical 10 pt title / 8 pt colorbar label."""
        from sbp_studio.core import load_profile
        from sbp_studio.viz.render import render_profile_figure
        prof = load_profile(profile_file, load_traces=True)
        fig = render_profile_figure(prof, prof.data, {})
        seis = next(a for a in fig.axes if a.get_images())
        cbar = next(a for a in fig.axes if not a.get_images())
        assert seis.title.get_fontsize() == pytest.approx(10.0)
        assert cbar.yaxis.label.get_size() == pytest.approx(8.0)
        fig.clear()


class TestFullLineAtViewScale:
    """The field workflow contract: the viewport authors the SCALE, never the
    extent. Whatever the zoom, the export covers the FULL line, sized by the
    per-axis scale snapshot — width from the X scale only, height from the Y
    scale only (independence is finalized by the data-box fit, tested after
    the loop fix)."""

    def _page(self, path, view_scale, margins=10.0):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        prof = load_profile(path, load_traces=True)
        cfg = dict(_cfg(), paper_size="Auto", margin_top=margins,
                   margin_bottom=margins)
        fig, _ = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_scale=view_scale)
        size = tuple(fig.get_size_inches())
        fig.clear()
        return size

    def test_full_extent_at_any_zoom_level(self, tmp_path):
        """Two different X scales (a compressed and a stretched view): BOTH
        pages span the FULL line km — page width == full_km x in_per_km
        exactly. Nothing is ever cropped by zooming."""
        import numpy as np
        from sbp_studio.core import load_profile
        big = str(tmp_path / "big.sgy")
        make_synthetic_segy(big, n_traces=600, ns=256)
        d = np.asarray(load_profile(big).dist_km, dtype=float)
        full_km = float(d[-1] - d[0])
        for ipk in (0.15, 0.6):                      # compressed / stretched
            w, _h = self._page(big, (ipk, 0.05))
            assert w == pytest.approx(full_km * ipk)

    def test_batch_payload_carries_the_scale(self, tmp_path):
        """The pool worker honors payload['view_scale']: same file, two
        scales -> saved PNG widths in the same ratio (each full-extent)."""
        from PIL import Image
        from sbp_studio.gui.export_headless import render_batch_item
        p = str(tmp_path / "b.sgy")
        make_synthetic_segy(p, n_traces=200, ns=160)
        widths = {}
        for tag, ipk in (("narrow", 0.2), ("wide", 0.4)):
            pl = _payload(p, str(tmp_path / f"{tag}.png"))
            pl["view_scale"] = (ipk, 0.05)
            out, ok, err = render_batch_item(pl)
            assert ok, err
            with Image.open(out) as im:
                widths[tag] = im.size[0]
        assert widths["wide"] / widths["narrow"] == pytest.approx(2.0, rel=0.05)
