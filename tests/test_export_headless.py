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


# ── Issue 1: WYSIWYG export extent (crop the file to the live viewport) ────────

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


class TestCropExportToView:
    def test_full_view_is_a_noop_identity(self):
        """A view covering the whole line returns the SAME object + array (no
        clone, no copy) — this is what keeps a full-line export byte-identical to
        the pre-feature behaviour and the batch/pool paths untouched."""
        from sbp_studio.gui.export_headless import crop_export_to_view
        obj, data = _synthetic_obj()
        o2, d2, p2 = crop_export_to_view(
            obj, data, 0.0, (-1.0, 11.0), (-10.0, 10_000.0), None)
        assert o2 is obj and d2 is data and p2 is None

    def test_zoomed_view_crops_both_axes(self):
        from sbp_studio.gui.export_headless import crop_export_to_view
        obj, data = _synthetic_obj()
        o2, d2, _ = crop_export_to_view(
            obj, data, 0.0, (2.0, 4.0), (10.0, 40.0), None)
        assert d2.shape[0] < data.shape[0] and d2.shape[1] < data.shape[1]
        # Clone geometry is consistent with the cropped matrix.
        assert o2.n_traces == d2.shape[1] and o2.ns == d2.shape[0]
        assert 0.0 < o2.total_km < 10.0
        # Distance axis sliced to the window (~2..4 km).
        assert o2.dist_km[0] >= 2.0 - 1e-6 and o2.dist_km[-1] <= 4.0 + 1e-6

    def test_picks_reindexed_and_filtered(self):
        """Picks shift by the crop's left column and those outside the crop's
        column/time window are dropped (so overlay picks land correctly on a
        zoomed export instead of clamping to the edge)."""
        from types import SimpleNamespace
        from sbp_studio.gui.export_headless import crop_export_to_view
        obj, data = _synthetic_obj()
        picks = [
            SimpleNamespace(trace_index=5, time_ms=20.0, id="left_out"),   # x < 2km
            SimpleNamespace(trace_index=60, time_ms=20.0, id="inside"),    # inside
            SimpleNamespace(trace_index=60, time_ms=90.0, id="time_out"),  # t > window
        ]
        o2, d2, p2 = crop_export_to_view(
            obj, data, 0.0, (2.0, 4.0), (10.0, 40.0), picks)
        kept = {p.id for p in p2}
        assert kept == {"inside"}
        inside = next(p for p in p2 if p.id == "inside")
        assert 0 <= inside.trace_index < d2.shape[1]     # re-indexed into the crop

    def test_render_export_full_view_matches_no_view(self, profile_file, tmp_path):
        """End-to-end byte-identity: rendering with a full-cover view_range yields
        the EXACT same file as passing no view_range — the WYSIWYG hook cannot
        regress the full-line export (or the batch/pool paths that pass None)."""
        import numpy as np
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        from sbp_studio.viz.render import save_figure
        p0 = load_profile(profile_file, load_traces=True)
        d = np.asarray(p0.dist_km, dtype=float)
        full_view = ((float(d.min()) - 1.0, float(d.max()) + 1.0), (-1e6, 1e6))
        digests = {}
        for tag, view in (("none", None), ("full", full_view)):
            prof = load_profile(profile_file, load_traces=True)
            fig, dpi = render_export_figure(
                prof, _cfg(), _params(), [], _scale_cfg(), False, "profile",
                _NoOpCancel(), view_range=view)
            out = str(tmp_path / f"{tag}.png")
            save_figure(fig, out, dpi=dpi, fmt="png", pdf_page="auto")
            fig.clear()
            digests[tag] = open(out, "rb").read()
        assert digests["none"] == digests["full"]

    def test_render_export_zoomed_view_shrinks_figure(self, tmp_path):
        """A genuinely zoomed view_range crops the trace span, so with a
        traces/cm scale the exported figure is physically NARROWER than the
        full-line export — proof the crop reaches the render, on a deterministic
        geometric assertion (not pixel content)."""
        import numpy as np
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        big = str(tmp_path / "big.sgy")
        make_synthetic_segy(big, n_traces=600, ns=192)
        # Horizontal scale tied to trace count → width = n_traces / tpc / 2.54.
        scale_cfg = dict(layout_mode="decoupled", mode="aspect", ratio=3.0,
                         traces_per_cm=20.0)
        p0 = load_profile(big, load_traces=True)
        d = np.asarray(p0.dist_km, dtype=float)
        lo, hi = float(d.min()), float(d.max())
        zoom = ((lo + 0.25 * (hi - lo), lo + 0.55 * (hi - lo)), (-1e6, 1e6))
        widths = {}
        for tag, view in (("full", None), ("zoom", zoom)):
            prof = load_profile(big, load_traces=True)
            fig, _dpi = render_export_figure(
                prof, _cfg(), _params(), [], scale_cfg, False, "profile",
                _NoOpCancel(), view_range=view)
            widths[tag] = float(fig.get_size_inches()[0])
            fig.clear()
        assert widths["zoom"] < widths["full"]


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
        view_figsize = (10.0, 6.0)
        cfg = dict(_cfg(), paper_size="Auto", mem_budget_gb=6.0)
        fig, render_dpi = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_figsize=view_figsize)
        fig.clear()
        expect = effective_export_dpi(view_figsize, (prof.ns, prof.n_traces),
                                      int(cfg["dpi"]))
        cap = dpi_for_budget(view_figsize, 6.0)
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
                _NoOpCancel(), view_figsize=(3.0, 2.0))
            fig.clear()
            dpis[budget] = render_dpi
        assert dpis[64.0] > dpis[0.5]    # 0.5 GB clamps below 2400; 64 GB doesn't
        assert dpis[64.0] == 2400        # tiny file: the full requested DPI


# The traces/cm ↔ zoom sync (traces_per_cm_on_screen + its tests) was removed
# with the manual 'Traces / cm' control: the viewport is the scale authority
# now (universal axis-wheel zoom), so there is nothing to sync a widget to.


# ── Field triage: every export logs its view decision, unconditionally ─────────

class TestViewDecisionLogging:
    """One line in app.log per export states which extent path ran (cropped /
    full view / no view_range) — the instrument that settles any future 'the
    export ignored my zoom' report from the log alone. core.logger sets the
    sbp_studio root logger propagate=False (records go only to app.log), so the
    tests re-enable propagation for caplog to see the records."""

    def _messages(self, caplog):
        return [r.getMessage() for r in caplog.records]

    def _render(self, path, view):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        prof = load_profile(path, load_traces=True)
        fig, _ = render_export_figure(
            prof, _cfg(), _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_range=view)
        fig.clear()

    @pytest.fixture(autouse=True)
    def _propagate(self, monkeypatch):
        import logging
        monkeypatch.setattr(logging.getLogger("sbp_studio"), "propagate", True)

    def test_no_view_range_logged(self, profile_file, caplog):
        import logging
        with caplog.at_level(logging.INFO, "sbp_studio.gui.export_headless"):
            self._render(profile_file, None)
        assert any("no view_range supplied" in m for m in self._messages(caplog))

    def test_full_view_no_crop_logged(self, profile_file, caplog):
        import logging
        with caplog.at_level(logging.INFO, "sbp_studio.gui.export_headless"):
            self._render(profile_file, ((-1e9, 1e9), (-1e9, 1e9)))
        assert any("no crop" in m for m in self._messages(caplog))

    def test_crop_bounds_logged(self, profile_file, caplog):
        import logging
        import numpy as np
        from sbp_studio.core import load_profile
        d = np.asarray(load_profile(profile_file).dist_km, dtype=float)
        lo, hi = float(d.min()), float(d.max())
        zoom = ((lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)), (-1e9, 1e9))
        with caplog.at_level(logging.INFO, "sbp_studio.gui.export_headless"):
            self._render(profile_file, zoom)
        assert any("cropped to the on-screen window" in m
                   for m in self._messages(caplog))


# ── Dynamic paper: 'Auto (match view)' page vs explicit sheets vs formula ──────

class TestDynamicPaper:
    """Figsize priority in render_export_figure: explicit paper (A4/A3/A0) >
    live viewport inches (Auto) > figsize_for_scale formula (batch/pool). The
    'A4 trap' regression pins: an Auto export must NEVER be squeezed onto a
    fixed sheet."""

    def _fig_size(self, path, cfg_extra, view_figsize=None):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.export_headless import render_export_figure, _NoOpCancel
        prof = load_profile(path, load_traces=True)
        cfg = dict(_cfg(), **cfg_extra)
        fig, _dpi = render_export_figure(
            prof, cfg, _params(), [], _scale_cfg(), False, "profile",
            _NoOpCancel(), view_figsize=view_figsize)
        size = tuple(fig.get_size_inches())
        fig.clear()
        return size

    def test_auto_page_takes_viewport_inches(self, profile_file):
        """'Auto' + a live viewport → the page IS the viewport. The WYSIWYG
        aspect-fit loop may refine the HEIGHT (decorations), but the width is
        the viewport's, exactly — never a fixed sheet's."""
        w, _h = self._fig_size(profile_file, {"paper_size": "Auto"},
                               view_figsize=(10.0, 6.0))
        assert w == pytest.approx(10.0)
        assert w != pytest.approx(11.69)          # not the old default A4 width

    def test_explicit_paper_still_wins(self, profile_file):
        """A user-chosen fixed sheet is respected verbatim (loop skipped)."""
        size = self._fig_size(profile_file, {"paper_size": "A4"},
                              view_figsize=(10.0, 6.0))
        assert size == (pytest.approx(11.69), pytest.approx(8.27))

    def test_auto_without_view_falls_back_to_formula(self, profile_file):
        """Batch/pool: 'Auto' with no live viewport sizes from the scale
        formula — the historical batch geometry, unchanged."""
        from sbp_studio.core import load_profile
        from sbp_studio.gui.tabs._render import figsize_for_scale
        prof = load_profile(profile_file)
        expect_w = figsize_for_scale(prof, _scale_cfg(), 300, 1500.0)[0]
        w, _h = self._fig_size(profile_file, {"paper_size": "Auto"})
        assert w == pytest.approx(expect_w)

    def test_degenerate_view_figsize_ignored(self, profile_file):
        from sbp_studio.core import load_profile
        from sbp_studio.gui.tabs._render import figsize_for_scale
        prof = load_profile(profile_file)
        expect_w = figsize_for_scale(prof, _scale_cfg(), 300, 1500.0)[0]
        w, _h = self._fig_size(profile_file, {"paper_size": "Auto"},
                               view_figsize=(0.0, 6.0))
        assert w == pytest.approx(expect_w)
