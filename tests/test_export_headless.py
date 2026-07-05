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
