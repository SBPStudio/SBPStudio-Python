"""
test_batch_parallel.py — process-parallel CLI batch export (roadmap #1).

The parallel orchestration (worker-count budgeting, as_completed progress
tally, per-file failure isolation) is tested deterministically by swapping the
ProcessPoolExecutor for a ThreadPoolExecutor — SAME submit/as_completed/shutdown
API, but in-process so there is no spawn/main-reimport flakiness under pytest.
The picklable worker contract and the estimator are tested directly, and one
opt-in real-process smoke test proves an actual pool renders correctly.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import pickle

import numpy as np
import pytest

from sbp_studio.cli import commands as C
from sbp_studio.cli.main import build_parser
from tests.make_synthetic_segy import make_synthetic_segy


def _make_batch_args(inputs, out_dir, fmt="png"):
    parser = build_parser()
    return parser.parse_args(["batch-export", *inputs, "--out", out_dir,
                              "--format", fmt])


def _payload(args, path, out):
    d = dict(vars(args))
    d["files"] = [path]
    d["out"] = out
    d["chain"] = False
    return d


@pytest.fixture
def batch_files(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    paths = []
    for i in range(4):
        p = str(d / f"line_{i}.sgy")
        make_synthetic_segy(p, n_traces=50 + i * 8, ns=192)
        paths.append(p)
    return str(d), paths


class TestBatchWorkerContract:
    """_batch_export_worker — the top-level picklable pool target. Called
    directly (in-process) so the contract is verified without a real pool."""

    def test_good_file_returns_ok_and_writes_output(self, batch_files, tmp_path):
        _in, paths = batch_files
        out = str(tmp_path / "out0.png")
        args = _make_batch_args([_in], str(tmp_path / "od"))
        res = C._batch_export_worker(_payload(args, paths[0], out))
        assert res == (out, True, "")
        assert os.path.exists(out)

    def test_bad_file_returns_failure_never_raises(self, tmp_path):
        bad = str(tmp_path / "corrupt.sgy")
        with open(bad, "wb") as fh:
            fh.write(b"definitely not segy" * 50)
        out = str(tmp_path / "bad.png")
        args = _make_batch_args([str(tmp_path)], str(tmp_path / "od"))
        out_path, ok, err = C._batch_export_worker(_payload(args, bad, out))
        assert out_path == out
        assert ok is False and err        # a message, but NO exception escaped

    def test_payload_is_picklable(self, batch_files, tmp_path):
        """spawn requires the payload to survive pickling."""
        _in, paths = batch_files
        args = _make_batch_args([_in], str(tmp_path / "od"))
        p = _payload(args, paths[0], str(tmp_path / "o.png"))
        assert pickle.loads(pickle.dumps(p))["files"] == [paths[0]]


class TestBatchWorkerBudget:
    def test_estimate_scales_with_largest_file(self, batch_files):
        _in, paths = batch_files
        est = C._estimate_batch_worker_bytes(paths)
        biggest = max(os.path.getsize(p) for p in paths)
        assert est == pytest.approx(biggest * C._BATCH_EXPORT_BYTES_PER_FILE)

    def test_estimate_survives_missing_file(self, batch_files):
        _in, paths = batch_files
        est = C._estimate_batch_worker_bytes(paths + ["/no/such/file.sgy"])
        assert est > 0


class TestBatchOrchestration:
    """cmd_batch_export's parallel branch, exercised via a ThreadPoolExecutor
    stand-in (identical API) so the as_completed tally / failure isolation /
    progress code runs deterministically in-process."""

    def _run(self, args, monkeypatch, n_workers):
        import sbp_studio.core._backends as B
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: n_workers)
        # Swap the process pool for a thread pool — same submit/as_completed/
        # shutdown(cancel_futures) surface, no spawn.
        monkeypatch.setattr(cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)
        C.cmd_batch_export(args)

    def test_parallel_path_writes_every_output(self, batch_files, tmp_path, monkeypatch):
        _in, paths = batch_files
        out_dir = tmp_path / "par"
        self._run(_make_batch_args([_in], str(out_dir)), monkeypatch, n_workers=3)
        assert sorted(os.listdir(out_dir)) == [f"line_{i}.png" for i in range(4)]

    def test_parallel_matches_sequential_bytes(self, batch_files, tmp_path, monkeypatch):
        """Same headless render path in both branches → identical files."""
        _in, paths = batch_files
        seq_dir, par_dir = tmp_path / "seq", tmp_path / "par"
        self._run(_make_batch_args([_in], str(seq_dir)), monkeypatch, n_workers=1)
        self._run(_make_batch_args([_in], str(par_dir)), monkeypatch, n_workers=3)
        for name in os.listdir(seq_dir):
            a = (seq_dir / name).read_bytes()
            b = (par_dir / name).read_bytes()
            assert a == b, f"{name} differs between sequential and parallel"

    def test_one_bad_file_does_not_abort_batch(self, batch_files, tmp_path, monkeypatch):
        _in, paths = batch_files
        with open(os.path.join(_in, "corrupt.sgy"), "wb") as fh:
            fh.write(b"nope" * 500)
        out_dir = tmp_path / "mixed"
        self._run(_make_batch_args([_in], str(out_dir)), monkeypatch, n_workers=3)
        produced = sorted(os.listdir(out_dir))
        assert len(produced) == 4                 # 4 good rendered
        assert "corrupt.png" not in produced      # the bad one skipped

    def test_single_worker_takes_sequential_path(self, batch_files, tmp_path, monkeypatch):
        """n_workers=1 must NOT construct a pool at all (byte-identical legacy
        path). Prove it by making ProcessPoolExecutor explode if touched."""
        _in, paths = batch_files
        import sbp_studio.core._backends as B
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: 1)

        def _boom(*a, **k):
            raise AssertionError("pool constructed on the 1-worker path")

        monkeypatch.setattr(cf, "ProcessPoolExecutor", _boom)
        out_dir = tmp_path / "seq_only"
        C.cmd_batch_export(_make_batch_args([_in], str(out_dir)))
        assert len(os.listdir(out_dir)) == 4


class TestBatchRealProcessPool:
    """One true multiprocessing smoke test — proves a spawned worker actually
    renders. Skipped if a real pool can't start cleanly in this environment
    (e.g. a spawn/main-reimport restriction) rather than hanging the suite."""

    def test_real_pool_renders_all(self, batch_files, tmp_path, monkeypatch):
        _in, paths = batch_files
        import sbp_studio.core._backends as B
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: 2)
        out_dir = tmp_path / "realpool"
        try:
            C.cmd_batch_export(_make_batch_args([_in], str(out_dir)))
        except (cf.process.BrokenProcessPool, OSError, RuntimeError) as exc:
            pytest.skip(f"real process pool unavailable here: {exc}")
        assert sorted(os.listdir(out_dir)) == [f"line_{i}.png" for i in range(4)]


class TestBatchRamBudgetDivision:
    """Audit fix: the parallel branch must divide ONE machine-wide raster
    budget across the pool — without it, each worker independently claimed
    _RAM_SAFE_FRACTION of the WHOLE machine's free RAM (N× oversubscription,
    the exact swap-a-laptop trap plan_workers exists to prevent)."""

    def _captured_budgets(self, args, monkeypatch, n_workers):
        import sbp_studio.core._backends as B
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: n_workers)
        monkeypatch.setattr(cf, "ProcessPoolExecutor", cf.ThreadPoolExecutor)
        seen = []
        real = C._batch_export_worker

        def spy(payload):
            seen.append(payload.get("mem_budget_gb"))
            return real(payload)

        monkeypatch.setattr(C, "_batch_export_worker", spy)
        C.cmd_batch_export(args)
        return seen

    def test_parser_default_budget_divided_across_workers(self, batch_files,
                                                          tmp_path, monkeypatch):
        """The CLI default (--mem-budget-gb 6.0, see main.py) is the WHOLE-batch
        rasteriser budget: 4 workers get 1.5 GB each — never 4 × 6 GB."""
        _in, _paths = batch_files
        budgets = self._captured_budgets(
            _make_batch_args([_in], str(tmp_path / "o")), monkeypatch, n_workers=4)
        assert budgets and all(b == pytest.approx(6.0 / 4) for b in budgets)

    def test_user_budget_split_n_ways(self, batch_files, tmp_path, monkeypatch):
        _in, _paths = batch_files
        args = _make_batch_args([_in], str(tmp_path / "o"))
        args.mem_budget_gb = 8.0
        budgets = self._captured_budgets(args, monkeypatch, n_workers=4)
        assert all(b == pytest.approx(2.0) for b in budgets)

    def test_per_worker_floor_engages_for_many_workers(self, batch_files,
                                                       tmp_path, monkeypatch):
        """A tiny total budget over many workers floors at 0.25 GB/worker so a
        single export never degrades to an unusably low pixel budget."""
        _in, _paths = batch_files
        args = _make_batch_args([_in], str(tmp_path / "o"))
        args.mem_budget_gb = 0.5
        budgets = self._captured_budgets(args, monkeypatch, n_workers=4)
        assert all(b == pytest.approx(0.25) for b in budgets)

    def test_sequential_path_keeps_budget_untouched(self, batch_files, tmp_path,
                                                    monkeypatch):
        """1 worker → the legacy in-process path: args.mem_budget_gb must stay
        exactly the parser default (6.0), never rewritten by the division."""
        import sbp_studio.core._backends as B
        _in, _paths = batch_files
        monkeypatch.setattr(B, "plan_workers", lambda *a, **k: 1)
        args = _make_batch_args([_in], str(tmp_path / "o"))
        C.cmd_batch_export(args)
        assert args.mem_budget_gb == pytest.approx(6.0)
