"""
test_capacity_planner.py — core._backends.plan_workers, the single shared
CPU + RAM budgeting authority for every parallel feature.

The contract under test (see _backends.py):

    workers = max(1, min(cpu_limit, hard_cap, ram_budget // bytes_per_worker))

with cpu_limit = cores−1 (affinity-aware, live) and ram_budget a fraction of
the RAM available AT CALL TIME. The floor of 1 is the zero-regression
guarantee for low-end hardware. psutil is mocked throughout so results are
deterministic regardless of the machine running the suite.
"""
from __future__ import annotations

import sys
import types

import pytest

import sbp_studio.core._backends as backends


GB = 1024 ** 3


def _fake_psutil(available_bytes: float, affinity_cores: int = 32):
    """A stub psutil module: fixed available RAM + CPU affinity."""
    mod = types.ModuleType("psutil")
    mod.virtual_memory = lambda: types.SimpleNamespace(available=available_bytes)

    class _Proc:
        def cpu_affinity(self):
            return list(range(affinity_cores))

    mod.Process = _Proc
    return mod


@pytest.fixture
def big_cpu(monkeypatch):
    """Pretend the machine has plenty of cores so RAM/cap are the binding
    constraints unless a test says otherwise."""
    monkeypatch.setattr(backends, "N_WORKERS", 16)
    yield


class TestAvailableRamBytes:
    def test_uses_psutil_when_present(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(7 * GB))
        assert backends.available_ram_bytes() == 7 * GB

    def test_conservative_fallback_without_psutil(self, monkeypatch):
        broken = types.ModuleType("psutil")   # no virtual_memory attribute

        monkeypatch.setitem(sys.modules, "psutil", broken)
        assert backends.available_ram_bytes() == 4.0 * GB

    def test_probe_is_live_not_cached(self, monkeypatch):
        """Two calls must see two different RAM states — callers depend on a
        LIVE probe (free RAM changes over a session)."""
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(8 * GB))
        first = backends.available_ram_bytes()
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(2 * GB))
        second = backends.available_ram_bytes()
        assert (first, second) == (8 * GB, 2 * GB)


class TestPlanWorkers:
    def test_ram_bound_on_modest_machine(self, monkeypatch, big_cpu):
        """8 GB free × 0.5 budget ÷ 1 GB/worker → exactly 4 workers."""
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(8 * GB))
        assert backends.plan_workers(1 * GB) == 4

    def test_low_ram_floor_is_one_never_zero(self, monkeypatch, big_cpu):
        """The zero-regression guarantee: a worker footprint bigger than the
        whole budget degrades to sequential (1), never to a crash or 0."""
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(1 * GB))
        assert backends.plan_workers(4 * GB) == 1

    def test_cpu_bound_with_abundant_ram(self, monkeypatch, big_cpu):
        """1 TB free: CPU (N_WORKERS=16) and the hard cap (12) bind instead."""
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(1024 * GB))
        assert backends.plan_workers(1 * GB) == 12          # HARD_WORKER_CAP

    def test_hard_cap_override(self, monkeypatch, big_cpu):
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(1024 * GB))
        assert backends.plan_workers(1 * GB, hard_cap=3) == 3

    def test_ram_fraction_override(self, monkeypatch, big_cpu):
        """A caller with a historically tighter budget (e.g. export_filter's
        0.30) can pass its own fraction: 10 GB × 0.3 ÷ 1 GB → 3."""
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(10 * GB))
        assert backends.plan_workers(1 * GB, ram_fraction=0.3) == 3

    def test_zero_footprint_means_cpu_limited_only(self, monkeypatch, big_cpu):
        """bytes_per_worker <= 0 (pure-CPU work): RAM must not be consulted
        at all — even a tiny available-RAM reading yields the CPU/cap limit."""
        monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(0.1 * GB))
        assert backends.plan_workers(0) == 12
        assert backends.plan_workers(-1) == 12

    def test_affinity_restriction_binds(self, monkeypatch, big_cpu):
        """A process confined to 3 cores plans at most 2 workers (cores−1),
        regardless of the machine's full core count."""
        monkeypatch.setitem(sys.modules, "psutil",
                            _fake_psutil(1024 * GB, affinity_cores=3))
        assert backends.plan_workers(1 * GB) == 2

    def test_single_core_laptop_floor(self, monkeypatch):
        """Worst field laptop: 1 effective core, tiny RAM → exactly 1 worker
        (today's sequential behaviour, bit for bit)."""
        monkeypatch.setattr(backends, "N_WORKERS", 1)
        monkeypatch.setitem(sys.modules, "psutil",
                            _fake_psutil(2 * GB, affinity_cores=1))
        assert backends.plan_workers(1 * GB) == 1

    def test_no_psutil_conservative_path(self, monkeypatch, big_cpu):
        """Without psutil the 4 GB fallback drives the RAM budget: 4 × 0.5 ÷
        1 GB → 2 workers — protective, not optimistic, on unmeasurable boxes."""
        broken = types.ModuleType("psutil")
        monkeypatch.setitem(sys.modules, "psutil", broken)
        assert backends.plan_workers(1 * GB) == 2
