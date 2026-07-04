"""
_backends.py — Hardware/library capability detection and acceleration setup.

Detected once at module-import time, never mutated afterward (constexpr-like).

Acceleration layers (in order of priority)
-------------------------------------------
1. CuPy/CUDA  — GPU operations (Hilbert, AGC, normalisation).
               Install: conda install -c conda-forge cupy cudatoolkit=<ver>
               Verify:  python -c "import cupy; cupy.array([1.0])"

2. pyfftw     — FFTW-backed FFTs replace scipy's pocketfft.
               2-5x faster Hilbert transform and bandpass filtering.
               Install: conda install -c conda-forge pyfftw
               Auto-activated here if importable.

3. ThreadPool — _parallel_apply splits trace-axis work across N_WORKERS.
               Always active; uses stdlib ThreadPoolExecutor (GIL-free for
               C-extension code like scipy/numpy).

Public API
----------
gpu_available()          -> bool
worker_count()           -> int
fftw_available()         -> bool
available_ram_bytes()    -> float  (LIVE free-RAM probe, psutil-backed)
plan_workers(bytes)      -> int    (CPU+RAM capacity planner — see below)
accel_info()             -> dict   (for CLI 'accel' subcommand)
XP                       : cupy or numpy
GPU, FFTW, N_WORKERS     : module-level constants
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np


# ── GPU (CuPy) ────────────────────────────────────────────────────────────────

def _detect_gpu() -> Tuple[Any, bool, str]:
    """Try CuPy; return (module, available, device_name)."""
    try:
        import cupy as cp
        _ = cp.array([1.0])
        try:
            dev = cp.cuda.Device(0)
            name = dev.attributes.get("DeviceName", "NVIDIA GPU")
            mem  = dev.mem_info  # (free_bytes, total_bytes)
            vram = f"{mem[1] / 1024**3:.1f} GB" if mem else "?"
        except Exception:
            name = "NVIDIA GPU"; vram = "?"
        return cp, True, f"{name} ({vram} VRAM)"
    except Exception:
        return np, False, "not available"


XP: Any
GPU: bool
_GPU_NAME: str
XP, GPU, _GPU_NAME = _detect_gpu()


# ── pyfftw ────────────────────────────────────────────────────────────────────

def _setup_fftw() -> Tuple[bool, str]:
    """
    If pyfftw is installed, register it as scipy.fft's global backend.
    This transparently accelerates ALL scipy.fft calls (Hilbert, sosfilt, etc.)
    without changing any other code.
    """
    try:
        import pyfftw
        import pyfftw.interfaces.scipy_fft as _pyfftw_scipy
        import scipy.fft as _sfft
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(30)
        _sfft.set_global_backend(_pyfftw_scipy, only=False)
        ver = getattr(pyfftw, "__version__", "?")
        return True, f"pyfftw {ver} (FFTW backend active)"
    except Exception:
        return False, "not available (install: conda install -c conda-forge pyfftw)"


FFTW: bool
_FFTW_MSG: str
FFTW, _FFTW_MSG = _setup_fftw()


# ── CPU workers ───────────────────────────────────────────────────────────────

N_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)


# ── Capacity planner — the single shared CPU + RAM budgeting authority ────────
#
# Every parallel feature (column-block DSP pools, batch export workers, load
# prefetching, …) must derive its worker count from plan_workers() instead of
# hand-rolling its own heuristic. The contract that makes "scale up on a
# workstation, never choke a field laptop" hold everywhere at once:
#
#   workers = max(1, min(cpu_limit, hard_cap, ram_budget // bytes_per_worker))
#
# The floor of 1 IS the zero-regression guarantee: on constrained hardware the
# degraded case is not a slower new code path — it is exactly today's
# sequential behaviour.

# Fraction of CURRENTLY-available RAM one parallel task may claim. Deliberately
# 0.5: leaves the other half for the OS, already-resident trace matrices, and
# whatever else the user is running. Callers with historically tighter budgets
# (e.g. export_filter's 0.30 for a single full-matrix DSP pass) pass their own.
RAM_BUDGET_FRACTION: float = 0.5

# Ceiling on any single pool regardless of core count — same rationale as
# processing._MAX_POOL_WORKERS: keeps worst-case oversubscription sane when
# several pooled features overlap (e.g. a batch export inside a busy session)
# while still using most of a high-core workstation.
HARD_WORKER_CAP: int = 12


def available_ram_bytes() -> float:
    """Best-effort probe of the RAM available RIGHT NOW (bytes).

    psutil when present (accurate, cross-OS); a conservative 4 GB estimate
    otherwise, so every budget derived from this stays protective rather
    than optimistic on an unmeasurable box. Probed LIVE on every call — free
    RAM an hour into a session is nothing like it was at import time, so
    callers must NOT cache this."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        return 4.0 * 1024 ** 3


def _effective_cores() -> int:
    """Core count the OS actually lets this process use — respects a user- or
    admin-restricted CPU affinity mask when psutil can read it, so a deliberately
    confined process never oversubscribes its allowance. Falls back to the
    plain logical core count."""
    try:
        import psutil
        aff = psutil.Process().cpu_affinity()
        if aff:
            return len(aff)
    except Exception:
        pass
    return os.cpu_count() or 2


def plan_workers(bytes_per_worker: float, *,
                 hard_cap: int = HARD_WORKER_CAP,
                 ram_fraction: float = RAM_BUDGET_FRACTION) -> int:
    """Worker count for a parallel task whose EACH worker holds about
    ``bytes_per_worker`` of peak memory (estimate it from headers/shape
    BEFORE loading anything: e.g. ns × n_traces × 4 bytes × working-copies).

    Returns ``max(1, min(cpu_limit, hard_cap, ram_budget // bytes_per_worker))``
    where ``cpu_limit`` is cores−1 (affinity-aware, live) and ``ram_budget``
    is ``ram_fraction`` of the RAM available AT CALL TIME. Always ≥ 1 — a
    task that fits sequentially today still runs sequentially on the same
    hardware tomorrow (never a new failure mode, possibly just no speedup).

    ``bytes_per_worker <= 0`` means "no meaningful per-worker footprint"
    (pure-CPU work): the result is CPU/cap-limited only."""
    cpu_limit = max(1, min(N_WORKERS, _effective_cores() - 1))
    n = min(cpu_limit, max(1, int(hard_cap)))
    if bytes_per_worker > 0:
        budget  = available_ram_bytes() * ram_fraction
        by_ram  = int(budget // float(bytes_per_worker))
        n = min(n, by_ram)
    return max(1, n)


# ── Public API ────────────────────────────────────────────────────────────────

def gpu_available() -> bool:
    """True if CuPy/CUDA is available and functional."""
    return GPU


def worker_count() -> int:
    """Number of thread-pool workers used by _parallel_apply."""
    return N_WORKERS


def fftw_available() -> bool:
    """True if pyfftw is installed and registered as scipy.fft backend."""
    return FFTW


def accel_info() -> Dict[str, Any]:
    """
    Return a dict describing all active acceleration layers.
    Used by the CLI 'accel' subcommand.
    """
    import numpy as _np
    import scipy as _sp
    info: Dict[str, Any] = {
        "cpu_cores":   os.cpu_count() or 1,
        "cpu_workers": N_WORKERS,
        "numpy":       _np.__version__,
        "scipy":       _sp.__version__,
        "gpu":         GPU,
        "gpu_device":  _GPU_NAME,
        "fftw":        FFTW,
        "fftw_status": _FFTW_MSG,
    }
    if GPU:
        try:
            import cupy as cp
            info["cupy"] = cp.__version__
            info["cuda"]  = cp.cuda.runtime.runtimeGetVersion()
        except Exception:
            pass
    return info
