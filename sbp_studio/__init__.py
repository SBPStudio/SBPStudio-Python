"""
sbp_studio — Headless computational core for SBP SEG-Y seismic profiles.

Package structure:
  sbp_studio.core  — processing, I/O, model, constants (GUI-free)
  sbp_studio.viz   — headless figure rendering (Agg/Pillow)
  sbp_studio.cli   — command-line interface

The original GUI (TopasSUITE.py) remains unchanged and continues to
work independently of this package.
"""

# ── BLAS thread pinning (MUST run before the first numpy/scipy import) ──────────
# The DSP layer parallelises over column blocks with its OWN thread pool (see
# core.processing._parallel_apply, now scaled to the real core count). If the
# underlying BLAS (OpenBLAS/MKL) ALSO spins up a thread per core inside each
# block call, the two layers fight: N column-workers × M BLAS-threads massively
# oversubscribes the CPU and is SLOWER than either alone. Pin BLAS to a single
# thread so the column-block pool is the one, clean place parallelism scales.
#
# ``setdefault`` respects an explicit user override (e.g. a power user who set
# OPENBLAS_NUM_THREADS in the environment for a BLAS-heavy custom workflow).
# This is the package root, imported before any sbp_studio submodule pulls in
# numpy, so it is the earliest reliable point to set these for both the GUI and
# CLI entry points (BLAS reads them once, at numpy import time).
import os as _os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.5.0"
