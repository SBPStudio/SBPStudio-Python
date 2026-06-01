"""
conftest.py — Pytest fixtures shared across all test modules.
"""
from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

from .make_synthetic_segy import make_synthetic_segy, make_chain_pair


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("topassuite_tests")


@pytest.fixture(scope="session")
def simple_segy(tmp_dir):
    """A minimal 50-trace SEG-Y with decimal-degree coords."""
    path = str(tmp_dir / "simple.sgy")
    return make_synthetic_segy(path, n_traces=50, ns=256, dt_us=250,
                               coord_unit=3, scalar_coord=-100)


@pytest.fixture(scope="session")
def arcsec_segy(tmp_dir):
    """SEG-Y with arc-second coords (coord_unit=2)."""
    path = str(tmp_dir / "arcsec.sgy")
    return make_synthetic_segy(path, n_traces=30, ns=128, dt_us=500,
                               coord_unit=2, scalar_coord=-100)


@pytest.fixture(scope="session")
def delay_segy(tmp_dir):
    """SEG-Y with variable delays (for delay-align tests)."""
    path = str(tmp_dir / "delay.sgy")
    return make_synthetic_segy(path, n_traces=40, ns=128, dt_us=250,
                               delay_ms=20, delay_variable=True)


@pytest.fixture(scope="session")
def spectrum_segy(tmp_dir):
    """SEG-Y with a known 2 kHz sinusoid injected."""
    path = str(tmp_dir / "spectrum.sgy")
    return make_synthetic_segy(path, n_traces=20, ns=512, dt_us=125,
                               inject_freq_hz=2000.0)


@pytest.fixture(scope="session")
def chain_pair(tmp_dir):
    """Two contiguous SEG-Y files for chain detection tests."""
    return make_chain_pair(str(tmp_dir), prefix="chain", gap_km=0.05)
