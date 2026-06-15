"""
test_import_chains.py — Batch import of chains from a campaign directory.

Covers the core ``import_chains_from_directory`` workflow accelerator:
  * one ProfileChain per subdirectory that holds SEG-Y files, named after it;
  * empty / non-SEG-Y subdirectories are skipped;
  * duplicates (already-present names + intra-batch) are skipped;
  * RAM-flat: chains and their constituent profiles stay lazy stubs (data=None);
  * progress is reported and a pre-cancelled token aborts the scan.
"""
from __future__ import annotations

import os

import pytest

from tests.make_synthetic_segy import make_synthetic_segy
from sbp_studio.core import import_chains_from_directory
from sbp_studio.core.tasks import CancelToken, Cancelled


def _campaign(tmp_path):
    """A campaign root: Line_A (2 SEG-Y), Line_B (1), Empty (none),
    Docs (a .txt, no SEG-Y)."""
    root = tmp_path / "campaign"
    (root / "Line_A").mkdir(parents=True)
    (root / "Line_B").mkdir(parents=True)
    (root / "Empty").mkdir(parents=True)
    (root / "Docs").mkdir(parents=True)
    make_synthetic_segy(str(root / "Line_A" / "a1.sgy"), n_traces=20, base_lon=-3.0)
    make_synthetic_segy(str(root / "Line_A" / "a2.sgy"), n_traces=20, base_lon=-2.9)
    make_synthetic_segy(str(root / "Line_B" / "b1.sgy"), n_traces=20, base_lon=10.0)
    (root / "Docs" / "notes.txt").write_text("not seismic")
    return root


def test_imports_one_chain_per_valid_subdir(tmp_path):
    root = _campaign(tmp_path)
    chains = import_chains_from_directory(str(root))
    names = sorted(c.label for c in chains)
    assert names == ["Line_A", "Line_B"]            # Empty + Docs skipped
    by_name = {c.label: c for c in chains}
    assert len(by_name["Line_A"].profiles) == 2     # both SEG-Y grouped
    assert len(by_name["Line_B"].profiles) == 1


def test_import_is_memory_flat(tmp_path):
    root = _campaign(tmp_path)
    chains = import_chains_from_directory(str(root))
    for ch in chains:
        assert ch.data is None                      # chain stays lazy
        assert ch.clip_p99 is None
        # n_traces summed from stubs, available without any matrix load
        assert ch.n_traces == sum(p.n_traces for p in ch.profiles)
        assert all(getattr(p, "data", None) is None for p in ch.profiles)


def test_existing_names_are_skipped(tmp_path):
    root = _campaign(tmp_path)
    chains = import_chains_from_directory(str(root), existing_names=["Line_A"])
    assert sorted(c.label for c in chains) == ["Line_B"]


def test_empty_root_returns_nothing(tmp_path):
    root = tmp_path / "blank"
    root.mkdir()
    assert import_chains_from_directory(str(root)) == []


def test_progress_is_reported(tmp_path):
    root = _campaign(tmp_path)
    seen = []
    import_chains_from_directory(str(root), progress=lambda f, m: seen.append((f, m)))
    assert seen and seen[-1][0] == 1.0              # finishes at 100 %
    # The subdirectory names are surfaced as progress messages.
    assert any(m == "Line_A" for _f, m in seen)


def test_cancelled_token_aborts(tmp_path):
    root = _campaign(tmp_path)
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        import_chains_from_directory(str(root), cancel=token)
