"""
test_state_lru.py — LRU eviction of trace matrices in AppState.

Verifies the RAM bound: only MAX_HOT_PROFILES trace matrices stay resident; the
oldest non-active ones are dropped back to lazy stubs and their NumPy arrays are
actually released (weakref dies after GC). Uses lightweight fake profiles — no
SEG-Y I/O — so it stays fast.
"""
from __future__ import annotations

import gc
import weakref

import numpy as np
from PyQt6.QtWidgets import QApplication

from sbp_studio.gui.state import AppState, MAX_HOT_PROFILES


def _qt():
    return QApplication.instance() or QApplication([])


class _FakeProfile:
    """Minimal stand-in exposing only what AppState touches."""

    def __init__(self, path: str, loaded: bool = True) -> None:
        self.path = path
        self.name = path.rsplit("/", 1)[-1]
        self.error = None
        if loaded:
            self.data = np.zeros((256, 512), dtype=np.float32)   # the heavy matrix
            self.amp_max = np.zeros(512, dtype=np.float32)
            self.clip_p99 = 1.0
        else:
            self.data = self.amp_max = self.clip_p99 = None


def _add_and_load(state: AppState, path: str, make_active: bool = True):
    """Add a stub, optionally select it, then deliver the loaded profile (as the
    worker would). Returns a weakref to the loaded matrix."""
    state.add_profile(_FakeProfile(path, loaded=False))
    if make_active:
        state.set_active_profile(path)
    full = _FakeProfile(path, loaded=True)
    ref = weakref.ref(full.data)
    state.update_profile_data(full)
    return ref


def _hot(state: AppState):
    return [k for k, p in state.profiles.items() if p.data is not None]


def test_lru_evicts_oldest_nonactive_and_frees_ram():
    _qt()
    state = AppState()
    refs = {i: _add_and_load(state, f"/p{i}.sgy", make_active=True)
            for i in range(MAX_HOT_PROFILES + 1)}

    p0 = state.profiles["/p0.sgy"]
    assert p0.data is None and p0.amp_max is None and p0.clip_p99 is None  # → stub
    assert state.active_profile.data is not None                          # active kept
    assert len(_hot(state)) == MAX_HOT_PROFILES                           # bounded

    gc.collect()
    assert refs[0]() is None        # the evicted matrix's RAM is actually released


def test_active_profile_is_never_evicted():
    _qt()
    state = AppState()
    _add_and_load(state, "/keep.sgy", make_active=True)
    for i in range(MAX_HOT_PROFILES + 2):
        _add_and_load(state, f"/bg{i}.sgy", make_active=False)
    assert state.profiles["/keep.sgy"].data is not None
    assert len(_hot(state)) <= MAX_HOT_PROFILES


def test_evicted_profile_reverts_to_reloadable_stub():
    _qt()
    state = AppState()
    for i in range(MAX_HOT_PROFILES + 1):
        _add_and_load(state, f"/p{i}.sgy", make_active=True)
    assert state.profiles["/p0.sgy"].data is None        # evicted → stub

    # Simulate the GUI re-loading it on re-selection (set active, worker delivers).
    state.set_active_profile("/p0.sgy")
    state.update_profile_data(_FakeProfile("/p0.sgy", loaded=True))
    assert state.profiles["/p0.sgy"].data is not None
    assert len(_hot(state)) == MAX_HOT_PROFILES


def test_remove_and_clear_keep_lru_consistent():
    _qt()
    state = AppState()
    _add_and_load(state, "/a.sgy", make_active=True)
    _add_and_load(state, "/b.sgy", make_active=False)
    state.remove_profile("/a.sgy")
    assert "/a.sgy" not in state._lru
    state.clear_profiles()
    assert state._lru == []
