"""
test_chain_lazy_gui.py — GUI wiring for lazy ProfileChain assembly.

Locks in the fix for the LRU-vs-eager-concat crash: chain DETECTION must stay
memory-flat (no trace matrices), and the stitched matrix must be assembled in a
background worker only when the chain is actually selected in the viewer.

Threaded steps use the bounded ``wait()`` + ``processEvents()`` pattern so the
test is deterministic and can never hang.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from tests.make_synthetic_segy import make_chain_pair
from sbp_studio.core import load_profile


def _qt():
    return QApplication.instance() or QApplication([])


def _drain(win, timeout_ms: int = 5000) -> None:
    app = QApplication.instance()
    for w in list(win._workers):
        assert w.wait(timeout_ms)
    app.processEvents()


def test_detection_is_memory_flat(tmp_path):
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    p1, p2 = make_chain_pair(str(tmp_path), prefix="g", gap_km=0.05)
    for path in (p1, p2):
        win.state.add_profile(load_profile(path, load_traces=False))

    win._detect_chains()
    _drain(win)

    assert len(win.state.chains) == 1
    ch = win.state.chains[0]
    # The crash scenario can't recur: detection built only lightweight metadata.
    assert ch.data is None
    assert all(getattr(p, "data", None) is None for p in win.state.profiles.values())


def test_selecting_chain_lazily_assembles_traces(tmp_path):
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    p1, p2 = make_chain_pair(str(tmp_path), prefix="g", gap_km=0.05)
    for path in (p1, p2):
        win.state.add_profile(load_profile(path, load_traces=False))

    win._detect_chains()
    _drain(win)                      # chains detected, none assembled yet
    ch = win.state.chains[0]
    assert ch.data is None

    # Selecting row 0 must kick the lazy-assembly worker.
    win.chain_list.setCurrentRow(0)
    _drain(win)

    assert ch.data is not None       # matrix now assembled
    assert ch.data.shape[1] == ch.n_traces
    assert win.state.active_chain is ch
