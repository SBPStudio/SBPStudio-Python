"""
test_add_to_map.py — Batch 'Add to map' workflow accelerator.

Guards the two hard requirements of the sidebar batch loader:

  * MEMORY  — adding navigation tracks to the map reads ONLY the spatial
              vectors already on the header stubs; it never loads a trace
              matrix and never touches the LRU hot set (RAM stays flat).
  * UX STATE — it never changes the active profile or disturbs the live
              seismic section; multi-selecting items never triggers a load.

The CoreWorker is awaited with a bounded ``wait()`` + ``processEvents()`` so the
test is deterministic and can never hang (same pattern as test_cancellation).
"""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from tests.make_synthetic_segy import make_synthetic_segy
from sbp_studio.core import load_profile


def _qt():
    return QApplication.instance() or QApplication([])


def _drain(win, timeout_ms: int = 5000) -> None:
    """Await every running CoreWorker, then deliver its queued success signal."""
    app = QApplication.instance()
    for w in list(win._workers):
        assert w.wait(timeout_ms)        # bounded → can never hang
    app.processEvents()                  # run _on_tracks_extracted on the GUI thread


def _make_stubs(tmp_path, n: int) -> list:
    """N header-only stub profiles (data=None) with real navigation tracks."""
    stubs = []
    for i in range(n):
        p = make_synthetic_segy(str(tmp_path / f"line{i:02d}.sgy"), n_traces=40,
                                base_lon=-3.0 + 0.01 * i, base_lat=43.0)
        stubs.append(load_profile(p, load_traces=False))
    return stubs


def test_add_ten_keeps_ram_flat_and_section_untouched(tmp_path):
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    stubs = _make_stubs(tmp_path, 10)
    for s in stubs:
        win.state.add_profile(s)

    # Pin an active profile + record the LRU baseline; the batch add must not
    # move either.
    active_before = win.state._active_profile_key
    lru_before = list(win.state._lru)
    map_view = win.tab_visualizer.map_view

    win._tracks_to_map(stubs, win.tab_visualizer)
    _drain(win)

    # MEMORY: no trace matrix was loaded and the LRU hot set is unchanged.
    assert all(getattr(s, "data", None) is None for s in stubs)
    assert win.state._lru == lru_before

    # UX STATE: active profile + section completely undisturbed.
    assert win.state._active_profile_key == active_before

    # All ten tracks landed as distinct managed map layers.
    assert map_view.layer_list.count() == 10
    assert map_view._layer_count == 10


def test_multi_selection_does_not_load_a_profile(tmp_path):
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    for s in _make_stubs(tmp_path, 3):
        win.state.add_profile(s)

    # Select ALL rows (range select) — the guard must keep the active profile
    # None and start no loading worker.
    win.prof_list.selectAll()
    QApplication.instance().processEvents()
    assert win.state._active_profile_key is None
    assert all(getattr(p, "data", None) is None
               for p in win.state.profiles.values())


def test_add_button_enabled_only_with_valid_selection(tmp_path):
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    assert win._btn_prof_to_map.isEnabled() is False     # nothing selected
    for s in _make_stubs(tmp_path, 2):
        win.state.add_profile(s)
    win._refresh_profile_list()

    win.prof_list.selectAll()
    QApplication.instance().processEvents()
    assert win._btn_prof_to_map.isEnabled() is True

    win.prof_list.clearSelection()
    QApplication.instance().processEvents()
    assert win._btn_prof_to_map.isEnabled() is False


def test_errored_profiles_are_skipped(tmp_path):
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    good = _make_stubs(tmp_path, 1)[0]
    bad = load_profile(str(tmp_path / "missing.sgy"), load_traces=False)  # .error set
    assert getattr(bad, "error", None)
    win.state.add_profile(good)
    win.state.add_profile(bad)

    win.prof_list.selectAll()
    QApplication.instance().processEvents()
    # Only the valid profile counts toward the selection / button.
    assert win._selected_profiles() == [good]
    assert win._btn_prof_to_map.isEnabled() is True

    win._tracks_to_map(win._selected_profiles(), win.tab_visualizer)
    _drain(win)
    assert win.tab_visualizer.map_view.layer_list.count() == 1
