"""
test_crs_map_redraw.py — the three CRS UX architectural changes:

1. Map redraw mechanism: AppState.crs_updated + SubTabbedTab._refresh_map_track
   (the 'ghost update' fix — the map didn't know set_crs_override had run).
2. Just-in-time CRS prompting on the Map sub-tab (not at file-load time).
3. MetadataInspectorDialog's 'Edit CRS…' → set_crs_override → notify →
   redraw round-trip.

All exercised against the real, audited MCS7 file (projected, no zone in
header) so these are genuine end-to-end checks, not just mocked plumbing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QDialog

from sbp_studio.core import load_profile
from sbp_studio.gui.components.crs_selector import CRSSelectorDialog
from sbp_studio.gui.components.metadata_inspector import MetadataInspectorDialog
from sbp_studio.gui.state import AppState
from sbp_studio.gui.tabs.visualizer_tab import VisualizerTab

_MCS7 = (Path(__file__).resolve().parent.parent / "examples" / "_real_in"
        / "MCS7" / "5_MCS7_MIG.segy")
_skip_no_sample = pytest.mark.skipif(not _MCS7.exists(), reason="real sample file not present")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


class _FakeTasks:
    def run_task(self, *a, **k): pass
    def notify(self, *a, **k): pass
    def show_error(self, *a, **k): pass


def _accept_with(crs: str):
    """Monkeypatch helper: makes CRSSelectorDialog.exec() behave as if the
    user picked `crs` and hit OK, without opening a real modal."""
    def fake_exec(self):
        self._selected = crs
        return QDialog.DialogCode.Accepted
    return fake_exec


def _reject():
    return lambda self: QDialog.DialogCode.Rejected


@_skip_no_sample
class TestMapRedrawAndJustInTimePrompt:
    def _tab_with_active_profile(self):
        state = AppState()
        tab = VisualizerTab(state, _FakeTasks())
        prof = load_profile(str(_MCS7), load_traces=False)
        state.add_profile(prof)
        state.set_active_profile(prof.path)
        return state, tab, prof

    def test_loading_does_not_prompt_map_stays_nan(self):
        """Task 2: no dialog at load time — confirmed by the map track simply
        being unresolved (NaN), since nothing ever called set_crs_override."""
        state, tab, prof = self._tab_with_active_profile()
        assert prof.detected_crs is None
        assert tab._map._x is None or np.all(np.isnan(tab._map._x))

    def test_switching_to_map_tab_prompts_and_cancel_leaves_it_unresolved(self):
        orig = CRSSelectorDialog.exec
        CRSSelectorDialog.exec = _reject()
        try:
            state, tab, prof = self._tab_with_active_profile()
            tab._on_subtab_changed(1)   # MAP index
        finally:
            CRSSelectorDialog.exec = orig
        assert prof.detected_crs is None
        assert tab._map._x is None or np.all(np.isnan(tab._map._x))

    def test_switching_to_map_tab_prompts_and_accept_resolves_and_redraws(self):
        """The core fix for Task 1 + Task 2 together: accepting a CRS at the
        Map-tab prompt must BOTH set detected_crs AND make the map track
        finite immediately — no separate manual refresh needed."""
        orig = CRSSelectorDialog.exec
        CRSSelectorDialog.exec = _accept_with("EPSG:32631")
        try:
            state, tab, prof = self._tab_with_active_profile()
            tab._on_subtab_changed(1)
        finally:
            CRSSelectorDialog.exec = orig
        assert prof.detected_crs == "EPSG:32631"
        assert np.all(np.isfinite(tab._map._x)) and np.all(np.isfinite(tab._map._y))
        assert float(tab._map._x[0]) == pytest.approx(4.052, abs=0.01)
        assert float(tab._map._y[0]) == pytest.approx(44.022, abs=0.01)

    def test_already_resolved_crs_is_not_re_prompted(self):
        """No dialog opens a second time once a CRS is set — exec() would
        raise if called, since we don't patch it here."""
        state, tab, prof = self._tab_with_active_profile()
        from sbp_studio.core import set_crs_override
        set_crs_override(prof, "EPSG:32631")
        tab._on_subtab_changed(1)   # must be a silent no-op (guard skips it)
        assert prof.detected_crs == "EPSG:32631"   # unchanged, no crash

    def test_crs_updated_signal_only_redraws_the_active_objects_own_map(self):
        """A CRS change on a DIFFERENT (inactive) profile must NOT touch this
        tab's map — _on_crs_updated checks identity against _active_object()."""
        state, tab, prof = self._tab_with_active_profile()
        other = load_profile(str(_MCS7), load_traces=False)   # a second, distinct object
        from sbp_studio.core import set_crs_override
        set_crs_override(other, "EPSG:32631")
        state.notify_crs_updated(other)
        # tab is showing `prof`, not `other` — its map must stay untouched/NaN.
        assert tab._map._x is None or np.all(np.isnan(tab._map._x))

    def test_no_infinite_loop_set_track_does_not_retrigger_anything(self):
        """Sanity guard matching the task's explicit ask: redrawing the map
        (set_track) must not itself emit crs_updated or change the subtab
        index, so the chain set_crs_override -> notify -> refresh terminates."""
        state, tab, prof = self._tab_with_active_profile()
        calls = []
        state.crs_updated.connect(lambda obj: calls.append(obj))
        from sbp_studio.core import set_crs_override
        set_crs_override(prof, "EPSG:32631")
        state.notify_crs_updated(prof)   # this is what _prompt_crs_if_needed does
        assert len(calls) == 1   # exactly the one explicit notification, no echo


@_skip_no_sample
class TestMetadataInspectorEditCrsRoundTrip:
    def test_dialog_shows_basic_stats_and_unresolved_crs(self):
        prof = load_profile(str(_MCS7), load_traces=False)
        dlg = MetadataInspectorDialog(obj=prof, state=None)
        text = dlg._info.text()
        assert str(prof.n_traces) in text
        assert str(prof.dt_us) in text
        assert "projected" in text.lower() or "metres" in text.lower() or "ft" in text.lower()
        assert "Unresolved" in dlg._crs_label.text()

    def test_edit_crs_applies_updates_text_and_notifies_state(self):
        state = AppState()
        prof = load_profile(str(_MCS7), load_traces=False)
        state.add_profile(prof)
        received = []
        state.crs_updated.connect(lambda obj: received.append(obj))

        dlg = MetadataInspectorDialog(obj=prof, state=state)
        orig = CRSSelectorDialog.exec
        CRSSelectorDialog.exec = _accept_with("EPSG:32631")
        try:
            dlg._on_edit_crs()
        finally:
            CRSSelectorDialog.exec = orig

        assert prof.detected_crs == "EPSG:32631"
        assert "EPSG:32631" in dlg._crs_label.text()
        assert received == [prof]

    def test_edit_crs_cancel_does_not_apply_or_notify(self):
        state = AppState()
        prof = load_profile(str(_MCS7), load_traces=False)
        state.add_profile(prof)
        received = []
        state.crs_updated.connect(lambda obj: received.append(obj))

        dlg = MetadataInspectorDialog(obj=prof, state=state)
        orig = CRSSelectorDialog.exec
        CRSSelectorDialog.exec = _reject()
        try:
            dlg._on_edit_crs()
        finally:
            CRSSelectorDialog.exec = orig

        assert prof.detected_crs is None
        assert received == []

    def test_works_without_a_state_reference(self):
        """state=None (e.g. a future standalone usage) must not crash —
        applying a CRS still works, it just can't notify any map."""
        prof = load_profile(str(_MCS7), load_traces=False)
        dlg = MetadataInspectorDialog(obj=prof, state=None)
        orig = CRSSelectorDialog.exec
        CRSSelectorDialog.exec = _accept_with("EPSG:32631")
        try:
            dlg._on_edit_crs()
        finally:
            CRSSelectorDialog.exec = orig
        assert prof.detected_crs == "EPSG:32631"
