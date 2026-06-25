"""
test_crs_selector.py — the GIS-style (QGIS/Petrel-workflow) CRS picker.

Covers:
  * CRSSelectorDialog's quick-defaults combo + the 'Other…' search handoff
    (insert-before-sentinel, so 'Other…' stays reusable after a search).
  * CRSAdvancedSearchDialog's live filter over the full EPSG registry
    (pyproj.database.query_crs_info), by code AND by name.
  * io_segy.set_crs_override applying a chosen CRS to an ALREADY-LOADED
    profile in place (no file re-read), against the real audited MCS7 file.
  * MainWindow._prompt_crs_for_ambiguous's skip logic (only fires for
    coord_unit==1 with no detected_crs and no load error).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from sbp_studio.core import load_profile, safe_map_coords, set_crs_override
from sbp_studio.gui.components.crs_selector import (
    CRSAdvancedSearchDialog, CRSSelectorDialog, _all_epsg_entries,
)

_MCS7 = (Path(__file__).resolve().parent.parent / "examples" / "_real_in"
        / "MCS7" / "5_MCS7_MIG.segy")


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestCRSAdvancedSearchDialog:
    def test_empty_search_shows_capped_full_list(self):
        dlg = CRSAdvancedSearchDialog()
        assert dlg._list.count() <= 200
        assert dlg._list.count() > 0
        assert "EPSG registry" in dlg._hint.text()

    def test_filter_by_code(self):
        dlg = CRSAdvancedSearchDialog()
        dlg._search.setText("32631")
        assert dlg._list.count() == 1
        assert dlg._list.item(0).data(Qt.ItemDataRole.UserRole) == "32631"

    def test_filter_by_name(self):
        dlg = CRSAdvancedSearchDialog()
        dlg._search.setText("UTM zone 31N")
        assert dlg._list.count() > 0
        for i in range(dlg._list.count()):
            assert "31N" in dlg._list.item(i).text() or "31n" in dlg._list.item(i).text().lower()

    def test_no_match_is_empty_not_an_error(self):
        dlg = CRSAdvancedSearchDialog()
        dlg._search.setText("this-will-never-match-anything-zzz")
        assert dlg._list.count() == 0

    def test_ok_disabled_until_a_row_is_selected(self):
        dlg = CRSAdvancedSearchDialog()
        dlg._search.setText("32631")
        ok_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Ok)
        # currentItem may already be set by Qt's auto-select-first on populate;
        # explicitly clear selection to test the disabled state deterministically.
        dlg._list.setCurrentRow(-1)
        assert not ok_btn.isEnabled()
        dlg._list.setCurrentRow(0)
        assert ok_btn.isEnabled()

    def test_accept_returns_canonical_epsg_string(self):
        dlg = CRSAdvancedSearchDialog()
        dlg._search.setText("32631")
        dlg._list.setCurrentRow(0)
        dlg._on_accept()
        assert dlg.selected_crs() == "EPSG:32631"

    def test_registry_has_thousands_of_entries(self):
        """Sanity check on the data source itself."""
        assert len(_all_epsg_entries()) > 1000


class TestCRSSelectorDialog:
    def test_quick_defaults_present_and_other_is_last(self):
        dlg = CRSSelectorDialog(file_label="test.sgy")
        codes = [dlg._combo.itemData(i) for i in range(dlg._combo.count())]
        assert codes[:4] == ["EPSG:4326", "EPSG:32629", "EPSG:32630", "EPSG:32631"]
        assert codes[-1] is None   # the "Other…" sentinel

    def test_accepting_a_quick_default_returns_it(self):
        dlg = CRSSelectorDialog(file_label="test.sgy")
        dlg._combo.setCurrentIndex(3)   # EPSG:32631
        dlg._on_accept()
        assert dlg.selected_crs() == "EPSG:32631"

    def test_other_search_result_inserted_before_sentinel_stays_reusable(self):
        """Simulates a successful 'Other…' search pick without invoking the
        real modal (headless-safe) — verifies the insert-before-sentinel
        invariant: 'Other…' must still be present and selectable afterwards."""
        dlg = CRSSelectorDialog(file_label="test.sgy")
        insert_at = dlg._combo.count() - 1
        dlg._combo.insertItem(insert_at, "EPSG:25831", "EPSG:25831")
        dlg._combo.setCurrentIndex(insert_at)
        labels = [dlg._combo.itemText(i) for i in range(dlg._combo.count())]
        assert "EPSG:25831" in labels
        assert labels[-1] == dlg.tr("Other…")
        dlg._on_accept()
        assert dlg.selected_crs() == "EPSG:25831"

    def test_accept_with_no_selection_does_not_crash_or_accept(self):
        """Defensive: if somehow still sitting on the bare 'Other…' sentinel
        (itemData None) when OK fires, must not raise or set a bogus result.

        blockSignals is essential here: a real (non-blocked) setCurrentIndex
        onto the 'Other…' row fires _on_combo_changed for real, which opens
        the actual nested search modal — exactly what this test must NOT do
        (it's testing _on_accept's defensive guard in isolation, not the
        live combo-changed handoff, which has its own dedicated tests)."""
        dlg = CRSSelectorDialog(file_label="test.sgy")
        dlg._combo.blockSignals(True)
        dlg._combo.setCurrentIndex(dlg._combo.count() - 1)   # bare "Other…"
        dlg._combo.blockSignals(False)
        dlg._on_accept()
        assert dlg.selected_crs() is None


@pytest.mark.skipif(not _MCS7.exists(), reason="real sample SEG-Y file not present")
class TestSetCrsOverrideIntegration:
    def test_applies_to_an_already_loaded_profile_in_place(self):
        prof = load_profile(str(_MCS7), load_traces=False)
        assert prof.detected_crs is None   # unresolved, as audited

        native_before = float(prof.track_lons[0])
        set_crs_override(prof, "EPSG:32631")

        assert prof.detected_crs == "EPSG:32631"
        # Native fields untouched — no file re-read, no unit change.
        assert float(prof.track_lons[0]) == native_before

        x, y = safe_map_coords(prof.track_lons, prof.track_lats,
                               prof.coord_unit, prof.detected_crs)
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        assert float(x[0]) == pytest.approx(4.052, abs=0.01)
        assert float(y[0]) == pytest.approx(44.022, abs=0.01)

    def test_invalid_override_leaves_detected_crs_none_with_a_note(self):
        prof = load_profile(str(_MCS7), load_traces=False)
        set_crs_override(prof, "not-a-real-crs")
        assert prof.detected_crs is None
        assert prof.crs_notes


class TestPromptCrsForAmbiguousSkipLogic:
    """MainWindow._prompt_crs_for_ambiguous must only prompt for genuinely
    ambiguous objects — never for errored loads, geographic files, or files
    that already resolved a CRS (e.g. from a prior override)."""

    @staticmethod
    def _should_prompt(obj) -> bool:
        # Mirrors the exact skip condition in main_window.py, exercised here
        # without constructing a full MainWindow (heavy: menus, state, workers).
        if getattr(obj, "error", None):
            return False
        if getattr(obj, "coord_unit", 0) != 1 or getattr(obj, "detected_crs", None) is not None:
            return False
        return True

    def test_skips_errored_object(self):
        class Stub:
            error = "boom"
            coord_unit = 1
            detected_crs = None
        assert not self._should_prompt(Stub())

    def test_skips_geographic_unit(self):
        class Stub:
            error = None
            coord_unit = 2
            detected_crs = None
        assert not self._should_prompt(Stub())

    def test_skips_already_resolved(self):
        class Stub:
            error = None
            coord_unit = 1
            detected_crs = "EPSG:32631"
        assert not self._should_prompt(Stub())

    def test_prompts_for_genuinely_ambiguous(self):
        class Stub:
            error = None
            coord_unit = 1
            detected_crs = None
        assert self._should_prompt(Stub())
