"""
crs_selector.py — Two-tier, GIS-style CRS picker (QGIS/Petrel-style workflow).

Resolves the ambiguity flagged by ``io_segy._detect_crs`` for PROJECTED
(CoordinateUnits=1) SEG-Y files: standard SEG-Y has no field for the zone/
projection, so a survey in UTM metres can't be auto-detected — without a CRS,
``core.spatial.safe_map_coords`` correctly refuses to plot it (NaN) rather
than feeding raw eastings/northings to a WGS84 map.

:class:`CRSSelectorDialog` — quick combo of marine-geophysics defaults, with
a trailing "Other…" that opens :class:`CRSAdvancedSearchDialog` — a live,
synchronous search over the full EPSG registry via
``pyproj.database.query_crs_info`` (~7000 entries; the query itself is the
only slow part, ~150ms, cached after the first call — filtering it per
keystroke is sub-2ms, no debounce needed).

Both dialogs return a canonical ``"EPSG:<code>"`` string via
``.selected_crs()`` (``None`` if cancelled).
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QVBoxLayout, QWidget,
)

# Sensible marine-geophysics defaults: WGS84 + the UTM zones spanning Western
# Europe (where this app's reference surveys sit) — same convention as
# core.spatial.CRS_CATALOG's UTM block, just a short curated subset for the
# common case. The label is shown as-is (domain data, like colormap names —
# not run through translation); only the dialog's OWN chrome is translated.
_QUICK_DEFAULTS = [
    ("EPSG:4326",  "EPSG:4326 (WGS 84 Geographic)"),
    ("EPSG:32629", "EPSG:32629 (UTM 29N)"),
    ("EPSG:32630", "EPSG:32630 (UTM 30N)"),
    ("EPSG:32631", "EPSG:32631 (UTM 31N)"),
]

_MAX_SEARCH_ROWS = 200   # cap rendered rows so a near-empty filter stays snappy

_epsg_cache: Optional[List] = None   # module-level cache — query_crs_info is ~150ms


def _all_epsg_entries() -> List:
    """Lazily fetch + cache the full EPSG registry (~7000 CRSInfo entries)
    from pyproj's internal SQLite database. First call costs ~150ms; every
    call after is instant. Cheap enough to filter synchronously on every
    keystroke afterwards (~1-2ms over the full list)."""
    global _epsg_cache
    if _epsg_cache is None:
        from pyproj.database import query_crs_info
        _epsg_cache = list(query_crs_info(auth_name="EPSG"))
    return _epsg_cache


class CRSAdvancedSearchDialog(QDialog):
    """Live-filtered search over the full EPSG registry — the 'Other…' path."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Search EPSG / CRS"))
        self.setMinimumSize(480, 420)
        self._selected: Optional[str] = None

        root = QVBoxLayout(self)
        self._search = QLineEdit()
        self._search.setPlaceholderText(
            self.tr("Type an EPSG code or name (e.g. '32631' or 'UTM zone 31')…"))
        root.addWidget(self._search)

        self._list = QListWidget()
        root.addWidget(self._list, 1)

        self._hint = QLabel()
        self._hint.setObjectName("sub")
        root.addWidget(self._hint)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._search.textChanged.connect(self._refilter)
        self._list.itemDoubleClicked.connect(lambda *_: self._on_accept())
        self._list.currentItemChanged.connect(self._update_ok_enabled)

        self._entries = _all_epsg_entries()
        self._refilter("")
        self._update_ok_enabled()
        self._search.setFocus()

    def _refilter(self, text: str) -> None:
        needle = text.strip().lower()
        self._list.clear()
        if not needle:
            shown = self._entries[:_MAX_SEARCH_ROWS]
            self._hint.setText(self.tr("Type to search the full EPSG registry ({0} entries)…")
                               .format(len(self._entries)))
        else:
            matches = [e for e in self._entries
                      if needle in e.code.lower() or needle in e.name.lower()]
            shown = matches[:_MAX_SEARCH_ROWS]
            self._hint.setText(
                self.tr("{0} match(es) (showing up to {1})")
                .format(len(matches), _MAX_SEARCH_ROWS))
        for e in shown:
            item = QListWidgetItem(f"EPSG:{e.code} — {e.name}")
            item.setData(Qt.ItemDataRole.UserRole, e.code)
            self._list.addItem(item)
        self._update_ok_enabled()

    def _update_ok_enabled(self, *_args) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._list.currentItem() is not None)

    def _on_accept(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        code = item.data(Qt.ItemDataRole.UserRole)
        self._selected = f"EPSG:{code}"
        self.accept()

    def selected_crs(self) -> Optional[str]:
        return self._selected


class CRSSelectorDialog(QDialog):
    """Quick combo of marine-geophysics defaults; the trailing 'Other…' opens
    :class:`CRSAdvancedSearchDialog`. Either path resolves to a canonical
    ``EPSG:<code>`` string, retrievable via :meth:`selected_crs`."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                file_label: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Select Coordinate Reference System"))
        self.setMinimumWidth(420)
        self._selected: Optional[str] = None

        root = QVBoxLayout(self)
        if file_label:
            msg = QLabel(self.tr(
                "'{0}' uses PROJECTED coordinates (metres/feet) — SEG-Y "
                "has no field for the zone, so it can't be auto-detected. "
                "Pick the CRS used by this survey, or Cancel to leave its "
                "map track empty:").format(file_label))
            msg.setWordWrap(True)
            root.addWidget(msg)

        self._combo = QComboBox()
        for code, label in _QUICK_DEFAULTS:
            self._combo.addItem(label, code)
        self._combo.addItem(self.tr("Other…"), None)
        root.addWidget(self._combo)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._combo.currentIndexChanged.connect(self._on_combo_changed)

    def _on_combo_changed(self, index: int) -> None:
        if self._combo.itemData(index) is not None:
            return   # a real CRS (default or a previously-searched one)
        # The trailing "Other…" sentinel — open the advanced search. Insert
        # the result as a NEW item just before it (rather than overwriting
        # "Other…" itself), so searching again later is always available.
        dlg = CRSAdvancedSearchDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_crs():
            chosen = dlg.selected_crs()
            insert_at = self._combo.count() - 1
            self._combo.insertItem(insert_at, chosen, chosen)
            self._combo.setCurrentIndex(insert_at)
        else:
            self._combo.setCurrentIndex(0)   # cancelled — don't strand on "Other…"

    def _on_accept(self) -> None:
        code = self._combo.currentData()
        if code is None:
            return   # only reachable if "Other…" search is still unresolved
        self._selected = code
        self.accept()

    def selected_crs(self) -> Optional[str]:
        return self._selected
