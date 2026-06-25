"""
metadata_inspector.py — Right-click 'Properties / Metadata' dialog.

Shows basic header stats for a sidebar-selected SegyProfile/ProfileChain
(traces, sample rate, the RAW SEG-Y CoordinateUnits flag) plus the resolved
CRS, with an "Edit CRS…" button that reuses :class:`CRSSelectorDialog`
(the same GIS-style picker the Map-tab just-in-time prompt uses — see
``_base.py``'s ``_prompt_crs_if_needed``). Applying a new CRS calls
``core.set_crs_override`` then ``AppState.notify_crs_updated`` so every tab
showing this object's track redraws (the Task 1 map-redraw mechanism).
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from .crs_selector import CRSSelectorDialog

_COORD_UNIT_LABELS = {
    1: "Length / projected (m or ft)",
    2: "Arc-seconds",
    3: "Decimal degrees",
    4: "DMS",
}


class MetadataInspectorDialog(QDialog):
    """Read-only header summary + a live CRS editor for one profile/chain."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                obj, state=None) -> None:
        super().__init__(parent)
        self._obj = obj
        self._state = state
        self.setWindowTitle(self.tr("Properties / Metadata"))
        self.setMinimumWidth(380)

        root = QVBoxLayout(self)
        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(self._info)

        crs_row = QHBoxLayout()
        self._crs_label = QLabel()
        self._crs_label.setWordWrap(True)
        crs_row.addWidget(self._crs_label, 1)
        self._btn_edit_crs = QPushButton(self.tr("Edit CRS…"))
        self._btn_edit_crs.clicked.connect(self._on_edit_crs)
        crs_row.addWidget(self._btn_edit_crs, 0)
        root.addLayout(crs_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._refresh_text()

    def _refresh_text(self) -> None:
        obj = self._obj
        name = getattr(obj, "name", None) or getattr(obj, "label", "") or ""
        n_traces = getattr(obj, "n_traces", None)
        dt_us = getattr(obj, "dt_us", None)
        coord_unit = getattr(obj, "coord_unit", None)
        unit_label = _COORD_UNIT_LABELS.get(
            coord_unit, self.tr("Unknown / unset ({0})").format(coord_unit))

        lines = [f"<b>{name}</b>"]
        lines.append(self.tr("Traces: {0}").format(
            n_traces if n_traces is not None else "—"))
        lines.append(self.tr("Sample interval: {0} µs").format(
            dt_us if dt_us is not None else "—"))
        if dt_us:
            lines.append(self.tr("Sample rate: {0:.1f} Hz").format(1_000_000.0 / dt_us))
        lines.append(self.tr("Coordinate units (raw SEG-Y): {0}").format(unit_label))
        self._info.setText("<br>".join(lines))

        crs = getattr(obj, "detected_crs", None)
        self._crs_label.setText(self.tr("CRS: {0}").format(
            crs or self.tr("Unresolved (projected, no zone in header)")))

    def _on_edit_crs(self) -> None:
        from sbp_studio.core import set_crs_override
        name = getattr(self._obj, "name", None) or getattr(self._obj, "label", "") or ""
        dlg = CRSSelectorDialog(self, file_label=name)
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.selected_crs():
            return
        set_crs_override(self._obj, dlg.selected_crs())
        self._refresh_text()
        if self._state is not None:
            self._state.notify_crs_updated(self._obj)
