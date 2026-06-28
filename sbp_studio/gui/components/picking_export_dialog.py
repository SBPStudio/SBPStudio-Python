"""
picking_export_dialog.py — "Export / Import" chooser for interpretation picks.

A small first-step dialog (mirrors metadata_inspector.py's lightweight style):
the user picks ONE of "Exportar" / "Importar"; the calling tab then drives the
actual native QFileDialog + core.picking I/O, exactly like the existing FIX-
marks export flow (_base.py's _on_export_fix_requested).

"Importar" is enabled ONLY when the current pick list is empty — importing
REPLACES the list wholesale (SeismicView.set_picks), so this prevents an
accidental import from silently destroying unsaved interpretation work.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)


class PickingExportImportDialog(QDialog):
    """Returns which action the user chose via :meth:`choice` after ``exec()``
    accepts (``EXPORT`` / ``IMPORT``), or ``None`` if cancelled."""

    EXPORT = "export"
    IMPORT = "import"

    def __init__(self, parent: Optional[QWidget] = None, *,
                has_existing_picks: bool) -> None:
        super().__init__(parent)
        self._choice: Optional[str] = None
        self._has_existing_picks = bool(has_existing_picks)
        self.setWindowTitle(self.tr("Interpretation Markers"))
        self.setMinimumWidth(320)

        root = QVBoxLayout(self)
        self._lbl = QLabel()
        self._lbl.setWordWrap(True)
        root.addWidget(self._lbl)

        row = QHBoxLayout()
        self.btn_export = QPushButton()
        self.btn_export.clicked.connect(self._on_export)
        row.addWidget(self.btn_export)
        self.btn_import = QPushButton()
        self.btn_import.setEnabled(not self._has_existing_picks)
        self.btn_import.clicked.connect(self._on_import)
        row.addWidget(self.btn_import)
        root.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._retranslate()

    def _retranslate(self) -> None:
        self._lbl.setText(self.tr(
            "Export the current interpretation markers, or import a "
            "previously saved session (.tps)."))
        self.btn_export.setText(self.tr("Export"))
        self.btn_import.setText(self.tr("Import"))
        if self._has_existing_picks:
            self.btn_import.setToolTip(self.tr(
                "Import is disabled while markers are already placed — "
                "importing replaces the current list. Clear the markers first."))
        else:
            self.btn_import.setToolTip("")

    def _on_export(self) -> None:
        self._choice = self.EXPORT
        self.accept()

    def _on_import(self) -> None:
        self._choice = self.IMPORT
        self.accept()

    def choice(self) -> Optional[str]:
        return self._choice
