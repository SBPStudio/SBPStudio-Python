"""
placeholder.py — Simple centred message panel.

A dumb display widget: it shows whatever (already-translated) text its owner
sets. The owning tab keeps the English source literal (so ``pylupdate6`` can
extract it) and re-pushes the translated text on language change.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderView(QWidget):
    """A widget showing a single centred message set by its owner."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self._label = QLabel(text)
        self._label.setObjectName("placeholder")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        lay.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)
