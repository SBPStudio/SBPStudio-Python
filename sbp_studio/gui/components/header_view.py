"""
header_view.py — SEG-Y Header Inspector (textual header + trace-header table).

A pro-QC view onto the raw file headers, essential for diagnosing non-standard
or proprietary coordinate layouts:

  * Top panel    — the 3200-byte textual header, read-only, monospace.
  * Main panel   — every trace's raw header fields in a QTableView.

The table is driven by :class:`TraceHeaderModel`, a custom
``QAbstractTableModel`` backed DIRECTLY by the per-field NumPy arrays. Only the
cells currently scrolled into view are ever queried, so 50 000+ traces stay
fluid — a ``QTableWidget`` (one widget object per cell) would allocate millions
of items and crash.

Bidirectional sync: clicking a trace on the Map or Seismic view calls
:meth:`HeaderView.select_trace` to scroll+highlight that row; clicking a row
emits :pyattr:`HeaderView.trace_selected` so the profile can recentre.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, pyqtSignal,
)
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QLabel, QPlainTextEdit, QSplitter, QTableView,
    QVBoxLayout, QWidget,
)

from ..i18n import language_manager
from ..theme import MONO, theme


class TraceHeaderModel(QAbstractTableModel):
    """Read-only table model over a ``{label: (n_traces,) ndarray}`` dict.

    Columns = header fields, rows = trace index. Lazy by construction (Qt only
    asks for visible cells), so memory and latency are independent of trace
    count."""

    def __init__(self, headers: Optional[Dict[str, np.ndarray]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._labels: List[str] = []
        self._cols: List[np.ndarray] = []
        self._rows: int = 0
        self.set_headers(headers or {})

    def set_headers(self, headers: Dict[str, np.ndarray]) -> None:
        """Swap in a new set of per-field arrays (full model reset)."""
        self.beginResetModel()
        self._labels = list(headers.keys())
        self._cols = [np.asarray(headers[k]) for k in self._labels]
        self._rows = int(min((c.shape[0] for c in self._cols), default=0))
        self.endResetModel()

    # ── Qt model interface ───────────────────────────────────────────────────

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._labels)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._cols[index.column()][index.row()]
            if isinstance(val, np.floating):
                return f"{float(val):.3f}"
            return str(int(val) if isinstance(val, np.integer) else val)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._labels[section] if 0 <= section < len(self._labels) else None
        return str(section)          # vertical header = 0-based trace index


class HeaderView(QWidget):
    """Textual header (top) + trace-header table (bottom) with selection sync."""

    # Emitted when the USER clicks a table row (not on programmatic selection).
    trace_selected = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # Cleanup banner — shown ONLY when the core purged duplicate-timestamp
        # traces from this source, so the displayed trace count is transparent.
        self.cleanup_banner = QLabel()
        self.cleanup_banner.setWordWrap(True)
        self.cleanup_banner.setVisible(False)
        lay.addWidget(self.cleanup_banner)

        self.split = QSplitter(Qt.Orientation.Vertical)

        # Textual header — read-only, monospace, no wrap (80-col cards).
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(QFont(MONO, 9))

        # Trace-header table — NumPy-backed model (fast for 50k+ rows).
        self.model = TraceHeaderModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setFont(QFont(MONO, 9))
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.clicked.connect(self._on_clicked)

        self.split.addWidget(self.text)
        self.split.addWidget(self.table)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([150, 600])
        lay.addWidget(self.split)

        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_source(self, obj: object) -> None:
        """Load the textual header + per-trace header fields from a
        SegyProfile / ProfileChain (or clear if ``obj`` is None)."""
        if obj is None:
            self.clear()
            return
        self.text.setPlainText(getattr(obj, "text_header", "") or "")
        self.model.set_headers(getattr(obj, "trace_headers", {}) or {})
        self._update_cleanup_banner(obj)

    def clear(self) -> None:
        self.text.setPlainText("")
        self.model.set_headers({})
        self.cleanup_banner.setVisible(False)

    def _update_cleanup_banner(self, obj: object) -> None:
        """Surface duplicate-timestamp cleanup: show ``Traces: N (cleaned from M
        original — K duplicate timestamps purged)`` when the core removed any
        traces, otherwise stay hidden."""
        purged = int(getattr(obj, "n_purged", 0) or 0)
        if purged <= 0:
            self.cleanup_banner.setVisible(False)
            return
        n_traces = int(getattr(obj, "n_traces", 0) or 0)
        original = int(getattr(obj, "original_n_traces", 0) or 0)
        self.cleanup_banner.setText(self.tr(
            "Traces: {0} (cleaned from {1} original — {2} duplicate timestamps "
            "purged)").format(n_traces, original, purged))
        self.cleanup_banner.setVisible(True)

    def select_trace(self, idx: int) -> None:
        """Scroll to and highlight a trace row. Programmatic — does NOT emit
        :pyattr:`trace_selected`, so it can't feed back into the click wiring."""
        n = self.model.rowCount()
        if n <= 0:
            return
        row = max(0, min(int(idx), n - 1))
        self.table.selectRow(row)
        self.table.scrollTo(self.model.index(row, 0),
                            QAbstractItemView.ScrollHint.PositionAtCenter)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _on_clicked(self, index: QModelIndex) -> None:
        if index.isValid():
            self.trace_selected.emit(int(index.row()))

    def _restyle(self, *_) -> None:
        bg, fg, sub = theme.color("panel"), theme.color("text"), theme.color("sub")
        warn = theme.color("warn")
        self.cleanup_banner.setStyleSheet(
            f"QLabel{{color:{warn};background:{bg};padding:4px 8px;"
            f"border-bottom:1px solid {sub};}}")
        self.text.setStyleSheet(
            f"QPlainTextEdit{{background:{bg};color:{fg};border:none;}}")
        self.table.setStyleSheet(
            f"QTableView{{background:{bg};color:{fg};gridline-color:{sub};"
            f"selection-background-color:{theme.color('highlight')};}}"
            f"QHeaderView::section{{background:{bg};color:{sub};border:0;"
            f"padding:2px 6px;}}")

    def _retranslate(self, *_) -> None:
        # Column labels come from the core (raw SEG-Y field names) and are shown
        # verbatim — domain data, not translated (same policy as CMAPS keys).
        pass
