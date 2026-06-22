"""
header_view.py — SEG-Y Header Inspector & Editor.

A pro-QC view onto the raw file headers, with full in-place editing support:

  * Textual header   — the 3200-byte block, EDITABLE, with a manual
                       ASCII/EBCDIC toggle (many modern SBP files violate the
                       SEG-Y standard and write this block in plain ASCII).
  * Binary header    — file-level dt_us is editable; ns (samples/trace) is a
                       READ-ONLY display, never editable — see "Zero file
                       corruption" note below.
  * Trace Header Calculator — SeiSee-style bulk expression edits over the
                       trace-header table (e.g. ``CDP = TraceNumber * 2``),
                       backed by a sandboxed AST evaluator (never ``eval()``).
  * Trace-header table — every trace's raw header fields, fed by
                       :class:`TraceHeaderModel`.
  * "Apply & Save"   — persists ALL pending changes (text, dt, calculator
                       edits) to the physical SEG-Y file in-place, then
                       reloads the profile so the rest of the app picks up
                       the corrected metadata immediately.

Zero file corruption
---------------------
``ns`` (samples/trace) is intentionally NOT editable here: changing the
binary header's declared trace length without resizing every trace's data
block on disk would misalign every trace boundary for any reader — there is
no in-place fix for that, only a full rewrite. The trace-header calculator
never touches the array it computes until the result has been validated
against the target field's SEG-Y integer type AND byte-width range (see
``core.header_calc`` / ``core.io_segy.patch_trace_header_field``); a bad
expression is blocked with an error message, never silently clamped.

The trace-header table is driven by :class:`TraceHeaderModel`, a custom
``QAbstractTableModel`` backed DIRECTLY by per-field NumPy arrays.  Only the
cells currently scrolled into view are ever queried, so 50 000+ traces stay
fluid — a ``QTableWidget`` (one widget object per cell) would allocate millions
of items and crash.

Bidirectional sync: clicking a trace on the Map or Seismic view calls
:meth:`HeaderView.select_trace` to scroll+highlight that row; clicking a row
emits :attr:`HeaderView.trace_selected` so the profile can recentre.

After a successful save, :attr:`HeaderView.source_refreshed` is emitted with
the freshly-reloaded :class:`SegyProfile` so the parent tab can push the
corrected metadata into :class:`AppState`.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, Qt, pyqtSignal,
)
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QRadioButton, QSpinBox, QSplitter, QTableView, QVBoxLayout,
    QWidget,
)

from ..i18n import language_manager
from ..theme import MONO, theme


# ── Binary-header field descriptors ────────────────────────────────────────────
# (profile_attr, display_label, unit_suffix, spin_min, spin_max)
# ONLY dt_us is here — it is the sole editable binary-header field. ns
# (samples/trace) is rendered as a separate read-only label; see the
# "Zero file corruption" note in the module docstring for why.
_BIN_FIELD_DEFS: Tuple = (
    ("dt_us", "Sample Interval (µs):", "µs",  1,       65535),
)


def _segyio_bin_keys() -> Tuple[int]:
    """Return (BinField.Interval,) as a plain int (lazy import)."""
    import segyio
    return (int(segyio.BinField.Interval),)


def _mono_font(size: int = 9) -> QFont:
    """A monospace font that degrades gracefully if MONO isn't installed."""
    font = QFont(MONO, size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFixedPitch(True)
    return font


def _pad_text_header_for_display(text: str) -> str:
    """Lay *text* out as the standard SEG-Y card deck: exactly 40 lines of
    80 columns, so the editor always shows the full, aligned grid — even
    though the loader right-trims trailing spaces off each line. Purely a
    display concern: the actual on-disk padding/truncation happens in
    ``io_segy._format_text_header_for_write`` when the user saves."""
    lines = (text or "").split("\n")[:40]
    lines = [ln[:80].ljust(80) for ln in lines]
    while len(lines) < 40:
        lines.append(" " * 80)
    return "\n".join(lines)


def _resolve_path(obj: object) -> Optional[str]:
    """A single on-disk file path for *obj*, or None if there isn't one
    (e.g. a multi-file ProfileChain, which has no single file to patch)."""
    path = getattr(obj, "path", None)
    if path:
        return path
    profiles = getattr(obj, "profiles", None)
    if profiles:
        return getattr(profiles[0], "path", None)
    return None


# ── Trace-header table model ────────────────────────────────────────────────────

class TraceHeaderModel(QAbstractTableModel):
    """Read-only table model over a ``{label: (n_traces,) ndarray}`` dict.

    Columns = header fields, rows = trace index.  Lazy by construction (Qt only
    asks for visible cells), so memory and latency are independent of trace
    count.
    """

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
        return str(section)


# ── Main widget ─────────────────────────────────────────────────────────────────

class HeaderView(QWidget):
    """Textual header + binary header panel + trace-header calculator +
    trace-header table, with selection sync and in-place SEG-Y header
    editing."""

    # Emitted when the USER clicks a table row (not on programmatic selection).
    trace_selected = pyqtSignal(int)
    # Emitted after a successful patch+reload: carries the updated SegyProfile.
    source_refreshed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Top action bar: in-place Save (disabled until an edit) + the
        # always-available "Save As Copy…" (writes edits to a NEW file). ──────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(0)
        self._save_btn = QPushButton()
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._save_as_btn = QPushButton()
        self._save_as_btn.setEnabled(False)
        self._save_as_btn.clicked.connect(self._on_save_as_copy_clicked)
        btn_row.addWidget(self._save_btn, 1)
        btn_row.addWidget(self._save_as_btn, 0)
        lay.addLayout(btn_row)

        # ── Cleanup banner ───────────────────────────────────────────────────
        self.cleanup_banner = QLabel()
        self.cleanup_banner.setWordWrap(True)
        self.cleanup_banner.setVisible(False)
        lay.addWidget(self.cleanup_banner)

        # ── Main splitter ────────────────────────────────────────────────────
        self.split = QSplitter(Qt.Orientation.Vertical)

        # ── Group 1: Textual header — EDITABLE, monospace, no wrap, laid out
        # as the standard 40-line x 80-col SEG-Y card deck, with a manual
        # ASCII/EBCDIC encoding toggle above it. ─────────────────────────────
        self._text_group = QGroupBox()
        text_lay = QVBoxLayout(self._text_group)
        text_lay.setContentsMargins(6, 4, 6, 6)
        text_lay.setSpacing(4)

        enc_row = QHBoxLayout()
        enc_row.setContentsMargins(0, 0, 0, 0)
        self._enc_label = QLabel()
        self._enc_ascii = QRadioButton("ASCII")
        self._enc_ebcdic = QRadioButton("EBCDIC")
        self._enc_latin1 = QRadioButton("Latin-1")
        self._enc_group = QButtonGroup(self)
        self._enc_group.addButton(self._enc_ascii, 0)
        self._enc_group.addButton(self._enc_ebcdic, 1)
        self._enc_group.addButton(self._enc_latin1, 2)
        self._enc_group.idClicked.connect(self._on_encoding_clicked)
        enc_row.addWidget(self._enc_label)
        enc_row.addWidget(self._enc_ascii)
        enc_row.addWidget(self._enc_ebcdic)
        enc_row.addWidget(self._enc_latin1)
        enc_row.addStretch(1)
        text_lay.addLayout(enc_row)
        # id → encoding string used by decode_text_header / the model field.
        self._enc_by_id = {0: "ascii", 1: "ebcdic", 2: "latin-1"}
        self._enc_radio_by_name = {
            "ascii": self._enc_ascii,
            "ebcdic": self._enc_ebcdic,
            "latin-1": self._enc_latin1,
        }

        self.text = QPlainTextEdit()
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setFont(_mono_font(9))
        self.text.textChanged.connect(self._on_text_user_edited)
        text_lay.addWidget(self.text)

        # ── Group 2: Binary header overrides — dt_us editable, ns READ-ONLY
        # (see module docstring "Zero file corruption"). ─────────────────────
        self._bin_group = QGroupBox()
        bin_form = QFormLayout(self._bin_group)
        bin_form.setContentsMargins(8, 6, 8, 6)
        bin_form.setSpacing(6)
        bin_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._bin_spins: List[QSpinBox] = []
        for _attr, label, unit, lo, hi in _BIN_FIELD_DEFS:
            spin = QSpinBox()
            spin.setRange(lo, hi)
            if unit:
                spin.setSuffix(f" {unit}")
            spin.setFont(_mono_font(9))
            spin.setMinimumWidth(110)
            spin.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
            spin.valueChanged.connect(self._mark_dirty)
            bin_form.addRow(label, spin)
            self._bin_spins.append(spin)

        self._ns_label = QLabel()
        self._ns_label.setFont(_mono_font(9))
        bin_form.addRow(self.tr("Samples per Trace:"), self._ns_label)

        # ── Group 3: Trace Header Calculator — SeiSee-style bulk expression
        # edits, sandboxed (no eval()), validated against the target field's
        # SEG-Y integer type/byte-width BEFORE anything is mutated. ──────────
        self._calc_group = QGroupBox()
        calc_lay = QVBoxLayout(self._calc_group)
        calc_lay.setContentsMargins(8, 6, 8, 6)
        calc_lay.setSpacing(4)
        calc_row = QHBoxLayout()
        calc_row.setContentsMargins(0, 0, 0, 0)
        self._calc_expr = QLineEdit()
        self._calc_expr.setPlaceholderText("CDP = TraceNumber * 2")
        self._calc_expr.setFont(_mono_font(9))
        self._calc_expr.returnPressed.connect(self._on_calc_apply)
        self._calc_apply_btn = QPushButton(self.tr("Apply"))
        self._calc_apply_btn.clicked.connect(self._on_calc_apply)
        self._calc_undo_btn = QPushButton(self.tr("Undo"))
        self._calc_undo_btn.setEnabled(False)
        self._calc_undo_btn.clicked.connect(self._on_calc_undo)
        self._calc_help_btn = QPushButton(self.tr("Help"))
        self._calc_help_btn.clicked.connect(self._on_calc_help)
        calc_row.addWidget(self._calc_expr, 1)
        calc_row.addWidget(self._calc_apply_btn)
        calc_row.addWidget(self._calc_undo_btn)
        calc_row.addWidget(self._calc_help_btn)
        calc_lay.addLayout(calc_row)
        self._calc_status = QLabel()
        self._calc_status.setWordWrap(True)
        calc_lay.addWidget(self._calc_status)

        # ── Group 4: Trace-header table — NumPy-backed model (fast for
        # 50 000+ rows). ──────────────────────────────────────────────────────
        self._table_group = QGroupBox()
        table_lay = QVBoxLayout(self._table_group)
        table_lay.setContentsMargins(6, 4, 6, 6)
        table_lay.setSpacing(0)
        self.model = TraceHeaderModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.setFont(_mono_font(9))
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.clicked.connect(self._on_clicked)
        table_lay.addWidget(self.table)

        # ── Bottom pane: binary overrides + calculator + trace table packed
        # tightly into ONE container so the user resizes only the single
        # splitter handle between the textual header and everything below. ───
        self._bottom = QWidget()
        bottom_lay = QVBoxLayout(self._bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(0)
        bottom_lay.addWidget(self._bin_group, 0)
        bottom_lay.addWidget(self._calc_group, 0)
        bottom_lay.addWidget(self._table_group, 1)   # table takes the slack

        self.split.addWidget(self._text_group)
        self.split.addWidget(self._bottom)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([180, 620])
        lay.addWidget(self.split)

        # ── Internal state ───────────────────────────────────────────────────
        self._profile: object = None   # currently displayed SegyProfile / None
        self._dirty: bool = False
        self._text_user_edited: bool = False   # True only on real user typing
        self._raw_text_bytes: bytes = b""      # cached raw 3200-byte block

        # Trace-header calculator staging (never touches disk until Save):
        #   _calc_undo_stack — [(field_name, previous_array), ...] for Undo
        #   _calc_baseline   — {field_name: array_as_first_loaded}, set once
        #                      per field on its first edit, used at Save time
        #                      to detect which fields actually differ from disk
        self._calc_undo_stack: List[Tuple[str, np.ndarray]] = []
        self._calc_baseline: Dict[str, np.ndarray] = {}

        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_source(self, obj: object) -> None:
        """Load textual header, binary fields, and trace-header table from a
        SegyProfile / ProfileChain (or clear if *obj* is None)."""
        if obj is None:
            self.clear()
            return

        self._profile = obj

        # Cache raw 3200-byte block for the ASCII/EBCDIC toggle (read-only,
        # never opens the file in write mode).
        path = _resolve_path(obj)
        self._raw_text_bytes = b""
        if path:
            try:
                from sbp_studio.core import read_raw_text_header
                self._raw_text_bytes = read_raw_text_header(path)
            except Exception:
                self._raw_text_bytes = b""

        # Encoding radio buttons reflect the auto-detected guess. setChecked()
        # alone doesn't fire idClicked (click-only signal), but we block
        # signals defensively. Toggles are usable only when raw bytes loaded.
        enc = getattr(obj, "text_header_encoding", "ascii") or "ascii"
        self._set_encoding_radio(enc)
        have_raw = bool(self._raw_text_bytes)
        for radio in self._enc_radio_by_name.values():
            radio.setEnabled(have_raw)

        # Textual header — block signals so programmatic load doesn't dirty.
        # Padded to the full 40 x 80 card grid so the layout is always visible,
        # even though the loader right-trims trailing spaces off each line.
        txt = getattr(obj, "text_header", "") or ""
        self.text.blockSignals(True)
        self.text.setPlainText(_pad_text_header_for_display(txt))
        self.text.blockSignals(False)
        self._text_user_edited = False

        # Binary header: dt_us editable, ns read-only display.
        attrs = [fd[0] for fd in _BIN_FIELD_DEFS]
        for spin, attr in zip(self._bin_spins, attrs):
            val = int(getattr(obj, attr, 0) or 0)
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._ns_label.setText(str(int(getattr(obj, "ns", 0) or 0)))

        # Trace headers + calculator staging reset (new source = clean slate).
        self.model.set_headers(getattr(obj, "trace_headers", {}) or {})
        self._calc_undo_stack = []
        self._calc_baseline = {}
        self._calc_undo_btn.setEnabled(False)
        self._calc_status.setText("")
        self._calc_expr.clear()
        self._update_cleanup_banner(obj)

        # "Save As Copy…" needs a single source file to duplicate (not a chain).
        self._save_as_btn.setEnabled(bool(path))

        # Reset dirty AFTER all programmatic updates.
        self._reset_dirty()

    def _set_encoding_radio(self, enc: str) -> None:
        """Check the radio for *enc* without firing :meth:`_on_encoding_clicked`."""
        radio = self._enc_radio_by_name.get(enc, self._enc_ascii)
        for r in self._enc_radio_by_name.values():
            r.blockSignals(True)
        radio.setChecked(True)
        for r in self._enc_radio_by_name.values():
            r.blockSignals(False)

    def clear(self) -> None:
        self._set_encoding_radio("ascii")
        for radio in self._enc_radio_by_name.values():
            radio.setEnabled(False)
        self._raw_text_bytes = b""

        self.text.blockSignals(True)
        self.text.setPlainText("")
        self.text.blockSignals(False)
        self._text_user_edited = False

        for spin in self._bin_spins:
            spin.blockSignals(True)
            spin.setValue(0)
            spin.blockSignals(False)
        self._ns_label.setText("")

        self.model.set_headers({})
        self._calc_undo_stack = []
        self._calc_baseline = {}
        self._calc_undo_btn.setEnabled(False)
        self._calc_status.setText("")
        self._calc_expr.clear()

        self.cleanup_banner.setVisible(False)
        self._save_as_btn.setEnabled(False)
        self._profile = None
        self._reset_dirty()

    def _update_cleanup_banner(self, obj: object) -> None:
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
        """Scroll to and highlight a trace row.  Programmatic — does NOT emit
        :attr:`trace_selected`, so it can't feed back into the click wiring."""
        n = self.model.rowCount()
        if n <= 0:
            return
        row = max(0, min(int(idx), n - 1))
        self.table.selectRow(row)
        self.table.scrollTo(self.model.index(row, 0),
                            QAbstractItemView.ScrollHint.PositionAtCenter)

    # ── Dirty-state helpers ───────────────────────────────────────────────────

    def _mark_dirty(self, *_) -> None:
        if not self._dirty:
            self._dirty = True
            self._save_btn.setEnabled(True)
            self._restyle_save_btn(dirty=True)

    def _reset_dirty(self) -> None:
        self._dirty = False
        self._save_btn.setEnabled(False)
        self._restyle_save_btn(dirty=False)

    def _restyle_save_btn(self, *, dirty: bool) -> None:
        warn  = theme.color("warn")
        bg    = theme.color("panel")
        sub   = theme.color("sub")
        if dirty:
            self._save_btn.setStyleSheet(
                f"QPushButton{{background:{warn};color:#fff;"
                f"font-weight:bold;padding:5px 10px;"
                f"border:none;border-radius:0;}}"
            )
        else:
            self._save_btn.setStyleSheet(
                f"QPushButton{{background:{bg};color:{sub};"
                f"padding:5px 10px;border:none;border-radius:0;}}"
                f"QPushButton:disabled{{color:{sub};}}"
            )

    # ── Textual header: ASCII / EBCDIC / Latin-1 toggle ──────────────────────

    def _on_text_user_edited(self) -> None:
        """Connected to QPlainTextEdit.textChanged. Every programmatic
        setPlainText() call in this file wraps with blockSignals(True/False),
        so textChanged only ever reaches here for genuine user keystrokes —
        that's what lets the encoding toggle tell "user typed something" apart
        from "we just re-decoded the display" without a separate
        change-tracking mechanism."""
        self._text_user_edited = True
        self._mark_dirty()

    def _on_encoding_clicked(self, click_id: int) -> None:
        """idClicked fires only on a real user click (not on setChecked()),
        so this is never triggered by set_source()'s programmatic init."""
        if not self._raw_text_bytes:
            QMessageBox.information(
                self, self.tr("No Raw Header"),
                self.tr("The raw 3200-byte textual header could not be read "
                        "from disk for this source, so it cannot be "
                        "re-decoded."))
            return

        if self._text_user_edited:
            reply = QMessageBox.question(
                self, self.tr("Discard Unsaved Edits?"),
                self.tr("Switching the encoding will re-decode the textual "
                        "header from the raw bytes, discarding your unsaved "
                        "text edits.\n\nContinue?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                # Revert the radio selection without re-triggering this slot.
                orig_enc = getattr(self._profile, "text_header_encoding", "ascii")
                self._set_encoding_radio(orig_enc)
                return

        from sbp_studio.core import decode_text_header
        enc = self._enc_by_id.get(click_id, "ascii")
        new_text = decode_text_header(self._raw_text_bytes, encoding=enc)

        self.text.blockSignals(True)
        self.text.setPlainText(_pad_text_header_for_display(new_text))
        self.text.blockSignals(False)
        self._text_user_edited = False

        if self._profile is not None:
            self._profile.text_header_encoding = enc
        self._mark_dirty()

    # ── Trace Header Calculator ──────────────────────────────────────────────

    def _show_calc_error(self, msg: str) -> None:
        self._calc_status.setText(msg)
        self._calc_status.setStyleSheet(f"QLabel{{color:{theme.color('warn')};}}")

    def _show_calc_success(self, msg: str) -> None:
        self._calc_status.setText(msg)
        self._calc_status.setStyleSheet(f"QLabel{{color:{theme.color('ok')};}}")

    def _on_calc_apply(self) -> None:
        if self._profile is None:
            self._show_calc_error(self.tr("No file loaded."))
            return

        expr_text = self._calc_expr.text().strip()
        if not expr_text:
            self._show_calc_error(
                self.tr("Enter an expression, e.g. CDP = TraceNumber * 2"))
            return

        from sbp_studio.core import (
            parse_assignment, evaluate_header_expr, validate_header_result,
            trace_field_int_range, HeaderExprError,
        )

        try:
            target, rhs = parse_assignment(expr_text)
        except HeaderExprError as exc:
            self._show_calc_error(str(exc))
            return

        headers = dict(getattr(self._profile, "trace_headers", {}) or {})
        if not headers:
            self._show_calc_error(self.tr("No trace headers loaded for this file."))
            return

        variables = {k: np.asarray(v) for k, v in headers.items()}
        try:
            result, is_float = evaluate_header_expr(rhs, variables)
        except HeaderExprError as exc:
            self._show_calc_error(str(exc))
            return

        int_range = trace_field_int_range(target)
        ok, int_values, msg = validate_header_result(target, result, is_float, int_range)
        if not ok:
            self._show_calc_error(msg)
            return

        n_traces = self.model.rowCount()
        if int_values.shape[0] != n_traces:
            self._show_calc_error(self.tr(
                "Expression produced {0} value(s) but there are {1} traces."
            ).format(int_values.shape[0], n_traces))
            return

        # Everything validated — now (and only now) stage the mutation.
        prev = np.asarray(headers.get(target, int_values)).copy()
        self._calc_undo_stack.append((target, prev))
        if target not in self._calc_baseline:
            self._calc_baseline[target] = prev.copy()
        self._calc_undo_btn.setEnabled(True)

        new_dtype = prev.dtype if prev.dtype.kind in "iu" else np.int32
        headers[target] = int_values.astype(new_dtype)
        self._profile.trace_headers = headers
        self.model.set_headers(headers)

        self._show_calc_success(self.tr(
            "Applied: {0} = {1}  ({2} traces, staged — not yet saved)"
        ).format(target, rhs, n_traces))
        self._mark_dirty()

    def _on_calc_undo(self) -> None:
        if not self._calc_undo_stack:
            return
        field, prev = self._calc_undo_stack.pop()
        headers = dict(getattr(self._profile, "trace_headers", {}) or {})
        headers[field] = prev
        self._profile.trace_headers = headers
        self.model.set_headers(headers)
        self._calc_undo_btn.setEnabled(bool(self._calc_undo_stack))
        self._show_calc_success(self.tr("Undid last change to {0}.").format(field))
        self._mark_dirty()

    def _on_calc_help(self) -> None:
        from sbp_studio.core import trace_field_names, available_functions
        fields = ", ".join(trace_field_names())
        funcs = ", ".join(available_functions())
        QMessageBox.information(
            self, self.tr("Trace Header Calculator — Help"),
            self.tr(
                "Write a single assignment: TARGET = expression\n"
                "Example:  CDP = TraceNumber * 2\n\n"
                "Available variables (trace-header fields):\n{0}\n\n"
                "Available functions:\n{1}\n\n"
                "Operators: + - * / // % ** and comparisons (<, <=, >, >=, ==, !=)\n\n"
                "Safety: expressions are parsed by a sandboxed evaluator — never "
                "eval(). If the result is floating-point (e.g. from /) or "
                "overflows the target field's SEG-Y integer type (int16/int32), "
                "the write is blocked with an error instead of corrupting the file."
            ).format(fields, funcs),
        )

    # ── Save handlers ─────────────────────────────────────────────────────────

    def _collect_pending_edits(self) -> Tuple[Optional[str], dict, Dict[str, np.ndarray]]:
        """Inspect the widgets and return ``(text_or_None, bin_updates,
        trace_field_updates)`` — the exact set of changes to write. PURE: no
        disk writes, no dialogs. Shared by both the in-place Save and the
        Save-As-Copy paths so they apply identical edits."""
        # Text: compare line-by-line (rstripped) rather than as a whole string,
        # since the editor displays the text padded to the full 40 x 80 card
        # grid — a whole-string rstrip() would only trim the very end and
        # falsely flag "changed" on every save.
        new_text     = self.text.toPlainText()
        orig_text    = getattr(self._profile, "text_header", "") or ""
        text_changed = (
            [ln.rstrip() for ln in new_text.split("\n")]
            != [ln.rstrip() for ln in orig_text.split("\n")]
        )

        # Binary-field changes (dt_us only — ns is read-only).
        bin_keys  = _segyio_bin_keys()  # (BinField.Interval,)
        bin_updates: dict = {}
        for spin, (attr, _label, _unit, _lo, _hi), bkey in zip(
                self._bin_spins, _BIN_FIELD_DEFS, bin_keys):
            new_val  = spin.value()
            orig_val = int(getattr(self._profile, attr, 0) or 0)
            if new_val != orig_val:
                bin_updates[bkey] = new_val

        # Trace-header calculator changes: only fields whose CURRENT value
        # actually differs from the baseline captured at first edit (so
        # undoing back to the original writes nothing).
        trace_field_updates: Dict[str, np.ndarray] = {}
        current_headers = getattr(self._profile, "trace_headers", {}) or {}
        for field, baseline in self._calc_baseline.items():
            current = current_headers.get(field)
            if current is not None and not np.array_equal(current, baseline):
                trace_field_updates[field] = current

        return (new_text if text_changed else None), bin_updates, trace_field_updates

    def _apply_edits_to_file(self, path: str, text_to_write: Optional[str],
                             bin_updates: dict,
                             trace_field_updates: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """Write the given edits to the file at *path*. Trace-header fields go
        first (each its own validated in-place patch), then the text/binary
        header. Returns ``(ok, error_message)``; stops at the first failure so
        a bad field can't be followed by further writes."""
        from sbp_studio.core import patch_trace_header_field, patch_segy_headers

        for field, values in trace_field_updates.items():
            ok, err = patch_trace_header_field(path, field, values)
            if not ok:
                return False, self.tr(
                    "trace-header field '{0}': {1}").format(field, err)

        if text_to_write is not None or bin_updates:
            enc = getattr(self._profile, "text_header_encoding", "ascii") or "ascii"
            ok, err = patch_segy_headers(
                path,
                text_header=text_to_write,
                text_encoding=enc,
                binary_updates=bin_updates or None,
            )
            if not ok:
                return False, err

        return True, ""

    def _hot_reload(self, path: str) -> None:
        """Reload metadata from *path* and switch the view to it, transplanting
        the in-memory trace matrix so the file isn't re-read. Notifies the
        parent tab via :attr:`source_refreshed`."""
        from sbp_studio.core import load_profile as _load_profile
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new_prof = _load_profile(path, load_traces=False)
        finally:
            QApplication.restoreOverrideCursor()

        if getattr(new_prof, "error", None):
            QMessageBox.warning(
                self, self.tr("Reload Warning"),
                self.tr("File patched but metadata reload failed:\n\n{0}").format(
                    new_prof.error),
            )
            return

        # Transplant the in-memory trace matrix (the data is byte-identical —
        # only headers changed — so re-reading the samples would be wasteful).
        old_data = getattr(self._profile, "data", None)
        if old_data is not None:
            new_prof.data     = old_data
            new_prof.amp_max  = getattr(self._profile, "amp_max",  None)
            new_prof.clip_p99 = getattr(self._profile, "clip_p99", None)

        self.set_source(new_prof)
        self.source_refreshed.emit(new_prof)

    def _on_save_clicked(self) -> None:
        if self._profile is None:
            return

        path = _resolve_path(self._profile)
        if not path:
            QMessageBox.warning(
                self, self.tr("Cannot Save"),
                self.tr(
                    "This view is showing a multi-file chain, which has no "
                    "single SEG-Y file to patch. Open an individual file to "
                    "save header edits, or use Save As Copy."),
            )
            return

        # Confirmation — writing is irreversible.
        reply = QMessageBox.warning(
            self,
            self.tr("Apply & Save to File"),
            self.tr(
                "This will PERMANENTLY modify the SEG-Y file:\n\n{0}\n\n"
                "This operation cannot be undone.\n\nProceed?"
            ).format(path),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        text_to_write, bin_updates, trace_field_updates = self._collect_pending_edits()
        if text_to_write is None and not bin_updates and not trace_field_updates:
            self._reset_dirty()
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, err = self._apply_edits_to_file(
                path, text_to_write, bin_updates, trace_field_updates)
        finally:
            QApplication.restoreOverrideCursor()

        if not ok:
            QMessageBox.critical(
                self, self.tr("Save Failed"),
                self.tr("Could not patch the SEG-Y file:\n\n{0}").format(err),
            )
            return

        self._hot_reload(path)

    def _on_save_as_copy_clicked(self) -> None:
        """Duplicate the source file (shutil.copy2), apply all pending edits to
        the COPY via the same safe patchers, then hot-reload onto the new file
        — the original is never touched."""
        if self._profile is None:
            return

        src_path = _resolve_path(self._profile)
        if not src_path:
            QMessageBox.warning(
                self, self.tr("Cannot Save As Copy"),
                self.tr(
                    "This view is showing a multi-file chain, which has no "
                    "single SEG-Y file to duplicate. Open an individual file "
                    "first."),
            )
            return

        src = Path(src_path)
        default_name = str(src.with_name(f"{src.stem}_copy{src.suffix or '.segy'}"))
        dst_path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save As Copy"), default_name,
            self.tr("SEG-Y Files (*.segy *.sgy *.seg);;All Files (*)"),
        )
        if not dst_path:
            return

        # Never let "Save As Copy" silently clobber the very file it copies from.
        if os.path.abspath(dst_path) == os.path.abspath(src_path):
            QMessageBox.warning(
                self, self.tr("Cannot Save As Copy"),
                self.tr(
                    "The destination is the same as the source file. Choose a "
                    "different name (use Apply & Save to File to overwrite the "
                    "original in place)."),
            )
            return

        text_to_write, bin_updates, trace_field_updates = self._collect_pending_edits()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            shutil.copy2(src_path, dst_path)
            ok, err = self._apply_edits_to_file(
                dst_path, text_to_write, bin_updates, trace_field_updates)
        except Exception as exc:                       # copy/IO failure
            ok, err = False, str(exc)
        finally:
            QApplication.restoreOverrideCursor()

        if not ok:
            QMessageBox.critical(
                self, self.tr("Save As Copy Failed"),
                self.tr("Could not write the copy:\n\n{0}\n\n"
                        "The original file was not modified.").format(err),
            )
            return

        QMessageBox.information(
            self, self.tr("Saved As Copy"),
            self.tr("Saved a copy with your edits:\n\n{0}\n\n"
                    "Now viewing the new file.").format(dst_path),
        )
        # Switch the app to the freshly-written copy.
        self._hot_reload(dst_path)

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
        group_frame = (
            f"QGroupBox{{color:{sub};border:1px solid {sub};border-radius:0;"
            f"margin-top:8px;padding-top:14px;font-weight:bold;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;top:2px;"
            f"color:{sub};}}"
        )
        self.text.setStyleSheet(
            f"QPlainTextEdit{{background:{bg};color:{fg};border:none;}}")
        self._text_group.setStyleSheet(
            group_frame + f"QLabel{{font-weight:normal;}}"
            f"QRadioButton{{color:{fg};font-weight:normal;}}")
        self._bin_group.setStyleSheet(
            group_frame +
            f"QSpinBox{{background:{bg};color:{fg};border:1px solid {sub};"
            f"padding:2px 4px;selection-background-color:"
            f"{theme.color('highlight')};}}"
            f"QLabel{{color:{fg};font-weight:normal;}}")
        self._calc_group.setStyleSheet(
            group_frame +
            f"QLineEdit{{background:{bg};color:{fg};border:1px solid {sub};"
            f"padding:2px 4px;}}"
            f"QLabel{{font-weight:normal;}}")
        self._table_group.setStyleSheet(group_frame)
        self.table.setStyleSheet(
            f"QTableView{{background:{bg};color:{fg};gridline-color:{sub};"
            f"selection-background-color:{theme.color('highlight')};}}"
            f"QHeaderView::section{{background:{bg};color:{sub};border:0;"
            f"padding:2px 6px;}}")
        # Save As Copy — neutral secondary button, always available.
        self._save_as_btn.setStyleSheet(
            f"QPushButton{{background:{theme.color('accent')};color:{fg};"
            f"padding:5px 12px;border:none;border-radius:0;}}"
            f"QPushButton:disabled{{color:{sub};}}")
        # Re-apply save button style without toggling dirty state.
        self._restyle_save_btn(dirty=self._dirty)

    def _retranslate(self, *_) -> None:
        self._save_btn.setText(self.tr("Apply && Save to File"))
        self._save_as_btn.setText(self.tr("Save As Copy…"))
        self._text_group.setTitle(self.tr("Textual Header (3200 bytes)"))
        self._enc_label.setText(self.tr("Encoding:"))
        # These are encoding-standard names (not prose), so the .ts entries
        # self-map them rather than translating them. The literal `or` fallback
        # is a second line of defence: it guarantees the label is never blank
        # even if a future .ts edit drops the entry (self.tr() itself already
        # falls back to the English source via i18n.TsTranslator.translate
        # returning None for an unmapped string — see that module).
        self._enc_ascii.setText(self.tr("ASCII") or "ASCII")
        self._enc_ebcdic.setText(self.tr("EBCDIC") or "EBCDIC")
        self._enc_latin1.setText(self.tr("Latin-1") or "Latin-1")
        self._bin_group.setTitle(self.tr("Binary Header Overrides"))
        self._calc_group.setTitle(self.tr("Trace Header Calculator"))
        self._calc_apply_btn.setText(self.tr("Apply"))
        self._calc_undo_btn.setText(self.tr("Undo"))
        self._calc_help_btn.setText(self.tr("Help"))
        self._table_group.setTitle(self.tr("Trace Headers"))
        # Form row labels — re-set via the layout.
        form: QFormLayout = self._bin_group.layout()
        labels = [self.tr("Sample Interval (µs):"), self.tr("Samples per Trace:")]
        for i, lbl in enumerate(labels):
            item = form.itemAt(i, QFormLayout.ItemRole.LabelRole)
            if item and item.widget():
                item.widget().setText(lbl)
