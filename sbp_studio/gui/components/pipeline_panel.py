"""
pipeline_panel.py — Dynamic workflow pipeline UI (drag-drop node list + editor).

Replaces the static hardcoded filter checkboxes with a reorderable list of DSP
nodes. The user adds/removes/reorders modules; selecting one populates a
property editor built dynamically from that node's :class:`ParamSpec` list.

Performance contract
--------------------
* Parameter edits are **debounced** (single-shot 200 ms QTimer): the heavy
  recompute fires only after the user stops dragging.
* Structural changes (add / remove / reorder) emit immediately.
* The panel only OWNS the node list + params; it never processes data. It
  emits :pyattr:`pipeline_changed`; a controller runs the (ViewBox-limited,
  memoized) :class:`~sbp_studio.gui.dsp.Pipeline` off-thread.

This is Phase-1 MVP scaffolding: only the AGC node is registered. The panel is
fully generic, so Phase 2 nodes appear automatically once added to
``NODE_REGISTRY``.
"""
from __future__ import annotations

from typing import List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from ..dsp import (
    NODE_REGISTRY, ChoiceSpec, DSPNode, ParamSpec, tr_node, tr_param,
)
from ..i18n import language_manager

_NODE_ROLE = Qt.ItemDataRole.UserRole


class _ParamRow(QWidget):
    """A labelled slider+spinbox bound to one node param. Emits on value change."""

    changed = pyqtSignal()

    def __init__(self, node: DSPNode, spec: ParamSpec,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._spec = spec
        self._scale = 10 ** spec.decimals  # slider works in integer steps
        self._is_int = spec.decimals == 0

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(spec.lo * self._scale), int(spec.hi * self._scale))
        self.slider.setSingleStep(max(1, int(spec.step * self._scale)))

        if spec.decimals > 0:
            self.spin: QDoubleSpinBox | QSpinBox = QDoubleSpinBox()
            self.spin.setDecimals(spec.decimals)
            self.spin.setRange(float(spec.lo), float(spec.hi))
            self.spin.setSingleStep(float(spec.step))
        else:
            # QSpinBox is integer-only: its range/step/value must be ints.
            self.spin = QSpinBox()
            self.spin.setRange(int(spec.lo), int(spec.hi))
            self.spin.setSingleStep(max(1, int(spec.step)))
        self.spin.setFixedWidth(72)
        if spec.unit:
            self.spin.setSuffix(f" {spec.unit}")

        val = float(node.params.get(spec.name, spec.default))
        self._set_value(val)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin, 0)

    def _set_value(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(int(round(val * self._scale)))
        self.spin.setValue(int(round(val)) if self._is_int else val)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def _commit(self, val: float) -> None:
        self._node.params[self._spec.name] = val
        self.changed.emit()

    def _on_slider(self, raw: int) -> None:
        val = raw / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(int(round(val)) if self._is_int else val)
        self.spin.blockSignals(False)
        self._commit(val)

    def _on_spin(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(val * self._scale)))
        self.slider.blockSignals(False)
        self._commit(float(val))


class _ChoiceRow(QWidget):
    """A combo box bound to one categorical node param. Emits on selection."""

    changed = pyqtSignal()

    def __init__(self, node: DSPNode, spec: ChoiceSpec,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._spec = spec

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        for value, display in spec.choices:   # choices are core domain data (not tr'd)
            self.combo.addItem(display, value)
        cur = node.params.get(spec.name, spec.default)
        idx = self.combo.findData(cur)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.currentIndexChanged.connect(self._on_change)
        lay.addWidget(self.combo, 1)

    def _on_change(self, *_) -> None:
        self._node.params[self._spec.name] = self.combo.currentData()
        self.changed.emit()


class PipelinePanel(QWidget):
    """Reorderable DSP node list + dynamic property editor (debounced)."""

    # Emitted (debounced for param edits, immediate for structural edits) when
    # the effective pipeline changes. A controller listens and re-runs preview.
    pipeline_changed = pyqtSignal()

    DEBOUNCE_MS = 200

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self.pipeline_changed.emit)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        self._hdr_pipeline = QLabel()
        self._hdr_pipeline.setObjectName("section")
        root.addWidget(self._hdr_pipeline)

        # ── Node list (drag-drop reorder) ──
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentItemChanged.connect(lambda *_: self._rebuild_editor())
        # rowsMoved fires after a drag-drop reorder completes.
        self.list.model().rowsMoved.connect(self._on_reordered)
        root.addWidget(self.list, 1)

        # ── Add / Remove ──
        btns = QHBoxLayout()
        self.btn_add = QPushButton()
        self.btn_remove = QPushButton()
        self.btn_add.clicked.connect(self._show_add_menu)
        self.btn_remove.clicked.connect(self._remove_selected)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_remove)
        btns.addStretch(1)
        root.addLayout(btns)

        line = QFrame()
        line.setObjectName("hline")
        root.addWidget(line)

        # ── Property editor (rebuilt per selection) ──
        self._hdr_props = QLabel()
        self._hdr_props.setObjectName("section")
        root.addWidget(self._hdr_props)

        self._editor_host = QWidget()
        self._editor_form = QFormLayout(self._editor_host)
        self._editor_form.setContentsMargins(0, 0, 0, 0)
        self._editor_rows: List[QWidget] = []
        root.addWidget(self._editor_host)
        self._editor_hint = QLabel()
        self._editor_hint.setObjectName("sub")
        self._editor_hint.setWordWrap(True)
        root.addWidget(self._editor_hint)

        root.addStretch(1)

        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()
        self._rebuild_editor()

    # ── Public API ──────────────────────────────────────────────────────────

    def nodes(self) -> List[DSPNode]:
        """Current ordered node list (top of the QListWidget = first applied)."""
        out: List[DSPNode] = []
        for i in range(self.list.count()):
            node = self.list.item(i).data(_NODE_ROLE)
            if node is not None:
                out.append(node)
        return out

    def add_node(self, node: DSPNode) -> None:
        item = QListWidgetItem(tr_node(node.KEY, node.DISPLAY))
        item.setData(_NODE_ROLE, node)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.pipeline_changed.emit()   # structural → immediate

    # ── Structural actions ──────────────────────────────────────────────────

    def _show_add_menu(self) -> None:
        menu = QMenu(self)
        for cls in NODE_REGISTRY:
            act = QAction(tr_node(cls.KEY, cls.DISPLAY), self)
            act.triggered.connect(lambda _=False, c=cls: self.add_node(c()))
            menu.addAction(act)
        menu.exec(self.btn_add.mapToGlobal(self.btn_add.rect().bottomLeft()))

    def _remove_selected(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        self.list.takeItem(row)
        self._rebuild_editor()
        self.pipeline_changed.emit()   # structural → immediate

    def _on_reordered(self, *_) -> None:
        # Drag-drop reorder changes order → pipeline differs → immediate.
        self.pipeline_changed.emit()

    # ── Property editor ─────────────────────────────────────────────────────

    def _rebuild_editor(self) -> None:
        # Tear down previous rows.
        for roww in self._editor_rows:
            roww.deleteLater()
        self._editor_rows.clear()
        while self._editor_form.rowCount():
            self._editor_form.removeRow(0)

        item = self.list.currentItem()
        node = item.data(_NODE_ROLE) if item is not None else None
        if node is None:
            self._editor_hint.setText(self.tr("Select a module to edit its parameters."))
            return
        self._editor_hint.setText("")
        for spec in node.SPECS:
            row = (_ChoiceRow(node, spec) if isinstance(spec, ChoiceSpec)
                   else _ParamRow(node, spec))
            row.changed.connect(self._schedule)   # param edit → debounced
            self._editor_rows.append(row)
            self._editor_form.addRow(tr_param(spec.label), row)

    def _schedule(self) -> None:
        """Restart the debounce timer; recompute fires only after settle."""
        self._debounce.start()

    # ── i18n ────────────────────────────────────────────────────────────────

    def retranslate_ui(self) -> None:
        self._hdr_pipeline.setText(self.tr("PROCESSING PIPELINE"))
        self._hdr_props.setText(self.tr("MODULE PARAMETERS"))
        self.btn_add.setText(self.tr("＋ Add module"))
        self.btn_remove.setText(self.tr("✖ Remove"))
        # Refresh visible node labels + current editor labels.
        for i in range(self.list.count()):
            it = self.list.item(i)
            node = it.data(_NODE_ROLE)
            if node is not None:
                it.setText(tr_node(node.KEY, node.DISPLAY))
        self._rebuild_editor()
