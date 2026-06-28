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

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QCoreApplication, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget, QWidgetAction,
)

from ..dsp import (
    NODE_REGISTRY, BoolSpec, ChoiceSpec, DSPNode, ParamSpec, PRESET_HEADER_VALUE, make_node,
    tr_node, tr_param, tr_preset_category, tr_tooltip,
)
from ..i18n import language_manager
from ..theme import CategoryHeaderItemDelegate, bump_font_size, theme

_NODE_ROLE = Qt.ItemDataRole.UserRole

# "Add module" menu grouping: a standard marine-seismic processing workflow
# order (physical correction → deconvolution/frequency → 2D spatial →
# gain/visual → interpretation) rather than NODE_REGISTRY's flat iteration
# order. Each tuple is (English source section header, node KEYs in the
# order they should appear within that section) — the header is translated
# via self.tr() in _show_add_menu, same convention as every other UI string.
_MENU_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Editing & Physical Correction",
     ("water_mute", "despike", "spherical_divergence")),
    ("Deconvolution & Frequency",
     ("decon", "whiten", "bandpass", "notch")),
    ("Spatial Filters (2D)",
     ("swell", "fk", "demultiple", "svd_filter", "bilateral_filter",
      "median_filter", "trace_mix")),
    ("Visual Gain Adjustments",
     ("trace_eq", "tvg", "agc", "clahe", "log_compress")),
    ("Interpretation",
     ("preset",)),
)


def _tr_section(name: str) -> str:
    """Localised "Add module" section header. ``self.tr(variable)`` cannot be
    extracted by pylupdate6 (it only sees literals) — same reason
    gui/dsp/node_i18n.py restates node DISPLAY/TOOLTIP strings as explicit
    QCoreApplication.translate(...) literals instead of ``self.tr(node.X)``.
    """
    table = {
        "Editing & Physical Correction": QCoreApplication.translate(
            "PipelinePanel", "Editing & Physical Correction"),
        "Deconvolution & Frequency": QCoreApplication.translate(
            "PipelinePanel", "Deconvolution & Frequency"),
        "Spatial Filters (2D)": QCoreApplication.translate(
            "PipelinePanel", "Spatial Filters (2D)"),
        "Visual Gain Adjustments": QCoreApplication.translate(
            "PipelinePanel", "Visual Gain Adjustments"),
        "Interpretation": QCoreApplication.translate(
            "PipelinePanel", "Interpretation"),
        "Other": QCoreApplication.translate(
            "PipelinePanel", "Other"),
    }
    return table.get(name, name)


# User preset store: one JSON file per saved pipeline under ~/.sbp_studio/presets.
# A preset is a flat list of {"key", "params", "enabled"} — exactly what
# make_node() needs to rebuild each stage, plus its mute state.
_PRESET_DIR = Path.home() / ".sbp_studio" / "presets"
_PRESET_EXT = ".json"


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
    """A combo box bound to one categorical node param. Emits on selection.

    Some ChoiceSpecs (currently PresetNode's "Type") group their choices
    under category headers — see nodes.PRESET_HEADER_VALUE for how a header
    entry is encoded in spec.choices. Those rows are rendered disabled and
    styled via the SAME CategoryHeaderItemDelegate ProcessingControls' own
    categorized preset combo uses, so both widgets present an identical
    visual hierarchy for what is, structurally, the same grouping.
    """

    changed = pyqtSignal()

    def __init__(self, node: DSPNode, spec: ChoiceSpec,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._spec = spec

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setItemDelegate(CategoryHeaderItemDelegate(self.combo))
        for i, (value, display) in enumerate(spec.choices):   # domain data, not tr'd
            if value == PRESET_HEADER_VALUE:
                # UI chrome, not domain data — translated here (at combo-build
                # time, so a runtime language switch is honoured) rather than
                # baked into nodes.py's frozen ChoiceSpec.choices tuple. See
                # node_i18n.tr_preset_category's docstring for why presets
                # themselves stay untranslated but headers don't. Clean text,
                # no "---" decoration — CategoryHeaderItemDelegate's own
                # disabled/bold/accent-colour rendering is what marks this row
                # as a header, matching the "Add module" menu's plain-text
                # QLabel headers exactly (one visual system, not two).
                self.combo.addItem(tr_preset_category(display), value)
                item = self.combo.model().item(i)
                item.setEnabled(False)
                font = item.font()
                font.setBold(True)
                font.setWeight(QFont.Weight.Black)
                item.setFont(font)
                continue
            self.combo.addItem(display, value)
            if spec.tooltips and value in spec.tooltips:
                self.combo.setItemData(i, spec.tooltips[value], Qt.ItemDataRole.ToolTipRole)
        cur = node.params.get(spec.name, spec.default)
        idx = self.combo.findData(cur)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.currentIndexChanged.connect(self._on_change)
        lay.addWidget(self.combo, 1)

    def _on_change(self, *_) -> None:
        self._node.params[self._spec.name] = self.combo.currentData()
        self.changed.emit()


class _BoolRow(QWidget):
    """A checkbox bound to one boolean node param. Emits on toggle."""

    changed = pyqtSignal()

    def __init__(self, node: DSPNode, spec: BoolSpec,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._node = node
        self._spec = spec

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(bool(node.params.get(spec.name, spec.default)))
        self.checkbox.toggled.connect(self._on_toggled)
        lay.addWidget(self.checkbox, 0)
        lay.addStretch(1)

    def _on_toggled(self, checked: bool) -> None:
        self._node.params[self._spec.name] = checked
        self.changed.emit()


class PipelinePanel(QWidget):
    """Reorderable DSP node list + dynamic property editor (debounced)."""

    # Emitted (debounced for param edits, immediate for structural edits) when
    # the effective pipeline changes. A controller listens and re-runs preview.
    pipeline_changed = pyqtSignal()

    DEBOUNCE_MS = 200

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # Guards the itemChanged handler while we programmatically set a row's
        # check state / style (so mute toggles only react to real user clicks).
        self._suppress_item_changed = False

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

        # ── Preset bar (load dropdown + Save) ──
        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self._on_preset_selected)
        self.btn_save_preset = QPushButton()
        self.btn_save_preset.clicked.connect(self._save_preset)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.btn_save_preset, 0)
        root.addLayout(preset_row)

        # ── Node list (drag-drop reorder; per-row mute checkbox) ──
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.currentItemChanged.connect(lambda *_: self._rebuild_editor())
        # Per-row checkbox = node mute/bypass (see _on_item_changed). itemChanged
        # also fires on text edits, but we only ever change text via setText with
        # the check-state preserved, so the handler simply re-syncs node.enabled.
        self.list.itemChanged.connect(self._on_item_changed)
        # rowsMoved fires after a drag-drop reorder completes.
        self.list.model().rowsMoved.connect(self._on_reordered)
        root.addWidget(self.list, 1)

        # ── Add / Remove / Clear (mirrors the Loaded Profiles tree's button row:
        # same QHBoxLayout margins, same Add/Remove-stretch-Clear order) ──
        btns = QHBoxLayout()
        btns.setContentsMargins(6, 0, 6, 0)
        self.btn_add = QPushButton()
        self.btn_remove = QPushButton()
        self.btn_clear = QPushButton()
        self.btn_add.clicked.connect(self._show_add_menu)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_pipeline)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_remove)
        btns.addStretch(1)
        btns.addWidget(self.btn_clear)
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
        self._refresh_presets()

    # ── Public API ──────────────────────────────────────────────────────────

    def nodes(self) -> List[DSPNode]:
        """EVERY ordered node, including muted ones (top = first applied).
        Used for serialization / save-preset, which must persist mute state."""
        out: List[DSPNode] = []
        for i in range(self.list.count()):
            node = self.list.item(i).data(_NODE_ROLE)
            if node is not None:
                out.append(node)
        return out

    def active_nodes(self) -> List[DSPNode]:
        """Only the ENABLED nodes, in order — the chain that actually executes.
        Both the live preview and the export read this, so a muted node is
        bypassed everywhere (Node Mute / bypass)."""
        return [n for n in self.nodes() if getattr(n, "enabled", True)]

    def add_node(self, node: DSPNode) -> None:
        item = QListWidgetItem(tr_node(node.KEY, node.DISPLAY))
        item.setData(_NODE_ROLE, node)
        # Checkable row = mute toggle. setData/flags BEFORE setCheckState so the
        # itemChanged that setCheckState fires already sees the node payload.
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self._suppress_item_changed = True
        item.setCheckState(Qt.CheckState.Checked if node.enabled
                           else Qt.CheckState.Unchecked)
        self._suppress_item_changed = False
        self._apply_muted_style(item, node.enabled)
        self.list.addItem(item)
        self.list.setCurrentItem(item)
        self.pipeline_changed.emit()   # structural → immediate

    # ── Node mute (bypass) ────────────────────────────────────────────────────

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """A row's checkbox toggled → mute/unmute its node and re-run."""
        if getattr(self, "_suppress_item_changed", False):
            return
        node = item.data(_NODE_ROLE)
        if node is None:
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        if bool(getattr(node, "enabled", True)) == enabled:
            return                     # no real change (e.g. text-only update)
        node.enabled = enabled
        self._apply_muted_style(item, enabled)
        self.pipeline_changed.emit()   # active node set changed → recompute

    def _apply_muted_style(self, item: QListWidgetItem, enabled: bool) -> None:
        """Grey + strike-through a muted row so it reads as 'bypassed'.

        BUG FIX: the enabled case used to reset the foreground to ``QColor()``
        — an INVALID QColor, which Qt paints as black. That per-item
        Qt::ForegroundRole override always wins over the stylesheet's
        ``QListWidget { color: ... }`` rule, so every enabled row rendered as
        unreadable black-on-dark-gray regardless of theme. Must use the
        theme's actual text colour instead (live via ``theme.color``, so a
        runtime theme switch still applies correctly)."""
        font = item.font()
        font.setStrikeOut(not enabled)
        item.setFont(font)
        item.setForeground(QColor(Qt.GlobalColor.gray) if not enabled
                           else QColor(theme.color("text")))

    # ── Structural actions ──────────────────────────────────────────────────

    def _show_add_menu(self) -> None:
        menu = QMenu(self)
        # Self-documenting "Add module" dropdown: a 1-2 sentence geophysical
        # explanation on hover. QMenu does NOT show QAction tooltips by
        # default (a long-standing Qt quirk) — setToolTipsVisible(True) is
        # REQUIRED, not just setToolTip() on each action, or hovering would
        # silently show nothing.
        menu.setToolTipsVisible(True)

        by_key = {cls.KEY: cls for cls in NODE_REGISTRY}
        categorized: set = set()
        first = True
        for section_en, keys in _MENU_CATEGORIES:
            if not first:
                menu.addSeparator()
            first = False
            self._add_section_header(menu, _tr_section(section_en))
            for key in keys:
                cls = by_key.get(key)
                if cls is None:
                    continue
                categorized.add(key)
                self._add_node_action(menu, cls)

        # Defensive catch-all: a node NOT listed in _MENU_CATEGORIES (e.g. a
        # future addition nobody re-categorised yet) still appears here
        # rather than silently vanishing from the menu — NODE_REGISTRY stays
        # the single source of truth for "what's addable".
        leftover = [cls for cls in NODE_REGISTRY if cls.KEY not in categorized]
        if leftover:
            if not first:
                menu.addSeparator()
            self._add_section_header(menu, _tr_section("Other"))
            for cls in leftover:
                self._add_node_action(menu, cls)

        menu.exec(self.btn_add.mapToGlobal(self.btn_add.rect().bottomLeft()))

    def _add_section_header(self, menu: QMenu, text: str) -> None:
        """An unclickable category header with a strong visual hierarchy.

        QMenu.addSection() is the "correct" native API for this, but once a
        QSS stylesheet is applied to QMenu (this app always has one — see
        theme.py), Qt's section-title paint path silently drops the text on
        several styles, leaving only a bare separator line — a long-standing
        Qt/QSS interaction quirk. A plain disabled QAction (the first fix
        attempted here) sidesteps THAT bug, but its text colour still comes
        from the active QStyle/QSS for a ":disabled" menu item, which only
        ever gives a muted grey — not the distinct accent colour a header
        needs to stand out from real, clickable entries.

        A QWidgetAction wrapping a real QLabel sidesteps both problems at
        once: the label paints itself, completely independent of QMenu's
        item-painting path, so neither addSection()'s text-drop bug nor the
        QSS-disabled-grey limitation can touch it. WA_TransparentForMouseEvents
        keeps it inert (clicks/hover pass through rather than highlighting
        or triggering it); setEnabled(False) on the action itself is a second,
        belt-and-suspenders guard against it ever being treated as the
        current/triggerable menu item.
        """
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        font.setWeight(QFont.Weight.Black)
        bump_font_size(font)
        label.setFont(font)
        label.setStyleSheet(f"color: {theme.color('bright')}; padding: 4px 10px;")
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        act = QWidgetAction(menu)
        act.setDefaultWidget(label)
        act.setEnabled(False)
        menu.addAction(act)

    def _add_node_action(self, menu: QMenu, cls: type) -> None:
        act = QAction(tr_node(cls.KEY, cls.DISPLAY), self)
        act.setToolTip(tr_tooltip(cls.KEY, cls.TOOLTIP))
        act.triggered.connect(lambda _=False, c=cls: self.add_node(c()))
        menu.addAction(act)

    def _remove_selected(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        self.list.takeItem(row)
        self._rebuild_editor()
        self.pipeline_changed.emit()   # structural → immediate

    def _clear_pipeline(self) -> None:
        if self.list.count() == 0:
            return
        self.list.clear()
        self._rebuild_editor()
        self.pipeline_changed.emit()   # structural → immediate

    def _on_reordered(self, *_) -> None:
        # Drag-drop reorder changes order → pipeline differs → immediate.
        self.pipeline_changed.emit()

    # ── Save / Load presets ───────────────────────────────────────────────────

    def _serialize(self) -> list:
        """The current stack as a JSON-ready list (key + params + mute state)."""
        return [{"key": n.KEY, "params": dict(n.params),
                 "enabled": bool(getattr(n, "enabled", True))}
                for n in self.nodes()]

    def set_pipeline(self, spec: list) -> None:
        """Rebuild the node stack from a serialized spec (load preset).

        Fully VALIDATED before any mutation: a malformed/hand-edited preset
        file (wrong field types, foreign/unknown keys) must never wipe the
        user's current pipeline. Each entry is checked independently — one
        bad entry is skipped, not fatal — and ``self.list`` is only cleared
        once every entry has been turned into a real node. Emits
        ``pipeline_changed`` once at the end (only if anything was built)."""
        if not isinstance(spec, list):
            QMessageBox.warning(
                self, self.tr("Load failed"),
                self.tr("Preset is not a valid pipeline (expected a list of modules)."))
            return

        built: List[DSPNode] = []
        for entry in spec:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key", "")
            params = entry.get("params") or {}
            if not isinstance(key, str) or not isinstance(params, dict):
                continue
            try:
                node = make_node(key, params, enabled=bool(entry.get("enabled", True)))
            except (KeyError, TypeError, ValueError):
                continue          # unknown node type / malformed params — skip gracefully
            built.append(node)

        # A non-empty spec that yields ZERO usable nodes is corruption, not an
        # intentionally empty preset — treat it as a failed load and keep the
        # current pipeline intact rather than silently wiping it.
        if spec and not built:
            QMessageBox.warning(
                self, self.tr("Load failed"),
                self.tr("Preset contains no valid modules — keeping the current pipeline."))
            return

        self.list.blockSignals(True)
        self.list.clear()
        self.list.blockSignals(False)
        for node in built:
            item = QListWidgetItem(tr_node(node.KEY, node.DISPLAY))
            item.setData(_NODE_ROLE, node)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._suppress_item_changed = True
            item.setCheckState(Qt.CheckState.Checked if node.enabled
                               else Qt.CheckState.Unchecked)
            self._suppress_item_changed = False
            self.list.addItem(item)
            self._apply_muted_style(item, node.enabled)
        self._rebuild_editor()
        self.pipeline_changed.emit()

    @staticmethod
    def _safe_name(name: str) -> str:
        """Filesystem-safe stem for a preset (collapse non-word chars)."""
        return re.sub(r"[^\w\- ]+", "_", name).strip() or "preset"

    def _refresh_presets(self) -> None:
        """Repopulate the load dropdown from the presets folder."""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(self.tr("Load preset…"), None)
        try:
            files = sorted(_PRESET_DIR.glob(f"*{_PRESET_EXT}"))
        except OSError:
            files = []
        for f in files:
            self.preset_combo.addItem(f.stem, str(f))
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self, self.tr("Save Preset"), self.tr("Preset name:"))
        name = (name or "").strip()
        if not ok or not name:
            return
        _PRESET_DIR.mkdir(parents=True, exist_ok=True)
        path = _PRESET_DIR / f"{self._safe_name(name)}{_PRESET_EXT}"
        if path.exists():
            reply = QMessageBox.question(
                self, self.tr("Overwrite preset?"),
                self.tr("A preset named “{0}” already exists. Overwrite it?")
                    .format(path.stem),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"name": name, "nodes": self._serialize()}, fh, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, self.tr("Save failed"), str(exc))
            return
        self._refresh_presets()
        idx = self.preset_combo.findData(str(path))
        if idx >= 0:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(idx)
            self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, index: int) -> None:
        path = self.preset_combo.itemData(index)
        if not path:
            return           # the "Load preset…" placeholder row
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError(self.tr("Preset file is not a valid pipeline (expected an object)."))
            nodes = data.get("nodes")
            if nodes is not None and not isinstance(nodes, list):
                raise ValueError(self.tr("Preset 'nodes' field is not a list."))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, self.tr("Load failed"), str(exc))
            return
        self.set_pipeline(nodes or [])

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
            if isinstance(spec, ChoiceSpec):
                row = _ChoiceRow(node, spec)
            elif isinstance(spec, BoolSpec):
                row = _BoolRow(node, spec)
            else:
                row = _ParamRow(node, spec)
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
        self.btn_clear.setText(self.tr("✖✖ Clear"))
        self.btn_save_preset.setText(self.tr("Save Preset"))
        # The placeholder row text (index 0) is language-dependent; refresh it.
        if self.preset_combo.count() > 0 and self.preset_combo.itemData(0) is None:
            self.preset_combo.setItemText(0, self.tr("Load preset…"))
        # Refresh visible node labels + current editor labels (preserve check state).
        self._suppress_item_changed = True
        for i in range(self.list.count()):
            it = self.list.item(i)
            node = it.data(_NODE_ROLE)
            if node is not None:
                it.setText(tr_node(node.KEY, node.DISPLAY))
        self._suppress_item_changed = False
        self._rebuild_editor()
