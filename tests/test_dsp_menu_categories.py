"""
test_dsp_menu_categories.py — categorized "Add module" menu and preset combo.

Covers the UX reorganization that groups DSP filters by geophysical
objective/execution order instead of a flat list:
  1. pipeline_panel.py's _show_add_menu — QMenu.addSection() category headers.
  2. processing_controls.py's preset combo — disabled-item category headers.

Constructs REAL widgets via pytest (confirmed to work fine in this
environment — see test_picking_gui.py's module docstring for why raw
`python -c` script invocation is avoided instead).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QMenu, QWidgetAction


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _header_label(action):
    """Category headers are QWidgetAction-wrapped QLabels (see
    pipeline_panel.py's _add_section_header) — the label, not the action
    itself, carries the visible text/font."""
    assert isinstance(action, QWidgetAction)
    label = action.defaultWidget()
    assert label is not None
    return label


class TestAddModuleMenuCategories:
    def test_every_node_registry_key_is_categorized_exactly_once(self):
        from sbp_studio.gui.components.pipeline_panel import _MENU_CATEGORIES
        from sbp_studio.gui.dsp import NODE_REGISTRY
        # MENU_HIDDEN nodes (e.g. AB_AnchorNode) live in the registry for
        # make_node() deserialization but are inserted via a dedicated button,
        # not the Add Module menu, so they are intentionally absent from
        # _MENU_CATEGORIES.
        all_keys = [c.KEY for c in NODE_REGISTRY
                    if not getattr(c, "MENU_HIDDEN", False)]
        categorized: list = []
        for _header, keys in _MENU_CATEGORIES:
            categorized.extend(keys)
        assert sorted(categorized) == sorted(all_keys)
        assert len(categorized) == len(set(categorized))   # no duplicates

    def test_category_order_and_membership_matches_spec(self):
        from sbp_studio.gui.components.pipeline_panel import _MENU_CATEGORIES
        expected = (
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
        assert _MENU_CATEGORIES == expected

    def test_tr_section_round_trips_every_known_header(self):
        from sbp_studio.gui.components.pipeline_panel import _MENU_CATEGORIES, _tr_section
        for header, _keys in _MENU_CATEGORIES:
            assert _tr_section(header) == header
        assert _tr_section("Other") == "Other"

    def test_tr_section_falls_back_for_unknown_name(self):
        from sbp_studio.gui.components.pipeline_panel import _tr_section
        assert _tr_section("Not A Real Category") == "Not A Real Category"

    def test_menu_structure_has_headers_then_actions_in_order(self):
        """Builds the SAME menu _show_add_menu constructs (minus the
        blocking exec()) and verifies: every category appears as a bold,
        disabled QAction header (NOT QMenu.addSection() — see
        _add_section_header's docstring for why addSection's text silently
        disappears once a QSS stylesheet is applied), preceded by a real
        separator (except the first), immediately followed by its node
        actions, in the exact specified order — no flat list."""
        from sbp_studio.gui.components.pipeline_panel import (
            PipelinePanel, _MENU_CATEGORIES, _tr_section,
        )
        from sbp_studio.gui.dsp import NODE_REGISTRY

        pp = PipelinePanel()
        menu = QMenu(pp)
        menu.setToolTipsVisible(True)
        by_key = {cls.KEY: cls for cls in NODE_REGISTRY}
        first = True
        for section_en, keys in _MENU_CATEGORIES:
            if not first:
                menu.addSeparator()
            first = False
            pp._add_section_header(menu, _tr_section(section_en))
            for key in keys:
                pp._add_node_action(menu, by_key[key])

        actions = menu.actions()
        i = 0
        for idx, (section_en, keys) in enumerate(_MENU_CATEGORIES):
            if idx > 0:
                assert actions[i].isSeparator()
                i += 1
            header = actions[i]
            label = _header_label(header)
            assert not header.isSeparator()
            assert label.text() == section_en
            assert not header.isEnabled()          # disabled -> unclickable
            assert label.font().bold()
            assert label.font().weight() == QFont.Weight.Black
            i += 1
            for key in keys:
                assert not actions[i].isSeparator()
                assert actions[i].isEnabled()
                i += 1
        assert i == len(actions)

    def test_section_headers_are_disabled_bold_black_and_not_triggerable(self):
        """A category header must be visibly distinct (bold, Black-weight,
        accent-coloured — see _add_section_header) and inert (disabled, so
        Qt can never deliver a click/triggered signal to it, and
        WA_TransparentForMouseEvents on the label so it never even gets
        hover-highlighted) — the properties the user explicitly required
        after addSection() rendered invisible text, and a plain disabled
        QAction rendered only a generic muted grey, under this app's QSS
        theme."""
        from PyQt6.QtCore import Qt
        from sbp_studio.gui.components.pipeline_panel import PipelinePanel
        from sbp_studio.gui.theme import theme
        pp = PipelinePanel()
        menu = QMenu(pp)
        pp._add_section_header(menu, "Test Section")
        header = menu.actions()[0]
        label = _header_label(header)
        assert not header.isSeparator()
        assert label.text() == "Test Section"
        assert not header.isEnabled()
        assert label.font().bold()
        assert label.font().weight() == QFont.Weight.Black
        assert label.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert theme.color("bright") in label.styleSheet()

    def test_leftover_node_still_appears_under_other_section(self):
        """Defensive catch-all: a node KEY not present in _MENU_CATEGORIES
        must still render (under an 'Other' section) rather than silently
        vanishing from the menu."""
        from sbp_studio.gui.components.pipeline_panel import PipelinePanel, _tr_section
        from sbp_studio.gui.dsp.nodes import DSPNode

        class _FakeExtraNode(DSPNode):
            KEY = "fake_extra_node"
            DISPLAY = "Fake Extra Node"
            TOOLTIP = "A node deliberately absent from _MENU_CATEGORIES."
            SPECS = ()
            def _apply(self, data, ctx):
                return data

        pp = PipelinePanel()
        menu = QMenu(pp)
        menu.setToolTipsVisible(True)
        # Replicate _show_add_menu's body with one extra, uncategorized node.
        from sbp_studio.gui.components.pipeline_panel import _MENU_CATEGORIES
        from sbp_studio.gui.dsp import NODE_REGISTRY
        registry = list(NODE_REGISTRY) + [_FakeExtraNode]
        by_key = {cls.KEY: cls for cls in registry}
        categorized = set()
        for section_en, keys in _MENU_CATEGORIES:
            pp._add_section_header(menu, _tr_section(section_en))
            for key in keys:
                categorized.add(key)
                pp._add_node_action(menu, by_key[key])
        leftover = [cls for cls in registry
                    if cls.KEY not in categorized
                    and not getattr(cls, "MENU_HIDDEN", False)]
        assert leftover == [_FakeExtraNode]
        pp._add_section_header(menu, _tr_section("Other"))
        for cls in leftover:
            pp._add_node_action(menu, cls)

        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert "Fake Extra Node" in texts


class TestPresetComboCategories:
    def test_every_filter_preset_key_is_categorized_exactly_once(self):
        from sbp_studio.gui.components.processing_controls import _PRESET_CATEGORIES
        from sbp_studio.core.constants import FILTER_PRESETS
        all_keys = [k for k in FILTER_PRESETS.values() if k != "none"]
        categorized: list = []
        for _header, keys in _PRESET_CATEGORIES:
            categorized.extend(keys)
        assert sorted(categorized) == sorted(all_keys)
        assert len(categorized) == len(set(categorized))

    def test_category_order_and_membership_matches_spec(self):
        from sbp_studio.gui.components.processing_controls import _PRESET_CATEGORIES
        expected = (
            ("Complex Trace Attributes",
             ("envelope", "inst_phase", "inst_freq", "cos_phase")),
            ("Structural Attributes",
             ("similarity", "sobel_v", "laplacian")),
            ("2D Image Filters",
             ("highboost", "median5", "wiener7")),
            ("Frequency & Smoothing Filters",
             ("gauss1", "topas_narrow", "topas_wide", "topas_hires",
              "derivative", "integral")),
        )
        assert _PRESET_CATEGORIES == expected

    def test_tr_preset_category_round_trips_every_known_header(self):
        from sbp_studio.gui.components.processing_controls import (
            _PRESET_CATEGORIES, _tr_preset_category,
        )
        for header, _keys in _PRESET_CATEGORIES:
            assert _tr_preset_category(header) == header
        assert _tr_preset_category("Other") == "Other"

    def test_combo_item_count_matches_presets_plus_headers(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        from sbp_studio.core.constants import FILTER_PRESETS
        from sbp_studio.gui.components.processing_controls import _PRESET_CATEGORIES
        pc = ProcessingControls()
        n_presets = len(FILTER_PRESETS)          # includes "none"
        n_headers = len(_PRESET_CATEGORIES)       # no leftover today (verified below)
        assert pc.preset_cb.count() == n_presets + n_headers

    def test_none_is_first_and_enabled(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        from sbp_studio.core.constants import FILTER_PRESETS
        pc = ProcessingControls()
        none_label = next(l for l, k in FILTER_PRESETS.items() if k == "none")
        assert pc.preset_cb.itemText(0) == none_label
        assert pc.preset_cb.model().item(0).isEnabled()

    def test_header_items_are_disabled_unselectable_bold_and_black(self):
        """Disabled + bold alone wasn't enough — a merely-disabled
        QStandardItem with a bold QFont still rendered indistinguishable
        from a normal row on at least one OS/style, because the active
        QStyle's item-view painting doesn't have to honour a QFont set on
        the item. The model-level state asserted here is the source of
        truth _PresetHeaderDelegate keys its hand-painted rendering off of
        (see test_combo_uses_header_paint_delegate for the paint side).
        Header text is clean, no "---" decoration — matching the "Add
        module" menu's plain QLabel headers exactly (one visual system)."""
        from PyQt6.QtGui import QFont
        from sbp_studio.gui.components.processing_controls import (
            ProcessingControls, _PRESET_CATEGORIES, _tr_preset_category,
        )
        pc = ProcessingControls()
        cb = pc.preset_cb
        headers = [i for i in range(cb.count())
                  if not cb.model().item(i).isEnabled()]
        assert len(headers) == 4   # the 4 explicit categories, no leftover
        expected_texts = [_tr_preset_category(h) for h, _keys in _PRESET_CATEGORIES]
        for i, expected in zip(headers, expected_texts):
            assert cb.itemText(i) == expected
            assert "---" not in cb.itemText(i)
            assert cb.model().item(i).font().bold()
            assert cb.model().item(i).font().weight() == QFont.Weight.Black

    def test_preset_items_are_enabled_and_not_bold(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        cb = pc.preset_cb
        for i in range(cb.count()):
            if cb.model().item(i).isEnabled():
                assert not cb.model().item(i).font().bold()

    def test_combo_uses_header_paint_delegate(self):
        """The actual on-screen rendering guarantee: preset_cb's popup must
        go through _PresetHeaderDelegate, which paints disabled rows by
        hand (accent colour, Black weight, larger size) instead of trusting
        the active QStyle/QSS — the fix for headers rendering as a
        completely flat, indistinguishable list on at least one OS/style
        combination even after the model-level disabled+bold fix."""
        from sbp_studio.gui.components.processing_controls import (
            ProcessingControls, _PresetHeaderDelegate,
        )
        pc = ProcessingControls()
        assert isinstance(pc.preset_cb.itemDelegate(), _PresetHeaderDelegate)

    def test_header_delegate_paints_disabled_rows_distinctly(self):
        """Renders a header row and a normal row through the real delegate
        onto an offscreen QPixmap and confirms they differ — the strongest
        check available short of a human looking at the popup: if the
        delegate fell back to default painting for header rows (e.g. the
        ItemIsEnabled flag check were inverted), both rows would render
        pixel-identical apart from text content."""
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPixmap, QPainter
        from PyQt6.QtWidgets import QStyleOptionViewItem
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        cb = pc.preset_cb
        delegate = cb.itemDelegate()
        header_idx = next(i for i in range(cb.count())
                           if not cb.model().item(i).isEnabled())
        normal_idx = next(i for i in range(cb.count())
                           if cb.model().item(i).isEnabled())

        def _render(row: int) -> QPixmap:
            pix = QPixmap(120, 24)
            pix.fill()
            painter = QPainter(pix)
            option = QStyleOptionViewItem()
            option.rect = QRect(0, 0, 120, 24)
            option.font = cb.font()
            delegate.paint(painter, option, cb.model().index(row, 0))
            painter.end()
            return pix

        header_pix = _render(header_idx)
        normal_pix = _render(normal_idx)
        assert header_pix.toImage() != normal_pix.toImage()

    def test_items_within_each_category_appear_in_specified_order(self):
        from sbp_studio.gui.components.processing_controls import (
            ProcessingControls, _PRESET_CATEGORIES,
        )
        from sbp_studio.core.constants import FILTER_PRESETS
        key_to_label = {k: l for l, k in FILTER_PRESETS.items()}
        pc = ProcessingControls()
        cb = pc.preset_cb
        all_texts = [cb.itemText(i) for i in range(cb.count())]
        for _header, keys in _PRESET_CATEGORIES:
            labels = [key_to_label[k] for k in keys]
            positions = [all_texts.index(l) for l in labels]
            assert positions == sorted(positions)   # appear in the specified order

    def test_selecting_a_categorized_preset_resolves_correct_description(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        from sbp_studio.core.constants import FILTER_PRESETS, FILTER_DESCRIPTIONS
        key_to_label = {k: l for l, k in FILTER_PRESETS.items()}
        pc = ProcessingControls()
        cb = pc.preset_cb
        idx = cb.findText(key_to_label["laplacian"])
        assert idx >= 0
        cb.setCurrentIndex(idx)
        assert pc.lbl_preset_desc.text() == FILTER_DESCRIPTIONS["laplacian"]

    def test_retranslate_preserves_current_selection(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        from sbp_studio.core.constants import FILTER_PRESETS
        key_to_label = {k: l for l, k in FILTER_PRESETS.items()}
        pc = ProcessingControls()
        cb = pc.preset_cb
        idx = cb.findText(key_to_label["wiener7"])
        cb.setCurrentIndex(idx)
        pc.retranslate_ui()
        assert cb.currentText() == key_to_label["wiener7"]

    def test_repopulate_does_not_change_item_count(self):
        """Calling _populate_preset_combo a second time (as retranslate_ui
        does on a language switch) must not duplicate or drop items."""
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        before = pc.preset_cb.count()
        pc._populate_preset_combo()
        assert pc.preset_cb.count() == before


class TestPresetNodeChoiceRowCategories:
    """The OTHER way a user reaches a specific preset: add the generic
    'Filtro preestablecido' node from the main menu, then pick the specific
    attribute from the QComboBox _ChoiceRow builds inside that node's own
    parameter panel — driven by PresetNode.SPECS' ChoiceSpec, a completely
    separate code path from ProcessingControls.preset_cb tested above. This
    dropdown was reported as a flat list with no headers at all; it must
    get the exact same categorized treatment.
    """

    def test_preset_node_choices_cover_every_real_preset_once(self):
        from sbp_studio.gui.dsp.nodes import PresetNode, PRESET_HEADER_VALUE
        from sbp_studio.core.constants import FILTER_PRESETS
        spec = PresetNode.SPECS[0]
        real_values = [v for v, _d in spec.choices if v != PRESET_HEADER_VALUE]
        expected = [k for k in FILTER_PRESETS.values() if k != "none"]
        assert sorted(real_values) == sorted(expected)
        assert len(real_values) == len(set(real_values))

    def test_preset_node_choices_grouped_in_category_order(self):
        """Walks spec.choices once, splitting it into segments at each
        header sentinel, and checks segment i's keys (in order) exactly
        match PRESET_CATEGORIES[i]'s keys — i.e. the SAME interleaved
        header/keys structure the static combo's _populate_preset_combo
        already builds, not just "all the right keys somewhere"."""
        from sbp_studio.gui.dsp.nodes import PresetNode, PRESET_CATEGORIES, PRESET_HEADER_VALUE
        spec = PresetNode.SPECS[0]
        segments: list = []
        for value, display in spec.choices:
            if value == PRESET_HEADER_VALUE:
                segments.append((display, []))
            else:
                segments[-1][1].append(value)
        assert [header for header, _keys in segments] == \
            [header for header, _keys in PRESET_CATEGORIES]   # no leftover today
        for (_header, actual_keys), (_header_en, expected_keys) in zip(segments, PRESET_CATEGORIES):
            assert actual_keys == list(expected_keys)

    def test_processing_controls_and_preset_node_share_categories(self):
        """The user's explicit requirement: "ordered logically just like
        the main menu" — both widgets must use the IDENTICAL category list,
        not two independently-maintained copies that could drift apart."""
        from sbp_studio.gui.dsp.nodes import PRESET_CATEGORIES
        from sbp_studio.gui.components.processing_controls import _PRESET_CATEGORIES
        assert PRESET_CATEGORIES is _PRESET_CATEGORIES

    def test_choice_row_renders_headers_disabled_bold_black(self):
        """Header text is clean, no "---" decoration — matching the "Add
        module" menu's plain QLabel headers exactly (one visual system)."""
        from PyQt6.QtGui import QFont
        from sbp_studio.gui.components.pipeline_panel import _ChoiceRow
        from sbp_studio.gui.dsp.nodes import PresetNode, PRESET_CATEGORIES, PRESET_HEADER_VALUE
        from sbp_studio.gui.dsp.node_i18n import tr_preset_category
        node = PresetNode()
        spec = node.SPECS[0]
        row = _ChoiceRow(node, spec)
        cb = row.combo
        header_indices = [i for i in range(cb.count())
                          if cb.itemData(i) == PRESET_HEADER_VALUE]
        assert len(header_indices) == 4   # the 4 PRESET_CATEGORIES, no leftover
        expected_texts = [tr_preset_category(h) for h, _keys in PRESET_CATEGORIES]
        for i, expected in zip(header_indices, expected_texts):
            item = cb.model().item(i)
            assert not item.isEnabled()
            assert cb.itemText(i) == expected
            assert "---" not in cb.itemText(i)
            assert item.font().bold()
            assert item.font().weight() == QFont.Weight.Black

    def test_choice_row_uses_header_paint_delegate(self):
        from sbp_studio.gui.components.pipeline_panel import _ChoiceRow
        from sbp_studio.gui.dsp.nodes import PresetNode
        from sbp_studio.gui.theme import CategoryHeaderItemDelegate
        node = PresetNode()
        row = _ChoiceRow(node, node.SPECS[0])
        assert isinstance(row.combo.itemDelegate(), CategoryHeaderItemDelegate)

    def test_choice_row_default_selection_is_a_real_preset_not_a_header(self):
        from sbp_studio.gui.components.pipeline_panel import _ChoiceRow
        from sbp_studio.gui.dsp.nodes import PresetNode, PRESET_HEADER_VALUE
        node = PresetNode()
        row = _ChoiceRow(node, node.SPECS[0])
        assert row.combo.currentData() == "envelope"
        assert row.combo.currentData() != PRESET_HEADER_VALUE

    def test_selecting_a_real_preset_updates_node_params(self):
        from sbp_studio.gui.components.pipeline_panel import _ChoiceRow
        from sbp_studio.gui.dsp.nodes import PresetNode
        node = PresetNode()
        row = _ChoiceRow(node, node.SPECS[0])
        idx = row.combo.findData("laplacian")
        assert idx >= 0
        row.combo.setCurrentIndex(idx)
        assert node.params["preset"] == "laplacian"

    def test_header_rows_keep_no_tooltip(self):
        """spec.tooltips is keyed by real FILTER_DESCRIPTIONS keys only —
        PRESET_HEADER_VALUE can never collide with one, so a header row
        must never pick up a (meaningless) tooltip."""
        from sbp_studio.gui.components.pipeline_panel import _ChoiceRow
        from sbp_studio.gui.dsp.nodes import PresetNode, PRESET_HEADER_VALUE
        node = PresetNode()
        row = _ChoiceRow(node, node.SPECS[0])
        cb = row.combo
        for i in range(cb.count()):
            if cb.itemData(i) == PRESET_HEADER_VALUE:
                assert cb.itemData(i, Qt.ItemDataRole.ToolTipRole) is None


class TestFontScalingSafety:
    """Regression for: the log being flooded with
    ``QFont::setPointSize: Point size <= 0 (-1), must be greater than 0``.

    Root cause: this app's base QSS sets ``font-size: 12px`` (build_qss) —
    a PIXEL size — so a widget's resolved font reports ``pointSize() ==
    -1`` (Qt's "not resolved via points" sentinel). The old
    ``font.setPointSize(font.pointSize() + 1)`` then computed
    ``setPointSize(0)``, which Qt logs a warning for. Verified empirically
    that ``QComboBox().ensurePolished()`` after applying the real app
    stylesheet reproduces ``pointSize() == -1, pixelSize() == 12`` exactly,
    and that Qt only routes the warning to stderr (capturable via pytest's
    fd-level ``capfd``) when ``QT_FORCE_STDERR_LOGGING=1`` — set per-test
    via ``monkeypatch`` so this works the same in this sandbox as in CI.
    """

    def test_bump_font_size_scales_pixel_only_font(self):
        from PyQt6.QtGui import QFont
        from sbp_studio.gui.theme import bump_font_size
        font = QFont()
        font.setPixelSize(12)
        assert font.pointSize() == -1
        bump_font_size(font)
        assert font.pixelSize() == 13
        assert font.pointSize() == -1   # still pixel-resolved, never touched

    def test_bump_font_size_scales_point_only_font(self):
        from PyQt6.QtGui import QFont
        from sbp_studio.gui.theme import bump_font_size
        font = QFont()
        font.setPointSize(10)
        bump_font_size(font)
        assert font.pointSize() == 11

    def test_bump_font_size_on_pixel_font_emits_no_qfont_warning(self, monkeypatch, capfd):
        from PyQt6.QtGui import QFont
        from sbp_studio.gui.theme import bump_font_size
        monkeypatch.setenv("QT_FORCE_STDERR_LOGGING", "1")
        font = QFont()
        font.setPixelSize(12)
        bump_font_size(font)
        captured = capfd.readouterr()
        assert "Point size" not in captured.err

    def test_category_header_delegate_paint_under_real_theme_qss_emits_no_warning(
        self, monkeypatch, capfd,
    ):
        """End-to-end: applies the REAL app stylesheet (the actual trigger —
        see class docstring) to this test's QApplication, polishes a combo
        so its font resolves to pixel size exactly like the live popup,
        then paints a header row through CategoryHeaderItemDelegate."""
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPixmap, QPainter
        from PyQt6.QtWidgets import QComboBox, QStyleOptionViewItem
        from sbp_studio.gui.theme import CategoryHeaderItemDelegate, build_qss, THEMES
        monkeypatch.setenv("QT_FORCE_STDERR_LOGGING", "1")
        app = QApplication.instance()
        old_stylesheet = app.styleSheet()
        app.setStyleSheet(build_qss(THEMES["dark"]))
        try:
            cb = QComboBox()
            cb.addItem("Complex Trace Attributes")
            cb.model().item(0).setEnabled(False)
            cb.ensurePolished()
            assert cb.font().pointSize() == -1   # confirms the trigger condition is live
            delegate = CategoryHeaderItemDelegate(cb)
            pix = QPixmap(160, 28)
            pix.fill()
            painter = QPainter(pix)
            option = QStyleOptionViewItem()
            option.rect = QRect(0, 0, 160, 28)
            option.font = cb.font()
            delegate.paint(painter, option, cb.model().index(0, 0))
            painter.end()
        finally:
            app.setStyleSheet(old_stylesheet)
        captured = capfd.readouterr()
        assert "Point size" not in captured.err

    def test_section_header_label_under_real_theme_qss_emits_no_warning(
        self, monkeypatch, capfd,
    ):
        """Same end-to-end check for the "Add module" menu's QWidgetAction
        QLabel path (_add_section_header), the other call site that hit
        the same bug."""
        from sbp_studio.gui.components.pipeline_panel import PipelinePanel
        from sbp_studio.gui.theme import build_qss, THEMES
        monkeypatch.setenv("QT_FORCE_STDERR_LOGGING", "1")
        app = QApplication.instance()
        old_stylesheet = app.styleSheet()
        app.setStyleSheet(build_qss(THEMES["dark"]))
        try:
            pp = PipelinePanel()
            pp.ensurePolished()
            assert pp.font().pointSize() == -1   # confirms the trigger condition is live
            menu = QMenu(pp)
            pp._add_section_header(menu, "Test Section")
        finally:
            app.setStyleSheet(old_stylesheet)
        captured = capfd.readouterr()
        assert "Point size" not in captured.err


class TestParamRowInteractionSignals:
    """Interactive LOD (see PreviewController._on_interaction_started/_ended)
    relies on _ParamRow's slider press/release reaching PipelinePanel's
    forwarded interactionStarted/interactionEnded signals — verify the whole
    chain, not just the slider in isolation, since _rebuild_editor's
    per-row connect is what actually wires it up end to end."""

    def _panel_with_agc(self):
        from sbp_studio.gui.components.pipeline_panel import PipelinePanel
        from sbp_studio.gui.dsp.nodes import AGCNode

        pp = PipelinePanel()
        pp.add_node(AGCNode())
        pp.list.setCurrentRow(0)  # selects it -> _rebuild_editor populates the row
        return pp

    def test_param_row_slider_press_release_emit_interaction_signals(self):
        from sbp_studio.gui.components.pipeline_panel import _ParamRow
        from sbp_studio.gui.dsp.nodes import AGCNode

        node = AGCNode()
        row = _ParamRow(node, node.SPECS[0])
        started = []
        ended = []
        row.interactionStarted.connect(lambda: started.append(True))
        row.interactionEnded.connect(lambda: ended.append(True))

        row.slider.sliderPressed.emit()
        assert started == [True]
        assert ended == []

        row.slider.sliderReleased.emit()
        assert ended == [True]

    def test_pipeline_panel_forwards_param_row_interaction_signals(self):
        pp = self._panel_with_agc()
        assert pp._editor_rows, "expected AGC's win_ms row to be built"
        row = pp._editor_rows[0]

        started = []
        ended = []
        pp.interactionStarted.connect(lambda: started.append(True))
        pp.interactionEnded.connect(lambda: ended.append(True))

        row.slider.sliderPressed.emit()
        assert started == [True]
        row.slider.sliderReleased.emit()
        assert ended == [True]

    def test_spinbox_edits_do_not_emit_interaction_signals(self):
        """Only the slider's own drag should toggle LOD — a discrete spinbox
        edit (typed value or arrow-click) is a single committed change, not
        a multi-frame drag worth masking compute cost for."""
        pp = self._panel_with_agc()
        row = pp._editor_rows[0]

        started = []
        pp.interactionStarted.connect(lambda: started.append(True))
        row.spin.setValue(row.spin.value() + 1)
        assert started == []
