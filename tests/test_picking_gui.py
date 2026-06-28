"""
test_picking_gui.py — Interpretation & Picking Module (Phase 3), GUI layer.

Constructs REAL SeismicView/ProcessingControls/dialog widgets via pytest
(confirmed to work fine in this environment — only raw `python -c` script
invocation hangs on Qt object construction here, not pytest-driven tests).
QInputDialog.getText / QMenu.exec are monkeypatched (they're blocking modal
calls) — the same established pattern used throughout this test suite for
dialog interactions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QInputDialog, QMenu

from sbp_studio.core import PickPoint


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


class _FakeSource:
    """Stand-in for a SegyProfile/ProfileChain's coordinate arrays."""
    def __init__(self, lons, lats):
        self.lons = lons
        self.lats = lats


class _FakeClickEvent:
    """Stand-in for pyqtgraph's MouseClickEvent — only the methods
    _on_scene_click actually calls."""
    def __init__(self, scene_pos, button=Qt.MouseButton.LeftButton,
                double=False, accepted=False):
        self._scene_pos = scene_pos
        self._button = button
        self._double = double
        self._accepted = accepted

    def scenePos(self):
        return self._scene_pos

    def button(self):
        return self._button

    def double(self):
        return self._double

    def isAccepted(self):
        return self._accepted


class _FakeSpot:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _FakeScatterClickEvent:
    def __init__(self, button, screen_pos):
        self._button = button
        self._screen_pos = screen_pos

    def button(self):
        return self._button

    def screenPos(self):
        return self._screen_pos


def _make_view():
    """A SeismicView with a real (small) image shown, so the ViewBox has a
    genuine view<->scene transform to round-trip through.

    NOTE: do NOT call ``sv.show()`` here — with ``useOpenGL=True`` (set at
    module level in seismic_view.py) an ACTUAL paint event in this sandbox
    (no real display/GPU driver) segfaults inside pyqtgraph's ImageItem.paint.
    Instead, ``mapSceneToView`` is called once as a harmless "warm-up": an
    unshown widget's scene<->view transform is not yet final, and this ONE
    call is what settles it (confirmed empirically — the SAME scene position
    measured before vs. after this call resolves to a different local-pixel
    coordinate otherwise). A real on-screen widget has already been through
    many such layout passes before a user can click on it, so this is a
    test-fixture-only concern, not a production one."""
    from sbp_studio.gui.components.seismic_view import SeismicView
    from PyQt6.QtCore import QPointF
    sv = SeismicView()
    sv.resize(400, 300)
    arr = np.zeros((50, 20), dtype=np.float32)
    sv.show_image(arr, dist0=0.0, dist1=2.0, t0=0.0, t1=500.0,
                 cmap_name="viridis", vmax=1.0)
    sv.set_distance_axis(np.linspace(0.0, 2.0, 20))
    sv.plot.getViewBox().mapSceneToView(QPointF(0.0, 0.0))   # settle the transform
    return sv


class TestPickingModeAndSource:
    def test_pick_mode_defaults_to_off(self):
        sv = _make_view()
        assert sv._pick_mode is False

    def test_set_pick_mode_toggles_flag(self):
        sv = _make_view()
        sv.set_pick_mode(True)
        assert sv._pick_mode is True
        sv.set_pick_mode(False)
        assert sv._pick_mode is False

    def test_set_picking_source_clears_picks_on_different_object(self):
        sv = _make_view()
        a, b = _FakeSource([1.0], [1.0]), _FakeSource([2.0], [2.0])
        sv.set_picking_source(a)
        sv._picks = [PickPoint(1, 0, 0.0, 1.0, 1.0, "x")]
        sv._redraw_picks()
        sv.set_picking_source(b)               # different object → cleared
        assert sv.get_picks() == []

    def test_set_picking_source_same_object_keeps_picks(self):
        sv = _make_view()
        a = _FakeSource([1.0], [1.0])
        sv.set_picking_source(a)
        sv._picks = [PickPoint(1, 0, 0.0, 1.0, 1.0, "x")]
        sv._redraw_picks()
        sv.set_picking_source(a)               # SAME object → kept
        assert len(sv.get_picks()) == 1

    def test_id_counter_survives_toggle_off_and_on(self):
        sv = _make_view()
        sv.set_pick_mode(True)
        sv._next_pick_id = 5
        sv.set_pick_mode(False)
        sv.set_pick_mode(True)
        assert sv._next_pick_id == 5

    def test_id_counter_not_reset_by_clear_picks(self):
        """Deleting/clearing markers must never let a future id collide with
        one that's already been exported/referenced."""
        sv = _make_view()
        sv._next_pick_id = 9
        sv.clear_picks()
        assert sv._next_pick_id == 9


class TestCreatePick:
    def _scene_pos_for(self, sv, km: float, ms: float):
        from PyQt6.QtCore import QPointF
        return sv.plot.getViewBox().mapViewToScene(QPointF(km, ms))

    def test_double_click_in_pick_mode_creates_a_point(self, monkeypatch):
        sv = _make_view()
        sv.set_picking_source(_FakeSource([0.0] * 20, [0.0] * 20))
        sv.set_pick_mode(True)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("Fault A", True)))
        scene_pos = self._scene_pos_for(sv, 1.0, 200.0)
        ev = _FakeClickEvent(scene_pos, double=True, accepted=False)
        sv._on_scene_click(ev)
        picks = sv.get_picks()
        assert len(picks) == 1
        assert picks[0].id == 1
        assert picks[0].description == "Fault A"
        assert picks[0].time_ms == pytest.approx(200.0, abs=1.0)

    def test_cancelling_dialog_creates_nothing_and_does_not_consume_an_id(self, monkeypatch):
        sv = _make_view()
        sv.set_picking_source(_FakeSource([0.0] * 20, [0.0] * 20))
        sv.set_pick_mode(True)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("", False)))
        scene_pos = self._scene_pos_for(sv, 1.0, 200.0)
        ev = _FakeClickEvent(scene_pos, double=True, accepted=False)
        next_id_before = sv._next_pick_id
        sv._on_scene_click(ev)
        assert sv.get_picks() == []
        assert sv._next_pick_id == next_id_before

    def test_single_click_does_not_create_a_pick_even_in_pick_mode(self, monkeypatch):
        sv = _make_view()
        sv.set_pick_mode(True)
        called = []
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: called.append(1) or ("x", True)))
        scene_pos = self._scene_pos_for(sv, 1.0, 200.0)
        ev = _FakeClickEvent(scene_pos, double=False, accepted=False)
        sv._on_scene_click(ev)
        assert called == []
        assert sv.get_picks() == []

    def test_double_click_with_pick_mode_off_does_not_create_a_pick(self, monkeypatch):
        sv = _make_view()
        sv.set_pick_mode(False)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("x", True)))
        scene_pos = self._scene_pos_for(sv, 1.0, 200.0)
        ev = _FakeClickEvent(scene_pos, double=True, accepted=False)
        sv._on_scene_click(ev)
        assert sv.get_picks() == []

    def test_double_click_already_accepted_by_a_marker_is_ignored(self, monkeypatch):
        """ev.isAccepted()==True means an existing marker's own
        mouseClickEvent already handled this click — must not ALSO create a
        new point stacked on top of it."""
        sv = _make_view()
        sv.set_pick_mode(True)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("x", True)))
        scene_pos = self._scene_pos_for(sv, 1.0, 200.0)
        ev = _FakeClickEvent(scene_pos, double=True, accepted=True)
        sv._on_scene_click(ev)
        assert sv.get_picks() == []

    def test_sequential_ids_assigned_in_order(self, monkeypatch):
        sv = _make_view()
        sv.set_pick_mode(True)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("p", True)))
        for km in (0.2, 0.6, 1.0):
            sv._on_scene_click(_FakeClickEvent(
                self._scene_pos_for(sv, km, 50.0), double=True))
        ids = [p.id for p in sv.get_picks()]
        assert ids == [1, 2, 3]

    def test_resolves_x_coord_y_coord_from_source(self, monkeypatch):
        """Picking click resolution uses _picking_trace_index_at_scene_pos —
        the window-relative linear-fraction INVERSE (see its docstring),
        NOT _index_at_scene_pos's dist_km-searchsorted resolution (used by
        hover/single-click/the ruler) — so the expected scene position is
        the FORWARD-mapped km for trace 10 within the current window
        (_img_trace_lo=0, _img_trace_hi=20, _img_km_lo/_hi=0..2, set by
        _make_view's show_image call), not dist_km[10] directly: that
        mapping is intentionally NOT a literal dist_km lookup (see
        _pick_x_km's docstring) — this round-trips through the SAME
        forward formula _redraw_picks would use to draw it back."""
        sv = _make_view()
        lons = [0.0] * 20
        lats = [0.0] * 20
        lons[10], lats[10] = -8.5, 43.1
        sv.set_picking_source(_FakeSource(lons, lats))
        sv.set_pick_mode(True)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("p", True)))
        forward_km = sv._pick_x_km(10, sv._dist_km.size)
        scene_pos = self._scene_pos_for(sv, forward_km, 200.0)
        sv._on_scene_click(_FakeClickEvent(scene_pos, double=True))
        picks = sv.get_picks()
        assert len(picks) == 1
        assert picks[0].trace_index == 10
        assert picks[0].x_coord == -8.5 and picks[0].y_coord == 43.1


class TestEditDeleteViaContextMenu:
    def test_right_click_on_marker_shows_menu_edit_updates_description(self, monkeypatch):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, -8.5, 43.1, "old")]
        sv._redraw_picks()
        monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: self.actions()[0])  # "Edit description"
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("new desc", True)))
        from PyQt6.QtCore import QPointF
        ev = _FakeScatterClickEvent(Qt.MouseButton.RightButton, QPointF(0, 0))
        sv._on_pick_scatter_clicked(None, [_FakeSpot(1)], ev)
        assert sv.get_picks()[0].description == "new desc"

    def test_right_click_delete_removes_the_point(self, monkeypatch):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, -8.5, 43.1, "old")]
        sv._redraw_picks()
        monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: self.actions()[1])  # "Delete marker"
        from PyQt6.QtCore import QPointF
        ev = _FakeScatterClickEvent(Qt.MouseButton.RightButton, QPointF(0, 0))
        sv._on_pick_scatter_clicked(None, [_FakeSpot(1)], ev)
        assert sv.get_picks() == []
        assert 1 not in sv._pick_labels

    def test_left_click_on_marker_does_nothing(self, monkeypatch):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, -8.5, 43.1, "old")]
        sv._redraw_picks()
        exec_called = []
        monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: exec_called.append(1))
        from PyQt6.QtCore import QPointF
        ev = _FakeScatterClickEvent(Qt.MouseButton.LeftButton, QPointF(0, 0))
        sv._on_pick_scatter_clicked(None, [_FakeSpot(1)], ev)
        assert exec_called == []
        assert sv.get_picks()[0].description == "old"

    def test_unknown_pick_id_is_a_safe_no_op(self):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, -8.5, 43.1, "old")]
        sv._redraw_picks()
        from PyQt6.QtCore import QPointF
        ev = _FakeScatterClickEvent(Qt.MouseButton.RightButton, QPointF(0, 0))
        sv._on_pick_scatter_clicked(None, [_FakeSpot(999)], ev)   # not a real id
        assert sv.get_picks()[0].description == "old"


class TestDrawingAndImportExportPlumbing:
    def test_redraw_creates_one_label_per_pick(self):
        sv = _make_view()
        sv._picks = [PickPoint(i, i, float(i * 10), 0.0, 0.0, f"p{i}") for i in range(1, 4)]
        sv._redraw_picks()
        assert len(sv._pick_labels) == 3
        assert sv._pick_scatter is not None

    def test_clear_picks_removes_all_labels(self):
        sv = _make_view()
        sv._picks = [PickPoint(1, 1, 10.0, 0.0, 0.0, "p")]
        sv._redraw_picks()
        sv.clear_picks()
        assert sv._pick_labels == {}
        assert sv.get_picks() == []

    def test_set_picks_replaces_list_and_advances_id_counter(self):
        sv = _make_view()
        sv._next_pick_id = 1
        imported = [PickPoint(7, 3, 30.0, 1.0, 1.0, "imported")]
        sv.set_picks(imported)
        assert sv.get_picks() == imported
        assert sv._next_pick_id == 8   # past the highest imported id

    def test_get_picks_returns_a_copy_not_the_live_list(self):
        sv = _make_view()
        sv._picks = [PickPoint(1, 1, 10.0, 0.0, 0.0, "p")]
        copy = sv.get_picks()
        copy.append(PickPoint(2, 2, 20.0, 0.0, 0.0, "q"))
        assert len(sv._picks) == 1   # the live list is untouched

    def test_redraw_clamps_trace_index_beyond_current_distance_array(self):
        """An imported session from a longer line must not crash when drawn
        against a shorter/different profile's distance axis."""
        sv = _make_view()
        sv._picks = [PickPoint(1, 99999, 10.0, 0.0, 0.0, "p")]
        sv._redraw_picks()   # must not raise
        assert len(sv._pick_labels) == 1


class TestProcessingControlsPickingButtons:
    def test_buttons_exist_and_toggle_is_checkable(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        assert pc.btn_toggle_picking.isCheckable()
        assert not pc.btn_toggle_picking.isChecked()

    def test_toggling_emits_picking_toggled_signal(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        received = []
        pc.picking_toggled.connect(received.append)
        pc.btn_toggle_picking.setChecked(True)
        assert received == [True]

    def test_export_import_button_emits_signal(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        received = []
        pc.export_import_picking_requested.connect(lambda: received.append(1))
        pc.btn_export_import_picking.click()
        assert received == [1]

    def test_toggle_button_style_changes_with_state(self):
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        pc = ProcessingControls()
        off_style = pc.btn_toggle_picking.styleSheet()
        pc.btn_toggle_picking.setChecked(True)
        on_style = pc.btn_toggle_picking.styleSheet()
        assert off_style != on_style
        assert "c62828" in off_style    # red while inactive
        assert "2e7d32" in on_style     # green while active


class TestPickingExportImportDialog:
    def test_import_disabled_when_picks_exist(self):
        from sbp_studio.gui.components.picking_export_dialog import PickingExportImportDialog
        dlg = PickingExportImportDialog(has_existing_picks=True)
        assert not dlg.btn_import.isEnabled()
        assert dlg.btn_export.isEnabled()

    def test_import_enabled_when_no_picks(self):
        from sbp_studio.gui.components.picking_export_dialog import PickingExportImportDialog
        dlg = PickingExportImportDialog(has_existing_picks=False)
        assert dlg.btn_import.isEnabled()

    def test_clicking_export_sets_choice_and_accepts(self):
        from sbp_studio.gui.components.picking_export_dialog import PickingExportImportDialog
        dlg = PickingExportImportDialog(has_existing_picks=False)
        dlg.btn_export.click()
        assert dlg.choice() == PickingExportImportDialog.EXPORT
        assert dlg.result() == QDialog.DialogCode.Accepted

    def test_clicking_import_sets_choice_and_accepts(self):
        from sbp_studio.gui.components.picking_export_dialog import PickingExportImportDialog
        dlg = PickingExportImportDialog(has_existing_picks=False)
        dlg.btn_import.click()
        assert dlg.choice() == PickingExportImportDialog.IMPORT

    def test_no_choice_before_any_click(self):
        from sbp_studio.gui.components.picking_export_dialog import PickingExportImportDialog
        dlg = PickingExportImportDialog(has_existing_picks=False)
        assert dlg.choice() is None


class TestPickScatterPlotItemRightClick:
    """The narrow but crucial fix: the base ScatterPlotItem.mouseClickEvent
    only ever handles LeftButton (every other button is ev.ignore()'d) — see
    pyqtgraph's own source. _PickScatterPlotItem must accept ANY button when
    it lands on a point, so a right-click reaches sigClicked instead of
    bubbling to the ViewBox's own context menu."""

    def test_subclass_overrides_mouse_click_event(self):
        from sbp_studio.gui.components.seismic_view import _PickScatterPlotItem
        import inspect
        src = inspect.getsource(_PickScatterPlotItem.mouseClickEvent)
        assert "LeftButton" not in src   # no button-type filtering at all

    def test_right_click_on_a_point_fires_sigclicked(self, monkeypatch):
        """pointsAt's own geometric hit-testing needs a real scene/transform
        (a bare unparented item has none) — that's pyqtgraph's own,
        already-tested code, not this override's logic. Stub pointsAt to a
        controlled non-empty result and verify ONLY this override's
        branching: ANY button reaching a hit gets accepted + forwarded."""
        from sbp_studio.gui.components.seismic_view import _PickScatterPlotItem
        from PyQt6.QtCore import QPointF
        scatter = _PickScatterPlotItem()
        monkeypatch.setattr(scatter, "pointsAt", lambda pos: ["hit"])
        received = []
        scatter.sigClicked.connect(lambda *a: received.append(a))

        class _Ev:
            def __init__(self):
                self._accepted = False
            def pos(self):
                return QPointF(0.0, 0.0)
            def button(self):
                return Qt.MouseButton.RightButton
            def accept(self):
                self._accepted = True
            def isAccepted(self):
                return self._accepted

        scatter.mouseClickEvent(_Ev())
        assert len(received) == 1

    def test_click_missing_every_point_is_ignored(self, monkeypatch):
        from sbp_studio.gui.components.seismic_view import _PickScatterPlotItem
        from PyQt6.QtCore import QPointF
        scatter = _PickScatterPlotItem()
        monkeypatch.setattr(scatter, "pointsAt", lambda pos: [])
        received = []
        scatter.sigClicked.connect(lambda *a: received.append(a))

        class _Ev:
            def pos(self):
                return QPointF(1000.0, 1000.0)
            def button(self):
                return Qt.MouseButton.RightButton
            def accept(self):
                raise AssertionError("must not accept a miss")
            def ignore(self):
                pass

        scatter.mouseClickEvent(_Ev())
        assert received == []


class TestZoomDriftFix:
    """Regression coverage for two related reported bugs in the picking
    overlay:

    1. Severe Y-axis offset on creation — double-clicking a reflector at,
       say, 400ms created a marker visibly higher up (~250ms). Root cause:
       the PREVIOUS fix's "image-local pixel grid" bookkeeping (_img_t0,
       _img_dt_ms, _img_c0, _img_stride) was populated ONLY by
       _push_full_image/_apply_zoom_update — the STATIC display-buffer
       path used by ``show_image`` (and these tests, via _make_view).
       The actually-running app, however, drives the view exclusively
       through ``show_preview`` (PreviewController; see
       ``_base.py``'s ``enable_preview(True)``), which never touched that
       state at all — so in production those values stayed frozen at their
       ``__init__`` defaults (t0=0.0, dt_ms=1.0) regardless of the real
       data's actual t0/dt, producing an arbitrary vertical offset.

    2. Visual drift / mismatch against the actual reflector — even at a
       fixed zoom, a pick plotted at the TRUE (non-uniform — ship speed
       varies) ``dist_km[trace_index]`` decouples from where that trace's
       data actually renders: self.img's array is always a UNIFORM
       resample (pyqtgraph's ImageItem.setRect stretches it linearly
       across whatever km window is shown, exactly like
       viz.render._colorize_for_target on the export side), with no notion
       of dist_km's real spacing.

    The fix: picks live directly in the ViewBox (self.plot.addItem, never
    parented to self.img), positioned via ``_pick_x_km`` — a LINEAR
    interpolation of trace_index into the CURRENT window's own trace-index
    bounds (``_img_trace_lo``/``_img_trace_hi``) mapped into that window's
    km bounds (``_img_km_lo``/``_img_km_hi``), set by
    ``show_preview``/``_apply_zoom_update``/``_push_full_image`` — the SAME
    linear approximation the image itself is built from, NOT a direct
    dist_km lookup. Y is the pick's own time_ms verbatim (the sample
    interval is constant by construction, so it never has this problem).

    A deliberate, unavoidable consequence: a pick's RENDERED km position
    can shift slightly between different zoom windows, since each
    window's own linear approximation of non-uniform spacing differs a
    little — the trade-off for staying glued to the image's actual pixels
    in whichever window is currently shown. ``_redraw_picks`` is therefore
    called again whenever the window's bounds change, not just when the
    pick LIST itself changes.
    """

    def test_scatter_lives_in_the_viewbox_not_parented_to_the_image(self):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, 0.0, 0.0, "p")]
        sv._redraw_picks()
        assert sv._pick_scatter.parentItem() is not sv.img
        assert sv._pick_scatter in sv.plot.items

    def test_labels_live_in_the_viewbox_not_parented_to_the_image(self):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, 0.0, 0.0, "p")]
        sv._redraw_picks()
        label = sv._pick_labels[1]
        assert label.parentItem() is not sv.img
        assert label in sv.plot.items

    def test_redraw_positions_scatter_via_linear_fraction_not_dist_km(self):
        """X is the LINEAR-FRACTION interpolation of trace_index into the
        current window's bounds (_pick_x_km) — NOT a dist_km[trace_index]
        lookup, which would decouple from the image whenever spacing is
        non-uniform (see TestZoomDriftFix's class docstring). Y is
        time_ms verbatim — no local-pixel-grid conversion at all."""
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 137.5, 0.0, 0.0, "p")]
        sv._redraw_picks()
        x = sv._pick_scatter.data["x"][0]
        y = sv._pick_scatter.data["y"][0]
        assert x == pytest.approx(sv._pick_x_km(5, sv._dist_km.size))
        assert y == pytest.approx(137.5)

    def test_label_position_matches_scatter_position(self):
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 137.5, 0.0, 0.0, "p")]
        sv._redraw_picks()
        label = sv._pick_labels[1]
        assert label.pos().x() == pytest.approx(sv._pick_x_km(5, sv._dist_km.size))
        assert label.pos().y() == pytest.approx(137.5)

    def test_marker_tracks_the_new_window_on_a_preview_recompute(self):
        """The actual fix's intent: a pick's rendered km position TRACKS
        whichever window is currently shown — a show_preview update that
        swaps in a DIFFERENT-shaped array over a DIFFERENT (narrower,
        shifted) km window, exactly what every real pan/zoom does in the
        live app, must reposition the pick to match that NEW window's own
        linear approximation (an intentional, unavoidable trade-off — see
        the class docstring), NOT leave it at the old window's position."""
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, 0.0, 0.0, "p")]
        sv._redraw_picks()
        x_before = sv._pick_scatter.data["x"][0]

        sv.set_colormap("viridis", 1.0)
        narrower = np.zeros((50, 7), dtype=np.float32)
        # trace 5 is now OUTSIDE this new window's bounds (2..9) but still
        # clamped/tracked relative to it — see _pick_x_km's clamp.
        sv.show_preview(narrower, dist0=0.4, dist1=1.0, t0=50.0, t1=450.0,
                        vmax=1.0, c0=2, c1=9)

        x_after = sv._pick_scatter.data["x"][0]
        assert x_after != x_before
        assert x_after == pytest.approx(sv._pick_x_km(5, sv._dist_km.size))

    def test_apply_zoom_update_repositions_picks_to_the_new_window(self):
        """_apply_zoom_update must update _img_trace_lo/_img_trace_hi (the
        SAME bookkeeping show_preview now maintains) and call
        _redraw_picks so existing picks track the new re-slice's window
        bounds.

        The NUMERIC position happens to stay unchanged for this specific
        re-slice (asserted explicitly below): unlike show_preview (which
        sources its window bounds from the REAL, non-uniform dist_km
        array), _apply_zoom_update derives sub_d0/sub_d1 from a SINGLE,
        internally self-consistent linear km-per-column formula applied
        uniformly across both the full buffer and any of its sub-windows
        — so re-framing trace_index through a narrower window of that SAME
        linear formula reproduces an identical answer algebraically. This
        does not contradict the fix: the bug this turn addresses is
        specifically the mismatch between an EXTERNAL non-uniform dist_km
        lookup and the image's own internal linear approximation, and this
        function never consults dist_km at all."""
        sv = _make_view()
        sv._picks = [PickPoint(1, 8, 100.0, 0.0, 0.0, "p")]
        sv._redraw_picks()
        x_before = sv._pick_scatter.data["x"][0]

        sv._rect = (0.0, 2.0, 0.0, 500.0)
        sv._pending_ranges = [(0.5, 1.0), (0.0, 500.0)]
        sv._last_zoom_key = None   # ensure the dedup guard doesn't short-circuit
        sv._apply_zoom_update()

        assert (sv._img_trace_lo, sv._img_trace_hi) == (5, 10)
        x_after = sv._pick_scatter.data["x"][0]
        assert x_after == pytest.approx(x_before)
        assert x_after == pytest.approx(sv._pick_x_km(8, sv._dist_km.size))

    def test_non_uniform_dist_km_does_not_affect_window_relative_rendering(self):
        """_pick_x_km never reads self._dist_km for ITS OWN positioning
        once window bounds are known (only as a clamp/fallback) — so
        replacing dist_km with a wildly non-uniform array must NOT change
        where an existing pick renders, proving the fix actually
        eliminated the dist_km dependency this bug was about."""
        sv = _make_view()
        sv._picks = [PickPoint(1, 5, 100.0, 0.0, 0.0, "p")]
        sv._redraw_picks()
        x_before = sv._pick_scatter.data["x"][0]

        sv.set_distance_axis(np.concatenate([
            np.linspace(0.0, 0.1, 10), np.linspace(0.1, 2.0, 10),
        ]))
        sv._redraw_picks()

        assert sv._pick_scatter.data["x"][0] == pytest.approx(x_before)

    def test_picking_click_resolution_round_trips_through_redraw(self, monkeypatch):
        """_picking_trace_index_at_scene_pos (click → trace_index) and
        _pick_x_km (trace_index → render position) must be exact inverses
        of each other under the SAME window bounds — this is what
        guarantees a newly-created pick redraws at the EXACT spot it was
        clicked, with zero snap, regardless of dist_km's spacing."""
        from PyQt6.QtCore import QPointF
        sv = _make_view()
        for trace_index in (0, 1, 9, 10, 19):
            forward_km = sv._pick_x_km(trace_index, sv._dist_km.size)
            scene_pos = self._scene_pos_for_view(sv, forward_km, 0.0)
            resolved = sv._picking_trace_index_at_scene_pos(scene_pos)
            assert resolved == trace_index

    @staticmethod
    def _scene_pos_for_view(sv, km: float, ms: float):
        from PyQt6.QtCore import QPointF
        return sv.plot.getViewBox().mapViewToScene(QPointF(km, ms))

    def test_non_uniform_spacing_end_to_end_click_and_render_agree(self, monkeypatch):
        """Full ground-truth scenario for the live view: a show_preview
        window sourced from REAL non-uniform dist_km (a "ship slowdown"),
        exactly like preview.py's real dist0=dist_km[c_vis0]/dist1=
        dist_km[c1_idx] — double-clicking a reflector must create a pick
        whose RENDERED position matches the EXACT clicked screen position,
        and that position must differ markedly from a naive
        dist_km[trace_index] lookup (the actual reported bug)."""
        from PyQt6.QtCore import QPointF
        sv = _make_view()
        dist_km = np.concatenate([
            np.linspace(0.0, 3.0, 40), np.linspace(3.0, 3.5, 30),
            np.linspace(3.5, 8.0, 30),
        ])
        sv.set_distance_axis(dist_km)
        sv.set_colormap("viridis", 1.0)
        c_vis0, c_vis1 = 30, 80   # a window straddling the non-uniform middle
        dist0, dist1 = float(dist_km[c_vis0]), float(dist_km[c_vis1 - 1])
        arr = np.zeros((50, c_vis1 - c_vis0), dtype=np.float32)
        sv.show_preview(arr, dist0=dist0, dist1=dist1, t0=0.0, t1=500.0,
                        vmax=1.0, c0=c_vis0, c1=c_vis1)
        sv.set_pick_mode(True)
        sv.plot.getViewBox().mapSceneToView(QPointF(0.0, 0.0))   # settle

        clicked_trace = 55
        click_km = sv._pick_x_km(clicked_trace, dist_km.size)
        scene_pos = sv.plot.getViewBox().mapViewToScene(QPointF(click_km, 100.0))
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("p", True)))
        sv._on_scene_click(_FakeClickEvent(scene_pos, double=True))

        picks = sv.get_picks()
        assert len(picks) == 1
        assert picks[0].trace_index == clicked_trace
        rendered_x = sv._pick_scatter.data["x"][0]
        assert rendered_x == pytest.approx(click_km, abs=1e-6)
        # The OLD (buggy) behaviour — dist_km[trace_index] directly — would
        # have placed this pick far from where it was actually clicked.
        assert abs(rendered_x - float(dist_km[clicked_trace])) > 0.3

    def test_double_click_creates_pick_at_exact_time_under_nonzero_t0_via_preview(
        self, monkeypatch,
    ):
        """The actual Y-offset regression, reproduced via show_preview (the
        path that was completely broken before — _push_full_image's t0/dt
        bookkeeping is never reached on this path at all) with a NON-zero
        t0, the case the old frozen default (t0=0.0) got wrong.

        Checks BOTH the stored value AND the actually-RENDERED scatter
        position: storing time_ms via mapSceneToView was already correct
        in the previous implementation too (that part was never the bug) —
        the real bug was purely in _redraw_picks's RENDERING of that stored
        value back onto the image's local pixel grid via the frozen
        t0=0.0/dt_ms=1.0 defaults. A test that only checked the stored
        value would have passed even with that bug present, so the
        rendered position is the assertion that actually matters here."""
        from PyQt6.QtCore import QPointF
        sv = _make_view()
        sv.set_colormap("viridis", 1.0)
        arr = np.zeros((50, 20), dtype=np.float32)
        sv.show_preview(arr, dist0=0.0, dist1=2.0, t0=100.0, t1=600.0, vmax=1.0)
        sv.set_distance_axis(np.linspace(0.0, 2.0, 20))
        sv.set_pick_mode(True)
        sv.plot.getViewBox().mapSceneToView(QPointF(0.0, 0.0))   # settle, see _make_view
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("p", True)))

        scene_pos = sv.plot.getViewBox().mapViewToScene(QPointF(1.0, 400.0))
        sv._on_scene_click(_FakeClickEvent(scene_pos, double=True))

        picks = sv.get_picks()
        assert len(picks) == 1
        assert picks[0].time_ms == pytest.approx(400.0, abs=1.0)
        # The rendered marker must appear at view-y == 400, not wherever the
        # old local-pixel-grid conversion (using frozen t0=0/dt_ms=1
        # defaults under this t0=100 source) would have placed it.
        assert sv._pick_scatter.data["y"][0] == pytest.approx(400.0, abs=1.0)
