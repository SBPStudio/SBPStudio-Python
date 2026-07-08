"""
test_gui_scale_modes.py — GUI scale/proportions math engine
(_render.figsize_for_scale / effective_aspect) and the horizontal control
in ProcessingControls.

History
-------
A "deep audit" pass once replaced "traces per cm" (tpc) with a reciprocal
"cm_per_100_traces" control, and made VE/hybrid height rigorously depend on
the ACTUAL resulting width (h = VE * w * depth_km / total_km) instead of a
fixed constant. Both changes were REVERTED at the user's explicit request:
the width-dependent VE recompute made every drag tick on the horizontal
slider trigger a full vertical-geometry recalculation, which was too slow/
laggy in practice. The control is back to "traces per cm" (higher = more
traces packed per cm = narrower figure — yes, that direction is intentional
again, traded for speed) and VE/hybrid height is back to the fixed
``GUI_X_SCALE`` constant, fully decoupled from the live width. This file
tests THAT (current, fast) behaviour.
"""
from __future__ import annotations

import pytest

from sbp_studio.gui.tabs._render import effective_aspect, figsize_for_scale


class _FakeSource:
    """Minimal stand-in for a SegyProfile/SegyMetadata — only the attributes
    figsize_for_scale actually reads."""

    def __init__(self, n_traces=1000, ns=2000, dt_us=250, total_km=5.0):
        self.n_traces = n_traces
        self.ns = ns
        self.dt_us = dt_us
        self.total_km = total_km


def _cfg(mode, **overrides):
    base = dict(mode=mode, ratio=3.0, ve=67.0, max_aspect=5.0,
               traces_per_cm=40.0, layout_mode="decoupled")
    base.update(overrides)
    return base


# ── 1. Horizontal control: cheap, width-only, inverse relationship (by design) ─

class TestTracesPerCm:
    def test_width_decreases_as_traces_per_cm_increases(self):
        """Intentional, reverted-to-original direction: MORE traces packed per
        cm means a NARROWER figure. Kept this way because the alternative
        (width-dependent VE recompute) was too slow while dragging."""
        src = _FakeSource()
        widths = [figsize_for_scale(src, _cfg("aspect", traces_per_cm=v), 100, 1500.0)[0]
                 for v in (10.0, 40.0, 100.0, 500.0)]
        assert widths == sorted(widths, reverse=True)

    def test_width_formula_is_exact(self):
        src = _FakeSource(n_traces=1000)
        w, _ = figsize_for_scale(src, _cfg("aspect", traces_per_cm=40.0), 100, 1500.0)
        assert w == pytest.approx(1000 / 40.0 / 2.54, rel=1e-9)


# ── 2. VE/hybrid height is DECOUPLED from traces_per_cm (the performance fix) ─

class TestVeDecoupledFromWidth:
    def test_ve_height_is_fixed_regardless_of_traces_per_cm(self):
        """The load-bearing performance property: dragging the horizontal
        slider must NEVER change VE mode's height — only width. This is what
        keeps the slider cheap/instant."""
        src = _FakeSource()
        heights = [figsize_for_scale(src, _cfg("ve", traces_per_cm=v), 100, 1500.0)[1]
                  for v in (10.0, 40.0, 100.0, 500.0)]
        assert all(h == pytest.approx(heights[0], rel=1e-12) for h in heights)

    def test_ve_height_matches_the_fixed_constant_formula(self):
        src = _FakeSource(ns=2000, dt_us=250)
        velocity = 1500.0
        ve = 67.0
        _, h = figsize_for_scale(src, _cfg("ve", ve=ve, traces_per_cm=40.0), 100, velocity)
        depth_km = (src.ns * src.dt_us / 1000.0) * velocity / 2_000_000.0
        from sbp_studio.gui.tabs._render import GUI_X_SCALE
        assert h == pytest.approx(ve * depth_km / GUI_X_SCALE, rel=1e-9)

    def test_ve_does_not_need_total_km_at_all(self):
        """Unlike the reverted rigorous formula, this one never reads
        total_km — must work identically whether it's set or zero."""
        with_geometry = _FakeSource(total_km=5.0)
        without_geometry = _FakeSource(total_km=0.0)
        cfg = _cfg("ve", traces_per_cm=40.0)
        _, h1 = figsize_for_scale(with_geometry, cfg, 100, 1500.0)
        _, h2 = figsize_for_scale(without_geometry, cfg, 100, 1500.0)
        assert h1 == pytest.approx(h2, rel=1e-12)


# ── 3. Hybrid cap: still works, operating on the fixed-constant VE height ───

class TestHybridCap:
    def test_cap_inactive_for_a_moderate_line(self):
        src = _FakeSource(ns=2000)
        w, h = figsize_for_scale(src, _cfg("hybrid", ve=67.0, max_aspect=5.0,
                                            traces_per_cm=40.0), 100, 1500.0)
        assert w / h < 5.0

    def test_cap_engages_when_aspect_would_exceed_max_aspect(self):
        """A narrow traces_per_cm (-> wide figure) against a small fixed VE
        height must trip the cap."""
        src = _FakeSource(n_traces=1000, ns=2000)
        w, h = figsize_for_scale(src, _cfg("hybrid", ve=1.0, max_aspect=5.0,
                                            traces_per_cm=1.0), 100, 1500.0)
        assert w / h == pytest.approx(5.0, rel=1e-6)

    def test_cap_does_not_override_ve_when_under_the_limit(self):
        src = _FakeSource(n_traces=1000, ns=2000)
        w, h = figsize_for_scale(src, _cfg("hybrid", ve=67.0, max_aspect=500.0,
                                            traces_per_cm=40.0), 100, 1500.0)
        w_ve, h_ve = figsize_for_scale(src, _cfg("ve", ve=67.0, traces_per_cm=40.0),
                                       100, 1500.0)
        assert (w, h) == pytest.approx((w_ve, h_ve), rel=1e-9)


# ── 4. effective_aspect: free mode + consistency with figsize_for_scale ─────

class TestEffectiveAspect:
    def test_free_mode_always_none(self):
        src = _FakeSource()
        for tpc in (10.0, 500.0):
            assert effective_aspect(src, _cfg("free", traces_per_cm=tpc)) is None

    @pytest.mark.parametrize("mode", ["aspect", "ve", "hybrid"])
    def test_matches_figsize_ratio(self, mode):
        src = _FakeSource()
        cfg = _cfg(mode)
        w, h = figsize_for_scale(src, cfg, 100, 1500.0)
        assert effective_aspect(src, cfg) == pytest.approx(w / h, rel=1e-12)


# ── 5. ProcessingControls widget: viewport-driven UX (no manual Traces/cm) ──

class TestTracesPerCmControl:
    """Qt widget construction — the math above is the load-bearing check.
    Deliberately a SINGLE test constructing ONE ProcessingControls instance
    (rather than one per test method): constructing multiple instances across
    separate pytest test functions in the same process hit an unrelated Qt
    singleton-lifecycle issue with the app-wide ``language_manager`` (its
    underlying C++ object got deleted between tests) — a pytest/PyQt test-
    isolation artifact, not a bug in this code."""

    def test_widget_matches_viewport_driven_ux(self):
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.processing_controls import ProcessingControls
        from sbp_studio.gui.tabs._render import DEFAULT_TRACES_PER_CM

        pc = ProcessingControls()

        # ── The manual 'Traces / cm' control is GONE (viewport-driven UX):
        # the horizontal scale is set by wheel-zooming the section axes, in
        # every mode. Only the fixed batch-sizing constant remains.
        assert not hasattr(pc, "sld_tpc") and not hasattr(pc, "sp_tpc")
        assert not hasattr(pc, "cap_tpc")
        assert not hasattr(pc, "set_traces_per_cm")
        assert pc.scale_config()["traces_per_cm"] == pytest.approx(
            DEFAULT_TRACES_PER_CM)

        # ── scale_changed still fires synchronously from the mode values ──
        received = []
        pc.scale_changed.connect(lambda: received.append(True))
        pc.sp_ve.setValue(100.0)
        assert len(received) == 1

        # ── Good parts kept: Free default + strict state machine ──
        pc2_defaults_free = pc.rb_free.isChecked()
        # (sp_ve edit above did not switch the MODE — only its value.)
        assert pc2_defaults_free
        assert not pc.sld_deform.isEnabled()
        assert not pc.sld_ve.isEnabled()
        assert not pc.sld_maxasp.isEnabled()

        pc.rb_hybrid.setChecked(True)
        assert pc.sld_ve.isEnabled()
        assert pc.sld_maxasp.isEnabled()
        assert not pc.sld_deform.isEnabled()


# ── 6. Preview refresh coalescing: a deferred fit is STICKY, never dropped ──

class TestPendingCoalesce:
    """The 'fit lost' field regression: selecting a new file queued
    _refresh(fit=True) while the OLD file's render was in flight; a pan-settle
    tick then overwrote the pending tuple with fit=False and the replayed
    frame kept the previous zoom. The coalesce must OR the flags."""

    def test_fit_survives_later_non_fit_tick(self):
        from sbp_studio.gui.dsp.preview import _coalesce_pending
        pending = _coalesce_pending(None, (True, True))     # queued selection fit
        pending = _coalesce_pending(pending, (False, False))  # pan settle tick
        assert pending == (True, True)

    def test_overlays_sticky_too(self):
        from sbp_studio.gui.dsp.preview import _coalesce_pending
        assert _coalesce_pending((False, True), (False, False)) == (False, True)

    def test_plain_replacement_when_nothing_pending(self):
        from sbp_studio.gui.dsp.preview import _coalesce_pending
        assert _coalesce_pending(None, (False, True)) == (False, True)


# ── 7. Export dialog: paper order + default (A4 first/default, Auto last) ──

class TestExportDialogPaperOrder:
    def test_a4_default_auto_last(self):
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from sbp_studio.gui.components.export_dialog import ExportDialog

        dlg = ExportDialog()
        items = [dlg.cb_papersize.itemText(i)
                 for i in range(dlg.cb_papersize.count())]
        assert items == ["A4", "A3", "A0", "Auto"]
        assert dlg.cb_papersize.currentText() == "A4"
        assert dlg.config()["paper_size"] == "A4"
        dlg.deleteLater()


# ── 8. SeismicView 'user-modified scale' tracker (stretch-vs-letterbox) ──

class TestUserScaledTracker:
    def test_tracker_state_machine(self):
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        # Known pytest/PyQt isolation artifact (see TestTracesPerCmControl's
        # docstring): earlier Qt-widget tests in this process can leave the
        # app-wide language_manager singleton's C++ object deleted, which
        # fails ANY later widget construction. Skip in that case — this test
        # is fully exercised when run in isolation / its own process.
        from sbp_studio.gui.i18n import language_manager
        try:
            language_manager.objectName()
        except RuntimeError:
            pytest.skip("language_manager C++ object deleted by an earlier "
                        "Qt test in this process (known isolation artifact)")
        from sbp_studio.gui.components.seismic_view import SeismicView

        view = SeismicView()
        # Fresh view: default (unmodified) state.
        assert view.view_is_user_scaled() is False

        # Applying a scale-mode preset is a deliberate proportion choice.
        view.set_aspect(2.0)
        assert view.view_is_user_scaled() is True

        # Libre (None) auto-fits → back to the default state.
        view.set_aspect(None)
        assert view.view_is_user_scaled() is False

        # Manual interaction that changes the EXTENTS → user-scaled…
        vb = view.plot.getViewBox()
        (x0, x1), (y0, y1) = vb.viewRange()
        view._fit_extents = (abs(x1 - x0), abs(y1 - y0))
        vb.setXRange(x0, x0 + (x1 - x0) * 2.0, padding=0)   # zoomed out 2x
        view._on_manual_range_change()
        assert view.view_is_user_scaled() is True

        # …but a pure PAN (same extents, shifted centre) stays default.
        view.set_aspect(None)                                # reset
        assert view.view_is_user_scaled() is False
        (x0, x1), (y0, y1) = vb.viewRange()
        view._fit_extents = (abs(x1 - x0), abs(y1 - y0))
        span = x1 - x0
        vb.setXRange(x0 + span * 0.3, x1 + span * 0.3, padding=0)
        view._on_manual_range_change()
        assert view.view_is_user_scaled() is False
        view.deleteLater()


# ── 9. _dispatch_pending must never swallow a queued fit (filters race) ──

class TestDispatchPendingFitCarryover:
    """The 'auto-fit fails when filters are active' race: a selection's
    fit=True request queued behind a busy worker was DISCARDED whenever a
    structural/param pipeline event was also pending (active filters emit
    those during a selection), and the replay re-rendered with fit=False at
    the previous zoom. The pending fit must be re-queued after the replay."""

    class _Fake:
        def __init__(self, pending, sync=False, params=False):
            self._pending = pending
            self._pending_sync = sync
            self._pending_params = params
            self.calls = []

        def _on_pipeline_changed(self):
            self.calls.append("pipeline")

        def _on_params_changed(self):
            self.calls.append("params")

        def _refresh(self, *, fit, overlays):
            self.calls.append(("refresh", fit, overlays))

    def _run(self, fake):
        from sbp_studio.gui.dsp.preview import PreviewController
        PreviewController._dispatch_pending(fake)
        return fake.calls

    def test_structural_replay_requeues_the_fit(self):
        calls = self._run(self._Fake((True, True), sync=True))
        assert calls == ["pipeline", ("refresh", True, True)]

    def test_param_replay_requeues_the_fit(self):
        calls = self._run(self._Fake((True, True), params=True))
        assert calls == ["params", ("refresh", True, True)]

    def test_plain_pending_unchanged(self):
        calls = self._run(self._Fake((True, False)))
        assert calls == [("refresh", True, False)]

    def test_structural_without_pending_stays_lean(self):
        calls = self._run(self._Fake(None, sync=True))
        assert calls == ["pipeline"]

    def test_pending_fitless_refresh_not_duplicated(self):
        # A queued plain pan refresh (no fit, no overlays) need not replay
        # after a structural rebuild — the rebuild's own render covers it.
        calls = self._run(self._Fake((False, False), sync=True))
        assert calls == ["pipeline"]
