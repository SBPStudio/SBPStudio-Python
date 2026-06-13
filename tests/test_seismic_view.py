"""
test_seismic_view.py — Viewport / aspect-ratio behaviour of the seismic canvas.

Locks in the Phase-3 zoom fix: a custom ViewBox, an aspect-lock ratio derived
from the WHOLE-profile geometry (invariant under zoom), and autoRange firing
ONLY on explicit fit — never on a routine re-lock.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QApplication

from topassuite.gui.components.seismic_view import SeismicView, _SeismicViewBox


def _qt():
    return QApplication.instance() or QApplication([])


def _fit_render(sv, dist0=0.0, dist1=10.0, t0=0.0, t1=500.0):
    """Render a full-section 'fit' frame (captures _full_rect)."""
    sv.set_colormap("viridis", 1.0)
    arr = np.zeros((100, 200), dtype=np.float32)
    sv.show_preview(arr, dist0, dist1, t0, t1, vmax=1.0, fit=True)


def test_custom_viewbox_installed():
    _qt()
    sv = SeismicView()
    assert isinstance(sv.plot.getViewBox(), _SeismicViewBox)


def test_aspect_ratio_from_full_rect_invariant_under_zoom():
    _qt()
    sv = SeismicView()
    _fit_render(sv, 0.0, 10.0, 0.0, 500.0)
    assert sv._full_rect == (0.0, 10.0, 0.0, 500.0)

    vb = sv.plot.getViewBox()
    sv.set_aspect(3.0, fit=True)               # lock 3:1 using the full geometry
    ratio = vb.state["aspectLocked"]
    assert abs(ratio - 3.0 * 500.0 / 10.0) < 1e-6     # aspect * y_ext / x_ext

    # A zoomed preview (fit=False) updates the transient rect but must NOT touch
    # the full geometry or the locked ratio → zoom preserves proportion.
    sv.show_preview(np.zeros((100, 200), np.float32),
                    2.0, 4.0, 100.0, 200.0, vmax=1.0, fit=False)
    assert sv._rect == (2.0, 4.0, 100.0, 200.0)        # transient tracks window
    assert sv._full_rect == (0.0, 10.0, 0.0, 500.0)    # geometry unchanged
    assert abs(vb.state["aspectLocked"] - ratio) < 1e-9


def test_set_aspect_autoranges_only_on_fit(monkeypatch):
    _qt()
    sv = SeismicView()
    _fit_render(sv)
    calls = []
    monkeypatch.setattr(sv.plot, "autoRange", lambda *a, **k: calls.append(1))

    sv.set_aspect(3.0, fit=False)              # routine re-lock (e.g. scale spinner)
    assert calls == []                         # must NOT yank the view to fit

    sv.set_aspect(6.0, fit=True)               # explicit fit
    assert calls == [1]


def test_rescale_recomputes_ratio_from_full_geometry():
    _qt()
    sv = SeismicView()
    _fit_render(sv, 0.0, 10.0, 0.0, 500.0)
    vb = sv.plot.getViewBox()
    sv.set_aspect(3.0, fit=True)
    r3 = vb.state["aspectLocked"]
    # Even after a deep zoom changed the transient rect, a scale change derives
    # the new ratio from the (unchanged) full geometry → exactly 2× for 3→6.
    sv.show_preview(np.zeros((100, 200), np.float32),
                    1.0, 2.0, 50.0, 100.0, vmax=1.0, fit=False)
    sv.set_aspect(6.0, fit=False)
    assert abs(vb.state["aspectLocked"] - 2.0 * r3) < 1e-6
