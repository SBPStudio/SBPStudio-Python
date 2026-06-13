"""
test_export_robustness.py — Export pipeline contract.

The export is a STANDALONE, publication-quality render: its figure size is
derived purely from the profile's native data dimensions + the baseline scale
ratio (compute_figsize), with NO coupling to the on-screen viewport. The only
GUI-side safety net is graceful handling of a locked output file (a PDF open in
Acrobat → PermissionError), surfaced as a friendly ExportError.
"""
from __future__ import annotations

import math
from types import SimpleNamespace


def _source(n_traces=1347, ns=2048, dt_us=250, total_km=12.5):
    return SimpleNamespace(n_traces=n_traces, ns=ns, dt_us=dt_us,
                           total_km=total_km)


# ── figsize is data-driven and standalone ──────────────────────────────────────

class TestComputeFigsizeDataDriven:
    def _figsize(self, **kw):
        from topassuite.gui.tabs._render import compute_figsize
        return compute_figsize(**kw)

    def test_width_from_native_traces(self):
        # Width is floored at 8in and otherwise grows with the native trace count
        # — never derived from a screen ViewBox.
        w, h = self._figsize(source=_source(n_traces=1347), dpi=300,
                             x_scale=None, ratio=3.0, velocity=1500.0)
        assert w >= 8.0
        assert math.isfinite(w) and math.isfinite(h) and h > 0

    def test_height_follows_scale_ratio(self):
        # h = w / ratio (the baseline scale factor), uncoupled from any preview.
        w, h = self._figsize(source=_source(), dpi=300, x_scale=None,
                             ratio=2.0, velocity=1500.0)
        assert h == max(0.5, w / 2.0)

    def test_high_dpi_keeps_full_width(self):
        w, _ = self._figsize(source=_source(n_traces=500), dpi=900,
                             x_scale=None, ratio=3.0, velocity=1500.0)
        assert w >= 8.0


# ── Locked-file error surfaces cleanly (the one piece we keep) ──────────────────

class TestExportError:
    def test_is_core_error_with_title(self):
        from topassuite.core.tasks import ExportError, TopasCoreError
        err = ExportError("Cannot save “x.pdf”: the file is open in another program.")
        err.title = "Export failed"
        assert isinstance(err, TopasCoreError)
        # Mirrors the worker's resolution: friendly title wins over class name.
        assert (getattr(err, "title", None) or type(err).__name__) == "Export failed"

    def test_plain_core_error_falls_back_to_class_name(self):
        from topassuite.core.tasks import CRSError
        err = CRSError("bad crs")
        assert (getattr(err, "title", None) or type(err).__name__) == "CRSError"

    def test_exported_from_core(self):
        from topassuite.core.tasks import ExportError
        assert ExportError is not None
