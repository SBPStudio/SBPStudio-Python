"""
test_cancellation.py — Cooperative cancellation of background tasks.

Three layers:
  * Core   — a pre-cancelled token aborts the reproject trace loop promptly.
  * Worker — CoreWorker honours a cancel requested mid-run (no success emitted).
  * GUI    — the status-bar Cancel button appears only while busy and requests
             cancellation on the active workers.
"""
from __future__ import annotations

import time

import pytest
from PyQt6.QtWidgets import QApplication


def _qt():
    return QApplication.instance() or QApplication([])


# ── Core: the trace loop honours a cancel token ────────────────────────────────

def test_reproject_aborts_on_cancelled_token(tmp_path):
    from tests.make_synthetic_segy import make_synthetic_segy
    from sbp_studio.core import load_profile, reproject_one
    from sbp_studio.core.tasks import CancelToken, Cancelled
    p = make_synthetic_segy(str(tmp_path / "c.sgy"), n_traces=50,
                            base_lon=-3.0, base_lat=43.0)
    prof = load_profile(p, load_traces=False)
    token = CancelToken()
    token.cancel()                              # already cancelled before we start
    out = str(tmp_path / "c_REPROY.sgy")
    with pytest.raises(Cancelled):
        reproject_one(prof, "EPSG:4326", "EPSG:32630", 2,
                      cancel=token, out_path=out)
    # Partial output must be cleaned up, not left half-written.
    import os
    assert not os.path.exists(out)


# ── Worker: cancel requested mid-run → no success, finishes cleanly ────────────

def test_coreworker_cancel_midrun_emits_no_success():
    app = _qt()
    from sbp_studio.gui.workers.base import CoreWorker
    succeeded, failed = [], []

    def job(progress, cancel):
        for _ in range(2000):
            cancel.check()                      # raises Cancelled once requested
            time.sleep(0.002)
        return "done"

    worker = CoreWorker(job)
    worker.succeeded.connect(succeeded.append)
    worker.failed.connect(lambda *a: failed.append(a))
    worker.start()
    time.sleep(0.1)                             # let it spin, then cancel
    worker.request_cancel()
    assert worker.wait(5000)                    # bounded wait → can never hang
    app.processEvents()                         # deliver any queued signals

    assert succeeded == []          # cancelled → result discarded
    assert failed == []             # cancellation is NOT an error
    assert worker.cancelled is True


def test_coreworker_runs_to_completion_without_cancel():
    app = _qt()
    from sbp_studio.gui.workers.base import CoreWorker
    got = []
    worker = CoreWorker(lambda progress, cancel: 21 * 2)
    worker.succeeded.connect(got.append)
    worker.start()
    assert worker.wait(5000)
    app.processEvents()
    assert got == [42]


# ── GUI: Cancel button lifecycle + wiring ──────────────────────────────────────

def test_cancel_button_lifecycle_and_wiring():
    _qt()
    from sbp_studio.gui.main_window import MainWindow
    win = MainWindow()
    # Use isHidden() (explicit-hidden flag) not isVisible() — the latter is False
    # whenever the top-level window hasn't been shown, regardless of our intent.
    assert win._btn_cancel.isHidden() is True          # idle → hidden

    # Inject a fake running worker and simulate task start.
    class _FakeWorker:
        def __init__(self):
            self.cancel_requested = False
        def isRunning(self):
            return True
        def request_cancel(self):
            self.cancel_requested = True

    fake = _FakeWorker()
    win._workers.add(fake)
    win.task_started("Working…")
    assert win._btn_cancel.isHidden() is False and win._btn_cancel.isEnabled()

    win._on_cancel_clicked()
    assert fake.cancel_requested is True               # request reached the worker
    assert win._btn_cancel.isEnabled() is False        # disabled as feedback
    assert win.status_lbl.text() == win.tr("Cancelling…")

    win._workers.discard(fake)
    win.task_finished()
    assert win._btn_cancel.isHidden() is True          # back to idle → hidden
