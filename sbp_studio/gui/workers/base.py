"""
base.py — Background worker that bridges the core task contract to Qt signals.

The core exposes long-running functions (load_profile, process_*_data,
reproject_*, …) that accept:

    * a ``ProgressCallback``  : Callable[[float, str], None]
    * a ``CancelToken``       : cooperative cancellation flag

:class:`CoreWorker` runs such a function on its own ``QThread`` and re-emits its
progress, success, and failure as thread-safe Qt signals. Cross-thread signal
emission is queued automatically by Qt, so slots connected to these signals run
on the GUI thread — safe for touching widgets.

Usage
-----
    def job(progress, cancel):
        return load_profile(path, progress=progress, cancel=cancel)

    worker = CoreWorker(job)
    worker.progress.connect(window.on_progress)
    worker.succeeded.connect(self._on_loaded)      # receives the return value
    worker.failed.connect(window.show_error)       # (title, message)
    worker.finished.connect(worker.deleteLater)    # built-in QThread signal
    worker.start()

    # later, e.g. from a Cancel button:
    worker.request_cancel()

The ``job`` callable MUST accept exactly ``(progress, cancel)`` positionally.
Wrap core calls in a small lambda/partial to adapt their signatures.
"""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal

from sbp_studio.core.logger import get_logger
from sbp_studio.core.tasks import CancelToken, Cancelled, TopasCoreError

_LOG = get_logger("worker")

# A unit of background work: receives a progress sink and a cancel token,
# returns any result (delivered via the ``succeeded`` signal).
Job = Callable[[Callable[[float, str], None], CancelToken], Any]


class CoreWorker(QThread):
    """Runs a single :data:`Job` on a background thread, reporting via signals."""

    # (fraction_0_to_1 | NaN, message) — forwarded from the core ProgressCallback.
    progress = pyqtSignal(float, str)
    # The job's return value, on successful (non-cancelled) completion.
    succeeded = pyqtSignal(object)
    # (title, message) — a handled error suitable for a QMessageBox.
    failed = pyqtSignal(str, str)
    # ``finished`` (no args) is inherited from QThread and always fires last.

    def __init__(self, job: Job, parent=None) -> None:
        super().__init__(parent)
        self._job = job
        self._token = CancelToken()

    # ── Cancellation ────────────────────────────────────────────────────────

    def request_cancel(self) -> None:
        """Cooperatively ask the running job to stop (idempotent)."""
        self._token.cancel()

    @property
    def cancelled(self) -> bool:
        return self._token.cancelled()

    # ── Thread body ─────────────────────────────────────────────────────────

    def run(self) -> None:  # executed on the worker thread
        def _progress(fraction: float, message: str = "") -> None:
            # Emitting across threads is queued to the GUI thread by Qt.
            self.progress.emit(float(fraction), str(message))

        try:
            result = self._job(_progress, self._token)
        except Cancelled:
            # Normal, user-requested stop — not an error.
            _LOG.debug("Background task cancelled by user.")
            return
        except TopasCoreError as exc:
            # Handled/expected error shown to the user — record it for the log.
            # An exception may carry a friendly ``title`` (e.g. ExportError for a
            # locked output file); otherwise fall back to the class name.
            _LOG.error("Background task failed: %s: %s", type(exc).__name__, exc)
            title = getattr(exc, "title", None) or type(exc).__name__
            self.failed.emit(title, str(exc))
            return
        except Exception as exc:  # unexpected: log the FULL traceback to app.log
            _LOG.exception("Unhandled error in background task")
            self.failed.emit("Error", f"{type(exc).__name__}: {exc}")
            return

        # A late cancel may have arrived while the job was finishing; honour it.
        if not self._token.cancelled():
            self.succeeded.emit(result)
