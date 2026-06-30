"""
preview_worker.py — Off-GUI-thread execution of Pipeline.process for the live
DSP preview.

A heavy node (the F-K dip filter's 2-D FFT, Predictive Deconvolution, Seabed
Multiple Suppression) computing on a near-cap ViewBox window can take well
over a frame; running it synchronously on the GUI thread would freeze
panning/zooming/dragging for that long. ``PipelineWorker`` runs exactly one
``Pipeline.process`` call on a QThread and reports the result back via a
queued (thread-safe) Qt signal.

It can ALSO, in the same background run, execute an optional
``global_levels_job`` callable — ``PreviewController._compute_global_levels``
bound to the not-yet-cached arguments — so that the "color pumping" fix's
full-resolution block sampling NEVER runs synchronously on the GUI thread
either (see PreviewController._refresh's cache-then-bundle logic). Both calls
share this same worker thread sequentially (never concurrently), so handing
the live ``Pipeline`` instance to both is safe — see the note below.

Only one worker is ever in flight at a time — see
``PreviewController._refresh``, which queues further triggers in
``_pending``/``_pending_sync`` rather than overlapping workers. This is what
makes it safe to hand the live ``Pipeline`` instance (and its mutable prefix
cache) straight to the worker thread: nothing else touches it while a worker
is running.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from sbp_studio.core.tasks import CancelToken, Cancelled

from .nodes import DSPContext
from .pipeline import Pipeline

# Sentinel message for the ``failed`` signal when the run aborted via
# cooperative cancellation rather than a genuine error — see
# PreviewController._on_pipeline_failed, which recognises this and skips
# the (alarming, unwarranted) error log line for an intentional abort.
CANCELLED_MESSAGE = "__cancelled__"


class PipelineWorker(QThread):
    """Runs ``pipeline.process(sub, ctx, input_token=token, cancel=...)`` on
    a worker thread, optionally followed by ``global_levels_job()`` (also
    off the GUI thread) — see module docstring.

    ``cancel`` — optional cooperative cancellation (see
    ``sbp_studio.core.tasks.CancelToken``): if a fresher request supersedes
    this one while it's still running, ``PreviewController._refresh``
    cancels this SAME token, and ``Pipeline.process``/individual nodes (see
    DSPContext.cancel) check it and raise ``Cancelled`` — caught below and
    reported via ``failed`` so the worker thread frees up immediately
    instead of finishing a now-stale computation."""

    # (token, result array, global_levels result-or-None)
    succeeded = pyqtSignal(object, object, object)
    failed = pyqtSignal(object, str)         # (token, error message)

    def __init__(self, pipeline: Pipeline, sub: np.ndarray, ctx: DSPContext,
                 token: tuple, *,
                 split_fn=None,
                 global_levels_job: Optional[Callable[[], Tuple[float, float]]] = None,
                 cancel: Optional[CancelToken] = None,
                 parent=None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._sub = sub
        self._ctx = ctx
        self._token = token
        self._split_fn = split_fn
        self._global_levels_job = global_levels_job
        self._cancel = cancel

    def run(self) -> None:  # executed on the worker thread
        try:
            if self._split_fn is not None:
                result = self._split_fn(self._cancel)
            else:
                result = self._pipeline.process(self._sub, self._ctx,
                                                input_token=self._token, cancel=self._cancel)
            levels = (self._global_levels_job() if self._global_levels_job is not None
                      else None)
        except Cancelled:
            self.failed.emit(self._token, CANCELLED_MESSAGE)
            return
        except Exception as exc:  # a bad node config must never hang the preview
            self.failed.emit(self._token, f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit(self._token, result, levels)


class BasePrepWorker(QThread):
    """Warms ``PreviewController._pc_array`` off the GUI thread when a pan
    crosses an evicted chain-segment boundary.

    No result payload — calling ``prep_fn`` is a pure side-effect: it runs
    ``_prepared_base(obj, c0, c1, _align=align)`` which rebuilds ``_pc_array``
    in place.  On ``succeeded``, ``PreviewController._on_base_prep_ready``
    re-calls ``_refresh``, which now finds the cache warm and dispatches the
    ``PipelineWorker`` instantly — no GUI thread ever blocks on disk I/O.

    ``_get_display()`` (reads Qt widget state) is NOT called inside ``prep_fn``
    — the ``_align`` value is captured on the GUI thread before dispatch and
    threaded through via ``_prepared_base``'s ``_align`` kwarg."""

    succeeded = pyqtSignal(object)       # token only  (side-effect warms _pc_array)
    failed    = pyqtSignal(object, str)  # (token, error message)

    def __init__(self, prep_fn, token, *, parent=None) -> None:
        super().__init__(parent)
        self._prep_fn = prep_fn
        self._token   = token

    def run(self) -> None:
        try:
            self._prep_fn()
        except Exception as exc:
            self.failed.emit(self._token, f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit(self._token)
