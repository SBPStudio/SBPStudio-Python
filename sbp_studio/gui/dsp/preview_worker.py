"""
preview_worker.py — Off-GUI-thread execution of Pipeline.process for the live
DSP preview.

A heavy node (the F-K dip filter's 2-D FFT, Predictive Deconvolution, Seabed
Multiple Suppression) computing on a near-cap ViewBox window can take well
over a frame; running it synchronously on the GUI thread would freeze
panning/zooming/dragging for that long. ``PipelineWorker`` runs exactly one
``Pipeline.process`` call on a QThread and reports the result back via a
queued (thread-safe) Qt signal.

Only one worker is ever in flight at a time — see
``PreviewController._refresh``, which queues further triggers in
``_pending``/``_pending_sync`` rather than overlapping workers. This is what
makes it safe to hand the live ``Pipeline`` instance (and its mutable prefix
cache) straight to the worker thread: nothing else touches it while a worker
is running.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from .nodes import DSPContext
from .pipeline import Pipeline


class PipelineWorker(QThread):
    """Runs ``pipeline.process(sub, ctx, input_token=token)`` on a worker thread."""

    succeeded = pyqtSignal(object, object)   # (token, result array)
    failed = pyqtSignal(object, str)         # (token, error message)

    def __init__(self, pipeline: Pipeline, sub: np.ndarray, ctx: DSPContext,
                 token: tuple, parent=None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._sub = sub
        self._ctx = ctx
        self._token = token

    def run(self) -> None:  # executed on the worker thread
        try:
            result = self._pipeline.process(self._sub, self._ctx, input_token=self._token)
        except Exception as exc:  # a bad node config must never hang the preview
            self.failed.emit(self._token, f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit(self._token, result)
