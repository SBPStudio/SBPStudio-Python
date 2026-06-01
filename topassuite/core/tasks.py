"""
tasks.py — Exception hierarchy, progress/cancel contracts, backend flags.

All public symbols here are used across core/, viz/, and cli/ to wire
progress reporting and graceful cancellation without coupling to any
specific toolkit.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

# ── Exception hierarchy ────────────────────────────────────────────────────────

class TopasCoreError(Exception):
    """Base class for all topassuite errors."""

class SegyLoadError(TopasCoreError):
    """Raised when a SEG-Y file cannot be read or is malformed."""

class CRSError(TopasCoreError):
    """Raised when a CRS string is invalid or cannot be resolved."""

class ReprojectionError(TopasCoreError):
    """Raised when a reprojection operation fails (partial output cleaned up)."""

class Cancelled(TopasCoreError):
    """Raised by CancelToken.check() when cancellation has been requested."""


# ── Progress callback type ─────────────────────────────────────────────────────

# Callers receive (fraction_0_to_1, message). fraction may be NaN for
# indeterminate progress. Message may be empty string.
ProgressCallback = Callable[[float, str], None]


def _noop_progress(_fraction: float, _msg: str) -> None:
    """Default no-op progress sink."""


# ── Cancel token ───────────────────────────────────────────────────────────────

class CancelToken:
    """
    Thread-safe cancel flag.

    Usage:
        tok = CancelToken()
        # in another thread:
        tok.cancel()
        # in the worker:
        tok.check()   # raises Cancelled if cancel() was called
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Signal cancellation. Idempotent."""
        self._event.set()

    def cancelled(self) -> bool:
        """True if cancel() has been called."""
        return self._event.is_set()

    def check(self) -> None:
        """Raise Cancelled if cancel() has been called."""
        if self._event.is_set():
            raise Cancelled("Operation cancelled by caller.")

    @classmethod
    def never(cls) -> "CancelToken":
        """Return a token that is never cancelled (sentinel for optional args)."""
        return cls()


# ── Log callback type (reprojection) ──────────────────────────────────────────

LogCallback = Callable[[str], None]


def _noop_log(_msg: str) -> None:
    """Default no-op log sink."""


# ── Phased timer ───────────────────────────────────────────────────────────────

class PhasedTimer:
    """
    Lightweight wall-clock timer that accumulates named phases.

    Usage
    -----
        t = PhasedTimer(enabled=True)
        with t.phase("loading"):
            data = load_profile(path)
        with t.phase("processing"):
            out = process_profile_data(...)
        print(t.report())

    Thread-safety: phases are accumulated in the calling thread only.
    Do not share a single PhasedTimer across threads.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._phases:  "dict[str, float]" = {}
        self._order:   "list[str]"         = []
        self._label:   Optional[str]       = None
        self._t0:      float               = 0.0

    # ── context-manager interface ──────────────────────────────────────────

    def phase(self, label: str) -> "PhasedTimer":
        """Use as `with timer.phase('name'): ...`"""
        self._pending_label = label
        return self

    def __enter__(self) -> "PhasedTimer":
        if self._enabled:
            import time
            self._label = getattr(self, "_pending_label", "unnamed")
            self._t0    = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self._enabled and self._label is not None:
            import time
            dt = time.perf_counter() - self._t0
            if self._label not in self._phases:
                self._phases[self._label] = 0.0
                self._order.append(self._label)
            self._phases[self._label] += dt
            self._label = None

    # ── manual start/stop ─────────────────────────────────────────────────

    def start(self, label: str) -> None:
        if self._enabled:
            import time
            self._label = label
            self._t0    = time.perf_counter()

    def stop(self) -> None:
        if self._enabled and self._label:
            import time
            dt = time.perf_counter() - self._t0
            if self._label not in self._phases:
                self._phases[self._label] = 0.0
                self._order.append(self._label)
            self._phases[self._label] += dt
            self._label = None

    # ── reporting ─────────────────────────────────────────────────────────

    def total(self) -> float:
        return sum(self._phases.values())

    def report(self, prefix: str = "") -> str:
        if not self._enabled or not self._phases:
            return ""
        total = self.total()
        lines = [f"{prefix}Timing breakdown:"]
        for label in self._order:
            t   = self._phases[label]
            pct = 100.0 * t / total if total > 0 else 0.0
            bar = "#" * int(pct / 5)
            lines.append(
                f"{prefix}  {label:<22} {t:6.2f}s  {pct:5.1f}%  {bar}")
        lines.append(f"{prefix}  {'TOTAL':<22} {total:6.2f}s")
        return "\n".join(lines)
