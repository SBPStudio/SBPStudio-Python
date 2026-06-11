"""
state.py — Centralised application state (the single source of truth).

The GUI follows an observer pattern: :class:`AppState` owns the loaded profiles
and detected chains, and emits Qt signals when they change. Tabs and side panels
*observe* these signals and re-read state, rather than each holding their own
duplicate copy of the data.

This object holds references only — all heavy data (trace matrices, coordinates)
lives inside the core ``SegyProfile`` / ``ProfileChain`` objects it stores. It
contains no DSP or I/O logic; loading/processing happens in workers that then
push results here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:  # avoid importing heavy core deps (segyio/pyproj) at GUI import time
    from topassuite.core.model import SegyProfile, ProfileChain


class AppState(QObject):
    """Observable container for the currently loaded profiles and chains."""

    # Emitted after the set of loaded profiles changes (add/remove/clear).
    profiles_changed = pyqtSignal()
    # Emitted after the list of detected chains changes.
    chains_changed = pyqtSignal()
    # Emitted with the newly active SegyProfile (or None).
    active_profile_changed = pyqtSignal(object)
    # Emitted with the newly active ProfileChain (or None).
    active_chain_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Keyed by profile path so re-adding the same file is idempotent and
        # selection can survive list reordering.
        self._profiles: "Dict[str, SegyProfile]" = {}
        self._chains: "List[ProfileChain]" = []
        self._active_profile_key: Optional[str] = None
        self._active_chain_index: Optional[int] = None

    # ── Profiles ────────────────────────────────────────────────────────────

    @property
    def profiles(self) -> "Dict[str, SegyProfile]":
        return self._profiles

    def add_profile(self, profile: "SegyProfile") -> None:
        """Insert/replace a profile (keyed by its path) and notify observers."""
        self._profiles[profile.path] = profile
        self.profiles_changed.emit()

    def remove_profile(self, key: str) -> None:
        if key in self._profiles:
            del self._profiles[key]
            if self._active_profile_key == key:
                self.set_active_profile(None)
            self.profiles_changed.emit()

    def clear_profiles(self) -> None:
        self._profiles.clear()
        self.set_active_profile(None)
        self.profiles_changed.emit()

    @property
    def active_profile(self) -> "Optional[SegyProfile]":
        if self._active_profile_key is None:
            return None
        return self._profiles.get(self._active_profile_key)

    def set_active_profile(self, key: Optional[str]) -> None:
        """Set the active profile by key (or None) and emit the change."""
        if key is not None and key not in self._profiles:
            key = None
        self._active_profile_key = key
        self.active_profile_changed.emit(self.active_profile)

    # ── Chains ──────────────────────────────────────────────────────────────

    @property
    def chains(self) -> "List[ProfileChain]":
        return self._chains

    def set_chains(self, chains: "List[ProfileChain]") -> None:
        """Replace the detected-chains list and reset the active chain."""
        self._chains = list(chains)
        self.set_active_chain(None)
        self.chains_changed.emit()

    @property
    def active_chain(self) -> "Optional[ProfileChain]":
        if self._active_chain_index is None:
            return None
        if 0 <= self._active_chain_index < len(self._chains):
            return self._chains[self._active_chain_index]
        return None

    def set_active_chain(self, index: Optional[int]) -> None:
        if index is not None and not (0 <= index < len(self._chains)):
            index = None
        self._active_chain_index = index
        self.active_chain_changed.emit(self.active_chain)

    # ── Lazy-load update ────────────────────────────────────────────────────

    def update_profile_data(self, profile: "SegyProfile") -> None:
        """Replace a header-only stub with the fully-loaded profile.

        Called by the trace-loader worker after it finishes. Re-emits
        ``active_profile_changed`` if this profile is currently active so tabs
        can transition from "Loading…" to "Press Render".
        """
        if profile.path not in self._profiles:
            return
        self._profiles[profile.path] = profile
        self.profiles_changed.emit()
        if self._active_profile_key == profile.path:
            self.active_profile_changed.emit(profile)
