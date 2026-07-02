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
    from sbp_studio.core.model import SegyProfile, ProfileChain

# Cap on simultaneously-loaded ("hot") trace matrices. Older, non-active profiles
# beyond this are evicted back to lazy stubs (data=None) to bound RAM when many
# heavy Antarctic lines are browsed in one session.
MAX_HOT_PROFILES = 3


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
    # Emitted with a profile/chain whose CRS was just resolved/changed IN
    # PLACE (core.set_crs_override mutates detected_crs without a file
    # re-read, so neither profiles_changed nor active_profile_changed fires
    # on its own). Tabs showing that object's track refresh their map via
    # core.safe_map_coords — this notification flows ONE way (CRS change →
    # map refresh) and the refresh never changes the CRS again, so there is
    # no cycle.
    crs_updated = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # Keyed by profile path so re-adding the same file is idempotent and
        # selection can survive list reordering.
        self._profiles: "Dict[str, SegyProfile]" = {}
        self._chains: "List[ProfileChain]" = []
        self._active_profile_key: Optional[str] = None
        self._active_chain_index: Optional[int] = None
        # Keys of profiles whose trace matrices are loaded, oldest → newest (LRU).
        self._lru: List[str] = []

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
            self._lru = [k for k in self._lru if k != key]
            if self._active_profile_key == key:
                self.set_active_profile(None)
            self.profiles_changed.emit()

    def clear_profiles(self) -> None:
        self._profiles.clear()
        self._lru.clear()
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
        # Selecting an already-loaded profile refreshes its recency so it isn't
        # evicted out from under the user.
        if key is not None and self._is_loaded(key):
            self._touch(key)
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

    def add_chains(self, chains: "List[ProfileChain]") -> int:
        """Append chains (e.g. a directory import), skipping any whose name is
        already present so re-importing a campaign never creates duplicates.
        Does NOT touch the active chain. Returns the number actually added."""
        existing = {getattr(c, "label", None) for c in self._chains}
        added = 0
        for ch in chains:
            label = getattr(ch, "label", None)
            if label in existing:
                continue
            self._chains.append(ch)
            existing.add(label)
            added += 1
        if added:
            self.chains_changed.emit()
        return added

    def remove_chain(self, index: int) -> None:
        if 0 <= index < len(self._chains):
            del self._chains[index]
            if self._active_chain_index == index:
                self.set_active_chain(None)
            elif self._active_chain_index is not None and self._active_chain_index > index:
                self._active_chain_index -= 1
            self.chains_changed.emit()

    def clear_chains(self) -> None:
        self._chains.clear()
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
        # A freshly-loaded matrix is now the hottest; cap total RAM by evicting
        # the oldest non-active loaded profiles back to lazy stubs.
        if getattr(profile, "data", None) is not None:
            self._touch(profile.path)
            self._evict_if_needed()
        self.profiles_changed.emit()
        if self._active_profile_key == profile.path:
            if self._active_chain_index is None:   # don't clobber an active chain view
                self.active_profile_changed.emit(profile)

    def update_chain_data(self, chain: "ProfileChain") -> None:
        """Re-emit ``active_chain_changed`` after a chain's trace matrix has been
        lazily assembled, so the Chains tab transitions from "Loading…" to the
        live section. The chain object is mutated in place by the loader worker,
        so we only need to re-notify (no list replacement)."""
        if chain is self.active_chain:
            self.active_chain_changed.emit(chain)

    def notify_crs_updated(self, obj) -> None:
        """Call after ``core.set_crs_override(obj, ...)`` mutates a profile's
        or chain's ``detected_crs`` in place. Every observing tab/dialog
        refreshes its OWN view of ``obj`` (e.g. re-running
        ``core.safe_map_coords`` for the map track) — this method itself
        does nothing but relay the notification, so it can never trigger
        another CRS change and cannot cycle."""
        self.crs_updated.emit(obj)

    # ── LRU eviction of trace matrices (RAM bound) ───────────────────────────

    def _is_loaded(self, key: str) -> bool:
        prof = self._profiles.get(key)
        return prof is not None and getattr(prof, "data", None) is not None

    def _touch(self, key: str) -> None:
        """Mark a loaded profile as most-recently used (move to the LRU tail)."""
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)

    def _evict_if_needed(self) -> None:
        """Drop the heavy arrays (``data``/``amp_max``) of the oldest non-active
        loaded profiles until at most ``MAX_HOT_PROFILES`` remain hot. Each
        evicted profile reverts to a lazy stub (``data=None``), so re-selecting it
        transparently re-loads from disk via the existing worker path."""
        self._lru = [k for k in self._lru if self._is_loaded(k)]   # resync
        i = 0
        while len(self._lru) > MAX_HOT_PROFILES and i < len(self._lru):
            key = self._lru[i]
            if key == self._active_profile_key:    # never evict what's on screen
                i += 1
                continue
            prof = self._profiles.get(key)
            if prof is not None:
                name = getattr(prof, "name", key)
                prof.data = None                   # release the (ns × n_traces) matrix
                prof.amp_max = None
                prof.clip_p99 = None               # → identical to a load_traces=False stub
                # Lazy import: keeps heavy core (segyio/pyproj) off the GUI import
                # path; by eviction time the core is long since loaded.
                from sbp_studio.core.logger import get_logger
                get_logger("state").debug(
                    "Evicted trace matrix for %s to free RAM", name)
            self._lru.pop(i)                       # list shrank; re-check at same index
