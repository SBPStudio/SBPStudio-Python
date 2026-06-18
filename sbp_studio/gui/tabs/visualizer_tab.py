"""
visualizer_tab.py — Tab A · Visualizer (unified profiles + chains).

A SINGLE seismic/map/spectrum/header viewer that renders EITHER an individual
:class:`SegyProfile` (selected in the sidebar's "Loaded profiles" list) OR a
merged :class:`ProfileChain` (selected in the "Detected chains" list). Both
sidebar lists feed this one tab.

Strategy routing (Disciplined Strategy + Adapter)
-------------------------------------------------
The type-specific behaviour is delegated to a :class:`SourceHandler` strategy
(``ProfileHandler`` / ``ChainHandler``, see ``_handlers.py``). The tab holds
``self._handler``, set by whichever sidebar list was selected LAST. There is NO
boolean ``_is_chain`` / ``_mode`` dispatch anywhere — ``_active_object`` and
``_export_basename`` resolve through the handler, and the shared
:class:`SubTabbedTab` base drives load/render/export entirely through the
handler interface, so it never knows which kind it is showing. "Last selection
wins" falls out automatically.

None-guards stop the INACTIVE list from clobbering the live view: removing a
profile fires ``active_profile_changed(None)`` even while a chain is on screen,
so a None notification only clears when its handler is the active one.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ._base import SubTabbedTab
from ._handlers import ChainHandler, ProfileHandler, SourceHandler
from ..state import AppState


class VisualizerTab(SubTabbedTab):
    """Tab A — unified live viewer for a selected profile OR chain."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(state, tasks, parent=parent)
        # One reusable strategy per source kind; selection just reassigns the
        # active handler. Constructed AFTER super().__init__() so the Qt object
        # (and the view widgets the handlers drive) already exist.
        self._handlers = {
            "profile": ProfileHandler(self),
            "chain": ChainHandler(self),
        }
        self._handler: SourceHandler | None = None
        self.state.active_profile_changed.connect(self._on_profile_selected)
        self.state.active_chain_changed.connect(self._on_chain_selected)

    def empty_message(self) -> str:
        return self.tr("Select a profile or chain in the list on the left")

    # ── Handler-resolved hooks (drive the shared base: export, HQ, DPI, preview) ──

    def _active_object(self):
        # getattr guard: the base class __init__ calls _update_dpi_estimate()
        # (→ _active_object) BEFORE this subclass assigns self._handler after
        # super().__init__(), so it may not exist yet during construction.
        handler = getattr(self, "_handler", None)
        return handler.active_object() if handler is not None else None

    def _export_basename(self) -> str:
        handler = getattr(self, "_handler", None)
        if handler is None:
            return "export"
        return handler.basename(handler.active_object())

    def source_handler(self, *, is_chain: bool) -> SourceHandler:
        """Return the strategy for a batch of the given kind, INDEPENDENT of the
        live view's current handler. A batch's kind comes from the sidebar list
        it was launched from (which can differ from what is on screen — e.g.
        rubber-band-selecting profiles while a chain is the active view), so the
        caller resolves it here rather than the type-agnostic base reading
        ``self._handler``."""
        return self._handlers["chain" if is_chain else "profile"]

    # ── Selection slots (signal wiring unchanged — they just set the handler) ──

    def _on_profile_selected(self, profile: object) -> None:
        if profile is None:
            # Only clear if a profile is what's currently showing; a None here
            # while a chain is active is the inactive list deselecting — ignore.
            if self._handler is self._handlers["profile"]:
                self._handler = None
                self._show_empty()
            return
        self._handler = self._handlers["profile"]
        self._handler.on_selected(profile)

    def _on_chain_selected(self, chain: object) -> None:
        if chain is None:
            if self._handler is self._handlers["chain"]:
                self._handler = None
                self._show_empty()
            return
        self._handler = self._handlers["chain"]
        self._handler.on_selected(chain)

    # ── Shared teardown ───────────────────────────────────────────────────────

    def _show_empty(self) -> None:
        self.preview.set_source(None)
        self._map.clear()
        self._headers.clear()
        for page in self.pages:
            page.show_placeholder(self.empty_message())
