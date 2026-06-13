"""
main_window.py — Primary application shell.

Hosts the menu bar (Language / Theme / Help), a top brand bar, a horizontal
splitter dividing the left sidebar (profile + chain managers) from the central
tab widget, and a status bar with a task spinner, a permanent progress bar and a
GPU/CPU badge.

The window owns the single :class:`AppState`, observes it to keep the sidebar
lists in sync, and drives background work through :class:`CoreWorker` (profile
loading, chain detection). It holds no DSP/I-O logic itself — it only calls core
functions and reflects their results into the state.

All user-facing strings are English source literals resolved through Qt's
``self.tr(...)``; translations come from the ``.ts`` files (see i18n.py).
"""
from __future__ import annotations

import math
from typing import List

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QSplitter,
    QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from .i18n import LANGUAGE_NAMES, language_manager
from .state import AppState
from .tabs import ChainsTab, ReprojectorTab, VisualizerTab
from .theme import THEMES, theme
from .workers import CoreWorker


def _gpu_available() -> bool:
    """Safe GPU probe — never let backend detection break the GUI."""
    try:
        from topassuite.core._backends import gpu_available
        return gpu_available()
    except Exception:
        return False


class MainWindow(QMainWindow):
    """Top-level window: menu, sidebar, tabs and status bar."""

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState(self)
        self._gpu = _gpu_available()
        self._workers: set[CoreWorker] = set()

        # Spinner state (animated while background tasks are active).
        self._active_tasks = 0
        self._spinner_chars = ["◐", "◓", "◑", "◒"]
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(120)
        self._spinner_timer.timeout.connect(self._spinner_tick)

        self.resize(1480, 900)
        self.setMinimumSize(1100, 680)

        self._build_menubar()
        self._build_ui()

        # Floating CLI console (hidden until toggled). Child overlay of the main
        # window so it floats over the top-right corner of the content area.
        from .components.cli_console import CliConsole
        self._cli = CliConsole(self)
        self._cli.hide()
        self._cli_guide = None             # lazily built Command Guide dialog

        # Observe state → refresh the sidebar lists.
        self.state.profiles_changed.connect(self._refresh_profile_list)
        self.state.chains_changed.connect(self._refresh_chain_list)
        language_manager.language_changed.connect(self.retranslate_ui)
        theme.theme_changed.connect(self._on_theme_changed)

        self.retranslate_ui()

    # ════════════════════════════════════════════════════════════════════════
    # Menu bar
    # ════════════════════════════════════════════════════════════════════════
    def _build_menubar(self) -> None:
        mb = self.menuBar()

        # ── Language ─────────────────────────────────────────────────────────
        self._menu_lang = mb.addMenu("")
        self._lang_group = QActionGroup(self)
        self._lang_group.setExclusive(True)
        self._lang_actions: dict[str, QAction] = {}
        for code in ("es", "en"):
            act = QAction(LANGUAGE_NAMES[code], self, checkable=True)
            act.setChecked(language_manager.language == code)
            act.triggered.connect(lambda _=False, c=code: language_manager.set_language(c))
            self._lang_group.addAction(act)
            self._menu_lang.addAction(act)
            self._lang_actions[code] = act

        # ── Theme ────────────────────────────────────────────────────────────
        self._menu_theme = mb.addMenu("")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for name in THEMES:
            act = QAction("", self, checkable=True)
            act.setChecked(theme.name == name)
            act.triggered.connect(lambda _=False, n=name: theme.set_theme(n))
            self._theme_group.addAction(act)
            self._menu_theme.addAction(act)
            self._theme_actions[name] = act

        # ── CLI menu (EXACTLY between Theme and Help) ────────────────────────
        # A dropdown with: Activate Console (toggle overlay) + Command Guide.
        self._menu_cli = mb.addMenu("")
        self._act_cli = QAction("", self, checkable=True)   # "Activate Console"
        self._act_cli.triggered.connect(self._toggle_cli)
        self._menu_cli.addAction(self._act_cli)
        self._act_cli_guide = QAction("", self)             # "Command Guide"
        self._act_cli_guide.triggered.connect(self._show_cli_guide)
        self._menu_cli.addAction(self._act_cli_guide)

        # ── Help ─────────────────────────────────────────────────────────────
        self._menu_help = mb.addMenu("")
        self._act_help_module = QAction("", self)
        self._act_help_docs = QAction("", self)
        self._act_help_about = QAction("", self)
        self._act_help_module.triggered.connect(self._show_module_help)
        self._act_help_docs.triggered.connect(self._show_docs)
        self._act_help_about.triggered.connect(self._show_about)
        self._menu_help.addAction(self._act_help_module)
        self._menu_help.addAction(self._act_help_docs)
        self._menu_help.addSeparator()
        self._menu_help.addAction(self._act_help_about)

    # ── CLI console (floating top-right overlay) ─────────────────────────────

    def _toggle_cli(self, checked: bool) -> None:
        if checked:
            self._position_cli()
            self._cli.show()
            self._cli.raise_()
            self._cli.focus_input()
        else:
            self._cli.hide()

    def _position_cli(self) -> None:
        """Pin the console to the top-right corner of the content area."""
        w, h = 420, 300
        margin = 10
        top = self.menuBar().height() + margin
        x = max(margin, self.width() - w - margin)
        self._cli.setGeometry(x, top, w, h)

    def _show_cli_guide(self) -> None:
        from .components.cli_console import CliGuideDialog
        if self._cli_guide is None:
            self._cli_guide = CliGuideDialog(self)
        self._cli_guide.show()
        self._cli_guide.raise_()
        self._cli_guide.activateWindow()

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        if getattr(self, "_cli", None) is not None and self._cli.isVisible():
            self._position_cli()

    # ════════════════════════════════════════════════════════════════════════
    # Layout
    # ════════════════════════════════════════════════════════════════════════
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_tabs())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([280, 1200])
        root.addWidget(body, 1)

        self._build_statusbar()

    # ── Top brand bar ─────────────────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(48)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 4, 12, 4)

        self._lbl_brand = QLabel("◈  TOPAS  SUITE")
        self._lbl_brand.setObjectName("title")
        self._lbl_brand_sub = QLabel()
        self._lbl_brand_sub.setObjectName("subtitle")
        lay.addWidget(self._lbl_brand)
        lay.addSpacing(8)
        lay.addWidget(self._lbl_brand_sub)
        lay.addStretch(1)
        return bar

    # ── Left sidebar: profiles + chains ───────────────────────────────────────
    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setMinimumWidth(240)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        split = QSplitter(Qt.Orientation.Vertical)

        # Loaded profiles
        pf = QFrame()
        pf.setObjectName("sidebar")
        pl = QVBoxLayout(pf)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(4)

        self._hdr_profiles = QLabel()
        self._hdr_profiles.setObjectName("hdr")
        pl.addWidget(self._hdr_profiles)

        self.prof_list = QListWidget()
        # Extended (multi) selection so several tracks can be batch-added to the
        # map at once. Selecting MANY items must not load any of them; only a
        # single-item selection drives the active seismic section (see
        # _on_profile_row_changed's guard).
        self.prof_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.prof_list.currentRowChanged.connect(self._on_profile_row_changed)
        self.prof_list.itemSelectionChanged.connect(self._update_map_buttons)
        pl.addWidget(self.prof_list, 1)

        row = QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)
        self._btn_add = QPushButton()
        self._btn_remove = QPushButton()
        self._btn_clear = QPushButton()
        self._btn_add.clicked.connect(self._add_profiles)
        self._btn_remove.clicked.connect(self._remove_profile)
        self._btn_clear.clicked.connect(self.state.clear_profiles)
        row.addWidget(self._btn_add)
        row.addWidget(self._btn_remove)
        row.addStretch(1)
        row.addWidget(self._btn_clear)
        pl.addLayout(row)

        # Dedicated batch-to-map action: extract the navigation tracks of the
        # selected profiles (headers only — no trace matrix load) and inject them
        # as managed layers on the Visualizer map. Disabled until ≥1 valid
        # profile is selected.
        self._btn_prof_to_map = QPushButton()
        self._btn_prof_to_map.setObjectName("addToMap")
        self._btn_prof_to_map.setEnabled(False)
        self._btn_prof_to_map.clicked.connect(self._add_profiles_to_map)
        pl.addWidget(self._btn_prof_to_map)
        split.addWidget(pf)

        # Detected chains
        cf = QFrame()
        cf.setObjectName("sidebar")
        cl = QVBoxLayout(cf)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(4)

        self._hdr_chains = QLabel()
        self._hdr_chains.setObjectName("hdr")
        cl.addWidget(self._hdr_chains)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(6, 0, 6, 0)
        self._lbl_threshold = QLabel()
        self._lbl_threshold.setObjectName("sub")
        ctrl.addWidget(self._lbl_threshold)
        self.chain_gap = QDoubleSpinBox()
        self.chain_gap.setRange(0.1, 50.0)
        self.chain_gap.setSingleStep(0.5)
        self.chain_gap.setValue(2.0)
        self.chain_gap.setFixedWidth(64)
        ctrl.addWidget(self.chain_gap)
        ctrl.addStretch(1)
        self._btn_detect = QPushButton()
        self._btn_detect.clicked.connect(self._detect_chains)
        ctrl.addWidget(self._btn_detect)
        cl.addLayout(ctrl)

        self.chain_list = QListWidget()
        self.chain_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.chain_list.currentRowChanged.connect(self._on_chain_row_changed)
        self.chain_list.itemSelectionChanged.connect(self._update_map_buttons)
        cl.addWidget(self.chain_list, 1)

        self._btn_chain_to_map = QPushButton()
        self._btn_chain_to_map.setObjectName("addToMap")
        self._btn_chain_to_map.setEnabled(False)
        self._btn_chain_to_map.clicked.connect(self._add_chains_to_map)
        cl.addWidget(self._btn_chain_to_map)

        self._lbl_chain_hint = QLabel()
        self._lbl_chain_hint.setObjectName("info")
        self._lbl_chain_hint.setWordWrap(True)
        self._lbl_chain_hint.setContentsMargins(8, 4, 8, 6)
        cl.addWidget(self._lbl_chain_hint)

        split.addWidget(cf)
        split.setSizes([450, 350])
        outer.addWidget(split)
        return panel

    # ── Central tabs ──────────────────────────────────────────────────────────
    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tab_visualizer = VisualizerTab(self.state, self)
        self.tab_reprojector = ReprojectorTab(self.state, self)
        self.tab_chains = ChainsTab(self.state, self)
        self.tabs.addTab(self.tab_visualizer, "")
        self.tabs.addTab(self.tab_reprojector, "")
        self.tabs.addTab(self.tab_chains, "")
        return self.tabs

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)

        self._spinner_lbl = QLabel("")
        sb.addWidget(self._spinner_lbl)

        self.status_lbl = QLabel()
        self.status_lbl.setObjectName("sub")
        sb.addWidget(self.status_lbl, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedWidth(220)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

        # Cancel button — only visible while a background task is running. Wired
        # to cancel every active CoreWorker (request_cancel is idempotent).
        self._btn_cancel = QPushButton("✕")
        self._btn_cancel.setObjectName("cancelBtn")
        self._btn_cancel.setFixedWidth(28)
        self._btn_cancel.setVisible(False)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        sb.addPermanentWidget(self._btn_cancel)

        self._gpu_badge = QLabel()
        sb.addPermanentWidget(self._gpu_badge)
        self._apply_dynamic_styles()

    def _apply_dynamic_styles(self) -> None:
        """(Re)apply the few inline colours that QSS can't reach, per theme."""
        self._spinner_lbl.setStyleSheet(f"color:{theme.color('bright')}; font-weight:bold;")
        color = theme.color("ok") if self._gpu else theme.color("sub")
        self._gpu_badge.setStyleSheet(f"color:{color}; font-weight:bold; padding:0 10px;")
        warn = theme.color("warn")
        self._btn_cancel.setStyleSheet(
            f"QPushButton#cancelBtn{{color:{warn}; font-weight:bold;"
            f" border:1px solid {warn}; padding:1px 4px;}}"
            f"QPushButton#cancelBtn:hover{{background:{warn}; color:#ffffff;}}"
            f"QPushButton#cancelBtn:disabled{{color:{theme.color('sub')};"
            f" border-color:{theme.color('sub')};}}")

    def _on_theme_changed(self, _name: str) -> None:
        self._apply_dynamic_styles()

    # ════════════════════════════════════════════════════════════════════════
    # State → view sync
    # ════════════════════════════════════════════════════════════════════════
    def _refresh_profile_list(self) -> None:
        self.prof_list.blockSignals(True)
        self.prof_list.clear()
        for prof in self.state.profiles.values():
            mark = "⚠ " if getattr(prof, "error", None) else ""
            self.prof_list.addItem(mark + prof.name)
        self.prof_list.blockSignals(False)

    def _refresh_chain_list(self) -> None:
        self.chain_list.blockSignals(True)
        self.chain_list.clear()
        for chain in self.state.chains:
            self.chain_list.addItem(getattr(chain, "label", getattr(chain, "name", "—")))
        self.chain_list.blockSignals(False)

    def _on_profile_row_changed(self, row: int) -> None:
        # Multi-selection (Ctrl/Shift) is for batch 'Add to map' ONLY — it must
        # not disturb the active section. Only a single highlighted row drives the
        # active profile + its (lazy) trace load.
        if len(self.prof_list.selectedItems()) > 1:
            return
        keys = list(self.state.profiles.keys())
        key = keys[row] if 0 <= row < len(keys) else None
        self.state.set_active_profile(key)
        # Kick off trace loading for stub profiles so the tab shows "Loading…"
        # instead of waiting for the user to click Render.
        if key is not None:
            prof = self.state.profiles.get(key)
            if prof is not None and getattr(prof, "data", None) is None and not getattr(prof, "error", None):
                self._load_profile_traces(prof)

    def _load_profile_traces(self, profile) -> None:
        """Background worker: load the trace matrix for a header-only stub."""
        path = profile.path
        self.task_started(self.tr("Loading traces…"))

        def job(progress, cancel):
            from topassuite.core import load_profile
            progress(float("nan"), "")
            return load_profile(path, load_traces=True)

        self._run_worker(job, self.state.update_profile_data)

    def _on_chain_row_changed(self, row: int) -> None:
        self.state.set_active_chain(row if row >= 0 else None)
        # Lazily assemble the stitched trace matrix the first time a chain is
        # viewed (mirrors the per-profile lazy load). The tab shows "Loading…"
        # meanwhile and transitions once update_chain_data re-emits.
        chain = self.state.active_chain
        if chain is not None and getattr(chain, "data", None) is None:
            self._load_chain_traces(chain)

    def _load_chain_traces(self, chain) -> None:
        """Background worker: assemble a chain's trace matrix on demand, loading
        any evicted constituent profiles from disk inside the core."""
        self.task_started(self.tr("Loading chain traces…"))

        def job(progress, cancel):
            progress(float("nan"), "")
            chain.load_chain_traces(cancel=cancel)
            return chain

        self._run_worker(job, self.state.update_chain_data)

    # ════════════════════════════════════════════════════════════════════════
    # Core-backed actions (run on background CoreWorker threads)
    # ════════════════════════════════════════════════════════════════════════
    def _add_profiles(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Add profiles"), "",
            self.tr("SEG-Y (*.sgy *.segy);;All files (*)"))
        if not paths:
            return
        self.task_started(self.tr("Reading headers…"))

        def job(progress, cancel) -> list:
            # Lazy-load: read only SEG-Y headers (no trace data).
            # Traces are loaded on first selection or on demand before rendering.
            from topassuite.core import load_profile
            loaded = []
            n = len(paths)
            for i, path in enumerate(paths):
                cancel.check()
                progress(i / n, "")
                loaded.append(load_profile(path, load_traces=False))
            progress(1.0, "")
            return loaded

        self._run_worker(job, self._on_profiles_loaded)

    def _on_profiles_loaded(self, profiles: list) -> None:
        ok = errors = 0
        for prof in profiles:
            if getattr(prof, "error", None):
                errors += 1
            else:
                ok += 1
            self.state.add_profile(prof)
        # Select the first newly loaded valid profile if none is active.
        if self.state.active_profile is None and self.prof_list.count():
            self.prof_list.setCurrentRow(0)
        self.status_lbl.setText(
            self.tr("Loaded {0} profile(s), {1} error(s).").format(ok, errors))

    def _detect_chains(self) -> None:
        profiles = list(self.state.profiles.values())
        if not profiles:
            self.status_lbl.setText(self.tr("Add profiles first."))
            return
        gap = float(self.chain_gap.value())
        self.task_started(self.tr("Detecting chains…"))

        def job(progress, cancel) -> list:
            from topassuite.core import detect_chains
            progress(float("nan"), "")
            # Detection + chain assembly is now LIGHTWEIGHT: it groups by dt_us,
            # geometry and timestamps — all resident on the header stubs. No trace
            # matrix is loaded here, so detection stays fast and RAM-flat. The
            # stitched matrix is assembled lazily when a chain is viewed.
            return detect_chains(profiles, gap_km=gap)

        self._run_worker(job, self._on_chains_detected)

    def _on_chains_detected(self, chains: list) -> None:
        self.state.set_chains(chains)
        self._lbl_chain_hint.setText(
            self.tr("{0} chain(s) detected.").format(len(chains)))
        if chains:
            self.chain_list.setCurrentRow(0)

    def _remove_profile(self) -> None:
        row = self.prof_list.currentRow()
        keys = list(self.state.profiles.keys())
        if 0 <= row < len(keys):
            self.state.remove_profile(keys[row])

    # ── Batch 'Add to map' (navigation tracks → map layers) ───────────────────
    def _selected_profiles(self) -> list:
        """SegyProfiles for the highlighted rows, skipping errored ones."""
        keys = list(self.state.profiles.keys())
        out = []
        for item in self.prof_list.selectedItems():
            row = self.prof_list.row(item)
            if 0 <= row < len(keys):
                prof = self.state.profiles.get(keys[row])
                if prof is not None and not getattr(prof, "error", None):
                    out.append(prof)
        return out

    def _selected_chains(self) -> list:
        out = []
        for item in self.chain_list.selectedItems():
            row = self.chain_list.row(item)
            if 0 <= row < len(self.state.chains):
                out.append(self.state.chains[row])
        return out

    def _update_map_buttons(self) -> None:
        """Enable each 'Add to map' button only when its list has a valid
        selection. Cheap — runs on every selection change."""
        self._btn_prof_to_map.setEnabled(bool(self._selected_profiles()))
        self._btn_chain_to_map.setEnabled(bool(self._selected_chains()))

    def _add_profiles_to_map(self) -> None:
        self._tracks_to_map(self._selected_profiles(), self.tab_visualizer)

    def _add_chains_to_map(self) -> None:
        self._tracks_to_map(self._selected_chains(), self.tab_chains)

    def _tracks_to_map(self, objs: list, tab) -> None:
        """Extract the navigation tracks of *objs* (profiles or chains) and add
        them to *tab*'s map as managed layers — in ONE background pass.

        Memory-safe: it reads ONLY the spatial vectors already resident on the
        header stubs (track_lons/track_lats); it never loads a trace matrix and
        never touches the LRU hot set. UX-safe: the active profile and the live
        seismic section are left completely undisturbed.
        """
        # Snapshot lightweight geometry on the GUI thread (trivial attribute
        # reads). Reprojection to WGS84 is the only real work → do it off-thread.
        specs = []
        for obj in objs:
            lons = getattr(obj, "track_lons", None)
            lats = getattr(obj, "track_lats", None)
            if lons is None or lats is None or len(lons) == 0:
                continue
            name = getattr(obj, "label", None) or getattr(obj, "name", "track")
            specs.append((name, lons, lats, getattr(obj, "detected_crs", None)))
        if not specs:
            self.status_lbl.setText(self.tr("No navigation tracks to add."))
            return
        self.task_started(self.tr("Extracting navigation tracks…"))

        def job(progress, cancel) -> list:
            from topassuite.core import to_geographic
            out = []
            n = len(specs)
            for i, (name, lons, lats, crs) in enumerate(specs):
                cancel.check()
                progress(i / n, "")
                x, y = to_geographic(lons, lats, crs)   # passthrough if geographic
                out.append((name, x, y))
            progress(1.0, "")
            return out

        self._run_worker(job, lambda tracks: self._on_tracks_extracted(tracks, tab))

    def _on_tracks_extracted(self, tracks: list, tab) -> None:
        tab.reveal_map()
        for name, x, y in tracks:
            tab.map_view.add_track_layer(name, x, y)
        self.tabs.setCurrentWidget(tab)
        self.status_lbl.setText(
            self.tr("Added {0} track(s) to the map.").format(len(tracks)))

    # ── Task service (used by tabs to dispatch background renders) ────────────
    def run_task(self, job, on_success, message: str = "") -> None:
        """Start a background job with status feedback. Public tab-facing API."""
        self.task_started(message)
        self._run_worker(job, on_success)

    # ── Worker plumbing ───────────────────────────────────────────────────────
    def _run_worker(self, job, on_success) -> None:
        worker = CoreWorker(job)
        worker.progress.connect(self.on_progress)
        worker.succeeded.connect(on_success)
        worker.failed.connect(self.show_error)
        worker.finished.connect(lambda w=worker: self._finish_worker(w))
        self._workers.add(worker)
        worker.start()

    def _finish_worker(self, worker: CoreWorker) -> None:
        self._workers.discard(worker)
        self.task_finished()

    # ════════════════════════════════════════════════════════════════════════
    # Status bar / progress (worker-facing slots)
    # ════════════════════════════════════════════════════════════════════════
    def task_started(self, message: str = "") -> None:
        self._active_tasks += 1
        if message:
            self.status_lbl.setText(message)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._btn_cancel.setEnabled(True)        # re-enable after a prior cancel
        self._btn_cancel.setVisible(True)
        if not self._spinner_timer.isActive():
            self._spinner_idx = 0
            self._spinner_lbl.setText(self._spinner_chars[0])
            self._spinner_timer.start()

    def on_progress(self, fraction: float, message: str = "") -> None:
        if not math.isnan(fraction):
            self._progress.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        if message:
            self.status_lbl.setText(message)

    def task_finished(self, message: str = "") -> None:
        self._active_tasks = max(0, self._active_tasks - 1)
        if message:
            self.status_lbl.setText(message)
        if self._active_tasks == 0:
            self._progress.setVisible(False)
            self._btn_cancel.setVisible(False)
            self._spinner_timer.stop()
            self._spinner_lbl.setText("")

    def _on_cancel_clicked(self) -> None:
        """Cooperatively cancel every running CoreWorker. Each task aborts at its
        next ``cancel.check()``; the button disables to show the request landed
        and hides once the task actually finishes."""
        requested = False
        for worker in list(self._workers):
            if worker.isRunning():
                worker.request_cancel()
                requested = True
        if requested:
            self.status_lbl.setText(self.tr("Cancelling…"))
            self._btn_cancel.setEnabled(False)

    def show_error(self, title: str, message: str) -> None:
        """Slot for CoreWorker.failed — shows a clean modal dialog."""
        QMessageBox.critical(self, title, message)

    def notify(self, message: str) -> None:
        """Set a transient status-bar message (used by tabs for export feedback)."""
        self.status_lbl.setText(message)

    def _spinner_tick(self) -> None:
        if self._active_tasks <= 0:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        self._spinner_lbl.setText(self._spinner_chars[self._spinner_idx])

    # ════════════════════════════════════════════════════════════════════════
    # Help actions
    # ════════════════════════════════════════════════════════════════════════
    def _show_module_help(self) -> None:
        title = self.tabs.tabText(self.tabs.currentIndex()).strip()
        QMessageBox.information(self, self.tr("How this module works"), title)

    def _show_docs(self) -> None:
        QMessageBox.information(self, self.tr("Documentation"),
                                "https://github.com/TopasSuite/TopasSuite-Python")

    def _show_about(self) -> None:
        QMessageBox.about(
            self, self.tr("About TOPAS Suite"),
            self.tr("TOPAS Suite — processing and visualisation of SEG-Y seismic "
                    "data (Topas sonars).\nPyQt6 interface, decoupled from the "
                    "processing core."))

    # ════════════════════════════════════════════════════════════════════════
    # i18n
    # ════════════════════════════════════════════════════════════════════════
    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("TOPAS Suite"))

        # Menus
        self._menu_lang.setTitle(self.tr("Language"))
        self._menu_theme.setTitle(self.tr("Theme"))
        self._menu_cli.setTitle(self.tr("CLI"))
        self._act_cli.setText(self.tr("Activate Console"))
        self._act_cli_guide.setText(self.tr("Command Guide"))
        self._menu_help.setTitle(self.tr("Help"))
        self._btn_cancel.setToolTip(self.tr("Cancel the current task"))
        self._theme_actions["dark"].setText(self.tr("Dark"))
        self._theme_actions["light"].setText(self.tr("Light"))
        self._act_help_module.setText(self.tr("How this module works"))
        self._act_help_docs.setText(self.tr("Documentation"))
        self._act_help_about.setText(self.tr("About TOPAS Suite"))
        # Keep checkable states in sync with the managers.
        for code, act in self._lang_actions.items():
            act.setChecked(language_manager.language == code)
        for name, act in self._theme_actions.items():
            act.setChecked(theme.name == name)

        # Top bar
        self._lbl_brand_sub.setText(self.tr("SEG-Y · Multi-profile · PyQt6"))

        # Sidebar
        self._hdr_profiles.setText(self.tr("LOADED PROFILES"))
        self._btn_add.setText(self.tr("＋ Add"))
        self._btn_remove.setText(self.tr("✖ Remove"))
        self._btn_clear.setText(self.tr("✖✖ Clear"))
        self._btn_prof_to_map.setText(self.tr("🗺 Add to map"))
        self._btn_prof_to_map.setToolTip(
            self.tr("Add the selected profiles' navigation tracks to the map"))
        self._btn_chain_to_map.setText(self.tr("🗺 Add to map"))
        self._btn_chain_to_map.setToolTip(
            self.tr("Add the selected chains' navigation tracks to the map"))
        self._hdr_chains.setText(self.tr("DETECTED CHAINS"))
        self._lbl_threshold.setText(self.tr("Threshold (km):"))
        self._btn_detect.setText(self.tr("🔍 Detect"))
        if not self.state.chains:
            self._lbl_chain_hint.setText(self.tr('Load profiles and click "Detect".'))

        # Tabs
        self.tabs.setTabText(0, self.tr("  ▣  Visualizer  "))
        self.tabs.setTabText(1, self.tr("  ⇄  Reprojector  "))
        self.tabs.setTabText(2, self.tr("  ⛓  Chains  "))

        # Status bar
        if self._active_tasks == 0:
            self.status_lbl.setText(self.tr("Ready — add one or more SEG-Y profiles."))
        self._gpu_badge.setText(self.tr("⚡ GPU") if self._gpu else self.tr("○ CPU"))
