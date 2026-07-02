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

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QPushButton, QSlider, QSplitter, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

from .i18n import LANGUAGE_NAMES, language_manager
from .state import AppState
from .tabs import ReprojectorTab, VisualizerTab
from .theme import THEMES, theme
from .workers import CoreWorker


def _gpu_available() -> bool:
    """Safe GPU probe — never let backend detection break the GUI."""
    try:
        from sbp_studio.core._backends import gpu_available
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
        self._cli_positioned = False       # spawn at bottom-left only the first time
        self._cli_guide = None             # lazily built Command Guide dialog
        self._help_dialog = None           # lazily built dynamic Help panel

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

        # ── Cruise menu (EXACTLY between CLI and Help) ───────────────────────
        # Campaign-management tools ported from former standalone scripts:
        # file/coordinate Excel generation + acquisition statistics.
        self._menu_cruise = mb.addMenu("")
        self._act_cruise_files = QAction("", self)
        self._act_cruise_files.triggered.connect(self._show_cruise_files)
        self._menu_cruise.addAction(self._act_cruise_files)
        self._act_cruise_stats = QAction("", self)
        self._act_cruise_stats.triggered.connect(self._show_cruise_stats)
        self._menu_cruise.addAction(self._act_cruise_stats)

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

    # ── CLI console (floating, undockable top-level window) ──────────────────

    def _toggle_cli(self, checked: bool) -> None:
        if checked:
            if not self._cli_positioned:
                self._position_cli()
                self._cli_positioned = True
            self._cli.show()
            self._cli.raise_()
            self._cli.activateWindow()
            self._cli.focus_input()
        else:
            self._cli.hide()

    def _position_cli(self) -> None:
        """Default spawn point: bottom-left corner of the main window's layout.

        Only used the FIRST time the console is shown — once the user drags it,
        its position is its own (real top-level window, not re-pinned on resize)."""
        w, h = 420, 300
        margin = 10
        x = margin
        y = max(self.menuBar().height() + margin, self.height() - h - margin)
        self._cli.resize(w, h)
        self._cli.move(self.mapToGlobal(QPoint(x, y)))

    def _show_cli_guide(self) -> None:
        from .components.cli_console import CliGuideDialog
        if self._cli_guide is None:
            self._cli_guide = CliGuideDialog(self)
        self._cli_guide.show()
        self._cli_guide.raise_()
        self._cli_guide.activateWindow()

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
        body.setSizes([210, 1200])
        root.addWidget(body, 1)

        self._build_statusbar()

    # ── Top brand bar ─────────────────────────────────────────────────────────
    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topbar")
        bar.setFixedHeight(54)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 6, 16, 6)

        self._lbl_brand = QLabel("◈  SBP Studio")
        self._lbl_brand.setObjectName("title")
        self._lbl_brand_sub = QLabel()
        self._lbl_brand_sub.setObjectName("subtitle")
        # Vertically centre both so the wordmark and its tagline share a clean
        # baseline band; the tagline sits just right of the title as a subtle
        # secondary label.
        lay.addWidget(self._lbl_brand, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(12)
        lay.addWidget(self._lbl_brand_sub, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addStretch(1)
        return bar

    # ── Left sidebar: profiles + chains ───────────────────────────────────────
    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setMinimumWidth(190)
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
        self.prof_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.prof_list.customContextMenuRequested.connect(self._prof_context_menu)
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

        # Threshold = label + slider + arrow-less numeric box (synced both ways);
        # the slider carries the fine adjustment so the box only needs to show
        # the value (no clunky up/down arrows clipping the digits).
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(6, 0, 6, 0)
        ctrl.setSpacing(4)
        self._lbl_threshold = QLabel()
        self._lbl_threshold.setObjectName("sub")
        ctrl.addWidget(self._lbl_threshold)
        self._THR_SCALE = 10.0            # slider int ↔ km float (×0.1)
        self.sld_threshold = QSlider(Qt.Orientation.Horizontal)
        self.sld_threshold.setRange(1, 500)      # 0.1 .. 50.0 km
        self.sld_threshold.setSingleStep(5)      # 0.5 km
        self.sld_threshold.setPageStep(25)
        self.sld_threshold.setValue(20)          # 2.0 km
        ctrl.addWidget(self.sld_threshold, 1)
        self.chain_gap = QDoubleSpinBox()
        self.chain_gap.setRange(0.1, 50.0)
        self.chain_gap.setSingleStep(0.5)
        self.chain_gap.setDecimals(1)
        self.chain_gap.setValue(2.0)
        self.chain_gap.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.chain_gap.setFixedWidth(48)
        ctrl.addWidget(self.chain_gap)
        cl.addLayout(ctrl)
        self._syncing_threshold = False
        self.sld_threshold.valueChanged.connect(self._on_threshold_slider)
        self.chain_gap.valueChanged.connect(self._on_threshold_spin)

        detect_row = QHBoxLayout()
        detect_row.setContentsMargins(6, 0, 6, 0)
        self._btn_detect = QPushButton()
        self._btn_detect.clicked.connect(self._detect_chains)
        detect_row.addWidget(self._btn_detect, 1)
        cl.addLayout(detect_row)

        self.chain_list = QListWidget()
        self.chain_list.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self.chain_list.currentRowChanged.connect(self._on_chain_row_changed)
        self.chain_list.itemSelectionChanged.connect(self._update_map_buttons)
        self.chain_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chain_list.customContextMenuRequested.connect(self._chain_context_menu)
        cl.addWidget(self.chain_list, 1)

        chain_row = QHBoxLayout()
        chain_row.setContentsMargins(6, 0, 6, 0)
        self._btn_add_chains = QPushButton()
        self._btn_chain_remove = QPushButton()
        self._btn_chain_clear = QPushButton()
        self._btn_add_chains.clicked.connect(self._add_chains_from_dir)
        self._btn_chain_remove.clicked.connect(self._remove_chain)
        self._btn_chain_clear.clicked.connect(self.state.clear_chains)
        chain_row.addWidget(self._btn_add_chains)
        chain_row.addWidget(self._btn_chain_remove)
        chain_row.addStretch(1)
        chain_row.addWidget(self._btn_chain_clear)
        cl.addLayout(chain_row)

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
        # Shares the Visualizer's live DSP filter chain so the Reprojector's
        # "Aplicar filtros" export can run the exact same active node chain
        # the user is editing there — see ReprojectorTab._active_node_cfg.
        self.tab_reprojector = ReprojectorTab(
            self.state, self, pipeline_panel=self.tab_visualizer.pipeline_panel)
        self.tabs.addTab(self.tab_visualizer, "")
        self.tabs.addTab(self.tab_reprojector, "")
        # File Switching (Part 1): clicking an "Add to map" reference track
        # activates that exact profile/chain, exactly as clicking its sidebar
        # row would — see _on_map_layer_source_clicked.
        self.tab_visualizer.map_view.layer_source_clicked.connect(
            self._on_map_layer_source_clicked)
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
            self._clear_other_list_selection(self.chain_list)
            prof = self.state.profiles.get(key)
            if prof is not None and getattr(prof, "data", None) is None and not getattr(prof, "error", None):
                self._load_profile_traces(prof)

    def _load_profile_traces(self, profile) -> None:
        """Background worker: load the trace matrix for a header-only stub."""
        path = profile.path
        self.task_started(self.tr("Loading traces…"))

        def job(progress, cancel):
            from sbp_studio.core import load_profile
            progress(float("nan"), "")
            return load_profile(path, load_traces=True)

        self._run_worker(job, self.state.update_profile_data)

    def _on_chain_row_changed(self, row: int) -> None:
        self.state.set_active_chain(row if row >= 0 else None)
        # Lazily assemble the stitched trace matrix the first time a chain is
        # viewed (mirrors the per-profile lazy load). The tab shows "Loading…"
        # meanwhile and transitions once update_chain_data re-emits.
        chain = self.state.active_chain
        if chain is not None:
            self._clear_other_list_selection(self.prof_list)
            if getattr(chain, "data", None) is None:
                self._load_chain_traces(chain)

    def _clear_other_list_selection(self, widget) -> None:
        """Clear the highlight on the sidebar list that did NOT drive the active
        view, so the unified Visualizer's source is unambiguous. Signals are
        blocked so clearing never cascades into a set_active(None) — the state's
        other-kind active object is intentionally left intact (the tab's mode
        flag, not the list highlight, governs what renders). Map buttons are
        refreshed manually since itemSelectionChanged is suppressed."""
        widget.blockSignals(True)
        widget.clearSelection()
        widget.setCurrentRow(-1)
        widget.blockSignals(False)
        self._update_map_buttons()

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
            self.tr("SEG-Y (*.sgy *.segy *.seg);;All files (*)"))
        if not paths:
            return
        self.task_started(self.tr("Reading headers…"))

        def job(progress, cancel) -> list:
            # Lazy-load: read only SEG-Y headers (no trace data).
            # Traces are loaded on first selection or on demand before rendering.
            from sbp_studio.core import load_profile
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
        # CRS prompting moved to SubTabbedTab._prompt_crs_if_needed (_base.py),
        # triggered just-in-time when the user actually switches to the Map
        # sub-tab — NOT here at load time, so loading a file for pure signal-
        # processing work is never interrupted by a dialog.
        ok = errors = purged = 0
        for prof in profiles:
            if getattr(prof, "error", None):
                errors += 1
            else:
                ok += 1
            purged += int(getattr(prof, "n_purged", 0) or 0)
            self.state.add_profile(prof)
        # Select the first newly loaded valid profile if none is active.
        if self.state.active_profile is None and self.prof_list.count():
            self.prof_list.setCurrentRow(0)
        msg = self.tr("Loaded {0} profile(s), {1} error(s).").format(ok, errors)
        # Surface duplicate-timestamp cleanup as a transient warning toast.
        if purged > 0:
            msg += "  " + self.tr(
                "Warning: purged {0} duplicate trace(s) based on timestamps."
            ).format(purged)
        self.status_lbl.setText(msg)

    # ── Threshold slider ↔ spin sync (guarded against feedback loops) ─────────

    def _on_threshold_slider(self, val: int) -> None:
        if self._syncing_threshold:
            return
        self._syncing_threshold = True
        self.chain_gap.setValue(val / self._THR_SCALE)
        self._syncing_threshold = False

    def _on_threshold_spin(self, val: float) -> None:
        if self._syncing_threshold:
            return
        self._syncing_threshold = True
        self.sld_threshold.setValue(int(round(val * self._THR_SCALE)))
        self._syncing_threshold = False

    def _detect_chains(self) -> None:
        profiles = list(self.state.profiles.values())
        if not profiles:
            self.status_lbl.setText(self.tr("Add profiles first."))
            return
        gap = float(self.chain_gap.value())
        self.task_started(self.tr("Detecting chains…"))

        def job(progress, cancel) -> list:
            from sbp_studio.core import detect_chains
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

    def _add_chains_from_dir(self) -> None:
        """Import pre-organised chains: each subdirectory of the chosen campaign
        folder that holds SEG-Y files becomes a chain named after the folder. The
        scan runs in a worker (header-only stubs → RAM-flat, no trace load)."""
        root = QFileDialog.getExistingDirectory(
            self, self.tr("Select campaign directory"))
        if not root:
            return
        # Pass the names already present so the core skips duplicates without
        # even reading their headers.
        existing = [getattr(c, "label", None) for c in self.state.chains]
        self.task_started(self.tr("Importing chains…"))

        def job(progress, cancel) -> list:
            from sbp_studio.core import import_chains_from_directory
            return import_chains_from_directory(
                root, progress=progress, cancel=cancel, existing_names=existing)

        self._run_worker(job, self._on_chains_imported)

    def _on_chains_imported(self, chains: list) -> None:
        added = self.state.add_chains(chains)
        self._lbl_chain_hint.setText(
            self.tr("{0} chain(s) imported.").format(added))

    def _remove_profile(self) -> None:
        row = self.prof_list.currentRow()
        keys = list(self.state.profiles.keys())
        if 0 <= row < len(keys):
            self.state.remove_profile(keys[row])

    def _remove_chain(self) -> None:
        row = self.chain_list.currentRow()
        if 0 <= row < len(self.state.chains):
            self.state.remove_chain(row)

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

    # ── Batch export (right-click → 'Export selected in batch…') ──────────────
    def _prof_context_menu(self, pos) -> None:
        self._list_context_menu(self.prof_list, self._selected_profiles(),
                                False, pos)

    def _chain_context_menu(self, pos) -> None:
        self._list_context_menu(self.chain_list, self._selected_chains(),
                                True, pos)

    def _list_context_menu(self, widget, items: list, is_chain: bool, pos) -> None:
        """Show the batch-export action for the current multi-selection, plus
        'View Properties / Metadata…' when EXACTLY one item is selected
        (inspecting metadata is inherently single-target)."""
        if not items:
            return
        menu = QMenu(self)
        act_export = menu.addAction(self.tr("Export selected in batch…"))
        act_props = None
        if len(items) == 1:
            act_props = menu.addAction(self.tr("View Properties / Metadata…"))
        chosen = menu.exec(widget.mapToGlobal(pos))
        if chosen is act_export:
            self._batch_export(items, is_chain)
        elif chosen is act_props:
            self._show_metadata_inspector(items[0])

    def _show_metadata_inspector(self, obj) -> None:
        from .components import MetadataInspectorDialog
        dlg = MetadataInspectorDialog(self, obj=obj, state=self.state)
        dlg.exec()

    def _batch_export(self, items: list, is_chain: bool) -> None:
        """Open the export options ONCE, then hand the whole selection to the
        unified Visualizer tab's worker-backed batch exporter (inherits its DSP
        settings). ``is_chain`` selects the render/load path for the batch items
        explicitly — decoupled from the live view's current mode."""
        from .components import ExportDialog
        dlg = ExportDialog(self, batch=True)
        try:
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            cfg = dlg.config()
        finally:
            dlg.deleteLater()
        handler = self.tab_visualizer.source_handler(is_chain=is_chain)
        self.tab_visualizer.export_batch(items, cfg, handler)

    def _add_profiles_to_map(self) -> None:
        self._tracks_to_map(self._selected_profiles(), self.tab_visualizer, is_chain=False)

    def _add_chains_to_map(self) -> None:
        self._tracks_to_map(self._selected_chains(), self.tab_visualizer, is_chain=True)

    def _tracks_to_map(self, objs: list, tab, *, is_chain: bool) -> None:
        """Extract the navigation tracks of *objs* (profiles or chains) and add
        them to *tab*'s map as managed layers — in ONE background pass.

        Memory-safe: it reads ONLY the spatial vectors already resident on the
        header stubs (track_lons/track_lats); it never loads a trace matrix and
        never touches the LRU hot set. UX-safe: the active profile and the live
        seismic section are left completely undisturbed.

        Each object becomes its OWN layer (one add_track_layer call per file/
        chain) — gaps between unrelated files are never artificially stitched
        into one polyline; only a chain's own (deliberately continuous) track
        is drawn as a single line, since that's what loading it as a chain
        means. ``source_id`` (profile path, or "chain:<label>") is carried
        through so File Switching (clicking the layer on the map) can later
        identify exactly which sidebar row to activate.
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
            source_id = f"chain:{name}" if is_chain else getattr(obj, "path", None)
            specs.append((name, lons, lats, getattr(obj, "coord_unit", 0),
                         getattr(obj, "detected_crs", None), source_id))
        if not specs:
            self.status_lbl.setText(self.tr("No navigation tracks to add."))
            return
        self.task_started(self.tr("Extracting navigation tracks…"))

        def job(progress, cancel) -> list:
            from sbp_studio.core import safe_map_coords
            out = []
            n = len(specs)
            for i, (name, lons, lats, coord_unit, crs, source_id) in enumerate(specs):
                cancel.check()
                progress(i / n, "")
                # Guaranteed-safe WGS84-or-NaN — never raw projected metres.
                x, y = safe_map_coords(lons, lats, coord_unit, crs)
                out.append((name, x, y, source_id))
            progress(1.0, "")
            return out

        self._run_worker(job, lambda tracks: self._on_tracks_extracted(tracks, tab))

    def _on_tracks_extracted(self, tracks: list, tab) -> None:
        tab.reveal_map()
        for name, x, y, source_id in tracks:
            tab.map_view.add_track_layer(name, x, y, source_id=source_id)
        self.tabs.setCurrentWidget(tab)
        self.status_lbl.setText(
            self.tr("Added {0} track(s) to the map.").format(len(tracks)))

    def _on_map_layer_source_clicked(self, source_id: str, idx: int) -> None:
        """File Switching: activate the exact profile/chain whose "Add to
        map" reference track was clicked — explicitly, not by merely nudging
        the sidebar's current row and hoping currentRowChanged cascades —
        then center the seismic view on the clicked trace once that
        profile/chain's data has actually finished loading.

        Bug fix: a bare ``setCurrentRow()`` does NOT clear an existing
        multi-selection in ExtendedSelection mode (confirmed empirically —
        Qt only changes the current index, leaving every previously
        selected row still selected). Multi-selection is how the user picks
        several tracks for a batch "Add to map" (see prof_list/chain_list's
        ExtendedSelection mode), so after any such batch action the sidebar
        is left with >1 row selected. _on_profile_row_changed's very first
        line is "if len(selectedItems()) > 1: return" — a guard meant to
        stop a multi-selection from disturbing the active profile — which
        silently swallowed a setCurrentRow()-only activation. That's why
        clicking a map track only "worked" when the target profile already
        happened to be active: nothing-to-switch-to read as "it worked".

        Fix: explicitly clear the stale selection AND call the row-changed
        handler directly (not just rely on the signal) — this reuses the
        EXACT same activation/lazy-load routine the sidebar itself uses
        (set_active_profile/set_active_chain + _load_profile_traces/
        _load_chain_traces for a not-yet-loaded stub), just invoked
        deterministically instead of through a state-dependent signal."""
        if source_id.startswith("chain:"):
            label = source_id[len("chain:"):]
            for i, chain in enumerate(self.state.chains):
                if getattr(chain, "label", None) == label:
                    self._activate_chain_row(i)
                    self._jump_to_trace_when_ready(source_id, idx)
                    return
        else:
            keys = list(self.state.profiles.keys())
            if source_id in keys:
                self._activate_profile_row(keys.index(source_id))
                self._jump_to_trace_when_ready(source_id, idx)

    def _clear_pending_jump(self) -> None:
        """Disconnect any still-armed deferred file-switch jump closure
        (Bug #8). A stale closure can linger when a profile/chain that was
        activated-then-superseded finishes loading WITHOUT becoming active
        again (so its active_*_changed never fires to self-disconnect).
        Tracking the single pending (signal, slot) and tearing it down here
        guarantees at most one is ever armed, and that rapid switching can't
        leave a lingering callback that recentres on a later, unrelated
        activation."""
        pending = getattr(self, "_pending_jump", None)
        if pending is not None:
            signal, slot = pending
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass        # already disconnected / receiver gone — fine
            self._pending_jump = None

    def _jump_to_trace_when_ready(self, source_id: str, idx: int) -> None:
        """Center the seismic view on trace ``idx`` once the profile/chain
        just activated by _on_map_layer_source_clicked actually has its
        trace data loaded — immediately if it was already hot, or deferred
        (via a one-shot signal connection) if a lazy load was just kicked
        off by _activate_profile_row/_activate_chain_row.

        Bug #8: a second click before the first finishes loading replaces the
        pending jump — _clear_pending_jump() below disconnects the prior
        closure first, so only ONE deferred jump is ever armed and no stale
        callback survives to fire on an unrelated later activation."""
        self._clear_pending_jump()
        obj = self.state.active_profile if not source_id.startswith("chain:") \
            else self.state.active_chain
        if obj is not None and getattr(obj, "data", None) is not None:
            self.tab_visualizer._center_on_trace(idx)
            return

        if source_id.startswith("chain:"):
            label = source_id[len("chain:"):]
            signal = self.state.active_chain_changed

            def _on_chain_ready(loaded, _idx=idx, _label=label) -> None:
                if loaded is not None and getattr(loaded, "data", None) is not None \
                        and getattr(loaded, "label", None) == _label:
                    self.tab_visualizer._center_on_trace(_idx)
                    self._clear_pending_jump()

            signal.connect(_on_chain_ready)
            self._pending_jump = (signal, _on_chain_ready)
        else:
            signal = self.state.active_profile_changed

            def _on_profile_ready(loaded, _idx=idx, _sid=source_id) -> None:
                if loaded is not None and getattr(loaded, "data", None) is not None \
                        and getattr(loaded, "path", None) == _sid:
                    self.tab_visualizer._center_on_trace(_idx)
                    self._clear_pending_jump()

            signal.connect(_on_profile_ready)
            self._pending_jump = (signal, _on_profile_ready)

    def _activate_profile_row(self, row: int) -> None:
        """Force-activate sidebar profile row ``row``, regardless of any
        stale multi-selection — see _on_map_layer_source_clicked."""
        self.prof_list.blockSignals(True)
        self.prof_list.clearSelection()
        self.prof_list.setCurrentRow(row)
        if 0 <= row < self.prof_list.count():
            self.prof_list.item(row).setSelected(True)
        self.prof_list.blockSignals(False)
        self._on_profile_row_changed(row)

    def _activate_chain_row(self, row: int) -> None:
        """Force-activate sidebar chain row ``row``, regardless of any
        stale multi-selection — see _on_map_layer_source_clicked."""
        self.chain_list.blockSignals(True)
        self.chain_list.clearSelection()
        self.chain_list.setCurrentRow(row)
        if 0 <= row < self.chain_list.count():
            self.chain_list.item(row).setSelected(True)
        self.chain_list.blockSignals(False)
        self._on_chain_row_changed(row)

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
        worker.finished.connect(worker.deleteLater)
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
    def _ensure_help_dialog(self) -> "object":
        """Lazily build the shared dynamic documentation panel."""
        if getattr(self, "_help_dialog", None) is None:
            from .components.help_dialog import HelpDialog
            self._help_dialog = HelpDialog(self)
        return self._help_dialog

    def _show_module_help(self) -> None:
        """Open the docs panel at the section for the CURRENT tab."""
        dlg = self._ensure_help_dialog()
        dlg.show_for_tab(self.tabs.currentIndex())
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_docs(self) -> None:
        """Open the full dynamic documentation panel (Overview)."""
        dlg = self._ensure_help_dialog()
        dlg.show_section("overview")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _show_about(self) -> None:
        QMessageBox.about(
            self, self.tr("About SBP Studio"),
            self.tr("SBP Studio — processing and visualisation of SEG-Y seismic "
                    "data (SBP sonars).\nPyQt6 interface, decoupled from the "
                    "processing core."))

    # ── Cruise (campaign) tools ──────────────────────────────────────────────
    def _show_cruise_files(self) -> None:
        from .components.cruise_dialogs import FilesCoordinatesDialog
        FilesCoordinatesDialog(self).exec()

    def _show_cruise_stats(self) -> None:
        from .components.cruise_dialogs import AcquisitionStatsDialog
        AcquisitionStatsDialog(self).exec()

    # ════════════════════════════════════════════════════════════════════════
    # i18n
    # ════════════════════════════════════════════════════════════════════════
    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("SBP Studio"))

        # Menus
        self._menu_lang.setTitle(self.tr("Language"))
        self._menu_theme.setTitle(self.tr("Theme"))
        self._menu_cli.setTitle(self.tr("CLI"))
        self._act_cli.setText(self.tr("Activate Console"))
        self._act_cli_guide.setText(self.tr("Command Guide"))
        self._menu_cruise.setTitle(self.tr("Cruise"))
        self._act_cruise_files.setText(self.tr("Files & Coordinates"))
        self._act_cruise_stats.setText(self.tr("Acquisition Stats"))
        self._menu_help.setTitle(self.tr("Help"))
        self._btn_cancel.setToolTip(self.tr("Cancel the current task"))
        self._theme_actions["dark"].setText(self.tr("Dark"))
        self._theme_actions["light"].setText(self.tr("Light"))
        self._act_help_module.setText(self.tr("How this module works"))
        self._act_help_docs.setText(self.tr("Documentation"))
        self._act_help_about.setText(self.tr("About SBP Studio"))
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
        self._btn_add_chains.setText(self.tr("📂 Add"))
        self._btn_add_chains.setToolTip(
            self.tr("Import chains from a campaign directory (one folder per chain)"))
        self._btn_chain_remove.setText(self.tr("✖ Remove"))
        self._btn_chain_clear.setText(self.tr("✖✖ Clear"))
        if not self.state.chains:
            self._lbl_chain_hint.setText(self.tr('Load profiles and click "Detect".'))

        # Tabs
        self.tabs.setTabText(0, self.tr("  ▣  Visualizer  "))
        self.tabs.setTabText(1, self.tr("  ⇄  Reprojector  "))

        # Status bar
        if self._active_tasks == 0:
            self.status_lbl.setText(self.tr("Ready — add one or more SEG-Y profiles."))
        self._gpu_badge.setText(self.tr("⚡ GPU") if self._gpu else self.tr("○ CPU"))
