"""
reprojector_tab.py — Tab B · Reprojector.

CRS management (EPSG/WKT presets + validation) and two background actions:

  * Reproject SEG-Y files to a new CRS → ``core.reproject_one`` /
    ``core.reproject_chain`` (writes ``*_REPROY`` files beside the source).
  * Export the navigation line (track) as SHP / GeoJSON / CSV → the core
    ``write_navline_*`` exporters, optionally reprojected via
    ``core.reproject_points``.

All projection/IO math lives in the CORE; this tab only collects parameters and
dispatches them onto a :class:`CoreWorker` via ``tasks.run_task``. Restores the
legacy ``TopasSUITE_qt.py`` reprojector on the new decoupled architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QTextEdit, QVBoxLayout, QWidget,
)

from ...core import CRS_CATALOG, CRS_PRESETS, resolve_crs, validate_crs
from ..i18n import language_manager
from ..state import AppState

# Selection modes.
_ALL, _ACTIVE, _CHAINS = range(3)
_FMTS = ("shp", "geojson", "csv")


class ReprojectorTab(QWidget):
    """Tab B — CRS reprojection of profiles/chains + navline export."""

    def __init__(self, state: AppState, tasks, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self.tasks = tasks                    # MainWindow task service (run_task/notify/show_error)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        self.root = QVBoxLayout(host)
        self.root.setContentsMargins(12, 10, 12, 10)
        self.root.setSpacing(6)
        scroll.setWidget(host)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._build_selection()
        self._build_crs_blocks()
        self._build_reproject_row()
        self._build_navline()
        self._build_log()
        self.root.addStretch(1)

        # React to data changes.
        self.state.profiles_changed.connect(self._on_data_changed)
        self.state.chains_changed.connect(self._on_data_changed)
        self.state.active_profile_changed.connect(lambda *_: self._on_data_changed())

        language_manager.language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()
        self._refresh_nav_sources()
        self._update_sel_label()

    # ── Construction ──────────────────────────────────────────────────────────

    def _section(self, attr: str) -> QLabel:
        lbl = QLabel()
        lbl.setObjectName("section")
        setattr(self, attr, lbl)
        self.root.addWidget(lbl)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("sub")
        self.root.addWidget(line)
        return lbl

    def _build_selection(self) -> None:
        self._section("_sec_sel")
        panel = QFrame(); panel.setObjectName("panel")
        v = QVBoxLayout(panel); v.setContentsMargins(8, 6, 8, 6)
        self.sel_group = QButtonGroup(self)
        self.rb_all = QRadioButton()
        self.rb_active = QRadioButton()
        self.rb_chains = QRadioButton()
        self.rb_all.setChecked(True)
        for i, rb in enumerate((self.rb_all, self.rb_active, self.rb_chains)):
            self.sel_group.addButton(rb, i)
            v.addWidget(rb)
        self.sel_group.idToggled.connect(
            lambda _id, on: self._update_sel_label() if on else None)
        self.lbl_sel = QLabel(); self.lbl_sel.setObjectName("sub")
        v.addWidget(self.lbl_sel)
        self.root.addWidget(panel)

    def _crs_block(self, which: str) -> QFrame:
        frame = QFrame(); frame.setObjectName("panel")
        v = QVBoxLayout(frame); v.setContentsMargins(8, 6, 8, 6); v.setSpacing(3)
        title = QLabel(); title.setObjectName("section")
        v.addWidget(title)
        lbl_preset = QLabel()
        v.addWidget(lbl_preset)
        preset = QComboBox()                  # filled from CRS_CATALOG (grouped)
        v.addWidget(preset)
        lbl_code = QLabel()
        v.addWidget(lbl_code)
        row = QHBoxLayout()
        entry = QLineEdit()
        row.addWidget(entry, 1)
        verify = QPushButton()
        verify.clicked.connect(self._verify_src if which == "src" else self._verify_dst)
        row.addWidget(verify)
        v.addLayout(row)
        status = QLabel(); status.setObjectName("sub")
        v.addWidget(status)
        if which == "src":
            (self.src_title, self.src_lbl_preset, self.src_preset, self.src_lbl_code,
             self.src_entry, self.src_verify, self.src_status) = (
                title, lbl_preset, preset, lbl_code, entry, verify, status)
        else:
            (self.dst_title, self.dst_lbl_preset, self.dst_preset, self.dst_lbl_code,
             self.dst_entry, self.dst_verify, self.dst_status) = (
                title, lbl_preset, preset, lbl_code, entry, verify, status)
        return frame

    def _build_crs_blocks(self) -> None:
        row = QHBoxLayout()
        row.addWidget(self._crs_block("src"))
        row.addWidget(self._crs_block("dst"))
        self.root.addLayout(row)
        self._fill_crs_combo(self.src_preset)
        self._fill_crs_combo(self.dst_preset)
        # Defaults via the stored EPSG code (BEFORE wiring signals).
        self._select_crs_code(self.src_preset, "EPSG:4326")
        self._select_crs_code(self.dst_preset, "EPSG:32630")
        self.src_entry.setText("EPSG:4326")
        self.dst_entry.setText("EPSG:32630")
        self.src_preset.currentIndexChanged.connect(lambda *_: self._on_preset_changed("src"))
        self.dst_preset.currentIndexChanged.connect(lambda *_: self._on_preset_changed("dst"))
        self._verify_src(); self._verify_dst()

    # ── Grouped CRS combos (QStandardItemModel with category separators) ───────

    def _tr_category(self, key: str) -> str:
        """Translate a CRS_CATALOG category key (literals → pylupdate6-extractable)."""
        return {
            "Geographic":           self.tr("Geographic"),
            "Polar Stereographic":  self.tr("Polar Stereographic"),
            "UTM North (WGS 84)":   self.tr("UTM North (WGS 84)"),
            "UTM South (WGS 84)":   self.tr("UTM South (WGS 84)"),
        }.get(key, key)

    def _fill_crs_combo(self, combo: QComboBox, prefix=()) -> None:
        """Populate a CRS combo from CRS_CATALOG with BOLD, non-selectable
        category separator rows. ``prefix`` = optional leading selectable
        (label, data) specials. Each real item stores its EPSG code in UserRole."""
        model = QStandardItemModel(combo)
        for label, data in prefix:
            it = QStandardItem(label)
            it.setData(data, Qt.ItemDataRole.UserRole)
            model.appendRow(it)
        for cat_key, items in CRS_CATALOG:
            sep = QStandardItem("— " + self._tr_category(cat_key) + " —")
            sep.setFlags(Qt.ItemFlag.NoItemFlags)        # non-selectable / disabled
            font = sep.font(); font.setBold(True); sep.setFont(font)
            model.appendRow(sep)
            for label, code in items:
                it = QStandardItem(label)
                it.setData(code, Qt.ItemDataRole.UserRole)
                model.appendRow(it)
        combo.setModel(model)

    def _select_crs_code(self, combo: QComboBox, code: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i, Qt.ItemDataRole.UserRole) == code:
                combo.setCurrentIndex(i)
                return

    def _on_preset_changed(self, which: str) -> None:
        combo = self.src_preset if which == "src" else self.dst_preset
        entry = self.src_entry if which == "src" else self.dst_entry
        code = combo.currentData(Qt.ItemDataRole.UserRole)
        if code:
            entry.setText(code)
            (self._verify_src if which == "src" else self._verify_dst)()

    def _build_reproject_row(self) -> None:
        row = QHBoxLayout()
        self.lbl_unit = QLabel()
        row.addWidget(self.lbl_unit)
        self.unit_cb = QComboBox()
        self.unit_cb.addItems(["1", "2", "3"])     # filled in retranslate
        self.unit_cb.setCurrentIndex(1)
        row.addWidget(self.unit_cb)
        row.addStretch(1)
        self.btn_reproject = QPushButton()
        self.btn_reproject.clicked.connect(self._run_reprojection)
        row.addWidget(self.btn_reproject)
        self.progress = QProgressBar(); self.progress.setFixedWidth(200)
        self.progress.setRange(0, 1); self.progress.setValue(0)
        row.addWidget(self.progress)
        self.root.addLayout(row)
        self.lbl_note = QLabel(); self.lbl_note.setObjectName("sub")
        self.lbl_note.setWordWrap(True)
        self.root.addWidget(self.lbl_note)

    def _build_navline(self) -> None:
        self._section("_sec_nav")
        panel = QFrame(); panel.setObjectName("panel")
        v = QVBoxLayout(panel); v.setContentsMargins(8, 6, 8, 6); v.setSpacing(4)
        r1 = QHBoxLayout()
        self.lbl_element = QLabel(); r1.addWidget(self.lbl_element)
        self.nav_source = QComboBox(); self.nav_source.setMinimumWidth(280)
        r1.addWidget(self.nav_source)
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.clicked.connect(self._refresh_nav_sources)
        r1.addWidget(self.btn_refresh)
        r1.addSpacing(12)
        self.lbl_outcrs = QLabel(); r1.addWidget(self.lbl_outcrs)
        self.nav_crs = QComboBox()
        self._fill_crs_combo(self.nav_crs, prefix=[
            (self.tr("(same as source CRS)"), "__SRC__"),
            (self.tr("(same as target CRS)"), "__DST__"),
        ])
        r1.addWidget(self.nav_crs)
        r1.addStretch(1)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        self.fmt_group = QButtonGroup(self)
        self.fmt_rbs: List[QRadioButton] = []
        for i in range(3):
            rb = QRadioButton()
            if i == 0:
                rb.setChecked(True)
            self.fmt_group.addButton(rb, i)
            self.fmt_rbs.append(rb)
            r2.addWidget(rb)
        self.nav_attrs = QCheckBox(); self.nav_attrs.setChecked(True)
        r2.addSpacing(12); r2.addWidget(self.nav_attrs)
        r2.addStretch(1)
        self.btn_navline = QPushButton()
        self.btn_navline.clicked.connect(self._export_navline)
        r2.addWidget(self.btn_navline)
        v.addLayout(r2)
        self.root.addWidget(panel)

    def _build_log(self) -> None:
        self._section("_sec_log")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.root.addWidget(self.log)

    # ── CRS presets / verify ──────────────────────────────────────────────────

    def _verify_src(self) -> None:
        self._verify(self.src_entry.text(), self.src_status)

    def _verify_dst(self) -> None:
        self._verify(self.dst_entry.text(), self.dst_status)

    def _verify(self, code: str, status: QLabel) -> None:
        try:
            crs = validate_crs(resolve_crs(code))
            status.setText("✔  " + crs.name[:55])
        except Exception as exc:
            status.setText("✘  " + str(exc)[:60])

    # ── Reprojection ──────────────────────────────────────────────────────────

    def _targets(self):
        mode = self.sel_group.checkedId()
        if mode == _ALL:
            return _ALL, [p for p in self.state.profiles.values() if not getattr(p, "error", None)]
        if mode == _ACTIVE:
            ap = self.state.active_profile
            return _ACTIVE, ([ap] if ap is not None and not getattr(ap, "error", None) else [])
        return _CHAINS, list(self.state.chains)

    def _run_reprojection(self) -> None:
        mode, targets = self._targets()
        if not targets:
            self.tasks.notify(self.tr("No items selected to reproject."))
            return
        try:
            src = resolve_crs(self.src_entry.text()); validate_crs(src)
        except Exception as exc:
            self.tasks.show_error(self.tr("Invalid source CRS"), str(exc)); return
        try:
            dst = resolve_crs(self.dst_entry.text()); validate_crs(dst)
        except Exception as exc:
            self.tasks.show_error(self.tr("Invalid target CRS"), str(exc)); return
        unit_hint = int(self.unit_cb.currentText()[0])
        is_chain = (mode == _CHAINS)

        # ── Interactive destination (no hardcoded suffix in the source folder) ──
        if len(targets) == 1 and not is_chain:
            src_p = Path(targets[0].path)
            default = str(src_p.with_name(src_p.stem + "_REPROY.sgy"))
            out, _ = QFileDialog.getSaveFileName(
                self, self.tr("Save reprojected SEG-Y"), default,
                self.tr("SEG-Y (*.sgy *.segy *.seg)"))
            if not out:
                return
            if not Path(out).suffix:
                out += ".sgy"
            out_paths = [out]
        else:
            folder = QFileDialog.getExistingDirectory(
                self, self.tr("Select output folder for reprojected files"))
            if not folder:
                return
            out_paths = [str(Path(folder) / f"{self._target_stem(t, is_chain)}_REPROY.sgy")
                         for t in targets]

        self.progress.setRange(0, 0)          # busy/indeterminate
        pairs = list(zip(targets, out_paths))

        def job(progress, cancel):
            from topassuite.core import reproject_one, reproject_chain
            outs, n = [], len(pairs)
            for i, (item, op) in enumerate(pairs, 1):
                cancel.check()
                progress(i / n, "")
                if is_chain:
                    outs.append(reproject_chain(item, src, dst, unit_hint,
                                                cancel=cancel, out_path=op))
                else:
                    outs.append(reproject_one(item, src, dst, unit_hint,
                                              cancel=cancel, out_path=op))
            return outs

        self.tasks.run_task(job, self._on_reprojected,
                            self.tr("Reprojecting {0} item(s)…").format(len(targets)))

    def _target_stem(self, item, is_chain: bool) -> str:
        """Output filename stem for a reprojection target (used in batch/folder
        mode): profile stem, or the joined-chain composite stem."""
        if is_chain:
            profs = getattr(item, "profiles", [])
            if profs:
                return f"{profs[0].stem}_a_{profs[-1].stem}_UNIDO"
            return getattr(item, "name", "cadena")
        return Path(item.path).stem

    def _on_reprojected(self, outs: list) -> None:
        self.progress.setRange(0, 1); self.progress.setValue(1)
        for out in outs:
            self.log.append("✔  " + Path(out).name)
        self.tasks.notify(self.tr("Reprojection complete: {0} file(s).").format(len(outs)))

    # ── Navline export ────────────────────────────────────────────────────────

    def _nav_items(self):
        """Return [(label, source_object)] for profiles + chains."""
        items = []
        for p in self.state.profiles.values():
            if not getattr(p, "error", None):
                items.append((f"{self.tr('Profile')}: {p.name}", p))
        for ch in self.state.chains:
            tag = self.tr("Chain") if len(getattr(ch, "profiles", [])) > 1 else self.tr("Profile")
            items.append((f"{tag}: {getattr(ch, 'label', ch.name)}", ch))
        return items

    def _refresh_nav_sources(self) -> None:
        self._nav_map = self._nav_items()
        self.nav_source.blockSignals(True)
        self.nav_source.clear()
        self.nav_source.addItems([lbl for lbl, _ in self._nav_map] or [self.tr("(no data loaded)")])
        self.nav_source.blockSignals(False)

    def _export_navline(self) -> None:
        idx = self.nav_source.currentIndex()
        nav_map = getattr(self, "_nav_map", [])
        if not nav_map or not (0 <= idx < len(nav_map)):
            self.tasks.notify(self.tr("Select a profile or chain.")); return
        source = nav_map[idx][1]

        # Output CRS from the selected item's stored data: __SRC__ (keep source),
        # __DST__ (target CRS), or a preset EPSG code.
        data = self.nav_crs.currentData(Qt.ItemDataRole.UserRole)
        out_crs: Optional[str] = None
        if data == "__DST__":
            out_crs = resolve_crs(self.dst_entry.text())
        elif data and data != "__SRC__":
            out_crs = data
        src_crs = resolve_crs(self.src_entry.text())

        fmt = _FMTS[self.fmt_group.checkedId()]
        stem = getattr(source, "stem", None) or getattr(source, "name", "navline")
        out, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export navline"), f"{stem}_NAVLINE.{fmt}",
            f"{fmt.upper()} (*.{fmt})")
        if not out:
            return
        include_attrs = self.nav_attrs.isChecked()
        lons = list(source.lons); lats = list(source.lats)
        dist = list(source.dist_km); wd = list(source.water_depth)
        ts = list(source.timestamps)

        def job(progress, cancel):
            import numpy as np
            from topassuite.core import (
                reproject_points, write_navline_shp, write_navline_geojson,
                write_navline_csv,
            )
            progress(float("nan"), "")
            lo, la = np.asarray(lons, float), np.asarray(lats, float)
            if out_crs and resolve_crs(out_crs) != src_crs:
                lo, la = reproject_points(lo, la, src_crs, out_crs)
            if fmt == "shp":
                write_navline_shp(out, lo, la, dist, wd, ts, include_attrs, out_crs)
            elif fmt == "geojson":
                write_navline_geojson(out, lo, la, dist, wd, ts, include_attrs, out_crs)
            else:
                write_navline_csv(out, lo, la, dist, wd, ts)
            return out

        self.tasks.run_task(job, self._on_navline_exported,
                            self.tr("Exporting navline…"))

    def _on_navline_exported(self, out: str) -> None:
        self.log.append("💾  " + Path(out).name)
        self.tasks.notify(self.tr("Navline saved: {0}").format(Path(out).name))

    # ── State reactions ───────────────────────────────────────────────────────

    def _on_data_changed(self) -> None:
        self._update_sel_label()
        self._refresh_nav_sources()

    def _update_sel_label(self) -> None:
        mode = self.sel_group.checkedId()
        if mode == _ALL:
            self.lbl_sel.setText(self.tr("→ {0} profile(s) queued").format(len(self.state.profiles)))
        elif mode == _ACTIVE:
            ap = self.state.active_profile
            self.lbl_sel.setText("→ " + (ap.name if ap is not None else self.tr("(none)")))
        else:
            self.lbl_sel.setText(self.tr("→ {0} chain(s) queued").format(len(self.state.chains)))

    # ── i18n ──────────────────────────────────────────────────────────────────

    def retranslate_ui(self) -> None:
        self._sec_sel.setText(self.tr("Profiles to reproject"))
        self.rb_all.setText(self.tr("All loaded profiles (individually)"))
        self.rb_active.setText(self.tr("Only the active profile"))
        self.rb_chains.setText(self.tr("Detected chains (join + reproject)"))
        self.src_title.setText(self.tr("Source CRS"))
        self.dst_title.setText(self.tr("Target CRS"))
        for lbl in (self.src_lbl_preset, self.dst_lbl_preset):
            lbl.setText(self.tr("Preset:"))
        for lbl in (self.src_lbl_code, self.dst_lbl_code):
            lbl.setText(self.tr("EPSG or WKT:"))
        for b in (self.src_verify, self.dst_verify):
            b.setText(self.tr("Verify"))
        self.lbl_unit.setText(self.tr("Unit interpretation if undetected:"))
        units = (self.tr("1 – Metres/feet"), self.tr("2 – Arc-seconds (TOPAS)"),
                 self.tr("3 – Decimal degrees"))
        for i, u in enumerate(units):
            self.unit_cb.setItemText(i, u)
        self.btn_reproject.setText(self.tr("Reproject"))
        self.lbl_note.setText(self.tr("Reprojected files are written beside the "
                                      "source with a _REPROY suffix."))
        self._sec_nav.setText(self.tr("Export navigation line"))
        self.lbl_element.setText(self.tr("Element:"))
        self.lbl_outcrs.setText(self.tr("Output CRS:"))
        for rb, lbl in zip(self.fmt_rbs, (self.tr("Shapefile (.shp)"),
                                          self.tr("GeoJSON (.geojson)"),
                                          self.tr("CSV (.csv)"))):
            rb.setText(lbl)
        self.nav_attrs.setText(self.tr("Include attributes (dist_km, wd, timestamp)"))
        self.btn_navline.setText(self.tr("Export navline"))
        self._sec_log.setText(self.tr("Log"))
        self._update_sel_label()
