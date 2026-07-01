"""
cruise_dialogs.py — native PyQt6 dialogs for the "Cruise" (Campaña) menu.

Two campaign-management tools, ported from former standalone tkinter scripts to
match the TopasSuite aesthetic. All heavy lifting lives GUI-free in
``sbp_studio.core.campaign``; these dialogs are thin front-ends:

  * :class:`FilesCoordinatesDialog` — generate the file-registry Excel and the
    line-coordinates/length Excel for one or more project phases.
  * :class:`AcquisitionStatsDialog` — compute ping rate, ping interval and
    vessel speed from the SEG-Y time/navigation headers.

Both open with a prominent banner describing the required directory structure
(a base dir containing ``SGY/`` and ``RAW/`` subfolders, each with one folder
per seismic line). Processing runs synchronously behind a wait cursor — the same
one-shot behaviour as the original tools.

Layout notes (post-mortem of the first cut)
--------------------------------------------
The very first version left the phase list squeezed into a small, immediately-
scrolling area while empty space sat below it — two concrete bugs, now fixed:
``AcquisitionStatsDialog`` hard-capped its scroll area at
``setMaximumHeight(220)`` regardless of how tall the dialog itself was, and
neither dialog ever called ``resize()`` (only ``setMinimumSize``), so both
opened pinned near their minimum floor. Fixed here via ``QGroupBox`` sections
(matching the original tkinter ``LabelFrame`` hierarchy), an explicit
``setMinimumHeight`` floor on each phase row (so text fields never collapse),
Expanding size policies with no artificial height cap on the phases scroll
area, and :func:`_grow_for_new_row`, which grows the dialog's own height by one
row's worth of space each time a phase is added — capped to the screen's
available height, past which the (still generously-sized) scroll area takes
over gracefully instead of squishing anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor, QFont
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ..theme import MONO, theme
from ...core import campaign


# ── Shared building blocks ──────────────────────────────────────────────────────

class _DirStructureBanner(QLabel):
    """Prominent, theme-aware info banner describing the expected folder layout.

    Shown at the top of both tools so the user knows what to point them at
    before anything runs."""

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self._restyle()

    def _restyle(self) -> None:
        self.setStyleSheet(
            f"background-color: {theme.color('highlight')};"
            f" color: {theme.color('text')};"
            f" border: 1px solid {theme.color('accent')};"
            " border-radius: 4px; padding: 8px 10px;")


def _group_box(title: str) -> QGroupBox:
    """A themed QGroupBox matching the app's bold/bright section-header look
    (see theme.py's ``QLabel#section`` rule) — the visual-hierarchy anchor the
    original tkinter tool used ``LabelFrame`` for ("Proyecto", "Fases del
    proyecto", "Archivos de salida"). Un-styled QGroupBox would otherwise
    render with the OS's light-mode default frame (the app only sets a QSS
    stylesheet, not a dark QPalette), clashing badly with the dark theme."""
    box = QGroupBox(title)
    box.setStyleSheet(
        "QGroupBox {"
        f"  border: 1px solid {theme.color('accent')};"
        "   border-radius: 6px;"
        "   margin-top: 12px;"
        "   font-weight: bold;"
        f"  color: {theme.color('bright')};"
        "   padding-top: 6px;"
        "}"
        "QGroupBox::title {"
        "   subcontrol-origin: margin;"
        "   left: 10px;"
        "   padding: 0 6px;"
        "}")
    return box


class _PhaseRowBase(QWidget):
    """Common framing for one removable phase row (bordered card + ✕ button)."""

    def __init__(self, remove_cb, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._remove_cb = remove_cb
        self.setStyleSheet(
            f"QWidget#phaseCard {{ border: 1px solid {theme.color('accent')};"
            " border-radius: 4px; }")
        self.setObjectName("phaseCard")


def _browse_dir(parent: QWidget, line_edit: QLineEdit, title: str) -> None:
    d = QFileDialog.getExistingDirectory(parent, title, line_edit.text().strip())
    if d:
        line_edit.setText(d)


def _grow_for_new_row(dialog: QDialog, row_min_height: int, spacing: int) -> None:
    """Grow ``dialog``'s height by one row's worth of space when a phase is
    added, so the scroll area genuinely gets bigger instead of immediately
    scrolling — capped to the screen's available height so the window never
    grows off-screen. Past that cap, the scroll area's own scrollbar (already
    enabled via setWidgetResizable + an Expanding size policy) takes over."""
    screen = dialog.screen() or QApplication.primaryScreen()
    max_h = (screen.availableGeometry().height() - 80) if screen is not None else 900
    new_h = min(dialog.height() + row_min_height + spacing, max_h)
    if new_h > dialog.height():
        dialog.resize(dialog.width(), new_h)


def _phases_scroll_area(phases_box: QVBoxLayout) -> QScrollArea:
    """Build the scrollable container hosting the phase rows. Expanding in
    BOTH directions with no maximum height — the earlier version's hard
    ``setMaximumHeight(220)`` on this exact widget was the concrete bug behind
    "a scrollbar appears immediately even though there is empty space below":
    it capped the phases area at 220px no matter how tall the dialog grew,
    while whatever sat below it (a log console, in one case) soaked up all the
    leftover space instead."""
    container = QWidget()
    container.setLayout(phases_box)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidget(container)
    return scroll


# ════════════════════════════════════════════════════════════════════════════════
# Tool A — Files & Coordinates
# ════════════════════════════════════════════════════════════════════════════════

# Two stacked rows (sheet/label/zone/detect/remove, then directory/browse) at
# normal control height, plus card margins/spacing — a floor so a row can
# never be squeezed below legible size regardless of how little room the
# scroll viewport currently has.
_FILES_ROW_MIN_HEIGHT = 92


class _FilesPhaseRow(_PhaseRowBase):
    """One phase: sheet name, section label, forced UTM zone (+ detect), dir."""

    def __init__(self, index: int, remove_cb, parent=None) -> None:
        super().__init__(remove_cb, parent)
        self.setMinimumHeight(_FILES_ROW_MIN_HEIGHT)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.ed_sheet = QLineEdit(self.tr("{n}ª FASE").format(n=index))
        self.ed_sheet.setMaximumWidth(110)
        self.ed_label = QLineEdit()
        self.ed_zone = QLineEdit()
        self.ed_zone.setMaximumWidth(48)
        self.ed_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_detect = QPushButton("🔍")
        self.btn_detect.setMaximumWidth(34)
        self.btn_detect.setToolTip(self.tr("Detect the UTM zone from the first SGY"))
        self.btn_detect.clicked.connect(self._detect_zone)
        btn_remove = QPushButton("✕")
        btn_remove.setMaximumWidth(30)
        btn_remove.clicked.connect(lambda: self._remove_cb(self))
        top.addWidget(QLabel(self.tr("Sheet:")))
        top.addWidget(self.ed_sheet)
        top.addWidget(QLabel(self.tr("Label:")))
        top.addWidget(self.ed_label, 1)
        top.addWidget(QLabel(self.tr("UTM zone:")))
        top.addWidget(self.ed_zone)
        top.addWidget(self.btn_detect)
        top.addWidget(btn_remove)
        lay.addLayout(top)

        bot = QHBoxLayout()
        bot.setSpacing(6)
        self.ed_dir = QLineEdit()
        self.ed_dir.setPlaceholderText(self.tr("Base directory (contains SGY/ and RAW/)"))
        btn_browse = QPushButton("📁")
        btn_browse.setMaximumWidth(34)
        btn_browse.clicked.connect(self._browse)
        bot.addWidget(QLabel(self.tr("Directory:")))
        bot.addWidget(self.ed_dir, 1)
        bot.addWidget(btn_browse)
        lay.addLayout(bot)

    def _browse(self) -> None:
        _browse_dir(self, self.ed_dir,
                    self.tr("Select base directory (must contain SGY/ and RAW/)"))
        if self.ed_dir.text().strip() and not self.ed_label.text().strip():
            self.ed_label.setText(Path(self.ed_dir.text().strip()).name)

    def _detect_zone(self) -> None:
        base = self.ed_dir.text().strip().strip('"').strip("'")
        if not base or not Path(base).is_dir():
            QMessageBox.warning(self, self.tr("No directory"),
                                self.tr("Select the base directory first."))
            return
        zone, fname, lon = campaign.detect_zone_for_base(base)
        if zone is None and fname is None:
            QMessageBox.information(
                self, self.tr("No SGY files"),
                self.tr("No .sgy files were found in SGY/ or RAW/.\n"
                        "Enter the zone manually."))
            return
        if zone is None:
            QMessageBox.warning(
                self, self.tr("Read error"),
                self.tr("Could not read the coordinate from:\n{name}").format(name=fname))
            return
        self.ed_zone.setText(str(zone))
        QMessageBox.information(
            self, self.tr("UTM zone detected"),
            self.tr("File analysed: {name}\nFirst-trace longitude: {lon:.5f}°\n\n"
                    "Detected UTM zone: {zone} → EPSG:326{zone:02d}\n\n"
                    "You can change it before generating.").format(
                        name=fname, lon=lon, zone=zone))

    def _parsed_zone(self) -> Optional[int]:
        raw = self.ed_zone.text().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def get_data(self) -> dict:
        label = self.ed_label.text().strip() or self.ed_sheet.text().strip()
        return {"sheet_name": self.ed_sheet.text().strip(), "label": label,
                "base_dir": self.ed_dir.text().strip(),
                "forced_zone": self._parsed_zone()}

    def validate(self) -> Optional[str]:
        d = self.get_data()
        if not d["sheet_name"]:
            return self.tr("The sheet name cannot be empty.")
        if not d["base_dir"]:
            return self.tr("Missing directory for phase '{p}'.").format(p=d["sheet_name"])
        if not Path(d["base_dir"]).is_dir():
            return self.tr("Directory not found:\n{p}").format(p=d["base_dir"])
        raw = self.ed_zone.text().strip()
        if raw:
            try:
                z = int(raw)
                if not (1 <= z <= 60):
                    return self.tr("UTM zone '{z}' out of range (1-60).").format(z=raw)
            except ValueError:
                return self.tr("UTM zone '{z}' is not a valid number.").format(z=raw)
        return None


class FilesCoordinatesDialog(QDialog):
    """Generate the file-registry Excel and the coordinates/length Excel."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Cruise — Files & Coordinates"))
        self.setMinimumSize(720, 560)
        self.resize(780, 680)
        self._rows: List[_FilesPhaseRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        banner = _DirStructureBanner(self.tr(
            "<b>Required directory structure.</b> The base directory of each "
            "phase must contain <b>SGY/</b> and <b>RAW/</b> subfolders, each with "
            "one folder per seismic line (e.g. <code>SGY/L1/*.sgy</code>). "
            "Alternatively the base directory may itself hold the per-line "
            "folders. Line start/end coordinates are read from the SEG-Y headers."))
        root.addWidget(banner)

        # ── "Proyecto" ──
        grp_project = _group_box(self.tr("Project"))
        proj_lay = QHBoxLayout(grp_project)
        proj_lay.setContentsMargins(10, 14, 10, 10)
        proj_lay.setSpacing(8)
        proj_lay.addWidget(QLabel(self.tr("Project name:")))
        self.ed_project = QLineEdit()
        proj_lay.addWidget(self.ed_project, 1)
        root.addWidget(grp_project)

        # ── "Fases del proyecto" — phase rows + Add button, exactly like the
        # original LabelFrame (which hosted both together). ──
        grp_phases = _group_box(self.tr("Phases of the project"))
        phases_lay = QVBoxLayout(grp_phases)
        phases_lay.setContentsMargins(10, 14, 10, 10)
        phases_lay.setSpacing(8)

        self._phases_box = QVBoxLayout()
        self._phases_box.setSpacing(8)
        scroll = _phases_scroll_area(self._phases_box)
        phases_lay.addWidget(scroll, 1)

        self.btn_add = QPushButton(self.tr("+ Add phase"))
        self.btn_add.clicked.connect(self._add_phase)
        phases_lay.addWidget(self.btn_add, 0, Qt.AlignmentFlag.AlignLeft)
        root.addWidget(grp_phases, 1)

        # ── "Archivos de salida" ──
        grp_out = _group_box(self.tr("Output files"))
        out_form = QFormLayout(grp_out)
        out_form.setContentsMargins(10, 14, 10, 10)
        out_form.setVerticalSpacing(8)
        out_form.setHorizontalSpacing(8)
        reg_row = QHBoxLayout()
        self.ed_out_reg = QLineEdit()
        btn_reg = QPushButton(self.tr("Browse…"))
        btn_reg.clicked.connect(lambda: self._browse_out(self.ed_out_reg, "Ficheros"))
        reg_row.addWidget(self.ed_out_reg, 1)
        reg_row.addWidget(btn_reg)
        coord_row = QHBoxLayout()
        self.ed_out_coord = QLineEdit()
        btn_coord = QPushButton(self.tr("Browse…"))
        btn_coord.clicked.connect(lambda: self._browse_out(self.ed_out_coord, "coordenadas"))
        coord_row.addWidget(self.ed_out_coord, 1)
        coord_row.addWidget(btn_coord)
        out_form.addRow(self.tr("Registry Excel:"), reg_row)
        out_form.addRow(self.tr("Coordinates Excel:"), coord_row)
        root.addWidget(grp_out)

        self.btn_generate = QPushButton(self.tr("Generate both Excel files"))
        self.btn_generate.setObjectName("primary")
        self.btn_generate.clicked.connect(self._generate)
        root.addWidget(self.btn_generate)

        self.lbl_status = QLabel(self.tr("Ready."))
        self.lbl_status.setObjectName("sub")
        root.addWidget(self.lbl_status)

        self._add_phase()

    def _add_phase(self) -> None:
        row = _FilesPhaseRow(len(self._rows) + 1, self._remove_phase)
        self._rows.append(row)
        self._phases_box.addWidget(row)
        _grow_for_new_row(self, _FILES_ROW_MIN_HEIGHT, self._phases_box.spacing())

    def _remove_phase(self, row: _FilesPhaseRow) -> None:
        if len(self._rows) == 1:
            QMessageBox.warning(self, self.tr("Notice"),
                                self.tr("At least one phase is required."))
            return
        self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _browse_out(self, line_edit: QLineEdit, prefix: str) -> None:
        proj = self.ed_project.text().strip() or "proyecto"
        start = line_edit.text().strip() or f"{prefix}_{proj}.xlsx"
        f, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save output file"), start,
            self.tr("Excel (*.xlsx)"))
        if f:
            line_edit.setText(f)

    def _generate(self) -> None:
        proj = self.ed_project.text().strip()
        if not proj:
            QMessageBox.critical(self, self.tr("Error"),
                                 self.tr("Enter the project name."))
            return
        phases = []
        for row in self._rows:
            err = row.validate()
            if err:
                QMessageBox.critical(self, self.tr("Phase error"), err)
                return
            phases.append(row.get_data())

        # Default output paths (beside the first phase's base dir) when blank.
        base0 = Path(phases[0]["base_dir"])
        out_reg = self.ed_out_reg.text().strip() or str(base0 / f"Ficheros_{proj}.xlsx")
        out_coord = self.ed_out_coord.text().strip() or str(base0 / f"coordenadas_{proj}.xlsx")
        if not out_reg.endswith(".xlsx"):
            out_reg += ".xlsx"
        if not out_coord.endswith(".xlsx"):
            out_coord += ".xlsx"

        self.btn_generate.setEnabled(False)
        self.lbl_status.setText(self.tr("Processing…"))
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        errors: List[str] = []
        sum_reg: List[str] = []
        sum_coord: List[str] = []
        try:
            try:
                sum_reg = campaign.build_registro_excel(phases, proj, out_reg)
            except PermissionError:
                errors.append(self.tr("Registry — file in use:\n{p}").format(p=out_reg))
            except Exception as e:  # noqa: BLE001 — surface any builder error
                errors.append(f"Registry — {e}")
            try:
                sum_coord = campaign.build_coordenadas_excel(phases, proj, out_coord)
            except PermissionError:
                errors.append(self.tr("Coordinates — file in use:\n{p}").format(p=out_coord))
            except Exception as e:  # noqa: BLE001
                errors.append(f"Coordinates — {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_generate.setEnabled(True)

        if errors:
            self.lbl_status.setText(self.tr("Completed with errors."))
            QMessageBox.critical(self, self.tr("Errors while generating"),
                                 "\n\n".join(errors))
            return
        self.lbl_status.setText(self.tr("Saved: {a}  |  {b}").format(
            a=Path(out_reg).name, b=Path(out_coord).name))
        reg_lines = "\n".join(f"  • {s}" for s in sum_reg)
        coord_lines = "\n".join(f"  • {s}" for s in sum_coord)
        QMessageBox.information(
            self, self.tr("Completed"),
            self.tr("Files generated successfully.\n\n"
                    "Registry ({rn}):\n{rl}\n\n"
                    "Coordinates ({cn}):\n{cl}").format(
                        rn=Path(out_reg).name, rl=reg_lines,
                        cn=Path(out_coord).name, cl=coord_lines))


# ════════════════════════════════════════════════════════════════════════════════
# Tool B — Acquisition Stats
# ════════════════════════════════════════════════════════════════════════════════

# One control row per phase (name/dir/browse/remove) at normal control height,
# plus card margins — same "never collapse below legible size" floor as
# _FILES_ROW_MIN_HEIGHT above, sized for this row's single line of controls.
_STATS_ROW_MIN_HEIGHT = 56


class _StatsPhaseRow(_PhaseRowBase):
    """One phase: name + directory."""

    def __init__(self, index: int, remove_cb, parent=None) -> None:
        super().__init__(remove_cb, parent)
        self.setMinimumHeight(_STATS_ROW_MIN_HEIGHT)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.ed_name = QLineEdit(self.tr("Phase {n}").format(n=index))
        self.ed_name.setMaximumWidth(130)
        self.ed_dir = QLineEdit()
        self.ed_dir.setPlaceholderText(self.tr("Base directory (SGY)"))
        btn_browse = QPushButton("📁")
        btn_browse.setMaximumWidth(34)
        btn_browse.clicked.connect(
            lambda: _browse_dir(self, self.ed_dir,
                                self.tr("Select base directory (SGY)")))
        btn_remove = QPushButton("✕")
        btn_remove.setMaximumWidth(30)
        btn_remove.clicked.connect(lambda: self._remove_cb(self))
        lay.addWidget(QLabel(self.tr("Name:")))
        lay.addWidget(self.ed_name)
        lay.addWidget(QLabel(self.tr("Directory:")))
        lay.addWidget(self.ed_dir, 1)
        lay.addWidget(btn_browse)
        lay.addWidget(btn_remove)

    def get_data(self):
        return self.ed_name.text().strip(), self.ed_dir.text().strip()

    def validate(self) -> Optional[str]:
        name, d = self.get_data()
        if not name:
            return self.tr("The phase name cannot be empty.")
        if not d:
            return self.tr("Missing directory for phase '{p}'.").format(p=name)
        if not Path(d).is_dir():
            return self.tr("Directory not found for phase '{p}':\n{d}").format(p=name, d=d)
        return None


class AcquisitionStatsDialog(QDialog):
    """Ping rate / interval / vessel speed from SEG-Y time & nav headers."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Cruise — Acquisition Stats"))
        self.setMinimumSize(760, 620)
        self.resize(820, 720)
        self._rows: List[_StatsPhaseRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        banner = _DirStructureBanner(self.tr(
            "<b>Required directory structure.</b> Each phase directory must "
            "contain the seismic lines as <code>*.sgy</code> files — either "
            "directly, in per-line subfolders, or under an <b>SGY/</b> subfolder. "
            "Ping rate, interval and vessel speed are computed from the SEG-Y "
            "time and navigation headers."))
        root.addWidget(banner)

        # ── "Fases a procesar" — phase rows + Add button, matching the
        # original LabelFrame (which hosted both together). ──
        grp_phases = _group_box(self.tr("Phases to process"))
        phases_lay = QVBoxLayout(grp_phases)
        phases_lay.setContentsMargins(10, 14, 10, 10)
        phases_lay.setSpacing(8)

        self._phases_box = QVBoxLayout()
        self._phases_box.setSpacing(8)
        scroll = _phases_scroll_area(self._phases_box)
        phases_lay.addWidget(scroll, 1)

        self.btn_add = QPushButton(self.tr("+ Add phase"))
        self.btn_add.clicked.connect(self._add_phase)
        phases_lay.addWidget(self.btn_add, 0, Qt.AlignmentFlag.AlignLeft)
        root.addWidget(grp_phases, 1)

        self.btn_calc = QPushButton(self.tr("⚡ Calculate metrics"))
        self.btn_calc.setObjectName("primary")
        self.btn_calc.clicked.connect(self._run)
        root.addWidget(self.btn_calc)

        # ── "Resultados" ──
        grp_log = _group_box(self.tr("Results"))
        log_lay = QVBoxLayout(grp_log)
        log_lay.setContentsMargins(10, 14, 10, 10)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont(MONO, 9))
        self.txt_log.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        log_lay.addWidget(self.txt_log)
        root.addWidget(grp_log, 2)

        self._add_phase()

    def _add_phase(self) -> None:
        row = _StatsPhaseRow(len(self._rows) + 1, self._remove_phase)
        self._rows.append(row)
        self._phases_box.addWidget(row)
        _grow_for_new_row(self, _STATS_ROW_MIN_HEIGHT, self._phases_box.spacing())

    def _remove_phase(self, row: _StatsPhaseRow) -> None:
        if len(self._rows) == 1:
            QMessageBox.warning(self, self.tr("Notice"),
                                self.tr("At least one phase is required."))
            return
        self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _run(self) -> None:
        phases = []
        for row in self._rows:
            err = row.validate()
            if err:
                QMessageBox.critical(self, self.tr("Phase error"), err)
                return
            phases.append(row.get_data())

        self.btn_calc.setEnabled(False)
        self.txt_log.clear()
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        all_hz: List[float] = []
        all_s: List[float] = []
        all_kn: List[float] = []
        try:
            for name, path in phases:
                log_str, hz, s, kn = campaign.calculate_metrics_for_phase(name, path)
                self.txt_log.appendPlainText(log_str + "\n")
                all_hz += hz
                all_s += s
                all_kn += kn
                QApplication.processEvents()

            self.txt_log.appendPlainText("=" * 70)
            if all_hz:
                avg_hz = sum(all_hz) / len(all_hz)
                avg_s = sum(all_s) / len(all_s)
                self.txt_log.appendPlainText(self.tr("🚀 PROJECT-WIDE AVERAGES:"))
                self.txt_log.appendPlainText(
                    self.tr("   Ping rate:      {v:.2f} Hz").format(v=avg_hz))
                self.txt_log.appendPlainText(
                    self.tr("   Ping interval:  {v:.3f} s").format(v=avg_s))
                if all_kn:
                    avg_v = sum(all_kn) / len(all_kn)
                    self.txt_log.appendPlainText(
                        self.tr("   Vessel speed:   {v:.2f} knots").format(v=avg_v))
                else:
                    self.txt_log.appendPlainText(
                        self.tr("   Vessel speed:   no navigation data"))
            else:
                self.txt_log.appendPlainText(
                    self.tr("⚠️ No valid data found in the whole project."))
            self.txt_log.appendPlainText("=" * 70)
        except Exception as e:  # noqa: BLE001 — report and keep the dialog usable
            QMessageBox.critical(self, self.tr("Calculation error"), str(e))
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_calc.setEnabled(True)
