"""
export_dialog.py — Compact options dialog for the high-quality (matplotlib) export.

The interactive views use PyQtGraph, but image/PDF export goes through the core
headless matplotlib renderer (``topassuite.viz.render``) — the same path as the
CLI ``export-image`` — for proper axes, ticks, time labels, FIX styling and
vector PDF output.

Only format, DPI and theme are exposed here. The display SCALE (aspect ratio)
is taken from the visualizer's own scale control so the export matches what is
on screen — it is NOT duplicated in this dialog. The finer styling (ticks, time
label format, FIX styling, margins, fill, PDF page) is baked to sensible
defaults in :meth:`config`. The DSP/processing comes from the main controls
panel — the export reflects how the section is processed (WYSIWYG).
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
    QVBoxLayout, QWidget,
)

# DPI choices offered in the dropdown (editable — any value can be typed).
DPI_CHOICES = ["96", "150", "200", "300", "450", "600", "900", "1200", "1800", "2400"]
DEFAULT_DPI = "600"


class ExportDialog(QDialog):
    """Collects the essential presentation options for a high-quality export."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(360)
        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self.cb_format = QComboBox()
        for label, ext in (("PDF (vector)", "pdf"), ("PNG", "png"),
                            ("TIFF", "tiff"), ("SVG (vector)", "svg")):
            self.cb_format.addItem(label, ext)
        self.cb_dpi = QComboBox()
        self.cb_dpi.setEditable(True)
        self.cb_dpi.addItems(DPI_CHOICES)
        self.cb_dpi.setCurrentText(DEFAULT_DPI)
        self.cb_theme = QComboBox()
        self.cb_theme.addItems(["dark", "light", "print"])
        self.cb_theme.setCurrentText("print")

        self._rows = [
            ("Format", self.cb_format),
            ("Resolution (DPI)", self.cb_dpi),
            ("Theme", self.cb_theme),
        ]
        self._label_widgets: list[QLabel] = []
        for _text, widget in self._rows:
            label = QLabel()
            self._label_widgets.append(label)
            form.addRow(label, widget)

        # File-boundary (red seam) lines in the export. Independent of the
        # interactive viewer's own "Show file boundaries" toggle. Unchecked
        # by default — boundaries are only drawn in the export if requested.
        self.cb_boundaries = QCheckBox()
        self.cb_boundaries.setChecked(False)
        root.addWidget(self.cb_boundaries)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.retranslate_ui()

    # ── Result ──────────────────────────────────────────────────────────────

    def _dpi(self) -> int:
        try:
            return max(50, min(4800, int(float(self.cb_dpi.currentText()))))
        except ValueError:
            return int(DEFAULT_DPI)

    def config(self) -> dict:
        """Exposed options + baked defaults (the latter mirror a good print recipe).

        Scale (aspect ratio) is intentionally absent — the export reads it from
        the visualizer's scale control so the output matches the on-screen view.
        """
        return dict(
            # ── Exposed ──
            format=self.cb_format.currentData(),
            dpi=self._dpi(),
            theme=self.cb_theme.currentText(),
            draw_file_boundaries=self.cb_boundaries.isChecked(),
            # ── Baked defaults ──
            velocity=1500.0,
            pdf_page="auto",
            x_tick=None,
            t_tick=None,
            time_ticks=5,
            time_fmt="full",
            time_font_size=6.0,
            time_align="left",
            fix_color=None,        # use theme FIX colour
            fix_bbox_alpha=0.0,
            margin_top=20.0,
            margin_bottom=20.0,
            fill_zero=True,
            grid=False,
        )

    # ── i18n ────────────────────────────────────────────────────────────────

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.tr("Export options"))
        labels = [self.tr("Format"), self.tr("Resolution (DPI)"), self.tr("Theme")]
        for label_widget, text in zip(self._label_widgets, labels):
            label_widget.setText(text)
        self.cb_boundaries.setText(self.tr("Include red file boundary lines in export"))
        # Standard buttons render with no visible text under the dark QSS — set
        # explicit, translated text so they're always readable.
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("Accept"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("Cancel"))
