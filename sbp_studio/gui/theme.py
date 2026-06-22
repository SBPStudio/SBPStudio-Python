"""
theme.py — Centralised theming: named palettes + a switchable ThemeManager.

All colours live here, never hard-coded inside individual widgets. Two modern
palettes are provided — ``dark`` (default) and ``light`` — sharing identical
keys so the same QSS template renders either. The :class:`ThemeManager`
singleton (:data:`theme`) owns the active palette, re-installs the application
stylesheet on switch, and emits :pyattr:`ThemeManager.theme_changed` so widgets
holding inline (non-QSS) colours can refresh.

Widgets must read colours through ``theme.color(...)`` (live) rather than
capturing a palette dict at construction, so a runtime theme switch is honoured.
"""
from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

# Monospace UI font used across the whole application.
MONO = "Courier New"

# ── Palettes ────────────────────────────────────────────────────────────────────
# Every palette defines the same keys; the QSS template below interpolates them.
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "bg":        "#1e1e1e",
        "panel":     "#2d2d2d",
        "sidebar":   "#252526",
        "accent":    "#3c3c3c",
        "highlight": "#094771",
        "bright":    "#4fc1ff",
        "warn":      "#f48771",
        "ok":        "#89d185",
        "text":      "#f0f0f0",
        "sub":       "#a0a6b0",
        "entry":     "#1a1a1a",
        "sel":       "#094771",
        # Header gradient stops (top → bottom) + its accent under-line.
        "topbar_a":  "#343438",
        "topbar_b":  "#242427",
        "topbar_ln": "#4fc1ff",
    },
    "light": {
        "bg":        "#f4f6fa",
        "panel":     "#ffffff",
        "sidebar":   "#eaeef5",
        "accent":    "#d6deea",
        "highlight": "#cfe2f7",
        "bright":    "#2f6fb0",
        "warn":      "#d23f57",
        "ok":        "#1f9d6b",
        "text":      "#1b2330",
        "sub":       "#5b6677",
        "entry":     "#ffffff",
        "sel":       "#cfe2f7",
        "topbar_a":  "#ffffff",
        "topbar_b":  "#e7ecf4",
        "topbar_ln": "#2f6fb0",
    },
}

DEFAULT_THEME = "dark"


# ── Stylesheet builder ──────────────────────────────────────────────────────────

def build_qss(palette: Dict[str, str]) -> str:
    """Return the application-wide QSS string, interpolating ``palette`` colours."""
    c = palette
    return f"""
    QWidget {{
        background-color: {c['bg']};
        color: {c['text']};
        font-family: "{MONO}";
        font-size: 12px;
    }}
    QFrame#panel    {{ background-color: {c['panel']};   }}
    QFrame#sidebar  {{ background-color: {c['sidebar']}; }}
    /* Premium IDE-style header: a soft top-to-bottom gradient lifts it off the
       flat workspace, and a 2px accent under-line gives a crisp, modern edge
       (QSS has no box-shadow, so the bright rule doubles as the 'drop shadow'). */
    QFrame#topbar   {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {c['topbar_a']}, stop:1 {c['topbar_b']});
        border-bottom: 2px solid {c['topbar_ln']};
    }}
    /* Children of the header inherit no background so the gradient shows through. */
    QFrame#topbar > QLabel {{ background: transparent; }}
    QFrame#hline    {{ background-color: {c['accent']}; max-height: 1px; min-height: 1px; }}

    QLabel#title    {{ color: {c['text']}; font-size: 20px; font-weight: bold;
                       padding: 0 2px; }}
    QLabel#subtitle {{ color: {c['sub']};    font-size: 11px; padding-bottom: 1px; }}
    QLabel#section  {{ color: {c['bright']}; font-size: 11px; font-weight: bold; }}
    QLabel#hdr      {{ color: {c['bright']}; font-size: 11px; font-weight: bold;
                       background-color: {c['accent']}; padding: 6px 10px; }}
    QLabel#sub      {{ color: {c['sub']};    font-size: 11px; }}
    QLabel#info     {{ color: {c['sub']};    font-size: 11px; }}
    QLabel#placeholder {{ color: {c['sub']}; font-size: 13px; }}

    QListWidget, QTextEdit, QLineEdit, QTableWidget {{
        background-color: {c['entry']};
        color: {c['text']};
        border: 1px solid {c['accent']};
        selection-background-color: {c['sel']};
        selection-color: {c['bright']};
    }}
    QListWidget::item {{ color: {c['text']}; }}
    QListWidget::item:selected {{ background-color: {c['sel']}; color: {c['bright']}; }}

    QHeaderView::section {{
        background-color: {c['accent']};
        color: {c['bright']};
        border: 0px;
        padding: 4px;
        font-weight: bold;
    }}
    QTableWidget {{ gridline-color: {c['accent']}; }}

    QMenuBar {{ background-color: {c['accent']}; color: {c['text']}; }}
    QMenuBar::item {{ background: transparent; padding: 4px 12px; }}
    QMenuBar::item:selected {{ background: {c['highlight']}; color: {c['bright']}; }}
    QMenu {{ background-color: {c['panel']}; color: {c['text']}; border: 1px solid {c['accent']}; }}
    QMenu::item:selected {{ background-color: {c['sel']}; color: {c['bright']}; }}
    QMenu::item:checked {{ color: {c['bright']}; }}

    QComboBox {{
        background-color: {c['entry']};
        color: {c['text']};
        border: 1px solid {c['accent']};
        padding: 3px 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['entry']};
        color: {c['text']};
        selection-background-color: {c['sel']};
        selection-color: {c['bright']};
        border: 1px solid {c['accent']};
    }}
    QComboBox::drop-down {{ border: 0px; width: 18px; }}

    QSpinBox, QDoubleSpinBox {{
        background-color: {c['entry']};
        color: {c['text']};
        border: 1px solid {c['accent']};
        padding: 2px 4px;
    }}

    QPushButton {{
        background-color: {c['accent']};
        color: {c['text']};
        border: 1px solid {c['highlight']};
        padding: 5px 12px;
    }}
    QPushButton:hover {{ background-color: {c['highlight']}; color: {c['bright']}; }}
    QPushButton:disabled {{ color: {c['sub']}; border-color: {c['accent']}; }}

    QCheckBox {{ color: {c['text']}; spacing: 6px; }}
    QCheckBox::indicator {{
        width: 13px; height: 13px;
        border: 1px solid {c['sub']};
        background: {c['entry']};
    }}
    QCheckBox::indicator:checked {{ background: {c['bright']}; border: 1px solid {c['bright']}; }}

    /* Indicator-less "segmented control": the whole option is a clickable pill
       that lights up cyan when selected (no radio circle at all). */
    QRadioButton::indicator {{ width: 0px; height: 0px; border: none; background: none; }}
    QRadioButton {{
        color: {c['text']};
        padding: 5px 12px;
        border: 1px solid {c['sub']};
        border-radius: 4px;
        background: {c['entry']};
    }}
    QRadioButton:hover {{ border-color: {c['bright']}; }}
    QRadioButton:checked {{
        background-color: {c['bright']};
        color: {c['bg']};
        font-weight: bold;
        border: 1px solid {c['bright']};
    }}

    QTabWidget::pane {{ border: 1px solid {c['accent']}; background: {c['bg']}; }}
    QTabBar::tab {{
        background: {c['accent']};
        color: {c['sub']};
        padding: 6px 14px;
        font-family: "{MONO}";
        font-size: 11px;
    }}
    QTabBar::tab:selected {{ background: {c['panel']}; color: {c['text']}; }}

    QScrollBar:vertical   {{ background: {c['bg']}; width: 12px; margin: 0; }}
    QScrollBar:horizontal {{ background: {c['bg']}; height: 12px; margin: 0; }}
    QScrollBar::handle    {{ background: {c['accent']}; border-radius: 3px; min-height: 24px; min-width: 24px; }}
    QScrollBar::handle:hover {{ background: {c['highlight']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

    QProgressBar {{
        background-color: {c['bg']};
        border: 0px; height: 4px; text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{ background-color: {c['bright']}; }}

    QStatusBar {{ background: {c['accent']}; color: {c['sub']}; }}
    QStatusBar::item {{ border: 0px; }}

    QSplitter::handle {{ background: {c['bg']}; }}
    """


# ── Theme manager ───────────────────────────────────────────────────────────────

class ThemeManager(QObject):
    """Owns the active palette, applies the stylesheet, and notifies on change."""

    theme_changed = pyqtSignal(str)

    def __init__(self, name: str = DEFAULT_THEME) -> None:
        super().__init__()
        self._name = name if name in THEMES else DEFAULT_THEME
        self._app = None  # set on first apply()

    @property
    def name(self) -> str:
        return self._name

    def palette(self) -> Dict[str, str]:
        return THEMES[self._name]

    def color(self, key: str) -> str:
        """Live colour lookup for inline (non-QSS) styling."""
        return THEMES[self._name].get(key, "#000000")

    def apply(self, app) -> None:
        """Install the current theme's stylesheet on ``app`` and remember it."""
        self._app = app
        app.setStyleSheet(build_qss(self.palette()))

    def set_theme(self, name: str) -> None:
        """Switch palette, re-apply the stylesheet and emit :pyattr:`theme_changed`."""
        if name not in THEMES or name == self._name:
            return
        self._name = name
        if self._app is not None:
            self._app.setStyleSheet(build_qss(self.palette()))
        self.theme_changed.emit(name)


# Module-level singleton shared by the whole GUI.
theme = ThemeManager()


def apply_theme(app, name: Optional[str] = None) -> None:
    """Convenience: optionally select a theme by name, then apply it to ``app``."""
    if name:
        theme._name = name if name in THEMES else theme._name
    theme.apply(app)
