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
        "bg":        "#12141a",
        "panel":     "#1a1d26",
        "sidebar":   "#141720",
        "accent":    "#1e2535",
        "highlight": "#2a3a5c",
        "bright":    "#4d9de0",
        "warn":      "#e94560",
        "ok":        "#3ddc97",
        "text":      "#dce3ee",
        "sub":       "#6a7a96",
        "entry":     "#0e1118",
        "sel":       "#253555",
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
    QFrame#topbar   {{ background-color: {c['accent']};  }}
    QFrame#hline    {{ background-color: {c['accent']}; max-height: 1px; min-height: 1px; }}

    QLabel#title    {{ color: {c['bright']}; font-size: 18px; font-weight: bold; }}
    QLabel#subtitle {{ color: {c['sub']};    font-size: 10px; }}
    QLabel#section  {{ color: {c['bright']}; font-size: 10px; font-weight: bold; }}
    QLabel#hdr      {{ color: {c['bright']}; font-size: 10px; font-weight: bold;
                       background-color: {c['accent']}; padding: 6px 10px; }}
    QLabel#sub      {{ color: {c['sub']};    font-size: 10px; }}
    QLabel#info     {{ color: {c['sub']};    font-size: 10px; }}
    QLabel#placeholder {{ color: {c['sub']}; font-size: 13px; }}

    QListWidget, QTextEdit, QLineEdit, QTableWidget {{
        background-color: {c['entry']};
        color: {c['text']};
        border: 1px solid {c['accent']};
        selection-background-color: {c['sel']};
        selection-color: {c['bright']};
    }}
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
