"""
i18n.py — Qt-native internationalisation (translation files, English as base).

Design
------
All user-facing strings are written in **English** directly in the widgets,
using Qt's own translation API (``QObject.tr`` / ``QCoreApplication.translate``).
There is no in-code translation dictionary and no hard-coded language mapping.

Translations live in standard Qt Linguist ``.ts`` files under
``topassuite/gui/translations/`` (e.g. ``topassuite_es.ts``), regenerated from
the source with ``pylupdate6``:

    pylupdate6 topassuite/gui/**/*.py -ts topassuite/gui/translations/topassuite_es.ts

At runtime the :class:`LanguageManager` installs a translator on the
``QApplication`` and emits :pyattr:`LanguageManager.language_changed`; every
widget implements ``retranslate_ui()`` (re-reading its strings via ``self.tr``)
and connects it to that signal.

Compiled ``.qm`` vs source ``.ts``
----------------------------------
The stock ``QTranslator`` can only load compiled ``.qm`` files (built with
``lrelease``). To keep the project buildable even where ``lrelease`` is not
installed, :class:`TsTranslator` loads a ``.qm`` when present and otherwise
parses the ``.ts`` XML directly. Once ``lrelease`` is available, dropping a
``.qm`` beside the ``.ts`` is picked up automatically — no code change.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import QObject, QTranslator, pyqtSignal

# Base (source) language — strings written in code. Others come from .ts/.qm.
BASE_LANG = "en"
# Human-readable language names (shown untranslated, in their own language).
LANGUAGE_NAMES: Dict[str, str] = {"en": "English", "es": "Español"}
DEFAULT_LANG = "es"

_TRANSLATIONS_DIR = Path(__file__).resolve().parent / "translations"


class TsTranslator(QTranslator):
    """A ``QTranslator`` that can load a Qt Linguist ``.ts`` file directly.

    Falls back gracefully: :meth:`load_lang` tries the compiled ``.qm`` first.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # (context, source, disambiguation) -> translation
        self._map: Dict[Tuple[str, str, str], str] = {}

    # ── Loading ─────────────────────────────────────────────────────────────

    def load_lang(self, lang: str) -> bool:
        """Load ``topassuite_<lang>.qm`` if present, else parse the ``.ts``."""
        qm = _TRANSLATIONS_DIR / f"topassuite_{lang}.qm"
        if qm.exists() and self.load(str(qm)):
            self._map.clear()  # native .qm handles lookups
            return True
        ts = _TRANSLATIONS_DIR / f"topassuite_{lang}.ts"
        if ts.exists():
            return self._load_ts(ts)
        return False

    def _load_ts(self, path: Path) -> bool:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            return False
        mapping: Dict[Tuple[str, str, str], str] = {}
        for ctx in root.findall("context"):
            name_el = ctx.find("name")
            context = name_el.text or "" if name_el is not None else ""
            for msg in ctx.findall("message"):
                src_el = msg.find("source")
                tr_el = msg.find("translation")
                if src_el is None or tr_el is None:
                    continue
                # type="unfinished"/"vanished" → treat as no translation.
                if tr_el.get("type") in ("unfinished", "vanished"):
                    continue
                text = tr_el.text or ""
                if not text:
                    continue
                com_el = msg.find("comment")
                disambig = (com_el.text or "") if com_el is not None else ""
                mapping[(context, src_el.text or "", disambig)] = text
        self._map = mapping
        return bool(mapping)

    # ── Lookup (only used for the .ts fallback path) ────────────────────────

    def translate(self, context, sourceText, disambiguation=None, n=-1):  # noqa: N802
        if not self._map:
            # .qm path: let the base class resolve it.
            return super().translate(context, sourceText, disambiguation, n)
        key = (context or "", sourceText or "", disambiguation or "")
        hit = self._map.get(key)
        if hit is None and disambiguation:
            hit = self._map.get((context or "", sourceText or "", ""))
        # Returning "" makes Qt fall back to the (English) source text.
        return hit or ""


class LanguageManager(QObject):
    """Installs/removes the active translator and notifies the UI on change."""

    language_changed = pyqtSignal(str)

    def __init__(self, lang: str = DEFAULT_LANG) -> None:
        super().__init__()
        self._lang = lang
        self._app = None
        self._translator: Optional[TsTranslator] = None

    @property
    def language(self) -> str:
        return self._lang

    def install(self, app, lang: Optional[str] = None) -> None:
        """Apply ``lang`` (or the current language) to ``app``."""
        self._app = app
        self._apply(lang or self._lang, emit=False)

    def set_language(self, lang: str) -> None:
        """Switch language at runtime and emit :pyattr:`language_changed`."""
        if lang == self._lang or lang not in LANGUAGE_NAMES:
            return
        self._apply(lang, emit=True)

    def _apply(self, lang: str, *, emit: bool) -> None:
        if self._app is not None and self._translator is not None:
            self._app.removeTranslator(self._translator)
            self._translator = None
        # The base language needs no translator (source strings are English).
        if lang != BASE_LANG and self._app is not None:
            tr = TsTranslator(self._app)
            if tr.load_lang(lang):
                self._app.installTranslator(tr)
                self._translator = tr
        self._lang = lang
        if emit:
            self.language_changed.emit(lang)


# Module-level singleton shared by the whole GUI.
language_manager = LanguageManager()
