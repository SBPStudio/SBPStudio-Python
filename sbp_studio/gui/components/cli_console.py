"""
cli_console.py — Integrated command-line console (floating, undockable window).

A functional internal terminal: an output log + an input line that parses typed
commands and prints responses. It is a real top-level window (``Qt.WindowType
.Tool``) loosely tied to the main window (stays on top of it, shown/hidden by
the menu-bar CLI toggle) but otherwise free — the user can drag it by its title
bar to anywhere on screen, including outside the main window's bounds. It
defaults to spawning at the bottom-left corner of the main window on first show.

The command set is intentionally small but real, with a ``register`` hook so core
batch operations can be wired in later without touching this widget.
"""
from __future__ import annotations

import contextlib
import io
import os
import shlex
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QTextBrowser, QVBoxLayout, QWidget,
)

from ...core.logger import get_logger
from ..i18n import language_manager
from ..theme import MONO, theme

_LOG = get_logger("cli")

# Legacy headless CLI commands surfaced inside the console (name, one-line help).
# Dispatched through the SAME argparse logic as `python -m sbp_studio.cli.main`.
_CORE_COMMANDS = [
    ("info",         "print SEG-Y metadata"),
    ("check",        "inspect headers + scan for anomalies"),
    ("patch-header", "patch dt/ns header fields in place (r+)"),
    ("process",      "run a headless DSP pipeline → new SEG-Y"),
    ("reproject",    "reproject SEG-Y file(s) to a new CRS"),
    ("join-chain",   "reproject + join a chain into one SEG-Y"),
    ("export-image", "export profile/chain as image (PNG/PDF/…)"),
    ("batch-export", "render many files/dirs into one folder"),
    ("spectrum",     "export the frequency-spectrum figure"),
    ("navline",      "export the navigation track (shp/geojson/csv)"),
    ("fix",          "export FIX-point marks (shp/geojson/csv)"),
    ("accel",        "show hardware acceleration status"),
]


# File extensions that mark a bare token as a path even without a separator.
_PATH_EXTS = {".sgy", ".seg", ".segy", ".shp", ".tif", ".tiff", ".geojson",
              ".json", ".csv", ".pdf", ".png", ".svg", ".gpkg"}


def _app_root() -> str:
    """Application root directory for resolving relative CLI paths.

    * Frozen (PyInstaller ``.exe``) → the folder CONTAINING the executable
      (``dirname(sys.executable)``) — i.e. next to the .exe on the user's disk,
      NOT the temporary ``sys._MEIPASS`` extraction dir. So a user dropping an
      ``examples\\`` folder beside the .exe gets ``.\\examples\\…`` to resolve.
    * Source run → the PROJECT ROOT, derived from this file's package location
      (``…/sbp_studio/gui/components/cli_console.py`` → parents[3]). This is the
      directory where ``examples/`` actually lives, and is correct regardless of
      how the app was launched (``applications/…`` launcher, ``python -m
      sbp_studio.gui``, an IDE, etc.) — unlike ``dirname(sys.argv[0])`` which
      points at the launcher's own sub-folder and breaks ``.\\examples\\…``."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return str(Path(__file__).resolve().parents[3])


def _is_drive_absolute(tok: str) -> bool:
    """True for an absolute path, incl. a Windows drive root like ``C:\\…``."""
    return os.path.isabs(tok) or (len(tok) >= 2 and tok[1] == ":")


class _StreamTee:
    """File-like object: forwards every write to a callback (→ the console),
    so a command's stdout/stderr streams live into the output area."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit

    def write(self, s: str) -> int:
        if s:
            self._emit(s)
        return len(s)

    def flush(self) -> None:
        pass


class _CommandInput(QLineEdit):
    """Line edit with Up/Down command-history recall."""

    def __init__(self, console: "CliConsole") -> None:
        super().__init__()
        self._console = console

    def keyPressEvent(self, ev: QKeyEvent) -> None:
        if ev.key() == Qt.Key.Key_Up:
            self._console.recall_history(-1)
        elif ev.key() == Qt.Key.Key_Down:
            self._console.recall_history(+1)
        else:
            super().keyPressEvent(ev)


class CliConsole(QWidget):
    """Frameless, undockable floating console — a real top-level window (not
    clipped to the main window's bounds) that the user can drag anywhere on
    screen by its title bar. Tied to the main window only via ``Qt.WindowType
    .Tool`` (stays on top of it, hides/shows together) — there is no docking
    or snapping back into the main layout."""

    # Emitted with the raw command line for any external listener (future hooks).
    command_entered = pyqtSignal(str)
    # Internal: worker-thread stdout/stderr writes, marshalled to the GUI thread.
    _log_line = pyqtSignal(str)

    def __init__(self, window: QWidget) -> None:
        super().__init__(window, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self._win = window
        self.setObjectName("cliConsole")
        self.setAutoFillBackground(True)
        self._drag_pos: Optional[QPoint] = None    # set while the title bar is being dragged

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(4)

        self._header_bar = QWidget()
        header = QHBoxLayout(self._header_bar)
        header.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel("›_  SBP Studio CLI")     # brand, not translated
        self._title.setObjectName("section")
        header.addWidget(self._title)
        header.addStretch(1)
        self._btn_close = QPushButton("✕")
        self._btn_close.setFixedWidth(24)
        self._btn_close.clicked.connect(self.hide)
        header.addWidget(self._btn_close)
        root.addWidget(self._header_bar)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.output.setFont(QFont(MONO, 9))
        root.addWidget(self.output, 1)

        self.input = _CommandInput(self)
        self.input.setFont(QFont(MONO, 9))
        self.input.returnPressed.connect(self._run)
        root.addWidget(self.input)

        # History + command registry.
        self._history: List[str] = []
        self._hist_pos = 0
        self._busy = False                     # one core command at a time
        self._guide: Optional[QDialog] = None
        self._cwd = _app_root()                # internal working dir (pwd/cd)
        self._commands: Dict[str, Callable[[List[str]], None]] = {}
        self._register_builtins()
        self._wire_core_commands()

        self._log_line.connect(self._append_raw)   # worker → GUI (queued)
        self._restyle()
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._restyle)
        self._intro()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, name: str, handler: Callable[[List[str]], None],
                 help_text: str = "") -> None:
        """Register an external command (e.g. a future core batch op)."""
        self._commands[name] = handler
        self._help_lines[name] = help_text

    def println(self, text: str = "") -> None:
        self.output.appendPlainText(text)

    def _append_raw(self, text: str) -> None:
        """Insert worker output verbatim (preserves the command's own newlines)."""
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.ensureCursorVisible()

    def focus_input(self) -> None:
        self.input.setFocus()

    # ── Drag-to-move (frameless top-level window has no native title bar) ──────

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and \
                self._header_bar.geometry().contains(ev.position().toPoint()):
            self._drag_pos = ev.globalPosition().toPoint() - self.pos()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag_pos is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag_pos)
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(ev)

    def recall_history(self, step: int) -> None:
        if not self._history:
            return
        self._hist_pos = max(0, min(len(self._history), self._hist_pos + step))
        self.input.setText(self._history[self._hist_pos]
                           if self._hist_pos < len(self._history) else "")

    # ── Command handling ──────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(line: str) -> List[str]:
        """Split a command line into tokens, honouring quotes BUT keeping Windows
        backslash paths intact (``posix=False`` so ``.\\examples\\x.sgy`` survives),
        then strip surrounding quotes (so ``--fix-color "#cc4444"`` → ``#cc4444``)."""
        try:
            toks = shlex.split(line, posix=False)
        except ValueError:
            toks = line.split()
        out = []
        for t in toks:
            if len(t) >= 2 and t[0] == t[-1] and t[0] in ("\"", "'"):
                t = t[1:-1]
            out.append(t)
        return out

    def _run(self) -> None:
        line = self.input.text().strip()
        self.input.clear()
        if not line:
            return
        self._history.append(line)
        self._hist_pos = len(self._history)
        self.println(f"› {line}")
        self.command_entered.emit(line)
        parts = self._tokenize(line)
        cmd, args = parts[0].lower(), parts[1:]
        handler = self._commands.get(cmd)
        if handler is None:
            self.println(f"Unknown command: {cmd!r}. Type 'help'.")
            return
        try:
            handler(args)
        except Exception as exc:                         # never let a command crash the UI
            self.println(f"Error: {exc}")

    def _register_builtins(self) -> None:
        self._help_lines: Dict[str, str] = {}
        reg = self.register
        reg("help", self._cmd_help, "list commands")
        reg("clear", lambda a: self.output.clear(), "clear the output")
        reg("echo", lambda a: self.println(" ".join(a)), "print text")
        reg("version", self._cmd_version, "show app / core versions")
        reg("theme", self._cmd_theme, "theme [dark|light]")
        reg("lang", self._cmd_lang, "lang [es|en]")
        reg("profiles", self._cmd_profiles, "list loaded profiles")
        reg("pwd", self._cmd_pwd, "print the working directory")
        reg("cd", self._cmd_cd, "cd <path> — change working directory")
        reg("close", lambda a: self.hide(), "hide the console")

    def _cmd_help(self, _args: List[str]) -> None:
        self.println("Commands:")
        for name in sorted(self._commands):
            self.println(f"  {name:<10} {self._help_lines.get(name, '')}")

    def _cmd_version(self, _args: List[str]) -> None:
        import sbp_studio
        self.println(f"SBP Studio {getattr(sbp_studio, '__version__', '?')}")

    def _cmd_theme(self, args: List[str]) -> None:
        if args and args[0] in ("dark", "light"):
            theme.set_theme(args[0])
            self.println(f"theme → {args[0]}")
        else:
            self.println(f"theme is '{theme.name}' (use: theme dark|light)")

    def _cmd_lang(self, args: List[str]) -> None:
        if args and args[0] in ("es", "en"):
            language_manager.set_language(args[0])
            self.println(f"lang → {args[0]}")
        else:
            self.println(f"lang is '{language_manager.language}' (use: lang es|en)")

    def _cmd_profiles(self, _args: List[str]) -> None:
        state = getattr(self._win, "state", None)
        profiles = getattr(state, "profiles", {}) if state is not None else {}
        if not profiles:
            self.println("(no profiles loaded)")
            return
        for i, p in enumerate(profiles.values()):
            self.println(f"  [{i}] {getattr(p, 'name', '?')}")

    def _cmd_pwd(self, _args: List[str]) -> None:
        self.println(self._cwd)

    def _cmd_cd(self, args: List[str]) -> None:
        if not args:                            # no arg → back to the app root
            self._cwd = _app_root()
            self.println(self._cwd)
            return
        target = args[0]
        if not _is_drive_absolute(target):
            target = os.path.normpath(os.path.join(self._cwd, target))
        if os.path.isdir(target):
            self._cwd = target
            self.println(self._cwd)
        else:
            self.println(f"cd: not a directory: {target}")

    # ── Relative-path resolution (against the internal working dir) ────────────

    @staticmethod
    def _looks_like_path(tok: str) -> bool:
        """Heuristic: a non-flag, non-absolute token that looks like a relative
        path — starts with ./ ../ .\\ ..\\, contains a separator, or ends in a
        known data/image extension. Deliberately skips flags (``-x``), CRS codes
        (``epsg:4326``), hex colours (``#cc4444``) and plain numbers/words."""
        if not tok or tok.startswith("-") or _is_drive_absolute(tok):
            return False
        if tok.startswith(("./", "../", ".\\", "..\\")):
            return True
        if "/" in tok or "\\" in tok:
            return True
        return os.path.splitext(tok)[1].lower() in _PATH_EXTS

    def _resolve_path(self, tok: str) -> str:
        """Resolve a relative-path token against the internal working dir; leave
        flags / non-path args untouched."""
        if self._looks_like_path(tok):
            return os.path.normpath(os.path.join(self._cwd, tok))
        return tok

    # ── Core CLI bridge (reuses sbp_studio.cli argparse + handlers) ────────────

    def _wire_core_commands(self) -> None:
        """Map the legacy headless CLI commands into the console. They share the
        EXACT argparse logic of ``python -m sbp_studio.cli.main`` — so the user
        types just ``export-image <file> --preset envelope …`` (no module prefix,
        no shell line continuations)."""
        for name, help_text in _CORE_COMMANDS:
            self.register(name, lambda args, n=name: self._run_core_command(n, args),
                          help_text)

    def _run_core_command(self, name: str, args: List[str]) -> None:
        if self._busy:
            self.println("Busy — wait for the current command to finish.")
            return
        # Resolve relative paths against the internal working dir (app-root by
        # default) so commands are portable across machines / a packaged .exe.
        args = [self._resolve_path(a) for a in args]
        # Import the headless CLI lazily (keeps GUI startup light).
        from ...cli.main import build_parser
        from ...cli import commands as cli_cmds

        dispatch = {
            "info": cli_cmds.cmd_info, "check": cli_cmds.cmd_check,
            "patch-header": cli_cmds.cmd_patch_header,
            "process": cli_cmds.cmd_process,
            "reproject": cli_cmds.cmd_reproject,
            "join-chain": cli_cmds.cmd_join_chain,
            "export-image": cli_cmds.cmd_export_image,
            "batch-export": cli_cmds.cmd_batch_export,
            "spectrum": cli_cmds.cmd_spectrum, "navline": cli_cmds.cmd_navline,
            "fix": cli_cmds.cmd_fix, "accel": cli_cmds.cmd_accel,
        }
        # Parse on the GUI thread (cheap); capture usage/errors for display.
        parser = build_parser()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ns = parser.parse_args([name] + args)
        except SystemExit:                       # argparse error / -h → message in buf
            msg = buf.getvalue().strip()
            if msg:
                self.println(msg)
            return

        handler = dispatch[name]
        self._busy = True
        self.println(f"running {name}…")
        emit = self._log_line.emit

        def job(progress, cancel):
            # Stream the command's stdout/stderr live into the console. The swap
            # is process-global but commands are serialised (self._busy), and
            # restored in finally.
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _StreamTee(emit)
            try:
                handler(ns)
            except SystemExit as exc:            # _err() in the handlers calls sys.exit
                if exc.code not in (0, None):
                    print(f"\n[exit code {exc.code}]")
                    _LOG.error("CLI command '%s' exited with code %s", name, exc.code)
            except Exception as exc:             # surfaced in-console, never crashes
                import traceback
                print(f"\nError: {exc}\n{traceback.format_exc()}")
                _LOG.exception("CLI command '%s' failed", name)
            finally:
                sys.stdout, sys.stderr = old_out, old_err
            return name

        runner = getattr(self._win, "run_task", None)
        if callable(runner):
            runner(job, self._on_core_done, f"CLI: {name}…")
        else:                                    # no task service (tests) → run inline
            job(None, None)
            self._on_core_done(name)

    def _on_core_done(self, name: str) -> None:
        self._busy = False
        self.println(f"✔ {name} finished.")

    # ── Internals ──────────────────────────────────────────────────────────────

    def _intro(self) -> None:
        self.println("SBP Studio integrated CLI — type 'help'.")

    def _restyle(self, *_) -> None:
        self.setStyleSheet(
            f"#cliConsole {{ background: {theme.color('panel')};"
            f" border: 1px solid {theme.color('highlight')}; border-radius: 6px; }}"
            f"QPlainTextEdit {{ background: {theme.color('bg')};"
            f" color: {theme.color('text')}; border: none; }}"
            f"QLineEdit {{ background: {theme.color('entry')};"
            f" color: {theme.color('bright')}; border: 1px solid {theme.color('sub')};"
            f" padding: 3px 5px; }}")

    def _retranslate(self, *_) -> None:
        self.input.setPlaceholderText(self.tr("Type a command…  (try 'help')"))
        self._btn_close.setToolTip(self.tr("Close"))


# ── Command Guide dialog ──────────────────────────────────────────────────────

_GUIDE_FILE = r".\examples\_real_in\ANT26\SGY\20260215115516.seg"
_GUIDE_OUT = r".\examples\_real_out"

_GUIDE_HTML_ES = f"""
<h2>Guía de Comandos — Consola interna de SBP Studio</h2>

<p>La consola interna ejecuta los <b>mismos comandos</b> que la herramienta de
línea de comandos de SBP Studio, pero <b>directamente dentro de la aplicación</b>.
Escribe el comando y sus argumentos y pulsa <b>Intro</b>.</p>

<h3>Formato</h3>
<pre>&lt;comando&gt; &lt;archivo(s)&gt; [--opción valor] [--bandera]</pre>
<ul>
  <li><b>NO</b> hace falta el prefijo <code>python -m sbp_studio.cli.main</code>.
      Escribe directamente el comando (p. ej. <code>export-image</code>).</li>
  <li><b>NO</b> uses los saltos de línea de PowerShell (el carácter <code>`</code>).
      Pega <b>todo el comando en una sola línea</b>.</li>
  <li>Encierra entre comillas los valores con caracteres especiales, como un color
      hexadecimal <code>"#cc4444"</code> o rutas con espacios.</li>
  <li>Las rutas de Windows con barra invertida se respetan tal cual.</li>
  <li><b>Rutas relativas portátiles:</b> las rutas como
      <code>.\\examples\\…</code> se resuelven respecto a la <b>carpeta de la
      aplicación</b> (no a la unidad <code>Z:\\</code> ni al directorio desde el
      que se lanzó). Funcionan igual ejecutando desde el código o desde un
      <code>.exe</code> empaquetado. Usa <code>pwd</code> para ver el directorio
      actual y <code>cd &lt;ruta&gt;</code> para cambiarlo.</li>
  <li>Usa las flechas <b>↑ / ↓</b> para recuperar comandos anteriores, y
      <code>help</code> para ver la lista completa.</li>
</ul>

<h3>Comandos disponibles</h3>
<ul>
  <li><code>info</code> — metadatos de un SEG-Y</li>
  <li><code>check</code> — inspecciona cabeceras (EBCDIC + binarias) y detecta anomalías</li>
  <li><code>patch-header</code> — corrige <code>dt</code>/<code>ns</code> en la cabecera, in situ (r+)</li>
  <li><code>process</code> — ejecuta una cadena DSP headless y guarda un nuevo SEG-Y</li>
  <li><code>export-image</code> — exporta el perfil/cadena como imagen (PNG/PDF…)</li>
  <li><code>batch-export</code> — renderiza varios archivos/carpetas a un directorio</li>
  <li><code>reproject</code> — reproyecta SEG-Y a otro CRS</li>
  <li><code>join-chain</code> — reproyecta y une una cadena en un solo SEG-Y</li>
  <li><code>spectrum</code> — figura del espectro de frecuencias</li>
  <li><code>navline</code> — exporta la traza de navegación (shp/geojson/csv)</li>
  <li><code>fix</code> — exporta marcas FIX (shp/geojson/csv)</li>
  <li><code>accel</code> — estado de la aceleración por hardware</li>
  <li><code>pwd</code> / <code>cd &lt;ruta&gt;</code> — directorio de trabajo interno</li>
</ul>

<h3>Ejemplos Prácticos</h3>
<p>Cada ejemplo es una <b>sola línea</b> lista para copiar y pegar (selecciónala y
cópiala con Ctrl+C). Usan un archivo real de la campaña ANT26:</p>

<p><b>1 · Inspección rápida</b> — cabeceras y estadísticas básicas:</p>
<pre>info {_GUIDE_FILE}</pre>

<p><b>2 · Exportación del Track</b> — extrae la navegación a un Shapefile:</p>
<pre>navline {_GUIDE_FILE} --format shp --out {_GUIDE_OUT}\\ANT26_track.shp</pre>

<p><b>3 · Reproyección</b> — convierte de geográficas (WGS84) a UTM 30N:</p>
<pre>reproject {_GUIDE_FILE} --src epsg:4326 --dst epsg:32630 --out-dir {_GUIDE_OUT}</pre>

<p><b>4 · Procesamiento y Gráficos</b> — exportación a PDF de alta calidad
(envelope + AGC + alineado + escala física + marcas FIX):</p>
<pre>export-image {_GUIDE_FILE} --preset envelope --cmap Greys --agc --align --fill-zero --x-scale 2 --ratio 3 --velocity 1500 --x-tick 5 --t-tick 50 --time-ticks 5 --time-fmt full --time-font-size 5.5 --time-align left --fix 5 --fix-color "#cc4444" --fix-bbox-alpha 0.0 --margin-top 20 --margin-bottom 20 --theme print --quality high --format pdf --pdf-page auto --timeit --out {_GUIDE_OUT}\\ANT26_L01.pdf</pre>

<p><b>5 · Diagnóstico de cabeceras</b> — texto EBCDIC, campos binarios y anomalías:</p>
<pre>check {_GUIDE_FILE}</pre>

<p><b>6 · Corrección de metadatos</b> — fija el intervalo de muestreo a 50 µs
(se propaga a todas las trazas) y el nº de muestras. <b>Modifica el archivo</b>:</p>
<pre>patch-header {_GUIDE_FILE} --dt 50 --ns 2048 --dry-run</pre>

<p><b>7 · Cadena DSP headless</b> — banda + blanqueo espectral + AGC, a un nuevo SEG-Y:</p>
<pre>process {_GUIDE_FILE} {_GUIDE_OUT}\\ANT26_proc.seg --pipeline "bandpass(1000,8000),whiten(1000,8000,300),agc(200)"</pre>

<p><b>8 · Exportación por lotes</b> — todas las líneas de una carpeta a PDF con la
paleta seismc y exageración vertical fija:</p>
<pre>batch-export .\\examples\\_real_in\\ANT26\\SGY --format pdf --cmap seismc --x-scale 2 --ve 15 --out {_GUIDE_OUT}</pre>

<p>El procesamiento se ejecuta en segundo plano: la interfaz no se bloquea y el
progreso y los mensajes aparecen en el área de la consola.</p>
"""

_GUIDE_HTML_EN = f"""
<h2>Command Guide — SBP Studio internal console</h2>

<p>The internal console runs the <b>same commands</b> as the SBP Studio
command-line tool, but <b>directly inside the application</b>. Type the
command and its arguments and press <b>Enter</b>.</p>

<h3>Format</h3>
<pre>&lt;command&gt; &lt;file(s)&gt; [--option value] [--flag]</pre>
<ul>
  <li>You do <b>NOT</b> need the <code>python -m sbp_studio.cli.main</code>
      prefix. Type the command directly (e.g. <code>export-image</code>).</li>
  <li>Do <b>NOT</b> use PowerShell line continuations (the <code>`</code>
      character). Paste the <b>whole command on a single line</b>.</li>
  <li>Quote values containing special characters, such as a hex colour
      <code>"#cc4444"</code> or paths with spaces.</li>
  <li>Windows paths with backslashes are respected as-is.</li>
  <li><b>Portable relative paths:</b> paths like
      <code>.\\examples\\…</code> are resolved relative to the
      <b>application folder</b> (not the <code>Z:\\</code> drive nor the
      directory the app was launched from). They work the same whether
      running from source or from a packaged <code>.exe</code>. Use
      <code>pwd</code> to see the current directory and
      <code>cd &lt;path&gt;</code> to change it.</li>
  <li>Use the <b>↑ / ↓</b> arrows to recall previous commands, and
      <code>help</code> for the full list.</li>
</ul>

<h3>Available commands</h3>
<ul>
  <li><code>info</code> — SEG-Y metadata</li>
  <li><code>check</code> — inspects headers (EBCDIC + binary) and detects anomalies</li>
  <li><code>patch-header</code> — fixes <code>dt</code>/<code>ns</code> in the header, in place (r+)</li>
  <li><code>process</code> — runs a headless DSP chain and saves a new SEG-Y</li>
  <li><code>export-image</code> — exports the profile/chain as an image (PNG/PDF…)</li>
  <li><code>batch-export</code> — renders several files/folders to a directory</li>
  <li><code>reproject</code> — reprojects SEG-Y to another CRS</li>
  <li><code>join-chain</code> — reprojects and joins a chain into a single SEG-Y</li>
  <li><code>spectrum</code> — frequency-spectrum figure</li>
  <li><code>navline</code> — exports the navigation track (shp/geojson/csv)</li>
  <li><code>fix</code> — exports FIX marks (shp/geojson/csv)</li>
  <li><code>accel</code> — hardware-acceleration status</li>
  <li><code>pwd</code> / <code>cd &lt;path&gt;</code> — internal working directory</li>
</ul>

<h3>Practical Examples</h3>
<p>Each example is a <b>single line</b> ready to copy and paste (select it and
copy with Ctrl+C). They use a real file from the ANT26 survey:</p>

<p><b>1 · Quick inspection</b> — headers and basic statistics:</p>
<pre>info {_GUIDE_FILE}</pre>

<p><b>2 · Track export</b> — extracts the navigation to a Shapefile:</p>
<pre>navline {_GUIDE_FILE} --format shp --out {_GUIDE_OUT}\\ANT26_track.shp</pre>

<p><b>3 · Reprojection</b> — converts from geographic (WGS84) to UTM 30N:</p>
<pre>reproject {_GUIDE_FILE} --src epsg:4326 --dst epsg:32630 --out-dir {_GUIDE_OUT}</pre>

<p><b>4 · Processing and Plotting</b> — high-quality PDF export
(envelope + AGC + aligned + physical scale + FIX marks):</p>
<pre>export-image {_GUIDE_FILE} --preset envelope --cmap Greys --agc --align --fill-zero --x-scale 2 --ratio 3 --velocity 1500 --x-tick 5 --t-tick 50 --time-ticks 5 --time-fmt full --time-font-size 5.5 --time-align left --fix 5 --fix-color "#cc4444" --fix-bbox-alpha 0.0 --margin-top 20 --margin-bottom 20 --theme print --quality high --format pdf --pdf-page auto --timeit --out {_GUIDE_OUT}\\ANT26_L01.pdf</pre>

<p><b>5 · Header diagnostics</b> — EBCDIC text, binary fields and anomalies:</p>
<pre>check {_GUIDE_FILE}</pre>

<p><b>6 · Metadata correction</b> — sets the sample interval to 50&nbsp;µs
(propagated to all traces) and the sample count. <b>Modifies the file</b>:</p>
<pre>patch-header {_GUIDE_FILE} --dt 50 --ns 2048 --dry-run</pre>

<p><b>7 · Headless DSP chain</b> — bandpass + spectral whitening + AGC, to a new SEG-Y:</p>
<pre>process {_GUIDE_FILE} {_GUIDE_OUT}\\ANT26_proc.seg --pipeline "bandpass(1000,8000),whiten(1000,8000,300),agc(200)"</pre>

<p><b>8 · Batch export</b> — every line in a folder to PDF with the seismc
palette and fixed vertical exaggeration:</p>
<pre>batch-export .\\examples\\_real_in\\ANT26\\SGY --format pdf --cmap seismc --x-scale 2 --ve 15 --out {_GUIDE_OUT}</pre>

<p>Processing runs in the background: the interface never freezes and
progress/messages appear in the console area.</p>
"""


def _guide_html() -> str:
    """Pick the CLI Command Guide body for the ACTIVE app language."""
    return _GUIDE_HTML_ES if language_manager.language == "es" else _GUIDE_HTML_EN


class CliGuideDialog(QDialog):
    """Read-only, language-aware command-reference dialog for the internal CLI."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.resize(720, 540)
        lay = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self._reload()
        lay.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        lay.addWidget(buttons)
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)

    def _reload(self, *_) -> None:
        """Rebuild the guide body in the ACTIVE language."""
        self.browser.setHtml(_guide_html())

    def _retranslate(self, *_) -> None:
        self.setWindowTitle(self.tr("Command Guide"))
        # The body was previously a single hardcoded-Spanish literal that this
        # signal never touched — switching to English left it in Spanish.
        self._reload()
