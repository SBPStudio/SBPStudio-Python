"""
help_dialog.py — Dynamic in-app Help / Documentation panel.

A single source of truth for "how the application works", whose sections mirror
the ACTUAL tab/sub-tab structure of the running app:

    Visualizer ▸ Seismic · Map · Spectrum · Headers      +      Reprojector

The DSP node list and the colormap palette are pulled LIVE from the code
(``NODE_REGISTRY`` and ``core.constants.CMAPS``) so the panel can never drift
out of sync with what the app actually ships. New nodes appear here the moment
they are registered.

Layout: a left navigation list (one entry per documented section) and a right
``QTextBrowser``. Selecting a section scrolls to its HTML anchor. The dialog is
theme-aware and retranslates its chrome.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QListWidget, QSplitter,
    QTextBrowser, QVBoxLayout, QWidget,
)

from ..i18n import language_manager
from ..theme import theme


# ── Section registry (anchor order = navigation order) ──────────────────────────
# Labels are resolved through _section_label() at lookup time (not stored as
# plain strings) so the nav list can be rebuilt in the active language whenever
# language_manager.language_changed fires — see HelpDialog._retranslate_nav.
_SECTION_ANCHORS: Tuple[str, ...] = (
    "overview", "seismic", "dsp", "map", "spectrum", "headers",
    "reprojector", "cli",
)

# Map a main-tab index → the anchor the "How this module works" action jumps to.
_TAB_ANCHOR = {0: "seismic", 1: "reprojector"}


def _section_label(anchor: str) -> str:
    """Localised nav-list label for a section anchor (pylupdate6-extractable —
    see node_i18n.py for why this needs literal translate() calls, not self.tr
    on a variable)."""
    table = {
        "overview":    QCoreApplication.translate("HelpDialog", "Overview"),
        "seismic":     QCoreApplication.translate("HelpDialog", "Visualizer · Seismic"),
        "dsp":         QCoreApplication.translate("HelpDialog", "Visualizer · Filters / DSP"),
        "map":         QCoreApplication.translate("HelpDialog", "Visualizer · Map"),
        "spectrum":    QCoreApplication.translate("HelpDialog", "Visualizer · Spectrum"),
        "headers":     QCoreApplication.translate("HelpDialog", "Visualizer · Headers"),
        "reprojector": QCoreApplication.translate("HelpDialog", "Reprojector"),
        "cli":         QCoreApplication.translate("HelpDialog", "CLI & Batch processing"),
    }
    return table[anchor]


def _current_sections() -> Tuple[Tuple[str, str], ...]:
    """(label, anchor) pairs in nav order, labels resolved in the ACTIVE language."""
    return tuple((_section_label(a), a) for a in _SECTION_ANCHORS)


def _dsp_nodes_html() -> str:
    """Build the DSP-node reference table LIVE from the node registry.

    Node/parameter labels go through node_i18n's tr_node/tr_param — the SAME
    localisation table the Filters/DSP panel itself uses — so this table is
    automatically correct in whichever language is currently installed."""
    try:
        from ..dsp.nodes import NODE_REGISTRY, ParamSpec, ChoiceSpec
        from ..dsp.node_i18n import tr_node, tr_param
    except Exception:
        return "<p><i>(node registry unavailable)</i></p>"

    choice_suffix = QCoreApplication.translate("HelpDialog", "(choice)")
    pre_crop = QCoreApplication.translate("HelpDialog", "PRE-CROP")
    rows: List[str] = []
    for cls in NODE_REGISTRY:
        params = []
        for s in getattr(cls, "SPECS", ()):
            if isinstance(s, ParamSpec):
                unit = f" {s.unit}" if s.unit else ""
                params.append(f"{tr_param(s.label)} [{s.lo:g}–{s.hi:g}{unit}]")
            elif isinstance(s, ChoiceSpec):
                params.append(f"{tr_param(s.label)} {choice_suffix}")
        pre = f"  ·  {pre_crop}" if getattr(cls, "PRECROP", False) else ""
        rows.append(
            f"<tr><td><b>{tr_node(cls.KEY, cls.DISPLAY)}</b><br>"
            f"<code>{cls.KEY}</code>{pre}</td>"
            f"<td>{'; '.join(params) or '—'}</td></tr>")
    node_hdr = QCoreApplication.translate("HelpDialog", "Node")
    params_hdr = QCoreApplication.translate("HelpDialog", "Parameters")
    return ("<table width='100%' cellpadding='4' cellspacing='0' border='1'>"
            f"<tr><th align='left'>{node_hdr}</th><th align='left'>{params_hdr}</th></tr>"
            + "".join(rows) + "</table>")


def _cmaps_html() -> str:
    """List the available colormaps (label → matplotlib name) from the core."""
    try:
        from ...core.constants import CMAPS
    except Exception:
        return ""
    items = "".join(f"<li><b>{label}</b> <code>({name})</code></li>"
                    for label, name in CMAPS.items())
    return f"<ul>{items}</ul>"


def _build_help_html_en() -> str:
    """Assemble the full documentation HTML with per-section anchors (English)."""
    return f"""
<h1>SBP Studio — User Guide</h1>
<p>Processing &amp; visualisation of marine Sub-Bottom Profiler (SBP) and
multichannel SEG-Y data. A PyQt6 interface sits on top of a fully headless,
GUI-independent processing core; everything you do interactively is also
scriptable from the CLI.</p>

<a name="overview"></a><h2>Overview — the application shell</h2>
<ul>
  <li><b>Left sidebar</b> — two managers: <i>Loaded profiles</i> (individual
      SEG-Y files, lazily header-loaded) and <i>Detected chains</i> (contiguous
      lines automatically grouped by geometry, time and sample interval). Select
      one profile <i>or</i> one chain to drive the central viewer; the
      "last selection wins". Multi-select feeds the batch <i>Add to map</i> and
      <i>Export selected in batch…</i> actions.</li>
  <li><b>Central tabs</b> — <b>Visualizer</b> (Seismic · Map · Spectrum ·
      Headers) and <b>Reprojector</b>.</li>
  <li><b>Status bar</b> — task spinner, cancellable progress bar and a
      GPU/CPU acceleration badge.</li>
  <li><b>CLI menu</b> — a built-in console (floating overlay) that runs the same
      headless commands as the external CLI, plus this Command Guide.</li>
</ul>

<a name="seismic"></a><h2>Visualizer ▸ Seismic — display &amp; render engine</h2>
<p>The interactive section is rendered by a custom PyQtGraph view tuned for very
large SEG-Y matrices. It offers a density (variable-density) raster plus an
optional <b>Wiggle / Variable-Area (VA)</b> overlay.</p>
<ul>
  <li><b>Zero-copy Wiggle/VA architecture</b> — the wiggle polyline and the
      variable-area fill are built as a single batched <code>QPainterPath</code>
      via PyQtGraph's <code>arrayToQPath</code>, writing straight into the packed
      vertex buffer with no per-sample Python objects and no intermediate copies.
      One path draws the whole frame, so thousands of traces stay fluid.</li>
  <li><b>Device-coordinate caching</b> — the display parameters read from the Qt
      widgets (gain, clip, colormap, VA toggle…) are cached and only re-read when
      a value actually changes (a dirty flag), so routine pan/zoom frames skip
      the widget round-trips and the DSP-context rebuild entirely.</li>
  <li><b>Performance limits</b> — independent decimation caps keep the GUI thread
      responsive on deep/dense full-fits:
      <ul>
        <li><code>MAX_PREVIEW_COLS / ROWS = 4000</code> — never process more
            samples/traces than a 4K screen can show; row decimation raises the
            effective <code>dt</code> fed to the DSP context so the maths stay
            physically correct (zoom in → strides fall to 1 → exact, full-res).</li>
        <li><code>WIGGLE_MAX_ROWS = 2000</code> — row cap for the wiggle line.</li>
        <li><code>VA_MAX_ROWS = 600</code> and
            <code>VA_TRACE_THRESHOLD = 600</code> — the variable-area fill uses
            its own tighter caps; above the trace threshold the lobes are too thin
            to read and the fill is suppressed so <code>arrayToQPath</code> never
            blocks the UI.</li>
      </ul></li>
  <li><b>Live pipeline preview</b> — pan/zoom and any filter edit re-run the DSP
      pipeline on just the <i>visible</i> ViewBox window (with halos), on a ~300 ms
      debounce, off the GUI thread.</li>
  <li><b>Colormaps</b> — including a diverging <i>Blue-White-Red</i> and a custom
      <i>seismc</i> palette for symmetric (−1…+1) amplitude display:
      {_cmaps_html()}</li>
  <li><b>Render Full</b> re-fits the whole section; <b>Render Viewport HQ</b>
      produces a Matplotlib-quality overlay of the current crop without exporting
      a file.</li>
</ul>

<a name="dsp"></a><h2>Visualizer ▸ Filters / DSP — the node pipeline</h2>
<p>Processing is an <b>ordered, reorderable pipeline of DSP nodes</b>. Each node
is a thin GUI wrapper that holds parameters and delegates to a single
<code>core.apply_*</code> function — the same code the CLI and the export engine
call, so the live preview is bit-identical to the final output. Registered nodes
(live from <code>NODE_REGISTRY</code>):</p>
{_dsp_nodes_html()}
<h3>Spectral Whitening (resolution enhancement)</h3>
<p>Flattens the amplitude spectrum within a chosen band to sharpen vertical
resolution — ideal for chirp / SBP data whose source signature rolls off with
frequency:</p>
<ul>
  <li>Forward <code>rfft</code> of each trace; the amplitude envelope is smoothed
      over a <i>Smooth window</i> (Hz) to estimate the spectral shape.</li>
  <li>Each in-band bin is divided by that smoothed envelope, <b>flattening
      (whitening) the magnitude spectrum</b> between <i>F&nbsp;low</i> and
      <i>F&nbsp;high</i>; out-of-band bins are left untouched.</li>
  <li><b>Phase is preserved exactly</b> — only the magnitude is normalised, so the
      operation is zero-phase and does not shift reflectors. Silent traces are
      passed through unchanged.</li>
  <li>Typical workflow: place it <i>after</i> Bandpass and judge the result on the
      Spectrum tab (the flat in-band plateau is the whitening at work).</li>
</ul>

<a name="map"></a><h2>Visualizer ▸ Map — navigation</h2>
<ul>
  <li><b>Dynamic trackline synchronisation</b> — the map and the seismic ViewBox
      are bi-directionally linked: as you pan/zoom the section a bright segment of
      the navigation track highlights exactly the visible traces (addressed by
      absolute trace index, so it stays locked even across GPS plateaus), and
      clicking the track jumps the section to that trace.</li>
  <li><b>Batch tracks</b> — multi-select profiles/chains in the sidebar and
      <i>Add to map</i> to overlay their navigation tracks (header-only, no trace
      load) as managed layers; GIS vector/raster layers can be added too.</li>
  <li>Selecting a trace on the map or the section also scrolls and highlights its
      row in the Headers table.</li>
</ul>

<a name="spectrum"></a><h2>Visualizer ▸ Spectrum — frequency QC</h2>
<p>An on-demand amplitude-spectrum panel (run via its <i>Generate</i> button,
decoupled from the live preview loop). Reports the peak and centroid frequency
and SNR — use it to set bandpass corners and to confirm spectral whitening has
flattened the in-band response.</p>

<a name="headers"></a><h2>Visualizer ▸ Headers — inspector &amp; editor</h2>
<p>QC and <b>safe editing</b> of the raw SEG-Y headers — essential because
SBP/MCS files often ship with missing or wrong metadata. The overriding design
rule here is <b>zero file corruption</b>: every edit is validated before a byte
is written, and any action that <i>could</i> corrupt the file is either blocked
or simply not offered.</p>
<ul>
  <li><b>Textual header</b> (3200 bytes) — editable, shown as the standard
      40&nbsp;×&nbsp;80 card grid in a monospaced font. An
      <b>Encoding toggle (ASCII · EBCDIC · Latin-1)</b> re-decodes the raw bytes
      live so you can read headers that violate the SEG-Y standard: many marine
      acquisition systems (Kongsberg TOPAS and others) write this block as
      plain ASCII instead of EBCDIC. The textual header is always read with
      plain file I/O — never via segyio's own text accessor, which
      <i>unconditionally</i> runs an EBCDIC-to-ASCII conversion table
      regardless of the file's actual encoding; on an already-ASCII file that
      double-translates clean text into garbage (confirmed by byte-level
      forensic comparison against real ANT26/L001A/MCS7 survey files — all
      three store plain ASCII). The same plain-I/O approach is used on
      <i>write</i>, so saving an edit never silently flips the file from ASCII
      to EBCDIC bytes on disk. NUL padding (a separate, also-seen quirk) is
      normalised to spaces before decoding so the text stays legible under any
      codec. <i>Latin-1</i> is a never-fails fallback for extended-ASCII files.
      Switching encodings re-reads the on-disk bytes only (never writes), and
      warns before discarding unsaved text edits.</li>
  <li><b>Binary header overrides</b> — the sample interval (<code>dt</code>) is
      editable. <b><code>ns</code> (samples/trace) is shown read-only by
      design</b>: changing the declared trace length without physically
      resizing every trace's data block would misalign every trace boundary for
      any reader, so it is intentionally not editable here.</li>
  <li><b>Trace Header Calculator</b> — SeiSee-style bulk edits over the
      per-trace header fields via a single assignment, e.g.
      <code>CDP = TraceNumber * 2</code>. Expressions are parsed by a
      <b>sandboxed evaluator (never <code>eval()</code>)</b> backed by the NumPy
      header arrays. The result is validated against the target field's SEG-Y
      integer type <i>before</i> the array is touched: a floating-point result
      (e.g. from <code>/</code>) or a value that would overflow the field's
      byte width (int16/int32) is <b>blocked with an error</b> rather than
      silently corrupting the file. <i>Apply</i> stages the change in memory,
      <i>Undo</i> reverts the last staged change, and <i>Help</i> lists the
      available variables (trace-header field names) and functions.</li>
  <li><b>The per-trace header table</b> is NumPy-backed so 50&nbsp;000+ traces
      scroll without allocating a widget per cell. The textual header sits in
      the top pane of a single splitter; the binary overrides, calculator and
      table share the bottom pane.</li>
  <li><b>Apply &amp; Save to File</b> — writes all pending edits (text, dt,
      calculator changes) <b>in place</b> (segyio <code>r+</code>, no temporary
      copy) after a confirmation dialog (the operation is irreversible).
      Changing <code>dt</code> is also <b>mass-propagated to every trace
      header</b> (<code>TRACE_SAMPLE_INTERVAL</code>), so readers that trust
      per-trace dt see the corrected value too.</li>
  <li><b>Save As Copy…</b> — duplicates the physical file (<code>shutil.copy2</code>)
      and applies your edits to the <b>new</b> file via the same safe patchers,
      leaving the original untouched. Pick a name/location in the dialog (it
      defaults to <code>&lt;name&gt;_copy</code> next to the original); the app
      then hot-reloads onto the copy so you are viewing the new file.</li>
  <li><b>Hot-reload</b> — after a successful save (in place or as a copy) the
      profile is reloaded and pushed back into the app state, so the Seismic
      view's vertical (time) scale and the DSP context refresh <i>immediately</i>
      — no re-open needed.</li>
</ul>

<a name="reprojector"></a><h2>Reprojector — CRS transforms &amp; joins</h2>
<ul>
  <li>Reproject one or many SEG-Y files to a new CRS (EPSG presets or custom),
      writing new files with corrected coordinate headers and scalars.</li>
  <li>Join a chain of contiguous lines into a single SEG-Y, with or without
      reprojection (a pure header-faithful copy when source = destination).</li>
  <li>Export navigation tracks and FIX-point marks to Shapefile / GeoJSON / CSV.</li>
</ul>

<a name="cli"></a><h2>CLI &amp; Batch processing</h2>
<p>Every core capability is available headlessly — from the built-in console
(<i>CLI ▸ Activate Console</i>) or via
<code>python -m sbp_studio.cli.main &lt;command&gt;</code>:</p>
<ul>
  <li><code>info</code> / <code>check</code> — metadata; <code>check</code> also
      dumps the EBCDIC text, binary fields, stats and a heuristic <b>anomaly
      scan</b> (zero <code>dt</code>, missing CRS, dead traces, purged
      duplicates…).</li>
  <li><code>patch-header FILE --dt µs [--text FILE]</code> — the headless
      counterpart of the Headers editor: safe <code>r+</code> patching with the
      same <code>dt</code> mass-propagation to all traces (add
      <code>--dry-run</code> to preview). <code>ns</code> is intentionally not
      patchable — see the Headers section above.</li>
  <li><code>process IN OUT --pipeline "bandpass(1000,8000),whiten(1000,8000,300),agc(200)"</code>
      — runs the DSP pipeline end-to-end and writes a new SEG-Y (all geometry and
      headers preserved). Ops: bandpass, whiten, decon, swell, water_mute, tvg,
      agc, preset, align.</li>
  <li><code>export-image</code> / <code>batch-export DIR --format pdf --cmap
      seismc --x-scale 2 --ve 15 --out DIR</code> — the vectorised
      max-abs-pooling render engine with custom colormaps, fixed (length-
      independent) vertical exaggeration, vector interpolation and an automatic
      RAM-safety down-scale; <code>batch-export</code> expands folders and renders
      every line into one output directory.</li>
  <li><code>reproject</code> · <code>join-chain</code> · <code>spectrum</code> ·
      <code>navline</code> · <code>fix</code> · <code>accel</code>.</li>
</ul>
<p>See <i>CLI ▸ Command Guide</i> for copy-paste examples.</p>
"""


def _build_help_html_es() -> str:
    """Assemble the full documentation HTML with per-section anchors (Spanish)."""
    return f"""
<h1>SBP Studio — Guía del Usuario</h1>
<p>Procesamiento y visualización de datos marinos de Perfilador de Subfondo (SBP)
y SEG-Y multicanal. Una interfaz PyQt6 se apoya sobre un núcleo de procesamiento
totalmente headless e independiente de la interfaz gráfica; todo lo que haces de
forma interactiva también se puede automatizar desde la CLI.</p>

<a name="overview"></a><h2>Resumen — la estructura de la aplicación</h2>
<ul>
  <li><b>Barra lateral izquierda</b> — dos gestores: <i>Perfiles cargados</i>
      (archivos SEG-Y individuales, cargados de forma diferida solo con
      cabecera) y <i>Cadenas detectadas</i> (líneas contiguas agrupadas
      automáticamente por geometría, tiempo e intervalo de muestreo). Selecciona
      un perfil <i>o</i> una cadena para controlar el visor central; "la última
      selección gana". La selección múltiple alimenta las acciones por lotes
      <i>Añadir al mapa</i> y <i>Exportar selección por lotes…</i>.</li>
  <li><b>Pestañas centrales</b> — <b>Visualizador</b> (Sísmica · Mapa · Espectro
      · Cabeceras) y <b>Reproyector</b>.</li>
  <li><b>Barra de estado</b> — indicador de tareas, barra de progreso cancelable
      e insignia de aceleración GPU/CPU.</li>
  <li><b>Menú CLI</b> — una consola integrada (superposición flotante) que
      ejecuta los mismos comandos headless que la CLI externa, además de esta
      Guía de Comandos.</li>
</ul>

<a name="seismic"></a><h2>Visualizador ▸ Sísmica — motor de visualización y render</h2>
<p>La sección interactiva se renderiza mediante una vista personalizada de
PyQtGraph optimizada para matrices SEG-Y muy grandes. Ofrece un raster de
densidad (densidad variable) más una superposición opcional de <b>Wiggle / Área
Variable (VA)</b>.</p>
<ul>
  <li><b>Arquitectura Wiggle/VA sin copias</b> — la polilínea wiggle y el
      relleno de área variable se construyen como un único <code>QPainterPath</code>
      por lotes mediante <code>arrayToQPath</code> de PyQtGraph, escribiendo
      directamente en el búfer de vértices empaquetado sin objetos Python por
      muestra ni copias intermedias. Un solo path dibuja todo el fotograma, de
      modo que miles de trazas se mantienen fluidas.</li>
  <li><b>Caché en coordenadas de dispositivo</b> — los parámetros de
      visualización leídos de los widgets de Qt (ganancia, clip, paleta de
      colores, alternancia de VA…) se cachean y solo se vuelven a leer cuando un
      valor realmente cambia (un indicador "dirty"), de modo que los fotogramas
      habituales de paneo/zoom se saltan por completo las idas y vueltas a los
      widgets y la reconstrucción del contexto DSP.</li>
  <li><b>Límites de rendimiento</b> — topes de diezmado independientes
      mantienen el hilo de la interfaz responsivo en ajustes completos
      profundos/densos:
      <ul>
        <li><code>MAX_PREVIEW_COLS / ROWS = 4000</code> — nunca procesa más
            muestras/trazas de las que puede mostrar una pantalla 4K; el
            diezmado de filas eleva el <code>dt</code> efectivo que se pasa al
            contexto DSP para que las matemáticas sigan siendo físicamente
            correctas (al acercar el zoom → los pasos caen a 1 → resolución
            exacta y completa).</li>
        <li><code>WIGGLE_MAX_ROWS = 2000</code> — tope de filas para la línea
            wiggle.</li>
        <li><code>VA_MAX_ROWS = 600</code> y <code>VA_TRACE_THRESHOLD = 600</code>
            — el relleno de área variable usa sus propios topes más estrictos;
            por encima del umbral de trazas los lóbulos son demasiado finos
            para leerse y el relleno se suprime para que
            <code>arrayToQPath</code> nunca bloquee la interfaz.</li>
      </ul></li>
  <li><b>Vista previa en vivo del pipeline</b> — el paneo/zoom y cualquier
      edición de filtro vuelven a ejecutar el pipeline DSP solo en la ventana
      visible del ViewBox (con halos), con un debounce de ~300 ms, fuera del
      hilo de la interfaz.</li>
  <li><b>Paletas de color</b> — incluyendo una divergente <i>Azul-Blanco-Rojo</i>
      y una paleta personalizada <i>seismc</i> para visualización
      simétrica de amplitud (−1…+1):
      {_cmaps_html()}</li>
  <li><b>Renderizar Completo</b> reajusta toda la sección; <b>Renderizar
      Viewport HQ</b> produce una superposición de calidad Matplotlib del
      recorte actual sin exportar un archivo.</li>
</ul>

<a name="dsp"></a><h2>Visualizador ▸ Filtros / DSP — el pipeline de nodos</h2>
<p>El procesamiento es un <b>pipeline de nodos DSP ordenado y reordenable</b>.
Cada nodo es un envoltorio ligero de la interfaz que contiene parámetros y
delega en una única función <code>core.apply_*</code> — el mismo código que
usan la CLI y el motor de exportación, de modo que la vista previa en vivo es
idéntica bit a bit a la salida final. Nodos registrados (en vivo desde
<code>NODE_REGISTRY</code>):</p>
{_dsp_nodes_html()}
<h3>Blanqueo Espectral (mejora de resolución)</h3>
<p>Aplana el espectro de amplitud dentro de una banda elegida para agudizar la
resolución vertical — ideal para datos chirp / SBP cuya firma de la fuente cae
con la frecuencia:</p>
<ul>
  <li><code>rfft</code> directa de cada traza; la envolvente de amplitud se
      suaviza en una <i>Ventana de suavizado</i> (Hz) para estimar la forma
      espectral.</li>
  <li>Cada bin dentro de banda se divide por esa envolvente suavizada,
      <b>aplanando (blanqueando) el espectro de magnitud</b> entre
      <i>F&nbsp;mínima</i> y <i>F&nbsp;máxima</i>; los bins fuera de banda
      quedan intactos.</li>
  <li><b>La fase se preserva exactamente</b> — solo se normaliza la magnitud,
      por lo que la operación es de fase cero y no desplaza los reflectores.
      Las trazas silenciosas pasan sin cambios.</li>
  <li>Flujo de trabajo típico: colócalo <i>después de</i> Bandpass y evalúa el
      resultado en la pestaña Espectro (la meseta plana dentro de banda es el
      blanqueo en acción).</li>
</ul>

<a name="map"></a><h2>Visualizador ▸ Mapa — navegación</h2>
<ul>
  <li><b>Sincronización dinámica de la traza de navegación</b> — el mapa y el
      ViewBox sísmico están vinculados de forma bidireccional: al desplazar o
      ajustar el zoom de la sección, un segmento brillante de la traza de
      navegación resalta exactamente las trazas visibles (referenciado por
      índice de traza absoluto, de modo que se mantiene fijo incluso a través
      de mesetas de GPS), y al hacer clic en la traza la sección salta a esa
      traza.</li>
  <li><b>Trazas por lotes</b> — selecciona varios perfiles/cadenas en la barra
      lateral y usa <i>Añadir al mapa</i> para superponer sus trazas de
      navegación (solo cabecera, sin cargar trazas) como capas gestionadas;
      también se pueden añadir capas GIS vectoriales/raster.</li>
  <li>Seleccionar una traza en el mapa o en la sección también desplaza y
      resalta su fila en la tabla de Cabeceras.</li>
</ul>

<a name="spectrum"></a><h2>Visualizador ▸ Espectro — control de calidad de frecuencia</h2>
<p>Un panel de espectro de amplitud a demanda (se ejecuta con su botón
<i>Generar</i>, desacoplado del bucle de vista previa en vivo). Informa de la
frecuencia de pico y centroide y la SNR — úsalo para fijar los cortes del
bandpass y confirmar que el blanqueo espectral ha aplanado la respuesta dentro
de banda.</p>

<a name="headers"></a><h2>Visualizador ▸ Cabeceras — inspector y editor</h2>
<p>Control de calidad y <b>edición segura</b> de las cabeceras SEG-Y crudas —
esencial porque los archivos SBP/MCS a menudo llegan con metadatos ausentes o
incorrectos. La regla de diseño dominante aquí es <b>cero corrupción de
archivos</b>: cada edición se valida antes de escribir un solo byte, y
cualquier acción que <i>pudiera</i> corromper el archivo se bloquea o
simplemente no se ofrece.</p>
<ul>
  <li><b>Cabecera textual</b> (3200 bytes) — editable, mostrada como la
      cuadrícula estándar de tarjetas de 40&nbsp;×&nbsp;80 en una fuente
      monoespaciada. Un <b>alternador de codificación (ASCII · EBCDIC ·
      Latin-1)</b> redecodifica los bytes crudos en vivo para que puedas leer
      cabeceras que incumplen el estándar SEG-Y: muchos sistemas de
      adquisición marina (Kongsberg TOPAS y otros) escriben este bloque como
      ASCII puro en lugar de EBCDIC. La cabecera textual siempre se lee con
      E/S de archivo plana — nunca mediante el accesor de texto propio de
      segyio, que ejecuta <i>incondicionalmente</i> una tabla de conversión
      EBCDIC-a-ASCII sin importar la codificación real del archivo; en un
      archivo ya en ASCII eso traduce dos veces el texto limpio y lo convierte
      en basura (confirmado mediante comparación forense a nivel de byte
      contra archivos reales de las campañas ANT26/L001A/MCS7 — los tres
      almacenan ASCII puro). Se usa el mismo enfoque de E/S plana al
      <i>escribir</i>, de modo que guardar una edición nunca cambia
      silenciosamente el archivo de ASCII a bytes EBCDIC en disco. El relleno
      con NUL (otra peculiaridad también observada, distinta) se normaliza a
      espacios antes de decodificar para que el texto siga siendo legible con
      cualquier códec. <i>Latin-1</i> es un respaldo que nunca falla para
      archivos ASCII extendido. Cambiar de codificación vuelve a leer solo los
      bytes en disco (nunca escribe), y avisa antes de descartar ediciones de
      texto sin guardar.</li>
  <li><b>Anulaciones de la cabecera binaria</b> — el intervalo de muestreo
      (<code>dt</code>) es editable. <b><code>ns</code> (muestras/traza) se
      muestra de solo lectura por diseño</b>: cambiar la longitud declarada de
      la traza sin redimensionar físicamente el bloque de datos de cada traza
      desalinearía cada límite de traza para cualquier lector, por lo que
      intencionadamente no es editable aquí.</li>
  <li><b>Calculadora de Cabeceras de Traza</b> — ediciones masivas al estilo
      SeiSee sobre los campos de cabecera por traza mediante una única
      asignación, p.&nbsp;ej. <code>CDP = TraceNumber * 2</code>. Las
      expresiones se analizan mediante un <b>evaluador en sandbox (nunca
      <code>eval()</code>)</b> respaldado por los arrays NumPy de cabeceras. El
      resultado se valida contra el tipo entero SEG-Y del campo destino
      <i>antes</i> de tocar el array: un resultado de coma flotante (p.&nbsp;ej.
      de <code>/</code>) o un valor que desbordaría el ancho de bytes del campo
      (int16/int32) se <b>bloquea con un error</b> en lugar de corromper
      silenciosamente el archivo. <i>Aplicar</i> deja el cambio preparado en
      memoria, <i>Deshacer</i> revierte el último cambio preparado, y
      <i>Ayuda</i> lista las variables disponibles (nombres de campos de
      cabecera de traza) y funciones.</li>
  <li><b>La tabla de cabeceras por traza</b> está respaldada por NumPy, de modo
      que 50&nbsp;000+ trazas se desplazan sin asignar un widget por celda. La
      cabecera textual ocupa el panel superior de un único divisor; las
      anulaciones binarias, la calculadora y la tabla comparten el panel
      inferior.</li>
  <li><b>Aplicar y Guardar en el Archivo</b> — escribe todas las ediciones
      pendientes (texto, <code>dt</code>, cambios de la calculadora) <b>in
      situ</b> (segyio <code>r+</code>, sin copia temporal) tras un diálogo de
      confirmación (la operación es irreversible). Cambiar <code>dt</code>
      también se <b>propaga masivamente a cada cabecera de traza</b>
      (<code>TRACE_SAMPLE_INTERVAL</code>), de modo que los lectores que
      confían en el <code>dt</code> por traza también ven el valor
      corregido.</li>
  <li><b>Guardar Como Copia…</b> — duplica el archivo físico
      (<code>shutil.copy2</code>) y aplica tus ediciones al archivo
      <b>nuevo</b> mediante los mismos parches seguros, dejando el original
      intacto. Elige un nombre/ubicación en el diálogo (por defecto
      <code>&lt;nombre&gt;_copy</code> junto al original); la aplicación
      entonces recarga en caliente sobre la copia, de modo que estás viendo el
      archivo nuevo.</li>
  <li><b>Recarga en caliente</b> — tras un guardado exitoso (in situ o como
      copia) el perfil se recarga y se reintroduce en el estado de la
      aplicación, de modo que la escala vertical (tiempo) de la vista Sísmica y
      el contexto DSP se actualizan <i>de inmediato</i> — sin necesidad de
      volver a abrir.</li>
</ul>

<a name="reprojector"></a><h2>Reproyector — transformaciones de CRS y uniones</h2>
<ul>
  <li>Reproyecta uno o varios archivos SEG-Y a un nuevo CRS (preajustes EPSG o
      personalizado), escribiendo archivos nuevos con cabeceras de coordenadas
      y escalares corregidos.</li>
  <li>Une una cadena de líneas contiguas en un único SEG-Y, con o sin
      reproyección (una copia fiel a la cabecera cuando origen = destino).</li>
  <li>Exporta trazas de navegación y marcas de punto FIX a Shapefile / GeoJSON
      / CSV.</li>
</ul>

<a name="cli"></a><h2>CLI y Procesamiento por Lotes</h2>
<p>Todas las capacidades del núcleo están disponibles de forma headless —
desde la consola integrada (<i>CLI ▸ Activar Consola</i>) o mediante
<code>python -m sbp_studio.cli.main &lt;comando&gt;</code>:</p>
<ul>
  <li><code>info</code> / <code>check</code> — metadatos; <code>check</code>
      también vuelca el texto EBCDIC, los campos binarios, estadísticas y un
      <b>escaneo heurístico de anomalías</b> (<code>dt</code> cero, CRS
      ausente, trazas muertas, duplicados purgados…).</li>
  <li><code>patch-header ARCHIVO --dt µs [--text ARCHIVO]</code> — el
      equivalente headless del editor de Cabeceras: parcheo seguro
      <code>r+</code> con la misma propagación masiva de <code>dt</code> a
      todas las trazas (añade <code>--dry-run</code> para previsualizar).
      <code>ns</code> intencionadamente no es parcheable — ver la sección
      Cabeceras arriba.</li>
  <li><code>process ENTRADA SALIDA --pipeline "bandpass(1000,8000),whiten(1000,8000,300),agc(200)"</code>
      — ejecuta el pipeline DSP de extremo a extremo y escribe un nuevo SEG-Y
      (toda la geometría y cabeceras se preservan). Operaciones: bandpass,
      whiten, decon, swell, water_mute, tvg, agc, preset, align.</li>
  <li><code>export-image</code> / <code>batch-export DIR --format pdf --cmap
      seismc --x-scale 2 --ve 15 --out DIR</code> — el motor de render
      vectorizado con max-abs-pooling, con paletas de color personalizadas,
      exageración vertical fija (independiente de la longitud), interpolación
      vectorial y una reducción automática de seguridad de RAM;
      <code>batch-export</code> expande carpetas y renderiza cada línea en un
      único directorio de salida.</li>
  <li><code>reproject</code> · <code>join-chain</code> · <code>spectrum</code> ·
      <code>navline</code> · <code>fix</code> · <code>accel</code>.</li>
</ul>
<p>Consulta <i>CLI ▸ Guía de Comandos</i> para ejemplos listos para copiar y
pegar.</p>
"""


def _help_html() -> str:
    """Pick the help body for the ACTIVE app language (extend here for more)."""
    return _build_help_html_es() if language_manager.language == "es" else _build_help_html_en()


class HelpDialog(QDialog):
    """Navigable, theme-aware documentation panel matching the app's structure."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.resize(900, 680)

        root = QVBoxLayout(self)
        split = QSplitter()

        self.nav = QListWidget()
        for label, _anchor in _current_sections():
            self.nav.addItem(label)
        self.nav.currentRowChanged.connect(self._on_nav)
        self.nav.setMaximumWidth(230)
        split.addWidget(self.nav)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        split.addWidget(self.browser)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([220, 680])
        root.addWidget(split, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._reload()
        self.nav.setCurrentRow(0)
        self._retranslate()
        language_manager.language_changed.connect(self._retranslate)
        theme.theme_changed.connect(self._reload)

    # ── Public API ────────────────────────────────────────────────────────────

    def show_section(self, anchor: str) -> None:
        """Jump to a documented section by its anchor (e.g. 'headers')."""
        for row, (_label, a) in enumerate(_current_sections()):
            if a == anchor:
                self.nav.setCurrentRow(row)
                return
        self.browser.scrollToAnchor(anchor)

    def show_for_tab(self, tab_index: int) -> None:
        """Open at the section matching a main-tab index (How this module works)."""
        self.show_section(_TAB_ANCHOR.get(tab_index, "overview"))

    # ── Internals ──────────────────────────────────────────────────────────────

    def _on_nav(self, row: int) -> None:
        sections = _current_sections()
        if 0 <= row < len(sections):
            self.browser.scrollToAnchor(sections[row][1])

    def _reload(self, *_) -> None:
        """Rebuild the HTML in the ACTIVE language (also re-applies theme colours)."""
        bg, fg = theme.color("panel"), theme.color("text")
        link = theme.color("highlight")
        css = (f"<style>body{{background:{bg};color:{fg};}}"
               f"a{{color:{link};}} code{{color:{theme.color('bright')};}}"
               f"th,td{{border-color:{theme.color('sub')};}}</style>")
        self.browser.setHtml(css + _help_html())

    def _retranslate_nav(self) -> None:
        """Rebuild the nav-list labels in the active language, preserving selection."""
        cur = self.nav.currentRow()
        self.nav.blockSignals(True)
        self.nav.clear()
        for label, _anchor in _current_sections():
            self.nav.addItem(label)
        self.nav.blockSignals(False)
        self.nav.setCurrentRow(cur if cur >= 0 else 0)

    def _retranslate(self, *_) -> None:
        self.setWindowTitle(self.tr("Documentation"))
        # A language switch must also rebuild the nav labels AND the HTML body —
        # both were previously hardcoded English literals that this signal never
        # touched, so switching to Spanish left the dialog's content in English.
        self._retranslate_nav()
        self._reload()
