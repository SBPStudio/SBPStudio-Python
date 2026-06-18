# GUI_REPORT — SBP Studio Desktop Interface

> Last updated: 2026-06-18
> Stack: **PyQt6** (Qt 6) + **PyQtGraph** (OpenGL-accelerated) on top of the headless `sbp_studio` core.
> Entry point: `python -m sbp_studio.gui` (or `applications/SBPStudio_GUI.py`).
> See [`CORE_REPORT.md`](CORE_REPORT.md) for the computation/CLI layer this GUI drives.

---

## 1. Design principles

| Principle | How it is enforced |
|-----------|--------------------|
| **Strict Core/GUI separation** | `sbp_studio.core` / `sbp_studio.viz` never import Qt. The GUI is a pure *presentation* layer; it calls the same functions as the CLI. |
| **Export ≠ preview** | The interactive view is a fast, decimated PyQtGraph raster. **Export always renders at 100 % native resolution** through the core matplotlib renderer (`viz/render.py`) — the on-screen decimation never leaks into a file. |
| **Off-thread compute** | Every heavy operation (load, chain assembly, reproject, export) runs on a `CoreWorker` (`QThread`) with a `CancelToken`; the UI thread never blocks. |
| **Memory-bounded** | Profiles load lazily (header-only stubs); trace matrices are loaded on demand and capped by an LRU (`MAX_HOT_PROFILES = 3`). Heavy exports honour a RAM budget. |
| **Qt-native i18n** | All user text is `self.tr(...)` / `QCoreApplication.translate(...)`; Spanish lives in `gui/translations/sbp_studio_es.ts` (parsed directly by a custom `TsTranslator`). |

---

## 2. Module map (`sbp_studio/gui/`)

| Module | Role |
|--------|------|
| `app.py` | Bootstrap: builds the `QApplication`, installs theme + translator, shows `MainWindow`, runs the event loop. Import-safe (tests build the window directly). |
| `__main__.py` | `python -m sbp_studio.gui` → `app.main()`. |
| `main_window.py` | Top-level window: menu bar (Language / Theme / CLI / Help), brand bar (**`◈  SBP Studio`**), the profile/chain sidebar, the tab stack and a status bar with a task spinner + progress bar. |
| `state.py` | `AppState` — observable store of loaded profiles/chains + active selection; owns the lazy-load + LRU eviction policy. Emits `profiles_changed` / `active_profile_changed` / `active_chain_changed`. |
| `theme.py` | Dark / light QSS themes + palette; `theme.color(key)`; live theme switching. |
| `i18n.py` | `LanguageManager` + `TsTranslator` (loads `.qm` if present, else parses the `.ts` XML directly). ES is the default display language. |
| `workers/base.py` | `CoreWorker(QThread)` + `run_task(job, on_success, msg)` wiring (progress, success, error, cancel). |

### Tabs (`gui/tabs/`)

| Tab | Class | Purpose |
|-----|-------|---------|
| Visualizer | `VisualizerTab` | **Unified** live viewer for *both* an individual profile **and** a detected chain. The sidebar's two lists (Loaded profiles / Detected chains) both feed this one tab. It holds a single `self._handler` (a `SourceHandler` strategy) reassigned on selection — there is **no** boolean `_mode` / `_is_chain` dispatch. The selection slots set the handler and delegate the state→preview transition to it; `_active_object()` / `_export_basename()` resolve through it. "Last selection wins" falls out of which slot fired last. |
| Reprojector | `ReprojectorTab` | CRS reprojection + navline/FIX geometry export (128-preset CRS catalog). Subclasses `QWidget` directly (not `SubTabbedTab`). |
| — | `SubTabbedTab` | **100 % type-agnostic** shared base: the controls column + a four-view sub-notebook (Profile / Spectrum / Map / Headers) and the full Render/Export dispatch. It never references `SegyProfile` / `ProfileChain` — every divergent operation (transition, load, render, basename, batch output path, post-batch release) goes through the `SourceHandler` interface. Map/Spectrum/Header signals are wired once here and operate on `_active_object()`. `export_batch(items, cfg, handler)` takes the **handler** for the batch's kind (resolved by the caller from the originating sidebar list), so a batch exports correctly regardless of what is on screen. |
| — | `_handlers.py` | The **`SourceHandler`** strategy interface (ABC) + `ProfileHandler` / `ChainHandler`. Each concrete handler encapsulates the one kind's divergent behaviour: `active_object()` (Adapter), `on_selected()`, `load_full()`, `render_figure()`, `basename()`, `source_path()`, `release_after_batch()`. |
| — | `_render.py` | Shared figure-size maths (`compute_figsize`, `figsize_for_scale`, `effective_aspect`, `effective_export_dpi`, `dpi_for_budget`) + the decimated display-buffer builder. |

> **Source-kind architecture (Disciplined Strategy + Adapter):** the profile and chain pipelines are **fundamentally decoupled** behind the `SourceHandler` interface — a profile is *never* wrapped as a "chain of length 1", so chain-level logic (stitching, spatial union) can never touch a single file. `VisualizerTab` selects the strategy; `SubTabbedTab` runs it knowing only the interface. See §10 for the maintenance contract.

### Components (`gui/components/`)

| Component | Role |
|-----------|------|
| `seismic_view.py` | `SeismicView` — the PyQtGraph section (`ImageItem` + LUT, `useOpenGL`, max-abs row pooling). Bidirectional sync with map/headers; the **HQ overlay** lives here. |
| `map_view.py` | `MapView` — navigation track + GIS overlays; click-to-jump synced to the section. |
| `spectrum_view.py` | `SpectrumView` — on-demand advanced frequency analysis (PyQtGraph). |
| `header_view.py` | `HeaderView` — NumPy-backed trace-header table (handles 50k+ traces) + the **duplicate-timestamp cleanup banner**. |
| `processing_controls.py` | The left controls column: palette/clip/FIX, geometry, the **scale/proportion** controls, the resolution control, and the Render/Export buttons. |
| `pipeline_panel.py` | The reorderable DSP node list (add / remove / edit / drag). |
| `export_dialog.py` | High-quality export options (format, DPI, theme, grid, fonts, margins, memory budget, live size/DPI estimate). |
| `cli_console.py` | In-app command console wrapping the real `cli.commands` (live streamed output) + a Spanish command guide. |
| `placeholder.py` | "Select a profile…" / "Loading…" placeholder views. |

### DSP node pipeline (`gui/dsp/`)

`pipeline.py` + `nodes.py` implement a **reorderable** processing chain applied to the
visible ViewBox window for the live preview, and to the full-resolution matrix for export.

Node `KEY`s: `decon`, `bandpass`, `preset`, `tvg`, `agc`, `water_mute`, `swell`.
`preview.py` (`PreviewController`) runs the chain on the cropped/decimated visible window,
debounced (~300 ms settle) so dragging never triggers a recompute mid-gesture; water-mute
and other pre-crop nodes run once on the full aligned array and are cached.

---

## 3. The live preview loop

1. The user pans/zooms → `SeismicView` emits `view_range_changed` (settle-debounced).
2. `PreviewController._refresh` extracts the visible window (`extract_visible_window`,
   column + max-abs-pooled row decimation, capped by `MAX_PREVIEW_COLS/ROWS = 4000`,
   scaled live by the **Pixels/trace** control), runs the DSP chain on that small bbox at
   the effective (decimated) sample interval, and pushes a float32 raster to the `ImageItem`
   (colourised on the GPU via a LUT).
3. The section, map segment and header row stay in lock-step by **absolute trace index**
   (`np.searchsorted` on the per-trace distance axis — plateau-proof).

Interactive sharpness is inherent to PyQtGraph (native-pixel `ImageItem`); the **Pixels/trace**
control raises the preview column cap, and **Render Viewport HQ** (below) gives an
export-quality look on demand.

---

## 4. Scale / proportion controls (`processing_controls.py`)

Three mutually-exclusive export modes (radio buttons) drive both the live aspect and the export:

| Mode | Inline editor | Meaning |
|------|---------------|---------|
| **Lock aspect ratio (W:H)** | horizontal-deformation slider + spin (synced, 0.5–20.0) | Constant figure shape; VE floats with length. |
| **Lock vertical exaggeration** | VE spin (default 67) | Constant VE → geologically comparable; long lines become wide/short. |
| **Hybrid (VE, capped)** | max-aspect spin (default 5) | Locks VE but caps the aspect so extreme lines don't become "noodles". |

Plus: a **Pixels/trace (resolution)** control (export width density + live viewer detail),
a **↺ Reset aspect settings** button, and a live **"Export ≈ N DPI · M Mpx"** readout computed
from the active line's dimensions (so the user sees the output resolution before rendering).
The live PyQtGraph aspect updates per mode via `effective_aspect` — VE/hybrid re-shape per line.

---

## 5. Rendering & export

| Action | What it does |
|--------|--------------|
| **⟳ Render Full** | Re-fits and re-renders the whole line in the live view. |
| **🔍 Render Viewport HQ** | Renders the currently-visible ViewBox crop at export quality (smooth bilinear colourise) and lays it over the live view as an **ephemeral `pg.ImageItem` overlay**, mapped 1:1 to the crop's km × ms bounds (`setRect`). Any pan/zoom auto-removes it (bound to `sigRangeChanged`), reverting to the fast view. No file written. |
| **💾 Export image** | Opens `ExportDialog`, then renders the **full-resolution** matrix through `viz/render.py` (matplotlib) in a background worker. Shared `_render_export_figure` is used by single **and** batch export so quality can never drift. |

Export safeguards:
- **`effective_export_dpi`** raises the render DPI just enough that `figsize × dpi ≥ (n_traces, ns)` so the embedded raster is never silently decimated (anti-blur), capped at 2400.
- **`dpi_for_budget`** then caps the DPI to the memory budget (parity with the CLI) — this is the fix for the GUI export **hang** (a long/deep line at 1200 DPI could balloon to gigapixels and stall the WYSIWYG draw loop). The export dialog shows the forced-safe DPI live and locks the field to the budget-safe maximum.
- **Free-RAM check**: a `QMessageBox` warns before rendering if the chosen memory budget exceeds the machine's free RAM (`psutil`).
- A **WYSIWYG aspect-fit loop** grows the figure so the data box hits the mode's effective aspect at full size.

`ExportDialog` fields: format (PDF/PNG/TIFF/SVG), DPI, theme, X/Y grid spacing + grid on/off,
axis font size, time-label font size, top/bottom margins, **memory budget (GB)**, red
file-boundary lines, and the live output-size/DPI status.

---

## 6. Memory & threading model

- **Lazy load:** `Add profiles` reads headers only (`load_profile(load_traces=False)`); trace
  matrices load on first selection/render.
- **LRU cap:** `AppState` keeps at most `MAX_HOT_PROFILES = 3` hot trace matrices, evicting the
  oldest non-active back to header stubs.
- **Lazy chains:** `ProfileChain` assembles its stitched matrix on demand (`load_chain_traces`),
  re-reading evicted segments from disk so it survives eviction.
- **Workers:** all I/O / DSP / export run on `CoreWorker(QThread)`; `plt.close(fig)` after every
  export job (single and batch) to release Matplotlib memory.

---

## 7. Internationalisation

Source strings are **English** literals via `self.tr(...)`. Spanish translations live in
`gui/translations/sbp_studio_es.ts` (Qt Linguist XML), regenerated with
`python tools/update_translations.py` (a `pylupdate6` merge over `gui/**/*.py` that preserves
existing translations, adds new strings as `unfinished`, and marks removed ones `vanished`).
ES is loaded by default; EN = no translator. No `.qm` is required — `TsTranslator` parses the
`.ts` directly (and prefers a `.qm` if one is ever dropped in). Gotcha: literals defined in a
base class but called from a subclass instance must use the explicit
`QCoreApplication.translate("Context", …)` form so PyQt resolves the right context.

---

## 8. Running & testing

```bash
# Launch the desktop app
python -m sbp_studio.gui
#   or
python applications/SBPStudio_GUI.py
```

GUI tests build the window directly (no `app.main()`), so they need a display/Qt platform.
On a headless CI box use an offscreen platform:

```bash
QT_QPA_PLATFORM=offscreen pytest tests/ -k "gui or chain_lazy or dsp or batch"
```

The headless **core + CLI** suites run anywhere; the **PyQt6/QApplication** suites require a
display (or `offscreen`) and are exercised on the developer desktop.

---

## 9. Dependencies (GUI-only, on top of the core)

| Library | Why |
|---------|-----|
| `pyqt` (PyQt6, Qt 6) | Widgets, signals/slots, the desktop application shell |
| `pyqtgraph>=0.13` | OpenGL-accelerated interactive seismic / map / spectrum views |
| `psutil>=5.9` | Free-RAM probe for the export memory-budget guard |

Installed by `env/environment.yml` (see `env/setup_env.sh` / `env/setup_env.ps1`). The core and
CLI do **not** require any of these.

---

## 10. Source-kind architecture (`SourceHandler`) & maintenance

### The pattern — Disciplined Strategy + Adapter

The unified `VisualizerTab` renders either a `SegyProfile` or a `ProfileChain` in one
UI space. The two kinds diverge in only a handful of operations; everything else (map,
spectrum, headers, DSP preview, scale, DPI) is already type-agnostic and driven by
duck-typed attributes. Those divergences are isolated behind one interface:

```
SourceHandler (ABC, _handlers.py)
├─ active_object()            # ADAPTER: hands views the current object (no concrete type leaks)
├─ on_selected(obj)           # GUI-thread: the type-specific state→preview transition
├─ load_full(obj, cancel)     # worker: ensure traces loaded; return the object carrying .data
├─ render_figure(obj, …)      # worker: pick the core Matplotlib renderer
├─ basename(obj)              # export filename stem
├─ source_path(obj)           # disk path for batch output naming
└─ release_after_batch(obj)   # free a JIT-loaded matrix between batch items
        ▲                               ▲
   ProfileHandler                  ChainHandler
```

`VisualizerTab` holds one `self._handler`, reassigned on selection; `SubTabbedTab` calls
`self._handler.<op>(...)` and **never knows** which kind it is driving. The profile and
chain pipelines are therefore fundamentally decoupled — a profile is never represented as
a "chain of length 1", so chain-only logic (stitching, spatial-union QC) cannot reach an
individual file. Worker-thread handler methods touch no Qt and are safe off-thread; the
caller snapshots a stable `handler` reference before each background job.

### Refactoring history (v0.5.0)

1. **Tab merge.** The standalone **Chains** tab (`ChainsTab`, `tabs/chains_tab.py`) was
   removed and folded into `VisualizerTab`; the central tab stack is now **Visualizer +
   Reprojector**. `MainWindow` clears the *inactive* sidebar list's highlight (under
   `blockSignals`) when the other list drives the view, so the source is never ambiguous.
2. **Boolean dispatch eliminated.** An interim `self._mode` flag + a base-class
   `_is_chain()` predicate were the first cut. Both were **deleted**: `_is_chain()` is gone
   from `SubTabbedTab`, and the four `if is_chain` branches (render-figure choice, lazy
   load, batch output path, post-batch release) were replaced by `SourceHandler` calls.
3. **Pipeline decoupling.** `_render_export_figure(...)` now takes a `handler` and calls
   `handler.render_figure(...)`; the single-export, HQ-preview, and batch worker jobs all
   load via `handler.load_full(...)`. `export_batch(items, cfg, handler)` receives the
   batch's handler from the caller (resolved via `VisualizerTab.source_handler(is_chain=…)`)
   so the batch kind is decoupled from the live view's kind.

### Maintenance guide — adding a new source type

To add a new object type (e.g. a *Survey* or *Composite*):

1. **Implement a new `SourceHandler` subclass** in `_handlers.py` providing the seven
   interface methods for the new kind.
2. **Register it in the sidebar selection logic** — instantiate it in
   `VisualizerTab.__init__` (`self._handlers[...]`), add a selection slot that assigns it,
   and (if it can be batch-exported) add its key to `source_handler()`.

**No changes to `SubTabbedTab` are required** — the base is closed to type knowledge and
open to new strategies. That is the whole point of the pattern.
