<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Python][python-shield]][python-url]
[![Conda][conda-shield]][conda-url]
[![Tests][tests-shield]][tests-url]
[![GPL License][license-shield]][license-url]

<!-- PROJECT LOGO / TITLE -->
<h1 align="center">SBP Studio — Python</h1>

<p align="center">
Headless SEG-Y processing core, plotter-quality image export and CLI for<br/>
Kongsberg TOPAS sub-bottom profiler data.<br/>
<br/>
<a href="https://www.kongsberg.com/maritime/products/mapping-systems/sub-bottom-profilers/">Kongsberg TOPAS</a>
·
<a href="https://github.com/AngelVeraHerrera/SBP-Studio/issues">Report Bug</a>
·
<a href="https://github.com/AngelVeraHerrera/SBP-Studio/issues">Request Feature</a>
</p>

---

## 📖 About This Project

**SBP Studio** is a sub-bottom-profiler (SBP) seismic suite for Kongsberg TOPAS SEG-Y data.
It pairs a clean, **headless computation core + CLI** with a decoupled **PyQt6 desktop GUI** —
both driven by the same code, so an image exported from the GUI is byte-for-byte the one the
CLI would produce.

**Core / CLI**
- Fully headless — no DISPLAY, no Qt required
- GPU acceleration via CuPy (optional, auto-detected) + pyfftw FFT backend (2-5×)
- Parallel CPU processing across all cores
- Lossless PDF export with physical scale control (img2pdf)
- Plotter-ready images: physical km/in + ms/in scale, UTC time axis, FIX marks
- **Constant vertical-exaggeration** scaling (`--ve` / `--max-aspect`) so lines of any length stay comparable
- **RAM-safe export** (`--mem-budget-gb`) — no more `ArrayMemoryError` on 100+ km lines
- **Duplicate-timestamp trace cleanup** for raw TOPAS files
- Three colour themes: `dark`, `light`, `print`

**Desktop GUI (PyQt6 + PyQtGraph)**
- Live, hardware-accelerated seismic / map / spectrum / header views
- Reorderable DSP node pipeline (decon · bandpass · preset · TVG · AGC · water-mute · swell)
- Aspect / VE / hybrid scaling controls with a live DPI & size readout
- **Render Viewport HQ** — an export-quality overlay of the zoomed-in area, on demand
- CRS reprojection + navline/FIX geometry export, integrated CLI console
- Memory-bounded (lazy load + LRU); RAM-budget guard before heavy exports

> ✅ **290 tests** — the headless core + CLI suites run anywhere; the PyQt6 GUI suites run on a desktop with a display.
> 📑 See [`CORE_REPORT.md`](CORE_REPORT.md) (computation/CLI) and [`GUI_REPORT.md`](GUI_REPORT.md) (desktop interface) for full architecture notes.

> ⚠️ This repository does **not** include proprietary SEG-Y survey data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📦 Package Structure

```text
sbp_studio/
├── core/                   GUI-free computation layer
│   ├── constants.py        CMAPS, PRESETS_CRS, FILTER_PRESETS, COORD_UNITS
│   ├── tasks.py            Exceptions, CancelToken, PhasedTimer
│   ├── _backends.py        GPU/CuPy + pyfftw detection, accel_info()
│   ├── model.py            SegyMetadata, SegyProfile, ProfileChain
│   ├── io_segy.py          load_metadata, load_profile, reproject_one/chain, join_profiles
│   ├── processing.py       Hilbert (parallel), AGC (parallel), full pipeline
│   ├── spectrum.py         compute_spectrum → SpectrumResult
│   ├── coordinates.py      resolve_crs, validate_crs, CRS unit detection
│   ├── chaining.py         detect_chains
│   ├── coloring.py         colormapped_rgba (GPU-accelerated when CuPy present)
│   └── geometry_export.py  write_fix_points_* / write_navline_* (stdlib only)
├── viz/
│   └── render.py           Headless Agg figures + raw-pixel TIFF/PNG/PDF export
├── cli/
│   ├── main.py             Entry point: python -m sbp_studio.cli.main
│   └── commands.py         All subcommand implementations
└── gui/                    PyQt6 + PyQtGraph desktop interface (optional)
    ├── app.py / __main__.py   Bootstrap — python -m sbp_studio.gui
    ├── main_window.py / state.py / theme.py / i18n.py
    ├── workers/            CoreWorker (QThread) + run_task
    ├── tabs/               Visualizer / Chains / Reprojector (+ shared _base, _render)
    ├── components/         seismic/map/spectrum/header views, controls, export dialog, CLI console
    ├── dsp/                Reorderable DSP node pipeline + live PreviewController
    └── translations/       sbp_studio_es.ts (Qt Linguist)
applications/
├── SBPStudio_CLI.py        Thin CLI launcher (adds repo root to sys.path)
└── SBPStudio_GUI.py        Thin GUI launcher
env/
├── environment.yml         Conda spec (core + GUI + optional accelerators)
├── setup_env.sh            Create/activate the env — Linux / macOS
└── setup_env.ps1           Create/activate the env — Windows PowerShell
examples/
├── generate_demo_data.py   Create synthetic SEG-Y fixtures
├── example_*.py            End-to-end Python API examples
├── run_demos.ps1           PowerShell runner (Windows)
└── run_demos.sh            Bash runner (Linux / macOS)
tests/
├── make_synthetic_segy.py  Parametric SEG-Y generator
├── conftest.py             Shared pytest fixtures
└── test_*.py               Unit, regression and CLI smoke tests
```

### Architecture boundary

| Layer | Imports | Constraint |
|-------|---------|-----------|
| `core/` | numpy, scipy, segyio, pyproj | No GUI libs. matplotlib only as lazy color-table lookup in `coloring.py` |
| `viz/` | core + matplotlib(Agg) + Pillow + img2pdf | Never imported by core |
| `cli/` | core + viz (+ psutil) | No GUI imports. Runs with no DISPLAY |
| `gui/` | core + viz + PyQt6 + PyQtGraph | Presentation only; **never imported by core/viz/cli**. Export uses the same `viz/render.py` as the CLI |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🛠️ Installation

**Recommended — one-shot setup script (creates + activates the conda env):**

```bash
# Linux / macOS  — source it so the env stays active in your shell
source env/setup_env.sh

# Windows PowerShell — dot-source it so activation sticks
. .\env\setup_env.ps1
```

The scripts create the `sbp_studio` env from `env/environment.yml` (or update it in
place with `--prune` if it already exists) and then activate it.

**Manual conda equivalent:**

```bash
conda env create -f env/environment.yml
conda activate sbp_studio
```

**Optional performance accelerators** (run `accel` after installing to verify):

```bash
# pyfftw — 2-5× faster FFT for envelope/bandpass (auto-activated at import)
conda install -n sbp_studio -c conda-forge pyfftw

# CuPy — NVIDIA GPU acceleration (check your CUDA version with nvidia-smi first)
pip install cupy-cuda12x          # for CUDA 12.x or 13.x drivers
# conda install -c conda-forge cupy cudatoolkit=12.6   # alternative
```

**pip only:**

```bash
# Core + CLI
pip install numpy scipy segyio pyproj matplotlib pillow img2pdf psutil

# Add the desktop GUI
pip install PyQt6 pyqtgraph
```

**Verify acceleration status:**

```bash
python -m sbp_studio.cli.main accel
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ⚡ Quick Start

**Desktop GUI:**

```bash
python -m sbp_studio.gui          # or:  python applications/SBPStudio_GUI.py
```

Add profiles, detect chains, tune the DSP node pipeline live, pick a scaling mode
(aspect / VE / hybrid), preview the zoomed-in area at full quality with **Render Viewport HQ**,
and export plotter-quality PDF/PNG/TIFF/SVG.

**CLI:**

```bash

# Inspect a SEG-Y file
python -m sbp_studio.cli.main info survey.sgy --json

# Export plotter-quality image (PDF, physical scale, UTC time axis)
python -m sbp_studio.cli.main join-chain f1.sgy f2.sgy f3.sgy \
    --no-reproject --out L01_joined.sgy

python -m sbp_studio.cli.main export-image L01_joined.sgy \
    --preset envelope --cmap Greys --agc --align --fill-zero \
    --x-scale 2 --ratio 3 --velocity 1500 \
    --x-tick 5 --t-tick 50 --time-ticks 5 --time-fmt full \
    --fix 5 --fix-color "#cc4444" --fix-bbox-alpha 0.0 \
    --margin-top 20 --margin-bottom 20 \
    --theme print --quality high --mem-budget-gb 6 --format pdf --pdf-page auto \
    --out L01_section.pdf

# Run the full demo suite
bash examples/run_demos.sh        # Linux / macOS   (PowerShell: .\examples\run_demos.ps1)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🖥️ CLI Reference

All subcommands write progress to **stderr**; errors produce a non-zero exit code and no DISPLAY is required.

### `info` — Inspect metadata

```bash
python -m sbp_studio.cli.main info FILE... [--json]
```

### `reproject` — Reproject coordinates

```bash
python -m sbp_studio.cli.main reproject FILE... \
    --src EPSG:4326 --dst EPSG:32629 \
    [--unit-hint 2] [--out-dir DIR]
```

### `join-chain` — Concatenate profiles into one SEG-Y

```bash
# Fast copy (no coordinate change) — recommended for same-CRS chains
python -m sbp_studio.cli.main join-chain f1.sgy f2.sgy f3.sgy \
    --no-reproject --out L01_joined.sgy

# Auto-detected pure join when src == dst
python -m sbp_studio.cli.main join-chain f1.sgy f2.sgy \
    --src EPSG:4326 --dst EPSG:4326 --out joined.sgy

# Full reprojection
python -m sbp_studio.cli.main join-chain f1.sgy f2.sgy \
    --src EPSG:4326 --dst EPSG:32629 --out joined_utm.sgy
```

### `export-image` — Seismic section images

```bash
python -m sbp_studio.cli.main export-image FILE... [OPTIONS]
```

**Quality / resolution:**

| Option | Default | Description |
|--------|---------|-------------|
| `--quality screen\|print\|high\|ultra` | — | DPI preset: 150/300/600/900 |
| `--dpi N` | 150 | Manual DPI (overrides `--quality`) |
| `--px-per-trace N` | 2.0 | Pixels per trace (horizontal) |
| `--auto-height` | off | figheight = ns/dpi (1 sample ≈ 1 px) |

**Physical scale (overrides px-per-trace/figheight):**

| Option | Description |
|--------|-------------|
| `--x-scale KM_PER_IN` | Horizontal: figwidth = total_km / x_scale |
| `--y-scale MS_PER_IN` | Vertical: figheight = record_ms / y_scale |
| `--velocity M_S` | Sound velocity m/s for physical scale (default: 1500) |
| `--ratio W_H` | Lock aspect ratio (VE floats with line length) |
| `--ve N` | **Lock vertical exaggeration** (with `--x-scale`) — constant VE, length-independent, so lines are comparable |
| `--max-aspect R` | Hybrid: lock VE but cap the aspect at `R:1` so an extreme line never becomes an "infinite noodle" |

**Memory safety & X-axis mode:**

| Option | Default | Description |
|--------|---------|-------------|
| `--mem-budget-gb GB` | 6 | Cap the raster to a RAM budget (scales both axes equally → preserves aspect **and** VE). Prevents `ArrayMemoryError`; warns if it exceeds free RAM |
| `--trace-axis` | — | One column per trace; bottom axis in trace number (no horizontal stretch) |
| `--km-no-stretch` | — | Km labels at real trace positions, no horizontal stretching |

**Processing:**

| Option | Description |
|--------|-------------|
| `--preset KEY` | `envelope`, `topas_narrow`, `topas_wide`, `similarity`, … |
| `--bandpass LO HI` | Butterworth bandpass (Hz) |
| `--agc` | Automatic gain control |
| `--align` / `--fill-zero` | Delay compensation; fill gaps with 0 (white in Greys) |
| `--clip P` / `--clip-lo P` | Upper/lower amplitude clip percentile |

**Axes, ticks, grid:**

| Option | Description |
|--------|-------------|
| `--x-tick KM` | Distance ticks every N km |
| `--t-tick MS` | Time ticks every N ms |
| `--time-ticks MIN` | Secondary top x-axis with UTC timestamps every N min |
| `--time-fmt hhmm\|fix\|position\|datetime\|full` | Label content |
| `--time-font-size PT` | Font size for time labels (default: 6.0) |
| `--time-align left\|center\|right` | Label position relative to tick (default: left) |
| `--grid` | Semi-transparent grid overlay |

**FIX marks:**

| Option | Description |
|--------|-------------|
| `--fix MIN` | Vertical FIX lines every N minutes |
| `--fix-color HEX` | FIX line and number colour (default: theme highlight) |
| `--fix-font-size PT` | FIX number font size (default: 5.0) |
| `--fix-bbox-alpha A` | Background box opacity (0.0 = no box, default: 0.12) |

**Margins:**

| Option | Description |
|--------|-------------|
| `--margin-top MS` | Zero-filled padding above the record (ms) |
| `--margin-bottom MS` | Zero-filled padding below the record (ms) |

**Theme / colours:**

| Option | Description |
|--------|-------------|
| `--theme dark\|light\|print` | Colour preset (default: dark) |
| `--bg-color HEX` | Figure background override |
| `--text-color HEX` | Text and tick label colour override |
| `--axes-bg-color HEX` | Seismic axes background override |

**Output:**

| Option | Description |
|--------|-------------|
| `--no-axes` | Raw pixels only — no matplotlib margins; exact px/trace |
| `--format png\|pdf\|tif\|svg` | Output format (default: png) |
| `--pdf-page auto\|A0\|A1\|A2\|A3\|A4` | PDF page size |
| `--title TEXT` | Override auto-generated title |
| `--timeit` | Print wall-clock breakdown per phase |

### `spectrum` — Frequency spectrum

```bash
python -m sbp_studio.cli.main spectrum FILE [--format png|pdf] [--out PATH]
```

### `navline` — Navigation track geometry

```bash
python -m sbp_studio.cli.main navline FILE... \
    [--chain] --format shp|geojson|csv [--crs EPSG] [--attrs] [--out PATH]
```

### `fix` — FIX-point export

```bash
python -m sbp_studio.cli.main fix FILE... \
    [--chain] --interval MIN --format shp|geojson|csv [--out PATH]
```

### `accel` — Hardware acceleration status

```bash
python -m sbp_studio.cli.main accel
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🖨️ Plotter Export Guide

Full pipeline for a publication-quality sub-bottom profile:

```bash
# Step 1 — join profiles
python -m sbp_studio.cli.main join-chain f1.sgy f2.sgy f3.sgy f4.sgy f5.sgy f6.sgy \
    --no-reproject --out L01_joined.sgy

# Step 2 — export PDF (natural proportions, UTC timestamps, red FIX marks)
python -m sbp_studio.cli.main export-image L01_joined.sgy \
    --preset envelope --cmap Greys --agc --align --fill-zero \
    --x-scale 2 --ratio 3 --velocity 1500 \
    --x-tick 5 --t-tick 50 --time-ticks 5 \
    --time-fmt full --time-font-size 5.5 --time-align left \
    --fix 5 --fix-color "#cc4444" --fix-bbox-alpha 0.0 \
    --margin-top 20 --margin-bottom 20 \
    --theme print --quality high \
    --format pdf --pdf-page auto \
    --timeit --out L01_section.pdf
```

**Scale selection guide** — `--ratio` vs physical scale:

| `--ratio` | `--velocity 1500` VE | Appearance |
|-----------|---------------------|------------|
| `3` | ≈60× | Typical sub-bottom section |
| `4` | ≈45× | More panoramic |
| `--x-scale 2 --velocity 1500` | 1× (true scale) | Extremely flat, reference only |

**Format comparison:**

| Format | Best for | Size |
|--------|----------|------|
| `--no-axes --format tif` | Plotter, exact 1px/trace, lossless | large |
| `--format pdf --pdf-page auto` | Plotter roll, vector text + raster | medium |
| `--format pdf --pdf-page A3` | Desk printer, aspect-fitted | medium |
| `--format png` | Screen review | small |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🧪 Run Examples

Generate synthetic data and run all examples:

```bash
# Generate synthetic SEG-Y fixtures
python examples/generate_demo_data.py

# Python API examples
python examples/example_inspect.py
python examples/example_export_image.py
python examples/example_spectrum.py
python examples/example_reproject.py
python examples/example_navline_fix.py
```

Run the full demo suite (synthetic + real data) with a single script:

```bash
# Linux / macOS
bash examples/run_demos.sh [tests|demo|real|accel]

# Windows PowerShell
.\examples\run_demos.ps1 [-Solo tests|demo|real|accel]
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ✅ Run Tests

```bash
# Headless core + CLI suites (run anywhere, no display needed)
pytest tests/ -v

# Regression tests for parallelised paths and bug fixes
pytest tests/test_regression_parallel.py -v

# Duplicate-timestamp cleanup, figure scaling, RAM-safe export
pytest tests/test_dedup_timestamps.py tests/test_cli_figsize_clamp.py tests/test_export_mem_safety.py -v
```

The PyQt6/GUI suites need a display; on a headless box use `QT_QPA_PLATFORM=offscreen`.
**290 tests** collected in total.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🔧 Two-Track Design

Every optimised function must pass a formal regression test before shipping:

```python
np.testing.assert_allclose(optimised_output, _ref_output, atol=TOLERANCE)
```

| Symbol | Meaning |
|--------|---------|
| `_ref_<name>` | Reference path — frozen, byte-identical to monolith |
| `<name>` | Public API — dispatches to optimised path (gate passed) or `_ref_` |

Currently optimised and regression-gated paths:

| Path | Tolerance | Test |
|------|-----------|------|
| Hilbert (envelope, inst_phase) via `_parallel_apply` | atol=1e-6 | `TestHilbertParallel*` |
| AGC via `_parallel_apply` | atol=1e-6 | `TestAGCParallel` |
| Vectorised reprojection (`_opt_reproject_coords_bulk`) | atol=1e-10 | `TestReprojectVectorised` |
| `join_profiles` (pure copy, no CRS transform) | exact | `TestJoinProfiles` |
| `out_sc = -10_000` fix (OQ-3) | exact | `TestOutScFix` |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ⚠️ Known Limitations

Carried forward from the original monolith (documented, not fixed):

- `delay_ms` uses `trace[0]` only — per-trace variability available in `delays` array
- `dist_km` for projected CRS assumes axis units are **metres** — warning emitted at load time if non-metre CRS detected
- `boundaries_km` in `ProfileChain` is based on per-profile `total_km` cumsum, not `dist_km` directly
- `out_sc = -10_000` for geographic reprojection gives 4 decimal places (≈ 11 m accuracy at equator)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📄 License

Distributed under the GNU General Public License v3.0 (GPL-3.0). See [`LICENSE`](LICENSE) for more information.

This license applies to the source code, scripts and documentation in this repository. It does not apply to SEG-Y survey data, Kongsberg TOPAS software, or any third-party proprietary materials.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🙏 Acknowledgments

* [Royal Institute and Observatory of the Spanish Navy (San Fernando, Spain)](https://armada.defensa.gob.es/ArmadaPortal/page/Portal/ArmadaEspannola/cienciaobservatorio/prefLang-es/)
* [segyio](https://github.com/equinor/segyio) — SEG-Y I/O
* [pyproj](https://pyproj4.github.io/pyproj/) — CRS transformations
* [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — desktop GUI toolkit
* [PyQtGraph](https://www.pyqtgraph.org/) — fast interactive plotting
* [img2pdf](https://gitlab.mister-muffin.de/josch/img2pdf) — lossless PDF embedding
* [pyfftw](https://pyfftw.readthedocs.io/) — FFTW-backed FFT acceleration
* [CuPy](https://cupy.dev/) — GPU array library
* [GPL License](https://choosealicense.com/licenses/gpl-3.0/)
* [Shields.io](https://shields.io)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[conda-shield]: https://img.shields.io/badge/conda--forge-sbp_studio-green?style=for-the-badge&logo=anaconda&logoColor=white
[conda-url]: https://conda-forge.org/
[tests-shield]: https://img.shields.io/badge/tests-290-brightgreen?style=for-the-badge&logo=pytest&logoColor=white
[tests-url]: https://docs.pytest.org/
[license-shield]: https://img.shields.io/badge/License-GPL%203.0-yellow.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/gpl-3.0
