<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Python][python-shield]][python-url]
[![Conda][conda-shield]][conda-url]
[![Tests][tests-shield]][tests-url]
[![GPL License][license-shield]][license-url]

<!-- PROJECT LOGO / TITLE -->
<h1 align="center">TOPAS Suite — Python</h1>

<p align="center">
Headless SEG-Y processing core, plotter-quality image export and CLI for<br/>
Kongsberg TOPAS sub-bottom profiler data.<br/>
<br/>
<a href="https://www.kongsberg.com/maritime/products/mapping-systems/sub-bottom-profilers/">Kongsberg TOPAS</a>
·
<a href="https://github.com/AngelVeraHerrera/TopasSuite-Python/issues">Report Bug</a>
·
<a href="https://github.com/AngelVeraHerrera/TopasSuite-Python/issues">Request Feature</a>
</p>

---

## 📖 About This Project

**TOPAS Suite Python** extracts the computational core of the original `TopasSUITE.py` GUI application into a clean, headless Python package with a full CLI.  
The original GUI monolith is **unchanged** and continues to run independently.

- ✅ Fully headless — no DISPLAY, no Qt, no tkinter required
- ✅ GPU acceleration via CuPy (optional, auto-detected)
- ✅ pyfftw backend for 2-5× faster FFT operations (optional, auto-activated)
- ✅ Parallel CPU processing across all available cores
- ✅ Lossless PDF export with physical scale control (img2pdf)
- ✅ Plotter-ready images: physical km/in + ms/in scale, UTC time axis, FIX marks
- ✅ Three colour themes: `dark`, `light`, `print`
- ✅ 115 / 115 tests passing

> ⚠️ This repository does **not** include proprietary SEG-Y survey data. Real data files must be provided by the authorised user.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 📦 Package Structure

```text
topassuite/
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
└── cli/
    ├── main.py             Entry point: python -m topassuite.cli.main
    └── commands.py         All subcommand implementations
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
| `cli/` | core + viz | No GUI imports. Runs with no DISPLAY |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🛠️ Installation

**Recommended — conda environment:**

```bash
conda env create -f environment.yml
conda activate topassuite
```

**Optional performance accelerators** (run `accel` after installing to verify):

```bash
# pyfftw — 2-5× faster FFT for envelope/bandpass (auto-activated at import)
conda install -n topassuite -c conda-forge pyfftw

# CuPy — NVIDIA GPU acceleration (check your CUDA version with nvidia-smi first)
pip install cupy-cuda12x          # for CUDA 12.x or 13.x drivers
# conda install -c conda-forge cupy cudatoolkit=12.6   # alternative
```

**pip only:**

```bash
pip install numpy scipy segyio pyproj matplotlib pillow img2pdf
```

**Verify acceleration status:**

```bash
python -m topassuite.cli.main accel
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## ⚡ Quick Start

```bash

# Inspect a SEG-Y file
python -m topassuite.cli.main info survey.sgy --json

# Export plotter-quality image (PDF, physical scale, UTC time axis)
python -m topassuite.cli.main join-chain f1.sgy f2.sgy f3.sgy \
    --no-reproject --out L01_joined.sgy

python -m topassuite.cli.main export-image L01_joined.sgy \
    --preset envelope --cmap Greys --agc --align --fill-zero \
    --x-scale 2 --ratio 3 --velocity 1500 \
    --x-tick 5 --t-tick 50 --time-ticks 5 --time-fmt full \
    --fix 5 --fix-color "#cc4444" --fix-bbox-alpha 0.0 \
    --margin-top 20 --margin-bottom 20 \
    --theme print --quality high --format pdf --pdf-page auto \
    --out L01_section.pdf

# Run the full demo suite
TODO
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🖥️ CLI Reference

All subcommands write progress to **stderr**; errors produce a non-zero exit code and no DISPLAY is required.

### `info` — Inspect metadata

```bash
python -m topassuite.cli.main info FILE... [--json]
```

### `reproject` — Reproject coordinates

```bash
python -m topassuite.cli.main reproject FILE... \
    --src EPSG:4326 --dst EPSG:32629 \
    [--unit-hint 2] [--out-dir DIR]
```

### `join-chain` — Concatenate profiles into one SEG-Y

```bash
# Fast copy (no coordinate change) — recommended for same-CRS chains
python -m topassuite.cli.main join-chain f1.sgy f2.sgy f3.sgy \
    --no-reproject --out L01_joined.sgy

# Auto-detected pure join when src == dst
python -m topassuite.cli.main join-chain f1.sgy f2.sgy \
    --src EPSG:4326 --dst EPSG:4326 --out joined.sgy

# Full reprojection
python -m topassuite.cli.main join-chain f1.sgy f2.sgy \
    --src EPSG:4326 --dst EPSG:32629 --out joined_utm.sgy
```

### `export-image` — Seismic section images

```bash
python -m topassuite.cli.main export-image FILE... [OPTIONS]
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
| `--ratio W_H` | Target width:height ratio — prints VE at given velocity |

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
python -m topassuite.cli.main spectrum FILE [--format png|pdf] [--out PATH]
```

### `navline` — Navigation track geometry

```bash
python -m topassuite.cli.main navline FILE... \
    [--chain] --format shp|geojson|csv [--crs EPSG] [--attrs] [--out PATH]
```

### `fix` — FIX-point export

```bash
python -m topassuite.cli.main fix FILE... \
    [--chain] --interval MIN --format shp|geojson|csv [--out PATH]
```

### `accel` — Hardware acceleration status

```bash
python -m topassuite.cli.main accel
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🖨️ Plotter Export Guide

Full pipeline for a publication-quality sub-bottom profile:

```bash
# Step 1 — join profiles
python -m topassuite.cli.main join-chain f1.sgy f2.sgy f3.sgy f4.sgy f5.sgy f6.sgy \
    --no-reproject --out L01_joined.sgy

# Step 2 — export PDF (natural proportions, UTC timestamps, red FIX marks)
python -m topassuite.cli.main export-image L01_joined.sgy \
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
# Full suite (115 tests, ~1 min)
pytest tests/ -v

# Regression tests for parallelised paths and bug fixes
pytest tests/test_regression_parallel.py -v

# CLI smoke tests only
pytest tests/test_cli_smoke.py -v
```

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

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

This license applies to the source code, scripts and documentation in this repository. It does not apply to SEG-Y survey data, Kongsberg TOPAS software, or any third-party proprietary materials.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🙏 Acknowledgments

* [Royal Institute and Observatory of the Spanish Navy (San Fernando, Spain)](https://armada.defensa.gob.es/ArmadaPortal/page/Portal/ArmadaEspannola/cienciaobservatorio/prefLang-es/)
* [segyio](https://github.com/equinor/segyio) — SEG-Y I/O
* [pyproj](https://pyproj4.github.io/pyproj/) — CRS transformations
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
[conda-shield]: https://img.shields.io/badge/conda--forge-topassuite-green?style=for-the-badge&logo=anaconda&logoColor=white
[conda-url]: https://conda-forge.org/
[tests-shield]: https://img.shields.io/badge/tests-115%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white
[tests-url]: https://docs.pytest.org/
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/gpl-3.0
