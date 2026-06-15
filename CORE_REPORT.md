# CORE_REPORT — SBP Studio Core Extraction

> Last updated: 2026-06-16
> Tests: **290 collected** (core + CLI run headless; the PyQt6/GUI suites run on a desktop with a display)
> `topas_core.py` (original draft): **deleted** — superseded by `sbp_studio/`
> Package renamed `topassuite` → **`sbp_studio`** (import path, launchers, translations).
> See [`GUI_REPORT.md`](GUI_REPORT.md) for the PyQt6 desktop interface built on top of this core.

---

## 1. Final layout + public API per module

### `sbp_studio/core/`

| Module | Public symbols | Notes |
|--------|---------------|-------|
| `constants.py` | `CMAPS`, `COORD_UNITS`, `PRESETS_CRS`, `FILTER_PRESETS`, `FILTER_DESCRIPTIONS` | Pure constants |
| `tasks.py` | `TopasCoreError`, `SegyLoadError`, `CRSError`, `ReprojectionError`, `Cancelled`, `ProgressCallback`, `CancelToken`, `LogCallback`, `PhasedTimer` | `PhasedTimer` for `--timeit` |
| `_backends.py` | `gpu_available()`, `worker_count()`, `fftw_available()`, `accel_info()`, `GPU`, `FFTW`, `N_WORKERS`, `XP` | pyfftw auto-activated; accel_info() feeds `accel` CLI |
| `model.py` | `SegyMetadata`, `SegyProfile`, `ProfileChain` | ref |
| `io_segy.py` | `load_metadata`, `load_profile`, `reproject_one`, `reproject_chain`, `join_profiles` | optimised path active; out_sc bug fixed; **duplicate-timestamp trace cleanup** (`_timestamp_dedup_mask`) |
| `processing.py` | `apply_predictive_decon`, `apply_filter_preset`, `process_profile_data`, `process_chain_data`, `time_window`, `_hilbert_parallel`, `_ref_agc` | Hilbert + AGC parallelised |
| `spectrum.py` | `compute_spectrum(data, fs)→SpectrumResult` | ref |
| `coordinates.py` | `resolve_crs(str)→str`, `validate_crs(str)→CRS` | ref |
| `chaining.py` | `detect_chains(profiles, gap_km)→list[ProfileChain]` | ref |
| `coloring.py` | `colormapped_rgba(data, cmap_name, vmin, vmax)→uint8` | GPU branch when CuPy present |
| `geometry_export.py` | `parse_timestamp`, `compute_fix_positions`, `write_fix_points_shp/geojson/csv`, `write_navline_shp/geojson/csv` | ref |

### `sbp_studio/viz/`

| Module | Public symbols | Notes |
|--------|---------------|-------|
| `render.py` | `render_profile_figure`, `render_chain_figure`, `render_spectrum_figure`, `save_figure`, `save_raw_rgba`, `build_theme`, `_THEMES`, `_save_pdf_raster` | Full theme system; margins; time axis; FIX customisation |

`render_profile_figure` / `render_chain_figure` complete keyword args:

```
figsize, dpi,
x_tick_km, t_tick_ms, show_grid, title_override, clip_lo,
time_tick_min, time_fmt, time_font_size, time_align,
margin_top_ms, margin_bottom_ms,
fix_font_size, fix_bbox_alpha, fix_color,
colors
```

`build_theme(theme, bg_color, text_color, axes_bg_color) → dict`
`_THEMES` keys: `"dark"`, `"light"`, `"print"`

`save_raw_rgba(source, data, path, params, render_opts, px_per_trace, dpi, figheight, is_chain, pdf_page)`

### `sbp_studio/cli/`

| Module | Subcommands |
|--------|-------------|
| `main.py` | `info`, `reproject`, `join-chain`, `export-image`, `spectrum`, `navline`, `fix`, `accel` |
| `commands.py` | All implementations + `_QUALITY_PRESETS`, `_load_profiles_parallel`, `cmd_accel` |

`export-image` complete flag set:

```
Processing:     --preset, --bandpass, --agc, --tvg, --align, --fill-zero, --clip, --clip-lo,
                --cmap, --invert
Quality:        --quality, --dpi, --px-per-trace, --figheight, --auto-height, --fill-zero
Physical scale: --x-scale, --y-scale, --velocity, --ratio, --ve, --max-aspect
X-axis mode:    --trace-axis, --km-no-stretch           (vs default distance/km extent)
Memory safety:  --mem-budget-gb                          (default 6; RAM-cap, anti-OOM)
Axes/ticks:     --x-tick, --t-tick, --time-ticks, --time-fmt, --time-font-size, --time-align
FIX marks:      --fix, --fix-color, --fix-font-size, --fix-bbox-alpha
Margins:        --margin-top, --margin-bottom
Theme/colour:   --theme, --bg-color, --text-color, --axes-bg-color
Output:         --no-axes, --format, --pdf-page, --title, --timeit, --out
```

**Scaling modes (new — keep many lines visually comparable):**

| Mode | Flags | Behaviour |
|------|-------|-----------|
| Lock aspect | `--ratio R` | Constant figure shape; vertical exaggeration (VE) floats with line length |
| Lock VE | `--x-scale K --ve N` | Constant VE → geologically comparable across lengths; `--ve` derives height as `N·depth_km/x_scale`, length-independent |
| Hybrid | `--x-scale K --ve N --max-aspect R` | Locks VE but caps the aspect at `R:1` so an extreme line never deforms into an "infinite noodle" (VE overridden only for that line) |

`--mem-budget-gb` caps the rasteriser to a RAM budget (default 6 GB) via a coupled
down-scale that preserves aspect **and** VE — fixes `numpy ArrayMemoryError` on
100+ km / high-DPI exports. Validated against free RAM before rendering; a prominent
warning is printed if the budget exceeds the machine's free RAM.

### `examples/`

| File | Purpose |
|------|---------|
| `generate_demo_data.py` | Creates synthetic SEG-Y in `_demo_in/` |
| `example_*.py` | End-to-end Python API examples |
| `run_demos.ps1` | PowerShell test runner (Windows) |
| `run_demos.sh` | Bash test runner (Linux / macOS) |

---

## 2. Behaviour-preservation checklist

| Behaviour | Module | Test |
|-----------|--------|------|
| Scalar fac: sc<0→1/abs, sc>0→sc, sc=0→1.0 | `io_segy._scalar_fac` | `test_scalar_fac_applied` |
| Arc-sec / 3600 for coord_unit==2 | `io_segy._populate_profile_from_file` | `test_arcsec_conversion` |
| Elevation scalar same formula | `io_segy._populate_profile_from_file` | `test_water_depth_shape` |
| Timestamp "YYYY-DOYnnn HH:MM:SS" | `io_segy._populate_profile_from_file` | `test_timestamps_format` |
| data (ns, n_traces) float32 | `io_segy._populate_profile_from_file` | `test_data_shape` |
| dist_km geo-vs-projected heuristic | `io_segy._dist_km` | `test_dist_km_shape` |
| delay_ms = int(delays[0]) | `io_segy._populate_profile_from_file` | `test_delay_ms_from_trace0` |
| Pipeline order: decon→bp→preset→tvg→agc→align | `processing._process_data_generic` | `TestProcessData.*` |
| apply_filter_preset "none" returns copy | `processing._ref_apply_filter_preset` | `test_none_returns_copy` |
| ProfileChain haversine concat | `model.ProfileChain._concat` | `TestProfileChainConcat.*` |
| compute_fix_positions algorithm | `geometry_export.compute_fix_positions` | `TestComputeFixPositions.*` |
| Reprojection: delay preserved | `io_segy.reproject_one` | `test_delay_preserved` |
| Reprojection: out_sc / new_uc rules | `io_segy.reproject_one` | `test_scalar_and_unit_*` |
| Reprojection: bin + text[0] copied | `io_segy.reproject_one` | `test_text_header_copied` |
| Chain TraceNumber sequential | `io_segy.reproject_chain` | `test_trace_number_sequential` |
| Partial output deleted on error | `io_segy._try_delete` | `test_partial_output_deleted` |

---

## 3. Library choices

### Core (required)

| Library | Why | Size |
|---------|-----|------|
| `numpy>=1.24` | Arrays, FFT | ~20 MB |
| `scipy>=1.10` | Filters, linalg | ~40 MB |
| `segyio>=1.9` | SEG-Y I/O | ~2 MB |
| `pyproj>=3.4` | CRS transforms | ~15 MB |
| `matplotlib>=3.7` | Colormaps (lazy in core), Agg (viz) | ~50 MB |
| `pillow>=9.0` | RGBA resize, PNG/TIFF/PDF | ~5 MB |
| `img2pdf` | Lossless PDF: PNG bytes → PDF stream, zero recompression, DPI metadata | ~4 MB + pikepdf |

### Optional accelerators

| Library | Benefit | Status |
|---------|---------|--------|
| `pyfftw>=0.13` | 2-5× faster Hilbert/bandpass via FFTW. Auto-activated via `scipy.fft.set_global_backend` | **Installed and active** |
| `cupy-cuda12x` | GPU: Hilbert, AGC, RGBA normalisation (fallback on CUDA OOM) | Installed by user (GTX 960M, 2 GB VRAM) |

---

## 4. Optimisations implemented

Benchmarked on real SBP data: 6-file chain, 7234 × 9677 samples, dt=31 µs, 8-core CPU, pyfftw active.

| Optimisation | Implementation | Measured speedup | Regression gate |
|-------------|---------------|-----------------|----------------|
| Hilbert (envelope/inst_phase/cos_phase/inst_freq) | `_parallel_apply` with `scipy.fft.set_workers(1)` per block | Part of 3× overall | `TestHilbertParallel*` — allclose atol=1e-6 |
| AGC CPU | `_parallel_apply(_agc_block)` | Part of 3× overall | `TestAGCParallel` — allclose atol=1e-6 |
| Bandpass in `apply_filter_preset` | `_parallel_apply(_sos_blk)` | Part of 3× overall | Implicit |
| pyfftw backend | `scipy.fft.set_global_backend(pyfftw)` at import | ~15-20% FFT speedup | Transparent |
| Float32-first resize (`_colorize_for_target`) | `scipy.ndimage.zoom(order=1)` on float32 BEFORE colourising when downsampling > 2:1 | **render+save: 22s → 5-7s** | Implicit |
| Parallel file loading | `ThreadPoolExecutor(max_workers=4)` over files | ~3-4× I/O speedup | Implicit |
| `--no-axes` mode | PIL/img2pdf direct save, bypasses matplotlib | render+save: 5.5s → 3s | N/A |
| Vectorised reprojection | `segyio.attributes` bulk read + single `tf.transform()` call | ~6% vs optimised path (both I/O-bound) | `TestReprojectVectorised` — allclose atol=1e-10 |
| `join_profiles` fast copy | Verbatim header copy, no Transformer call | ~6% vs identity transform | `TestJoinProfiles.*` |
| `out_sc = -10_000` (OQ-3 fix) | Correctness fix — no longer overflows 16-bit SEG-Y field | Correctness | `TestOutScFix` |

**End-to-end (chain, quality=print, no-axes):**
- Before: ~60s estimated
- After: **~20s measured**
- **~3× speedup** on 8-core CPU without GPU

---

## 5. Gaps filled (vs TopasSUITE.py monolith)

| Gap | GUI source (~line) | Module | Function |
|-----|---------------------|--------|----------|
| Spectrum computation | `_compute_spectrum` (~L2709) | `core/spectrum.py` | `compute_spectrum` |
| Navline writers (shp/geojson/csv) | `_write_navline_*` (~L2427-2597) | `core/geometry_export.py` | `write_navline_*` |
| CRS resolution | `_resolve_crs` | `core/coordinates.py` | `resolve_crs`, `validate_crs` |
| Header-only loader | — (new) | `core/io_segy.py` | `load_metadata` |
| FIX writer name disambiguation | Draft `topas_core.py` | `core/geometry_export.py` | `write_fix_points_*` vs `write_navline_*` |
| Headless image export | GUI canvas only | `viz/render.py` | `save_raw_rgba`, `_save_pdf_raster` |
| PDF export | Not in monolith | `viz/render.py` | `_save_pdf_raster` (img2pdf, lossless) |
| Quality/scale presets | GUI sliders | `cli/main.py` | `--quality`, `--px-per-trace`, `--auto-height` |
| Physical scale | GUI display | `cli/main.py` | `--x-scale`, `--y-scale`, `--velocity`, `--ratio` |
| UTC time axis | — (new) | `viz/render.py` | `_add_time_axis`, `--time-ticks` |
| Colour themes | GUI hardcoded | `viz/render.py` | `build_theme`, `_THEMES` |
| Zero-padded margins | — (new) | `viz/render.py` | `_pad_margins`, `--margin-top/bottom` |
| FIX mark customisation | — (new) | `viz/render.py` | `_draw_fix_marks` params, `--fix-color/font-size/bbox-alpha` |
| Acceleration diagnostics | — (new) | `core/_backends.py` + CLI | `accel_info()`, `cmd_accel` |
| Timing measurements | — (new) | `core/tasks.py` + CLI | `PhasedTimer`, `--timeit` |
| Pure join (no CRS) | — (new) | `core/io_segy.py` | `join_profiles`, `--no-reproject` |

---

## 6. Bug fixes

| Bug | Location | Fix |
|-----|----------|-----|
| `out_sc = -10_000_000` overflows 16-bit SEG-Y field | `io_segy._build_transformer` | Changed to `-10_000` (max valid SEG-Y scalar for geographic CRS) |
| DOY index `ts_str[6:9]` incorrect in time label | `viz/render._format_time_label` | Fixed to `ts_str[8:11]` (DOY digits in "YYYY-DOYnnn ...") |
| Axis tick colours grey in non-dark themes | `viz/render._apply_axes_options` | Now accepts and uses `colors` dict; removed hardcoded `_C` references |
| FIX mark bbox uses highlight colour as background | `viz/render._draw_fix_marks` | Changed to `fc=C["entry"]` (axes background) |
| `--preset envelope` silently ignored via CLI | `processing._process_data_generic` | Now accepts both display name ("Envelope (...)") and direct key ("envelope") |
| `start_process` on Windows doesn't inherit conda DLLs | CLI docs | Must use bare `python` command (not absolute path or `Start-Process`) |

---

## 7. Carried-forward risks / known limitations

| Risk | Location | Status |
|------|----------|--------|
| `delay_ms` uses `trace[0]` only | `io_segy._populate_profile_from_file` | Preserved. `delays` array exposes per-trace values. |
| Projected `dist_km` assumes metres | `io_segy._dist_km` | Preserved. `_check_crs_units` warns at load time for non-metre projected CRS. |
| `boundaries_km` vs `dist_km` offset at joins | `model.ProfileChain._concat` | Preserved. `boundaries_km` uses `total_km` cumsum. |
| Hilbert/AGC lack formal `allclose` regression gate | `processing._hilbert_parallel`, `_agc_block` | Gated by 20 tests in `test_regression_parallel.py`. |
| GTX 960M (2 GB VRAM) may OOM for chain | `_backends.py` GPU branches | Fallback to CPU on CUDA OOM. Single profiles fit; chains may not. |

---

## 8. Performance benchmarks

6-file chain, 7234 × 9677 samples, dt=31 µs.  
Hardware: 8-core CPU (7 workers), pyfftw active, no GPU. `--quality print --px-per-trace 1 --chain`

| Phase | Time | % |
|-------|------|---|
| Loading (6 files, parallel) | ~2.6s | 13% |
| Chain detect | ~1.7s | 9% |
| Processing (envelope + AGC) | ~9-10s | 45% |
| Render + save (float32-first resize + PIL/mpl) | ~5-7s | 33% |
| **TOTAL (no-axes PNG/TIFF)** | **~19-21s** | — |
| **TOTAL (with axes, matplotlib PDF)** | **~28-32s** | — |

With CuPy (GTX 960M, 2 GB VRAM, single profile):
- Processing: 9-10s → ~2-3s (GPU Hilbert + AGC)
- Chain: likely CPU fallback due to VRAM limit

---

## 9. Open questions

### Resolved

- ~~pyfftw integration~~ — active, marginal benefit for chains (memory-bound)
- ~~OQ-1 join-chain pure-join~~ — `join_profiles` + `--no-reproject`; auto-detect when `src==dst`
- ~~OQ-2 projected-CRS distance units~~ — `_check_crs_units` warns at load time
- ~~OQ-3 out_sc overflow~~ — `out_sc = -10_000`; `TestOutScFix` gates it
- ~~OQ-4 regression gate for parallel paths~~ — 20 tests in `test_regression_parallel.py`
- ~~OQ-5 CuPy installation~~ — installed by user

### Pending

**OQ-6** — Vectorised reprojection is I/O-bound at current data sizes; further gains would require bulk header write (not exposed by segyio's public API).

**OQ-7** — CRS unit warning (`_check_crs_units`) only fires when `detected_crs` is set. Files with projected coordinates (coord_unit=1) after reprojection have `detected_crs=None`. A future `--assume-crs` CLI option would allow users to declare the CRS of non-geographic files.

---

## 10. Recommendations (next steps)

| Priority | Action | Status |
|----------|--------|--------|
| ~~High~~ | ~~Install CuPy~~ | Done |
| ~~High~~ | ~~Regression tests for Hilbert/AGC~~ | Done — 20 tests |
| ~~Medium~~ | ~~Fix out_sc overflow~~ | Done — `-10_000` |
| ~~Medium~~ | ~~join-chain fast path~~ | Done — `join_profiles` |
| ~~Medium~~ | ~~CRS unit warning~~ | Done — partial |
| ~~Low~~ | ~~Vectorised bulk reprojection~~ | Done — gated |
| **Low** | Expand CRS detection to projected files (OQ-7) | Pending |
| **Low** | Chain streaming / lazy concat (peak memory) | Pending |
| **Low** | `--scale-bar` graphical scale bar on images | Pending |
| **Low** | `--velocity V` depth axis (requires velocity model) | Pending |
| **Low** | Bulk header write for reprojection (segyio internal API) | Pending |

---

## 11. Recent additions (2026-06)

| Feature | Module | Summary |
|---------|--------|---------|
| **Duplicate-timestamp cleanup** | `io_segy._timestamp_dedup_mask` + `_populate_profile_from_file` | TOPAS files stamp blocks of consecutive traces with an identical DayOfYear/Hour/Minute/Second. The loader builds the `(n,4)` time matrix, diffs consecutive rows, and drops repeats (first trace always kept; bypassed if the time words are all-zero). The **same** keep-mask filters every per-trace array (data, lons/lats, delays, water depth, timestamps, inspector headers) so nothing misaligns. Records `original_n_traces` / `n_purged`; updates `n_traces`. File-copy ops (`reproject_*` / `join_profiles`) iterate the file's true `tracecount`, so they stay byte-faithful — the purge is a display/analysis feature. Tests: `test_dedup_timestamps.py`. |
| **Constant-VE scaling** | `cli/commands._figsize`, `cli/main` | `--ve N` (with `--x-scale`) fixes vertical exaggeration length-independently (`h = N·depth_km/x_scale`). `--max-aspect R` is the "infinite-noodle" guard that caps the aspect only when a `--ve` line would otherwise be extreme. Closed form: `VE = x_scale·2e6 / (velocity·y_scale)`. Tests: `test_cli_figsize_clamp.py`. |
| **Rasteriser RAM safeguard** | `cli/commands._safe_pixel_budget`, `_figsize` | Caps the output raster to a RAM budget (default **6 GB**, `--mem-budget-gb`) by scaling both figure axes by a single factor `s = √(budget/required)` — preserves aspect **and** VE, only lowering pixel density. Prevents `numpy ArrayMemoryError` on very long / high-DPI lines. Tests: `test_export_mem_safety.py`. |
| **X-axis modes** | `viz/render` (`_x_axis_extent`, `_label_x_axis`) | `--trace-axis` (one column per trace, axis in trace number) and `--km-no-stretch` (km labels at real trace positions, no horizontal stretch) alongside the historical distance/km extent. |
| **Chain blockiness fix** | `viz/render.render_chain_figure` | The stitched native-width chain image is now smoothly resized (bilinear/LANCZOS) to the output pixel width before `imshow`, instead of nearest-upscaling each trace into a `px_per_trace`-wide block. |
| **Configurable axis fonts + grid** | `viz/render` | `axis_font_size`, `grid_alpha`, `grid_lw` threaded through `render_profile_figure` / `render_chain_figure` (defaults reproduce the historical look). |
