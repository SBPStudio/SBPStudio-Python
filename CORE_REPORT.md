# CORE_REPORT — TOPAS Suite Core Extraction

## 1. Final layout + public API per module

### `topassuite/core/`

| Module | Public symbols | Track |
|--------|---------------|-------|
| `constants.py` | `CMAPS`, `COORD_UNITS`, `PRESETS_CRS`, `FILTER_PRESETS`, `FILTER_DESCRIPTIONS` | — |
| `tasks.py` | `TopasCoreError`, `SegyLoadError`, `CRSError`, `ReprojectionError`, `Cancelled`, `ProgressCallback`, `CancelToken`, `LogCallback` | — |
| `_backends.py` | `gpu_available()`, `worker_count()`, `GPU`, `N_WORKERS`, `XP` | — |
| `model.py` | `SegyMetadata` (dataclass), `SegyProfile`, `ProfileChain` | ref |
| `io_segy.py` | `load_metadata(path)→SegyMetadata`, `load_profile(path, load_traces)→SegyProfile`, `reproject_one(…)→str`, `reproject_chain(…)→str` | ref |
| `processing.py` | `apply_predictive_decon`, `apply_filter_preset`, `process_profile_data`, `process_chain_data`, `time_window` | ref |
| `spectrum.py` | `compute_spectrum(data, fs)→SpectrumResult` | ref |
| `coordinates.py` | `resolve_crs(str)→str`, `validate_crs(str)→CRS` | ref |
| `chaining.py` | `detect_chains(profiles, gap_km)→list[ProfileChain]` | ref |
| `coloring.py` | `colormapped_rgba(data, cmap_name, vmin, vmax)→uint8` | ref |
| `geometry_export.py` | `parse_timestamp`, `compute_fix_positions`, `write_fix_points_shp/geojson/csv`, `write_navline_shp/geojson/csv` | ref |

### `topassuite/viz/`

| Module | Public symbols |
|--------|---------------|
| `render.py` | `render_profile_figure`, `render_chain_figure`, `render_spectrum_figure`, `save_figure` |

### `topassuite/cli/`

| Module | Subcommands |
|--------|------------|
| `main.py` | CLI entry point (`python -m topassuite.cli.main`) |
| `commands.py` | `cmd_info`, `cmd_reproject`, `cmd_join_chain`, `cmd_export_image`, `cmd_spectrum`, `cmd_navline`, `cmd_fix` |

---

## 2. Behaviour-preservation checklist

| Behaviour | Covered by | Test |
|-----------|-----------|------|
| Scalar fac: sc<0→1/abs, sc>0→sc, sc=0→1.0 | `io_segy._scalar_fac` | `test_metadata.TestLoadMetadata.test_scalar_fac_applied` |
| Arc-sec / 3600 for coord_unit==2 | `io_segy._populate_profile_from_file` | `test_metadata.TestLoadMetadata.test_arcsec_conversion` |
| Elevation scalar same formula | `io_segy._populate_profile_from_file` | `test_metadata.TestLoadMetadata.test_water_depth_shape` |
| Timestamp "YYYY-DOYnnn HH:MM:SS" | `io_segy._populate_profile_from_file` | `test_metadata.TestLoadMetadata.test_timestamps_format` |
| data (ns, n_traces) float32 | `io_segy._populate_profile_from_file` | `test_processing.TestProcessData.test_data_shape` |
| dist_km geo-vs-projected heuristic | `io_segy._dist_km` | `test_metadata.TestLoadMetadata.test_dist_km_shape` |
| delay_ms = int(delays[0]) | `io_segy._populate_profile_from_file` | `test_metadata.TestLoadMetadata.test_delay_ms_from_trace0` |
| Pipeline order: decon→bp→preset→tvg→agc→align | `processing._process_data_generic` | `test_processing.TestProcessData.*` |
| apply_filter_preset "none" returns copy | `processing._ref_apply_filter_preset` | `test_processing.TestFilterPreset.test_none_returns_copy` |
| ProfileChain haversine concat | `model.ProfileChain._concat` | `test_chaining.TestProfileChainConcat.*` |
| compute_fix_positions algorithm | `geometry_export.compute_fix_positions` | `test_geometry_export.TestComputeFixPositions.*` |
| Reprojection header rules (delay preserved) | `io_segy.reproject_one` | `test_reprojection.TestReprojectOne.test_delay_preserved` |
| out_sc=-10M geographic / -100 projected | `io_segy.reproject_one` | `test_reprojection.TestReprojectOne.test_scalar_and_unit_*` |
| bin + text[0] copied | `io_segy.reproject_one` | `test_reprojection.TestReprojectOne.test_text_header_copied` |
| TraceNumber sequential for chains | `io_segy.reproject_chain` | `test_reprojection.TestReprojectChain.test_trace_number_sequential` |
| Partial output deleted on error | `io_segy._try_delete` | `test_reprojection.TestReprojectOne.test_partial_output_deleted_on_error` |

---

## 3. Library choices

| Library | Why | Measured / expected benefit | Install impact | Regression status |
|---------|-----|----------------------------|---------------|-------------------|
| `numpy` | Core numerical arrays | required | ~20 MB | N/A |
| `scipy` | FFT, filters, linalg.solve_toeplitz | required | ~40 MB | N/A |
| `segyio` | SEG-Y I/O | required | ~2 MB | N/A |
| `pyproj` | CRS transformations | required | ~15 MB | N/A |
| `matplotlib` | Colormaps (lazy, core); Agg renderer (viz) | required for viz | ~50 MB | N/A |
| `Pillow` | High-quality RGBA resize in viz/render.py | 2-pass Lanczos resizing for high-DPI export | ~5 MB | N/A |

**Not added (considered)**:
- `pyfftw`: No optimized path implemented yet; gated by regression test.
- `numba`: Decon inner loop bottleneck not measured on real data yet.
- `dask/zarr`: Chunked loading deferred; lazy/mmap approach sufficient for current scope.
- `PyQtGraph`: **Forbidden** in core and viz; may only be evaluated for a future GUI backend.
- `QtCore`: Discouraged (heavy dep for no numeric gain); stdlib threading used instead.

---

## 4. Optimisations implemented

None in this release. Every public function dispatches to its `_ref_` path.

The following optimisations are **planned** and will be enabled after their
regression tests pass (tolerance documented inline):

| Optimisation | Planned tolerance | Status |
|-------------|------------------|--------|
| Vectorised reprojection (bulk Transformer.transform) | coord match atol=1e-6 m | Not implemented |
| Avoid full-matrix copy in trace loading | identical output | Not implemented |
| Lazy/mmap trace loading for large files | identical output | Not implemented |
| scipy.fft workers / pyfftw for spectrum | allclose atol=1e-5 | Not implemented |
| Reduce transient memory in chain concat | identical output | Not implemented |

---

## 5. Gaps filled

| Gap | Source in GUI | Module | Function |
|-----|--------------|--------|---------|
| Spectrum computation | `TopasSUITE._compute_spectrum` (~L2709) | `core/spectrum.py` | `_ref_compute_spectrum`, `compute_spectrum` |
| Navline SHP writer | `TopasSUITE._write_navline_shp` (~L2427) | `core/geometry_export.py` | `write_navline_shp` |
| Navline GeoJSON writer | `TopasSUITE._write_navline_geojson` (~L2543) | `core/geometry_export.py` | `write_navline_geojson` |
| Navline CSV writer | `TopasSUITE._write_navline_csv` (~L2586) | `core/geometry_export.py` | `write_navline_csv` |
| CRS resolution/validation | `TopasSUITE._resolve_crs` | `core/coordinates.py` | `resolve_crs`, `validate_crs` |
| SegyMetadata (header-only) | — (new, not in monolith) | `core/model.py` | `SegyMetadata`, `load_metadata` |
| FIX writer name disambiguation | Name clash in topas_core.py | `core/geometry_export.py` | `write_fix_points_*` vs `write_navline_*` |

---

## 6. Carried-forward risks / known limitations

| Risk | Location | Status |
|------|----------|--------|
| `delay_ms` uses `trace[0]` only | `io_segy._populate_profile_from_file`, `processing.time_window` | Preserved, NOT fixed. Per-trace delays available via `delays` array. |
| Projected-CRS `dist_km` assumes metres | `io_segy._dist_km` | Preserved, NOT fixed. Documented in docstring. |
| `boundaries_km` vs `dist_km` offset | `model.ProfileChain._concat` | Preserved. `boundaries_km` uses `total_km` cumulative sum, not `dist_km`. |
| Scalar null-handling divergence | `io_segy._scalar_fac(0) = 1.0` | Consistent with monolith; no change. |
| `apply_filter_preset("none")` semantics | `processing._ref_apply_filter_preset` | Returns `data.copy()` — explicit new array, never aliased. |
| Geographic reprojection `out_sc = -10_000_000` | `io_segy.reproject_one/chain` | `SEG-Y SourceGroupScalar` is a 16-bit field; -10_000_000 overflows, segyio stores truncated value (27008). Preserved from monolith. On reload this truncated scalar gives wrong coordinate scaling — this is a monolith bug, carried forward unchanged. |

---

## 7. Threading / GIL / cancellation

- `_parallel_apply` uses `ThreadPoolExecutor` (same as monolith). The GIL
  limits CPU-bound work per thread; this is acceptable since most heavy ops
  (FFT via numpy/scipy, filter via scipy) release the GIL internally.
- `CancelToken` is wired in `reproject_one` and `reproject_chain` — cancel
  checks occur every trace.
- All other operations (`process_*_data`, `compute_spectrum`, rendering) do
  not currently check the cancel token (deferred as per spec).
- Peak memory for chain concat: `ProfileChain._concat` holds all profiles'
  data simultaneously plus the concatenated matrix. Peak ≈ 2× concatenated
  size. No streaming or chunking implemented; large chains (>GB) should use
  the CLI's join-chain path which never loads all data into Python memory.

---

## 8. Open questions

1. **join-chain pure-join semantics**: `--src X --dst X` passes an identity
   transform through `reproject_chain`. `Transformer.from_crs(X, X)` is a
   no-op numerically but still rewrites every header. A true copy-only
   code path (no pyproj call) would be faster for the pure-join case.
   **Awaiting confirmation of intended semantics.**

2. **Projected units validation**: `dist_km` silently assumes projected CRS
   units are in metres. Should we raise `CRSError` or warn when the CRS unit
   is not metres?

3. **pyfftw integration**: No benchmark has been run against real TOPAS data.
   Should this be prioritised before the next release?
