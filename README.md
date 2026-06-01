# TOPAS Suite — Python Package

Headless SEG-Y seismic processing core, visualisation layer, and CLI for
Kongsberg TOPAS sub-bottom profiler data.

The original GUI monolith (`TopasSUITE.py`) is **unchanged** and continues
to work independently.

---

## Package structure

```
topassuite/
  core/           # All computation — GUI-free, importable headless
    constants.py  # CMAPS, PRESETS_CRS, FILTER_PRESETS, COORD_UNITS
    tasks.py      # Exceptions, ProgressCallback, CancelToken
    model.py      # SegyMetadata, SegyProfile, ProfileChain
    io_segy.py    # load_metadata, load_profile, reproject_one, reproject_chain
    processing.py # apply_predictive_decon, apply_filter_preset, process_*_data
    spectrum.py   # compute_spectrum → SpectrumResult
    coordinates.py# resolve_crs, validate_crs
    chaining.py   # detect_chains
    coloring.py   # colormapped_rgba (lazy matplotlib import)
    geometry_export.py  # write_fix_points_* / write_navline_* (stdlib only)
    _backends.py  # GPU/CuPy detection, worker count
  viz/
    render.py     # Headless Matplotlib Agg figures
  cli/
    main.py       # python -m topassuite.cli.main
    commands.py   # Per-subcommand implementations
examples/
  generate_demo_data.py
  example_*.py
tests/
  make_synthetic_segy.py
  conftest.py
  test_*.py
```

### Core / Viz / CLI boundary

- `core/` imports **only** non-GUI libraries. No tkinter, no PyQt/PySide, no
  matplotlib.pyplot/Figure at module level. The only matplotlib usage in core
  is a **lazy import** of color tables inside `coloring.py`.
- `viz/` may use matplotlib (Agg backend) and Pillow. It is **never** imported
  by `core/`.
- `cli/` imports both `core` and `viz`; no GUI imports, no DISPLAY required.

---

## Two-track design

Every function with an optimized implementation follows this contract:

| Symbol | Meaning |
|--------|---------|
| `_ref_<name>` | Reference path — frozen, byte-identical to monolith |
| `<name>` | Public API — dispatches to optimized path if validated, else `_ref_` |

An **optimized path is only activated after its regression test passes**:
```python
np.testing.assert_allclose(public_output, _ref_output, atol=<documented_tolerance>)
```

Current status: all public functions dispatch to their `_ref_` implementations.
No optimized paths have been introduced yet.

---

## Install

```bash
conda env create -f environment.yml
conda activate topassuite
```

Or with pip:
```bash
pip install numpy scipy segyio pyproj matplotlib Pillow
```

---

## CLI usage

```bash
# Metadata inspection
python -m topassuite.cli.main info file.sgy [--json]

# Reprojection
python -m topassuite.cli.main reproject file.sgy \
    --src EPSG:4326 --dst EPSG:32630 [--unit-hint 2] [--out-dir DIR]

# Join chain into one SEG-Y (--src == --dst → pure join, no coord change)
python -m topassuite.cli.main join-chain f1.sgy f2.sgy \
    --src EPSG:4326 --dst EPSG:4326 [--gap-km 2.0] [--out joined.sgy]

# Export image
python -m topassuite.cli.main export-image file.sgy \
    [--chain] [--preset envelope] [--bandpass 200 4000] [--agc] \
    [--tvg 0.5] [--align] [--clip 99] [--cmap Viridis] [--invert] \
    [--fix 5] [--dpi 150] [--format png] [--out out.png]

# Spectrum
python -m topassuite.cli.main spectrum file.sgy [--format png] [--out out.png]

# Navigation track
python -m topassuite.cli.main navline file.sgy \
    [--chain] --format geojson [--crs EPSG:32630] [--attrs] [--out nav.geojson]

# FIX marks
python -m topassuite.cli.main fix file.sgy \
    [--chain] --interval 5 --format geojson [--out fix.geojson]
```

---

## Run examples

```bash
python examples/generate_demo_data.py   # create synthetic SEG-Y data
python examples/example_inspect.py
python examples/example_export_image.py
python examples/example_spectrum.py
python examples/example_reproject.py
python examples/example_navline_fix.py
```

---

## Run tests

```bash
pytest tests/ -v
```

To run only the CLI smoke tests:
```bash
pytest tests/test_cli_smoke.py -v
```

---

## Known limitations (carried forward from monolith)

- `delay_ms` uses `trace[0]` only; per-trace delay variability is exposed via
  the `delays` array but the scalar is fixed to the first trace.
- Projected-CRS distance in `dist_km` assumes coordinates are in **metres**;
  this is not validated.
- `boundaries_km` in `ProfileChain` is computed from per-profile `total_km`
  cumulative sum, not from `dist_km` directly — may differ by the inter-profile
  gap distance at join points.

---

## Open questions

1. **join-chain pure-join semantics**: `join-chain --src X --dst X` passes an
   identity transform through `reproject_chain`. Coordinates are transformed
   through `Transformer.from_crs(X, X)`, which should be a no-op but still
   reads/rewrites every header. Is a true copy-only code path preferred for
   the `src == dst` case? (Currently not implemented — please confirm.)

2. **Projected units assumption**: Should `dist_km` validate that projected CRS
   units are metres rather than silently dividing by 1000?

