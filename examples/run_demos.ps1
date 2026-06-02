# examples/run_demos.ps1 — Test runner for topassuite (PowerShell)
#
# Run from the REPO ROOT with the conda env active:
#   conda activate topassuite
#   cd T:\workspace_py\TopasSuite-Python
#   .\examples\run_demos.ps1
#
# IMPORTANT: always use bare 'python' (not an absolute path or Start-Process)
# so it inherits the active conda environment's DLLs correctly on Windows.
#
# Options:
#   .\examples\run_demos.ps1              run everything
#   .\examples\run_demos.ps1 -Solo tests  unit tests only
#   .\examples\run_demos.ps1 -Solo demo   synthetic data examples only
#   .\examples\run_demos.ps1 -Solo real   real data exports only
#   .\examples\run_demos.ps1 -Solo accel  show acceleration status only

param([string]$Solo = "todo")

# ── Ensure we are at repo root ─────────────────────────────────────────────────
$REPO = Split-Path -Parent $PSScriptRoot
Set-Location $REPO

# ── Paths ─────────────────────────────────────────────────────────────────────
$DEMO_IN  = "examples\_demo_in"
$DEMO_OUT = "examples\_demo_out"
$REAL_IN  = "examples\_real_data_in\sgy_files\line_example_1"
$REAL_OUT = "examples\_real_data_out"

$F1 = "$REAL_IN\20260527061612.sgy"
$F2 = "$REAL_IN\20260527070833.sgy"
$F3 = "$REAL_IN\20260527072542.sgy"
$F4 = "$REAL_IN\20260527074151.sgy"
$F5 = "$REAL_IN\20260527080151.sgy"
$F6 = "$REAL_IN\20260527082722.sgy"
$ALL = @($F1, $F2, $F3, $F4, $F5, $F6)

New-Item -ItemType Directory -Force -Path $DEMO_OUT | Out-Null
New-Item -ItemType Directory -Force -Path $REAL_OUT | Out-Null

function titulo($txt) { Write-Host "`n=== $txt ===" -ForegroundColor Cyan }
function ok($txt)     { Write-Host "    OK: $txt"  -ForegroundColor Green }

# ============================================================
# ACCEL — hardware acceleration status
# ============================================================
if ($Solo -in @("todo","accel")) {
    titulo "ACCEL — hardware/software acceleration"
    python -m topassuite.cli.main accel
}

# ============================================================
# BLOQUE 1 — UNIT TESTS
# ============================================================
if ($Solo -in @("todo","tests")) {
    titulo "TESTS — full suite (~1 min)"
    pytest tests/ -q
    ok "tests completed"
}

# ============================================================
# BLOQUE 2 — SYNTHETIC DATA EXAMPLES
# ============================================================
if ($Solo -in @("todo","demo")) {

    titulo "DEMO 1/6 — Generate synthetic SEG-Y data"
    python examples/generate_demo_data.py

    titulo "DEMO 2/6 — Inspect metadata (header-only + full)"
    python examples/example_inspect.py

    titulo "DEMO 3/6 — Export image (profile + chain)"
    python examples/example_export_image.py

    titulo "DEMO 4/6 — Frequency spectrum"
    python examples/example_spectrum.py

    titulo "DEMO 5/6 — Reprojection + join-chain"
    python examples/example_reproject.py

    titulo "DEMO 6/6 — Navline + FIX marks"
    python examples/example_navline_fix.py

    titulo "Demo outputs"
    Get-ChildItem $DEMO_OUT | Select-Object Name, @{N="KB";E={[int]($_.Length/1KB)}}
}

# ============================================================
# BLOQUE 3 — REAL DATA (line_example_1, 6 files)
# ============================================================
if ($Solo -in @("todo","real")) {

    titulo "REAL 1 — Info: metadata of all 6 files (text)"
    python -m topassuite.cli.main info @ALL

    titulo "REAL 2 — Info: JSON output"
    python -m topassuite.cli.main info --json @ALL

    titulo "REAL 3 — Export: single profile, quality=screen, Greys"
    python -m topassuite.cli.main export-image $F1 `
        --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality screen --px-per-trace 1 `
        --out "$REAL_OUT\F1_screen.png"

    titulo "REAL 4 — Export: single profile, quality=print, with ticks"
    python -m topassuite.cli.main export-image $F1 `
        --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality print --px-per-trace 1 `
        --x-tick 5 --t-tick 50 --grid `
        --out "$REAL_OUT\F1_print_ticks.png"

    titulo "REAL 5 — Export: single profile, no-axes (exact pixels)"
    python -m topassuite.cli.main export-image $F1 `
        --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality print --px-per-trace 1 --no-axes `
        --out "$REAL_OUT\F1_print_noaxes.tif"

    titulo "REAL 6 — Export: full chain, quality=print, with timing"
    python -m topassuite.cli.main export-image @ALL `
        --chain --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality print --px-per-trace 1 `
        --x-tick 10 --t-tick 50 --grid `
        --title "L01 - TOPAS Sub-bottom Profiler - 27/05/2026" `
        --timeit --out "$REAL_OUT\L01_print.png"

    titulo "REAL 7 — Export: full chain, no-axes TIFF (plotter)"
    python -m topassuite.cli.main export-image @ALL `
        --chain --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality high --px-per-trace 1 --no-axes `
        --timeit --format tif --out "$REAL_OUT\L01_high_noaxes.tif"

    titulo "REAL 8 — Export: full chain, PDF natural size (plotter)"
    python -m topassuite.cli.main export-image @ALL `
        --chain --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality print --px-per-trace 1 --no-axes `
        --format pdf --pdf-page auto `
        --timeit --out "$REAL_OUT\L01_print_plotter.pdf"

    titulo "REAL 9 — Export: full chain, PDF A3 for desk printing"
    python -m topassuite.cli.main export-image @ALL `
        --chain --preset envelope --cmap Greys --agc --align --fill-zero `
        --quality high --px-per-trace 1 --no-axes `
        --format pdf --pdf-page A3 `
        --timeit --out "$REAL_OUT\L01_high_A3.pdf"

    titulo "REAL 10 — Spectrum: first file"
    python -m topassuite.cli.main spectrum $F1 `
        --out "$REAL_OUT\F1_spectrum.png"

    titulo "REAL 11 — Navline: chain as GeoJSON"
    python -m topassuite.cli.main navline @ALL `
        --chain --format geojson `
        --out "$REAL_OUT\navline_L01.geojson"

    titulo "REAL 12 — Navline: chain as Shapefile"
    python -m topassuite.cli.main navline @ALL `
        --chain --format shp `
        --out "$REAL_OUT\navline_L01.shp"

    titulo "REAL 13 — FIX marks: chain, every 10 min, Shapefile"
    python -m topassuite.cli.main fix @ALL `
        --chain --interval 10 --format shp `
        --out "$REAL_OUT\fix_L01.shp"

    titulo "REAL 14 — Join-chain: pure join (same CRS)"
    python -m topassuite.cli.main join-chain @ALL `
        --src EPSG:4326 --dst EPSG:4326 `
        --out "$REAL_OUT\L01_joined.sgy"

    titulo "REAL 15 — Join-chain: reproject to UTM 29N"
    python -m topassuite.cli.main join-chain @ALL `
        --src EPSG:4326 --dst EPSG:32629 `
        --out "$REAL_OUT\L01_joined_UTM29N.sgy"

    titulo "Real data outputs"
    Get-ChildItem $REAL_OUT |
        Select-Object Name, @{N="KB";E={[int]($_.Length/1KB)}} |
        Format-Table -AutoSize
}
