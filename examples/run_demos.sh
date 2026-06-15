#!/usr/bin/env bash
# examples/run_demos.sh — Test runner for sbp_studio (bash / Linux / macOS)
#
# Run from the REPO ROOT with the conda env active:
#   conda activate sbp_studio
#   cd /path/to/SBP-Studio
#   bash examples/run_demos.sh
#
# Options (first argument):
#   (none)   run everything
#   tests    unit tests only
#   demo     synthetic data examples only
#   real     real data exports only
#   accel    show acceleration status only
#
# NOTE: real data paths assume Linux-style separators. Edit REAL_IN
# below if your data lives elsewhere.

set -euo pipefail

SOLO="${1:-todo}"

# ── Repo root (script lives in examples/) ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

# ── Paths ─────────────────────────────────────────────────────────────────────
DEMO_IN="examples/_demo_in"
DEMO_OUT="examples/_demo_out"
REAL_IN="examples/_real_data_in/sgy_files/line_example_1"
REAL_OUT="examples/_real_data_out"

F1="$REAL_IN/20260527061612.sgy"
F2="$REAL_IN/20260527070833.sgy"
F3="$REAL_IN/20260527072542.sgy"
F4="$REAL_IN/20260527074151.sgy"
F5="$REAL_IN/20260527080151.sgy"
F6="$REAL_IN/20260527082722.sgy"

mkdir -p "$DEMO_OUT" "$REAL_OUT"

titulo() { echo; echo "=== $* ==="; }
ok()     { echo "    OK: $*"; }

# ============================================================
# ACCEL — hardware acceleration status
# ============================================================
if [[ "$SOLO" == "todo" || "$SOLO" == "accel" ]]; then
    titulo "ACCEL — hardware/software acceleration"
    python -m sbp_studio.cli.main accel
fi

# ============================================================
# BLOQUE 1 — UNIT TESTS
# ============================================================
if [[ "$SOLO" == "todo" || "$SOLO" == "tests" ]]; then
    titulo "TESTS — full suite (~1 min)"
    pytest tests/ -q
    ok "tests completed"
fi

# ============================================================
# BLOQUE 2 — SYNTHETIC DATA EXAMPLES
# ============================================================
if [[ "$SOLO" == "todo" || "$SOLO" == "demo" ]]; then

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
    ls -lh "$DEMO_OUT" 2>/dev/null || echo "(empty)"
fi

# ============================================================
# BLOQUE 3 — REAL DATA (line_example_1, 6 files)
# ============================================================
if [[ "$SOLO" == "todo" || "$SOLO" == "real" ]]; then

    titulo "REAL 1 — Info: metadata of all 6 files (text)"
    python -m sbp_studio.cli.main info "$F1" "$F2" "$F3" "$F4" "$F5" "$F6"

    titulo "REAL 2 — Info: JSON output"
    python -m sbp_studio.cli.main info --json "$F1" "$F2" "$F3" "$F4" "$F5" "$F6"

    titulo "REAL 3 — Export: single profile, quality=screen, Greys"
    python -m sbp_studio.cli.main export-image "$F1" \
        --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality screen --px-per-trace 1 \
        --out "$REAL_OUT/F1_screen.png"

    titulo "REAL 4 — Export: single profile, quality=print, with ticks"
    python -m sbp_studio.cli.main export-image "$F1" \
        --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality print --px-per-trace 1 \
        --x-tick 5 --t-tick 50 --grid \
        --out "$REAL_OUT/F1_print_ticks.png"

    titulo "REAL 5 — Export: single profile, no-axes TIFF (exact pixels)"
    python -m sbp_studio.cli.main export-image "$F1" \
        --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality print --px-per-trace 1 --no-axes \
        --out "$REAL_OUT/F1_print_noaxes.tif"

    titulo "REAL 6 — Export: full chain, quality=print, timed"
    python -m sbp_studio.cli.main export-image \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality print --px-per-trace 1 \
        --x-tick 10 --t-tick 50 --grid \
        --title "L01 - SBP Sub-bottom Profiler - 27/05/2026" \
        --timeit --out "$REAL_OUT/L01_print.png"

    titulo "REAL 7 — Export: full chain, no-axes TIFF (plotter)"
    python -m sbp_studio.cli.main export-image \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality high --px-per-trace 1 --no-axes \
        --timeit --format tif --out "$REAL_OUT/L01_high_noaxes.tif"

    titulo "REAL 8 — Export: full chain, PDF natural size (plotter)"
    python -m sbp_studio.cli.main export-image \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality print --px-per-trace 1 --no-axes \
        --format pdf --pdf-page auto \
        --timeit --out "$REAL_OUT/L01_print_plotter.pdf"

    titulo "REAL 9 — Export: full chain, PDF A3 for desk printing"
    python -m sbp_studio.cli.main export-image \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --preset envelope --cmap Greys --agc --align --fill-zero \
        --quality high --px-per-trace 1 --no-axes \
        --format pdf --pdf-page A3 \
        --timeit --out "$REAL_OUT/L01_high_A3.pdf"

    titulo "REAL 10 — Spectrum: first file"
    python -m sbp_studio.cli.main spectrum "$F1" \
        --out "$REAL_OUT/F1_spectrum.png"

    titulo "REAL 11 — Navline: chain as GeoJSON"
    python -m sbp_studio.cli.main navline \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --format geojson \
        --out "$REAL_OUT/navline_L01.geojson"

    titulo "REAL 12 — Navline: chain as Shapefile"
    python -m sbp_studio.cli.main navline \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --format shp \
        --out "$REAL_OUT/navline_L01.shp"

    titulo "REAL 13 — FIX marks: chain, every 10 min, Shapefile"
    python -m sbp_studio.cli.main fix \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --chain --interval 10 --format shp \
        --out "$REAL_OUT/fix_L01.shp"

    titulo "REAL 14 — Join-chain: pure join (same CRS)"
    python -m sbp_studio.cli.main join-chain \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --src EPSG:4326 --dst EPSG:4326 \
        --out "$REAL_OUT/L01_joined.sgy"

    titulo "REAL 15 — Join-chain: reproject to UTM 29N"
    python -m sbp_studio.cli.main join-chain \
        "$F1" "$F2" "$F3" "$F4" "$F5" "$F6" \
        --src EPSG:4326 --dst EPSG:32629 \
        --out "$REAL_OUT/L01_joined_UTM29N.sgy"

    titulo "Real data outputs"
    ls -lh "$REAL_OUT" 2>/dev/null || echo "(empty)"
fi
