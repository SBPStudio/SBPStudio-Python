# run_demos.ps1 — Comandos de prueba para topassuite
#
# Requisito previo (ejecutar UNA sola vez antes de lanzar este script):
#   conda activate topassuite
#   cd T:\workspace_py\TopasSuite-Python
#
# IMPORTANTE: usar 'python' (no ruta absoluta ni Start-Process) para que
# herede correctamente el entorno de conda activo.
#
# Uso:
#   .\run_demos.ps1              <- ejecuta todo
#   .\run_demos.ps1 -Solo real   <- solo bloque de datos reales
#   .\run_demos.ps1 -Solo demo   <- solo bloque de datos sinteticos
#   .\run_demos.ps1 -Solo tests  <- solo tests

param([string]$Solo = "todo")

# ── Rutas ────────────────────────────────────────────────────────────────────
$DEMO_IN   = "examples\_demo_in"
$DEMO_OUT  = "examples\_demo_out"
$REAL_IN   = "examples\_real_data_in\sgy_files\line_example_1"
$REAL_OUT  = "examples\_real_data_out"

$F1 = "$REAL_IN\20260527061612.sgy"
$F2 = "$REAL_IN\20260527070833.sgy"
$F3 = "$REAL_IN\20260527072542.sgy"
$F4 = "$REAL_IN\20260527074151.sgy"
$F5 = "$REAL_IN\20260527080151.sgy"
$F6 = "$REAL_IN\20260527082722.sgy"
$ALL_REAL  = @($F1, $F2, $F3, $F4, $F5, $F6)

New-Item -ItemType Directory -Force -Path $DEMO_OUT | Out-Null
New-Item -ItemType Directory -Force -Path $REAL_OUT | Out-Null

function titulo($txt) { Write-Host "`n=== $txt ===" -ForegroundColor Cyan }
function ok($txt)     { Write-Host "    OK: $txt"  -ForegroundColor Green }

# ============================================================
# BLOQUE 1 — TESTS
# ============================================================
if ($Solo -in @("todo","tests")) {

    titulo "TESTS — suite completa (95 tests, ~1 min)"
    pytest tests/ -q
    ok "tests completados"

}

# ============================================================
# BLOQUE 2 — DATOS SINTETICOS
# ============================================================
if ($Solo -in @("todo","demo")) {

    titulo "DEMO 1/6 — Generar datos sinteticos"
    python examples/generate_demo_data.py

    titulo "DEMO 2/6 — Inspeccionar metadatos"
    python examples/example_inspect.py

    titulo "DEMO 3/6 — Exportar imagen (perfil + cadena)"
    python examples/example_export_image.py

    titulo "DEMO 4/6 — Espectro de frecuencia"
    python examples/example_spectrum.py

    titulo "DEMO 5/6 — Reproyeccion + join chain"
    python examples/example_reproject.py

    titulo "DEMO 6/6 — Navline + FIX marks"
    python examples/example_navline_fix.py

    titulo "Resultados demo"
    Get-ChildItem $DEMO_OUT | Select-Object Name, @{N="KB";E={[int]($_.Length/1KB)}}

}

# ============================================================
# BLOQUE 3 — DATOS REALES
# ============================================================
if ($Solo -in @("todo","real")) {

    titulo "REAL 1 — INFO texto (metadatos de los 6 archivos)"
    python -m topassuite.cli.main info @ALL_REAL

    titulo "REAL 2 — INFO JSON"
    python -m topassuite.cli.main info --json @ALL_REAL

    titulo "REAL 3 — Imagen: un perfil sin procesado"
    python -m topassuite.cli.main export-image $F1 `
        --out "$REAL_OUT\F1_raw.png"

    titulo "REAL 4 — Imagen: un perfil con bandpass + AGC"
    python -m topassuite.cli.main export-image $F1 `
        --bandpass 200 6000 --agc `
        --out "$REAL_OUT\F1_agc.png"

    titulo "REAL 5 — Imagen: perfil con delays + FIX cada 5 min"
    python -m topassuite.cli.main export-image $F1 `
        --bandpass 200 6000 --agc --align --fix 5 `
        --out "$REAL_OUT\F1_full.png"

    titulo "REAL 6 — Imagen: cadena completa (6 archivos)"
    python -m topassuite.cli.main export-image @ALL_REAL `
        --chain --bandpass 200 6000 --agc --fix 5 `
        --out "$REAL_OUT\L01_cadena.png"

    titulo "REAL 7 — Espectro: primer archivo"
    python -m topassuite.cli.main spectrum $F1 `
        --out "$REAL_OUT\F1_spectrum.png"

    titulo "REAL 8 — Espectro: todos los archivos"
    foreach ($f in $ALL_REAL) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($f)
        python -m topassuite.cli.main spectrum $f `
            --out "$REAL_OUT\spectrum_$stem.png"
    }

    titulo "REAL 9 — Navline: GeoJSON de un perfil"
    python -m topassuite.cli.main navline $F1 `
        --format geojson `
        --out "$REAL_OUT\navline_F1.geojson"

    titulo "REAL 10 — Navline: CSV de un perfil"
    python -m topassuite.cli.main navline $F1 `
        --format csv `
        --out "$REAL_OUT\navline_F1.csv"

    titulo "REAL 11 — Navline: cadena completa como Shapefile"
    python -m topassuite.cli.main navline @ALL_REAL `
        --chain --format shp `
        --out "$REAL_OUT\navline_L01.shp"

    titulo "REAL 12 — Navline: cadena completa como GeoJSON"
    python -m topassuite.cli.main navline @ALL_REAL `
        --chain --format geojson `
        --out "$REAL_OUT\navline_L01.geojson"

    titulo "REAL 13 — FIX marks: GeoJSON, primer archivo, cada 5 min"
    python -m topassuite.cli.main fix $F1 `
        --interval 5 --format geojson `
        --out "$REAL_OUT\fix_F1.geojson"

    titulo "REAL 14 — FIX marks: cadena completa, Shapefile, cada 10 min"
    python -m topassuite.cli.main fix @ALL_REAL `
        --chain --interval 10 --format shp `
        --out "$REAL_OUT\fix_L01.shp"

    titulo "REAL 15 — Join-chain: pure join (sin cambiar coordenadas)"
    python -m topassuite.cli.main join-chain @ALL_REAL `
        --src EPSG:4326 --dst EPSG:4326 `
        --out "$REAL_OUT\L01_joined.sgy"

    titulo "REAL 16 — Join-chain + reproyeccion a UTM 29N"
    python -m topassuite.cli.main join-chain @ALL_REAL `
        --src EPSG:4326 --dst EPSG:32629 `
        --out "$REAL_OUT\L01_joined_UTM29N.sgy"

    titulo "Resultados datos reales"
    Get-ChildItem $REAL_OUT | Select-Object Name, @{N="KB";E={[int]($_.Length/1KB)}}

}
