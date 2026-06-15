"""
constants.py — Domain constants shared across the sbp_studio package.

All constants are module-level, never mutated at runtime. They mirror
exactly the values used in the GUI monolith (TopasSUITE.py).
"""
from __future__ import annotations

from typing import Dict

# Colormap palette: human-readable label → matplotlib cmap name.
CMAPS: Dict[str, str] = {
    "Blanco / Negro": "Greys",
    "Viridis":        "viridis",
    "Inferno":        "inferno",
    "Jet":            "jet",
    "Terrain":        "terrain",
}

# SEG-Y CoordinateUnits field values.
COORD_UNITS: Dict[int, str] = {1: "m/ft", 2: "arc-sec", 3: "decimal°", 4: "DMS"}

# Named CRS presets: human label → EPSG string (or "CUSTOM").
PRESETS_CRS: Dict[str, str] = {
    "WGS84 Geográfico (EPSG:4326)":      "EPSG:4326",
    "ED50 Geográfico (EPSG:4230)":        "EPSG:4230",
    "ETRS89 Geográfico (EPSG:4258)":      "EPSG:4258",
    "SIRGAS 2000 (EPSG:4674)":            "EPSG:4674",
    "UTM WGS84 Zona 29N (EPSG:32629)":   "EPSG:32629",
    "UTM WGS84 Zona 30N (EPSG:32630)":   "EPSG:32630",
    "UTM WGS84 Zona 31N (EPSG:32631)":   "EPSG:32631",
    "UTM WGS84 Zona 18S (EPSG:32718)":   "EPSG:32718",
    "UTM WGS84 Zona 19S (EPSG:32719)":   "EPSG:32719",
    "UTM WGS84 Zona 20S (EPSG:32720)":   "EPSG:32720",
    "UTM WGS84 Zona 21S (EPSG:32721)":   "EPSG:32721",
    "UTM WGS84 Zona 22S (EPSG:32722)":   "EPSG:32722",
    "ETRS89 UTM 30N (EPSG:25830)":       "EPSG:25830",
    "ED50 UTM 30N (EPSG:23030)":         "EPSG:23030",
    "Personalizado…":                     "CUSTOM",
}

# Filter preset keys: human label → internal key.
FILTER_PRESETS: Dict[str, str] = {
    "── Sin filtro preestablecido ──":       "none",
    "Envelope  (amplitud instantánea)":      "envelope",
    "Fase instantánea":                      "inst_phase",
    "Frecuencia instantánea":                "inst_freq",
    "Cos(fase instantánea)":                 "cos_phase",
    "Atributo de similitud (traza ±1)":      "similarity",
    "Sobel vertical  (bordes horizontales)": "sobel_v",
    "Laplaciano  (realce bordes)":           "laplacian",
    "High-Boost  (nitidez ×2)":              "highboost",
    "Mediana  (ventana 5 muestras)":         "median5",
    "Wiener  (ventana 7 muestras)":          "wiener7",
    "Suavizado Gaussiano  (σ=1)":            "gauss1",
    "SBP banda estrecha  (2–4 kHz)":       "topas_narrow",
    "SBP banda ancha    (1–8 kHz)":        "topas_wide",
    "SBP alta resolución (4–10 kHz)":      "topas_hires",
    "Derivada temporal  (realce flancos)":   "derivative",
    "Integración temporal  (suavizado)":     "integral",
}

# Tooltip descriptions for each filter preset key.
FILTER_DESCRIPTIONS: Dict[str, str] = {
    "none":         "Sin procesado adicional.",
    "envelope":     "Amplitud de la señal analítica (módulo de Hilbert). "
                    "Resalta reflectores brillantes independientemente de la polaridad.",
    "inst_phase":   "Fase instantánea (ángulo de la señal analítica). "
                    "Resalta continuidad de reflectores incluso con baja amplitud.",
    "inst_freq":    "Frecuencia instantánea (derivada de la fase). "
                    "Sensible a cambios litológicos y efectos de fluidos.",
    "cos_phase":    "Coseno de la fase instantánea. Versión normalizada de la fase, "
                    "facilita correlación lateral de reflectores.",
    "similarity":   "Similitud entre traza central y sus vecinas (±1). "
                    "Resalta discontinuidades y fallas.",
    "sobel_v":      "Gradiente vertical de Sobel. Detecta bordes horizontales "
                    "(reflectores con cambio brusco de amplitud).",
    "laplacian":    "Laplaciano 2D. Realza límites de capas y finas láminas.",
    "highboost":    "Filtro High-Boost (original + 2× paso-alto). "
                    "Incrementa nitidez sin amplificar ruido de alta frecuencia.",
    "median5":      "Filtro de mediana a lo largo de cada traza (ventana 5 muestras). "
                    "Elimina spikes sin borrar reflectores.",
    "wiener7":      "Filtro de Wiener adaptativo (ventana 7). "
                    "Óptimo para reducción de ruido aditivo gaussiano.",
    "gauss1":       "Suavizado Gaussiano (σ=1 muestra). "
                    "Atenúa ruido de alta frecuencia suavemente.",
    "topas_narrow": "Pasabanda Butterworth 4º orden, 2–4 kHz. "
                    "Óptimo para SBP en modo sub-bottom profiler estrecho.",
    "topas_wide":   "Pasabanda Butterworth 4º orden, 1–8 kHz. "
                    "Configuración amplia para perfiles SBP estándar.",
    "topas_hires":  "Pasabanda Butterworth 4º orden, 4–10 kHz. "
                    "Alta resolución para sedimentos superficiales.",
    "derivative":   "Derivada primera a lo largo del tiempo. "
                    "Realza flancos de reflexiones y discontinuidades.",
    "integral":     "Integración temporal acumulada. "
                    "Suaviza la señal, útil para visualizar tendencias de baja frecuencia.",
}
