#!/usr/bin/env python3
"""
TOPAS Suite — Gestión, Visualización y Reproyección de perfiles SEG-Y
Versión multi-perfil integrada.
Pestañas: [A] Visualizador  [B] Reproyector  (C próximamente: RAW→SGY)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, sys
import concurrent.futures as _cf
from pathlib import Path

# ── Dependencias ─────────────────────────────────────────────────────────────
MISSING = []
# Mapa: nombre_importable -> nombre_pip
_DEPS = {
    "numpy":      "numpy",
    "segyio":     "segyio",
    "matplotlib": "matplotlib",
    "scipy":      "scipy",
    "pyproj":     "pyproj",
    "PIL":        "Pillow",   # Pillow se importa como PIL
}
for import_name, pip_name in _DEPS.items():
    try:
        __import__(import_name)
    except ImportError:
        MISSING.append(pip_name)

if MISSING:
    import subprocess
    print(f"Instalando: {MISSING}")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--break-system-packages", "-q"] + MISSING)
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import segyio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from scipy import signal as sp_signal
import scipy.linalg
from pyproj import CRS, Transformer


# ── Compatibilidad de colormaps entre versiones de matplotlib ─────────────────
def _get_colormap(name: str):
    """
    Obtiene un colormap compatible con todas las versiones de matplotlib:
      - matplotlib >= 3.5 → matplotlib.colormaps[name]
      - matplotlib <  3.5 → matplotlib.cm.get_cmap(name)
    """
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return matplotlib.cm.get_cmap(name)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMA Y CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":       "#12141a",
    "panel":    "#1a1d26",
    "sidebar":  "#141720",
    "accent":   "#1e2535",
    "highlight":"#2a3a5c",
    "bright":   "#4d9de0",
    "warn":     "#e94560",
    "ok":       "#3ddc97",
    "text":     "#dce3ee",
    "sub":      "#6a7a96",
    "entry":    "#0e1118",
    "sel":      "#253555",
}

# Paletas estrictamente secuenciales
CMAPS = {
    "Blanco / Negro": "Greys",
    "Viridis":        "viridis",
    "Inferno":        "inferno",
    "Jet":            "jet",
    "Terrain":        "terrain",
}

COORD_UNITS = {1: "m/ft", 2: "arc-sec", 3: "decimal°", 4: "DMS"}

PRESETS_CRS = {
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════
class SegyProfile:
    """Carga y cachea un perfil SEG-Y completo."""

    def __init__(self, filepath: str):
        self.path  = filepath
        self.name  = Path(filepath).name
        self.stem  = Path(filepath).stem
        self.error = None
        self._load()

    def _load(self):
        try:
            with segyio.open(self.path, ignore_geometry=True) as f:
                self.n_traces = f.tracecount
                self.ns       = f.samples.size
                self.dt_us    = int(f.bin[segyio.BinField.Interval])

                h0 = f.header[0]
                # int() con fallback a 0 por si segyio devuelve None en archivos no conformes
                self.scalar_coord = int(h0[segyio.TraceField.SourceGroupScalar] or 0)
                self.scalar_elev  = int(h0[segyio.TraceField.ElevationScalar]   or 0)
                self.coord_unit   = int(h0[segyio.TraceField.CoordinateUnits]   or 0)
                
                # Carga de todos los headers en una sola pasada al disco
                _fields = [
                    segyio.TraceField.DelayRecordingTime,
                    segyio.TraceField.SourceX,
                    segyio.TraceField.SourceY,
                    segyio.TraceField.SourceWaterDepth,
                    segyio.TraceField.YearDataRecorded,
                    segyio.TraceField.DayOfYear,
                    segyio.TraceField.HourOfDay,
                    segyio.TraceField.MinuteOfHour,
                    segyio.TraceField.SecondOfMinute,
                ]
                _hdr = {fld: np.asarray(f.attributes(fld)[:]) for fld in _fields}

                delays_raw = _hdr[segyio.TraceField.DelayRecordingTime]
                self.delays   = delays_raw
                self.min_delay = float(np.min(delays_raw))
                self.max_delay = float(np.max(delays_raw))
                self.delay_ms  = int(delays_raw[0])

                sc   = self.scalar_coord
                fac  = (1.0/abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)
                sxs  = _hdr[segyio.TraceField.SourceX].astype(float)  * fac
                sys_ = _hdr[segyio.TraceField.SourceY].astype(float)  * fac

                if self.coord_unit == 2:          # arc-seconds → degrees
                    self.lons = sxs  / 3600.0
                    self.lats = sys_ / 3600.0
                else:
                    self.lons = sxs
                    self.lats = sys_

                es   = self.scalar_elev
                efac = (1.0/abs(es)) if es < 0 else (float(es) if es > 0 else 1.0)
                self.water_depth = _hdr[segyio.TraceField.SourceWaterDepth].astype(float) * efac

                yrs  = _hdr[segyio.TraceField.YearDataRecorded]
                doys = _hdr[segyio.TraceField.DayOfYear]
                hrs  = _hdr[segyio.TraceField.HourOfDay]
                mins = _hdr[segyio.TraceField.MinuteOfHour]
                secs = _hdr[segyio.TraceField.SecondOfMinute]

                # Padding de DOY corregido a 3 dígitos para mantener ordenación alfanumérica geométrica
                self.timestamps = [
                    f"{int(y)}-DOY{int(d):03d} {int(h):02d}:{int(m):02d}:{int(s):02d}"
                    for y, d, h, m, s in zip(yrs, doys, hrs, mins, secs)]

                self.data = f.trace.raw[:].T.astype(np.float32)

            # amplitude cache — computed once from the already-loaded matrix
            self.amp_max = np.max(np.abs(self.data), axis=0)   # shape (n_traces,)

            self.t_ms    = self.delay_ms + np.arange(self.ns) * self.dt_us / 1000.0
            self.dur_ms  = self.ns * self.dt_us / 1000.0
            self.clip_p99 = float(np.percentile(np.abs(self.data), 99))

            dlat = np.diff(self.lats)
            dlon = np.diff(self.lons)
            # Comprobar si las coordenadas parecen geográficas (grados).
            # Si son proyectadas (metros/pies), la fórmula esférica daría
            # distancias completamente erróneas → usar distancia euclídea.
            _is_geo = self.coord_unit in (2, 3) or (
                -180 <= float(self.lons[0]) <= 180 and
                -90  <= float(self.lats[0]) <= 90)
            if _is_geo:
                lat_m = np.mean(self.lats)
                d = np.sqrt((dlat * 111.32)**2 +
                            (dlon * 111.32 * np.cos(np.radians(lat_m)))**2)
            else:
                # CRS proyectado: distancia euclídea en metros → convertir a km
                d = np.sqrt(dlat**2 + dlon**2) / 1000.0
            self.dist_km  = np.concatenate([[0.0], np.cumsum(d)])
            self.total_km = float(self.dist_km[-1])

            self.detected_crs, self.crs_notes = self._detect_crs()

        except Exception as e:
            self.error = str(e)

    def _detect_crs(self):
        if self.coord_unit == 2:
            if -180 <= self.lons[0] <= 180 and -90 <= self.lats[0] <= 90:
                return "EPSG:4326", ["✔ Arc-seconds → WGS84 detectado"]
        elif self.coord_unit == 3:
            if -180 <= self.lons[0] <= 180 and -90 <= self.lats[0] <= 90:
                return "EPSG:4326", ["✔ Grados decimales → WGS84 detectado"]
        return None, ["⚠ No detectado automáticamente"]

    def summary(self):
        if self.error:
            return f"ERROR: {self.error}"
        return (f"{self.n_traces} trazas · {self.dur_ms:.0f} ms · "
                f"{self.total_km:.1f} km · "
                f"WD {np.nanmean(self.water_depth):.0f} m")


# ═══════════════════════════════════════════════════════════════════════════════
# ACELERACIÓN GPU (CuPy / CUDA) — opcional, fallback silencioso a NumPy/SciPy
# ═══════════════════════════════════════════════════════════════════════════════
def _detect_gpu():
    """
    Intenta importar CuPy (wrapper CUDA para NumPy).
    Devuelve (cupy_module, True) si hay GPU disponible, (numpy, False) si no.
    """
    try:
        import cupy as cp
        _ = cp.array([1.0])          # test rápido de asignación en VRAM
        return cp, True
    except Exception:
        return np, False

_xp, _GPU = _detect_gpu()          # _xp = cupy o numpy; _GPU = bool

# ─── Motor de procesado paralelo ────────────────────────────────────────────
# Número de núcleos físicos disponibles (sin saturar el SO)
_N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

def _parallel_apply(fn, data: np.ndarray, *args, n_workers: int = _N_WORKERS, **kwargs) -> np.ndarray:
    """
    Divide 'data' (ns × n_traces) en n_workers bloques de columnas,
    aplica fn(bloque, *args, **kwargs) en paralelo con ProcessPoolExecutor
    y reensambla el resultado.

    Heurística: si n_traces < 64 o el array es pequeño (<4 MB) no merece
    el overhead del multiproceso — se ejecuta en el hilo actual.
    """
    ns, n_traces = data.shape
    if n_traces < 64 or data.nbytes < 4 * 1024 * 1024 or n_workers <= 1:
        return fn(data, *args, **kwargs)

    # Dividir en bloques de columnas lo más iguales posible
    chunk_size = max(1, n_traces // n_workers)
    slices = [slice(j, min(j + chunk_size, n_traces))
              for j in range(0, n_traces, chunk_size)]
    chunks = [data[:, sl] for sl in slices]

    results = [None] * len(chunks)
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(fn, ch, *args, **kwargs): k
                for k, ch in enumerate(chunks)}
        for fut in _cf.as_completed(futs):
            k = futs[fut]
            results[k] = fut.result()

    return np.concatenate(results, axis=1)



def _colormapped_image_parallel(data: np.ndarray, cmap_name: str,
                                 vmin: float, vmax: float,
                                 n_workers: int = _N_WORKERS) -> np.ndarray:
    """
    Aplica colormap a 'data' (ns × n_traces) dividiendo en franjas verticales
    procesadas en paralelo.  Devuelve array RGBA uint8 (ns, n_traces, 4).

    Si hay GPU disponible, intenta usar CuPy para la normalización antes de
    enviar a los workers de CPU (que sólo hacen la colorización final).
    """
    ns, n_traces = data.shape

    # ── GPU: normalización numérica ──────────────────────────────────────────
    if _GPU:
        import cupy as cp
        try:
            g    = cp.asarray(data, dtype=cp.float32)
            lo   = cp.float32(vmin)
            hi   = cp.float32(vmax) if vmax != vmin else cp.float32(vmin + 1e-9)
            norm_data = cp.clip((g - lo) / (hi - lo), 0.0, 1.0)
            data_norm = cp.asnumpy(norm_data)          # (ns, n_traces) float32 in [0,1]
        except Exception:
            data_norm = None
    else:
        data_norm = None

    # Fallback numpy normalisation
    if data_norm is None:
        lo = float(vmin); hi = float(vmax) if vmax != vmin else vmin + 1e-9
        data_norm = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    # ── CPU multi-núcleo: colorización ──────────────────────────────────────
    if n_traces < 128 or n_workers <= 1:
        cmap = _get_colormap(cmap_name)
        return (cmap(data_norm) * 255).astype(np.uint8)

    chunk_size = max(1, n_traces // n_workers)
    slices  = [slice(j, min(j + chunk_size, n_traces))
               for j in range(0, n_traces, chunk_size)]
    tiles   = [data_norm[:, sl] for sl in slices]

    results = [None] * len(tiles)
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_render_rgba_tile_norm, t, cmap_name): k
                for k, t in enumerate(tiles)}
        for fut in _cf.as_completed(futs):
            k = futs[fut]
            results[k] = fut.result()

    return np.concatenate(results, axis=1)   # (ns, n_traces, 4)


def _render_rgba_tile_norm(tile_norm: np.ndarray, cmap_name: str) -> np.ndarray:
    """Coloriza un tile ya normalizado [0,1] → RGBA uint8."""
    cmap = _get_colormap(cmap_name)
    return (cmap(tile_norm) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESADO SÍSMICO
# ═══════════════════════════════════════════════════════════════════════════════
def apply_predictive_decon(data: np.ndarray, dt_us: int, op_len_ms: float, gap_ms: float, white_noise_pct: float) -> np.ndarray:
    """
    Aplica deconvolución predictiva traza a traza (Wiener-Levinson).
    Paraleliza sobre columnas usando _parallel_apply.
    """
    ns, n_traces = data.shape
    dt_ms = dt_us / 1000.0
    nl = max(2, int(op_len_ms / dt_ms))
    gap = max(1, int(gap_ms / dt_ms))
    mu = white_noise_pct / 100.0
    max_lag = nl + gap
    n_fft = 2**int(np.ceil(np.log2(2 * ns - 1)))

    def _decon_block(block: np.ndarray) -> np.ndarray:
        _ns, _nt = block.shape
        out = np.zeros_like(block)
        for i in range(_nt):
            tr = block[:, i]
            X  = np.fft.fft(tr, n_fft)
            r  = np.fft.ifft(X * np.conj(X)).real[:max_lag]
            if r[0] == 0:
                out[:, i] = tr
                continue
            r[0] *= (1.0 + mu)
            try:
                a = scipy.linalg.solve_toeplitz(r[0:nl], r[gap:max_lag])
            except scipy.linalg.LinAlgError:
                out[:, i] = tr
                continue
            f = np.zeros(max_lag)
            f[0] = 1.0
            f[gap:] = -a
            out[:, i] = sp_signal.lfilter(f, [1.0], tr)
        return out

    return _parallel_apply(_decon_block, data)


# Nombre visible → clave interna
FILTER_PRESETS = {
    "── Sin filtro preestablecido ──": "none",
    # ── Atributos sísmicos (estilo Kingdom Suite) ────────────────────────────
    "Envelope  (amplitud instantánea)":      "envelope",
    "Fase instantánea":                      "inst_phase",
    "Frecuencia instantánea":                "inst_freq",
    "Cos(fase instantánea)":                 "cos_phase",
    "Atributo de similitud (traza ±1)":      "similarity",
    # ── Realce de reflectores ────────────────────────────────────────────────
    "Sobel vertical  (bordes horizontales)": "sobel_v",
    "Laplaciano  (realce bordes)":           "laplacian",
    "High-Boost  (nitidez ×2)":              "highboost",
    # ── Suavizado / reducción de ruido ──────────────────────────────────────
    "Mediana  (ventana 5 muestras)":         "median5",
    "Wiener  (ventana 7 muestras)":          "wiener7",
    "Suavizado Gaussiano  (σ=1)":            "gauss1",
    # ── Pasabanda estándar TOPAS ─────────────────────────────────────────────
    "TOPAS banda estrecha  (2–4 kHz)":       "topas_narrow",
    "TOPAS banda ancha    (1–8 kHz)":        "topas_wide",
    "TOPAS alta resolución (4–10 kHz)":      "topas_hires",
    # ── Derivada / integración ───────────────────────────────────────────────
    "Derivada temporal  (realce flancos)":   "derivative",
    "Integración temporal  (suavizado)":     "integral",
}

# Tooltips descriptivos (se muestran en la etiqueta de estado)
FILTER_DESCRIPTIONS = {
    "none":         "Sin procesado adicional.",
    "envelope":     "Amplitud de la señal analítica (módulo de Hilbert). "
                    "Resalta reflectores brillantes independientemente de la polaridad. ",
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
                    "Óptimo para TOPAS en modo sub-bottom profiler estrecho.",
    "topas_wide":   "Pasabanda Butterworth 4º orden, 1–8 kHz. "
                    "Configuración amplia para perfiles TOPAS estándar.",
    "topas_hires":  "Pasabanda Butterworth 4º orden, 4–10 kHz. "
                    "Alta resolución para sedimentos superficiales.",
    "derivative":   "Derivada primera a lo largo del tiempo. "
                    "Realza flancos de reflexiones y discontinuidades.",
    "integral":     "Integración temporal acumulada. "
                    "Suaviza la señal, útil para visualizar tendencias de baja frecuencia.",
}


def apply_filter_preset(data: np.ndarray, key: str, dt_us: int) -> np.ndarray:
    """
    Aplica el filtro preestablecido identificado por `key` sobre `data` (ns x n_traces).
    Devuelve un array float32 del mismo shape.
    dt_us: intervalo de muestreo en microsegundos.
    Usa CuPy (GPU) cuando esta disponible para operaciones pesadas;
    siempre devuelve un ndarray numpy estandar.
    """
    if key == "none" or not key:
        return data

    from scipy.signal import hilbert, medfilt, wiener
    from scipy.ndimage import gaussian_filter, laplace, sobel

    fs  = 1e6 / dt_us
    out = data.astype(np.float32)

    # -- Atributos de Hilbert --
    if key in ("envelope", "inst_phase", "inst_freq", "cos_phase"):
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.signal import hilbert as cu_hilbert
                an = cu_hilbert(cp.asarray(out), axis=0)
                if key == "envelope":
                    out = cp.asnumpy(cp.abs(an)).astype(np.float32)
                elif key == "inst_phase":
                    out = cp.asnumpy(cp.angle(an)).astype(np.float32)
                elif key == "cos_phase":
                    out = cp.asnumpy(cp.cos(cp.angle(an))).astype(np.float32)
                elif key == "inst_freq":
                    phase = cp.unwrap(cp.angle(an), axis=0)
                    freq  = cp.diff(phase, axis=0, prepend=phase[:1, :]) / (2 * np.pi * dt_us * 1e-6)
                    out   = cp.asnumpy(cp.clip(freq, 0, fs / 2)).astype(np.float32)
            except Exception:
                analytic = hilbert(out, axis=0)
                if key == "envelope":
                    out = np.abs(analytic).astype(np.float32)
                elif key == "inst_phase":
                    out = np.angle(analytic).astype(np.float32)
                elif key == "cos_phase":
                    out = np.cos(np.angle(analytic)).astype(np.float32)
                elif key == "inst_freq":
                    phase = np.unwrap(np.angle(analytic), axis=0)
                    freq  = np.diff(phase, axis=0, prepend=phase[:1, :]) / (2 * np.pi * dt_us * 1e-6)
                    out   = np.clip(freq, 0, fs / 2).astype(np.float32)
        else:
            analytic = hilbert(out, axis=0)
            if key == "envelope":
                out = np.abs(analytic).astype(np.float32)
            elif key == "inst_phase":
                out = np.angle(analytic).astype(np.float32)
            elif key == "cos_phase":
                out = np.cos(np.angle(analytic)).astype(np.float32)
            elif key == "inst_freq":
                phase = np.unwrap(np.angle(analytic), axis=0)
                freq  = np.diff(phase, axis=0, prepend=phase[:1, :]) / (2 * np.pi * dt_us * 1e-6)
                out   = np.clip(freq, 0, fs / 2).astype(np.float32)

    # -- Similitud -- (GPU vectorizada, o CPU paralela por bloques)
    elif key == "similarity":
        if _GPU:
            import cupy as cp
            try:
                g    = cp.asarray(out)
                n_tr = g.shape[1]
                sim  = cp.ones_like(g)
                # Vectorizado sin bucle Python: desplazamiento de columnas
                if n_tr > 2:
                    left  = g[:, :-2]
                    mid   = g[:, 1:-1]
                    right = g[:, 2:]
                    num = left * mid + mid * right
                    den = (cp.sqrt(left**2  + mid**2 + 1e-12) *
                           cp.sqrt(mid**2   + right**2 + 1e-12))
                    sim[:, 1:-1] = num / (den + 1e-12)
                out = cp.asnumpy(sim).astype(np.float32)
            except Exception:
                # CPU fallback vectorizado
                sim = np.ones_like(out)
                if out.shape[1] > 2:
                    left  = out[:, :-2]; mid = out[:, 1:-1]; right = out[:, 2:]
                    num = left * mid + mid * right
                    den = (np.sqrt(left**2 + mid**2 + 1e-12) *
                           np.sqrt(mid**2  + right**2 + 1e-12))
                    sim[:, 1:-1] = num / (den + 1e-12)
                out = sim.astype(np.float32)
        else:
            sim = np.ones_like(out)
            if out.shape[1] > 2:
                left  = out[:, :-2]; mid = out[:, 1:-1]; right = out[:, 2:]
                num = left * mid + mid * right
                den = (np.sqrt(left**2 + mid**2 + 1e-12) *
                       np.sqrt(mid**2  + right**2 + 1e-12))
                sim[:, 1:-1] = num / (den + 1e-12)
            out = sim.astype(np.float32)

    # -- Realce de bordes -- (GPU donde posible, CPU paralela si no)
    elif key == "sobel_v":
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.ndimage import sobel as cu_sobel
                out = cp.asnumpy(cu_sobel(cp.asarray(out, dtype=cp.float64), axis=0)).astype(np.float32)
            except Exception:
                out = _parallel_apply(lambda b: sobel(b.astype(np.float64), axis=0).astype(np.float32), out)
        else:
            out = _parallel_apply(lambda b: sobel(b.astype(np.float64), axis=0).astype(np.float32), out)
    elif key == "laplacian":
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.ndimage import laplace as cu_laplace
                out = cp.asnumpy(cu_laplace(cp.asarray(out, dtype=cp.float64))).astype(np.float32)
            except Exception:
                out = laplace(out.astype(np.float64)).astype(np.float32)
        else:
            out = laplace(out.astype(np.float64)).astype(np.float32)
    elif key == "highboost":
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.ndimage import gaussian_filter as cu_gauss
                g = cp.asarray(out, dtype=cp.float64)
                sm = cu_gauss(g, sigma=1.0)
                out = cp.asnumpy(g + 2.0 * (g - sm)).astype(np.float32)
            except Exception:
                smooth = gaussian_filter(out.astype(np.float64), sigma=1.0)
                out    = (out + 2.0 * (out - smooth)).astype(np.float32)
        else:
            smooth = gaussian_filter(out.astype(np.float64), sigma=1.0)
            out    = (out + 2.0 * (out - smooth)).astype(np.float32)

    # -- Reducción de ruido --
    elif key == "median5":
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.ndimage import median_filter as cu_med
                out = cp.asnumpy(cu_med(cp.asarray(out), size=(5, 1))).astype(np.float32)
            except Exception:
                out = _parallel_apply(lambda b: medfilt(b, kernel_size=(5, 1)).astype(np.float32), out)
        else:
            out = _parallel_apply(lambda b: medfilt(b, kernel_size=(5, 1)).astype(np.float32), out)
    elif key == "wiener7":
        def _wiener_block(block: np.ndarray) -> np.ndarray:
            res = np.empty_like(block)
            for j in range(block.shape[1]):
                res[:, j] = wiener(block[:, j].astype(np.float64), mysize=7)
            return res.astype(np.float32)
        # CuPy no tiene wiener nativo → siempre CPU paralela
        out = _parallel_apply(_wiener_block, out)
    elif key == "gauss1":
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.ndimage import gaussian_filter as cu_gauss
                out = cp.asnumpy(cu_gauss(cp.asarray(out).astype(cp.float64),
                                          sigma=(1.0, 0.0))).astype(np.float32)
            except Exception:
                out = gaussian_filter(out.astype(np.float64), sigma=(1.0, 0.0)).astype(np.float32)
        else:
            out = gaussian_filter(out.astype(np.float64), sigma=(1.0, 0.0)).astype(np.float32)

    # -- Pasabanda TOPAS --
    elif key in ("topas_narrow", "topas_wide", "topas_hires"):
        bands = {"topas_narrow": (2000, 4000),
                 "topas_wide":   (1000, 8000),
                 "topas_hires":  (4000, 10000)}
        flo, fhi = bands[key]
        flo = max(10, flo);  fhi = min(fs / 2 - 1, fhi)
        if flo < fhi:
            sos = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")
            out = sp_signal.sosfilt(sos, out, axis=0).astype(np.float32)

    # -- Derivada / integral --
    elif key == "derivative":
        if _GPU:
            import cupy as cp
            try:
                out = cp.asnumpy(cp.gradient(cp.asarray(out).astype(cp.float64), axis=0)).astype(np.float32)
            except Exception:
                out = np.gradient(out.astype(np.float64), axis=0).astype(np.float32)
        else:
            out = np.gradient(out.astype(np.float64), axis=0).astype(np.float32)
    elif key == "integral":
        if _GPU:
            import cupy as cp
            try:
                out = cp.asnumpy(cp.cumsum(cp.asarray(out).astype(cp.float64), axis=0)).astype(np.float32)
            except Exception:
                out = np.cumsum(out.astype(np.float64), axis=0).astype(np.float32)
        else:
            out = np.cumsum(out.astype(np.float64), axis=0).astype(np.float32)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
class ProfileChain:
    """
    Agrupa N perfiles SEG-Y contiguos en un único objeto concatenado.
    La detección de continuidad se basa en:
      1. Distancia geográfica entre el último punto del perfil i y el primero del i+1.
      2. Intervalo de muestreo dt idéntico (misma configuración de adquisición).
    Los datos se concatenan horizontalmente (eje de trazas) y las distancias
    acumuladas se encadenan correctamente para que el eje X sea continuo.
    """
    # Umbral máximo en km entre extremos de dos perfiles para considerarlos contiguos
    GAP_KM_MAX = 2.0

    def __init__(self, profiles: list):
        """profiles: lista ordenada de SegyProfile ya cargados sin error."""
        self.profiles   = profiles
        self.name       = " ⛓ ".join(p.stem for p in profiles)
        if len(profiles) > 1:
            self.label = f"[{len(profiles)} perfiles]  {profiles[0].name} … {profiles[-1].name}"
        else:
            self.label = f"[1 perfil]  {profiles[0].name}"
        self._concat()

    def _concat(self):
        # Datos sísmicos concatenados por trazas
        self.data = np.concatenate([p.data for p in self.profiles], axis=1)

        # Delays
        self.delays = np.concatenate([p.delays for p in self.profiles])
        self.min_delay = float(np.min(self.delays))
        self.max_delay = float(np.max(self.delays))

        # Coordenadas
        self.lons = np.concatenate([p.lons for p in self.profiles])
        self.lats = np.concatenate([p.lats for p in self.profiles])

        # Water depth
        self.water_depth = np.concatenate([p.water_depth for p in self.profiles])

        # Timestamps
        self.timestamps = []
        for p in self.profiles:
            self.timestamps.extend(p.timestamps)

        # Distancias acumuladas continuas.
        # Se incluye el gap geográfico real entre perfiles contiguos para que
        # el eje X sea continuo y geográficamente correcto, sin saltos artificiales.
        segments = []
        offset = 0.0
        for idx_p, p in enumerate(self.profiles):
            segments.append(p.dist_km + offset)
            offset += p.total_km
            # Añadir la distancia del gap al siguiente perfil (si lo hay)
            if idx_p + 1 < len(self.profiles):
                nxt = self.profiles[idx_p + 1]
                gap_km = ProfileChain._haversine_km(
                    float(p.lons[-1]), float(p.lats[-1]),
                    float(nxt.lons[0]), float(nxt.lats[0]))
                offset += gap_km
        self.dist_km  = np.concatenate(segments)
        self.total_km = float(self.dist_km[-1])

        # Metadatos del primero (dt, ns, delay deben ser iguales)
        p0 = self.profiles[0]
        self.dt_us    = p0.dt_us
        self.ns       = p0.ns
        self.dur_ms   = p0.dur_ms
        self.delay_ms = p0.delay_ms
        self.n_traces = self.data.shape[1]

        # Fronteras entre perfiles (distancia km donde empieza cada uno)
        self.boundaries_km = []
        d = 0.0
        for p in self.profiles[:-1]:
            d += p.total_km
            self.boundaries_km.append(d)

        # clip global p99
        self.clip_p99 = float(np.percentile(np.abs(self.data), 99))

    @staticmethod
    def _haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    @classmethod
    def detect(cls, profiles: list, gap_km: float = None) -> list:
        """
        Recibe una lista de SegyProfile válidos y devuelve una lista de
        ProfileChain con las agrupaciones de perfiles contiguos detectadas.
        Perfiles con distintos dt_us se mantienen separados.
        Si hay un único perfil válido, devuelve una lista con una cadena de
        un solo elemento. Devuelve lista vacía solo si la lista de entrada
        está vacía o todos los perfiles tienen error.
        """
        if not profiles:
            return []

        gap = gap_km if gap_km is not None else cls.GAP_KM_MAX

        # Ordenar por timestamp del primer registro
        valid = [p for p in profiles if not p.error]
        if len(valid) < 2:
            return [cls([p]) for p in valid] if valid else []

        # Ordenar cronológicamente por timestamp
        def _ts_key(p):
            return p.timestamps[0] if p.timestamps else ""
        valid_sorted = sorted(valid, key=_ts_key)

        # Agrupar greedy: añadir al grupo actual si contiguos, sino nuevo grupo
        groups = []
        current = [valid_sorted[0]]

        for nxt in valid_sorted[1:]:
            prev = current[-1]
            # Condición 1: mismo dt
            same_dt = (prev.dt_us == nxt.dt_us)
            # Condición 2: distancia entre último punto de prev y primero de nxt
            gap_d = cls._haversine_km(
                prev.lons[-1], prev.lats[-1],
                nxt.lons[0],  nxt.lats[0])
            if same_dt and gap_d <= gap:
                current.append(nxt)
            else:
                groups.append(current)
                current = [nxt]
        groups.append(current)

        return [cls(g) for g in groups]


class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, window, callback=None):
        self._custom_zoom_callback = callback
        super().__init__(canvas, window, pack_toolbar=False)
        
    def release_zoom(self, event):
        super().release_zoom(event)
        if self._custom_zoom_callback:
            self.window.after(10, self._custom_zoom_callback)

    def release_pan(self, event):
        super().release_pan(event)
        if self._custom_zoom_callback:
            self.window.after(10, self._custom_zoom_callback)
            
    def home(self, *args, **kwargs):
        super().home(*args, **kwargs)
        if self._custom_zoom_callback:
            self.window.after(10, self._custom_zoom_callback)

    def back(self, *args, **kwargs):
        super().back(*args, **kwargs)
        if self._custom_zoom_callback:
            self.window.after(10, self._custom_zoom_callback)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        if self._custom_zoom_callback:
            self.window.after(10, self._custom_zoom_callback)

# ═══════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class TopasSuite(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("TOPAS Suite")
        self.configure(bg=C["bg"])
        self.geometry("1480x900")
        self.minsize(1100, 680)

        # Tipado corregido para evitar crasheo TypeError en versiones < Python 3.9
        self.profiles: "dict[str, SegyProfile]" = {}
        self.active_profile: "SegyProfile | None" = None
        self.chains: "list[ProfileChain]" = []
        self.active_chain: "ProfileChain | None" = None
        self._last_fixes:          list = []
        self._last_fixes_sd:       object = None
        self._last_chain_fixes:    list = []
        self._last_chain_fixes_ch: object = None

        # Buffer del log de reproyección — inicializado aquí para evitar
        # race condition si un hilo llama a _rlog antes de la primera escritura
        self._rlog_buf: list = []
        self._rlog_pending: bool = False

        # IDs de render — evitan que getattr() devuelva valores inesperados
        # si un callback de zoom se dispara antes del primer render
        self._render_id:            int  = 0
        self._chain_render_id:      int  = 0
        self._spec_render_id:       int  = 0
        self._spec_chain_render_id: int  = 0

        # Datos actuales del render activo
        self._current_profile_data: object = None
        self._current_chain_data:   object = None

        # Figuras activas
        self._profile_fig:          object = None
        self._chain_fig:            object = None

        # Referencias de scroll y canvas (evitan AttributeError en callbacks rápidos)
        self._profile_canvas_obj:   object = None
        self._profile_scroll_cv:    object = None
        self._profile_inner_frame:  object = None
        self._chain_canvas_obj:     object = None
        self._chain_scroll_cv:      object = None

        # Referencia de tiempo del último render (para zoom coherente)
        self._last_profile_t0:      float  = 0.0
        self._last_chain_t0:        float  = 0.0

        self._style()
        self._build_ui()
        self.after(80, self._center)

    def _center(self):
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h   = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")

    # ── Estilos ttk ──────────────────────────────────────────────────────────
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",           background=C["bg"],    borderwidth=0)
        s.configure("TNotebook.Tab",       background=C["accent"],foreground=C["sub"],
                    font=("Courier New", 9), padding=[12, 4])
        s.map("TNotebook.Tab",
              background=[("selected", C["panel"])],
              foreground=[("selected", C["text"])])
        s.configure("Treeview",            background=C["entry"], foreground=C["text"],
                    fieldbackground=C["entry"], font=("Courier New", 8), rowheight=20)
        s.configure("Treeview.Heading",    background=C["accent"],foreground=C["bright"],
                    font=("Courier New", 8, "bold"))
        s.map("Treeview",                  background=[("selected", C["sel"])])
        s.configure("TScrollbar",          background=C["accent"],troughcolor=C["bg"])
        s.configure("TCombobox",
                    fieldbackground=C["entry"], background=C["accent"],
                    foreground=C["text"],
                    selectbackground=C["entry"],
                    selectforeground=C["text"],
                    insertcolor=C["text"],
                    arrowcolor=C["text"],
                    bordercolor=C["accent"])
        # Cubrir todos los estados interactivos: readonly, focus, hover, disabled
        s.map("TCombobox",
              fieldbackground=[("readonly", C["entry"]),
                               ("disabled", C["panel"]),
                               ("active",   C["entry"]),
                               ("focus",    C["entry"])],
              foreground=[("readonly",          C["text"]),
                          ("disabled",          C["sub"]),
                          ("focus",             C["text"]),
                          ("active",            C["text"])],
              selectbackground=[("readonly",    C["entry"]),
                                ("focus",       C["entry"]),
                                ("active",      C["entry"])],
              selectforeground=[("readonly",    C["text"]),
                                ("focus",       C["text"]),
                                ("active",      C["text"])],
              background=[("active",            C["accent"]),
                          ("pressed",           C["accent"])],
              arrowcolor=[("disabled",          C["sub"]),
                          ("pressed",           C["bright"]),
                          ("active",            C["bright"])])
        s.configure("TSeparator",          background=C["accent"])

    # ════════════════════════════════════════════════════════════════════════
    # LAYOUT PRINCIPAL
    # ════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Topbar ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=C["accent"], height=48)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="◈  TOPAS  SUITE",
                 bg=C["accent"], fg=C["bright"],
                 font=("Courier New", 14, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(topbar, text="SEG-Y · Multi-perfil · Visualización · Reproyección",
                 bg=C["accent"], fg=C["sub"],
                 font=("Courier New", 8)).pack(side="left", padx=4)

        self._btn(topbar, "📂  Añadir perfiles", self._add_profiles,
                  C["bright"], side="right", padx=12, pady=10)

        # ── Cuerpo: panel de perfiles izq + contenido dcha ──────────────────
        body = tk.PanedWindow(self, orient="horizontal",
                              bg=C["bg"], sashwidth=6,
                              sashrelief="flat", sashpad=0)
        body.pack(fill="both", expand=True)

        left  = self._build_profile_panel(body)
        right = self._build_right(body)
        body.add(left,  minsize=260, width=280)
        body.add(right, minsize=700)

        # ── Barra inferior: estado + barra de progreso ──────────────────────
        bottom_bar = tk.Frame(self, bg=C["accent"])
        bottom_bar.pack(fill="x", side="bottom")

        # ── Fila superior: icono de tarea + texto de estado + badge GPU ───
        top_row = tk.Frame(bottom_bar, bg=C["accent"])
        top_row.pack(fill="x")

        # Badge GPU fijo al extremo derecho
        gpu_color = C["ok"] if _GPU else C["sub"]
        gpu_text  = "⚡ GPU" if _GPU else "○ CPU"
        tk.Label(top_row, text=gpu_text,
                 bg=C["accent"], fg=gpu_color,
                 font=("Courier New", 7, "bold"), padx=10
                 ).pack(side="right", pady=3)

        # Spinner (etiqueta rotatoria que indica tarea activa)
        self._spinner_lbl = tk.Label(top_row, text="",
                                     bg=C["accent"], fg=C["bright"],
                                     font=("Courier New", 8, "bold"), padx=6)
        self._spinner_lbl.pack(side="left", pady=3)
        self._spinner_chars = ["◐", "◓", "◑", "◒"]
        self._spinner_idx   = 0
        self._spinner_job   = None   # after() id para cancelar

        self.statusbar = tk.Label(top_row,
                                  text="Listo  —  añade uno o más perfiles SEG-Y.",
                                  bg=C["accent"], fg=C["sub"],
                                  font=("Courier New", 8), anchor="w", padx=2)
        self.statusbar.pack(side="left", fill="x", expand=True, pady=3)

        # ── Fila inferior: barra de progreso a ancho completo ─────────────
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Topas.Horizontal.TProgressbar",
                        troughcolor=C["bg"],
                        background=C["bright"],
                        borderwidth=0, relief="flat",
                        thickness=3)
        self._progress_bar = ttk.Progressbar(
            bottom_bar, orient="horizontal", mode="determinate", maximum=100,
            style="Topas.Horizontal.TProgressbar")
        # No se hace pack aquí — _progress_start lo hace cuando se necesita

        # Contador de tareas activas (thread-safe vía after())
        self._active_tasks = 0

    # ── Panel izquierdo: doble lista de perfiles y cadenas ──────────────────
    def _build_profile_panel(self, parent):
        pw = tk.PanedWindow(parent, orient="vertical", bg=C["bg"], 
                            sashwidth=6, sashrelief="flat", sashpad=0)

        # ── BLOQUE SUPERIOR: PERFILES CARGADOS ──
        frame_profs = tk.Frame(pw, bg=C["sidebar"])
        
        hdr = tk.Frame(frame_profs, bg=C["accent"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="PERFILES CARGADOS",
                 bg=C["accent"], fg=C["bright"],
                 font=("Courier New", 8, "bold"),
                 padx=10, pady=6).pack(side="left")

        lf = tk.Frame(frame_profs, bg=C["sidebar"])
        lf.pack(fill="both", expand=True, pady=(2, 0))

        vsb = ttk.Scrollbar(lf, orient="vertical")
        self.prof_list = tk.Listbox(
            lf, bg=C["entry"], fg=C["text"],
            selectbackground=C["sel"], selectforeground=C["bright"],
            font=("Courier New", 9), relief="flat",
            highlightthickness=0, activestyle="none",
            yscrollcommand=vsb.set)
        vsb.config(command=self.prof_list.yview)
        vsb.pack(side="right", fill="y")
        self.prof_list.pack(fill="both", expand=True)
        self.prof_list.bind("<<ListboxSelect>>", self._on_profile_select)

        bf = tk.Frame(frame_profs, bg=C["sidebar"], pady=6)
        bf.pack(fill="x")
        self._btn(bf, "＋ Añadir",  self._add_profiles,  C["bright"],  side="left",  padx=6)
        self._btn(bf, "✖ Quitar",   self._remove_profile, C["warn"],   side="left",  padx=2)
        self._btn(bf, "✖✖ Limpiar", self._clear_profiles, C["accent"], side="right", padx=6)

        sep = tk.Frame(frame_profs, bg=C["accent"], height=1)
        sep.pack(fill="x")

        self.lbl_prof_info = tk.Label(
            frame_profs, text="", bg=C["sidebar"], fg=C["sub"],
            font=("Courier New", 7), anchor="w",
            justify="left", wraplength=230, padx=8, pady=6)
        self.lbl_prof_info.pack(fill="x")

        pw.add(frame_profs, minsize=200)

        # ── BLOQUE INFERIOR: CADENAS DETECTADAS ──
        frame_chains = tk.Frame(pw, bg=C["sidebar"])
        
        hdr_ch = tk.Frame(frame_chains, bg=C["accent"])
        hdr_ch.pack(fill="x")
        tk.Label(hdr_ch, text="CADENAS DETECTADAS",
                 bg=C["accent"], fg=C["bright"],
                 font=("Courier New", 8, "bold"),
                 padx=10, pady=6).pack(side="left")

        ctrl_ch = tk.Frame(frame_chains, bg=C["sidebar"], pady=4)
        ctrl_ch.pack(fill="x", padx=6)
        tk.Label(ctrl_ch, text="Umbral(km):", bg=C["sidebar"], fg=C["sub"],
                 font=("Courier New", 7)).pack(side="left")
        self.chain_gap_var = tk.DoubleVar(value=2.0)
        tk.Spinbox(ctrl_ch, textvariable=self.chain_gap_var,
                   from_=0.1, to=50.0, increment=0.5, width=4,
                   bg=C["entry"], fg=C["text"], insertbackground=C["text"],
                   relief="flat", font=("Courier New", 7),
                   buttonbackground=C["accent"]).pack(side="left", padx=4)
        self._btn(ctrl_ch, "🔍 Detectar", self._run_chain_detection,
                  C["bright"], side="right", padx=0)

        lf_ch = tk.Frame(frame_chains, bg=C["sidebar"])
        lf_ch.pack(fill="both", expand=True, pady=(2, 0))

        vsb_ch = ttk.Scrollbar(lf_ch, orient="vertical")
        self.chain_list = tk.Listbox(
            lf_ch, bg=C["entry"], fg=C["text"],
            selectbackground=C["sel"], selectforeground=C["bright"],
            font=("Courier New", 9), relief="flat",
            highlightthickness=0, activestyle="none",
            yscrollcommand=vsb_ch.set)
        vsb_ch.config(command=self.chain_list.yview)
        vsb_ch.pack(side="right", fill="y")
        self.chain_list.pack(fill="both", expand=True)
        self.chain_list.bind("<<ListboxSelect>>", self._on_chain_select)

        sep2 = tk.Frame(frame_chains, bg=C["accent"], height=1)
        sep2.pack(fill="x")

        self.lbl_chain_info = tk.Label(
            frame_chains, text='Carga perfiles y pulsa "🔍 Detectar".',
            bg=C["sidebar"], fg=C["sub"],
            font=("Courier New", 7), anchor="w",
            justify="left", wraplength=230, padx=8, pady=6)
        self.lbl_chain_info.pack(fill="x")

        pw.add(frame_chains, minsize=200)

        return pw

    # ── Panel derecho: notebook de herramientas ──────────────────────────────
    def _build_right(self, parent):
        frame = tk.Frame(parent, bg=C["bg"])
        self.nb = ttk.Notebook(frame)
        self.nb.pack(fill="both", expand=True, padx=2, pady=2)

        self.tab_vis = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(self.tab_vis, text="  ▣  Visualizador  ")
        self._build_tab_visualizer(self.tab_vis)

        self.tab_rep = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(self.tab_rep, text="  ⇄  Reproyector  ")
        self._build_tab_reprojector(self.tab_rep)

        self.tab_chain = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(self.tab_chain, text="  ⛓  Cadenas  ")
        self._build_tab_chains(self.tab_chain)

        self.tab_raw = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(self.tab_raw, text="  ⬡  RAW → SGY  ")
        tk.Label(self.tab_raw,
                 text="\n\n\n◈  Próximamente: conversión de archivos RAW de TOPAS a SEG-Y\n"
                      "con ajuste de parámetros, sincronización GPS y metadatos.",
                 bg=C["bg"], fg=C["sub"],
                 font=("Courier New", 11)).pack(expand=True)

        return frame

    # ════════════════════════════════════════════════════════════════════════
    # PESTAÑA A — VISUALIZADOR
    # ════════════════════════════════════════════════════════════════════════
    def _build_tab_visualizer(self, parent):
        paned = tk.PanedWindow(parent, orient="horizontal",
                               bg=C["bg"], sashwidth=5)
        paned.pack(fill="both", expand=True)

        ctrl = self._build_vis_controls(paned)
        paned.add(ctrl, minsize=200, width=220)

        right = tk.Frame(paned, bg=C["bg"])
        paned.add(right, minsize=500)

        self.vis_nb = ttk.Notebook(right)
        self.vis_nb.pack(fill="both", expand=True)

        self.vtab_profile  = tk.Frame(self.vis_nb, bg=C["bg"])
        self.vtab_map      = tk.Frame(self.vis_nb, bg=C["bg"])
        self.vtab_spectrum = tk.Frame(self.vis_nb, bg=C["bg"])
        self.vtab_headers  = tk.Frame(self.vis_nb, bg=C["bg"])

        self.vis_nb.add(self.vtab_profile,  text="  Perfil  ")
        self.vis_nb.add(self.vtab_map,      text="  Mapa  ")
        self.vis_nb.add(self.vtab_spectrum, text="  Espectro  ")
        self.vis_nb.add(self.vtab_headers,  text="  Cabeceras  ")

        for tab in (self.vtab_profile, self.vtab_map,
                    self.vtab_spectrum, self.vtab_headers):
            self._placeholder(tab, "Selecciona un perfil en la lista de la izquierda")

    def _build_vis_controls(self, parent):
        outer = tk.Frame(parent, bg=C["panel"], width=230)
        outer.pack_propagate(False)

        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        cv = tk.Canvas(outer, bg=C["panel"], yscrollcommand=vsb.set,
                       highlightthickness=0, width=210)
        cv.pack(side="left", fill="both", expand=True)
        vsb.config(command=cv.yview)

        sb = tk.Frame(cv, bg=C["panel"])
        win_id = cv.create_window((0, 0), window=sb, anchor="nw")

        def _on_frame_configure(e): cv.configure(scrollregion=cv.bbox("all"))
        def _on_canvas_configure(e): cv.itemconfig(win_id, width=e.width)
        def _on_mousewheel(e):
            delta = getattr(e, 'delta', 0)
            if delta:
                cv.yview_scroll(int(-delta / 120), "units")

        sb.bind("<Configure>", _on_frame_configure)
        cv.bind("<Configure>", _on_canvas_configure)
        cv.bind("<MouseWheel>",  _on_mousewheel)
        cv.bind("<Button-4>",    lambda e: cv.yview_scroll(-1, "units"))
        cv.bind("<Button-5>",    lambda e: cv.yview_scroll( 1, "units"))
        def _ctrl_scroll_vis(e):
            try:
                rx, ry = outer.winfo_rootx(), outer.winfo_rooty()
                rw, rh = outer.winfo_width(), outer.winfo_height()
                if rx <= e.x_root < rx + rw and ry <= e.y_root < ry + rh:
                    cv.yview_scroll(int(-e.delta / 120), "units")
            except Exception:
                pass
        outer.bind_all("<MouseWheel>", _ctrl_scroll_vis, add="+")

        def sec(t):
            tk.Frame(sb, bg=C["accent"], height=1).pack(fill="x", pady=(8, 2))
            tk.Label(sb, text=t, bg=C["panel"], fg=C["bright"],
                     font=("Courier New", 7, "bold"), anchor="w").pack(
                     fill="x", padx=8, pady=(0, 4))
        def lbl(t):
            tk.Label(sb, text=t, bg=C["panel"], fg=C["sub"],
                     font=("Courier New", 7), anchor="w").pack(
                     fill="x", padx=8, pady=(4, 0))
        def scale(var, lo, hi, res=1, length=190):
            tk.Scale(sb, from_=lo, to=hi, orient="horizontal",
                     variable=var, bg=C["panel"], fg=C["text"],
                     highlightthickness=0, troughcolor=C["accent"],
                     activebackground=C["bright"], font=("Courier New", 7),
                     length=length, resolution=res).pack(padx=8)

        # ── Paleta ──
        sec("PALETA")
        self.cmap_var = tk.StringVar(value="Blanco / Negro")
        ttk.Combobox(sb, textvariable=self.cmap_var, values=list(CMAPS.keys()), state="readonly",
                     font=("Courier New", 7)).pack(fill="x", padx=8, pady=2)
        self.inv_cmap_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Invertir colores", variable=self.inv_cmap_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(0, 4))

        # ── Deconvolución Predictiva ──
        sec("DECONVOLUCIÓN PREDICTIVA")
        self.decon_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Activar Deconvolución", variable=self.decon_var, bg=C["panel"], fg=C["warn"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7, "bold")).pack(anchor="w", padx=8, pady=(2, 0))
        
        lbl("Longitud operador (ms):"); self.decon_op = tk.DoubleVar(value=10.0)
        scale(self.decon_op, 1.0, 50.0, res=1.0)
        
        lbl("Gap / Lag predicción (ms):"); self.decon_gap = tk.DoubleVar(value=2.0)
        scale(self.decon_gap, 0.1, 20.0, res=0.1)

        lbl("Ruido blanco pre-whitening (%):"); self.decon_wn = tk.DoubleVar(value=1.0)
        scale(self.decon_wn, 0.1, 10.0, res=0.1)

        # ── Filtro Pasabanda ──
        sec("FILTRO PASABANDA")
        self.filt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Activar filtro", variable=self.filt_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7)).pack(anchor="w", padx=8)
        lbl("F low (Hz):"); self.flo = tk.IntVar(value=2000)
        scale(self.flo, 500, 15000, res=100)
        lbl("F high (Hz):"); self.fhi = tk.IntVar(value=7000)
        scale(self.fhi, 500, 15000, res=100)

        # ── Ganancia y Filtros ──
        sec("CLIP  /  GANANCIA")
        lbl("Clip amplitud (%):"); self.clip_var = tk.IntVar(value=98)
        scale(self.clip_var, 80, 100)

        # --- TVG (Atenuación α) ---
        self.tvg_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="TVG Adaptativo (Compensar α)", variable=self.tvg_var, bg=C["panel"], fg=C["highlight"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7, "bold")).pack(anchor="w", padx=8, pady=(4,0))
        lbl("Coef. atenuación α:"); self.tvg_alpha = tk.DoubleVar(value=15.0)
        scale(self.tvg_alpha, 0.0, 150.0, res=1)

        self.agc_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Aplicar AGC", variable=self.agc_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(4,0))
        lbl("Ventana AGC (ms):"); self.agc_win = tk.IntVar(value=20)
        scale(self.agc_win, 5, 100, res=5)

        self.align_delays_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sb, text="Compensar delays (alinear grupos)", variable=self.align_delays_var,
                       bg=C["panel"], fg=C["ok"], selectcolor=C["accent"], activebackground=C["panel"],
                       font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(4, 0))

        # ── Filtros preestablecidos ──
        sec("FILTROS  PREESTABLECIDOS")
        self.preset_var = tk.StringVar(value=list(FILTER_PRESETS.keys())[0])
        ttk.Combobox(sb, textvariable=self.preset_var, values=list(FILTER_PRESETS.keys()),
                     state="readonly", font=("Courier New", 7)).pack(fill="x", padx=8, pady=(2, 4))
        self.lbl_preset_desc = tk.Label(sb, text="", bg=C["panel"], fg=C["sub"], font=("Courier New", 6),
                                        anchor="w", justify="left", wraplength=200, padx=8)
        self.lbl_preset_desc.pack(fill="x")
        def _update_preset_desc(*_):
            key = FILTER_PRESETS.get(self.preset_var.get(), "none")
            self.lbl_preset_desc.config(text=FILTER_DESCRIPTIONS.get(key, ""))
        self.preset_var.trace_add("write", _update_preset_desc)
        _update_preset_desc()

        # ── Marcas FIX de tiempo ──
        sec("MARCAS  FIX")
        self.fix_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Mostrar marcas FIX", variable=self.fix_var, bg=C["panel"], fg=C["highlight"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7, "bold")).pack(anchor="w", padx=8)
        fix_row = tk.Frame(sb, bg=C["panel"])
        fix_row.pack(fill="x", padx=8, pady=(2, 4))
        tk.Label(fix_row, text="Intervalo (min):", bg=C["panel"], fg=C["sub"], font=("Courier New", 7)).pack(side="left")
        self.fix_interval_var = tk.IntVar(value=15)
        tk.Spinbox(fix_row, textvariable=self.fix_interval_var, from_=1, to=120, increment=1, width=5,
                   bg=C["entry"], fg=C["text"], insertbackground=C["text"], relief="flat", font=("Courier New", 7),
                   buttonbackground=C["accent"]).pack(side="left", padx=4)

        tk.Frame(sb, bg=C["accent"], height=1).pack(fill="x", pady=10)
        self._btn(sb, "⟳  RENDERIZAR", self._render_all, C["bright"], fill="x", padx=8, pady=4)
        self._btn(sb, "📊  Calcular Espectro", self._render_spectrum_vis, C["accent"], fill="x", padx=8, pady=(0, 8))
        
        # ── Control de DPI para exportación ──
        sec("EXPORTACIÓN")
        lbl("Resolución (DPI):")
        self.dpi_var = tk.StringVar(value="300")
        ttk.Combobox(sb, textvariable=self.dpi_var, values=["150", "300", "600", "1200"], font=("Courier New", 7)).pack(fill="x", padx=8, pady=(2, 8))

        self._btn(sb, "💾  Exportar imagen", self._export_vis, C["highlight"], fill="x", padx=8, pady=(0, 4))
        self._btn(sb, "🗺  Exportar FIX → SHP / GeoJSON / CSV", self._export_fix_shp_vis, C["warn"], fill="x", padx=8, pady=(0, 12))

        return outer

    # ════════════════════════════════════════════════════════════════════════
    # PESTAÑA B — REPROYECTOR
    # ════════════════════════════════════════════════════════════════════════
    def _build_tab_reprojector(self, parent):
        top = tk.Frame(parent, bg=C["bg"])
        top.pack(fill="x", padx=12, pady=10)

        bot = tk.Frame(parent, bg=C["bg"])
        bot.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        def sec(parent, text):
            tk.Label(parent, text=text, bg=C["bg"], fg=C["bright"],
                     font=("Courier New", 8, "bold"), anchor="w").pack(
                     fill="x", pady=(0, 4))
            tk.Frame(parent, bg=C["accent"], height=1).pack(fill="x", pady=(0, 8))

        sec(top, "▸  PERFILES A REPROYECTAR")

        sel_frame = tk.Frame(top, bg=C["panel"], pady=6, padx=8)
        sel_frame.pack(fill="x")

        # Modificación para soportar unión de cadenas
        self.rep_mode_var = tk.IntVar(value=0) # 0: Todos, 1: Activo, 2: Cadenas
        tk.Radiobutton(sel_frame, text="Todos los perfiles cargados (individualmente)",
                       variable=self.rep_mode_var, value=0,
                       command=self._on_rep_sel_change,
                       bg=C["panel"], fg=C["text"], selectcolor=C["accent"],
                       activebackground=C["panel"],
                       font=("Courier New", 8)).pack(anchor="w")
        tk.Radiobutton(sel_frame, text="Solo el perfil activo (seleccionado en la lista)",
                       variable=self.rep_mode_var, value=1,
                       command=self._on_rep_sel_change,
                       bg=C["panel"], fg=C["text"], selectcolor=C["accent"],
                       activebackground=C["panel"],
                       font=("Courier New", 8)).pack(anchor="w")
        tk.Radiobutton(sel_frame, text="Cadenas detectadas (unir perfiles contiguos y reproyectar el resultado)",
                       variable=self.rep_mode_var, value=2,
                       command=self._on_rep_sel_change,
                       bg=C["panel"], fg=C["ok"], selectcolor=C["accent"],
                       activebackground=C["panel"],
                       font=("Courier New", 8, "bold")).pack(anchor="w")

        self.lbl_rep_sel = tk.Label(sel_frame, text="",
                                    bg=C["panel"], fg=C["sub"],
                                    font=("Courier New", 7), anchor="w")
        self.lbl_rep_sel.pack(anchor="w", pady=(4, 0))

        crs_frame = tk.Frame(top, bg=C["bg"])
        crs_frame.pack(fill="x", pady=(10, 0))

        left_col = tk.Frame(crs_frame, bg=C["bg"])
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right_col = tk.Frame(crs_frame, bg=C["bg"])
        right_col.pack(side="left", fill="both", expand=True, padx=(8, 0))

        def crs_block(parent, title, preset_var, entry_var, status_lbl_name, verify_cmd):
            sec(parent, title)
            pf = tk.Frame(parent, bg=C["panel"], pady=6, padx=8)
            pf.pack(fill="x")
            tk.Label(pf, text="Preset:", bg=C["panel"], fg=C["sub"],
                     font=("Courier New", 7)).pack(anchor="w")
            ttk.Combobox(pf, textvariable=preset_var,
                         values=list(PRESETS_CRS.keys()),
                         state="readonly", font=("Courier New", 7),
                         width=36).pack(fill="x", pady=2)
            tk.Label(pf, text="EPSG o WKT:", bg=C["panel"], fg=C["sub"],
                     font=("Courier New", 7)).pack(anchor="w", pady=(4, 0))
            ef = tk.Frame(pf, bg=C["panel"])
            ef.pack(fill="x")
            e = tk.Entry(ef, textvariable=entry_var, bg=C["entry"],
                         fg=C["text"], insertbackground=C["text"],
                         font=("Courier New", 8), relief="flat")
            e.pack(side="left", fill="x", expand=True)
            self._btn(ef, "✔", verify_cmd, C["highlight"], side="right", padx=(4, 0))
            sl = tk.Label(pf, text="", bg=C["panel"],
                          font=("Courier New", 7), anchor="w")
            sl.pack(anchor="w", pady=(2, 0))
            setattr(self, status_lbl_name, sl)

        self.src_preset = tk.StringVar(value="WGS84 Geográfico (EPSG:4326)")
        self.src_epsg   = tk.StringVar(value="4326")
        self.dst_preset = tk.StringVar(value="UTM WGS84 Zona 30N (EPSG:32630)")
        self.dst_epsg   = tk.StringVar(value="32630")

        crs_block(left_col,  "▸  CRS ORIGEN",  self.src_preset, self.src_epsg,
                  "lbl_src_ok", self._verify_src)
        crs_block(right_col, "▸  CRS DESTINO", self.dst_preset, self.dst_epsg,
                  "lbl_dst_ok", self._verify_dst)

        self.src_preset.trace_add("write", lambda *_: self._on_src_preset())
        self.dst_preset.trace_add("write", lambda *_: self._on_dst_preset())

        tk.Label(top, text="Nota: el archivo reproyectado se guarda en la misma carpeta con sufijo _REPROY (o _UNIDO_REPROY).",
                 bg=C["bg"], fg=C["sub"], font=("Courier New", 7)).pack(anchor="w", pady=(8, 0))

        uh = tk.Frame(top, bg=C["bg"])
        uh.pack(fill="x", pady=4)
        tk.Label(uh, text="Interpretación unidades si no detectadas:",
                 bg=C["bg"], fg=C["sub"], font=("Courier New", 7)).pack(side="left")
        self.unit_hint = tk.StringVar(value="2 – Arc-seconds (TOPAS)")
        ttk.Combobox(uh, textvariable=self.unit_hint, state="readonly",
                     font=("Courier New", 7), width=28,
                     values=["1 – Metros/pies", "2 – Arc-seconds (TOPAS)",
                             "3 – Grados decimales"]).pack(side="left", padx=6)

        self.rep_progress = ttk.Progressbar(top, mode="determinate", maximum=100, length=200)
        self.rep_progress.pack(side="right", padx=8, pady=8)
        self._btn(top, "▶  REPROYECTAR", self._run_reprojection,
                  C["ok"], side="right", pady=8)

        # ── Exportación de línea de navegación ─────────────────────────────
        sec(bot, "▸  EXPORTAR LÍNEA DE NAVEGACIÓN")

        nav_frame = tk.Frame(bot, bg=C["panel"], pady=6, padx=8)
        nav_frame.pack(fill="x", pady=(0, 6))

        # Fila 1: selector de elemento + CRS de salida
        nav_row1 = tk.Frame(nav_frame, bg=C["panel"])
        nav_row1.pack(fill="x", pady=(0, 4))

        tk.Label(nav_row1, text="Elemento:", bg=C["panel"], fg=C["sub"],
                 font=("Courier New", 7)).pack(side="left")
        self.nav_source_var = tk.StringVar(value="active")
        self._nav_source_menu = ttk.Combobox(
            nav_row1, textvariable=self.nav_source_var,
            state="readonly", font=("Courier New", 7), width=42)
        self._nav_source_menu.pack(side="left", padx=6)
        self._btn(nav_row1, "⟳", self._refresh_nav_sources,
                  C["sub"], side="left", padx=(0, 12))

        tk.Label(nav_row1, text="CRS salida:", bg=C["panel"], fg=C["sub"],
                 font=("Courier New", 7)).pack(side="left")
        self.nav_crs_var = tk.StringVar(value="(mismo que CRS origen)")
        self._nav_crs_cb = ttk.Combobox(
            nav_row1, textvariable=self.nav_crs_var,
            font=("Courier New", 7), width=32,
            values=["(mismo que CRS origen)", "(mismo que CRS destino)"] +
                   list(PRESETS_CRS.keys()))
        self._nav_crs_cb.pack(side="left", padx=6)

        # Fila 2: opciones + botón exportar
        nav_row2 = tk.Frame(nav_frame, bg=C["panel"])
        nav_row2.pack(fill="x")

        self.nav_fmt_var = tk.StringVar(value="shp")
        for val, lbl in (("shp", "Shapefile (.shp)"),
                         ("geojson", "GeoJSON (.geojson)"),
                         ("csv",     "CSV (.csv)")):
            tk.Radiobutton(nav_row2, text=lbl, variable=self.nav_fmt_var, value=val,
                           bg=C["panel"], fg=C["text"], selectcolor=C["accent"],
                           activebackground=C["panel"],
                           font=("Courier New", 7)).pack(side="left", padx=(0, 10))

        self.nav_attrs_var = tk.BooleanVar(value=True)
        tk.Checkbutton(nav_row2, text="Incluir atributos (dist_km, wd, timestamp)",
                       variable=self.nav_attrs_var,
                       bg=C["panel"], fg=C["sub"], selectcolor=C["accent"],
                       activebackground=C["panel"],
                       font=("Courier New", 7)).pack(side="left", padx=(10, 0))

        self._btn(nav_row2, "💾  Exportar navline",
                  self._export_navline, C["bright"],
                  side="right", padx=(8, 0))

        sec(bot, "▸  LOG")
        self.rep_log = tk.Text(bot, bg=C["entry"], fg=C["text"],
                               font=("Courier New", 8), relief="flat",
                               state="disabled", wrap="word",
                               highlightthickness=1,
                               highlightbackground=C["accent"])
        vsb = ttk.Scrollbar(bot, orient="vertical", command=self.rep_log.yview)
        self.rep_log.config(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.rep_log.pack(fill="both", expand=True)

        self._verify_src()
        self._verify_dst()
        self._update_rep_sel_label()

    # ════════════════════════════════════════════════════════════════════════
    # GESTIÓN DE PERFILES
    # ════════════════════════════════════════════════════════════════════════
    def _add_profiles(self):
        paths = filedialog.askopenfilenames(
            title="Abrir archivos SEG-Y",
            filetypes=[("SEG-Y", "*.sgy *.segy *.SGY *.SEGY"), ("Todos", "*.*")])
        if not paths:
            return
        new_paths = [p for p in paths if p not in self.profiles]
        if not new_paths:
            self._status("Todos los archivos seleccionados ya estaban cargados.")
            return
        self._progress_start(f"Leyendo {len(new_paths)} perfil(es) SEG-Y…")
        self.update_idletasks()

        def load_all():
            for i, p in enumerate(new_paths, 1):
                self.after(0, lambda n=Path(p).name, i=i: self._progress_update(
                    f"Leyendo perfil {i}/{len(new_paths)}  ·  {n}…"))
                self.after(0, lambda p_c=(i/len(new_paths)*100): self._progress_set(p_c))
                sd = SegyProfile(p)
                self.after(0, lambda s=sd: self._register_profile(s))
            self.after(0, lambda: self._progress_stop(
                f"✔ {len(self.profiles)} perfil(es) cargado(s)."))

        threading.Thread(target=load_all, daemon=True).start()

    def _register_profile(self, sd: SegyProfile):
        self.profiles[sd.path] = sd
        tag = "✘ " if sd.error else "✔ "
        self.prof_list.insert(tk.END, f"{tag}{sd.name}")
        fg = C["warn"] if sd.error else C["ok"]
        self.prof_list.itemconfig(tk.END, fg=fg)
        if len(self.profiles) == 1:
            self.prof_list.selection_set(0)
            self._activate_profile(sd)
        # Actualizar cadenas automáticamente si hay ≥2 perfiles válidos
        valid = [p for p in self.profiles.values() if not p.error]
        if len(valid) >= 2:
            gap = float(self.chain_gap_var.get())
            self.chains = ProfileChain.detect(valid, gap_km=gap)
            self._refresh_chain_list()
        self._update_rep_sel_label()
        self._refresh_nav_sources()

    def _remove_profile(self):
        sel = self.prof_list.curselection()
        if not sel:
            return
        idx = sel[0]
        path = list(self.profiles.keys())[idx]
        del self.profiles[path]
        self.prof_list.delete(idx)
        if self.active_profile and self.active_profile.path == path:
            self.active_profile = None
            self.lbl_prof_info.config(text="")
        # Recalcular cadenas
        valid = [p for p in self.profiles.values() if not p.error]
        if len(valid) >= 2:
            gap = float(self.chain_gap_var.get())
            self.chains = ProfileChain.detect(valid, gap_km=gap)
        else:
            self.chains = []
            self.active_chain = None
        self._refresh_chain_list()
        self._update_rep_sel_label()

    def _clear_profiles(self):
        if not self.profiles:
            return
        if not messagebox.askyesno("Limpiar lista",
                                   "¿Eliminar todos los perfiles cargados?"):
            return
        self.profiles.clear()
        self.prof_list.delete(0, tk.END)
        self.active_profile = None
        self.lbl_prof_info.config(text="")
        self.chains = []
        self.active_chain = None
        self._refresh_chain_list()
        self._placeholder_chain()
        self._update_rep_sel_label()

    def _on_profile_select(self, event):
        sel = self.prof_list.curselection()
        if not sel:
            return
        idx  = sel[0]
        path = list(self.profiles.keys())[idx]
        sd   = self.profiles[path]
        self._activate_profile(sd)
        self._update_rep_sel_label()

    def _activate_profile(self, sd: SegyProfile):
        self.active_profile = sd
        if sd.error:
            self.lbl_prof_info.config(
                text=f"⚠ Error: {sd.error}", fg=C["warn"])
            return
        ts0 = sd.timestamps[0]  if sd.timestamps else "—"
        ts1 = sd.timestamps[-1] if sd.timestamps else "—"
        info = (
            f"Trazas: {sd.n_traces}\n"
            f"Muestras: {sd.ns}  ·  dt: {sd.dt_us} µs\n"
            f"Duración: {sd.dur_ms:.0f} ms\n"
            f"Longitud: {sd.total_km:.2f} km\n"
            f"Prof. agua: {np.nanmean(sd.water_depth):.0f} m\n"
            f"CRS: {sd.detected_crs or '?'}\n"
            f"Inicio: {ts0}\n"
            f"Fin:    {ts1}"
        )
        self.lbl_prof_info.config(text=info, fg=C["sub"])

        if sd.detected_crs:
            self.src_epsg.set(sd.detected_crs.replace("EPSG:", ""))
            self._verify_src()

        self._status(f"Perfil activo: {sd.name}  ·  {sd.summary()}")


    # ════════════════════════════════════════════════════════════════════════
    # ESPECTRO — Cálculo independiente (no bloquea el renderizado de imagen)
    # ════════════════════════════════════════════════════════════════════════
    def _render_spectrum_vis(self):
        """Calcula y muestra el espectro del perfil activo sin re-renderizar la imagen."""
        sd = self.active_profile
        if sd is None or sd.error:
            messagebox.showwarning("Sin perfil", "Selecciona un perfil válido.")
            return

        self._spec_render_id = getattr(self, "_spec_render_id", 0) + 1
        current_id = self._spec_render_id

        # Capturar params en hilo principal (thread-safe)
        spec_params = dict(
            decon    = self.decon_var.get(),    decon_op  = self.decon_op.get(),
            decon_gap= self.decon_gap.get(),    decon_wn  = self.decon_wn.get(),
            filt     = self.filt_var.get(),     flo       = self.flo.get(),
            fhi      = self.fhi.get(),          preset    = self.preset_var.get(),
            tvg      = self.tvg_var.get(),      tvg_alpha = self.tvg_alpha.get(),
            agc      = self.agc_var.get(),      agc_win   = self.agc_win.get(),
            align    = self.align_delays_var.get(),
            clip     = self.clip_var.get(),     cmap      = self.cmap_var.get(),
            inv_cmap = self.inv_cmap_var.get(), fix       = self.fix_var.get(),
            fix_iv   = self.fix_interval_var.get(),
        )

        self._progress_start(f"Calculando espectro  ·  {sd.name}…")

        def worker():
            try:
                data = self._process_data(sd, spec_params)
                self._current_profile_data = data
                if getattr(self, "_spec_render_id", None) != current_id:
                    self.after(0, lambda: self._progress_stop()); return
                fig = self._create_spectrum_fig(sd, data)
                if getattr(self, "_spec_render_id", None) != current_id:
                    if fig: import matplotlib.pyplot as _plt; _plt.close(fig)
                    self.after(0, lambda: self._progress_stop()); return
                if fig:
                    self.after(0, lambda: self._embed(self.vtab_spectrum, fig))
                    self.after(0, lambda: self.vis_nb.select(self.vtab_spectrum))
                self.after(0, lambda: self._progress_stop(
                    f"✔ Espectro calculado  ·  {sd.name}"))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._progress_stop("✘ Error al calcular espectro"))
                self.after(0, lambda: messagebox.showerror("Error espectro", err))

        threading.Thread(target=worker, daemon=True).start()

    def _render_spectrum_chain(self):
        """Calcula y muestra el espectro de la cadena activa sin re-renderizar la imagen."""
        ch = self.active_chain
        if ch is None:
            messagebox.showwarning("Sin cadena", "Detecta cadenas primero y selecciona una.")
            return

        self._spec_chain_render_id = getattr(self, "_spec_chain_render_id", 0) + 1
        current_id = self._spec_chain_render_id

        # Capturar params en hilo principal (thread-safe)
        spec_chain_params = dict(
            decon     = self.chain_decon_var.get(),    decon_op  = self.chain_decon_op.get(),
            decon_gap = self.chain_decon_gap.get(),    decon_wn  = self.chain_decon_wn.get(),
            filt      = self.chain_filt_var.get(),     flo       = self.chain_flo.get(),
            fhi       = self.chain_fhi.get(),          preset    = self.chain_preset_var.get(),
            tvg       = self.chain_tvg_var.get(),      tvg_alpha = self.chain_tvg_alpha.get(),
            agc       = self.chain_agc_var.get(),      agc_win   = self.chain_agc_win.get(),
            align     = self.chain_align_delays_var.get(),
            clip      = self.chain_clip_var.get(),     cmap      = self.chain_cmap_var.get(),
            inv_cmap  = self.chain_inv_var.get(),      fix       = self.chain_fix_var.get(),
            fix_iv    = self.chain_fix_interval_var.get(),
        )

        self._progress_start(f"Calculando espectro cadena  ·  {ch.label}…")

        def worker():
            try:
                data = self._process_chain_data(ch, spec_chain_params)
                self._current_chain_data = data
                if getattr(self, "_spec_chain_render_id", None) != current_id:
                    self.after(0, lambda: self._progress_stop()); return
                fig = self._create_chain_spectrum_fig(ch, data)
                if getattr(self, "_spec_chain_render_id", None) != current_id:
                    if fig: import matplotlib.pyplot as _plt; _plt.close(fig)
                    self.after(0, lambda: self._progress_stop()); return
                if fig:
                    self.after(0, lambda: self._embed(self.ctab_spectrum, fig))
                    self.after(0, lambda: self.chain_nb.select(self.ctab_spectrum))
                self.after(0, lambda: self._progress_stop(
                    f"✔ Espectro calculado  ·  {ch.label}"))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._progress_stop("✘ Error al calcular espectro de cadena"))
                self.after(0, lambda: messagebox.showerror("Error espectro cadena", err))

        threading.Thread(target=worker, daemon=True).start()

    # ════════════════════════════════════════════════════════════════════════
    # RENDERIZADO — Visualizador
    # ════════════════════════════════════════════════════════════════════════
    def _render_all(self):
        sd = self.active_profile
        if sd is None or sd.error:
            messagebox.showwarning("Sin perfil", "Selecciona un perfil válido.")
            return

        self._render_id = getattr(self, "_render_id", 0) + 1
        current_id = self._render_id

        # Capturar todos los valores de tk.Var en el hilo principal antes de
        # lanzar el worker — tk.Variable.get() no es thread-safe en Windows
        render_params = dict(
            decon       = self.decon_var.get(),
            decon_op    = self.decon_op.get(),
            decon_gap   = self.decon_gap.get(),
            decon_wn    = self.decon_wn.get(),
            filt        = self.filt_var.get(),
            flo         = self.flo.get(),
            fhi         = self.fhi.get(),
            preset      = self.preset_var.get(),
            tvg         = self.tvg_var.get(),
            tvg_alpha   = self.tvg_alpha.get(),
            agc         = self.agc_var.get(),
            agc_win     = self.agc_win.get(),
            align       = self.align_delays_var.get(),
            clip        = self.clip_var.get(),
            cmap        = self.cmap_var.get(),
            inv_cmap    = self.inv_cmap_var.get(),
            fix         = self.fix_var.get(),
            fix_iv      = self.fix_interval_var.get(),
        )

        self._progress_start(f"Renderizando {sd.name}…")
        self.update_idletasks()

        def worker():
            # Heavy processing and Figure creation — runs off the main thread
            self.after(0, lambda: self._progress_update(f"Procesando datos  ·  {sd.name}…"))
            data = self._process_data(sd, render_params)
            self._current_profile_data = data
            if self._render_id != current_id:
                self.after(0, lambda: self._progress_stop()); return

            self.after(0, lambda: self._progress_update(f"Dibujando perfil sísmico  ·  {sd.name}…"))
            fig_prof = self._create_profile_fig(sd, data, render_params)
            if self._render_id != current_id:
                plt.close(fig_prof)
                self.after(0, lambda: self._progress_stop()); return

            self.after(0, lambda: self._progress_update(f"Dibujando mapa  ·  {sd.name}…"))
            fig_map = self._create_map_fig(sd)
            if self._render_id != current_id:
                plt.close(fig_prof); plt.close(fig_map)
                self.after(0, lambda: self._progress_stop()); return

            # Schedule UI updates back on the main thread
            self.after(0, lambda: self._display_profile_figure(fig_prof, sd))
            self.after(0, lambda: self._embed(self.vtab_map, fig_map))
            self.after(0, lambda: self._draw_headers(sd, current_id))
            self.after(0, lambda: self._progress_stop(f"✔ {sd.name}  ·  perfil renderizado"))

        threading.Thread(target=worker, daemon=True).start()

    def _process_data(self, sd: SegyProfile, params: dict = None):
        """Aplica el pipeline de procesado a los datos del perfil.
        params: dict de valores ya leídos desde el hilo principal (thread-safe).
                Si es None, se leen directamente de los tk.Var (solo desde el hilo principal).
        """
        p = params  # alias corto
        def _get(key, tkvar):
            return p[key] if p is not None else tkvar.get()

        data = sd.data.copy()

        # 1. Deconvolución Predictiva
        if _get("decon", self.decon_var):
            data = apply_predictive_decon(data, sd.dt_us,
                                          _get("decon_op",  self.decon_op),
                                          _get("decon_gap", self.decon_gap),
                                          _get("decon_wn",  self.decon_wn))

        # 2. Filtro Pasabanda (paralelo multi-núcleo)
        if _get("filt", self.filt_var):
            fs  = 1e6 / sd.dt_us
            flo = max(10, _get("flo", self.flo))
            fhi = min(fs/2 - 1, _get("fhi", self.fhi))
            if flo < fhi:
                sos  = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")
                def _sosfilt_block(block, _sos=sos):
                    return sp_signal.sosfilt(_sos, block, axis=0).astype(np.float32)
                data = _parallel_apply(_sosfilt_block, data)

        # 3. Atributos / Filtros Preestablecidos
        preset_key = FILTER_PRESETS.get(_get("preset", self.preset_var), "none")
        if preset_key != "none":
            data = apply_filter_preset(data, preset_key, sd.dt_us)

        # 4. Compensación TVG exponencial por atenuación
        if _get("tvg", self.tvg_var):
            alpha = _get("tvg_alpha", self.tvg_alpha)
            t_sec = np.arange(data.shape[0], dtype=np.float32) * (sd.dt_us / 1e6)
            gain_curve = np.clip(np.exp(alpha * t_sec), 0.0, 1e9)
            data *= gain_curve[:, np.newaxis]

        # 5. Control Automático de Ganancia (AGC)
        if _get("agc", self.agc_var):
            win_s = max(3, int(_get("agc_win", self.agc_win) / (sd.dt_us / 1000.0)))
            if win_s % 2 == 0:
                win_s += 1
            if _GPU:
                import cupy as cp
                try:
                    from cupyx.scipy.ndimage import uniform_filter1d as cu_uf
                    g = cp.asarray(np.abs(data))
                    rms = cu_uf(g, size=win_s, axis=0)
                    rms = cp.maximum(rms, 1e-9)
                    data = cp.asnumpy(cp.asarray(data) / rms).astype(np.float32)
                except Exception:
                    # fallback: vectorized numpy
                    k = np.ones(win_s, dtype=np.float32) / win_s
                    env = np.abs(data)
                    from scipy.ndimage import uniform_filter1d
                    rms = uniform_filter1d(env, size=win_s, axis=0)
                    data = (data / np.maximum(rms, 1e-9)).astype(np.float32)
            else:
                from scipy.ndimage import uniform_filter1d
                env = np.abs(data)
                rms = uniform_filter1d(env, size=win_s, axis=0)
                data = (data / np.maximum(rms, 1e-9)).astype(np.float32)

        # 6. Compensación de delays — vectorizado
        if _get("align", self.align_delays_var):
            dt_ms = sd.dt_us / 1000.0
            offsets = np.round((sd.delays - sd.min_delay) / dt_ms).astype(int)
            extra_samples = int(offsets.max())
            new_ns = sd.ns + extra_samples

            aligned_data = np.full((new_ns, sd.n_traces), np.nan, dtype=np.float32)
            row_idx = (np.arange(sd.ns)[:, None] + offsets[None, :])   # (ns, n_traces)
            col_idx = np.arange(sd.n_traces)[None, :]                   # broadcast
            aligned_data[row_idx, col_idx] = data

            data = aligned_data
            
        return data

    def _time_window(self, sd: SegyProfile, data_ns: int, params: dict = None):
        i0 = 0
        i1 = data_ns
        align = params["align"] if params is not None else self.align_delays_var.get()
        if align:
            t0 = sd.min_delay
            t1 = sd.min_delay + data_ns * sd.dt_us / 1000.0
        else:
            t0 = sd.delay_ms
            t1 = sd.delay_ms + data_ns * sd.dt_us / 1000.0
        return i0, i1, t0, t1

    def _embed(self, parent, fig):
        # Destruir todos los widgets anteriores y cerrar las figuras viejas
        # para liberar memoria matplotlib (evitar leak de figuras huérfanas).
        for w in parent.winfo_children():
            old_fig = getattr(w, "figure_ref", None)
            if old_fig is not None:
                try:
                    plt.close(old_fig)
                except Exception:
                    pass
            w.destroy()

        # Toolbar PRIMERO (side="bottom") para que el canvas llene el resto.
        # pack_toolbar=False nos permite controlarlo manualmente.
        canvas = FigureCanvasTkAgg(fig, master=parent)
        tb = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        tb.config(bg=C["panel"])
        tb.update()
        tb.pack(side="bottom", fill="x")

        widget = canvas.get_tk_widget()
        widget.figure_ref = fig       # referencia para exportación y cierre
        widget.pack(fill="both", expand=True)

        # Forzar layout antes de draw() para que el canvas conozca su
        # tamaño real y evitar la imagen pequeña sobre fondo oscuro grande.
        parent.update_idletasks()
        canvas.draw()

    def _display_profile_figure(self, fig: Figure, sd):
        if hasattr(self, "_profile_fig") and self._profile_fig is not None:
            try: plt.close(self._profile_fig)
            except Exception: pass
        self._profile_fig = fig

        for w in list(self.vtab_profile.winfo_children()):
            old_f = getattr(w, "figure_ref", None)
            if old_f is not None:
                try: plt.close(old_f)
                except Exception: pass
            try: w.destroy()
            except Exception: pass

        avail_w_px = max(400, self.vtab_profile.winfo_width())
        avail_h_px = max(300, self.vtab_profile.winfo_height() - 60)
        
        fig.set_size_inches(avail_w_px / 100.0, avail_h_px / 100.0)
        fig_px_w, fig_px_h = avail_w_px, avail_h_px

        tb_frame = tk.Frame(self.vtab_profile, bg=C["panel"])
        tb_frame.pack(side="bottom", fill="x")

        hbar = ttk.Scrollbar(self.vtab_profile, orient="horizontal")
        hbar.pack(side="bottom", fill="x")

        scroll_cv = tk.Canvas(self.vtab_profile, bg=C["panel"], xscrollcommand=hbar.set, highlightthickness=0)
        vbar = ttk.Scrollbar(self.vtab_profile, orient="vertical", command=scroll_cv.yview)
        vbar.pack(side="right", fill="y")
        scroll_cv.config(yscrollcommand=vbar.set)
        scroll_cv.pack(fill="both", expand=True)
        hbar.config(command=scroll_cv.xview)

        inner = tk.Frame(scroll_cv, bg=C["panel"], width=fig_px_w, height=fig_px_h)
        inner.pack_propagate(False)
        scroll_cv.create_window(0, 0, anchor="nw", window=inner)
        scroll_cv.config(scrollregion=(0, 0, fig_px_w, fig_px_h))

        mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
        mpl_widget = mpl_canvas.get_tk_widget()
        mpl_widget.config(width=fig_px_w, height=fig_px_h)
        mpl_widget.figure_ref = fig
        mpl_widget.pack(fill="both", expand=True)

        tb = CustomToolbar(mpl_canvas, tb_frame, callback=self._on_profile_zoom)
        tb.config(bg=C["panel"])
        tb.update()
        tb.pack(fill="x")

        def _hscroll(e): scroll_cv.xview_scroll(int(-e.delta / 120), "units")
        scroll_cv.bind("<Shift-MouseWheel>", _hscroll)
        scroll_cv.bind("<MouseWheel>", _hscroll)

        mpl_canvas.draw()
        self._profile_canvas_obj = mpl_canvas
        self._profile_scroll_cv = scroll_cv
        self._profile_inner_frame = inner
        
    def _on_profile_zoom(self):
        data = getattr(self, "_current_profile_data", None)
        sd = self.active_profile
        if data is None or sd is None or not hasattr(self, "_profile_canvas_obj"): return
        if not hasattr(self, "_profile_fig") or self._profile_fig is None: return
        if not self._profile_fig.axes or not self._profile_fig.axes[0].images: return
        
        fig = self._profile_fig
        ax = fig.axes[0]
        x0, x1 = ax.get_xlim()
        y1, y0 = ax.get_ylim()
        
        dist = sd.dist_km
        tr0 = max(0, min(np.searchsorted(dist, min(x0, x1)), len(dist) - 1))
        tr1 = max(0, min(np.searchsorted(dist, max(x0, x1)), len(dist)))
        if tr1 <= tr0: tr1 = tr0 + 1
        
        t_max, t_min = max(y0, y1), min(y0, y1)
        dt_ms = sd.dt_us / 1000.0
        # Use the same time reference the render used (min_delay if aligned, else delay_ms)
        t_base = getattr(self, "_last_profile_t0", sd.delay_ms)
        s0 = max(0, min(int((t_min - t_base) / dt_ms), data.shape[0] - 1))
        s1 = max(0, min(int((t_max - t_base) / dt_ms), data.shape[0]))
        if s1 <= s0: s1 = s0 + 1

        new_d = data[s0:s1, tr0:tr1]
        
        is_full = (tr1 - tr0) >= len(dist) * 0.95 and (s1 - s0) >= data.shape[0] * 0.95
        avail_w_px = max(400, self.vtab_profile.winfo_width())
        avail_h_px = max(300, self.vtab_profile.winfo_height() - 60)
        
        if is_full:
            new_fig_px_w, new_fig_px_h = avail_w_px, avail_h_px
        else:
            new_fig_px_w = avail_w_px
            orig_px_per_trace = avail_w_px / len(dist)
            orig_px_per_sample = avail_h_px / data.shape[0]
            aspect_ratio = orig_px_per_sample / orig_px_per_trace
            new_px_per_trace = new_fig_px_w / (tr1 - tr0)
            new_px_per_sample = new_px_per_trace * aspect_ratio
            new_fig_px_h = new_px_per_sample * (s1 - s0)
            
        fig.set_size_inches(new_fig_px_w / 100.0, new_fig_px_h / 100.0)
        ax.images[0].set_data(new_d)
        ax.images[0].set_extent([dist[tr0], dist[tr1-1], t_max, t_min])
        
        self._profile_inner_frame.config(width=new_fig_px_w, height=int(new_fig_px_h))
        self._profile_canvas_obj.get_tk_widget().config(width=new_fig_px_w, height=int(new_fig_px_h))
        self._profile_scroll_cv.config(scrollregion=(0, 0, new_fig_px_w, int(new_fig_px_h)))
        self._profile_canvas_obj.draw_idle()

    # ── Perfil sísmico ────────────────────────────────────────────────────
    @staticmethod
    def _parse_timestamp(ts: str):
        from datetime import datetime, timedelta
        try:
            date_part, time_part = ts.split(" ")
            year = int(date_part.split("-")[0])
            doy  = int(date_part.split("DOY")[1])
            h, m, s = (int(x) for x in time_part.split(":"))
            return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=h, minutes=m, seconds=s)
        except Exception:
            return None

    @staticmethod
    def _compute_fix_positions(timestamps: list, dist_km: np.ndarray,
                                lons: np.ndarray, lats: np.ndarray,
                                interval_min: int) -> list:
        from datetime import datetime, timedelta, timezone

        # Parse all timestamps once into seconds-since-epoch array
        epoch = datetime(1970, 1, 1)
        t_sec = np.empty(len(timestamps), dtype=np.float64)
        for k, ts in enumerate(timestamps):
            dt = TopasSuite._parse_timestamp(ts)
            t_sec[k] = (dt - epoch).total_seconds() if dt else np.nan

        valid = ~np.isnan(t_sec)
        if not valid.any():
            return []

        t0   = t_sec[valid][0]
        t_end = t_sec[valid][-1]
        iv_sec = interval_min * 60.0

        # First fix tick at next round multiple
        first_rem = t0 % iv_sec
        first_fix = t0 + (iv_sec - first_rem) if first_rem else t0 + iv_sec

        fix_times = np.arange(first_fix, t_end + 1e-3, iv_sec)
        if fix_times.size == 0:
            return []

        # For each fix time, find nearest trace index via searchsorted
        t_valid = t_sec.copy()
        t_valid[~valid] = np.inf          # keep indexing aligned to original array

        t_sorted_idx = np.argsort(t_sec, kind="stable")   # argsort on full array
        t_sorted     = t_sec[t_sorted_idx]

        fixes = []
        half_iv = iv_sec / 2.0
        for num, ft in enumerate(fix_times, 1):
            pos = np.searchsorted(t_sorted, ft)
            # Check neighbours
            best_idx = None
            best_diff = np.inf
            for cand_pos in (pos - 1, pos):
                if 0 <= cand_pos < len(t_sorted_idx):
                    orig_idx = t_sorted_idx[cand_pos]
                    if np.isnan(t_sec[orig_idx]):
                        continue
                    diff = abs(t_sec[orig_idx] - ft)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx  = orig_idx
            if best_idx is not None and best_diff < half_iv:
                hhmm = datetime.fromtimestamp(ft, tz=timezone.utc).strftime("%H:%M")
                fixes.append((num, float(dist_km[best_idx]), hhmm,
                              float(lons[best_idx]), float(lats[best_idx])))
        return fixes

    @staticmethod
    def _draw_fix_marks(ax, fixes: list, color: str = "#FFD700"):
        if not fixes: return
        for num, dist, label, _lon, _lat in fixes:
            ax.axvline(dist, color=color, lw=0.8, ls="--", alpha=0.65, zorder=3)
            ax.text(dist, 0.98, f" {num} · {label}", color=color, fontsize=5.5, fontfamily="Courier New",
                    rotation=90, va="top", ha="center", zorder=5, clip_on=True, transform=ax.get_xaxis_transform(),
                    bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none", alpha=0.20))

    def _create_profile_fig(self, sd, data, params: dict = None):
        i0, i1, t0, t1 = self._time_window(sd, data.shape[0], params)
        d = data[i0:i1, :]
        valid_vals = np.abs(d[~np.isnan(d)])
        clip_pct = params["clip"] if params is not None else self.clip_var.get()
        vmax = float(np.percentile(valid_vals, clip_pct)) if valid_vals.size > 0 else 1.0
        vmax = vmax or 1.0
        vmin = 0.0

        cmap_base = CMAPS[params["cmap"] if params is not None else self.cmap_var.get()]
        cmap_name = cmap_base + "_r" if (params["inv_cmap"] if params is not None else self.inv_cmap_var.get()) else cmap_base

        fig = Figure(figsize=(10, 6), facecolor=C["panel"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(C["panel"])

        ax.imshow(d, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax,
                  extent=[sd.dist_km[0], sd.dist_km[-1], t1, t0], interpolation="bilinear")

        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(vmin, vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.015)
        cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=7)
        cb.set_label("Amplitud", color=C["sub"], fontsize=8)

        ax.set_xlabel("Distancia (km)", color=C["text"], fontsize=9)
        ax.set_ylabel("Tiempo (ms)", color=C["text"], fontsize=9)
        _preset_name = params["preset"] if params is not None else self.preset_var.get()
        preset_key = FILTER_PRESETS.get(_preset_name, "none")
        preset_lbl = (f"  ·  {_preset_name}" if preset_key != "none" else "")
        _align = params["align"] if params is not None else self.align_delays_var.get()
        delay_lbl  = ("  ·  delays compensados" if _align else "  ·  delays SIN compensar")
        ax.set_title(f"{sd.name}  ·  {sd.n_traces} trazas  ·  {sd.dt_us} µs{preset_lbl}{delay_lbl}",
                     color=C["text"], fontsize=10, pad=8)
        ax.tick_params(colors=C["text"], labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(C["accent"])

        show_fix = params["fix"] if params is not None else self.fix_var.get()
        fix_iv   = params["fix_iv"] if params is not None else self.fix_interval_var.get()
        if show_fix:
            fixes = self._compute_fix_positions(sd.timestamps, sd.dist_km, sd.lons, sd.lats, int(fix_iv))
            self._last_fixes = fixes
            self._last_fixes_sd = sd
            self._draw_fix_marks(ax, fixes, color=C["highlight"])
        else:
            self._last_fixes = []
            self._last_fixes_sd = None

        # Store time reference for _on_profile_zoom (must match the alignment used here)
        self._last_profile_t0 = t0
        fig.tight_layout(pad=1.2)
        return fig

    # ── Exportar FIX a Shapefile ──────────────────────────────────────────
    # ── Escritura SHP pura (stdlib struct, sin pyshp) ─────────────────────
    @staticmethod
    def _write_shp_pure(path: str, points: list):
        """
        Escribe un Shapefile de puntos (POINT 2D) usando solo la stdlib.
        points: lista de (lon, lat, atributos…)  — lon/lat float WGS84.
        Genera .shp / .shx / .dbf / .prj sin ninguna dependencia externa.
        """
        import struct, json as _json

        records = points   # list of (fix_num, fix_hora, lon, lat)

        # ── .shp / .shx ──────────────────────────────────────────────────
        # File header: 100 bytes (big-endian fields + little-endian fields)
        def _shp_record(lon, lat):
            # Record header (big-endian): record number + content length (in 16-bit words)
            # Content: shape type (4 bytes LE) + X + Y (8 bytes LE each) = 20 bytes → 10 words
            content = struct.pack("<i dd", 1, lon, lat)   # shape type 1 = POINT
            return content

        shp_records = []
        offsets = []
        cur_offset = 50   # file header = 100 bytes = 50 16-bit words

        for rec in records:
            lon, lat = rec[2], rec[3]
            content = _shp_record(lon, lat)
            offsets.append(cur_offset)
            shp_records.append(content)
            # Record header = 2 × int32 (big-endian) = 8 bytes = 4 sixteen-bit words
            cur_offset += 4 + len(content) // 2   # 4 words (rec hdr) + content words

        file_length = cur_offset   # in 16-bit words

        # Bounding box
        lons = [r[2] for r in records]; lats = [r[3] for r in records]
        xmin, xmax = min(lons), max(lons)
        ymin, ymax = min(lats), max(lats)

        def _file_header(file_len):
            return struct.pack(
                ">iiiiiii",
                9994, 0, 0, 0, 0, 0, file_len
            ) + struct.pack(
                "<ii dddddddd",
                1000,         # version
                1,            # shape type POINT
                xmin, ymin, xmax, ymax,
                0.0, 0.0, 0.0, 0.0   # Xmin,Ymin,Xmax,Ymax,Zmin,Zmax,Mmin,Mmax
            )

        shp_path = path if path.endswith(".shp") else path + ".shp"
        shx_path = shp_path.replace(".shp", ".shx")
        dbf_path = shp_path.replace(".shp", ".dbf")
        prj_path = shp_path.replace(".shp", ".prj")

        with open(shp_path, "wb") as shp_f, open(shx_path, "wb") as shx_f:
            hdr = _file_header(file_length)
            shx_file_len = 50 + 4 * len(records)   # header + 8 bytes/record
            shp_f.write(hdr)
            shx_f.write(_file_header(shx_file_len))

            for idx, (content, offset) in enumerate(zip(shp_records, offsets)):
                rec_num = idx + 1
                content_len = len(content) // 2   # in 16-bit words
                rec_hdr = struct.pack(">ii", rec_num, content_len)
                shp_f.write(rec_hdr + content)
                shx_f.write(struct.pack(">ii", offset, content_len))

        # ── .dbf (dBASE III) ─────────────────────────────────────────────
        # Fields: fix_num (N,6), fix_hora (C,8), lon (N,18,8), lat (N,18,8)
        fields = [
            (b"fix_num\x00\x00\x00\x00",  b"N", 6,  0),
            (b"fix_hora\x00\x00\x00",     b"C", 8,  0),
            (b"lon\x00\x00\x00\x00\x00\x00\x00\x00", b"N", 18, 8),
            (b"lat\x00\x00\x00\x00\x00\x00\x00\x00", b"N", 18, 8),
        ]
        record_len = 1 + sum(f[2] for f in fields)   # 1 = deletion flag
        header_len = 32 + 32 * len(fields) + 1        # + terminator

        with open(dbf_path, "wb") as dbf:
            # DBF header corrigida a formato estándar de 32 bytes para dBASE III
            dbf.write(struct.pack("<B B B B I H H 20s",
                3,              # version dBASE III
                125, 1, 1,      # last update YY MM DD (placeholder)
                len(records),   # num records (I = 4 bytes)
                header_len,     # (H = 2 bytes)
                record_len,     # (H = 2 bytes)
                b"\x00" * 20))  # reserved 20 bytes
                
            # Field descriptors
            for fname, ftype, flen, fdec in fields:
                dbf.write(fname[:11].ljust(11, b"\x00"))
                dbf.write(ftype)
                dbf.write(b"\x00" * 4)         # reserved
                dbf.write(struct.pack("B", flen))
                dbf.write(struct.pack("B", fdec))
                dbf.write(b"\x00" * 14)
            dbf.write(b"\r")   # header terminator

            for rec in records:
                fix_num, fix_hora, lon, lat = rec[0], rec[1], rec[2], rec[3]
                dbf.write(b" ")   # not deleted
                dbf.write(str(fix_num).rjust(6).encode("ascii"))
                dbf.write(fix_hora.ljust(8).encode("ascii"))
                dbf.write(f"{lon:.8f}".rjust(18).encode("ascii"))
                dbf.write(f"{lat:.8f}".rjust(18).encode("ascii"))
            dbf.write(b"\x1a")   # EOF marker

        # ── .prj (WGS84) ─────────────────────────────────────────────────
        with open(prj_path, "w") as prj:
            prj.write('GEOGCS["GCS_WGS_1984",'
                      'DATUM["D_WGS_1984",'
                      'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                      'PRIMEM["Greenwich",0.0],'
                      'UNIT["Degree",0.0174532925199433]]')

    # ── Escritura GeoJSON (stdlib json, zero deps) ────────────────────────
    @staticmethod
    def _write_geojson(path: str, points: list):
        """Escribe un GeoJSON FeatureCollection de puntos WGS84."""
        import json as _json
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"fix_num": num, "fix_hora": hora,
                               "lon": round(lon, 8), "lat": round(lat, 8)},
            }
            for num, _dist, hora, lon, lat in points
        ]
        fc = {"type": "FeatureCollection",
              "crs": {"type": "name",
                      "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
              "features": features}
        geojson_path = path if path.endswith(".geojson") else path
        with open(geojson_path, "w", encoding="utf-8") as f:
            _json.dump(fc, f, ensure_ascii=False, indent=2)

    # ── Escritura CSV (stdlib csv, zero deps) ────────────────────────────
    @staticmethod
    def _write_csv(path: str, points: list):
        """Escribe un CSV con columnas fix_num, fix_hora, lon, lat."""
        import csv as _csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["fix_num", "fix_hora", "lon", "lat"])
            for num, _dist, hora, lon, lat in points:
                w.writerow([num, hora, round(lon, 8), round(lat, 8)])

    # ════════════════════════════════════════════════════════════════════════
    # EXPORTACIÓN — LÍNEA DE NAVEGACIÓN
    # ════════════════════════════════════════════════════════════════════════

    def _export_navline(self):
        """Exporta la línea de navegación del elemento seleccionado como
        POLYLINE shapefile, GeoJSON LineString o CSV de trazas."""

        sel = self.nav_source_var.get()
        if not sel or sel.startswith("(sin"):
            messagebox.showwarning("Sin selección",
                                   "Selecciona un perfil o cadena en el menú.")
            return

        # ── Resolver el objeto seleccionado ──────────────────────────────
        # Prefijos — deben coincidir exactamente con los usados en _refresh_nav_sources
        _PFX_PERFIL  = "Perfil: "
        _PFX_CADENA1 = "Perfil (cadena 1): "
        _PFX_CADENA  = "Cadena: "

        source = None
        if sel.startswith(_PFX_PERFIL):
            name = sel[len(_PFX_PERFIL):]
            for sd in self.profiles.values():
                if sd.name == name and not sd.error:
                    source = sd; break
        elif sel.startswith(_PFX_CADENA1):
            name = sel[len(_PFX_CADENA1):]
            for sd in self.profiles.values():
                if sd.name == name and not sd.error:
                    source = sd; break
        elif sel.startswith(_PFX_CADENA):
            label = sel[len(_PFX_CADENA):]
            for ch in self.chains:
                if ch.label == label:
                    source = ch; break

        if source is None:
            messagebox.showwarning("No encontrado",
                                   "No se pudo identificar el elemento seleccionado.\n"
                                   "Pulsa ⟳ para refrescar la lista.")
            return

        is_chain = isinstance(source, ProfileChain)
        stem = (f"{source.profiles[0].stem}_a_{source.profiles[-1].stem}_NAVLINE"
                if is_chain else f"{source.stem}_NAVLINE")

        # ── CRS de salida ─────────────────────────────────────────────────
        nav_crs_val = self.nav_crs_var.get()
        out_crs_str = None
        tf = None
        if nav_crs_val == "(mismo que CRS destino)":
            out_crs_str = self._resolve_crs(self.dst_epsg.get())
        elif nav_crs_val in PRESETS_CRS:
            out_crs_str = PRESETS_CRS[nav_crs_val]
        # else: None → sin reproyectar (coordenadas originales)

        if out_crs_str:
            try:
                from pyproj import CRS as _CRS, Transformer as _Tf
                src_crs_str = self._resolve_crs(self.src_epsg.get())
                src_crs = _CRS.from_user_input(src_crs_str)
                dst_crs = _CRS.from_user_input(out_crs_str)
                if src_crs != dst_crs:
                    tf = _Tf.from_crs(src_crs, dst_crs, always_xy=True)
            except Exception as e:
                messagebox.showerror("Error CRS",
                                     f"No se pudo preparar la reproyección:\n{e}")
                return

        # ── Diálogo de guardado ───────────────────────────────────────────
        fmt = self.nav_fmt_var.get()
        ext_map  = {"shp": ".shp", "geojson": ".geojson", "csv": ".csv"}
        ft_map   = {"shp":  [("Shapefile", "*.shp")],
                    "geojson": [("GeoJSON", "*.geojson")],
                    "csv":  [("CSV", "*.csv")]}
        out = filedialog.asksaveasfilename(
            initialfile=stem,
            filetypes=ft_map[fmt] + [("Todos", "*.*")],
            defaultextension=ext_map[fmt])
        if not out:
            return

        include_attrs = self.nav_attrs_var.get()

        # ── Worker ────────────────────────────────────────────────────────
        self.after(0, lambda: self._progress_start(
            f"Exportando navline  ·  {Path(out).name}…"))

        def export_worker():
            try:
                lons = np.array(source.lons, dtype=float)
                lats = np.array(source.lats, dtype=float)
                dist = np.array(source.dist_km, dtype=float)
                wd   = np.array(source.water_depth, dtype=float)
                ts   = list(source.timestamps)

                if tf is not None:
                    lons, lats = tf.transform(lons, lats)

                n = len(lons)

                if fmt == "shp":
                    self._write_navline_shp(out, lons, lats, dist, wd, ts,
                                            include_attrs, out_crs_str)
                elif fmt == "geojson":
                    self._write_navline_geojson(out, lons, lats, dist, wd, ts,
                                                include_attrs, out_crs_str)
                else:
                    self._write_navline_csv(out, lons, lats, dist, wd, ts)

                self.after(0, lambda: self._progress_stop(
                    f"✔ Navline exportada  ·  {n} trazas  ·  {Path(out).name}"))
                self.after(0, lambda: messagebox.showinfo(
                    "Navline exportada",
                    f"Línea de navegación guardada en:\n{out}\n"
                    f"{n} trazas  ·  {source.total_km:.2f} km" +
                    (f"\nCRS: {out_crs_str}" if out_crs_str else "")))
                self.after(0, lambda: self._rlog(
                    f"✔ Navline: {Path(out).name}  ({n} trazas, "
                    f"{source.total_km:.2f} km"
                    + (f", CRS: {out_crs_str}" if out_crs_str else "") + ")"))
            except Exception as e:
                import traceback as _tb
                err = str(e); tb = _tb.format_exc()
                self.after(0, lambda: self._progress_stop("✘ Error al exportar navline."))
                self.after(0, lambda: messagebox.showerror("Error navline", err))
                self.after(0, lambda: self._rlog(f"✘ Error navline: {err}\n{tb}"))

        threading.Thread(target=export_worker, daemon=True).start()

    # ── Escritores de polilínea ──────────────────────────────────────────
    @staticmethod
    def _write_navline_shp(path: str, lons, lats, dist, wd, ts,
                           include_attrs: bool, crs_str):
        """Shapefile POLYLINE (tipo 3) + .dbf con atributos por vértice."""
        import struct

        n = len(lons)
        shp_path = path if path.lower().endswith(".shp") else path + ".shp"
        shx_path = shp_path[:-4] + ".shx"
        dbf_path = shp_path[:-4] + ".dbf"
        prj_path = shp_path[:-4] + ".prj"

        # ── Contenido del registro POLYLINE ─────────────────────────────
        # Shape type 3: 4(type) + 32(bbox) + 4(numparts) + 4(numpoints)
        #               + 4*numparts(parts array) + 16*numpoints(XY pairs)
        num_parts  = 1
        num_points = n
        content_len_bytes = 4 + 32 + 4 + 4 + 4 * num_parts + 16 * num_points
        content_len_words = content_len_bytes // 2   # in 16-bit words

        xmin, xmax = float(np.min(lons)), float(np.max(lons))
        ymin, ymax = float(np.min(lats)), float(np.max(lats))

        content = struct.pack("<i", 3)                         # shape type POLYLINE
        content += struct.pack("<dddd", xmin, ymin, xmax, ymax)  # bbox
        content += struct.pack("<ii", num_parts, num_points)
        content += struct.pack("<i", 0)                        # parts[0] = 0
        for x, y in zip(lons, lats):
            content += struct.pack("<dd", float(x), float(y))

        # ── .shp ────────────────────────────────────────────────────────
        file_len_words = 50 + 4 + content_len_words   # header + rec_hdr + content

        def _fhdr(flen):
            return (struct.pack(">iiiiiii", 9994, 0, 0, 0, 0, 0, flen) +
                    struct.pack("<ii dddddddd",
                                1000, 3,
                                xmin, ymin, xmax, ymax,
                                0.0, 0.0, 0.0, 0.0))

        with open(shp_path, "wb") as shp, open(shx_path, "wb") as shx:
            shp.write(_fhdr(file_len_words))
            shx.write(_fhdr(50 + 4))   # shx: header + 1 record × 8 bytes = 4 words
            rec_hdr = struct.pack(">ii", 1, content_len_words)
            shp.write(rec_hdr + content)
            shx.write(struct.pack(">ii", 50, content_len_words))  # offset, content_len

        # ── .dbf ────────────────────────────────────────────────────────
        # NOTE: for a POLYLINE shp the .dbf has 1 record (the whole line).
        # We store summary stats: total traces, start/end lon/lat, total km.
        # For per-vertex attributes users should use GeoJSON or CSV.
        sum_fields = [
            (b"n_traces\x00\x00\x00",  b"N", 7,  0),
            (b"total_km\x00\x00\x00",  b"N", 12, 4),
            (b"lon_start\x00\x00",     b"N", 18, 8),
            (b"lat_start\x00\x00",     b"N", 18, 8),
            (b"lon_end\x00\x00\x00\x00", b"N", 18, 8),
            (b"lat_end\x00\x00\x00\x00", b"N", 18, 8),
            (b"ts_start\x00\x00\x00",  b"C", 24, 0),
            (b"ts_end\x00\x00\x00\x00\x00",   b"C", 24, 0),
        ]
        if include_attrs:
            sum_fields += [
                (b"wd_mean\x00\x00\x00\x00",  b"N", 10, 2),
                (b"wd_min\x00\x00\x00\x00\x00",  b"N", 10, 2),
                (b"wd_max\x00\x00\x00\x00\x00",  b"N", 10, 2),
            ]

        record_len = 1 + sum(f[2] for f in sum_fields)
        header_len = 32 + 32 * len(sum_fields) + 1

        with open(dbf_path, "wb") as dbf:
            dbf.write(struct.pack("<B B B B I H H 20s",
                                  3, 125, 1, 1, 1,
                                  header_len, record_len, b"\x00" * 20))
            for fname, ftype, flen, fdec in sum_fields:
                dbf.write(fname[:11].ljust(11, b"\x00"))
                dbf.write(ftype)
                dbf.write(b"\x00" * 4)
                dbf.write(struct.pack("B", flen))
                dbf.write(struct.pack("B", fdec))
                dbf.write(b"\x00" * 14)
            dbf.write(b"\r")

            dbf.write(b" ")   # not deleted
            dbf.write(str(n).rjust(7).encode("ascii"))
            dbf.write(f"{float(dist[-1]):.4f}".rjust(12).encode("ascii"))
            dbf.write(f"{float(lons[0]):.8f}".rjust(18).encode("ascii"))
            dbf.write(f"{float(lats[0]):.8f}".rjust(18).encode("ascii"))
            dbf.write(f"{float(lons[-1]):.8f}".rjust(18).encode("ascii"))
            dbf.write(f"{float(lats[-1]):.8f}".rjust(18).encode("ascii"))
            dbf.write((ts[0][:24] if ts else "").ljust(24).encode("ascii"))
            dbf.write((ts[-1][:24] if ts else "").ljust(24).encode("ascii"))
            if include_attrs:
                dbf.write(f"{float(np.nanmean(wd)):.2f}".rjust(10).encode("ascii"))
                dbf.write(f"{float(np.nanmin(wd)):.2f}".rjust(10).encode("ascii"))
                dbf.write(f"{float(np.nanmax(wd)):.2f}".rjust(10).encode("ascii"))
            dbf.write(b"\x1a")

        # ── .prj ────────────────────────────────────────────────────────
        with open(prj_path, "w") as prj:
            if crs_str and crs_str not in ("EPSG:4326", "4326"):
                try:
                    from pyproj import CRS as _CRS
                    prj.write(_CRS.from_user_input(crs_str).to_wkt())
                except Exception:
                    prj.write(TopasSuite._WGS84_PRJ)
            else:
                prj.write(TopasSuite._WGS84_PRJ)

    _WGS84_PRJ = ('GEOGCS["GCS_WGS_1984",'
                  'DATUM["D_WGS_1984",'
                  'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                  'PRIMEM["Greenwich",0.0],'
                  'UNIT["Degree",0.0174532925199433]]')

    @staticmethod
    def _write_navline_geojson(path: str, lons, lats, dist, wd, ts,
                               include_attrs: bool, crs_str):
        """GeoJSON LineString con coordenadas por vértice y atributos opcionales."""
        import json as _json

        coords = [[float(x), float(y)] for x, y in zip(lons, lats)]

        props: dict = {
            "n_traces":   len(lons),
            "total_km":   round(float(dist[-1]), 4) if len(dist) else 0.0,
            "lon_start":  round(float(lons[0]),  8),
            "lat_start":  round(float(lats[0]),  8),
            "lon_end":    round(float(lons[-1]), 8),
            "lat_end":    round(float(lats[-1]), 8),
            "ts_start":   ts[0]  if ts else "",
            "ts_end":     ts[-1] if ts else "",
        }
        if include_attrs:
            props["wd_mean"] = round(float(np.nanmean(wd)), 2)
            props["wd_min"]  = round(float(np.nanmin(wd)), 2)
            props["wd_max"]  = round(float(np.nanmax(wd)), 2)
            # Per-vertex arrays stored as JSON arrays
            props["dist_km"]    = [round(float(v), 4) for v in dist]
            props["water_depth"]= [round(float(v), 2) for v in wd]
            props["timestamps"] = list(ts)

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": props,
        }
        crs_name = ("urn:ogc:def:crs:OGC:1.3:CRS84" if not crs_str
                    else f"urn:ogc:def:crs:EPSG::{crs_str.upper().replace('EPSG:','')}")
        fc = {
            "type": "FeatureCollection",
            "crs":  {"type": "name", "properties": {"name": crs_name}},
            "features": [feature],
        }
        out_path = path if path.lower().endswith(".geojson") else path + ".geojson"
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(fc, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _write_navline_csv(path: str, lons, lats, dist, wd, ts):
        """CSV con una fila por traza: trace_idx, lon, lat, dist_km, water_depth, timestamp."""
        import csv as _csv
        out_path = path if path.lower().endswith(".csv") else path + ".csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["trace_idx", "lon", "lat", "dist_km", "water_depth_m", "timestamp"])
            for i, (x, y, d, wdv, t) in enumerate(zip(lons, lats, dist, wd, ts)):
                w.writerow([i + 1,
                            round(float(x), 8), round(float(y), 8),
                            round(float(d), 4), round(float(wdv), 2),
                            t])

    # ── Exportación FIX: selector de formato ─────────────────────────────
    def _export_fix_shp(self, fixes: list, default_stem: str):
        if not fixes:
            messagebox.showwarning(
                "Sin FIX",
                "No hay marcas FIX calculadas.\n"
                "Activa 'Mostrar marcas FIX' y renderiza primero.")
            return

        out = filedialog.asksaveasfilename(
            initialfile=f"{default_stem}_FIX",
            filetypes=[
                ("Shapefile",  "*.shp"),
                ("GeoJSON",    "*.geojson"),
                ("CSV",        "*.csv"),
            ],
            defaultextension=".shp")
        if not out:
            return

        ext = Path(out).suffix.lower()
        # Garantizar extensión correcta si el diálogo no la añade
        if ext not in (".shp", ".geojson", ".csv"):
            out += ".shp"; ext = ".shp"

        # Preparar lista de puntos normalizada
        pts = [(num, hora, float(lon), float(lat))
               for num, _dist, hora, lon, lat in fixes]

        try:
            if ext == ".shp":
                self._write_shp_pure(out, pts)
                fmt_label = "Shapefile"
            elif ext == ".geojson":
                self._write_geojson(out, [(n, 0, h, lo, la) for n, h, lo, la in pts])
                fmt_label = "GeoJSON"
            else:
                self._write_csv(out, [(n, 0, h, lo, la) for n, h, lo, la in pts])
                fmt_label = "CSV"

            n = len(fixes)
            self._status(f"✔ {n} marcas FIX → {fmt_label}: {Path(out).name}")
            messagebox.showinfo(
                f"{fmt_label} exportado",
                f"{n} marcas FIX exportadas a:\n{out}\n\n"
                f"Atributos: fix_num · fix_hora · lon · lat\n"
                f"CRS: WGS84 (EPSG:4326)\n"
                f"(sin dependencias externas)")
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))
            self._status("✘ Error al exportar FIX.")

    def _export_fix_shp_vis(self):
        sd = self._last_fixes_sd
        stem = sd.stem if sd else "perfil"
        self._export_fix_shp(self._last_fixes, stem)

    def _export_fix_shp_chain(self):
        ch = self._last_chain_fixes_ch
        stem = (f"cadena_{len(ch.profiles)}p" if ch else "cadena")
        self._export_fix_shp(self._last_chain_fixes, stem)

    # ── Mapa ──────────────────────────────────────────────────────────────
    def _create_map_fig(self, sd):
        fig = Figure(figsize=(9, 6), facecolor=C["panel"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(C["entry"])

        wd   = sd.water_depth
        norm = mcolors.Normalize(np.nanmin(wd), np.nanmax(wd))
        cmap = _get_colormap("viridis")

        # Vectorized LineCollection — single draw call instead of N ax.plot() calls
        from matplotlib.collections import LineCollection
        pts    = np.column_stack([sd.lons, sd.lats])          # (N, 2)
        segs   = np.stack([pts[:-1], pts[1:]], axis=1)        # (N-1, 2, 2)
        wd_mid = (wd[:-1] + wd[1:]) / 2.0
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.0)
        lc.set_array(wd_mid)
        ax.add_collection(lc)
        ax.autoscale()

        _ts0 = sd.timestamps[0]  if sd.timestamps else "—"
        _ts1 = sd.timestamps[-1] if sd.timestamps else "—"
        ax.scatter(sd.lons[0],  sd.lats[0],  s=60, color=C["ok"],   zorder=5, label=f"Inicio ({_ts0})", edgecolors="white", lw=0.5)
        ax.scatter(sd.lons[-1], sd.lats[-1], s=60, color=C["warn"], zorder=5, label=f"Fin ({_ts1})",    edgecolors="white", lw=0.5)

        step = max(1, sd.n_traces // 10)
        for i in range(0, sd.n_traces, step):
            ax.annotate(str(i), (sd.lons[i], sd.lats[i]), color=C["sub"], fontsize=6, xytext=(3, 3), textcoords="offset points")

        cb = fig.colorbar(lc, ax=ax, pad=0.01, fraction=0.02)
        cb.set_label("Prof. agua (m)", color=C["sub"], fontsize=8)
        cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=7)

        ax.set_xlabel("Longitud (°)", color=C["text"], fontsize=9)
        ax.set_ylabel("Latitud (°)",  color=C["text"], fontsize=9)
        ax.set_title(f"Trayectoria  ·  {sd.total_km:.2f} km  ·  {sd.n_traces} trazas", color=C["text"], fontsize=10)
        ax.tick_params(colors=C["text"], labelsize=8)
        ax.grid(True, color=C["accent"], alpha=0.3, lw=0.5)
        ax.legend(facecolor=C["panel"], edgecolor=C["accent"], labelcolor=C["text"], fontsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(C["accent"])
        fig.tight_layout(pad=1.2)
        return fig

    # ══════════════════════════════════════════════════════════════════════════
    # ESPECTRO DE FRECUENCIA
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_spectrum(d_clean: np.ndarray, fs: float):
        """
        Calcula el espectro completo a partir de una matriz (ns × n_traces).

        Devuelve un dict con:
          freqs        – array de frecuencias (Hz)
          spec_mean_db – espectro medio en dB (Welch multi-traza, ventana Hann)
          spec_p10_db  – percentil 10  (trazas más débiles)
          spec_p50_db  – percentil 50  (mediana)
          spec_p90_db  – percentil 90  (trazas más fuertes)
          spec_2d_db   – espectro 2-D: (n_freqs × n_traces) en dB para mapa f-t
          peak_hz      – frecuencia de pico del espectro medio (Hz)
          centroid_hz  – frecuencia centroide (Hz)
          bw_3db_lo/hi – bordes de banda -3 dB (Hz)
          bw_6db_lo/hi – bordes de banda -6 dB (Hz)
          snr_db       – SNR estimado (señal 1–15 kHz vs. ruido >15 kHz)
          roll_off_hz  – frecuencia donde se acumula el 85 % de la energía (Hz)
        """
        from scipy.signal.windows import hann

        ns, n_tr = d_clean.shape
        # Potencia de 2 más cercana, mínimo 512, máximo 4096
        nfft = min(4096, max(512, 2 ** int(np.ceil(np.log2(ns)))))
        nfft = min(nfft, ns)
        # Aviso cuando el perfil es tan corto que nfft cae por debajo de 512:
        # la resolución frecuencial resultante es muy baja (Δf = fs/nfft).
        _low_res = nfft < 512

        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        win   = hann(nfft)

        # Espectro por traza: media de ventanas solapadas (Welch manual, 50 % overlap)
        step = nfft // 2
        n_frames = max(1, (ns - nfft) // step + 1)

        # Acumular potencia por traza con ventanas solapadas
        pwr = np.zeros((len(freqs), n_tr), dtype=np.float64)
        for k in range(n_frames):
            seg = d_clean[k * step: k * step + nfft, :] * win[:, None]
            pwr += np.abs(np.fft.rfft(seg, axis=0)) ** 2
        pwr /= n_frames

        # Normalizar cada traza a su máximo propio para comparación relativa
        pwr_norm = pwr / (pwr.max(axis=0, keepdims=True) + 1e-30)
        spec_2d_db = 10 * np.log10(pwr_norm + 1e-30)

        # Estadísticas sobre el ensemble de trazas
        pwr_mean = pwr.mean(axis=1)
        ref = pwr_mean.max() + 1e-30
        spec_mean_db = 10 * np.log10(pwr_mean / ref)

        pwr_p10 = np.percentile(pwr, 10, axis=1)
        pwr_p50 = np.percentile(pwr, 50, axis=1)
        pwr_p90 = np.percentile(pwr, 90, axis=1)
        spec_p10_db = 10 * np.log10(pwr_p10 / ref)
        spec_p50_db = 10 * np.log10(pwr_p50 / ref)
        spec_p90_db = 10 * np.log10(pwr_p90 / ref)

        # ── Métricas ──────────────────────────────────────────────────────
        peak_hz     = float(freqs[np.argmax(pwr_mean)])

        # Centroide espectral
        total_pwr   = pwr_mean.sum() + 1e-30
        centroid_hz = float((freqs * pwr_mean).sum() / total_pwr)

        # Ancho de banda -3 dB y -6 dB
        def _bandwidth(pwr_arr, freqs_arr, db_drop):
            thresh = pwr_arr.max() / (10 ** (db_drop / 10))
            above  = freqs_arr[pwr_arr >= thresh]
            if above.size < 2:
                if above.size == 0:
                    return float(freqs_arr[0]), float(freqs_arr[-1])
                return float(above[0]), float(above[0])
            return float(above[0]), float(above[-1])

        bw3_lo,  bw3_hi  = _bandwidth(pwr_mean, freqs, 3.0)
        bw6_lo,  bw6_hi  = _bandwidth(pwr_mean, freqs, 6.0)

        # Roll-off al 85 % de la energía acumulada
        cum_energy   = np.cumsum(pwr_mean)
        cum_energy  /= cum_energy[-1] + 1e-30
        roll_off_idx = np.searchsorted(cum_energy, 0.85)
        roll_off_hz  = float(freqs[min(roll_off_idx, len(freqs) - 1)])

        # SNR estimado: banda 0.5–15 kHz vs. >15 kHz
        sig_mask  = (freqs >= 500)  & (freqs <= 15000)
        noise_mask= freqs > 15000
        sig_pwr   = pwr_mean[sig_mask].mean()   if sig_mask.any()   else 1e-30
        noise_pwr = pwr_mean[noise_mask].mean() if noise_mask.any() else 1e-30
        snr_db    = float(10 * np.log10(sig_pwr / (noise_pwr + 1e-30)))

        return dict(
            freqs        = freqs,
            nfft         = nfft,
            n_frames     = n_frames,
            low_res      = _low_res,   # True when nfft < 512 (perfil muy corto)
            spec_mean_db = spec_mean_db,
            spec_p10_db  = spec_p10_db,
            spec_p50_db  = spec_p50_db,
            spec_p90_db  = spec_p90_db,
            spec_2d_db   = spec_2d_db,      # (n_freqs, n_traces)
            peak_hz      = peak_hz,
            centroid_hz  = centroid_hz,
            bw_3db_lo    = bw3_lo,  bw_3db_hi = bw3_hi,
            bw_6db_lo    = bw6_lo,  bw_6db_hi = bw6_hi,
            roll_off_hz  = roll_off_hz,
            snr_db       = snr_db,
        )

    @staticmethod
    def _draw_spectrum_figure(sp: dict, fs: float, title: str,
                               n_traces: int, dist_km: np.ndarray = None,
                               boundaries_km: list = None) -> "Figure":
        """
        Construye la figura completa del espectro con 3 paneles:
          [0] Espectro medio con percentiles + métricas
          [1] Mapa espectral 2-D (frecuencia vs. traza)
          [2] Tabla de métricas acústicas
        """
        freqs        = sp["freqs"]
        freqs_khz    = freqs / 1000.0
        f_max_khz    = min(20.0, fs / 2000.0)

        fig = Figure(figsize=(10, 8), facecolor=C["panel"])
        gs  = GridSpec(3, 2, figure=fig,
                       height_ratios=[2.8, 2.2, 0.9],
                       width_ratios=[3, 1],
                       hspace=0.52, wspace=0.35)

        # ── Panel 0 izquierda: Espectro medio + percentiles ────────────────
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor(C["entry"])

        # Relleno de incertidumbre p10–p90
        ax1.fill_between(freqs_khz,
                          sp["spec_p10_db"], sp["spec_p90_db"],
                          alpha=0.15, color=C["bright"], label="P10–P90")
        # Mediana
        ax1.plot(freqs_khz, sp["spec_p50_db"],
                 color=C["sub"], lw=0.9, ls="--", alpha=0.8, label="Mediana (P50)")
        # Media (Welch)
        ax1.plot(freqs_khz, sp["spec_mean_db"],
                 color=C["bright"], lw=1.5, label="Media (Welch)")

        # Pico
        pk_khz = sp["peak_hz"] / 1000.0
        pk_db  = float(sp["spec_mean_db"][np.argmax(sp["spec_mean_db"])])
        ax1.axvline(pk_khz, color=C["warn"], lw=1.0, ls="--", alpha=0.85)
        ax1.annotate(f" Pico\n {pk_khz:.2f} kHz",
                     xy=(pk_khz, pk_db), xytext=(pk_khz + 0.15, pk_db - 8),
                     color=C["warn"], fontsize=7, fontfamily="Courier New",
                     arrowprops=dict(arrowstyle="-", color=C["warn"], lw=0.7))

        # Centroide
        ct_khz = sp["centroid_hz"] / 1000.0
        ax1.axvline(ct_khz, color=C["highlight"], lw=0.9, ls=":", alpha=0.8,
                    label=f"Centroide {ct_khz:.2f} kHz")

        # Banda -3 dB
        bw3_lo_k = sp["bw_3db_lo"] / 1000.0
        bw3_hi_k = sp["bw_3db_hi"] / 1000.0
        ax1.axvspan(bw3_lo_k, bw3_hi_k, alpha=0.07, color=C["ok"],
                    label=f"BW -3 dB  ({bw3_lo_k:.1f}–{bw3_hi_k:.1f} kHz)")

        # Banda -6 dB (solo bordes)
        bw6_lo_k = sp["bw_6db_lo"] / 1000.0
        bw6_hi_k = sp["bw_6db_hi"] / 1000.0
        ax1.axvline(bw6_lo_k, color=C["ok"], lw=0.7, ls="-.", alpha=0.55)
        ax1.axvline(bw6_hi_k, color=C["ok"], lw=0.7, ls="-.", alpha=0.55,
                    label=f"BW -6 dB  ({bw6_lo_k:.1f}–{bw6_hi_k:.1f} kHz)")

        ax1.set_xlim(0, f_max_khz)
        ax1.set_ylim(-80, 3)
        ax1.set_xlabel("Frecuencia (kHz)", color=C["text"], fontsize=9)
        ax1.set_ylabel("Amplitud (dB re. máx)", color=C["text"], fontsize=9)
        _low_res_warn = "  ⚠ baja resolución (perfil corto)" if sp.get("low_res") else ""
        ax1.set_title(f"Espectro Welch  ·  {title}  ·  {n_traces} trazas  ·  "
                      f"NFFT={sp['nfft']}  ·  {sp['n_frames']} ventanas/traza{_low_res_warn}",
                      color=C["text"] if not sp.get("low_res") else C["warn"], fontsize=9)
        ax1.tick_params(colors=C["text"], labelsize=8)
        ax1.grid(True, color=C["accent"], alpha=0.3, lw=0.5)
        ax1.legend(facecolor=C["panel"], edgecolor=C["accent"],
                   labelcolor=C["text"], fontsize=6.5, loc="lower left",
                   ncol=3)
        for sp_ in ax1.spines.values(): sp_.set_edgecolor(C["accent"])

        # ── Panel 1 izquierda: Mapa espectral 2-D (f vs. traza) ───────────
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(C["entry"])

        spec_2d = sp["spec_2d_db"]                      # (n_freqs, n_traces)
        x_axis  = (dist_km if dist_km is not None
                   else np.arange(n_traces))
        x_label = "Distancia (km)" if dist_km is not None else "Traza"

        # Mostrar solo hasta f_max
        f_mask = freqs <= f_max_khz * 1000
        ax2.pcolormesh(
            x_axis, freqs_khz[f_mask], spec_2d[f_mask, :],
            cmap="inferno", vmin=-50, vmax=0, shading="auto")

        # Líneas de banda
        ax2.axhline(bw3_lo_k, color=C["ok"], lw=0.8, ls="--", alpha=0.6)
        ax2.axhline(bw3_hi_k, color=C["ok"], lw=0.8, ls="--", alpha=0.6)
        ax2.axhline(pk_khz,   color=C["warn"], lw=0.7, ls=":", alpha=0.7)

        # Fronteras de perfiles (solo cadenas)
        if boundaries_km:
            for b in boundaries_km:
                ax2.axvline(b, color=C["warn"], lw=0.7, ls="--", alpha=0.6)

        ax2.set_ylim(0, f_max_khz)
        ax2.set_xlabel(x_label,          color=C["text"], fontsize=8)
        ax2.set_ylabel("Frecuencia (kHz)", color=C["text"], fontsize=8)
        ax2.set_title("Mapa espectral 2-D  (frecuencia vs. distancia)",
                      color=C["text"], fontsize=8)
        ax2.tick_params(colors=C["text"], labelsize=7)
        for sp_ in ax2.spines.values(): sp_.set_edgecolor(C["accent"])

        # Barra de color del mapa 2-D
        sm2 = plt.cm.ScalarMappable(cmap="inferno", norm=mcolors.Normalize(-50, 0))
        sm2.set_array([])
        cb2 = fig.colorbar(sm2, ax=ax2, pad=0.01, fraction=0.03)
        cb2.set_label("dB re. máx", color=C["sub"], fontsize=7)
        cb2.ax.yaxis.set_tick_params(color=C["sub"], labelsize=6)

        # ── Panel 1 derecha: Distribución de energía por banda ────────────
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(C["entry"])

        # Banda TOPAS por defecto + otras bandas de interés
        bands = [
            ("< 1 kHz",   0,     1000),
            ("1–2 kHz",   1000,  2000),
            ("2–4 kHz",   2000,  4000),
            ("4–7 kHz",   4000,  7000),
            ("7–10 kHz",  7000,  10000),
            ("10–15 kHz", 10000, 15000),
            ("> 15 kHz",  15000, fs / 2),
        ]
        # Filtrar bandas a las que tienen datos
        pwr_mean_lin = 10 ** (sp["spec_mean_db"] / 10)
        band_labels = []; band_powers = []
        for label, flo, fhi in bands:
            mask = (freqs >= flo) & (freqs < fhi)
            if mask.any() and fhi <= fs / 2:
                pwr_band = pwr_mean_lin[mask].sum()
                band_labels.append(label)
                band_powers.append(pwr_band)

        total_bp = sum(band_powers) + 1e-30
        band_pcts = [100 * p / total_bp for p in band_powers]

        colors_bar = [C["accent"]] * len(band_labels)
        # Destacar la banda con más energía
        max_idx = int(np.argmax(band_pcts))
        colors_bar[max_idx] = C["bright"]

        bars = ax3.barh(band_labels, band_pcts, color=colors_bar,
                        edgecolor=C["bg"], linewidth=0.5)
        for bar, pct in zip(bars, band_pcts):
            if pct > 3:
                ax3.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{pct:.1f}%",
                         va="center", ha="left",
                         color=C["text"], fontsize=6, fontfamily="Courier New")

        ax3.set_xlabel("Energía (%)", color=C["text"], fontsize=7)
        ax3.set_title("Distribución\npor banda", color=C["text"], fontsize=8)
        ax3.tick_params(colors=C["text"], labelsize=6.5)
        ax3.set_xlim(0, max(band_pcts) * 1.18)
        for sp_ in ax3.spines.values(): sp_.set_edgecolor(C["accent"])
        ax3.set_facecolor(C["entry"])

        # ── Panel 2: tabla de métricas ─────────────────────────────────────
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")
        ax4.set_facecolor(C["panel"])

        bw3   = (sp["bw_3db_hi"]  - sp["bw_3db_lo"])  / 1000.0
        bw6   = (sp["bw_6db_hi"]  - sp["bw_6db_lo"])  / 1000.0
        ro_k  = sp["roll_off_hz"] / 1000.0

        metrics = [
            ("Frec. pico",    f"{pk_khz:.3f} kHz"),
            ("Centroide",     f"{ct_khz:.3f} kHz"),
            ("BW -3 dB",      f"{bw3:.2f} kHz  ({bw3_lo_k:.2f}–{bw3_hi_k:.2f})"),
            ("BW -6 dB",      f"{bw6:.2f} kHz  ({bw6_lo_k:.2f}–{bw6_hi_k:.2f})"),
            ("Roll-off 85%",  f"{ro_k:.2f} kHz"),
            ("SNR estimado",  f"{sp['snr_db']:.1f} dB"),
            ("NFFT",          f"{sp['nfft']}"),
            ("Ventanas",      f"{sp['n_frames']}/traza  (Welch 50%)"),
        ]

        col_x = [0.02, 0.14, 0.39, 0.52, 0.77, 0.89]
        row_y = [0.78, 0.44, 0.10]
        for col_idx, (lbl, val) in enumerate(metrics):
            cx = col_x[col_idx % 3 * 2]
            vx = col_x[col_idx % 3 * 2 + 1]
            cy = row_y[col_idx // 3]
            ax4.text(cx, cy, lbl + ":",
                     transform=ax4.transAxes,
                     color=C["sub"], fontsize=7, fontfamily="Courier New",
                     ha="left", va="center")
            ax4.text(vx, cy, val,
                     transform=ax4.transAxes,
                     color=C["text"], fontsize=7, fontfamily="Courier New",
                     ha="left", va="center", fontweight="bold")

        fig.tight_layout(pad=1.2)
        return fig

    # ── Espectro perfil ───────────────────────────────────────────────────
    def _create_spectrum_fig(self, sd, data):
        i0, i1, *_ = self._time_window(sd, data.shape[0])
        d = data[i0:i1, :]

        valid_traces = ~np.isnan(d).all(axis=0)
        if valid_traces.sum() == 0:
            return

        fs      = 1e6 / sd.dt_us
        d_clean = np.nan_to_num(d[:, valid_traces], nan=0.0)

        sp = self._compute_spectrum(d_clean, fs)
        fig = self._draw_spectrum_figure(
            sp, fs,
            title    = sd.name,
            n_traces = int(valid_traces.sum()),
            dist_km  = sd.dist_km[valid_traces],
            boundaries_km = []   # perfil individual: sin uniones
        )
        return fig

    # ── Cabeceras ─────────────────────────────────────────────────────────
    def _draw_headers(self, sd, token=None):
        if token is not None and getattr(self, "_render_id", None) != token: return
        parent = self.vtab_headers
        for w in parent.winfo_children():
            w.destroy()

        cols = ("Traza", "Dist (km)", "Lon (°)", "Lat (°)", "Prof. agua (m)", "TWT fondo (ms)", "Tiempo UTC", "Amp. máx")
        tree = ttk.Treeview(parent, columns=cols, show="headings")
        vsb  = ttk.Scrollbar(parent, orient="vertical",   command=tree.yview)
        hsb  = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")

        twt  = (sd.water_depth / 1500.0) * 2000.0
        amps = sd.amp_max          # cached at load — no disk access

        # Preformat all strings at once (much faster than per-row f-strings in a loop)
        rows = [
            (str(i+1),
             f"{sd.dist_km[i]:.3f}", f"{sd.lons[i]:.6f}", f"{sd.lats[i]:.6f}",
             f"{sd.water_depth[i]:.1f}", f"{twt[i]:.1f}",
             sd.timestamps[i], f"{amps[i]:.5f}")
            for i in range(sd.n_traces)
        ]
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        _hts0 = sd.timestamps[0]  if sd.timestamps else "—"
        _hts1 = sd.timestamps[-1] if sd.timestamps else "—"
        tk.Label(parent, text=(
            f"  {sd.n_traces} trazas  ·  {sd.total_km:.2f} km  ·  "
            f"Prof. {sd.water_depth.min():.0f}–{sd.water_depth.max():.0f} m  ·  "
            f"{_hts0}  →  {_hts1}"),
            bg=C["panel"], fg=C["sub"], font=("Courier New", 7)).pack(fill="x")

        # Insertar filas en chunks para no congelar la UI con perfiles grandes.
        # Cada lote se programa con after(0) para devolver el control al event loop.
        CHUNK = 500
        def _insert_chunk(start=0):
            tree.config(selectmode="none")
            for row in rows[start:start + CHUNK]:
                tree.insert("", "end", values=row)
            if start + CHUNK < len(rows):
                self.after(0, lambda: _insert_chunk(start + CHUNK))
        _insert_chunk()

    # ── Exportación optimizada en hilo secundario ──────────────────────────
    def _export_vis(self):
        sd = self.active_profile
        if sd is None:
            messagebox.showwarning("Sin perfil", "Selecciona un perfil primero.")
            return

        idx   = self.vis_nb.index(self.vis_nb.select())
        names = ["perfil", "mapa", "espectro", "cabeceras"]
        suf   = names[idx] if idx < len(names) else "export"

        out = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{sd.stem}_{suf}.png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not out: return

        try:
            dpi_val = int(self.dpi_var.get())
        except ValueError:
            dpi_val = 300

        if idx == 0:
            # Capturar params en hilo principal antes de lanzar el thread
            export_params = dict(
                decon    = self.decon_var.get(),    decon_op  = self.decon_op.get(),
                decon_gap= self.decon_gap.get(),    decon_wn  = self.decon_wn.get(),
                filt     = self.filt_var.get(),     flo       = self.flo.get(),
                fhi      = self.fhi.get(),          preset    = self.preset_var.get(),
                tvg      = self.tvg_var.get(),      tvg_alpha = self.tvg_alpha.get(),
                agc      = self.agc_var.get(),      agc_win   = self.agc_win.get(),
                align    = self.align_delays_var.get(),
                clip     = self.clip_var.get(),     cmap      = self.cmap_var.get(),
                inv_cmap = self.inv_cmap_var.get(), fix       = self.fix_var.get(),
                fix_iv   = self.fix_interval_var.get(),
            )
            self._progress_start(f"Exportando perfil a {dpi_val} DPI… (puede tardar)")
            self.update_idletasks()

            def export_task():
                try:
                    data = self._process_data(sd, export_params)
                    self._current_profile_data = data
                    i0, i1, t0, t1 = self._time_window(sd, data.shape[0], export_params)
                    d = data[i0:i1, :]
                    data_ns, data_tr = d.shape

                    # ── Dimensiones exactas al DPI elegido ──────────────────
                    if hasattr(self, "_profile_fig") and self._profile_fig is not None:
                        orig_w, orig_h = self._profile_fig.get_size_inches()
                    else:
                        orig_w, orig_h = 10.0, 6.0

                    fig_w, fig_h = orig_w, orig_h
                    target_w_px = int(fig_w * dpi_val)
                    target_h_px = int(fig_h * dpi_val)

                    vmax = float(np.percentile(np.abs(d[~np.isnan(d)]), export_params["clip"])) or 1.0

                    cmap_base = CMAPS[export_params["cmap"]]
                    cmap_name = cmap_base + "_r" if export_params["inv_cmap"] else cmap_base

                    # Remuestrear el RGBA al tamaño exacto de píxeles de la figura
                    # para que imshow no tenga que interpolar: 1 px figura = 1 px dato
                    self.after(0, lambda: self._progress_set(20))
                    rgba_full = _colormapped_image_parallel(d, cmap_name, 0.0, vmax)
                    from PIL import Image as _PILImage
                    self.after(0, lambda: self._progress_set(40))
                    rgba_resized = np.array(
                        _PILImage.fromarray(rgba_full).resize(
                            (target_w_px, target_h_px), _PILImage.LANCZOS))

                    self.after(0, lambda: self._progress_set(60))
                    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi_val, facecolor=C["panel"])
                    ax  = fig.add_subplot(111)
                    ax.set_facecolor(C["panel"])
                    ax.imshow(rgba_resized, aspect="auto", interpolation="none",
                              extent=[sd.dist_km[0], sd.dist_km[-1], t1, t0])

                    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(0.0, vmax))
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.015)
                    cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=7)
                    cb.set_label("Amplitud", color=C["sub"], fontsize=8)

                    preset_key = FILTER_PRESETS.get(export_params["preset"], "none")
                    preset_lbl = (f"  ·  {export_params['preset']}" if preset_key != "none" else "")
                    delay_lbl  = ("  ·  delays compensados" if export_params["align"] else "  ·  delays SIN compensar")
                    par_lbl    = f"  ·  {_N_WORKERS} núcleos" + ("  ·  GPU" if _GPU else "")
                    ax.set_title(f"{sd.name}  ·  {sd.n_traces} trazas  ·  {dpi_val} DPI{preset_lbl}{delay_lbl}{par_lbl}",
                                 color=C["text"], fontsize=10, pad=8)
                    ax.set_xlabel("Distancia (km)", color=C["text"], fontsize=9)
                    ax.set_ylabel("Tiempo (ms)",    color=C["text"], fontsize=9)
                    ax.tick_params(colors=C["text"], labelsize=8)
                    for sp in ax.spines.values(): sp.set_edgecolor(C["accent"])

                    if export_params["fix"] and self._last_fixes:
                        self._draw_fix_marks(ax, self._last_fixes, color=C["highlight"])

                    fig.tight_layout(pad=1.2)
                    self.after(0, lambda: self._progress_set(80))
                    fig.savefig(out, dpi=dpi_val, bbox_inches="tight", facecolor=C["panel"])
                    fig.clear()

                    # Medir dimensiones reales del fichero guardado
                    try:
                        from PIL import Image as _PILImg
                        with _PILImg.open(out) as _im:
                            real_w, real_h = _im.size
                        res_str = f"{real_w}×{real_h} px  ·  {dpi_val} DPI"
                    except Exception:
                        res_str = f"~{target_w_px}×{target_h_px} px  ·  {dpi_val} DPI"

                    self.after(0, lambda: self._progress_stop(
                        f"✔ Exportado a {dpi_val} DPI ({res_str}): {Path(out).name}"))
                    self.after(0, lambda: messagebox.showinfo(
                        "Exportación completada",
                        f"Imagen guardada en:\n{out}\nResolución: {res_str}"
                        f"\n({_N_WORKERS} núcleos CPU" + ("  +  GPU)" if _GPU else ")")))
                except Exception as e:
                    err = str(e)
                    try: fig.clear(); plt.close(fig)
                    except Exception: pass
                    self.after(0, lambda: self._progress_stop("✘ Error durante la exportación."))
                    self.after(0, lambda m=err: messagebox.showerror("Error al exportar", m))
            threading.Thread(target=export_task, daemon=True).start()

        else:
            tabs = [self.vtab_profile, self.vtab_map, self.vtab_spectrum, self.vtab_headers]
            target_fig = None
            for w in tabs[idx].winfo_children():
                if hasattr(w, "figure_ref"):
                    target_fig = w.figure_ref; break
                elif hasattr(w, "figure"):
                    target_fig = w.figure; break

            if target_fig is None:
                messagebox.showinfo("Vista vacía", "Renderiza primero la vista antes de exportar.")
                return

            self.after(0, lambda: self._progress_start(f"Exportando {suf} a {dpi_val} DPI…"))

            def save_task(fig=target_fig, _suf=suf):
                try:
                    # Re-serializar la figura al DPI de exportación para que
                    # todos los elementos vectoriales se rendericen a la resolución
                    # correcta en lugar de escalar la figura de pantalla.
                    import pickle as _pkl
                    import io as _io
                    buf = _io.BytesIO()
                    _pkl.dump(fig, buf)
                    buf.seek(0)
                    export_fig = _pkl.load(buf)
                    export_fig.set_dpi(dpi_val)

                    export_fig.savefig(out, dpi=dpi_val, bbox_inches="tight",
                                       facecolor=C["panel"])
                    export_fig.clear()

                    try:
                        from PIL import Image as _PILImg
                        with _PILImg.open(out) as _im:
                            real_w, real_h = _im.size
                        res_str = f"{real_w}×{real_h} px  ·  {dpi_val} DPI"
                    except Exception:
                        res_str = f"{dpi_val} DPI"

                    self.after(0, lambda: self._progress_stop(
                        f"✔ Exportado ({res_str}): {Path(out).name}"))
                    self.after(0, lambda: messagebox.showinfo(
                        "Exportación completada",
                        f"Imagen guardada en:\n{out}\nResolución: {res_str}"))
                except Exception as e:
                    err = str(e)
                    try: export_fig.clear(); plt.close(export_fig)
                    except Exception: pass
                    self.after(0, lambda: self._progress_stop("✘ Error durante la exportación."))
                    self.after(0, lambda m=err: messagebox.showerror("Error al exportar", m))
            threading.Thread(target=save_task, daemon=True).start()

    # ════════════════════════════════════════════════════════════════════════
    # PESTAÑA C — CADENAS DE PERFILES CONTIGUOS
    # ════════════════════════════════════════════════════════════════════════
    def _build_tab_chains(self, parent):
        paned = tk.PanedWindow(parent, orient="horizontal", bg=C["bg"], sashwidth=5)
        paned.pack(fill="both", expand=True)

        ctrl = self._build_chain_controls(paned)
        paned.add(ctrl, minsize=200, width=220)

        right = tk.Frame(paned, bg=C["bg"])
        paned.add(right, minsize=500)

        # Zona de visualización: Notebook con subpestañas
        self.chain_nb = ttk.Notebook(right)
        self.chain_nb.pack(fill="both", expand=True)

        self.ctab_profile  = tk.Frame(self.chain_nb, bg=C["bg"])
        self.ctab_map      = tk.Frame(self.chain_nb, bg=C["bg"])
        self.ctab_spectrum = tk.Frame(self.chain_nb, bg=C["bg"])
        self.ctab_headers  = tk.Frame(self.chain_nb, bg=C["bg"])

        self.chain_nb.add(self.ctab_profile,  text="  Perfil  ")
        self.chain_nb.add(self.ctab_map,      text="  Mapa  ")
        self.chain_nb.add(self.ctab_spectrum, text="  Espectro  ")
        self.chain_nb.add(self.ctab_headers,  text="  Cabeceras  ")

        for tab in (self.ctab_profile, self.ctab_map, self.ctab_spectrum, self.ctab_headers):
            self._placeholder(tab, "Detecta cadenas primero y selecciona una en la lista izquierda")

        self._chain_fig = None

    def _build_chain_controls(self, parent):
        outer = tk.Frame(parent, bg=C["panel"], width=230)
        outer.pack_propagate(False)

        vsb = ttk.Scrollbar(outer, orient="vertical")
        vsb.pack(side="right", fill="y")

        cv = tk.Canvas(outer, bg=C["panel"], yscrollcommand=vsb.set, highlightthickness=0, width=210)
        cv.pack(side="left", fill="both", expand=True)
        vsb.config(command=cv.yview)

        sb = tk.Frame(cv, bg=C["panel"])
        win_id = cv.create_window((0, 0), window=sb, anchor="nw")

        def _on_frame_configure(e): cv.configure(scrollregion=cv.bbox("all"))
        def _on_canvas_configure(e): cv.itemconfig(win_id, width=e.width)
        def _on_mousewheel(e):
            delta = getattr(e, 'delta', 0)
            if delta:
                cv.yview_scroll(int(-delta / 120), "units")

        sb.bind("<Configure>", _on_frame_configure)
        cv.bind("<Configure>", _on_canvas_configure)
        cv.bind("<MouseWheel>",  _on_mousewheel)
        cv.bind("<Button-4>",    lambda e: cv.yview_scroll(-1, "units"))
        cv.bind("<Button-5>",    lambda e: cv.yview_scroll( 1, "units"))
        def _ctrl_scroll_chain(e):
            try:
                rx, ry = outer.winfo_rootx(), outer.winfo_rooty()
                rw, rh = outer.winfo_width(), outer.winfo_height()
                if rx <= e.x_root < rx + rw and ry <= e.y_root < ry + rh:
                    cv.yview_scroll(int(-e.delta / 120), "units")
            except Exception:
                pass
        outer.bind_all("<MouseWheel>", _ctrl_scroll_chain, add="+")

        def sec(t):
            tk.Frame(sb, bg=C["accent"], height=1).pack(fill="x", pady=(8, 2))
            tk.Label(sb, text=t, bg=C["panel"], fg=C["bright"], font=("Courier New", 7, "bold"), anchor="w").pack(fill="x", padx=8, pady=(0, 4))
        def lbl(t): tk.Label(sb, text=t, bg=C["panel"], fg=C["sub"], font=("Courier New", 7), anchor="w").pack(fill="x", padx=8, pady=(4, 0))
        def scale(var, lo, hi, res=1, length=190):
            tk.Scale(sb, from_=lo, to=hi, orient="horizontal", variable=var, bg=C["panel"], fg=C["text"],
                     highlightthickness=0, troughcolor=C["accent"], activebackground=C["bright"], font=("Courier New", 7),
                     length=length, resolution=res).pack(padx=8)

        # ── Paleta ──
        sec("PALETA")
        self.chain_cmap_var = tk.StringVar(value="Blanco / Negro")
        ttk.Combobox(sb, textvariable=self.chain_cmap_var, values=list(CMAPS.keys()), state="readonly", font=("Courier New", 7)).pack(fill="x", padx=8, pady=2)
        self.chain_inv_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Invertir colores", variable=self.chain_inv_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(0, 4))

        # ── Deconvolución Predictiva Cadenas ──
        sec("DECONVOLUCIÓN PREDICTIVA")
        self.chain_decon_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Activar Deconvolución", variable=self.chain_decon_var, bg=C["panel"], fg=C["warn"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7, "bold")).pack(anchor="w", padx=8, pady=(2, 0))
        
        lbl("Longitud operador (ms):"); self.chain_decon_op = tk.DoubleVar(value=10.0)
        scale(self.chain_decon_op, 1.0, 50.0, res=1.0)
        
        lbl("Gap / Lag predicción (ms):"); self.chain_decon_gap = tk.DoubleVar(value=2.0)
        scale(self.chain_decon_gap, 0.1, 20.0, res=0.1)
        
        lbl("Ruido blanco pre-whitening (%):"); self.chain_decon_wn = tk.DoubleVar(value=1.0)
        scale(self.chain_decon_wn, 0.1, 10.0, res=0.1)

        # ── Filtro Pasabanda Cadenas ──
        sec("FILTRO Pasabanda")
        self.chain_filt_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Activar filtro", variable=self.chain_filt_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7)).pack(anchor="w", padx=8)
        lbl("F low (Hz):"); self.chain_flo = tk.IntVar(value=2000)
        scale(self.chain_flo, 500, 15000, res=100)
        lbl("F high (Hz):"); self.chain_fhi = tk.IntVar(value=7000)
        scale(self.chain_fhi, 500, 15000, res=100)

        # ── Ganancia y Filtros ──
        sec("CLIP  /  GANANCIA")
        lbl("Clip amplitud (%):"); self.chain_clip_var = tk.IntVar(value=98)
        scale(self.chain_clip_var, 80, 100)

        # --- TVG Cadenas ---
        self.chain_tvg_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="TVG Adaptativo (Compensar α)", variable=self.chain_tvg_var, bg=C["panel"], fg=C["highlight"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7, "bold")).pack(anchor="w", padx=8, pady=(4,0))
        lbl("Coef. atenuación α:"); self.chain_tvg_alpha = tk.DoubleVar(value=15.0)
        scale(self.chain_tvg_alpha, 0.0, 150.0, res=1)

        self.chain_agc_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Aplicar AGC", variable=self.chain_agc_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(4,0))
        lbl("Ventana AGC (ms):"); self.chain_agc_win = tk.IntVar(value=20)
        scale(self.chain_agc_win, 5, 100, res=5)

        self.chain_align_delays_var = tk.BooleanVar(value=True)
        tk.Checkbutton(sb, text="Compensar delays (alinear grupos)", variable=self.chain_align_delays_var,
                       bg=C["panel"], fg=C["ok"], selectcolor=C["accent"], activebackground=C["panel"],
                       font=("Courier New", 7)).pack(anchor="w", padx=8, pady=(4, 0))

        # ── Filtros preestablecidos ──
        sec("FILTROS  PREESTABLECIDOS")
        self.chain_preset_var = tk.StringVar(value=list(FILTER_PRESETS.keys())[0])
        ttk.Combobox(sb, textvariable=self.chain_preset_var, values=list(FILTER_PRESETS.keys()),
                     state="readonly", font=("Courier New", 7)).pack(fill="x", padx=8, pady=(2, 4))
        self.lbl_chain_preset_desc = tk.Label(sb, text="", bg=C["panel"], fg=C["sub"], font=("Courier New", 6),
                                        anchor="w", justify="left", wraplength=200, padx=8)
        self.lbl_chain_preset_desc.pack(fill="x")
        def _update_chain_preset_desc(*_):
            key = FILTER_PRESETS.get(self.chain_preset_var.get(), "none")
            self.lbl_chain_preset_desc.config(text=FILTER_DESCRIPTIONS.get(key, ""))
        self.chain_preset_var.trace_add("write", _update_chain_preset_desc)
        _update_chain_preset_desc()

        # ── Marcas FIX de tiempo ──
        sec("MARCAS  FIX")
        self.chain_fix_var = tk.BooleanVar(value=False)
        tk.Checkbutton(sb, text="Mostrar marcas FIX", variable=self.chain_fix_var, bg=C["panel"], fg=C["highlight"],
                       selectcolor=C["accent"], activebackground=C["panel"], font=("Courier New", 7, "bold")).pack(anchor="w", padx=8)
        fix_row = tk.Frame(sb, bg=C["panel"])
        fix_row.pack(fill="x", padx=8, pady=(2, 4))
        tk.Label(fix_row, text="Intervalo (min):", bg=C["panel"], fg=C["sub"], font=("Courier New", 7)).pack(side="left")
        self.chain_fix_interval_var = tk.IntVar(value=15)
        tk.Spinbox(fix_row, textvariable=self.chain_fix_interval_var, from_=1, to=120, increment=1, width=5,
                   bg=C["entry"], fg=C["text"], insertbackground=C["text"], relief="flat", font=("Courier New", 7),
                   buttonbackground=C["accent"]).pack(side="left", padx=4)

        tk.Frame(sb, bg=C["accent"], height=1).pack(fill="x", pady=10)
        self._btn(sb, "⟳  RENDERIZAR", self._render_chain, C["bright"], fill="x", padx=8, pady=4)
        self._btn(sb, "📊  Calcular Espectro", self._render_spectrum_chain, C["accent"], fill="x", padx=8, pady=(0, 8))
        
        # ── Control de DPI para exportación ──
        sec("EXPORTACIÓN")
        lbl("Resolución (DPI):")
        self.chain_dpi_var = tk.StringVar(value="300")
        ttk.Combobox(sb, textvariable=self.chain_dpi_var, values=["150", "300", "600", "1200"], font=("Courier New", 7)).pack(fill="x", padx=8, pady=(2, 8))

        self._btn(sb, "💾  Exportar vista", self._export_chain, C["highlight"], fill="x", padx=8, pady=(0, 4))
        self._btn(sb, "🗺  Exportar FIX → SHP / GeoJSON / CSV", self._export_fix_shp_chain, C["warn"], fill="x", padx=8, pady=(0, 12))

        return outer

    # ── Placeholder ─────────────────────────────────────────────────────────
    def _placeholder_chain(self):
        for tab in (self.ctab_profile, self.ctab_map, self.ctab_spectrum, self.ctab_headers):
            self._placeholder(tab, "Detecta y renderiza una cadena para ver sus datos.")

    # ════════════════════════════════════════════════════════════════════════
    # LÓGICA DE DETECCIÓN
    # ════════════════════════════════════════════════════════════════════════
    def _run_chain_detection(self):
        valid = [p for p in self.profiles.values() if not p.error]
        if len(valid) < 1:
            messagebox.showwarning("Sin perfiles",
                                   "No hay perfiles SEG-Y válidos cargados.")
            return

        gap = float(self.chain_gap_var.get())
        self.chains = ProfileChain.detect(valid, gap_km=gap)
        self._refresh_chain_list()

        n_multi = sum(1 for c in self.chains if len(c.profiles) > 1)
        msg = (f"✔ {len(self.chains)} cadena(s) detectada(s) "
               f"({n_multi} con ≥2 perfiles) · umbral {gap:.1f} km.")
        self._status(msg)
        self._update_rep_sel_label()

    def _refresh_chain_list(self):
        self.chain_list.delete(0, tk.END)
        for i, ch in enumerate(self.chains):
            n = len(ch.profiles)
            tag = "⛓" if n > 1 else "○"
            label = f"{tag} [{n}p · {ch.total_km:.1f} km]  {self.chains[i].profiles[0].name}"
            self.chain_list.insert(tk.END, label)
            fg = C["bright"] if n > 1 else C["sub"]
            self.chain_list.itemconfig(tk.END, fg=fg)
        if self.chains:
            self.chain_list.selection_set(0)
            self._activate_chain(self.chains[0])
        else:
            self.lbl_chain_info.config(
                text='Carga perfiles y pulsa "🔍 Detectar".',
                fg=C["sub"])

    def _on_chain_select(self, event):
        sel = self.chain_list.curselection()
        if not sel:
            return
        self._activate_chain(self.chains[sel[0]])

    def _activate_chain(self, ch: ProfileChain):
        self.active_chain = ch
        n = len(ch.profiles)
        cts0 = ch.timestamps[0]  if ch.timestamps else "—"
        cts1 = ch.timestamps[-1] if ch.timestamps else "—"
        info = (
            f"Perfiles: {n}\n"
            f"Trazas totales: {ch.n_traces}\n"
            f"Longitud: {ch.total_km:.2f} km\n"
            f"dt: {ch.dt_us} µs  ·  ns: {ch.ns}\n"
            f"Inicio: {cts0}\n"
            f"Fin:    {cts1}\n"
            + ("Uniones en: " + ", ".join(f"{b:.1f} km" for b in ch.boundaries_km)
               if ch.boundaries_km else "Sin uniones (perfil único)")
        )
        self.lbl_chain_info.config(text=info, fg=C["sub"])
        self._status(f"Cadena activa: {n} perfil(es) · {ch.total_km:.2f} km")

    # ════════════════════════════════════════════════════════════════════════
    # RENDERIZADO DE CADENA
    # ════════════════════════════════════════════════════════════════════════
    def _render_chain(self):
        ch = self.active_chain
        if ch is None:
            messagebox.showwarning("Sin cadena",
                                   "Detecta cadenas primero y selecciona una.")
            return

        self._chain_render_id = getattr(self, "_chain_render_id", 0) + 1
        current_id = self._chain_render_id

        self.update_idletasks()
        # Calculamos el espacio vertical disponible restando un margen para la barra de herramientas y scroll
        avail_h_px = max(400, self.ctab_profile.winfo_height() - 60)

        # Capturar todos los valores de tk.Var antes del hilo (thread-safe)
        chain_params = dict(
            decon     = self.chain_decon_var.get(),    decon_op  = self.chain_decon_op.get(),
            decon_gap = self.chain_decon_gap.get(),    decon_wn  = self.chain_decon_wn.get(),
            filt      = self.chain_filt_var.get(),     flo       = self.chain_flo.get(),
            fhi       = self.chain_fhi.get(),          preset    = self.chain_preset_var.get(),
            tvg       = self.chain_tvg_var.get(),      tvg_alpha = self.chain_tvg_alpha.get(),
            agc       = self.chain_agc_var.get(),      agc_win   = self.chain_agc_win.get(),
            align     = self.chain_align_delays_var.get(),
            clip      = self.chain_clip_var.get(),     cmap      = self.chain_cmap_var.get(),
            inv_cmap  = self.chain_inv_var.get(),      fix       = self.chain_fix_var.get(),
            fix_iv    = self.chain_fix_interval_var.get(),
        )

        self._progress_start(f"Renderizando cadena {ch.label}…")
        self.update_idletasks()

        def worker():
            try:
                self.after(0, lambda: self._progress_update(
                    f"Procesando datos de cadena  ·  {ch.label}  ·  {len(ch.profiles)} perfiles…"))
                data = self._process_chain_data(ch, chain_params)
                self._current_chain_data = data
                if self._chain_render_id != current_id:
                    self.after(0, lambda: self._progress_stop()); return

                self.after(0, lambda: self._progress_update(
                    f"Dibujando perfil de cadena  ·  {ch.label}  ·  {ch.total_km:.1f} km…"))
                fig_prof = self._build_chain_figure(ch, data, avail_h_px, chain_params)
                if self._chain_render_id != current_id:
                    plt.close(fig_prof)
                    self.after(0, lambda: self._progress_stop()); return

                


                self.after(0, lambda: self._progress_update(
                    f"Generando mapa de cadena  ·  {ch.label}…"))
                fig_map = self._create_chain_map_fig(ch)
                if self._chain_render_id != current_id:
                    plt.close(fig_prof); plt.close(fig_map)
                    self.after(0, lambda: self._progress_stop()); return

                


                self.after(0, lambda: self._placeholder(self.ctab_spectrum, "Pulsa 'Calcular Espectro' para visualizarlo."))

                self.after(0, lambda: self._display_chain_figure(fig_prof, ch))
                self.after(0, lambda: self._embed(self.ctab_map, fig_map))
                self.after(0, lambda: self._draw_chain_headers(ch, current_id))
                self.after(0, lambda: self._progress_stop(
                    f"✔ Cadena {ch.label}  ·  {len(ch.profiles)} perfil(es)  ·  {ch.total_km:.2f} km"))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._progress_stop(f"✘ Error al renderizar cadena"))
                self.after(0, lambda: messagebox.showerror(
                    "Error al renderizar", err))

        threading.Thread(target=worker, daemon=True).start()

    def _process_chain_data(self, ch: ProfileChain, params: dict = None):
        """params: dict capturado en hilo principal; si None, lee tk.Var directamente."""
        p = params
        def _get(key, tkvar):
            return p[key] if p is not None else tkvar.get()

        data = ch.data.copy()

        # 1. Deconvolución Predictiva Cadenas
        if _get("decon", self.chain_decon_var):
            data = apply_predictive_decon(data, ch.dt_us,
                                          _get("decon_op",  self.chain_decon_op),
                                          _get("decon_gap", self.chain_decon_gap),
                                          _get("decon_wn",  self.chain_decon_wn))

        # 2. Filtro Pasabanda Cadenas
        if _get("filt", self.chain_filt_var):
            fs  = 1e6 / ch.dt_us
            flo = max(10, _get("flo", self.chain_flo))
            fhi = min(fs/2 - 1, _get("fhi", self.chain_fhi))
            if flo < fhi:
                sos  = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")
                def _sosfilt_chain(block, _sos=sos):
                    return sp_signal.sosfilt(_sos, block, axis=0).astype(np.float32)
                data = _parallel_apply(_sosfilt_chain, data)

        # 3. Atributos / Filtros Preestablecidos
        preset_key = FILTER_PRESETS.get(_get("preset", self.chain_preset_var), "none")
        if preset_key != "none":
            data = apply_filter_preset(data, preset_key, ch.dt_us)

        # 4. Compensación TVG Cadenas
        if _get("tvg", self.chain_tvg_var):
            alpha = _get("tvg_alpha", self.chain_tvg_alpha)
            t_sec = np.arange(data.shape[0], dtype=np.float32) * (ch.dt_us / 1e6)
            gain_curve = np.clip(np.exp(alpha * t_sec), 0.0, 1e9)
            data *= gain_curve[:, np.newaxis]

        # 5. Control Automático de Ganancia (AGC)
        if _get("agc", self.chain_agc_var):
            win_s = max(3, int(_get("agc_win", self.chain_agc_win) / (ch.dt_us / 1000.0)))
            if win_s % 2 == 0:
                win_s += 1
            if _GPU:
                import cupy as cp
                try:
                    from cupyx.scipy.ndimage import uniform_filter1d as cu_uf
                    g = cp.asarray(np.abs(data))
                    rms = cu_uf(g, size=win_s, axis=0)
                    rms = cp.maximum(rms, 1e-9)
                    data = cp.asnumpy(cp.asarray(data) / rms).astype(np.float32)
                except Exception:
                    from scipy.ndimage import uniform_filter1d
                    env = np.abs(data)
                    rms = uniform_filter1d(env, size=win_s, axis=0)
                    data = (data / np.maximum(rms, 1e-9)).astype(np.float32)
            else:
                from scipy.ndimage import uniform_filter1d
                env = np.abs(data)
                rms = uniform_filter1d(env, size=win_s, axis=0)
                data = (data / np.maximum(rms, 1e-9)).astype(np.float32)

        # 6. Compensación de delays — vectorizado
        if _get("align", self.chain_align_delays_var):
            dt_ms = ch.dt_us / 1000.0
            offsets = np.round((ch.delays - ch.min_delay) / dt_ms).astype(int)
            extra_samples = int(offsets.max())
            new_ns = ch.ns + extra_samples

            aligned_data = np.full((new_ns, ch.n_traces), np.nan, dtype=np.float32)
            row_idx = (np.arange(ch.ns)[:, None] + offsets[None, :])
            col_idx = np.arange(ch.n_traces)[None, :]
            aligned_data[row_idx, col_idx] = data

            data = aligned_data
            
        return data

    def _chain_time_window(self, ch: ProfileChain, data_ns: int, params: dict = None):
        i0 = 0
        i1 = data_ns
        align = params["align"] if params is not None else self.chain_align_delays_var.get()
        if align:
            t0 = ch.min_delay
            t1 = ch.min_delay + data_ns * ch.dt_us / 1000.0
        else:
            t0 = ch.delay_ms
            t1 = ch.delay_ms + data_ns * ch.dt_us / 1000.0
        return i0, i1, t0, t1

    def _build_chain_figure(self, ch: ProfileChain, data: np.ndarray, avail_h_px: int = 600,
                            params: dict = None) -> Figure:
        p = params
        def _get(key, tkvar):
            return p[key] if p is not None else tkvar.get()
        cmap_base = CMAPS[_get("cmap", self.chain_cmap_var)]
        cmap_name = cmap_base + "_r" if _get("inv_cmap", self.chain_inv_var) else cmap_base
        clip_pct  = int(_get("clip", self.chain_clip_var))

        # ── Time window (needed for imshow extent) ────────────────────────────
        i0, i1, t0, t1 = self._chain_time_window(ch, data.shape[0], params)

        # ── Auto-sizing ──────────────────────────────────────────────────────
        # Exact same as single profile viewer
        # Each sub-profile gets the same 10-inch width as the individual viewer.
        # Height stays at 6 inches, same as individual. This preserves proportions.
        n_profs = max(1, len(ch.profiles))
        base_fig_w = 10.0 * n_profs
        base_fig_h = 6.0
        
        # Ajustamos el alto al espacio disponible y escalamos el ancho proporcionalmente
        fig_h = max(4.0, avail_h_px / 100.0)  # Asumiendo 100 DPI por defecto de Matplotlib
        scale = fig_h / base_fig_h
        fig_w = base_fig_w * scale

        # ── Colorización por segmento: cada perfil con su propio vmax ──────
        # Esto garantiza el mismo contraste que el visor individual.
        # Un vmax global elevaría el rango si un perfil tiene amplitudes
        # atípicas, lavando el resto de la cadena.
        rgba_segs = []
        vmax_list = []
        start_tr  = 0
        for prof in ch.profiles:
            end_tr = start_tr + prof.n_traces
            seg    = data[:, start_tr:end_tr]
            # NaN → 0 antes de colorizar (zonas vacías = amplitud cero, no negro)
            seg_clean = np.nan_to_num(seg, nan=0.0)
            valid  = np.abs(seg_clean[seg_clean != 0.0])
            vm     = float(np.percentile(valid, clip_pct)) if valid.size > 0 else 1.0
            vm     = vm or 1.0
            vmax_list.append(vm)
            rgba_segs.append(_colormapped_image_parallel(seg_clean, cmap_name, 0.0, vm))
            start_tr = end_tr

        rgba = np.concatenate(rgba_segs, axis=1) if rgba_segs else np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
        # vmax para la colorbar: mediana de los vmaxes por perfil
        vmax = float(np.median(vmax_list)) if vmax_list else 1.0
        vmin = 0.0

        fig = Figure(figsize=(fig_w, fig_h), facecolor=C["panel"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(C["panel"])

        ax.imshow(rgba, aspect="auto", interpolation="bilinear",
                  extent=[ch.dist_km[0], ch.dist_km[-1], t1, t0])

        for b_km in ch.boundaries_km:
            ax.axvline(b_km, color=C["warn"], lw=0.8, ls="--", alpha=0.7)
            ax.text(b_km + 0.05, t0 + 4,
                    f"↕ {b_km:.1f} km",
                    color=C["warn"], fontsize=6,
                    fontfamily="Courier New", va="top")

        sm = plt.cm.ScalarMappable(cmap=cmap_name,
                                    norm=mcolors.Normalize(vmin, vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.005, fraction=0.008)
        cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=6)
        cb.set_label("Amplitud", color=C["sub"], fontsize=7)

        ax.set_xlabel("Distancia acumulada (km)", color=C["text"], fontsize=9)
        ax.set_ylabel("Tiempo (ms)", color=C["text"], fontsize=9)
        n = len(ch.profiles)
        _preset_name = _get("preset", self.chain_preset_var)
        preset_key = FILTER_PRESETS.get(_preset_name, "none")
        preset_lbl = (f"  ·  {_preset_name}" if preset_key != "none" else "")
        delay_lbl  = ("  ·  delays compensados" if _get("align", self.chain_align_delays_var) else "  ·  delays SIN compensar")
        ax.set_title(
            f"Cadena  ·  {n} perfil(es)  ·  {ch.total_km:.2f} km  ·  "
            f"{ch.n_traces} trazas  ·  {ch.dt_us} µs{preset_lbl}{delay_lbl}",
            color=C["text"], fontsize=9, pad=6)
        ax.tick_params(colors=C["text"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["accent"])

        # ── Marcas FIX ──
        if _get("fix", self.chain_fix_var):
            fixes = self._compute_fix_positions(
                ch.timestamps, ch.dist_km, ch.lons, ch.lats,
                int(_get("fix_iv", self.chain_fix_interval_var)))
            self._last_chain_fixes    = fixes
            self._last_chain_fixes_ch = ch
            self._draw_fix_marks(ax, fixes, color=C["highlight"])
        else:
            self._last_chain_fixes    = []
            self._last_chain_fixes_ch = None

        # Store time reference for _on_chain_zoom
        self._last_chain_t0 = t0
        fig.tight_layout(pad=1.0)
        return fig

    def _display_chain_figure(self, fig: Figure, ch: ProfileChain):
        """
        Embed chain figure in a horizontal-scroll canvas at its natural pixel size.

        The figure is generated at 10 inches × n_profiles width (same per-trace density
        as the individual viewer) and displayed inside a scrollable area so the user
        can pan horizontally.  The NavigationToolbar sits below, outside the scroll
        area, so Matplotlib zoom/pan work exactly like in the single-profile viewer.
        """
        # Close old figure
        if self._chain_fig is not None:
            try: plt.close(self._chain_fig)
            except Exception: pass
        self._chain_fig = fig

        # Destroy all current children of ctab_profile
        for w in list(self.ctab_profile.winfo_children()):
            old_f = getattr(w, "figure_ref", None)
            if old_f is not None:
                try: plt.close(old_f)
                except Exception: pass
            try: w.destroy()
            except Exception: pass

        # Figure pixel dimensions (100 dpi default for Matplotlib figures)
        dpi     = fig.get_dpi()
        fig_px_w = int(fig.get_figwidth()  * dpi)
        fig_px_h = int(fig.get_figheight() * dpi)

        # ── Toolbar at bottom (outside scroll) ────────────────────────────────
        tb_frame = tk.Frame(self.ctab_profile, bg=C["panel"])
        tb_frame.pack(side="bottom", fill="x")

        # ── Horizontal scrollbar ──────────────────────────────────────────────
        hbar = ttk.Scrollbar(self.ctab_profile, orient="horizontal")
        hbar.pack(side="bottom", fill="x")

        # ── Viewport canvas (fills remaining space) ───────────────────────────
        scroll_cv = tk.Canvas(
            self.ctab_profile, bg=C["panel"],
            xscrollcommand=hbar.set,
            highlightthickness=0)
        scroll_cv.pack(fill="both", expand=True)
        hbar.config(command=scroll_cv.xview)

        # ── Inner frame fixed at exact figure pixel dimensions ────────────────
        inner = tk.Frame(scroll_cv, bg=C["panel"],
                         width=fig_px_w, height=fig_px_h)
        inner.pack_propagate(False)
        scroll_cv.create_window(0, 0, anchor="nw", window=inner)
        scroll_cv.config(scrollregion=(0, 0, fig_px_w, fig_px_h))

        # ── Matplotlib canvas, pinned to exact figure size ────────────────────
        mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
        mpl_widget = mpl_canvas.get_tk_widget()
        mpl_widget.config(width=fig_px_w, height=fig_px_h)
        mpl_widget.figure_ref = fig
        mpl_widget.pack(fill="both", expand=True)

        # ── Toolbar (attached to Matplotlib canvas) ───────────────────────────
        tb = CustomToolbar(mpl_canvas, tb_frame, callback=self._on_chain_zoom)
        tb.config(bg=C["panel"])
        tb.update()
        tb.pack(fill="x")

        # ── Scroll bindings ───────────────────────────────────────────────────
        def _hscroll(e):
            scroll_cv.xview_scroll(int(-e.delta / 120), "units")
        scroll_cv.bind("<Shift-MouseWheel>", _hscroll)
        scroll_cv.bind("<MouseWheel>",       _hscroll)   # horizontal by default in chain

        mpl_canvas.draw()

        # Save references for any follow-up code
        self._chain_canvas_obj  = mpl_canvas
        self._chain_scroll_cv   = scroll_cv

        self._status(
            f"✔ Cadena renderizada  ·  {len(ch.profiles)} perfil(es)  ·  "
            f"{ch.total_km:.2f} km  ·  {fig_px_w}×{fig_px_h} px")

    def _on_chain_zoom(self):
        data = getattr(self, "_current_chain_data", None)
        ch = self.active_chain
        if data is None or ch is None or not hasattr(self, "_chain_canvas_obj"): return
        if not hasattr(self, "_chain_fig") or self._chain_fig is None: return
        if not self._chain_fig.axes or not self._chain_fig.axes[0].images: return
        
        fig = self._chain_fig
        ax = fig.axes[0]
        x0, x1 = ax.get_xlim()
        y1, y0 = ax.get_ylim()
        
        dist = ch.dist_km
        tr0 = max(0, min(np.searchsorted(dist, min(x0, x1)), len(dist) - 1))
        tr1 = max(0, min(np.searchsorted(dist, max(x0, x1)), len(dist)))
        if tr1 <= tr0: tr1 = tr0 + 1
        
        t_max, t_min = max(y0, y1), min(y0, y1)
        dt_ms = ch.dt_us / 1000.0
        # Use the same time reference _build_chain_figure used
        t_base_ch = getattr(self, "_last_chain_t0", ch.delay_ms)
        s0 = max(0, min(int((t_min - t_base_ch) / dt_ms), data.shape[0] - 1))
        s1 = max(0, min(int((t_max - t_base_ch) / dt_ms), data.shape[0]))
        if s1 <= s0: s1 = s0 + 1
        
        avail_w_px = max(400, self.ctab_profile.winfo_width())
        avail_h_px = max(300, self.ctab_profile.winfo_height() - 60)
        is_full = (tr1 - tr0) >= len(dist) * 0.95 and (s1 - s0) >= data.shape[0] * 0.95
        
        if is_full:
            new_fig_px_w = int(10.0 * len(ch.profiles) * 100)
            new_fig_px_h = int(max(4.0, avail_h_px / 100.0) * 100)
        else:
            new_fig_px_w = avail_w_px
            orig_px_per_trace = (10.0 * len(ch.profiles) * 100) / len(dist)
            orig_px_per_sample = (max(4.0, avail_h_px / 100.0) * 100) / data.shape[0]
            aspect_ratio = orig_px_per_sample / orig_px_per_trace
            new_px_per_trace = new_fig_px_w / (tr1 - tr0)
            new_px_per_sample = new_px_per_trace * aspect_ratio
            new_fig_px_h = new_px_per_sample * (s1 - s0)
            
        cmap_base = CMAPS[self.chain_cmap_var.get()]
        cmap_name = cmap_base + "_r" if self.chain_inv_var.get() else cmap_base
        clip_pct  = int(self.chain_clip_var.get())

        rgba_segs = []
        start_tr = 0
        for prof in ch.profiles:
            end_tr = start_tr + prof.n_traces
            overlap_start = max(start_tr, tr0)
            overlap_end = min(end_tr, tr1)
            if overlap_start < overlap_end:
                seg = data[s0:s1, overlap_start:overlap_end]
                seg_clean = np.nan_to_num(seg, nan=0.0)
                valid = np.abs(seg_clean[seg_clean != 0.0])
                vm = float(np.percentile(valid, clip_pct)) if valid.size > 0 else 1.0
                vm = vm or 1.0
                rgba_segs.append(_colormapped_image_parallel(seg_clean, cmap_name, 0.0, vm))
            start_tr = end_tr

        if rgba_segs:
            new_rgba = np.concatenate(rgba_segs, axis=1)
            ax.images[0].set_data(new_rgba)
            ax.images[0].set_extent([dist[tr0], dist[tr1-1], t_max, t_min])
            
        fig.set_size_inches(new_fig_px_w / 100.0, new_fig_px_h / 100.0)
        
        # update inner frame
        inner = None
        for w in self._chain_scroll_cv.winfo_children():
            if isinstance(w, tk.Frame):
                inner = w; break
        if inner:
            inner.config(width=new_fig_px_w, height=int(new_fig_px_h))
            
        self._chain_canvas_obj.get_tk_widget().config(width=new_fig_px_w, height=int(new_fig_px_h))
        self._chain_scroll_cv.config(scrollregion=(0, 0, new_fig_px_w, int(new_fig_px_h)))
        self._chain_canvas_obj.draw_idle()

    def _create_chain_map_fig(self, ch: ProfileChain):
        fig = Figure(figsize=(9, 6), facecolor=C["panel"])
        ax  = fig.add_subplot(111)
        ax.set_facecolor(C["entry"])

        wd   = ch.water_depth
        norm = mcolors.Normalize(np.nanmin(wd), np.nanmax(wd))
        cmap = _get_colormap("viridis")

        from matplotlib.collections import LineCollection
        pts    = np.column_stack([ch.lons, ch.lats])
        segs   = np.stack([pts[:-1], pts[1:]], axis=1)
        wd_mid = (wd[:-1] + wd[1:]) / 2.0
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=2.0)
        lc.set_array(wd_mid)
        ax.add_collection(lc)
        ax.autoscale()

        _cts0 = ch.timestamps[0]  if ch.timestamps else "—"
        _cts1 = ch.timestamps[-1] if ch.timestamps else "—"
        ax.scatter(ch.lons[0],  ch.lats[0],  s=60, color=C["ok"],
                   zorder=5, label=f"Inicio ({_cts0})",
                   edgecolors="white", lw=0.5)
        ax.scatter(ch.lons[-1], ch.lats[-1], s=60, color=C["warn"],
                   zorder=5, label=f"Fin ({_cts1})",
                   edgecolors="white", lw=0.5)

        step = max(1, ch.n_traces // 10)
        for i in range(0, ch.n_traces, step):
            ax.annotate(str(i), (ch.lons[i], ch.lats[i]),
                        color=C["sub"], fontsize=6,
                        xytext=(3, 3), textcoords="offset points")

        cb = fig.colorbar(lc, ax=ax, pad=0.01, fraction=0.02)
        cb.set_label("Prof. agua (m)", color=C["sub"], fontsize=8)
        cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=7)

        ax.set_xlabel("Longitud (°)", color=C["text"], fontsize=9)
        ax.set_ylabel("Latitud (°)",  color=C["text"], fontsize=9)
        ax.set_title(f"Trayectoria Cadena  ·  {ch.total_km:.2f} km  ·  {ch.n_traces} trazas",
                     color=C["text"], fontsize=10)
        ax.tick_params(colors=C["text"], labelsize=8)
        ax.grid(True, color=C["accent"], alpha=0.3, lw=0.5)
        ax.legend(facecolor=C["panel"], edgecolor=C["accent"],
                  labelcolor=C["text"], fontsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor(C["accent"])
        fig.tight_layout(pad=1.2)
        return fig

    def _create_chain_spectrum_fig(self, ch: ProfileChain, data: np.ndarray):
        valid_traces = ~np.isnan(data).all(axis=0)
        if valid_traces.sum() == 0:
            return None

        fs      = 1e6 / ch.dt_us
        d_clean = np.nan_to_num(data[:, valid_traces], nan=0.0)

        sp = self._compute_spectrum(d_clean, fs)
        fig = self._draw_spectrum_figure(
            sp, fs,
            title         = f"Cadena  ·  {len(ch.profiles)} perfiles",
            n_traces      = int(valid_traces.sum()),
            dist_km       = ch.dist_km[valid_traces],
            boundaries_km = list(ch.boundaries_km)   # líneas de unión entre perfiles
        )
        return fig

    def _draw_chain_headers(self, ch: ProfileChain, token=None):
        # Comprobar token ANTES de construir las filas — con cadenas grandes
        # esto evita bloquear el hilo principal si el render fue cancelado.
        if token is not None and getattr(self, "_chain_render_id", None) != token:
            return

        parent = self.ctab_headers
        for w in parent.winfo_children():
            w.destroy()

        cols = ("Traza Global", "Archivo Origen", "Dist (km)", "Lon (°)", "Lat (°)",
                "Prof. agua (m)", "TWT fondo (ms)", "Tiempo UTC", "Amp. máx")
        tree = ttk.Treeview(parent, columns=cols, show="headings")
        vsb  = ttk.Scrollbar(parent, orient="vertical",   command=tree.yview)
        hsb  = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.config(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor="center")
        tree.column("Archivo Origen", width=180)

        twt  = (ch.water_depth / 1500.0) * 2000.0
        amps = np.max(np.abs(ch.data), axis=0)

        # Build origin name list in one pass
        prof_names = np.empty(ch.n_traces, dtype=object)
        idx = 0
        for p in ch.profiles:
            prof_names[idx:idx + p.n_traces] = p.name
            idx += p.n_traces

        rows = [
            (str(i+1), prof_names[i],
             f"{ch.dist_km[i]:.3f}", f"{ch.lons[i]:.6f}", f"{ch.lats[i]:.6f}",
             f"{ch.water_depth[i]:.1f}", f"{twt[i]:.1f}",
             ch.timestamps[i], f"{amps[i]:.5f}")
            for i in range(ch.n_traces)
        ]

        # Segunda comprobación de token tras construir las filas
        if token is not None and getattr(self, "_chain_render_id", None) != token:
            return

        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)

        _chts0 = ch.timestamps[0]  if ch.timestamps else "—"
        _chts1 = ch.timestamps[-1] if ch.timestamps else "—"
        tk.Label(parent,
                 text=(f"  {ch.n_traces} trazas  ·  {len(ch.profiles)} perfiles  ·  "
                       f"{ch.total_km:.2f} km  ·  "
                       f"Prof. {ch.water_depth.min():.0f}–{ch.water_depth.max():.0f} m  ·  "
                       f"{_chts0}  →  {_chts1}"),
                 bg=C["panel"], fg=C["sub"],
                 font=("Courier New", 7)).pack(fill="x")

        # Insertar en chunks para no congelar la UI con cadenas de muchas trazas
        CHUNK = 500
        _tok = token
        def _insert_chunk_ch(start=0):
            if _tok is not None and getattr(self, "_chain_render_id", None) != _tok:
                return  # render cancelado — abortar inserción
            tree.config(selectmode="none")
            for row in rows[start:start + CHUNK]:
                tree.insert("", "end", values=row)
            if start + CHUNK < len(rows):
                self.after(0, lambda: _insert_chunk_ch(start + CHUNK))
        _insert_chunk_ch()

    # ════════════════════════════════════════════════════════════════════════
    # EXPORTACIÓN DE CADENA
    # ════════════════════════════════════════════════════════════════════════
    def _export_chain(self):
        ch = self.active_chain
        if ch is None:
            messagebox.showwarning("Sin cadena", "Selecciona una cadena primero.")
            return

        idx = self.chain_nb.index(self.chain_nb.select())
        names = ["perfil", "mapa", "espectro", "cabeceras"]
        suf = names[idx] if idx < len(names) else "export"

        stem = f"cadena_{len(ch.profiles)}p_{ch.total_km:.1f}km_{suf}"
        out  = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{stem}.png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"),
                       ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not out:
            return

        try:
            dpi_val = int(self.chain_dpi_var.get())
        except ValueError:
            dpi_val = 300

        # Capturar chain params antes del thread
        chain_export_params = dict(
            decon     = self.chain_decon_var.get(),    decon_op  = self.chain_decon_op.get(),
            decon_gap = self.chain_decon_gap.get(),    decon_wn  = self.chain_decon_wn.get(),
            filt      = self.chain_filt_var.get(),     flo       = self.chain_flo.get(),
            fhi       = self.chain_fhi.get(),          preset    = self.chain_preset_var.get(),
            tvg       = self.chain_tvg_var.get(),      tvg_alpha = self.chain_tvg_alpha.get(),
            agc       = self.chain_agc_var.get(),      agc_win   = self.chain_agc_win.get(),
            align     = self.chain_align_delays_var.get(),
            clip      = self.chain_clip_var.get(),     cmap      = self.chain_cmap_var.get(),
            inv_cmap  = self.chain_inv_var.get(),      fix       = self.chain_fix_var.get(),
            fix_iv    = self.chain_fix_interval_var.get(),
        )

        self._progress_start(f"Exportando {suf} de cadena a {dpi_val} DPI…")
        self.update_idletasks()

        if idx == 0:
            def save_task():
                import gc
                import tempfile
                import os
                from matplotlib.backends.backend_agg import FigureCanvasAgg as _Agg
                fig = None
                try:
                    # ── Fase 1: Procesar datos a resolución NATIVA ──────────
                    self.after(0, lambda: self._progress_set(5))
                    data = self._process_chain_data(ch, chain_export_params)
                    self._current_chain_data = data
                    i0, i1, t0, t1 = self._chain_time_window(ch, data.shape[0], chain_export_params)
                    d = data[i0:i1, :]
                    data_ns, data_tr = d.shape

                    cmap_base = CMAPS[chain_export_params["cmap"]]
                    cmap_name = cmap_base + "_r" if chain_export_params["inv_cmap"] else cmap_base
                    clip_pct  = int(chain_export_params["clip"])

                    if hasattr(self, "_chain_fig") and self._chain_fig is not None:
                        orig_w, orig_h = self._chain_fig.get_size_inches()
                    else:
                        orig_w, orig_h = 10.0 * max(1, len(ch.profiles)), 6.0

                    fig_w, fig_h = orig_w, orig_h
                    self.after(0, lambda: self._progress_set(10))

                    # ── Fase 2: Búfer en DISCO (np.memmap) ──────────────────
                    # En lugar de usar PIL y RAM, creamos un archivo temporal en el disco duro.
                    # Mantenemos la resolución original de las trazas (data_tr), NO la multiplicamos por DPI.
                    tmp_file = os.path.join(tempfile.gettempdir(), f'topas_export_buf_{id(self)}.dat')
                    
                    # shape: (alto_muestras, ancho_trazas, RGBA)
                    disk_buffer = np.memmap(tmp_file, dtype=np.uint8, mode='w+', shape=(data_ns, data_tr, 4))

                    vmax_list = []
                    n_profs   = len(ch.profiles)
                    start_tr  = 0

                    for pi, prof in enumerate(ch.profiles):
                        end_tr   = start_tr + prof.n_traces
                        seg      = d[:, start_tr:end_tr]
                        seg_flat = np.nan_to_num(seg, nan=0.0)
                        
                        valid_s  = np.abs(seg_flat[seg_flat != 0.0])
                        vm       = float(np.percentile(valid_s, clip_pct)) if valid_s.size > 0 else 1.0
                        vm       = vm or 1.0
                        vmax_list.append(vm)

                        # Colorizamos el bloque a su resolución original
                        lo = 0.0; hi = vm if vm != 0.0 else 1e-9
                        seg_norm = np.clip((seg_flat.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
                        cmap_fn = _get_colormap(cmap_name)
                        tile_rgba = (cmap_fn(seg_norm) * 255).astype(np.uint8)

                        # Volcamos directamente al disco duro
                        disk_buffer[:, start_tr:end_tr, :] = tile_rgba
                        
                        # Limpieza intensiva de RAM por cada bloque
                        del seg, seg_flat, seg_norm, tile_rgba
                        gc.collect()

                        start_tr = end_tr
                        pct = 10 + int(50 * (pi + 1) / n_profs)
                        self.after(0, lambda p=pct: self._progress_set(p))

                    # Forzar la escritura final en disco
                    disk_buffer.flush()
                    vmax = float(np.median(vmax_list)) if vmax_list else 1.0

                    # ── Fase 3: Construir figura Matplotlib ─────────────────
                    self.after(0, lambda: self._progress_set(62))
                    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi_val, facecolor=C["panel"])
                    _Agg(fig)
                    ax = fig.add_subplot(111)
                    ax.set_facecolor(C["panel"])

                    # Pasamos el memmap a imshow. Matplotlib es eficiente leyendo esto por bloques.
                    # Interpolation "none" o "nearest" evita que Matplotlib intente suavizar arrays gigantes.
                    ax.imshow(disk_buffer, aspect="auto", interpolation="nearest",
                              extent=[ch.dist_km[0], ch.dist_km[-1], t1, t0])

                    for b_km in ch.boundaries_km:
                        ax.axvline(b_km, color=C["warn"], lw=0.8, ls="--", alpha=0.7)
                        ax.text(b_km + 0.05, t0 + 4, f"↕ {b_km:.1f} km",
                                color=C["warn"], fontsize=6, fontfamily="Courier New", va="top")

                    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=mcolors.Normalize(0.0, vmax))
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, pad=0.005, fraction=0.008)
                    cb.ax.yaxis.set_tick_params(color=C["sub"], labelsize=6)
                    cb.set_label("Amplitud", color=C["sub"], fontsize=7)

                    ax.set_title(f"Cadena  ·  {len(ch.profiles)} perfil(es)  ·  {ch.total_km:.2f} km",
                                 color=C["text"], fontsize=9, pad=6)
                    ax.set_xlabel("Distancia acumulada (km)", color=C["text"], fontsize=9)
                    ax.set_ylabel("Tiempo (ms)", color=C["text"], fontsize=9)
                    ax.tick_params(colors=C["text"], labelsize=8)
                    for sp_ in ax.spines.values():
                        sp_.set_edgecolor(C["accent"])

                    # ── Fase 4: Guardar y liberar disco ─────────────────────
                    self.after(0, lambda: self._progress_set(75))
                    fig.tight_layout(pad=1.2)
                    self.after(0, lambda: self._progress_set(82))
                    
                    # savefig se encargará de escalar el vector y los textos al DPI establecido, 
                    # sin explotar la RAM leyendo el raster gracias al memmap.
                    fig.savefig(out, dpi=dpi_val, bbox_inches="tight", facecolor=C["panel"])
                    
                    # Cerrar y eliminar el archivo temporal
                    fig.clear()
                    plt.close(fig)
                    fig = None
                    del disk_buffer
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                    gc.collect()

                    # ── Fase 5: Notificar ───────────────────────────────────
                    self.after(0, lambda: self._progress_set(100))
                    self.after(0, lambda: self._progress_stop(f"✔ Exportado a {dpi_val} DPI: {Path(out).name}"))
                    self.after(0, lambda: messagebox.showinfo(
                        "Exportación completada",
                        f"Cadena guardada en:\n{out}\n\nOptimizado con búfer en disco."))
                
                except Exception as e:
                    err = str(e)
                    if fig is not None:
                        try: fig.clear(); plt.close(fig)
                        except Exception: pass
                    try:
                        # Asegurarse de borrar el archivo temporal si hay error
                        if 'tmp_file' in locals() and os.path.exists(tmp_file):
                            del disk_buffer
                            os.remove(tmp_file)
                    except Exception: pass
                    
                    self.after(0, lambda: self._progress_stop("✘ Error al exportar cadena."))
                    self.after(0, lambda m=err: messagebox.showerror("Error al exportar", m))
            threading.Thread(target=save_task, daemon=True).start()
        else:
            tabs = [self.ctab_profile, self.ctab_map, self.ctab_spectrum, self.ctab_headers]
            target_fig = None
            for w in tabs[idx].winfo_children():
                if hasattr(w, "figure_ref"):
                    target_fig = w.figure_ref; break
                elif hasattr(w, "figure"):
                    target_fig = w.figure; break

            if target_fig is None:
                messagebox.showinfo("Vista vacía", "Renderiza la vista antes de exportar.")
                return

            self.after(0, lambda: self._progress_start(f"Exportando {suf} de cadena a {dpi_val} DPI…"))

            def save_task(fig=target_fig):
                try:
                    # Recrear al DPI de exportación para re-renderizar vectores
                    import pickle as _pkl, io as _io
                    buf = _io.BytesIO()
                    _pkl.dump(fig, buf)
                    buf.seek(0)
                    export_fig = _pkl.load(buf)
                    export_fig.set_dpi(dpi_val)

                    export_fig.savefig(out, dpi=dpi_val, bbox_inches="tight",
                                       facecolor=C["panel"])
                    export_fig.clear()

                    try:
                        from PIL import Image as _PILImg
                        with _PILImg.open(out) as _im:
                            real_w, real_h = _im.size
                        res_str = f"{real_w}×{real_h} px  ·  {dpi_val} DPI"
                    except Exception:
                        res_str = f"{dpi_val} DPI"

                    self.after(0, lambda: self._progress_stop(
                        f"✔ Exportado ({res_str}): {Path(out).name}"))
                    self.after(0, lambda: messagebox.showinfo(
                        "Exportación completada",
                        f"Imagen guardada en:\n{out}\nResolución: {res_str}"))
                except Exception as e:
                    err = str(e)
                    try: export_fig.clear(); plt.close(export_fig)
                    except Exception: pass
                    self.after(0, lambda: self._progress_stop("✘ Error al exportar."))
                    self.after(0, lambda m=err: messagebox.showerror("Error al exportar", m))
            threading.Thread(target=save_task, daemon=True).start()

    # ════════════════════════════════════════════════════════════════════════
    # REPROYECCIÓN
    # ════════════════════════════════════════════════════════════════════════
    def _on_rep_sel_change(self):
        self._update_rep_sel_label()
        self._refresh_nav_sources()

    def _refresh_nav_sources(self):
        """Actualiza el combo de fuentes de navegación con perfiles y cadenas actuales."""
        if not hasattr(self, "_nav_source_menu"):
            return
        items = []
        # Perfiles individuales
        for sd in self.profiles.values():
            if not sd.error:
                items.append(f"Perfil: {sd.name}")          # debe coincidir con _PFX_PERFIL
        # Cadenas
        for ch in self.chains:
            if len(ch.profiles) > 1:
                items.append(f"Cadena: {ch.label}")           # debe coincidir con _PFX_CADENA
            else:
                items.append(f"Perfil (cadena 1): {ch.profiles[0].name}")  # _PFX_CADENA1
        if not items:
            items = ["(sin perfiles cargados)"]
        self._nav_source_menu["values"] = items
        # Seleccionar el activo si está disponible
        if self.active_profile and not self.active_profile.error:
            target = f"Perfil: {self.active_profile.name}"
            if target in items:
                self.nav_source_var.set(target)
                return
        if items:
            self.nav_source_var.set(items[0])

    def _update_rep_sel_label(self):
        mode = self.rep_mode_var.get()
        if mode == 0:
            n = len(self.profiles)
            self.lbl_rep_sel.config(
                text=f"  → {n} perfil(es) en cola" if n else "  → (ningún perfil cargado)")
        elif mode == 1:
            name = self.active_profile.name if self.active_profile else "(ninguno)"
            self.lbl_rep_sel.config(text=f"  → {name}")
        elif mode == 2:
            n = len(self.chains)
            n_multi = sum(1 for c in self.chains if len(c.profiles) > 1)
            self.lbl_rep_sel.config(
                text=f"  → {n} cadena(s) en cola ({n_multi} a unir)" if n else "  → (ninguna cadena detectada)")

    def _on_src_preset(self):
        v = PRESETS_CRS.get(self.src_preset.get())
        if v and v != "CUSTOM":
            self.src_epsg.set(v.replace("EPSG:", ""))
            self._verify_src()

    def _on_dst_preset(self):
        v = PRESETS_CRS.get(self.dst_preset.get())
        if v and v != "CUSTOM":
            self.dst_epsg.set(v.replace("EPSG:", ""))
            self._verify_dst()

    def _resolve_crs(self, raw: str) -> str:
        raw = raw.strip()
        return f"EPSG:{raw}" if raw.isdigit() else raw

    def _verify_src(self):
        self._verify_crs(self.src_epsg.get(), self.lbl_src_ok)

    def _verify_dst(self):
        self._verify_crs(self.dst_epsg.get(), self.lbl_dst_ok)

    def _verify_crs(self, code, lbl):
        try:
            crs = CRS.from_user_input(self._resolve_crs(code))
            lbl.config(text=f"✔  {crs.name[:55]}", fg=C["ok"])
        except Exception as e:
            lbl.config(text=f"✘  {str(e)[:60]}", fg=C["warn"])

    def _run_reprojection(self):
        mode = self.rep_mode_var.get()
        if mode == 0:
            targets = list(self.profiles.values())
        elif mode == 1:
            targets = [self.active_profile] if self.active_profile else []
        elif mode == 2:
            targets = self.chains

        if not targets:
            messagebox.showwarning("Sin selección", "No hay elementos seleccionados para reproyectar.")
            return

        src_str = self._resolve_crs(self.src_epsg.get())
        dst_str = self._resolve_crs(self.dst_epsg.get())

        try: CRS.from_user_input(src_str)
        except Exception as e:
            messagebox.showerror("CRS origen inválido", str(e)); return
        try: CRS.from_user_input(dst_str)
        except Exception as e:
            messagebox.showerror("CRS destino inválido", str(e)); return

        unit_hint = int(self.unit_hint.get()[0])
        self.rep_progress["value"] = 0
        self.after(0, lambda: self._progress_start(
            f"Reproyectando {len(targets)} elemento(s)…"))

        def worker():
            ok = 0
            for i, item in enumerate(targets, 1):
                if mode == 2:
                    name = item.label
                else:
                    if item.error:
                        self._rlog(f"⚠ Saltando {item.name} (error al cargar).")
                        continue
                    name = item.name
                self.after(0, lambda n=name, i=i: self._progress_update(
                    f"Reproyectando {i}/{len(targets)}  ·  {n}…"))
                self.after(0, lambda p_c=(i/len(targets)*100): self.rep_progress.configure(value=p_c))
                self.after(0, lambda p_c=(i/len(targets)*100): self._progress_set(p_c))
                if mode == 2:
                    self._reproject_chain(item, src_str, dst_str, unit_hint)
                else:
                    self._reproject_one(item, src_str, dst_str, unit_hint)
                ok += 1
            self.after(0, lambda: self.rep_progress.configure(value=100))
            self.after(0, lambda: self._progress_stop(
                f"✔ Reproyección completada: {ok}/{len(targets)} procesados."))
            self.after(0, lambda: messagebox.showinfo(
                "Completado",
                f"✔ {ok}/{len(targets)} elemento(s) reproyectados correctamente."))

        threading.Thread(target=worker, daemon=True).start()

    def _reproject_one(self, sd: SegyProfile, src_str, dst_str, unit_hint):
        self._rlog(f"\n{'─'*55}")
        self._rlog(f"Procesando : {sd.name}")
        self._rlog(f"CRS origen : {src_str}")
        self._rlog(f"CRS destino: {dst_str}")
        try:
            src_crs = CRS.from_user_input(src_str)
            dst_crs = CRS.from_user_input(dst_str)
            dst_geo = dst_crs.is_geographic
            out_sc  = -10_000_000 if dst_geo else -100
            new_uc  = 3 if dst_geo else 1
            tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

            p       = Path(sd.path)
            outpath = str(p.with_name(p.stem + "_REPROY" + p.suffix))
            self._rlog(f"Salida     : {Path(outpath).name}")

            div = (1.0 / abs(out_sc)) if out_sc < 0 else float(out_sc)

            INT32_MAX = 2_147_483_647
            def _safe_coord(v):
                return int(np.clip(round(v / div), -INT32_MAX, INT32_MAX))

            with segyio.open(sd.path, ignore_geometry=True) as src:
                spec = segyio.tools.metadata(src)
                with segyio.create(outpath, spec) as dst:
                    dst.bin = src.bin
                    # Fix Crítico: Copiar EBCDIC
                    dst.text[0] = src.text[0]
                    for i in range(sd.n_traces):
                        if i % 200 == 0:
                            self._rlog(f"  traza {i+1}/{sd.n_traces}…")
                        h  = src.header[i]
                        sc = int(h[segyio.TraceField.SourceGroupScalar])
                        uc = int(h[segyio.TraceField.CoordinateUnits]) or unit_hint
                        fac = (1.0/abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)
                        sx  = int(h[segyio.TraceField.SourceX]) * fac
                        sy  = int(h[segyio.TraceField.SourceY]) * fac
                        if uc == 2:
                            sx /= 3600.0; sy /= 3600.0
                        nx, ny = tf.transform(sx, sy)

                        dst.header[i] = h
                        dst.header[i].update({
                            segyio.TraceField.SourceX:           _safe_coord(nx),
                            segyio.TraceField.SourceY:           _safe_coord(ny),
                            segyio.TraceField.GroupX:            _safe_coord(nx),
                            segyio.TraceField.GroupY:            _safe_coord(ny),
                            segyio.TraceField.SourceGroupScalar: out_sc,
                            segyio.TraceField.CoordinateUnits:   new_uc,
                        })
                        dst.trace[i] = src.trace[i]

            self._rlog(f"✔ Guardado: {Path(outpath).name}")
        except Exception as e:
            self._rlog(f"✘ Error: {e}")
            import traceback; self._rlog(traceback.format_exc())

    def _reproject_chain(self, ch: ProfileChain, src_str, dst_str, unit_hint):
        self._rlog(f"\n{'─'*55}")
        self._rlog(f"Procesando CADENA: {ch.label}")
        self._rlog(f"CRS origen : {src_str}")
        self._rlog(f"CRS destino: {dst_str}")
        try:
            src_crs = CRS.from_user_input(src_str)
            dst_crs = CRS.from_user_input(dst_str)
            dst_geo = dst_crs.is_geographic
            out_sc  = -10_000_000 if dst_geo else -100
            new_uc  = 3 if dst_geo else 1
            tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

            # Usar la ruta del primer perfil como base para el nombre del archivo final
            p0 = Path(ch.profiles[0].path)
            stem = f"{p0.stem}_a_{Path(ch.profiles[-1].path).stem}_UNIDO_REPROY"
            outpath = str(p0.with_name(stem + p0.suffix))
            self._rlog(f"Salida     : {Path(outpath).name}")

            div = (1.0 / abs(out_sc)) if out_sc < 0 else float(out_sc)

            INT32_MAX = 2_147_483_647
            def _safe_coord(v):
                return int(np.clip(round(v / div), -INT32_MAX, INT32_MAX))

            # Obtener spec del primer perfil y actualizar el tracecount total
            with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
                spec = segyio.tools.metadata(src0)
                spec.tracecount = ch.n_traces

            with segyio.create(outpath, spec) as dst:
                # Copiar cabeceras globales (binaria y texto) del primero
                with segyio.open(ch.profiles[0].path, ignore_geometry=True) as src0:
                    dst.bin = src0.bin
                    dst.text[0] = src0.text[0]

                global_trace_idx = 0
                for p_idx, sd in enumerate(ch.profiles):
                    self._rlog(f"  Integrando perfil {p_idx+1}/{len(ch.profiles)}: {sd.name}...")
                    with segyio.open(sd.path, ignore_geometry=True) as src:
                        for i in range(sd.n_traces):
                            if global_trace_idx % 500 == 0:
                                self._rlog(f"    traza global {global_trace_idx+1}/{ch.n_traces}…")
                            
                            h = src.header[i]
                            sc = int(h[segyio.TraceField.SourceGroupScalar])
                            uc = int(h[segyio.TraceField.CoordinateUnits]) or unit_hint
                            fac = (1.0/abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)
                            sx  = int(h[segyio.TraceField.SourceX]) * fac
                            sy  = int(h[segyio.TraceField.SourceY]) * fac
                            
                            if uc == 2:
                                sx /= 3600.0; sy /= 3600.0
                            nx, ny = tf.transform(sx, sy)

                            dst.header[global_trace_idx] = h
                            dst.header[global_trace_idx].update({
                                segyio.TraceField.SourceX:           _safe_coord(nx),
                                segyio.TraceField.SourceY:           _safe_coord(ny),
                                segyio.TraceField.GroupX:            _safe_coord(nx),
                                segyio.TraceField.GroupY:            _safe_coord(ny),
                                segyio.TraceField.SourceGroupScalar: out_sc,
                                segyio.TraceField.CoordinateUnits:   new_uc,
                                segyio.TraceField.TraceNumber:       global_trace_idx + 1
                            })
                            dst.trace[global_trace_idx] = src.trace[i]
                            global_trace_idx += 1

            self._rlog(f"✔ Guardado: {Path(outpath).name}")
        except Exception as e:
            self._rlog(f"✘ Error: {e}")
            import traceback; self._rlog(traceback.format_exc())

    def _rlog(self, msg):
        """Thread-safe log append with batched UI updates.
        _rlog_buf y _rlog_pending se inicializan en __init__ para evitar
        race condition cuando múltiples hilos llaman a _rlog simultáneamente."""
        self._rlog_buf.append(msg)

        if not self._rlog_pending:
            self._rlog_pending = True
            self.after(80, self._rlog_flush)

    def _rlog_flush(self):
        if not self._rlog_buf:
            self._rlog_pending = False
            return
        lines, self._rlog_buf = self._rlog_buf, []
        self._rlog_pending = False
        text = "\n".join(lines) + "\n"
        self.rep_log.config(state="normal")
        self.rep_log.insert(tk.END, text)
        self.rep_log.see(tk.END)
        self.rep_log.config(state="disabled")

    # ════════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ════════════════════════════════════════════════════════════════════════
    def _btn(self, parent, text, cmd, color, side=None, **pack_kw):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=color, fg="white", relief="flat",
                      font=("Courier New", 8, "bold"),
                      padx=10, pady=4, cursor="hand2",
                      activebackground=color)
        if side:
            b.pack(side=side, **pack_kw)
        else:
            b.pack(**pack_kw)
        return b

    def _placeholder(self, parent, text):
        tk.Label(parent, text=f"\n\n\n{text}",
                 bg=C["bg"], fg=C["sub"],
                 font=("Courier New", 10)).pack(expand=True)

    # ── Spinner animado ───────────────────────────────────────────────────
    def _spinner_tick(self):
        """Avanza un frame del spinner cada 120 ms mientras haya tareas activas."""
        if self._active_tasks <= 0:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        self._spinner_lbl.config(text=self._spinner_chars[self._spinner_idx])
        self._spinner_job = self.after(120, self._spinner_tick)

    # ── Progreso central ─────────────────────────────────────────────────

    def _progress_set(self, pct: float):
        """Actualiza el porcentaje de la barra de progreso (0-100)."""
        self._progress_bar["value"] = max(0.0, min(100.0, float(pct)))
        self._progress_bar.update_idletasks()

    def _progress_start(self, msg: str = ""):
        """Muestra la barra de progreso y el spinner.
        Thread-safe: llamar siempre vía after(0, ...)."""
        self._active_tasks = max(0, self._active_tasks) + 1
        if msg:
            self.statusbar.config(text=msg, fg=C["bright"])
        # Mostrar la barra de progreso si no está visible (pack dentro del bottom_bar)
        if not self._progress_bar.winfo_ismapped():
            self._progress_bar.pack(fill="x")
        self._progress_bar["value"] = 0
        self._progress_bar.update_idletasks()
        # Arrancar el spinner si no está corriendo
        if self._spinner_job is None:
            self._spinner_idx = 0
            self._spinner_lbl.config(text=self._spinner_chars[0])
            self._spinner_job = self.after(120, self._spinner_tick)

    def _progress_stop(self, msg: str = ""):
        """Oculta barra y spinner cuando no queda ninguna tarea activa."""
        self._active_tasks = max(0, self._active_tasks - 1)
        color = C["warn"] if (msg.startswith("✘")) else (
                C["ok"]   if (msg.startswith("✔")) else C["sub"])
        if msg:
            self.statusbar.config(text=msg, fg=color)
        if self._active_tasks == 0:
            # Lleva la barra al 100% un momento antes de ocultarla
            self._progress_bar["value"] = 100
            self._progress_bar.update_idletasks()
            self._progress_bar.pack_forget()
            # Detener spinner
            if self._spinner_job is not None:
                self.after_cancel(self._spinner_job)
                self._spinner_job = None
            self._spinner_lbl.config(text="")

    def _progress_update(self, msg: str):
        """Actualiza el texto de estado sin tocar el contador de tareas."""
        self.statusbar.config(text=msg, fg=C["bright"])

    def _status(self, msg: str):
        color = C["warn"] if msg.startswith("✘") else (
                C["ok"]   if msg.startswith("✔") else C["sub"])
        self.statusbar.config(text=msg, fg=color)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = TopasSuite()
    app.mainloop()