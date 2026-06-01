#!/usr/bin/env python3
"""
topas_core.py — Backend núcleo de TOPAS Suite (PHASE 1 extraction).

Capa de MODELO + lógica de negocio totalmente desacoplada de la GUI.

DISEÑO (mirror de paradigmas C++17):
  * Tipado estricto (type hints en toda firma pública).
  * Encapsulación estricta: métodos/funciones "privadas" prefijadas con "_".
    El API público NO lleva guión bajo y constituye la "interfaz" (header).
  * Sin estado global mutable salvo capacidades de hardware detectadas una
    sola vez al cargar (constexpr-like): _XP, _GPU, _N_WORKERS.

GARANTÍAS DE AISLAMIENTO (RULE 2 — ZERO GUI):
  * Este módulo NO importa tkinter, PyQt6 ni matplotlib a nivel de módulo.
  * La única dependencia de colormaps (tablas RGBA de matplotlib) se carga
    de forma diferida (lazy import) dentro del cuerpo de la función que la
    necesita, igual que se aislaría un subsistema pesado tras una fachada en
    C++17. Así el backend es 100% importable en un entorno headless.

PRESERVACIÓN DE LÓGICA (RULE 3 / RULE 4):
  * Matemática geofísica, aceleración CuPy y ThreadPoolExecutor intactos.
  * El metadato "delay recording time" (bytes 109-110 de la cabecera de traza,
    segyio.TraceField.DelayRecordingTime) se lee, propaga y usa para restaurar
    perfiles SIN ninguna alteración de la lógica de bytes original.
"""

from __future__ import annotations

import os
import concurrent.futures as _cf
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Any

import numpy as np
import segyio
from scipy import signal as sp_signal
import scipy.linalg
from pyproj import CRS, Transformer


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE DOMINIO (constantes — equivalentes a `static constexpr`)
# ═══════════════════════════════════════════════════════════════════════════════

# Paletas estrictamente secuenciales (nombre visible → nombre de colormap mpl).
CMAPS: Dict[str, str] = {
    "Blanco / Negro": "Greys",
    "Viridis":        "viridis",
    "Inferno":        "inferno",
    "Jet":            "jet",
    "Terrain":        "terrain",
}

COORD_UNITS: Dict[int, str] = {1: "m/ft", 2: "arc-sec", 3: "decimal°", 4: "DMS"}

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

# Nombre visible → clave interna del filtro/atributo sísmico.
FILTER_PRESETS: Dict[str, str] = {
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

# Tooltips descriptivos (texto de dominio — el consumidor de UI los muestra).
FILTER_DESCRIPTIONS: Dict[str, str] = {
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


# ═══════════════════════════════════════════════════════════════════════════════
# COLORMAP (lazy facade — RULE 2: matplotlib NO se importa a nivel de módulo)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_colormap(name: str):
    """
    Obtiene un colormap compatible con todas las versiones de matplotlib:
      - matplotlib >= 3.5 → matplotlib.colormaps[name]
      - matplotlib <  3.5 → matplotlib.cm.get_cmap(name)

    NOTA DE AISLAMIENTO: matplotlib se importa de forma DIFERIDA aquí dentro.
    Es una tabla de lookup de color puramente numérica (no abre ventana ni
    backend gráfico), de modo que el backend sigue siendo headless: nada de
    matplotlib se carga salvo que se solicite explícitamente colorización.
    """
    import matplotlib  # lazy import — backend permanece GUI-free hasta este punto
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return matplotlib.cm.get_cmap(name)


# ═══════════════════════════════════════════════════════════════════════════════
# ACELERACIÓN GPU (CuPy / CUDA) — opcional, fallback silencioso a NumPy/SciPy
# ═══════════════════════════════════════════════════════════════════════════════
def _detect_gpu() -> Tuple[Any, bool]:
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


_XP, _GPU = _detect_gpu()          # _XP = cupy o numpy; _GPU = bool

# ─── Motor de procesado paralelo ────────────────────────────────────────────
# Número de núcleos físicos disponibles (sin saturar el SO)
_N_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)


def gpu_available() -> bool:
    """API público: True si CuPy/CUDA está disponible para acelerar."""
    return _GPU


def worker_count() -> int:
    """API público: número de workers usados por el motor paralelo."""
    return _N_WORKERS


def _parallel_apply(fn: Callable[..., np.ndarray], data: np.ndarray, *args,
                    n_workers: int = _N_WORKERS, **kwargs) -> np.ndarray:
    """
    Divide 'data' (ns × n_traces) en n_workers bloques de columnas,
    aplica fn(bloque, *args, **kwargs) en paralelo con ThreadPoolExecutor
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

    results: List[Optional[np.ndarray]] = [None] * len(chunks)
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(fn, ch, *args, **kwargs): k
                for k, ch in enumerate(chunks)}
        for fut in _cf.as_completed(futs):
            k = futs[fut]
            results[k] = fut.result()

    return np.concatenate(results, axis=1)


def colormapped_image_parallel(data: np.ndarray, cmap_name: str,
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
    data_norm: Optional[np.ndarray]
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

    results: List[Optional[np.ndarray]] = [None] * len(tiles)
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


# ═══════════════════════════════════════════════════════════════════════════════
# MODELO DE DATOS — SegyProfile
# ═══════════════════════════════════════════════════════════════════════════════
class SegyProfile:
    """Carga y cachea un perfil SEG-Y completo."""

    def __init__(self, filepath: str) -> None:
        self.path: str  = filepath
        self.name: str  = Path(filepath).name
        self.stem: str  = Path(filepath).stem
        self.error: Optional[str] = None
        self._load()

    def _load(self) -> None:
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

                # ── DELAY RECORDING TIME (bytes 109-110) — CRÍTICO (RULE 4) ──────
                # Se lee tal cual de la cabecera de traza; NO se altera la lógica
                # de bytes. Estos delays se usan para restaurar/alinear perfiles
                # más adelante (process_profile_data → "delay compensation").
                delays_raw = _hdr[segyio.TraceField.DelayRecordingTime]
                self.delays    = delays_raw
                self.min_delay = float(np.min(delays_raw))
                self.max_delay = float(np.max(delays_raw))
                self.delay_ms  = int(delays_raw[0])

                sc   = self.scalar_coord
                fac  = (1.0 / abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)
                sxs  = _hdr[segyio.TraceField.SourceX].astype(float)  * fac
                sys_ = _hdr[segyio.TraceField.SourceY].astype(float)  * fac

                if self.coord_unit == 2:          # arc-seconds → degrees
                    self.lons = sxs  / 3600.0
                    self.lats = sys_ / 3600.0
                else:
                    self.lons = sxs
                    self.lats = sys_

                es   = self.scalar_elev
                efac = (1.0 / abs(es)) if es < 0 else (float(es) if es > 0 else 1.0)
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

    def _detect_crs(self) -> Tuple[Optional[str], List[str]]:
        if self.coord_unit == 2:
            if -180 <= self.lons[0] <= 180 and -90 <= self.lats[0] <= 90:
                return "EPSG:4326", ["✔ Arc-seconds → WGS84 detectado"]
        elif self.coord_unit == 3:
            if -180 <= self.lons[0] <= 180 and -90 <= self.lats[0] <= 90:
                return "EPSG:4326", ["✔ Grados decimales → WGS84 detectado"]
        return None, ["⚠ No detectado automáticamente"]

    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        return (f"{self.n_traces} trazas · {self.dur_ms:.0f} ms · "
                f"{self.total_km:.1f} km · "
                f"WD {np.nanmean(self.water_depth):.0f} m")


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESADO SÍSMICO — Deconvolución predictiva
# ═══════════════════════════════════════════════════════════════════════════════
def apply_predictive_decon(data: np.ndarray, dt_us: int, op_len_ms: float,
                           gap_ms: float, white_noise_pct: float) -> np.ndarray:
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
# MODELO DE DATOS — ProfileChain
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
    GAP_KM_MAX: float = 2.0

    def __init__(self, profiles: List[SegyProfile]) -> None:
        """profiles: lista ordenada de SegyProfile ya cargados sin error."""
        self.profiles = profiles
        self.name     = " ⛓ ".join(p.stem for p in profiles)
        if len(profiles) > 1:
            self.label = f"[{len(profiles)} perfiles]  {profiles[0].name} … {profiles[-1].name}"
        else:
            self.label = f"[1 perfil]  {profiles[0].name}"
        self._concat()

    def _concat(self) -> None:
        # Datos sísmicos concatenados por trazas
        self.data = np.concatenate([p.data for p in self.profiles], axis=1)

        # Delays — el metadato "delay recording time" se concatena intacto
        # y se conserva para la restauración/alineación posterior (RULE 4).
        self.delays = np.concatenate([p.delays for p in self.profiles])
        self.min_delay = float(np.min(self.delays))
        self.max_delay = float(np.max(self.delays))

        # Coordenadas
        self.lons = np.concatenate([p.lons for p in self.profiles])
        self.lats = np.concatenate([p.lats for p in self.profiles])

        # Water depth
        self.water_depth = np.concatenate([p.water_depth for p in self.profiles])

        # Timestamps
        self.timestamps: List[str] = []
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
        self.boundaries_km: List[float] = []
        d = 0.0
        for p in self.profiles[:-1]:
            d += p.total_km
            self.boundaries_km.append(d)

        # clip global p99
        self.clip_p99 = float(np.percentile(np.abs(self.data), 99))

    @staticmethod
    def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    @classmethod
    def detect(cls, profiles: List[SegyProfile],
               gap_km: Optional[float] = None) -> List["ProfileChain"]:
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
        def _ts_key(p: SegyProfile) -> str:
            return p.timestamps[0] if p.timestamps else ""
        valid_sorted = sorted(valid, key=_ts_key)

        # Agrupar greedy: añadir al grupo actual si contiguos, sino nuevo grupo
        groups: List[List[SegyProfile]] = []
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


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE DE PROCESADO (extraído de la GUI → funciones libres del backend)
#
# La GUI original leía los parámetros desde tk.Var; aquí se reciben en un dict
# `params` puro para mantener el backend 100% desacoplado de cualquier toolkit.
# La firma del dict (claves) replica EXACTAMENTE las usadas por la capa de UI:
#   decon, decon_op, decon_gap, decon_wn,
#   filt, flo, fhi,
#   preset,
#   tvg, tvg_alpha,
#   agc, agc_win,
#   align
# ═══════════════════════════════════════════════════════════════════════════════
def _process_data_generic(obj: Any, params: Dict[str, Any]) -> np.ndarray:
    """
    Implementación común del pipeline para SegyProfile y ProfileChain.
    `obj` debe exponer: .data, .dt_us, .delays, .min_delay, .ns, .n_traces.

    Etapas (orden preservado bit a bit respecto al monolito):
      1. Deconvolución predictiva
      2. Filtro pasabanda (paralelo multi-núcleo)
      3. Atributos / filtros preestablecidos
      4. Compensación TVG exponencial
      5. AGC (control automático de ganancia)
      6. Compensación de delays (usa "delay recording time" — RULE 4)
    """
    data = obj.data.copy()

    # 1. Deconvolución Predictiva
    if params.get("decon"):
        data = apply_predictive_decon(data, obj.dt_us,
                                      params["decon_op"],
                                      params["decon_gap"],
                                      params["decon_wn"])

    # 2. Filtro Pasabanda (paralelo multi-núcleo)
    if params.get("filt"):
        fs  = 1e6 / obj.dt_us
        flo = max(10, params["flo"])
        fhi = min(fs / 2 - 1, params["fhi"])
        if flo < fhi:
            sos  = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")
            def _sosfilt_block(block: np.ndarray, _sos=sos) -> np.ndarray:
                return sp_signal.sosfilt(_sos, block, axis=0).astype(np.float32)
            data = _parallel_apply(_sosfilt_block, data)

    # 3. Atributos / Filtros Preestablecidos
    preset_key = FILTER_PRESETS.get(params.get("preset", ""), "none")
    if preset_key != "none":
        data = apply_filter_preset(data, preset_key, obj.dt_us)

    # 4. Compensación TVG exponencial por atenuación
    if params.get("tvg"):
        alpha = params["tvg_alpha"]
        t_sec = np.arange(data.shape[0], dtype=np.float32) * (obj.dt_us / 1e6)
        gain_curve = np.clip(np.exp(alpha * t_sec), 0.0, 1e9)
        data *= gain_curve[:, np.newaxis]

    # 5. Control Automático de Ganancia (AGC)
    if params.get("agc"):
        win_s = max(3, int(params["agc_win"] / (obj.dt_us / 1000.0)))
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
                from scipy.ndimage import uniform_filter1d
                env = np.abs(data)
                rms = uniform_filter1d(env, size=win_s, axis=0)
                data = (data / np.maximum(rms, 1e-9)).astype(np.float32)
        else:
            from scipy.ndimage import uniform_filter1d
            env = np.abs(data)
            rms = uniform_filter1d(env, size=win_s, axis=0)
            data = (data / np.maximum(rms, 1e-9)).astype(np.float32)

    # 6. Compensación de delays — vectorizado (RULE 4)
    #    Usa el "delay recording time" (bytes 109-110) para realinear cada
    #    traza al delay mínimo del grupo y restaurar la geometría temporal.
    if params.get("align"):
        dt_ms = obj.dt_us / 1000.0
        offsets = np.round((obj.delays - obj.min_delay) / dt_ms).astype(int)
        extra_samples = int(offsets.max())
        new_ns = obj.ns + extra_samples

        aligned_data = np.full((new_ns, obj.n_traces), np.nan, dtype=np.float32)
        row_idx = (np.arange(obj.ns)[:, None] + offsets[None, :])   # (ns, n_traces)
        col_idx = np.arange(obj.n_traces)[None, :]                   # broadcast
        aligned_data[row_idx, col_idx] = data

        data = aligned_data

    return data


def process_profile_data(sd: SegyProfile, params: Dict[str, Any]) -> np.ndarray:
    """API público: aplica el pipeline completo a un SegyProfile."""
    return _process_data_generic(sd, params)


def process_chain_data(ch: ProfileChain, params: Dict[str, Any]) -> np.ndarray:
    """API público: aplica el pipeline completo a una ProfileChain."""
    return _process_data_generic(ch, params)


def time_window(obj: Any, data_ns: int, align: bool) -> Tuple[int, int, float, float]:
    """
    Calcula la ventana temporal (i0, i1, t0, t1) en ms del render.
    Si `align` es True usa min_delay como referencia (delays compensados);
    si no, usa delay_ms (delay del primer registro — bytes 109-110).
    Válido tanto para SegyProfile como para ProfileChain.
    """
    i0 = 0
    i1 = data_ns
    if align:
        t0 = obj.min_delay
        t1 = obj.min_delay + data_ns * obj.dt_us / 1000.0
    else:
        t0 = obj.delay_ms
        t1 = obj.delay_ms + data_ns * obj.dt_us / 1000.0
    return i0, i1, t0, t1


# ═══════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE MARCAS FIX (navegación temporal) — lógica pura, sin GUI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_timestamp(ts: str):
    """Convierte 'YYYY-DOYnnn HH:MM:SS' → datetime. None si no parseable."""
    from datetime import datetime, timedelta
    try:
        date_part, time_part = ts.split(" ")
        year = int(date_part.split("-")[0])
        doy  = int(date_part.split("DOY")[1])
        h, m, s = (int(x) for x in time_part.split(":"))
        return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=h, minutes=m, seconds=s)
    except Exception:
        return None


def compute_fix_positions(timestamps: List[str], dist_km: np.ndarray,
                          lons: np.ndarray, lats: np.ndarray,
                          interval_min: int) -> List[Tuple[int, float, str, float, float]]:
    """
    Calcula las posiciones FIX a intervalos regulares de tiempo.
    Devuelve lista de tuplas (fix_num, dist_km, "HH:MM", lon, lat).
    """
    from datetime import datetime, timezone

    # Parse all timestamps once into seconds-since-epoch array
    epoch = datetime(1970, 1, 1)
    t_sec = np.empty(len(timestamps), dtype=np.float64)
    for k, ts in enumerate(timestamps):
        dt = parse_timestamp(ts)
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

    fixes: List[Tuple[int, float, str, float, float]] = []
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


# ═══════════════════════════════════════════════════════════════════════════════
# ESCRITORES DE GEOMETRÍA — zero-dependency (stdlib struct/json/csv)
# ═══════════════════════════════════════════════════════════════════════════════
def write_shp_pure(path: str, points: list) -> None:
    """
    Escribe un Shapefile de puntos (POINT 2D) usando solo la stdlib.
    points: lista de (fix_num, fix_hora, lon, lat) — lon/lat float WGS84.
    Genera .shp / .shx / .dbf / .prj sin ninguna dependencia externa.
    """
    import struct

    records = points   # list of (fix_num, fix_hora, lon, lat)

    # ── .shp / .shx ──────────────────────────────────────────────────
    # File header: 100 bytes (big-endian fields + little-endian fields)
    def _shp_record(lon: float, lat: float) -> bytes:
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

    def _file_header(file_len: int) -> bytes:
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


def write_geojson(path: str, points: list) -> None:
    """Escribe un GeoJSON FeatureCollection de puntos WGS84.
    points: lista de (fix_num, _dist, fix_hora, lon, lat)."""
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


def write_csv(path: str, points: list) -> None:
    """Escribe un CSV con columnas fix_num, fix_hora, lon, lat.
    points: lista de (fix_num, _dist, fix_hora, lon, lat)."""
    import csv as _csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["fix_num", "fix_hora", "lon", "lat"])
        for num, _dist, hora, lon, lat in points:
            w.writerow([num, hora, round(lon, 8), round(lat, 8)])


# ═══════════════════════════════════════════════════════════════════════════════
# REPROYECCIÓN SEG-Y — escritura con preservación estricta de cabeceras
#
# RULE 4: el "delay recording time" (bytes 109-110) NUNCA se toca. La escritura
# copia la cabecera de traza completa (`dst.header[i] = h`) y SOLO sobrescribe
# los campos de coordenadas/escala/unidad. El DelayRecordingTime se preserva
# byte a byte al copiar la cabecera origen sin modificarlo.
#
# `log` es un callback opcional (Callable[[str], None]) para reportar progreso
# sin acoplar a ningún widget. Por defecto, no-op.
# ═══════════════════════════════════════════════════════════════════════════════
_INT32_MAX = 2_147_483_647


def _noop_log(_msg: str) -> None:
    """Sumidero por defecto del callback de log (sin GUI)."""
    pass


def reproject_one(sd: SegyProfile, src_str: str, dst_str: str,
                  unit_hint: int, log: Callable[[str], None] = _noop_log) -> Optional[str]:
    """
    Reproyecta un único perfil SEG-Y a un nuevo CRS.
    Devuelve la ruta de salida en éxito, o None si hubo error.

    Preserva la cabecera de traza completa (incluido DelayRecordingTime,
    bytes 109-110) y solo reescribe SourceX/Y, GroupX/Y, escala y unidad.
    """
    log(f"\n{'─' * 55}")
    log(f"Procesando : {sd.name}")
    log(f"CRS origen : {src_str}")
    log(f"CRS destino: {dst_str}")
    try:
        src_crs = CRS.from_user_input(src_str)
        dst_crs = CRS.from_user_input(dst_str)
        dst_geo = dst_crs.is_geographic
        out_sc  = -10_000_000 if dst_geo else -100
        new_uc  = 3 if dst_geo else 1
        tf = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        p       = Path(sd.path)
        outpath = str(p.with_name(p.stem + "_REPROY" + p.suffix))
        log(f"Salida     : {Path(outpath).name}")

        div = (1.0 / abs(out_sc)) if out_sc < 0 else float(out_sc)

        def _safe_coord(v: float) -> int:
            return int(np.clip(round(v / div), -_INT32_MAX, _INT32_MAX))

        with segyio.open(sd.path, ignore_geometry=True) as src:
            spec = segyio.tools.metadata(src)
            with segyio.create(outpath, spec) as dst:
                dst.bin = src.bin
                # Fix Crítico: Copiar EBCDIC
                dst.text[0] = src.text[0]
                for i in range(sd.n_traces):
                    if i % 200 == 0:
                        log(f"  traza {i+1}/{sd.n_traces}…")
                    h  = src.header[i]
                    sc = int(h[segyio.TraceField.SourceGroupScalar])
                    uc = int(h[segyio.TraceField.CoordinateUnits]) or unit_hint
                    fac = (1.0 / abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)
                    sx  = int(h[segyio.TraceField.SourceX]) * fac
                    sy  = int(h[segyio.TraceField.SourceY]) * fac
                    if uc == 2:
                        sx /= 3600.0; sy /= 3600.0
                    nx, ny = tf.transform(sx, sy)

                    # Copia íntegra de la cabecera (DelayRecordingTime intacto)
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

        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except Exception as e:
        log(f"✘ Error: {e}")
        import traceback
        log(traceback.format_exc())
        return None


def reproject_chain(ch: ProfileChain, src_str: str, dst_str: str,
                    unit_hint: int, log: Callable[[str], None] = _noop_log) -> Optional[str]:
    """
    Reproyecta y UNE una cadena de perfiles SEG-Y contiguos a un nuevo CRS,
    escribiendo un único archivo concatenado.
    Devuelve la ruta de salida en éxito, o None si hubo error.

    Cada cabecera de traza se copia íntegra (DelayRecordingTime, bytes 109-110,
    preservado), reescribiendo solo coordenadas, escala, unidad y TraceNumber.
    """
    log(f"\n{'─' * 55}")
    log(f"Procesando CADENA: {ch.label}")
    log(f"CRS origen : {src_str}")
    log(f"CRS destino: {dst_str}")
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
        log(f"Salida     : {Path(outpath).name}")

        div = (1.0 / abs(out_sc)) if out_sc < 0 else float(out_sc)

        def _safe_coord(v: float) -> int:
            return int(np.clip(round(v / div), -_INT32_MAX, _INT32_MAX))

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
                log(f"  Integrando perfil {p_idx+1}/{len(ch.profiles)}: {sd.name}...")
                with segyio.open(sd.path, ignore_geometry=True) as src:
                    for i in range(sd.n_traces):
                        if global_trace_idx % 500 == 0:
                            log(f"    traza global {global_trace_idx+1}/{ch.n_traces}…")

                        h = src.header[i]
                        sc = int(h[segyio.TraceField.SourceGroupScalar])
                        uc = int(h[segyio.TraceField.CoordinateUnits]) or unit_hint
                        fac = (1.0 / abs(sc)) if sc < 0 else (float(sc) if sc > 0 else 1.0)
                        sx  = int(h[segyio.TraceField.SourceX]) * fac
                        sy  = int(h[segyio.TraceField.SourceY]) * fac

                        if uc == 2:
                            sx /= 3600.0; sy /= 3600.0
                        nx, ny = tf.transform(sx, sy)

                        # Copia íntegra de cabecera (DelayRecordingTime intacto)
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

        log(f"✔ Guardado: {Path(outpath).name}")
        return outpath
    except Exception as e:
        log(f"✘ Error: {e}")
        import traceback
        log(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# API PÚBLICO EXPORTADO (la "interfaz" del header del módulo)
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Configuración de dominio
    "CMAPS", "COORD_UNITS", "PRESETS_CRS", "FILTER_PRESETS", "FILTER_DESCRIPTIONS",
    # Capacidades de hardware
    "gpu_available", "worker_count",
    # Modelo de datos
    "SegyProfile", "ProfileChain",
    # Procesado sísmico
    "apply_predictive_decon", "apply_filter_preset",
    "process_profile_data", "process_chain_data", "time_window",
    # Colorización
    "colormapped_image_parallel",
    # Navegación / FIX
    "parse_timestamp", "compute_fix_positions",
    # IO geometría
    "write_shp_pure", "write_geojson", "write_csv",
    # Reproyección
    "reproject_one", "reproject_chain",
]
