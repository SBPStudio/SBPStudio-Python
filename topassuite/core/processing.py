"""
processing.py — Seismic processing pipeline.

Two-track design
----------------
Every public function dispatches to the optimized path when one exists and
has been regression-tested. If no optimized path exists, the reference
implementation (_ref_*) is used directly.

Current status
--------------
apply_predictive_decon : reference only (no optimization implemented)
apply_filter_preset    : reference only (GPU branches preserved from monolith)
process_data           : reference only
time_window            : reference only (trivial; no optimization needed)

Array contract
--------------
- Input data : (ns, n_traces) float32
- All public functions return NEW arrays; inputs are never mutated.
- apply_filter_preset("none") returns data.copy() (new array, not aliased).

Known limitations
-----------------
- delay_ms uses trace[0] only (passed in via obj.delay_ms).
- Projected-CRS distance assumed in metres (see model.py).
"""
from __future__ import annotations

import concurrent.futures as _cf
from typing import Any, Dict

import numpy as np
import scipy.linalg
from scipy import signal as sp_signal

from ._backends import XP as _XP, GPU as _GPU, N_WORKERS as _N_WORKERS


# ── Parallel helper ────────────────────────────────────────────────────────────

def _parallel_apply(fn, data: np.ndarray, *args,
                    n_workers: int = _N_WORKERS, **kwargs) -> np.ndarray:
    """
    Split data (ns × n_traces) into n_workers column-blocks, apply fn to
    each in a ThreadPoolExecutor, then concatenate results.

    Heuristic: single-threaded when n_traces < 64 or array < 4 MB.
    """
    ns, n_traces = data.shape
    if n_traces < 64 or data.nbytes < 4 * 1024 * 1024 or n_workers <= 1:
        return fn(data, *args, **kwargs)

    chunk_size = max(1, n_traces // n_workers)
    slices     = [slice(j, min(j + chunk_size, n_traces))
                  for j in range(0, n_traces, chunk_size)]
    chunks     = [data[:, sl] for sl in slices]
    results    = [None] * len(chunks)

    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(fn, ch, *args, **kwargs): k
                for k, ch in enumerate(chunks)}
        for fut in _cf.as_completed(futs):
            results[futs[fut]] = fut.result()

    return np.concatenate(results, axis=1)


# ── Reference deconvolution ────────────────────────────────────────────────────

def _ref_apply_predictive_decon(
    data: np.ndarray,
    dt_us: int,
    op_len_ms: float,
    gap_ms: float,
    white_noise_pct: float,
) -> np.ndarray:
    """
    Reference predictive deconvolution (Wiener-Levinson), trace by trace.
    Path: reference. Backend: NumPy FFT + scipy.linalg.solve_toeplitz.
    """
    ns, _nt = data.shape
    dt_ms   = dt_us / 1000.0
    nl      = max(2, int(op_len_ms / dt_ms))
    gap     = max(1, int(gap_ms / dt_ms))
    mu      = white_noise_pct / 100.0
    max_lag = nl + gap
    n_fft   = 2 ** int(np.ceil(np.log2(2 * ns - 1)))

    def _decon_block(block: np.ndarray) -> np.ndarray:
        _ns, _nt_b = block.shape
        out = np.zeros_like(block)
        for i in range(_nt_b):
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
            f       = np.zeros(max_lag)
            f[0]    = 1.0
            f[gap:] = -a
            out[:, i] = sp_signal.lfilter(f, [1.0], tr)
        return out

    return _parallel_apply(_decon_block, data)


def apply_predictive_decon(
    data: np.ndarray,
    dt_us: int,
    op_len_ms: float,
    gap_ms: float,
    white_noise_pct: float,
) -> np.ndarray:
    """
    Predictive deconvolution (Wiener-Levinson), trace by trace.

    Parameters
    ----------
    data           : (ns, n_traces) float32 — input NOT mutated
    dt_us          : sample interval in microseconds
    op_len_ms      : operator length in ms
    gap_ms         : prediction gap in ms
    white_noise_pct: regularisation (% of zero-lag autocorrelation)

    Returns
    -------
    (ns, n_traces) float32 — new array

    Path: reference (_ref_apply_predictive_decon).
    No optimized path implemented; regression gate not applicable.
    """
    return _ref_apply_predictive_decon(data, dt_us, op_len_ms, gap_ms, white_noise_pct)


# ── Hilbert parallelisation helper ────────────────────────────────────────────

def _hilbert_parallel(out: np.ndarray, key: str, dt_us: int, fs: float) -> np.ndarray:
    """
    Compute Hilbert-based attributes (envelope / inst_phase / cos_phase /
    inst_freq) using _parallel_apply so that all CPU cores are saturated.

    Each worker sets scipy.fft workers=1 to prevent thread over-subscription:
    N_WORKERS parallel blocks × 1 FFT thread each = N_WORKERS threads total.

    Path: reference (same math as the CPU scalar path).
    """
    from scipy.signal import hilbert as _hilbert

    def _block(blk: np.ndarray, _key=key, _dt_us=dt_us, _fs=fs) -> np.ndarray:
        try:
            from scipy.fft import set_workers as _sw
            ctx = _sw(1)
            ctx.__enter__()
        except Exception:
            ctx = None
        try:
            an = _hilbert(blk, axis=0)
            if _key == "envelope":
                return np.abs(an).astype(np.float32)
            if _key == "inst_phase":
                return np.angle(an).astype(np.float32)
            if _key == "cos_phase":
                return np.cos(np.angle(an)).astype(np.float32)
            # inst_freq
            phase = np.unwrap(np.angle(an), axis=0)
            freq  = np.diff(phase, axis=0, prepend=phase[:1, :]) / (2 * np.pi * _dt_us * 1e-6)
            return np.clip(freq, 0, _fs / 2).astype(np.float32)
        finally:
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass

    return _parallel_apply(_block, out)


# ── Reference filter presets ───────────────────────────────────────────────────

def _ref_apply_filter_preset(data: np.ndarray, key: str, dt_us: int) -> np.ndarray:
    """
    Reference filter preset implementation. Mirrors TopasSUITE.py exactly.
    Path: reference. GPU branches preserved from monolith where applicable.

    "none" returns data.copy() — always a new array.
    """
    if key == "none" or not key:
        return data.copy()

    from scipy.signal import hilbert, medfilt, wiener
    from scipy.ndimage import gaussian_filter, laplace, sobel

    fs  = 1e6 / dt_us
    out = data.astype(np.float32)

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
                out = _hilbert_parallel(out, key, dt_us, fs)
        else:
            # CPU: parallelise over trace columns to saturate all cores.
            # Each worker uses scipy.fft with 1 internal thread to avoid
            # over-subscription (N workers × 1 FFT-thread = N threads total).
            out = _hilbert_parallel(out, key, dt_us, fs)

    elif key == "similarity":
        if _GPU:
            import cupy as cp
            try:
                g   = cp.asarray(out)
                sim = cp.ones_like(g)
                if g.shape[1] > 2:
                    left  = g[:, :-2]; mid = g[:, 1:-1]; right = g[:, 2:]
                    num = left * mid + mid * right
                    den = (cp.sqrt(left**2 + mid**2 + 1e-12) *
                           cp.sqrt(mid**2  + right**2 + 1e-12))
                    sim[:, 1:-1] = num / (den + 1e-12)
                out = cp.asnumpy(sim).astype(np.float32)
            except Exception:
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
                g  = cp.asarray(out, dtype=cp.float64)
                sm = cu_gauss(g, sigma=1.0)
                out = cp.asnumpy(g + 2.0 * (g - sm)).astype(np.float32)
            except Exception:
                smooth = gaussian_filter(out.astype(np.float64), sigma=1.0)
                out    = (out + 2.0 * (out - smooth)).astype(np.float32)
        else:
            smooth = gaussian_filter(out.astype(np.float64), sigma=1.0)
            out    = (out + 2.0 * (out - smooth)).astype(np.float32)

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

    elif key in ("topas_narrow", "topas_wide", "topas_hires"):
        bands = {"topas_narrow": (2000, 4000),
                 "topas_wide":   (1000, 8000),
                 "topas_hires":  (4000, 10000)}
        flo, fhi = bands[key]
        flo = max(10, flo)
        fhi = min(fs / 2 - 1, fhi)
        if flo < fhi:
            sos = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")
            def _sos_blk(blk: np.ndarray, _s=sos) -> np.ndarray:
                return sp_signal.sosfilt(_s, blk, axis=0).astype(np.float32)
            out = _parallel_apply(_sos_blk, out)

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


def apply_filter_preset(data: np.ndarray, key: str, dt_us: int) -> np.ndarray:
    """
    Apply a named filter preset to the data matrix.

    Parameters
    ----------
    data  : (ns, n_traces) float32
    key   : internal preset key (see constants.FILTER_PRESETS values)
    dt_us : sample interval in microseconds

    Returns
    -------
    (ns, n_traces) float32 — new array ("none" returns data.copy())

    Path: reference (_ref_apply_filter_preset).
    GPU branches active when CuPy is available (see _backends.GPU).
    """
    return _ref_apply_filter_preset(data, key, dt_us)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _process_data_generic(obj: Any, params: Dict[str, Any]) -> np.ndarray:
    """
    Common pipeline implementation for SegyProfile and ProfileChain.

    Stage order (preserved bit-for-bit from monolith):
      1. Predictive deconvolution
      2. Bandpass filter (Butterworth SOS, parallel blocks)
      3. Filter preset / seismic attribute
      4. TVG exponential gain
      5. AGC (uniform sliding RMS)
      6. Delay-align compensation

    obj must expose: .data, .dt_us, .delays, .min_delay, .ns, .n_traces.
    params keys: decon, decon_op, decon_gap, decon_wn,
                 filt, flo, fhi, preset,
                 tvg, tvg_alpha, agc, agc_win, align,
                 fill_value (float, default NaN — used when align=True;
                             pass 0.0 to fill delay gaps with white).
    """
    from .constants import FILTER_PRESETS as _FP

    data = obj.data.copy()

    if params.get("decon"):
        data = apply_predictive_decon(data, obj.dt_us,
                                      params["decon_op"],
                                      params["decon_gap"],
                                      params["decon_wn"])

    if params.get("filt"):
        fs  = 1e6 / obj.dt_us
        flo = max(10, params["flo"])
        fhi = min(fs / 2 - 1, params["fhi"])
        if flo < fhi:
            sos = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")
            def _sosfilt_block(block: np.ndarray, _sos=sos) -> np.ndarray:
                return sp_signal.sosfilt(_sos, block, axis=0).astype(np.float32)
            data = _parallel_apply(_sosfilt_block, data)

    # Resolve preset: accept either the display name (GUI path) or the
    # direct key (CLI path, e.g. "envelope" instead of "Envelope (amplitud...)")
    _raw_preset = params.get("preset", "")
    preset_key  = _FP.get(_raw_preset, None)
    if preset_key is None:
        preset_key = _raw_preset if _raw_preset in set(_FP.values()) else "none"
    if preset_key != "none":
        data = apply_filter_preset(data, preset_key, obj.dt_us)

    if params.get("tvg"):
        alpha = params["tvg_alpha"]
        t_sec = np.arange(data.shape[0], dtype=np.float32) * (obj.dt_us / 1e6)
        gain_curve = np.clip(np.exp(alpha * t_sec), 0.0, 1e9)
        data *= gain_curve[:, np.newaxis]

    if params.get("agc"):
        win_s = max(3, int(params["agc_win"] / (obj.dt_us / 1000.0)))
        if win_s % 2 == 0:
            win_s += 1
        if _GPU:
            import cupy as cp
            try:
                from cupyx.scipy.ndimage import uniform_filter1d as cu_uf
                g    = cp.asarray(np.abs(data))
                rms  = cu_uf(g, size=win_s, axis=0)
                rms  = cp.maximum(rms, 1e-9)
                data = cp.asnumpy(cp.asarray(data) / rms).astype(np.float32)
            except Exception:
                # GPU failed → fall through to parallel CPU
                _GPU_fallback = True
            else:
                _GPU_fallback = False
        else:
            _GPU_fallback = True

        if not _GPU or _GPU_fallback:
            # Parallel CPU: split across trace blocks.
            # uniform_filter1d operates along axis=0 (time) for each trace
            # independently → perfect for _parallel_apply.
            from scipy.ndimage import uniform_filter1d as _uf

            def _agc_block(blk: np.ndarray, _w=win_s) -> np.ndarray:
                env = np.abs(blk)
                rms = _uf(env, size=_w, axis=0)
                return (blk / np.maximum(rms, 1e-9)).astype(np.float32)

            data = _parallel_apply(_agc_block, data)

    if params.get("align"):
        dt_ms      = obj.dt_us / 1000.0
        offsets    = np.round((obj.delays - obj.min_delay) / dt_ms).astype(int)
        extra      = int(offsets.max())
        new_ns     = obj.ns + extra
        fill_value = float(params.get("fill_value", np.nan))

        aligned = np.full((new_ns, obj.n_traces), fill_value, dtype=np.float32)
        row_idx = np.arange(obj.ns)[:, None] + offsets[None, :]
        col_idx = np.arange(obj.n_traces)[None, :]
        aligned[row_idx, col_idx] = data
        data = aligned

    return data


def process_profile_data(sd: Any, params: Dict[str, Any]) -> np.ndarray:
    """Apply the full processing pipeline to a SegyProfile. Returns new array."""
    return _process_data_generic(sd, params)


def process_chain_data(ch: Any, params: Dict[str, Any]) -> np.ndarray:
    """Apply the full processing pipeline to a ProfileChain. Returns new array."""
    return _process_data_generic(ch, params)


# ── Time window ────────────────────────────────────────────────────────────────

def time_window(obj: Any, data_ns: int, align: bool) -> tuple:
    """
    Compute the (i0, i1, t0_ms, t1_ms) render window for a profile or chain.

    If align is True, t0 = min_delay (delays compensated reference).
    If align is False, t0 = delay_ms (trace[0] delay — known limitation).
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
