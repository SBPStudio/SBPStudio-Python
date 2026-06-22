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
from .logger import get_logger

_LOG = get_logger("processing")


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

    # Cap at 4 to avoid N×N contention with BLAS threads when running inside
    # a CoreWorker export job (BLAS already threads per worker internally).
    with _cf.ThreadPoolExecutor(max_workers=min(n_workers, 4)) as pool:
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


# ── AGC (Automatic Gain Control) ───────────────────────────────────────────────

def apply_agc(data: np.ndarray, win_ms: float, dt_us: int) -> np.ndarray:
    """
    Automatic Gain Control: divide each sample by the local sliding RMS.

    This is the standalone, composable form of the AGC stage previously
    inlined in :func:`_process_data_generic`. Behaviour is identical
    (GPU path when available, else parallel-CPU uniform_filter1d over the
    time axis). Exposed so the GUI node pipeline can call it directly
    instead of re-implementing the math.

    Parameters
    ----------
    data   : (ns, n_traces) float32 — input NOT mutated
    win_ms : AGC window length in ms
    dt_us  : sample interval in microseconds

    Returns
    -------
    (ns, n_traces) float32 — new array

    Path: reference math (matches _ref_agc); dispatches GPU/parallel-CPU.
    """
    win_s = max(3, int(win_ms / (dt_us / 1000.0)))
    if win_s % 2 == 0:
        win_s += 1

    if _GPU:
        import cupy as cp
        try:
            from cupyx.scipy.ndimage import uniform_filter1d as cu_uf
            g   = cp.asarray(np.abs(data))
            rms = cu_uf(g, size=win_s, axis=0)
            rms = cp.maximum(rms, 1e-9)
            return cp.asnumpy(cp.asarray(data) / rms).astype(np.float32)
        except Exception:
            pass  # fall through to parallel CPU

    from scipy.ndimage import uniform_filter1d as _uf

    def _agc_block(blk: np.ndarray, _w=win_s) -> np.ndarray:
        env = np.abs(blk)
        rms = _uf(env, size=_w, axis=0)
        return (blk / np.maximum(rms, 1e-9)).astype(np.float32)

    return _parallel_apply(_agc_block, data)


# ── Bandpass filter ─────────────────────────────────────────────────────────────

def apply_bandpass(data: np.ndarray, flo: float, fhi: float, dt_us: int) -> np.ndarray:
    """
    Zero-distortion-band Butterworth (4th order SOS) bandpass along time.

    Standalone, composable form of the bandpass stage inlined in
    ``_process_data_generic``. ``flo`` is clamped to ≥10 Hz and ``fhi`` to just
    below Nyquist; if the band collapses (flo ≥ fhi) the input is returned
    unchanged (a copy).

    Parameters
    ----------
    data  : (ns, n_traces) float32 — input NOT mutated
    flo   : low cut frequency (Hz)
    fhi   : high cut frequency (Hz)
    dt_us : sample interval in microseconds

    Returns
    -------
    (ns, n_traces) float32 — new array
    """
    fs  = 1e6 / dt_us
    flo = max(10, flo)
    fhi = min(fs / 2 - 1, fhi)
    if flo >= fhi:
        return data.copy()
    sos = sp_signal.butter(4, [flo, fhi], btype="bandpass", fs=fs, output="sos")

    def _sosfilt_block(block: np.ndarray, _sos=sos) -> np.ndarray:
        return sp_signal.sosfilt(_sos, block, axis=0).astype(np.float32)

    return _parallel_apply(_sosfilt_block, data)


# ── TVG (time-variant exponential gain) ─────────────────────────────────────────

def apply_tvg(data: np.ndarray, alpha: float, dt_us: int) -> np.ndarray:
    """
    Time-variant gain: multiply each sample by ``exp(alpha · t)`` (t in seconds).

    Standalone, composable form of the TVG stage inlined in
    ``_process_data_generic`` (gain clipped to [0, 1e9]).

    Parameters
    ----------
    data  : (ns, n_traces) float32 — input NOT mutated
    alpha : exponential attenuation-compensation coefficient
    dt_us : sample interval in microseconds

    Returns
    -------
    (ns, n_traces) float32 — new array
    """
    t_sec = np.arange(data.shape[0], dtype=np.float32) * (dt_us / 1e6)
    gain_curve = np.clip(np.exp(alpha * t_sec), 0.0, 1e9)
    return (data * gain_curve[:, np.newaxis]).astype(np.float32)


# ── Delay alignment (geometry) ──────────────────────────────────────────────────

def apply_delay_alignment(data: np.ndarray, delays: np.ndarray,
                          min_delay: float, dt_us: int,
                          fill_value: float = 0.0) -> np.ndarray:
    """
    Compensate per-trace recording delays so events line up in two-way-time.

    Each trace is shifted DOWN by ``round((delay − min_delay) / dt_ms)`` samples;
    the matrix grows by the largest shift and the exposed gaps are filled with
    ``fill_value`` (0.0 for white, NaN for transparent). Standalone, composable
    form of the align stage inlined in ``_process_data_generic``.

    Parameters
    ----------
    data       : (ns, n_traces) float32 — input NOT mutated
    delays     : (n_traces,) per-trace DelayRecordingTime (ms)
    min_delay  : reference delay (ms) — the alignment origin
    dt_us      : sample interval in microseconds
    fill_value : value for the exposed delay gaps (default 0.0)

    Returns
    -------
    (ns + extra, n_traces) float32 — new, taller array
    """
    ns, n_traces = data.shape
    dt_ms   = dt_us / 1000.0
    offsets = np.round((np.asarray(delays) - min_delay) / dt_ms).astype(int)
    # Despike corrupt DelayRecordingTime headers: a single bad byte produces an
    # offset that spikes and *immediately returns* to baseline.  Detect by the
    # 3-condition neighbor-agreement test — valid step-changes (fault scarps,
    # deep-water canyons, operator window shifts) are preserved by construction
    # because at a real step the right neighbor already holds the new value, so
    # |mid − hi| ≈ 0 and the spike mask is never set.
    if offsets.size >= 3:
        lo  = offsets[:-2].astype(np.int64)
        mid = offsets[1:-1].astype(np.int64)
        hi  = offsets[2:].astype(np.int64)
        spike_mask = (
            (np.abs(mid - lo) > ns) &
            (np.abs(mid - hi) > ns) &
            (np.abs(lo  - hi) <= ns)
        )
        n_spikes = int(spike_mask.sum())
        if n_spikes:
            idx = np.where(spike_mask)[0] + 1   # +1: mid starts at index 1
            offsets[idx] = (
                (offsets[idx - 1].astype(np.int64) +
                 offsets[idx + 1].astype(np.int64)) // 2
            ).astype(int)
            _LOG.warning(
                "apply_delay_alignment: %d isolated delay spike(s) removed "
                "(corrupt SEG-Y header bytes); valid step-changes preserved.",
                n_spikes,
            )
    extra   = int(offsets.max()) if offsets.size else 0
    aligned = np.full((ns + extra, n_traces), fill_value, dtype=np.float32)
    row_idx = np.arange(ns)[:, None] + offsets[None, :]
    col_idx = np.arange(n_traces)[None, :]
    aligned[row_idx, col_idx] = data
    return aligned


# ── Water-column mute ───────────────────────────────────────────────────────────

def apply_water_mute(data: np.ndarray, threshold_pct: float,
                     margin_ms: float, dt_us: int) -> np.ndarray:
    """
    Mute the water column above the seabed, trace by trace (fully vectorised).

    For each trace the seabed is picked as the FIRST sample whose |amplitude|
    reaches ``threshold_pct`` % of that trace's peak |amplitude|. Everything from
    t=0 down to ``pick − margin_ms`` is zeroed (the margin keeps a little signal
    above the seabed so the reflector itself is never clipped).

    Parameters
    ----------
    data          : (ns, n_traces) float32 — input NOT mutated
    threshold_pct : seabed pick threshold, % of per-trace peak (e.g. 30)
    margin_ms     : protect this many ms above the pick (mute stops there)
    dt_us         : sample interval in microseconds

    Returns
    -------
    (ns, n_traces) float32 — new array

    Notes
    -----
    Vectorised: per-trace peak via ``max(axis=0)``, first-crossing via
    ``argmax`` on the boolean threshold mask, mute via a broadcast row<limit
    mask. All-zero traces (peak=0 ⇒ threshold=0 ⇒ pick=0) mute nothing.
    """
    ns, n_traces = data.shape
    dt_ms  = dt_us / 1000.0
    abs_d  = np.abs(data)
    peak   = np.nanmax(abs_d, axis=0)                  # (n_traces,) — NaN-safe for aligned data
    thresh = (threshold_pct / 100.0) * peak            # (n_traces,)

    # First sample per trace reaching the threshold (the peak always qualifies,
    # so argmax always finds a real crossing; ties resolve to the earliest).
    exceed = abs_d >= thresh[None, :]                  # (ns, n_traces) bool
    pick   = np.argmax(exceed, axis=0)                 # (n_traces,)

    margin_s   = int(round(margin_ms / dt_ms))
    mute_until = np.maximum(pick - margin_s, 0)        # (n_traces,) — exclusive
    rows       = np.arange(ns)[:, None]                # (ns, 1)
    mute_mask  = rows < mute_until[None, :]            # (ns, n_traces)

    out = data.copy()
    out[mute_mask] = 0.0
    return out.astype(np.float32)


# ── Swell filter (algorithmic heave correction) ─────────────────────────────────

def apply_swell_filter(data: np.ndarray, window_traces: int,
                       max_shift_ms: float, dt_us: int) -> np.ndarray:
    """
    Remove wave-induced heave by flattening each trace to a smooth spatial
    reference via cross-correlation static shifts (vectorised over traces).

    For each trace a reference is built from the rolling mean of its
    ``window_traces`` neighbours (a smooth, heave-free seabed estimate). The
    per-trace vertical static is the lag in [−max_shift, +max_shift] samples
    that maximises the (unit-normalised) cross-correlation with that reference;
    the trace is then rolled by that lag to align it, cancelling the heave.

    Parameters
    ----------
    data          : (ns, n_traces) float32 — input NOT mutated
    window_traces : neighbourhood width for the smooth reference (traces)
    max_shift_ms  : maximum |static shift| searched/applied (ms)
    dt_us         : sample interval in microseconds

    Returns
    -------
    (ns, n_traces) float32 — new array

    Notes
    -----
    Reference: ``uniform_filter1d`` across traces (axis=1). Cross-correlation is
    a loop over the (small) lag range, each step a vectorised multiply+sum over
    all traces. Zero-lag is the baseline so ties bias to *no* shift (minimal
    heave). The final shift uses ``take_along_axis`` with zero-fill at the edges.
    """
    from scipy.ndimage import uniform_filter1d

    ns, n_traces = data.shape
    dt_ms     = dt_us / 1000.0
    max_shift = max(1, int(round(max_shift_ms / dt_ms)))
    win       = max(3, int(window_traces))
    if n_traces < 3:
        return data.copy().astype(np.float32)

    # Smooth spatial reference (heave-free seabed estimate).
    ref = uniform_filter1d(data, size=win, axis=1, mode="nearest").astype(np.float32)

    # Unit-normalise per trace → cosine cross-correlation (robust peak picking).
    eps = 1e-12
    dn = data / (np.linalg.norm(data, axis=0, keepdims=True) + eps)
    rn = ref  / (np.linalg.norm(ref,  axis=0, keepdims=True) + eps)

    # Baseline = zero lag; only strictly better lags override (ties → no shift).
    best_corr = np.sum(dn * rn, axis=0)                # (n_traces,)
    best_lag  = np.zeros(n_traces, dtype=int)
    for lag in range(-max_shift, max_shift + 1):
        if lag == 0:
            continue
        rr = np.roll(rn, lag, axis=0)
        if lag > 0:
            rr[:lag, :] = 0.0
        else:
            rr[lag:, :] = 0.0
        corr = np.sum(dn * rr, axis=0)
        upd = corr > best_corr
        best_corr[upd] = corr[upd]
        best_lag[upd] = lag

    # Align each trace to the reference: out[i] = data[i + lag] (zero-filled).
    rows  = np.arange(ns)[:, None]
    src   = rows + best_lag[None, :]                   # (ns, n_traces)
    valid = (src >= 0) & (src < ns)
    out   = np.take_along_axis(data, np.clip(src, 0, ns - 1), axis=0)
    out[~valid] = 0.0
    return out.astype(np.float32)


# ── Spectral whitening (resolution enhancement) ────────────────────────────────

def _ref_apply_spectral_whitening(
    data: np.ndarray,
    dt_us: int,
    flo: float,
    fhi: float,
    smooth_hz: float,
) -> np.ndarray:
    """
    Reference spectral whitening, trace by trace (vectorised over columns).

    Algorithm (zero-phase):
      1. rfft each column at its original length → complex spectrum X
      2. Compute amplitude envelope A = |X|, smooth it along the frequency
         axis with a uniform filter of width ``smooth_hz``
      3. Inside [flo, fhi]: divide X by the smoothed envelope (normalise)
      4. Outside [flo, fhi]: keep X unchanged so the node composes cleanly
         before/after a BandpassNode without hard-cutting the flanks
      5. irfft → output trace (same length, same phase, flattened amplitude)

    Path: reference.  Parallelised over column blocks via ``_parallel_apply``.
    """
    from scipy.ndimage import uniform_filter1d

    ns, _nt = data.shape
    fs      = 1e6 / dt_us
    nyquist = fs / 2.0
    flo     = max(0.0, flo)
    fhi     = min(nyquist - 1.0, fhi)
    if flo >= fhi:
        return data.copy()

    # Frequency-axis metadata for a length-ns real FFT (computed once, shared
    # across all column blocks via closure — bin count never changes for a
    # fixed (ns, dt_us) pair).
    freqs  = np.fft.rfftfreq(ns, d=dt_us / 1e6)          # (n_freqs,) Hz
    bin_hz = float(freqs[1]) if freqs.size > 1 else 1.0   # Hz per bin
    band   = (freqs >= flo) & (freqs <= fhi)              # (n_freqs,) bool

    if not np.any(band):
        return data.copy()

    # Smoothing window in bins; odd so uniform_filter1d is symmetric.
    smooth_b = max(3, int(smooth_hz / bin_hz))
    if smooth_b % 2 == 0:
        smooth_b += 1

    def _whiten_block(block: np.ndarray) -> np.ndarray:
        # rfft axis=0 (time axis) gives one spectrum per column — fully
        # vectorised: no inner trace loop.
        X   = np.fft.rfft(block.astype(np.float64), n=block.shape[0], axis=0)
        amp = np.abs(X)                                    # (n_freqs, _nt_b)

        # Smooth the amplitude envelope along the frequency axis (same
        # uniform_filter1d approach as AGC uses on the time axis).
        env = uniform_filter1d(amp, size=smooth_b, axis=0, mode='nearest')

        # Per-trace in-band peak → adaptive eps floor that stays stable
        # for quiet/all-zero traces without clamping real signal.
        # peak shape (1, _nt_b) for broadcasting with (n_band, _nt_b).
        peak = env[band, :].max(axis=0, keepdims=True)    # (1, _nt_b)
        eps  = np.maximum(peak * 1e-9, 1e-30)

        X_out = X.copy()
        X_out[band, :] = X[band, :] / np.maximum(env[band, :], eps)

        # Silent traces (peak ≤ eps floor) pass through unchanged so the
        # node never amplifies numerical noise on dead channels.
        silent = peak[0, :] <= 1e-30                      # (_nt_b,) bool
        if np.any(silent):
            X_out[:, silent] = X[:, silent]

        return np.fft.irfft(X_out, n=block.shape[0], axis=0).astype(np.float32)

    return _parallel_apply(_whiten_block, data)


def apply_spectral_whitening(
    data: np.ndarray,
    dt_us: int,
    flo: float,
    fhi: float,
    smooth_hz: float,
) -> np.ndarray:
    """
    Spectral whitening (resolution enhancement), trace by trace.

    Flattens the amplitude spectrum within [flo, fhi] by dividing each trace's
    complex rfft by a smoothed version of its own amplitude envelope.  The
    operation is strictly zero-phase: the complex phase is preserved exactly,
    only the magnitude changes.  The net effect is broader effective bandwidth
    and sharper temporal resolution — the primary resolution-enhancement step
    for SBP / TOPAS sub-bottom profiler data, also applicable to high-frequency
    MCS surveys.

    Parameters
    ----------
    data      : (ns, n_traces) float32 — input NOT mutated
    dt_us     : sample interval in microseconds
    flo       : lower band limit (Hz) — whitening starts at this frequency
    fhi       : upper band limit (Hz) — whitening ends at this frequency
    smooth_hz : smoothing window applied to the spectral envelope (Hz).
                Narrower values flatten more aggressively (wider effective
                bandwidth); wider values apply gentle broad-band equalization.
                Typical SBP range: 200–600 Hz.

    Returns
    -------
    (ns, n_traces) float32 — new array

    Notes
    -----
    The smoothed envelope is computed via ``scipy.ndimage.uniform_filter1d``
    along the frequency axis — the same approach AGC uses on the time axis.
    Outside [flo, fhi] the original spectrum is preserved unchanged, so this
    node composes cleanly before or after a BandpassNode.  Dead traces (all-zero
    amplitude in band) pass through unmodified.  Parallelised over trace-column
    blocks via ``_parallel_apply``.

    Path: reference (_ref_apply_spectral_whitening).
    """
    return _ref_apply_spectral_whitening(data, dt_us, flo, fhi, smooth_hz)


# ── F-K (frequency–wavenumber) dip filter ───────────────────────────────────────

def apply_fk_filter(data: np.ndarray, dt_us: int, dip_ms: float,
                    width_ms: float, mode: str = "reject_both") -> np.ndarray:
    """2-D frequency–wavenumber (F-K) dip filter — rejects (or isolates) a fan
    of coherently DIPPING events (side-echoes, diffraction tails, towfish/
    cable noise) by their apparent slope, leaving flat reflectors intact.

    A linear event advancing ``p`` samples per trace maps in the 2-D FFT to the
    radial line ``k = -p·f`` (normalised cycles/trace vs cycles/sample). For
    every (f, k) bin the apparent dip is ``p = -k/f``; the fan
    ``[dip-width, dip+width]`` (converted ms/trace → samples/trace) is the
    reject (or, in ``pass`` mode, the keep) region. The mask is lightly
    Gaussian-smoothed (wrap-around) to soften the cut and limit ringing.

    Parameters
    ----------
    data     : (ns, n_traces) float32 — input NOT mutated. MUST be the true,
               un-decimated viewport block: decimating the columns would
               corrupt the wavenumber axis (the GUI runs this on full-res
               viewport data, see PreviewController / DSPNode.NEEDS_FULL_RES).
    dt_us    : sample interval (µs) — sets samples↔ms for the dip conversion.
    dip_ms   : centre apparent dip of the fan (ms per trace; sign = direction).
    width_ms : half-width of the fan (ms per trace).
    mode     : "reject_both" (reject ±dip), "reject_one" (reject the signed
               fan only), or "pass" (keep ONLY the fan, remove everything else).

    Returns
    -------
    (ns, n_traces) float32 — new array.
    """
    ns, nt = data.shape
    if ns < 4 or nt < 4:
        return data.copy()
    from scipy.fft import fft2, ifft2
    from scipy.ndimage import gaussian_filter

    dt_ms = (dt_us or 1) / 1000.0
    # float32 throughout the mask build: at the 4096² cap a float64 (f, k, p,
    # mask) intermediate would double the memory of the complex64 spectrum
    # itself for no precision benefit (the mask is a soft 0..1 gate, not a
    # value that accumulates error).
    f = np.fft.fftfreq(ns).astype(np.float32)[:, None]   # cycles/sample (ns, 1)
    k = np.fft.fftfreq(nt).astype(np.float32)[None, :]   # cycles/trace  (1, nt)
    p_c = dip_ms / dt_ms                                  # centre dip, samples/trace
    p_w = max(abs(width_ms), 1e-6) / dt_ms               # half-width, samples/trace
    p_lo, p_hi = p_c - p_w, p_c + p_w
    with np.errstate(divide="ignore", invalid="ignore"):
        p = -k / f                                       # (ns, nt); ±inf on the f=0 row
    fan = (p >= p_lo) & (p <= p_hi)
    if mode == "reject_both":
        fan = fan | ((p >= -p_hi) & (p <= -p_lo))

    if mode == "pass":
        keep = fan.copy()
        keep[0, :] = True       # always keep the f=0 (time-DC) row …
        keep[:, 0] = True       # … and the k=0 (zero-dip) column → preserve flat events
        mask = keep.astype(np.float32)
    else:
        mask = (~fan).astype(np.float32)
    mask = gaussian_filter(mask, sigma=1.5, mode="wrap")  # soften the cut (anti-ring)

    spec = fft2(data.astype(np.float32))                  # complex64 (scipy preserves dtype)
    spec *= mask                                          # in-place: no extra complex64 buffer
    del mask
    out = ifft2(spec).real
    del spec                                              # release before the final copy below
    return np.ascontiguousarray(out, dtype=np.float32)


# ── Seabed (water-bottom) multiple suppression ──────────────────────────────────

def apply_multiple_suppression(data: np.ndarray, dt_us: int,
                               threshold_pct: float, period_ms: float = 0.0,
                               max_gain: float = 1.0) -> np.ndarray:
    """Suppress the first water-bottom MULTIPLE by adaptive predictive
    subtraction at the seabed period (critical for shallow SBP, where the
    seabed multiple masks the sub-bottom).

    The first multiple is a delayed copy of the primary arriving one seabed
    two-way-time later, so a 1-tap predictor at lag = seabed sample removes it:
    for each trace the best-fit scalar gain ``g = Σ x·x₋ₗ / Σ x₋ₗ²`` (the
    least-squares match of the lag-shifted trace to itself) predicts the
    repeating event, and ``x − g·x₋ₗ`` cancels it. Fully vectorised over
    traces (per-trace lag via ``take_along_axis``).

    Parameters
    ----------
    data          : (ns, n_traces) float32 — input NOT mutated.
    dt_us         : sample interval (µs).
    threshold_pct : seabed pick threshold, % of each trace's peak |amp| — the
                    seabed sample sets the per-trace prediction lag (the
                    primary→multiple period). Ignored when ``period_ms`` > 0.
    period_ms     : fixed prediction period (ms). 0 → auto-pick per trace.
    max_gain      : clamp on the adaptive gain so a mis-pick can't over-subtract
                    / invert a primary.

    Returns
    -------
    (ns, n_traces) float32 — new array.
    """
    ns, nt = data.shape
    if ns < 4:
        return data.copy()
    dt_ms = (dt_us or 1) / 1000.0

    if period_ms and period_ms > 0:
        lag = np.full(nt, max(1, int(round(period_ms / dt_ms))), dtype=int)
    else:
        abs_d = np.abs(data)
        peak = np.nanmax(abs_d, axis=0)
        thr = (threshold_pct / 100.0) * peak
        onset = np.argmax(abs_d >= thr[None, :], axis=0)   # first seabed crossing
        # The prediction period is the primary→multiple separation = the seabed
        # PEAK two-way time, NOT the threshold onset (which precedes the peak by
        # the wavelet rise and would mis-align the predictor). Refine each pick
        # to the local |amplitude| peak in a short window after the onset.
        w = max(5, int(round(5.0 / dt_ms)))                # ~5 ms search window
        rows0 = np.arange(ns)[:, None]
        inwin = (rows0 >= onset[None, :]) & (rows0 < (onset + w)[None, :])
        sb = np.argmax(np.where(inwin, abs_d, -1.0), axis=0)
        lag = np.maximum(sb.astype(int), 1)                # primary→multiple period (samples)

    rows = np.arange(ns)[:, None]
    src = rows - lag[None, :]                            # 'one period earlier' index
    valid = src >= 0
    shifted = np.take_along_axis(data, np.clip(src, 0, ns - 1), axis=0)
    shifted = np.where(valid, shifted, 0.0).astype(np.float32)

    # Estimate the gain ONLY within the first-multiple window [lag, 2.5·lag].
    # The lag-shifted trace also carries a deeper "ghost" of the multiple
    # itself (around 3·lag) that has no real event to match; including it in
    # the least-squares denominator would dilute the gain and leave the
    # multiple under-subtracted. Restricting to the first-multiple band makes
    # the 1-tap predictor cancel it cleanly without touching deeper data.
    win_hi = np.minimum(2 * lag + lag // 2, ns)
    wmask = (rows >= lag[None, :]) & (rows < win_hi[None, :])
    num = np.sum(data * shifted * wmask, axis=0)
    den = np.sum(shifted * shifted * wmask, axis=0)
    g = np.where(den > 1e-12, num / den, 0.0)
    g = np.clip(g, -abs(max_gain), abs(max_gain))
    out = data - g[None, :] * shifted
    return np.ascontiguousarray(out, dtype=np.float32)


# ── Notch (surgical band-stop) filter ───────────────────────────────────────────

def apply_notch(data: np.ndarray, dt_us: int, freq: float,
                q: float = 30.0) -> np.ndarray:
    """Surgically remove a single narrow interference frequency (electrical
    resonance, tow-cable/strumming tone) with a zero-phase IIR notch.

    ``scipy.signal.iirnotch(freq, Q, fs)`` designs a 2nd-order notch; it is
    applied with ``filtfilt`` (forward-backward → zero phase, no event shift)
    along the time axis. Higher ``Q`` = narrower notch (surgical); lower Q
    removes a wider band around ``freq``.

    Parameters
    ----------
    data  : (ns, n_traces) float32 — input NOT mutated.
    dt_us : sample interval (µs).
    freq  : notch centre frequency (Hz). A no-op (copy) if ≤0 or ≥ Nyquist.
    q     : quality factor (centre/bandwidth).

    Returns
    -------
    (ns, n_traces) float32 — new array.
    """
    ns, _nt = data.shape
    fs = 1e6 / (dt_us or 1)
    if freq <= 0 or freq >= fs / 2 or ns < 4:
        return data.copy()
    b, a = sp_signal.iirnotch(freq, max(0.1, q), fs)
    # filtfilt needs > 3*max(len(a),len(b)) samples; guard tiny windows.
    if ns <= 3 * max(len(a), len(b)):
        def _notch_blk(blk, _b=b, _a=a):
            return sp_signal.lfilter(_b, _a, blk, axis=0).astype(np.float32)
    else:
        def _notch_blk(blk, _b=b, _a=a):
            return sp_signal.filtfilt(_b, _a, blk, axis=0).astype(np.float32)
    return _parallel_apply(_notch_blk, data)


# ── Amplitude spectrum (FFT) ────────────────────────────────────────────────────

def compute_amplitude_spectrum(data: np.ndarray, dt_us: int) -> tuple:
    """
    Mean-trace amplitude spectrum of a seismic section (for filter tuning).

    The traces are averaged into a single mean 1-D trace (stacking improves the
    SNR of the estimate), Hann-tapered to limit spectral leakage, transformed
    with a real FFT, and the magnitude is lightly smoothed so the spectral
    envelope is readable.

    Parameters
    ----------
    data  : (ns, n_traces) float32, or a 1-D (ns,) trace
    dt_us : sample interval in microseconds

    Returns
    -------
    (freqs_hz, amplitude_db) : both 1-D float32 arrays of length ns//2 + 1.
        ``freqs_hz`` spans 0 … Nyquist (1e6 / dt_us / 2). ``amplitude_db`` is the
        power magnitude in decibels, NORMALISED so the peak is 0 dB; the 0 Hz
        (DC) bin is forced to the floor so it never compresses the plot.
    """
    arr = np.asarray(data)
    trace = arr.mean(axis=1) if arr.ndim == 2 else arr
    trace = np.nan_to_num(trace, nan=0.0).astype(np.float64)
    n = trace.size
    if n < 4:
        return np.zeros(1, np.float32), np.zeros(1, np.float32)

    trace = trace - trace.mean()                       # coarse DC removal
    spec  = np.fft.rfft(trace * np.hanning(n))         # Hann taper → less leakage
    freqs = np.fft.rfftfreq(n, d=dt_us / 1e6)          # Hz
    amp   = np.abs(spec)
    amp[0] = 0.0                                        # kill DC BEFORE smoothing
                                                       # so the spike can't smear

    # Light moving-average smoothing for a readable envelope (~1 % of the band).
    if amp.size > 8:
        from scipy.ndimage import uniform_filter1d
        amp = uniform_filter1d(amp, size=max(3, amp.size // 100))
    amp[0] = 0.0                                        # STRICT: 0 Hz forced to 0

    # Decibel (power) scale — essential for the huge dynamic range of seismic
    # spectra — normalised so the spectral peak sits at 0 dB. The DC bin → very
    # negative; clamp the floor so the plot/auto-range stays sane.
    amp_db = 20.0 * np.log10(amp + 1e-12)
    amp_db -= amp_db.max()
    amp_db = np.maximum(amp_db, -120.0)

    return freqs.astype(np.float32), amp_db.astype(np.float32)


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

    data = obj.data   # no upfront copy — each active stage returns a new array

    if params.get("decon"):
        data = apply_predictive_decon(data, obj.dt_us,
                                      params["decon_op"],
                                      params["decon_gap"],
                                      params["decon_wn"])

    if params.get("filt"):
        # Single source of truth — the standalone apply_bandpass.
        data = apply_bandpass(data, params["flo"], params["fhi"], obj.dt_us)

    # Resolve preset: accept either the display name (GUI path) or the
    # direct key (CLI path, e.g. "envelope" instead of "Envelope (amplitud...)")
    _raw_preset = params.get("preset", "")
    preset_key  = _FP.get(_raw_preset, None)
    if preset_key is None:
        preset_key = _raw_preset if _raw_preset in set(_FP.values()) else "none"
    if preset_key != "none":
        data = apply_filter_preset(data, preset_key, obj.dt_us)

    if params.get("tvg"):
        data = apply_tvg(data, params["tvg_alpha"], obj.dt_us)

    if params.get("agc"):
        # Single source of truth — the standalone apply_agc (same GPU/parallel
        # dispatch as before; the GUI node pipeline calls the same function).
        data = apply_agc(data, params["agc_win"], obj.dt_us)

    if params.get("align"):
        fill_value = float(params.get("fill_value", np.nan))
        data = apply_delay_alignment(data, obj.delays, obj.min_delay,
                                     obj.dt_us, fill_value=fill_value)

    if data is obj.data:   # no stage ran — copy to honour "new array" contract
        data = data.copy()
    return data


def process_profile_data(sd: Any, params: Dict[str, Any]) -> np.ndarray:
    """Apply the full processing pipeline to a SegyProfile. Returns new array."""
    return _process_data_generic(sd, params)


def process_chain_data(ch: Any, params: Dict[str, Any]) -> np.ndarray:
    """Apply the full processing pipeline to a ProfileChain. Returns new array."""
    return _process_data_generic(ch, params)


# ── Reference AGC (sequential, for regression tests) ──────────────────────────

def _ref_agc(data: np.ndarray, win_ms: float, dt_us: int) -> np.ndarray:
    """
    Reference AGC: sequential uniform_filter1d on the FULL matrix.
    Path: reference.  Used by regression tests as ground truth vs parallel path.

    Parameters
    ----------
    data   : (ns, n_traces) float32
    win_ms : AGC window length in ms
    dt_us  : sample interval in µs
    """
    from scipy.ndimage import uniform_filter1d
    win_s = max(3, int(win_ms / (dt_us / 1000.0)))
    if win_s % 2 == 0:
        win_s += 1
    env  = np.abs(data)
    rms  = uniform_filter1d(env, size=win_s, axis=0)
    return (data / np.maximum(rms, 1e-9)).astype(np.float32)


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
