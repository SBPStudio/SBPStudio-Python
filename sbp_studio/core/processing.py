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
- Exception: apply_dc_removal(data, inplace=True) is an explicit, opt-in
  escape hatch for the ONE call site (io_segy.py's load path) that owns a
  freshly-allocated array exclusively and benefits from skipping the extra
  allocation. Every other caller, and the default (inplace=False), keeps the
  contract above.

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


# ── DC offset removal (mandatory, first stage — see io_segy.py's load path) ─────

def apply_dc_removal(data: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Mandatory DC-offset removal: subtract each trace's OWN mean so every
    trace is strictly centred on zero.

    This is "stage 0" of the pipeline — applied to ``prof.data`` immediately
    after the raw trace matrix is loaded (see
    ``io_segy._populate_profile_from_file``), BEFORE any DSP node, the legacy
    generic pipeline (:func:`process_profile_data`), or the live preview ever
    sees the data. Raw instrumental DC bias is harmless on its own, but once
    a downstream gain stage (AGC, TVG) or filter (bandpass) touches it, a
    per-trace offset gets amplified/redistributed into vertical 'striping'
    artifacts in the water column — this removes the cause at the source,
    once, rather than letting every filter fight a symptom.

    Hardware/survey-agnostic by construction: the only assumption is the
    array contract itself (axis 0 = samples/time, axis 1 = traces) — no
    trace-length, sample-rate, or instrument-specific constant anywhere, so
    it is identically correct for a 9000-sample TOPAS SBP trace or a
    3000-sample deep-MCS trace. Pure subtraction with NO division anywhere,
    so it is provably NaN/Inf-safe even for a degenerate all-zero or
    constant (dead-channel) trace.

    Parameters
    ----------
    data    : (ns, n_traces) float32 (or any float dtype) — per the module
              contract, NOT mutated unless ``inplace=True``
    inplace : subtract the mean DIRECTLY into ``data`` instead of allocating
              a new array — a real saving for the multi-GB matrices this app
              handles, but ONLY safe when the caller exclusively owns
              ``data`` and no other reference depends on it staying
              unchanged (e.g. immediately after
              ``f.trace.raw[:].T.astype(...)`` at load time, before
              anything else can have touched it). Defaults to False to
              match every other ``apply_*`` function's contract.

    Returns
    -------
    (ns, n_traces) — ``data`` itself if ``inplace``, else a new array of the
    same dtype as ``data``.
    """
    # float64 accumulator keeps the mean numerically robust regardless of
    # trace length (summing many float32 samples can otherwise lose
    # precision) — the tiny (1, n_traces) mean array is the only extra
    # allocation either way.
    mean = data.mean(axis=0, keepdims=True, dtype=np.float64)
    if inplace:
        data -= mean
        return data
    return (data - mean).astype(data.dtype, copy=False)


# ── Trace equalization (selectable pipeline node, NOT mandatory) ────────────────

def apply_trace_equalization(data: np.ndarray) -> np.ndarray:
    """
    Trace Equalization (RMS Balance): divide each trace by its own RMS
    amplitude, so every trace carries comparable energy along the line
    regardless of source/receiver coupling or range-dependent attenuation.

    Unlike :func:`apply_dc_removal` (mandatory, applied once at load time —
    see io_segy.py — because zero-centring is a precondition for a
    meaningful RMS), this is a SELECTABLE, order-sensitive pipeline node
    (see gui/dsp/nodes.py's TraceEqualizationNode): a downstream gain stage
    (AGC) or local-contrast stage (CLAHE) amplifies whatever relative
    trace-to-trace imbalance still survives at that point in the chain,
    which is what produces vertical 'striping' in the water column.
    Inserting this node BEFORE such a stage removes the imbalance there;
    after it instead re-balances whatever that stage produced — the user's
    placement decides which, exactly like every other reorderable node.

    Safety: RMS == 0 (or numerically negligible — a dead/disconnected
    channel) would otherwise divide live values by ~0 and explode to
    Inf/NaN; that trace is returned as all-zero instead, the same
    degenerate-trace contract every other node in this module uses (e.g.
    apply_water_mute's all-zero-trace guard).

    Hardware/survey-agnostic: the only assumption is the array contract
    itself (axis 0 = samples, axis 1 = traces) — no trace-length,
    sample-rate, or instrument-specific constant anywhere.

    Parameters
    ----------
    data : (ns, n_traces) float32 (or any float dtype) — NOT mutated.

    Returns
    -------
    (ns, n_traces) — new array, same dtype as ``data``.
    """
    # Squaring straight into a float64 OUTPUT (rather than upcasting `data`
    # first) keeps this to one extra (1, n_traces)-shaped float64 mean plus
    # one same-size-as-data float64 temporary — not two.
    mean_sq = np.mean(np.square(data, dtype=np.float64), axis=0, keepdims=True)
    rms = np.sqrt(mean_sq)
    safe = rms > 1e-6
    out = np.zeros_like(data, dtype=np.float64)
    np.divide(data, rms, out=out, where=safe)
    return out.astype(data.dtype, copy=False)


# ── Trace mixing (selectable pipeline node, NOT mandatory) ──────────────────────

def apply_trace_mixing(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Trace Mixing (Horizontal Spatial Smoothing): rolling average ACROSS
    traces (axis=1) at each sample row independently. A continuous reflector
    has coherent amplitude/phase from one trace to the next, so neighbouring
    traces interfere CONSTRUCTIVELY and the event survives the average;
    incoherent random ("salt-and-pepper") noise has no such cross-trace
    correlation, so it interferes DESTRUCTIVELY and is attenuated — exactly
    the noise a downstream gain stage (AGC, TVG) would otherwise amplify
    into visible speckle in deep, low-SNR sections.

    ``window_size`` MUST be odd so the average stays centred on each trace
    without laterally shifting events — an even window has no centre column,
    which would shift every reflector by half a trace. Coerced up to the
    next odd value if given even; floored at 1 (a no-op pass-through).

    Implementation: ``scipy.ndimage.uniform_filter1d`` along axis=1 — a
    single optimized C rolling-average pass, not a Python loop over traces.
    ``mode="nearest"`` (edge replication) avoids the artificial mirrored
    duplicate traces ``mode="reflect"`` would otherwise introduce at the two
    ends of the line.

    Hardware/survey-agnostic: the only assumption is the array contract
    itself (axis 0 = samples, axis 1 = traces) — no trace-length, sample-
    rate, or instrument-specific constant anywhere.

    Parameters
    ----------
    data        : (ns, n_traces) float32 (or any float dtype) — NOT mutated.
    window_size : number of traces averaged together (odd, >= 1). Default 3.

    Returns
    -------
    (ns, n_traces) — new array, same dtype as ``data``.
    """
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= 1:
        return data.copy()

    from scipy.ndimage import uniform_filter1d
    out = uniform_filter1d(data, size=window_size, axis=1, mode="nearest")
    return out.astype(data.dtype, copy=False)


# ── Median filter (edge-preserving alternative to Trace Mixing) ─────────────────

def apply_median_filter(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Median Filter (Edge-Preserving Spatial Denoise): replaces each sample
    with the MEDIAN of itself and its horizontal neighbours (same sample
    row, ``window_size`` adjacent traces). Unlike :func:`apply_trace_mixing`
    (a MEAN, i.e. a low-pass filter), the median is unmoved by a single
    outlier value in the window, so an isolated salt-and-pepper noise spike
    is rejected outright rather than smeared across its neighbours — the
    mean's "watercolor" blurring of sharp lateral discontinuities (fault
    edges, steeply-dipping reflectors) never happens, because the median is
    always one of the ACTUAL input values, never an interpolated blend.

    ``window_size`` MUST be odd, for the same reason as ``apply_trace_mixing``
    (a centred window with no half-trace lateral shift); coerced up to the
    next odd value if given even, floored at 1 (a no-op pass-through).

    Implementation: ``scipy.ndimage.median_filter`` with ``size=(1,
    window_size)`` — NOT a square/symmetric footprint. The footprint's FIRST
    axis (samples) is fixed at 1 so the median is taken strictly along axis=1
    (traces) at each sample row independently; a >1 first-axis would also
    blend across TIME, destroying vertical/temporal resolution, which this
    filter must never touch. ``mode="nearest"`` (edge replication) avoids the
    artificial mirrored duplicate traces ``mode="reflect"`` would introduce
    at the two ends of the line.

    Hardware/survey-agnostic: the only assumption is the array contract
    itself (axis 0 = samples, axis 1 = traces) — no trace-length, sample-
    rate, or instrument-specific constant anywhere.

    Parameters
    ----------
    data        : (ns, n_traces) float32 (or any float dtype) — NOT mutated.
    window_size : number of traces evaluated together (odd, >= 1). Default 3.

    Returns
    -------
    (ns, n_traces) — new array, same dtype as ``data``.
    """
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= 1:
        return data.copy()

    from scipy.ndimage import median_filter
    out = median_filter(data, size=(1, window_size), mode="nearest")
    return out.astype(data.dtype, copy=False)


# ── SVD filter / Karhunen-Loeve transform (coherent-signal enhancement) ─────────

def apply_svd_filter(data: np.ndarray, num_components: int = 10) -> np.ndarray:
    """
    SVD Filter (Karhunen-Loeve Transform): rebuild the matrix from only its
    ``num_components`` largest singular values/vectors —
    ``X_clean = U_k @ diag(S_k) @ V_k^T``. A coherent reflector is spatially
    repetitive across many traces, so it concentrates almost all of its
    energy into a handful of dominant singular components; dense, spatially
    incoherent thermal/random noise spreads thinly across ALL of them.
    Truncating to the top ``num_components`` keeps the former and discards
    the latter — a fundamentally different (and stronger) mechanism than
    Trace Mixing/Median Filter's local neighbour-window averaging, useful
    when those are too conservative against widespread noise in deep,
    low-SNR sections.

    Math/performance: a FULL SVD of an (ns, n_traces) matrix is O(ns·n_traces·
    min(ns,n_traces)) time and materialises min(ns,n_traces) singular vectors
    — wasteful when only a handful are ever kept. ``scipy.sparse.linalg.svds``
    (ARPACK, Lanczos iteration) computes ONLY the requested top-k directly,
    without ever forming the full decomposition — the appropriate tool here,
    not ``np.linalg.svd``.

    ``num_components`` is clipped to ``[1, min(ns, n_traces) - 1]`` (svds'
    own hard constraint: ``k`` must be a strictly smaller than the matrix's
    smaller dimension) — silently, so an unusually small viewport/chain chunk
    never crashes the pipeline.

    Safety: an EXACTLY all-zero chunk (e.g. a fully-muted/dead window) has no
    nonzero starting vector for ARPACK's Lanczos iteration and would raise an
    ``ArpackError``; detected up front and returned as a zero copy instead —
    consistent with every other degenerate-input guard in this module.

    Hardware/survey-agnostic: the only assumption is the array contract
    itself (axis 0 = samples, axis 1 = traces) — no trace-length, sample-
    rate, or instrument-specific constant anywhere.

    Parameters
    ----------
    data           : (ns, n_traces) float32 (or any float dtype) — NOT mutated.
    num_components : number of dominant singular components kept. Default 10.

    Returns
    -------
    (ns, n_traces) — new array, same dtype as ``data``.
    """
    ns, n_traces = data.shape
    max_k = min(ns, n_traces) - 1
    if max_k < 1 or not np.any(data):
        return data.copy()
    k = max(1, min(int(num_components), max_k))

    from scipy.sparse.linalg import svds
    u, s, vt = svds(data.astype(np.float64, copy=False), k=k)
    out = (u * s) @ vt
    return out.astype(data.dtype, copy=False)


# ── Bilateral filter (edge-preserving spatial smoothing) ────────────────────────

def apply_bilateral_filter(data: np.ndarray, window_size: int = 5,
                           sigma_space: float = 2.0, sigma_color: float = 0.5) -> np.ndarray:
    """
    Bilateral Filter (Edge-Preserving Spatial Smoothing): a weighted average
    of horizontal neighbour traces, where each neighbour's weight is the
    PRODUCT of two Gaussians — one on spatial distance (``sigma_space``,
    "how far"), one on amplitude difference (``sigma_color``, "how similar").
    Unlike Trace Mixing's plain mean (which weights every neighbour equally
    regardless of how different it is), a neighbour whose amplitude differs
    sharply from the centre trace — exactly what happens across a fault or a
    steep reflector edge — gets a near-zero weight and is excluded from its
    own average, so the smoothing never crosses the discontinuity. Where
    amplitudes ARE locally similar (a quiet stretch with only random noise
    riding on it), neighbours get full spatial weight and the noise is
    averaged down same as Trace Mixing would. A tunable middle ground
    between Trace Mixing (smooth everywhere, blurs edges) and Median Filter
    (preserves edges, but a harder on/off decision with no weighted blend).

    ``sigma_color`` is a UNITLESS, RELATIVE tolerance, not a raw amplitude:
    internally it is scaled by this chunk's own amplitude spread
    (``data.std()``), so e.g. ``sigma_color=0.5`` always means "half a
    standard deviation of THIS data," regardless of whether the section is
    recorded in raw counts, volts, or an already-gained display unit — the
    user never has to guess an abstract absolute float. Standard deviation
    (rather than max-abs) is used because a single noise spike would
    otherwise inflate the whole tolerance window and weaken edge
    preservation everywhere else in the chunk.

    ``window_size`` MUST be odd (centred window, no lateral event shift),
    coerced up to the next odd value if given even, floored at 1 (a no-op
    pass-through) — same convention as Trace Mixing/Median Filter.

    Implementation: a Python loop over the (small) window OFFSETS — not over
    traces or samples — each iteration a fully vectorised NumPy pass over the
    whole array (shift via edge-padding, squared-difference, two Gaussians,
    accumulate). ``window_size`` iterations of O(ns·n_traces) vector ops, not
    ``ns·n_traces`` individual Python-level pixel computations.

    Safety: the centre offset (k=0) always has zero amplitude difference and
    distance, so its weight is exactly 1 and the per-position weight sum can
    never be zero — division-safe by construction, no epsilon flooring
    needed. A literally constant (zero-variance) chunk has nothing to
    normalise sigma_color against and is returned as a pass-through copy.
    ``sigma_space``/``sigma_color`` are floored at a tiny epsilon so a
    careless 0.0 from the UI can't divide by zero in the Gaussian exponents.

    Hardware/survey-agnostic: the only assumption is the array contract
    itself (axis 0 = samples, axis 1 = traces) — no trace-length, sample-
    rate, or instrument-specific constant anywhere.

    Parameters
    ----------
    data        : (ns, n_traces) float32 (or any float dtype) — NOT mutated.
    window_size : number of traces evaluated together (odd, >= 1). Default 5.
    sigma_space : spatial Gaussian spread, in TRACES. Default 2.0.
    sigma_color : amplitude-similarity tolerance, as a multiple of this
                  chunk's own std-dev. Default 0.5.

    Returns
    -------
    (ns, n_traces) — new array, same dtype as ``data``.
    """
    window_size = max(1, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    if half == 0:
        return data.copy()

    data64 = data.astype(np.float64, copy=False)
    scale = float(np.std(data64))
    if scale < 1e-12:
        return data.copy()

    sigma_space = max(float(sigma_space), 1e-6)
    sigma_color = max(float(sigma_color), 1e-6)
    space_denom = 2.0 * sigma_space ** 2
    color_denom = 2.0 * (sigma_color * scale) ** 2

    n_traces = data64.shape[1]
    padded = np.pad(data64, ((0, 0), (half, half)), mode="edge")

    acc = np.zeros_like(data64)
    wsum = np.zeros_like(data64)
    for k in range(-half, half + 1):
        neighbor = padded[:, half + k: half + k + n_traces]
        spatial_w = np.exp(-(k * k) / space_denom)
        diff = neighbor - data64
        w = spatial_w * np.exp(-(diff * diff) / color_denom)
        acc += w * neighbor
        wsum += w

    out = acc / wsum
    return out.astype(data.dtype, copy=False)


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

    from scipy.signal import hilbert, medfilt
    from scipy.ndimage import gaussian_filter, laplace, sobel, uniform_filter1d

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
        # Audit #6: vectorized over the whole block — the same math as the old
        # per-column scipy.signal.wiener(col, mysize=7) loop (zero-padded
        # 7-sample windowed mean/variance along axis 0 only, `noise` estimated
        # PER TRACE), without the per-trace Python loop that serialized the
        # worker pool on the GIL. Deliberately NOT wiener(block, (7, 1)): that
        # would estimate `noise` as the mean local variance of the whole
        # worker block, making output depend on how _parallel_apply happened
        # to split the array (worker-count-dependent, non-reproducible).
        def _wiener_block(block: np.ndarray) -> np.ndarray:
            im = block.astype(np.float64)
            k = 7
            l_mean = uniform_filter1d(im, k, axis=0, mode="constant", cval=0.0)
            l_var  = (uniform_filter1d(im * im, k, axis=0, mode="constant",
                                       cval=0.0) - l_mean * l_mean)
            noise  = l_var.mean(axis=0, keepdims=True)         # per trace
            with np.errstate(divide="ignore", invalid="ignore"):
                res  = im - l_mean
                res *= 1.0 - noise / l_var
                res += l_mean
                filt = np.where(l_var < noise, l_mean, res)
            return filt.astype(np.float32)
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

def apply_tvg(data: np.ndarray, alpha: float, dt_us: int,
              threshold_pct: float = 30.0) -> np.ndarray:
    """
    Topography-aware ("Smart") time-variant gain: multiply each sample by
    ``exp(alpha · t_since_seabed)``, where ``t_since_seabed`` is measured from
    each TRACE'S OWN picked water-bottom time — not a single global ``t=0``.
    Above the pick (in the water column) the exponent is clamped to 0, so
    those samples get unity gain instead of being boosted for no reason.

    Water-bottom picking re-uses :func:`apply_water_mute`'s energy-threshold
    logic (first sample whose envelope reaches ``threshold_pct`` % of that
    trace's own peak), with one addition: the envelope is briefly smoothed
    (~0.5 ms rolling RMS) first, so a single noise spike can't fool the pick
    the way it could from raw per-sample amplitude — a lightweight STA-style
    step, not a full STA/LTA ratio test.

    Fallback for noisy/dead traces: if the pick lands on sample 0 (peak at
    the very first sample — a dead trace, or noise dominating the whole
    trace), ``t_since_seabed`` reduces exactly to the legacy global ``t``,
    so that trace transparently falls back to the old globally-ramped
    behaviour instead of producing a nonsensical pick. ``threshold_pct=0``
    disables picking entirely for ALL traces (exact legacy behaviour).

    Parameters
    ----------
    data          : (ns, n_traces) float32 — input NOT mutated
    alpha         : exponential attenuation-compensation coefficient
    dt_us         : sample interval in microseconds
    threshold_pct : seabed-pick threshold, % of per-trace peak envelope
                    (0 ⇒ disable picking, identical to the pre-upgrade
                    global-``t=0`` TVG)

    Returns
    -------
    (ns, n_traces) float32 — new array
    """
    ns = data.shape[0]
    dt_s = dt_us / 1e6
    t_sec = np.arange(ns, dtype=np.float32) * dt_s

    if threshold_pct <= 0.0:
        gain_curve = np.clip(np.exp(alpha * t_sec), 0.0, 1e9)
        return (data * gain_curve[:, np.newaxis]).astype(np.float32)

    from scipy.ndimage import uniform_filter1d

    smooth_n = max(3, int(round(0.0005 / dt_s)))   # ~0.5 ms noise-robust envelope
    if smooth_n % 2 == 0:
        smooth_n += 1
    env = uniform_filter1d(np.abs(data), size=smooth_n, axis=0)

    peak   = np.max(env, axis=0)                       # (n_traces,)
    thresh = (threshold_pct / 100.0) * peak
    exceed = env >= thresh[None, :]
    pick   = np.argmax(exceed, axis=0)                  # (n_traces,); 0 ⇒ fallback

    pick_t_sec = pick.astype(np.float32) * dt_s         # (n_traces,)
    t_since_seabed = np.maximum(t_sec[:, np.newaxis] - pick_t_sec[np.newaxis, :], 0.0)
    gain = np.clip(np.exp(alpha * t_since_seabed), 0.0, 1e9)
    return (data * gain).astype(np.float32)


# ── Spherical divergence correction (deterministic, physics-based gain) ────────

def pick_seabed(data: np.ndarray, threshold_pct: float = 30.0,
                smooth_samples: int = 7) -> np.ndarray:
    """
    Robust, fully-vectorised seabed (first significant energy break) picker.

    Algorithm: MEDIAN-smooth the rectified trace, then — per trace,
    independently — the first sample whose smoothed envelope reaches
    ``threshold_pct`` % of THAT trace's own peak envelope. A relative
    (per-trace) threshold rather than an absolute one, since traces in the
    same chunk can differ wildly in overall amplitude (coupling, range).

    MEDIAN, not a box/mean average: a single impulsive water-column noise
    spike sits in only 1 of ``smooth_samples`` window positions, so the
    median categorically rejects it (it can never become the middle value
    as long as it's a minority of the window) — a mean would instead drag
    the smoothed envelope UP toward the spike's own huge amplitude and could
    still fool the threshold test. Ignoring exactly this kind of outlier is
    the picker's whole job.

    Deliberately expressed purely in SAMPLES, not absolute time/``dt_us``:
    early noise is typically only 1-2 samples wide regardless of sample
    rate, so a small fixed sample-count smoother suppresses it without
    needing the acquisition's sample interval at all — keeping this picker
    a self-contained, reusable building block.

    Fully vectorised: ONE ``median_filter`` pass (axis confined to samples
    via a ``(smooth_samples, 1)`` footprint, C-optimised), one
    max-reduction, one broadcasted comparison, one argmax-reduction — no
    Python-level loop over traces, regardless of ``n_traces``.

    Degenerate traces (no energy ever reaches the threshold — e.g. a fully
    dead/muted channel, or a perfectly flat constant trace) pick index 0:
    callers that build a "time since seabed" gain from this pick then
    transparently fall back to a from-t=0 curve for exactly that trace,
    rather than failing or picking a nonsensical position.

    Parameters
    ----------
    data           : (ns, n_traces) float32 (or any float dtype).
    threshold_pct  : per-trace envelope threshold, as a % of that trace's
                     own peak. Default 30.0.
    smooth_samples : median window width in SAMPLES (coerced to odd,
                     floored at 1). Default 7.

    Returns
    -------
    (n_traces,) int64 — the picked sample index per trace, in ``[0, ns-1]``.
    """
    from scipy.ndimage import median_filter

    smooth_samples = max(1, int(smooth_samples))
    if smooth_samples % 2 == 0:
        smooth_samples += 1

    env = median_filter(np.abs(data), size=(smooth_samples, 1), mode="nearest")
    peak = np.max(env, axis=0)                          # (n_traces,)
    thresh = (threshold_pct / 100.0) * peak
    exceed = env >= thresh[np.newaxis, :]
    pick = np.argmax(exceed, axis=0)                    # 0 ⇒ no break found / fallback
    return pick.astype(np.int64)


def apply_spherical_divergence(data: np.ndarray, exponent: float = 1.0,
                               reference_seabed: bool = True) -> np.ndarray:
    """
    Spherical Divergence Correction (True Amplitude Recovery): a
    DETERMINISTIC, physics-based gain curve ``g(t) = t**exponent`` — the
    deterministic counterpart to AGC's purely STATISTICAL windowed-RMS gain
    (AGC reacts to whatever amplitude each trace happens to have; this
    applies a curve derived from the wavefront geometry, not the data).

    ``exponent=1.0`` is the theoretical pure geometric-spreading loss
    (amplitude falls off ∝ 1/range, so the compensating gain is ∝ range ∝
    time for a roughly constant-velocity medium). ``exponent>1.0`` lets the
    user empirically push the curve further to ALSO compensate for
    inelastic (absorption) attenuation, which spherical spreading alone does
    not model — geophysically, real sub-bottom losses are usually somewhat
    stronger than the pure 1/range law, hence the >1.0 headroom.

    ``reference_seabed`` (the geophysically correct default) — applying the
    gain from a single global t=0 is a flaw: it keeps boosting EVERY trace
    by the same growing curve regardless of where the seafloor actually is,
    so a deep-water trace (where the seabed reflection itself arrives late)
    gets blown out, while a shallow-water trace (seabed arrives early) is
    comparatively washed out, even though physically the gain should always
    be referenced to round-trip distance FROM THE SEAFLOOR'S OWN ARRIVAL,
    not an arbitrary recording-start clock. With ``reference_seabed=True``:
      - :func:`pick_seabed` finds each trace's OWN seabed sample.
      - The water column (every sample ABOVE that pick) gets a flat,
        constant gain — there is no "divergence" to correct above the
        seafloor in this single-bounce model.
      - Every sample AT OR BELOW the pick gets ``(t_since_seabed + 1)
        ** exponent`` — 1-indexed (not 0) for the same reason as the
        legacy curve below: a 0-indexed exponent>0 curve would give an
        exact-zero gain immediately at the seafloor — a discontinuity vs.
        the water column's flat gain just above it, with no physical basis.
    With ``reference_seabed=False``, the legacy single global ``t**exponent``
    curve (1-indexed sample number, identical for every trace) is used
    instead — kept for parity/comparison and as a cheaper, geometry-blind
    fallback when no clear seabed reflection exists in the chunk at all.

    The raw curve is always computed in float64 (``t**exponent`` can reach
    large magnitudes for long traces at the high end of the exponent range,
    and float32 would risk overflow), then NORMALISED before being applied,
    so the result doesn't blow out the display/export's clipping range:
      - Legacy (``reference_seabed=False``): normalised by the WHOLE curve's
        own mean (its only component).
      - Dynamic (``reference_seabed=True``): normalised by the mean of ONLY
        the post-seabed growth values, IGNORING the water column's flat
        entries — a long flat-1.0 water column (e.g. a deep-water trace
        where the seabed pick lands late) would otherwise dilute the mean
        toward 1.0 regardless of how strong the actual sub-seabed growth
        is, defeating the normalisation's purpose. This is a single GLOBAL
        scalar for the whole chunk (not per-trace), so relative amplitude
        relationships between traces — the entire point of "true amplitude"
        recovery — are preserved; only the overall scale is adjusted.

    Safety: for ``exponent >= 0`` and any 1-indexed sample count ``>= 1``,
    every gain value is always ``>= 1``, so every normalising mean is always
    ``>= 1`` — division can never be by zero, by construction (no epsilon
    flooring needed) in either path. ``exponent`` is floored at 0.0 (a
    negative exponent would invert the curve into an attenuation, the
    opposite of this filter's purpose).

    Parameters
    ----------
    data             : (ns, n_traces) float32 (or any float dtype) — NOT
                       mutated.
    exponent         : gain-curve power. ``0.0`` is a no-op (unity gain
                       everywhere, in EITHER path). Default 1.0 (pure
                       geometric spreading).
    reference_seabed : if True (default), reference the gain to each
                       trace's own picked seabed; if False, use the legacy
                       single global curve from sample 0.

    Returns
    -------
    (ns, n_traces) — new array, same dtype as ``data``.
    """
    exponent = max(0.0, float(exponent))
    ns = data.shape[0]
    idx = np.arange(ns, dtype=np.float64)[:, np.newaxis]      # (ns, 1)

    if not reference_seabed:
        t = idx + 1.0                                          # 1-indexed, shared by every trace
        gain = t ** exponent
        gain /= gain.mean()
        out = data.astype(np.float64, copy=False) * gain
        return out.astype(data.dtype, copy=False)

    seabed = pick_seabed(data).astype(np.float64)[np.newaxis, :]   # (1, n_traces)
    below = idx >= seabed                                          # (ns, n_traces) bool
    # 1-indexed depth below the seabed (continuous with the water column's
    # flat gain right at the boundary: t_rel=1 there, so 1**exponent=1).
    # Clamped at 1.0 even ABOVE the seabed purely to keep the (otherwise
    # discarded-by-np.where) power computation from raising a negative base
    # to a fractional exponent there, which would emit a spurious warning.
    t_rel = np.maximum(idx - seabed + 1.0, 1.0)
    gain = np.where(below, t_rel ** exponent, 1.0)

    sub_seabed_mean = gain[below].mean() if np.any(below) else 1.0
    gain /= sub_seabed_mean

    out = data.astype(np.float64, copy=False) * gain
    return out.astype(data.dtype, copy=False)


# ── Log compression (HDR dynamic-range compression) ─────────────────────────────

def apply_log_compression(data: np.ndarray, k: float) -> np.ndarray:
    """
    Phase-preserving logarithmic dynamic-range compression ("Seismic HDR").

    Normalises to [-1, 1] by the array's own peak amplitude so ``k`` behaves
    consistently regardless of the raw SEG-Y amplitude scale, applies
    ``sign(x) * log1p(k*|x|) / log1p(k)`` (sign-preserving, so polarity/phase
    is never altered), then rescales back to the original peak amplitude.

    Parameters
    ----------
    data : (ns, n_traces) float32 — input NOT mutated
    k    : compression strength (k=0 ⇒ no compression; higher k ⇒ stronger
           boost of weak amplitudes relative to strong ones)

    Returns
    -------
    (ns, n_traces) float32 — new array
    """
    max_amp = float(np.max(np.abs(data))) + 1e-12   # prevent division by zero
    norm_data = data / max_amp
    comp_data = np.sign(norm_data) * (np.log1p(k * np.abs(norm_data)) / np.log1p(k))
    return (comp_data * max_amp).astype(np.float32)


# ── CLAHE (adaptive local-contrast HDR) ──────────────────────────────────────────

def apply_clahe(data: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) — 2-D, spatially
    adaptive local contrast enhancement ("Seismic HDR", tile-based).

    Unlike :func:`apply_log_compression` (one GLOBAL gain curve over time),
    this equalises contrast independently within local (time x trace) tiles,
    so a weak reflector sitting next to a strong one is boosted even where a
    single global curve cannot serve both. Phase-preserving: ``sign(data)``
    is carried through untouched; only the rectified magnitude is histogram-
    equalised — same normalise-by-own-peak / restore contract as
    ``apply_log_compression``, so ``clip_limit`` behaves consistently
    regardless of the raw SEG-Y amplitude scale.

    Requires ``opencv-python-headless`` (``cv2``), imported lazily here so the
    rest of the core stays importable without it (same discipline as the
    lazy matplotlib import in ``coloring.py``).

    Parameters
    ----------
    data       : (ns, n_traces) float32 — input NOT mutated
    clip_limit : OpenCV CLAHE clip limit (contrast strength; higher = stronger,
                 also more noise amplification)
    tile_grid  : side length, in tiles, of the square local window (e.g. 8 means
                 an 8x8 grid of equalization tiles across the array)

    Returns
    -------
    (ns, n_traces) float32 — new array
    """
    import cv2

    sign = np.sign(data)
    mag = np.abs(data)
    max_amp = float(np.max(mag)) + 1e-12             # prevent division by zero

    norm_u16 = np.ascontiguousarray((mag / max_amp * 65535.0).astype(np.uint16))
    tiles = max(1, int(tile_grid))
    clahe = cv2.createCLAHE(clipLimit=max(0.01, float(clip_limit)),
                            tileGridSize=(tiles, tiles))
    eq_u16 = clahe.apply(norm_u16)

    out = sign * (eq_u16.astype(np.float32) / 65535.0) * max_amp
    return out.astype(np.float32)


# ── Despike (impulsive-noise removal) ────────────────────────────────────────────

def apply_despike(data: np.ndarray, window_size: int, threshold: float) -> np.ndarray:
    """
    Impulsive-noise (spike) removal via a robust rolling-median filter.

    For each trace independently, along time: a rolling MEDIAN gives a
    spike-free local baseline; the rolling MEDIAN ABSOLUTE DEVIATION (MAD)
    around that baseline gives a robust local noise-scale estimate — unlike
    a rolling std, a single huge spike can't blow up the very statistic
    meant to detect it. Any sample whose deviation from the local median
    exceeds ``threshold`` times the local robust standard deviation
    (``1.4826 * MAD``, the usual MAD→std factor for Gaussian noise) is
    replaced by the local median; everything else passes through unchanged.

    Parameters
    ----------
    data        : (ns, n_traces) float32 — input NOT mutated
    window_size : rolling window length in SAMPLES (forced odd internally so
                  the window is symmetric)
    threshold   : spike threshold, in multiples of the local robust standard
                  deviation (lower ⇒ more aggressive despiking)

    Returns
    -------
    (ns, n_traces) float32 — new array
    """
    from scipy.ndimage import median_filter

    win = max(3, int(window_size))
    if win % 2 == 0:
        win += 1

    baseline  = median_filter(data, size=(win, 1), mode="nearest")
    deviation = np.abs(data - baseline)
    mad       = median_filter(deviation, size=(win, 1), mode="nearest")
    robust_std = 1.4826 * mad

    spike_mask = deviation > (threshold * np.maximum(robust_std, 1e-9))
    out = np.where(spike_mask, baseline, data)
    return out.astype(np.float32)


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

    Stage 0 — mandatory DC-offset removal (apply_dc_removal) — already ran on
    ``obj.data`` at load time (see io_segy._populate_profile_from_file), so it
    is NOT repeated here; every stage below already sees a zero-centred trace.

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
