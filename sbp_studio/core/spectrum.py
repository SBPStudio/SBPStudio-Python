"""
spectrum.py — Frequency spectrum computation.

Public API
----------
compute_spectrum(data, fs) → SpectrumResult

The reference implementation (_ref_compute_spectrum) is ported verbatim
from TopasSUITE._compute_spectrum (~L2709). The public function dispatches
to it directly (no optimized path implemented yet).

Array contract
--------------
- Input data  : (ns, n_traces) float32 or float64
- freqs       : (n_freqs,) float64 — Hz
- spec_*_db   : (n_freqs,) float64 — dB
- spec_2d_db  : (n_freqs, n_traces) float64 — per-trace spectrum in dB
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# SBP bands of interest for the per-band energy distribution (Hz). The last
# band's upper edge is replaced with Nyquist at runtime; bands above Nyquist are
# dropped. Mirrors TopasSUITE._draw_spectrum_figure.
TOPAS_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("< 1 kHz",   0.0,     1000.0),
    ("1–2 kHz",   1000.0,  2000.0),
    ("2–4 kHz",   2000.0,  4000.0),
    ("4–7 kHz",   4000.0,  7000.0),
    ("7–10 kHz",  7000.0,  10000.0),
    ("10–15 kHz", 10000.0, 15000.0),
    ("> 15 kHz",  15000.0, np.inf),
)

# Audit #4: column budget for the percentile pass. np.percentile promotes its
# working copy to float64 (an unbounded 2×-pwr transient — ~820 MB at 50k
# traces) and partitions every row; above this many traces a uniform column
# subsample is used instead, bounding the transient at ~134 MB. 8192 samples
# per frequency bin keep the envelope curves visually identical: measured
# worst case is the p10 of a single-frame white-noise Welch (its power is
# exponential-distributed, the steepest dB quantile) at ≈0.08 dB mean /
# ≲0.5 dB worst-bin jitter — under the plotted line width on a 40+ dB scale.
_PCT_MAX_COLS = 8192


@dataclass
class SpectrumResult:
    """
    Output of compute_spectrum.

    All spectrum arrays are in dB relative to the per-ensemble maximum.
    spec_2d_db is normalised per trace (each column relative to its own max).
    """
    freqs:        np.ndarray   # (n_freqs,) float64  — Hz
    nfft:         int
    n_frames:     int
    low_res:      bool         # True when nfft < 512 (short profile)
    spec_mean_db: np.ndarray   # (n_freqs,) — Welch mean in dB
    spec_p10_db:  np.ndarray   # (n_freqs,) — 10th percentile
    spec_p50_db:  np.ndarray   # (n_freqs,) — median
    spec_p90_db:  np.ndarray   # (n_freqs,) — 90th percentile
    spec_2d_db:     np.ndarray  # (n_freqs, n_traces) — per-trace normalised dB
    # (n_traces,) float32 — per-trace dB offset, 10·log10(col_max/ensemble_max).
    # Audit #4: replaces the stored spec_2d_db_global matrix (which doubled the
    # spectrum RAM for a view most sessions never open); see the property below.
    spec_db_offset: np.ndarray
    peak_hz:      float        # frequency of max power (Hz)
    centroid_hz:  float        # spectral centroid (Hz)
    bw_3db_lo:    float        # -3 dB bandwidth lower edge (Hz)
    bw_3db_hi:    float        # -3 dB bandwidth upper edge (Hz)
    bw_6db_lo:    float        # -6 dB bandwidth lower edge (Hz)
    bw_6db_hi:    float        # -6 dB bandwidth upper edge (Hz)
    roll_off_hz:  float        # frequency where 85% of cumulative energy reached (Hz)
    snr_db:       float        # estimated SNR: 0.5–15 kHz vs >15 kHz (dB)
    band_labels:  List[str]    = field(default_factory=list)   # SBP energy bands
    band_pcts:    List[float]  = field(default_factory=list)   # % of total energy per band

    @property
    def spec_2d_db_global(self) -> np.ndarray:
        """Ensemble-max normalised dB matrix (#16), materialized lazily:
        per-trace dB + per-trace offset. Identical to 10·log10(pwr/global_max)
        wherever power is non-zero; only exact-zero floor bins (≤ −300 dB,
        far below any display clamp) differ by the offset. Deliberately NOT
        cached — the add is memory-bandwidth cheap and caching would restore
        the two-matrices-resident footprint this replaces (audit #4)."""
        return self.spec_2d_db + self.spec_db_offset[None, :]


def _ref_compute_spectrum(data: np.ndarray, fs: float) -> SpectrumResult:
    """
    Reference spectrum computation — ported verbatim from
    TopasSUITE._compute_spectrum (~L2709).

    Path: reference.
    Backend: NumPy FFT (scipy.signal.windows.hann for the window).

    Parameters
    ----------
    data : (ns, n_traces) float32/64
    fs   : sample rate in Hz (= 1e6 / dt_us)
    """
    from scipy.signal.windows import hann

    ns, n_tr = data.shape
    nfft     = min(4096, max(512, 2 ** int(np.ceil(np.log2(ns)))))
    nfft     = min(nfft, ns)
    low_res  = nfft < 512

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    win   = hann(nfft)

    step     = nfft // 2
    n_frames = max(1, (ns - nfft) // step + 1)

    # Cast to float32 early: the Welch frame loop otherwise builds a complex128
    # intermediate (nfft × n_traces) per frame, spiking RAM by ~1.5 GB on wide
    # profiles (50k traces, nfft 4096). float32 input → complex64 rfft → float32
    # power accumulator halves both the per-frame spike and the pwr matrix.
    data_f32 = np.ascontiguousarray(data, dtype=np.float32)
    win_f32  = win.astype(np.float32)

    pwr = np.zeros((len(freqs), n_tr), dtype=np.float32)
    for k in range(n_frames):
        seg  = data_f32[k * step: k * step + nfft, :] * win_f32[:, None]
        pwr += np.abs(np.fft.rfft(seg, axis=0)) ** 2
    pwr /= n_frames

    # #18: true one-sided Welch PSD normalization.
    # Dividing by (fs × Σwin²) converts raw |FFT|² accumulator into
    # physical power-spectral-density (u²/Hz). One-sided spectrum: all
    # interior bins (neither DC nor Nyquist) are doubled because the two-
    # sided energy folds onto them symmetrically.
    _psd_norm = np.float32(float(fs) * float(np.sum(win_f32 ** 2)))
    if _psd_norm > 0:
        pwr /= _psd_norm
        pwr[1:-1] *= np.float32(2.0)

    col_max    = pwr.max(axis=0, keepdims=True) + np.float32(1e-30)
    pwr_norm   = pwr / col_max
    spec_2d_db = (10 * np.log10(pwr_norm + np.float32(1e-30))).astype(np.float32)

    # #16 + audit #4: the global-normalization variant (single ensemble max,
    # preserving relative amplitude across ALL traces) is now DERIVED, not
    # stored: global dB = per-trace dB + 10·log10(col_max/global_max). A
    # (n_traces,) offset vector replaces the second full (n_freqs, n_traces)
    # matrix — see SpectrumResult.spec_2d_db_global.
    global_max     = pwr.max() + np.float32(1e-30)
    spec_db_offset = (10 * np.log10(col_max[0] / global_max)).astype(np.float32)

    # Upcast only the small per-frequency mean (n_freqs elements, not n_traces)
    # to float64 for the downstream statistics; pwr stays float32.
    pwr_mean     = pwr.mean(axis=1).astype(np.float64)
    ref          = pwr_mean.max() + 1e-30
    spec_mean_db = 10 * np.log10(pwr_mean / ref)

    # Single 3-quantile pass — 3× cheaper than three separate percentile calls
    # over the (n_freqs, n_traces) pwr matrix. Audit #4: above _PCT_MAX_COLS
    # traces, run it on a uniform column subsample — np.percentile promotes
    # its working copy to float64, so this bounds a 2×-pwr transient while the
    # curves stay statistically indistinguishable.
    if n_tr > _PCT_MAX_COLS:
        pct_src = pwr[:, np.linspace(0, n_tr - 1, _PCT_MAX_COLS).astype(np.intp)]
    else:
        pct_src = pwr
    pwr_p10, pwr_p50, pwr_p90 = np.percentile(pct_src, [10, 50, 90], axis=1)
    spec_p10_db = 10 * np.log10(pwr_p10.astype(np.float64) / ref)
    spec_p50_db = 10 * np.log10(pwr_p50.astype(np.float64) / ref)
    spec_p90_db = 10 * np.log10(pwr_p90.astype(np.float64) / ref)

    peak_hz     = float(freqs[np.argmax(pwr_mean)])

    total_pwr   = pwr_mean.sum() + 1e-30
    centroid_hz = float((freqs * pwr_mean).sum() / total_pwr)

    def _bandwidth(pwr_arr, freqs_arr, db_drop):
        thresh = pwr_arr.max() / (10 ** (db_drop / 10))
        above  = freqs_arr[pwr_arr >= thresh]
        if above.size < 2:
            if above.size == 0:
                return float(freqs_arr[0]), float(freqs_arr[-1])
            return float(above[0]), float(above[0])
        return float(above[0]), float(above[-1])

    bw3_lo, bw3_hi = _bandwidth(pwr_mean, freqs, 3.0)
    bw6_lo, bw6_hi = _bandwidth(pwr_mean, freqs, 6.0)

    cum_energy   = np.cumsum(pwr_mean)
    cum_energy  /= cum_energy[-1] + 1e-30
    roll_off_idx = np.searchsorted(cum_energy, 0.85)
    roll_off_hz  = float(freqs[min(roll_off_idx, len(freqs) - 1)])

    # #17: Nyquist-aware SNR — gate the noise band so it never extends past
    # 85 % of Nyquist (avoids aliasing artefacts at the band edge).
    # When Nyquist is below ~17.6 kHz the noise band becomes empty; fall
    # back to a bandwidth-relative SNR (lower-75% vs upper-25%) so the
    # metric is always meaningful, guarded by noise_mask.any().
    nyq            = fs / 2.0
    noise_floor_hz = min(15000.0, nyq * 0.85)
    sig_mask   = (freqs >= 500) & (freqs <= noise_floor_hz)
    noise_mask = freqs > noise_floor_hz
    sig_pwr    = pwr_mean[sig_mask].mean() if sig_mask.any() else 1e-30
    if noise_mask.any():
        noise_pwr = pwr_mean[noise_mask].mean()
    else:
        # Nyquist too low for a dedicated noise band: relative SNR fallback —
        # signal = lower 75% of the available bandwidth, noise = upper 25%.
        bw_split   = float(freqs[0]) + 0.75 * (float(freqs[-1]) - float(freqs[0]))
        sig_mask_r = freqs <= bw_split
        nse_mask_r = freqs >  bw_split
        sig_pwr    = pwr_mean[sig_mask_r].mean() if sig_mask_r.any() else 1e-30
        noise_pwr  = pwr_mean[nse_mask_r].mean() if nse_mask_r.any() else 1e-30
    snr_db = float(10 * np.log10(sig_pwr / (noise_pwr + 1e-30)))

    # Per-band energy distribution (% of total) over the SBP bands of interest.
    pwr_mean_lin = 10.0 ** (spec_mean_db / 10.0)
    band_labels: list = []
    band_powers: list = []
    for label, flo, fhi in TOPAS_BANDS:
        fhi_eff = min(fhi, nyq)
        mask = (freqs >= flo) & (freqs < fhi_eff)
        if mask.any() and flo < nyq:
            band_labels.append(label)
            band_powers.append(float(pwr_mean_lin[mask].sum()))
    total_bp  = sum(band_powers) + 1e-30
    band_pcts = [100.0 * p / total_bp for p in band_powers]

    return SpectrumResult(
        freqs=freqs, nfft=nfft, n_frames=n_frames, low_res=low_res,
        spec_mean_db=spec_mean_db,
        spec_p10_db=spec_p10_db,
        spec_p50_db=spec_p50_db,
        spec_p90_db=spec_p90_db,
        spec_2d_db=spec_2d_db,
        spec_db_offset=spec_db_offset,
        peak_hz=peak_hz,
        centroid_hz=centroid_hz,
        bw_3db_lo=bw3_lo, bw_3db_hi=bw3_hi,
        bw_6db_lo=bw6_lo, bw_6db_hi=bw6_hi,
        roll_off_hz=roll_off_hz,
        snr_db=snr_db,
        band_labels=band_labels,
        band_pcts=band_pcts,
    )


def compute_spectrum(data: np.ndarray, fs: float) -> SpectrumResult:
    """
    Compute the Welch power spectrum of a seismic data matrix.

    Parameters
    ----------
    data : (ns, n_traces) float32 or float64
           NaN values should be zeroed before calling (nan_to_num).
    fs   : sample rate in Hz (= 1e6 / dt_us)

    Returns
    -------
    SpectrumResult dataclass — all arrays are float64.

    Path: reference (_ref_compute_spectrum).
    No optimized path implemented; regression gate not applicable.
    """
    return _ref_compute_spectrum(data, fs)
