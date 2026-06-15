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
    spec_2d_db:   np.ndarray   # (n_freqs, n_traces) — per-trace normalised dB
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

    pwr = np.zeros((len(freqs), n_tr), dtype=np.float64)
    for k in range(n_frames):
        seg  = data[k * step: k * step + nfft, :] * win[:, None]
        pwr += np.abs(np.fft.rfft(seg, axis=0)) ** 2
    pwr /= n_frames

    pwr_norm   = pwr / (pwr.max(axis=0, keepdims=True) + 1e-30)
    spec_2d_db = 10 * np.log10(pwr_norm + 1e-30)

    pwr_mean     = pwr.mean(axis=1)
    ref          = pwr_mean.max() + 1e-30
    spec_mean_db = 10 * np.log10(pwr_mean / ref)

    pwr_p10 = np.percentile(pwr, 10, axis=1)
    pwr_p50 = np.percentile(pwr, 50, axis=1)
    pwr_p90 = np.percentile(pwr, 90, axis=1)
    spec_p10_db = 10 * np.log10(pwr_p10 / ref)
    spec_p50_db = 10 * np.log10(pwr_p50 / ref)
    spec_p90_db = 10 * np.log10(pwr_p90 / ref)

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

    sig_mask   = (freqs >= 500) & (freqs <= 15000)
    noise_mask = freqs > 15000
    sig_pwr    = pwr_mean[sig_mask].mean()   if sig_mask.any()   else 1e-30
    noise_pwr  = pwr_mean[noise_mask].mean() if noise_mask.any() else 1e-30
    snr_db     = float(10 * np.log10(sig_pwr / (noise_pwr + 1e-30)))

    # Per-band energy distribution (% of total) over the SBP bands of interest.
    nyq          = fs / 2.0
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
