"""
coloring.py — Colormapped RGBA conversion for seismic data.

GUI independence contract
-------------------------
matplotlib is imported lazily inside _get_colormap() only — never at module
level. This keeps core/ importable in headless/no-display environments
without triggering any matplotlib backend initialization.

Public API
----------
colormapped_rgba(data, cmap_name, vmin, vmax) → np.ndarray (ns, n_tr, 4) uint8

Two-track
---------
Reference path: _ref_colormapped_rgba (NumPy normalisation, lazy cmap).
No optimized path implemented; GPU normalisation preserved from monolith
as an opportunistic branch (not a registered optimization).
"""
from __future__ import annotations

import concurrent.futures as _cf
from typing import Optional

import numpy as np

from ._backends import GPU as _GPU, N_WORKERS as _N_WORKERS


# ── Lazy colormap helper ───────────────────────────────────────────────────────

def _get_colormap(name: str):
    """
    Return a matplotlib colormap compatible with all matplotlib versions.
    matplotlib is imported lazily here — no GUI backend is initialised.
    """
    import matplotlib  # noqa: PLC0415  (lazy import, intentional)
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return matplotlib.cm.get_cmap(name)


def _render_rgba_tile_norm(tile_norm: np.ndarray, cmap_name: str) -> np.ndarray:
    """Colorise a [0,1]-normalised tile → RGBA uint8."""
    cmap = _get_colormap(cmap_name)
    return (cmap(tile_norm) * 255).astype(np.uint8)


# ── Reference implementation ───────────────────────────────────────────────────

def _ref_colormapped_rgba(
    data: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    n_workers: int = _N_WORKERS,
) -> np.ndarray:
    """
    Reference colormapped RGBA conversion.

    GPU normalisation is an opportunistic branch carried from the monolith —
    it is NOT a registered optimization (no regression gate).

    Path: reference.
    """
    ns, n_traces = data.shape

    # GPU normalisation (opportunistic, not a registered optimization)
    data_norm: Optional[np.ndarray] = None
    if _GPU:
        import cupy as cp
        try:
            g         = cp.asarray(data, dtype=cp.float32)
            lo        = cp.float32(vmin)
            hi        = cp.float32(vmax) if vmax != vmin else cp.float32(vmin + 1e-9)
            data_norm = cp.asnumpy(cp.clip((g - lo) / (hi - lo), 0.0, 1.0))
        except Exception:
            data_norm = None

    if data_norm is None:
        lo        = float(vmin)
        hi        = float(vmax) if vmax != vmin else vmin + 1e-9
        data_norm = np.clip((data.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)

    if n_traces < 128 or n_workers <= 1:
        cmap = _get_colormap(cmap_name)
        return (cmap(data_norm) * 255).astype(np.uint8)

    chunk_size = max(1, n_traces // n_workers)
    slices     = [slice(j, min(j + chunk_size, n_traces))
                  for j in range(0, n_traces, chunk_size)]
    tiles      = [data_norm[:, sl] for sl in slices]
    results    = [None] * len(tiles)

    with _cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_render_rgba_tile_norm, t, cmap_name): k
                for k, t in enumerate(tiles)}
        for fut in _cf.as_completed(futs):
            results[futs[fut]] = fut.result()

    return np.concatenate(results, axis=1)


def colormapped_rgba(
    data: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    n_workers: int = _N_WORKERS,
) -> np.ndarray:
    """
    Convert a 2-D seismic data matrix to a colormapped RGBA image.

    Parameters
    ----------
    data      : (ns, n_traces) float32
    cmap_name : matplotlib colormap name (may include "_r" suffix)
    vmin/vmax : clip range for normalisation
    n_workers : thread-pool size for parallel tile colorisation

    Returns
    -------
    (ns, n_traces, 4) uint8  — RGBA image

    Path: reference (_ref_colormapped_rgba).
    """
    return _ref_colormapped_rgba(data, cmap_name, vmin, vmax, n_workers)
