"""
example_spectrum.py — Compute and export the frequency spectrum.

Run from repo root:
    python examples/generate_demo_data.py
    python examples/example_spectrum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sbp_studio.core import load_profile, compute_spectrum
from sbp_studio.viz import render_spectrum_figure, save_figure

DATA_DIR = Path(__file__).parent / "_demo_in"
OUT_DIR  = Path(__file__).parent / "_demo_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sd  = load_profile(str(DATA_DIR / "demo_spectrum.sgy"))
    fs  = 1e6 / sd.dt_us
    d   = np.nan_to_num(sd.data, nan=0.0)

    print(f"Computing spectrum for {sd.name}…")
    sp = compute_spectrum(d, fs)
    print(f"  peak_hz     : {sp.peak_hz:.1f} Hz")
    print(f"  centroid_hz : {sp.centroid_hz:.1f} Hz")
    print(f"  BW -3 dB    : {sp.bw_3db_lo:.0f}–{sp.bw_3db_hi:.0f} Hz")
    print(f"  SNR         : {sp.snr_db:.1f} dB")

    fig = render_spectrum_figure(sp, fs, sd.name, sd.n_traces, sd.dist_km)
    out = str(OUT_DIR / "demo_spectrum.png")
    save_figure(fig, out, dpi=150)
    print(f"  ✔ {out}")


if __name__ == "__main__":
    main()
