"""
example_export_image.py — Export a seismic profile image and a chain image.

Run from repo root:
    python examples/generate_demo_data.py
    python examples/example_export_image.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sbp_studio.core import load_profile, detect_chains, process_profile_data, process_chain_data
from sbp_studio.viz import render_profile_figure, render_chain_figure, save_figure

DATA_DIR = Path(__file__).parent / "_demo_in"
OUT_DIR  = Path(__file__).parent / "_demo_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = dict(
    decon=False, decon_op=10.0, decon_gap=1.0, decon_wn=0.1,
    filt=True, flo=200.0, fhi=4000.0,
    preset="── Sin filtro preestablecido ──",
    tvg=False, tvg_alpha=0.0,
    agc=True, agc_win=50.0,
    align=False,
    clip=99.0, cmap="Viridis", inv_cmap=False,
    fix=True, fix_iv=1,
)


def main() -> None:
    # ── Single profile ────────────────────────────────────────────────────
    print("Exporting single profile image…")
    sd   = load_profile(str(DATA_DIR / "demo_single.sgy"))
    data = process_profile_data(sd, PARAMS)
    fig  = render_profile_figure(sd, data, PARAMS, figsize=(12, 6), dpi=150)
    save_figure(fig, str(OUT_DIR / "demo_single.png"))
    print(f"  ✔ {OUT_DIR / 'demo_single.png'}")

    # ── Chain ─────────────────────────────────────────────────────────────
    print("Exporting chain image…")
    p1, p2 = [str(DATA_DIR / f"demo_chain_{i}.sgy") for i in (1, 2)]
    profiles = [load_profile(p1), load_profile(p2)]
    chains   = detect_chains(profiles, gap_km=1.0)
    for ch in chains:
        data = process_chain_data(ch, PARAMS)
        fig  = render_chain_figure(ch, data, PARAMS, figsize=(14, 6), dpi=150)
        out  = str(OUT_DIR / f"demo_chain.png")
        save_figure(fig, out)
        print(f"  ✔ {out}")


if __name__ == "__main__":
    main()
