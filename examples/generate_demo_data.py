"""
generate_demo_data.py — Generate synthetic SEG-Y demo data for examples.

Creates:
  examples/_demo_in/demo_single.sgy      — single profile (decimal degrees)
  examples/_demo_in/demo_arcsec.sgy      — arc-second coords
  examples/_demo_in/demo_chain_1.sgy     — first of a contiguous pair
  examples/_demo_in/demo_chain_2.sgy     — second of a contiguous pair
  examples/_demo_in/demo_spectrum.sgy    — known 2 kHz sinusoid

Run from repo root:
    python examples/generate_demo_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.make_synthetic_segy import make_synthetic_segy, make_chain_pair

OUT_DIR = Path(__file__).parent / "_demo_in"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing demo data to {OUT_DIR}/")

    # Smooth curved trackline. A fine coordinate scalar (1e-4° grid — the finest
    # that still fits the 2-byte SourceGroupScalar field, |s| ≤ 32767) keeps every
    # trace on a DISTINCT position. Coarse scalars quantise small steps onto the
    # same stored integer, collapsing traces into overlapping plateaus that make
    # the map's click-to-jump ambiguous.
    make_synthetic_segy(
        str(OUT_DIR / "demo_single.sgy"),
        n_traces=80, ns=512, dt_us=250,
        scalar_coord=-10000, coord_unit=3,
        base_lon=-8.5, base_lat=43.2,
        arc_deg=60.0, arc_radius_deg=0.08, water_depth=150.0,
        delay_ms=5,
    )
    print("  ✔ demo_single.sgy")

    # Diagonal (both lon AND lat advance) so the track isn't a degenerate line.
    make_synthetic_segy(
        str(OUT_DIR / "demo_arcsec.sgy"),
        n_traces=60, ns=256, dt_us=500,
        scalar_coord=-100, coord_unit=2,
        base_lon=-8.3, base_lat=43.1,
        lon_step=0.001, lat_step=0.0006,
    )
    print("  ✔ demo_arcsec.sgy")

    make_chain_pair(
        str(OUT_DIR), prefix="demo_chain",
        n_traces=60, ns=256, dt_us=250, gap_km=0.08,
    )
    print("  ✔ demo_chain_1.sgy  demo_chain_2.sgy")

    make_synthetic_segy(
        str(OUT_DIR / "demo_spectrum.sgy"),
        n_traces=30, ns=1024, dt_us=125,
        inject_freq_hz=2500.0,
        water_depth=80.0,
    )
    print("  ✔ demo_spectrum.sgy")

    print(f"\nDone — {len(list(OUT_DIR.glob('*.sgy')))} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
