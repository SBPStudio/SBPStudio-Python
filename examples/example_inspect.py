"""
example_inspect.py — Load and inspect a demo SEG-Y profile.

Run from repo root after generating demo data:
    python examples/generate_demo_data.py
    python examples/example_inspect.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sbp_studio.core import load_metadata, load_profile, detect_chains

DATA_DIR = Path(__file__).parent / "_demo_in"


def main() -> None:
    segy = str(DATA_DIR / "demo_single.sgy")
    print(f"=== Metadata only (no trace read) ===")
    md = load_metadata(segy)
    print(json.dumps(md.to_dict(), indent=2, default=str))

    print(f"\n=== Full profile ===")
    sd = load_profile(segy)
    print(sd.summary())
    print(f"  data shape : {sd.data.shape}")
    print(f"  dist range : {sd.dist_km[0]:.3f} – {sd.dist_km[-1]:.3f} km")
    print(f"  lon range  : {sd.lons[0]:.6f} – {sd.lons[-1]:.6f} °")
    print(f"  delay range: {sd.min_delay:.0f} – {sd.max_delay:.0f} ms")
    print(f"  clip p99   : {sd.clip_p99:.4f}")
    print(f"  CRS        : {sd.detected_crs}")

    print(f"\n=== Chain detection (two-file chain) ===")
    p1 = str(DATA_DIR / "demo_chain_1.sgy")
    p2 = str(DATA_DIR / "demo_chain_2.sgy")
    profiles = [load_profile(p1), load_profile(p2)]
    chains   = detect_chains(profiles, gap_km=1.0)
    for ch in chains:
        print(f"  {ch.label}  ({ch.total_km:.2f} km, {ch.n_traces} traces)")


if __name__ == "__main__":
    main()
