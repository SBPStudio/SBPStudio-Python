"""
example_reproject.py — Reproject a single profile and join a chain.

Run from repo root:
    python examples/generate_demo_data.py
    python examples/example_reproject.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from topassuite.core import (
    load_profile, detect_chains,
    reproject_one, reproject_chain,
)

DATA_DIR = Path(__file__).parent / "_demo_in"
OUT_DIR  = Path(__file__).parent / "_demo_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(msg)


def main() -> None:
    # ── Reproject single profile ──────────────────────────────────────────
    src = str(OUT_DIR / "rep_single.sgy")
    shutil.copy(str(DATA_DIR / "demo_single.sgy"), src)
    sd  = load_profile(src)

    print("Reprojecting single profile (WGS84 → UTM 30N)…")
    out = reproject_one(sd, "EPSG:4326", "EPSG:32630", log=_log)
    print(f"  ✔ {Path(out).name}")

    # ── Join chain ────────────────────────────────────────────────────────
    c1 = str(OUT_DIR / "join_c1.sgy")
    c2 = str(OUT_DIR / "join_c2.sgy")
    shutil.copy(str(DATA_DIR / "demo_chain_1.sgy"), c1)
    shutil.copy(str(DATA_DIR / "demo_chain_2.sgy"), c2)

    profiles = [load_profile(c1), load_profile(c2)]
    chains   = detect_chains(profiles, gap_km=1.0)
    print(f"\nJoining chain: {chains[0].label}")
    out = reproject_chain(chains[0], "EPSG:4326", "EPSG:32630", log=_log)
    print(f"  ✔ {Path(out).name}")


if __name__ == "__main__":
    main()
