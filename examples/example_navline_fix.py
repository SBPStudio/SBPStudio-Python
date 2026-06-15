"""
example_navline_fix.py — Export navline and FIX marks to GeoJSON and CSV.

Run from repo root:
    python examples/generate_demo_data.py
    python examples/example_navline_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sbp_studio.core import (
    load_profile, compute_fix_positions,
    write_navline_geojson, write_navline_csv,
    write_fix_points_geojson, write_fix_points_csv,
)

DATA_DIR = Path(__file__).parent / "_demo_in"
OUT_DIR  = Path(__file__).parent / "_demo_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sd = load_profile(str(DATA_DIR / "demo_single.sgy"))

    # ── Navline ───────────────────────────────────────────────────────────
    write_navline_geojson(
        str(OUT_DIR / "navline.geojson"),
        sd.lons, sd.lats, sd.dist_km, sd.water_depth, sd.timestamps,
        include_attrs=True,
    )
    print("  ✔ navline.geojson")

    write_navline_csv(
        str(OUT_DIR / "navline.csv"),
        sd.lons, sd.lats, sd.dist_km, sd.water_depth, sd.timestamps,
    )
    print("  ✔ navline.csv")

    # ── FIX marks ─────────────────────────────────────────────────────────
    fixes = compute_fix_positions(sd.timestamps, sd.dist_km, sd.lons, sd.lats, 1)
    print(f"  {len(fixes)} FIX marks at 1-minute interval")

    if fixes:
        write_fix_points_geojson(str(OUT_DIR / "fixes.geojson"), fixes)
        print("  ✔ fixes.geojson")

        write_fix_points_csv(str(OUT_DIR / "fixes.csv"), fixes)
        print("  ✔ fixes.csv")
    else:
        print("  (no FIX marks — profile timestamps too close)")


if __name__ == "__main__":
    main()
