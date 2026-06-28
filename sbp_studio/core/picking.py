"""
picking.py — Interpretation pick-point data model, persistence, and export.

A "pick" is a single user-placed interpretation marker on a seismic section:
a (trace, time) position plus a free-text description. This module is the
headless, GUI-free core for that feature — the GUI (SeismicView) owns the
interactive placement/drawing; this module owns the data structure and every
on-disk format.

Formats
-------
- .tps (JSON): the INTERNAL format. Round-trips every field exactly
  (id, trace_index, time_ms, x_coord, y_coord, description), so a re-import
  redraws points on the pyqtgraph time/trace grid byte-identically. This is
  the ONLY format ``load_picks_tps`` reads — CSV/GeoJSON/SHP are one-way
  exports for external GIS tools, not re-importable (they drop trace_index/
  time_ms, which a GIS has no use for and no way to express).
- .csv / .geojson / .shp: standard GIS exports (id, x, y, description) via
  ``core.geometry_export``'s pick writers — same pure-stdlib methodology as
  the existing FIX-mark writers (no new GIS dependency).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class PickPoint:
    """One interpretation marker.

    ``id`` is a sequential, never-reused integer (assigned by the caller —
    see SeismicView's monotonically-increasing counter, which survives
    toggling picking mode off and on). ``trace_index``/``time_ms`` are the
    pyqtgraph-grid position (trace column, time row in ms) — what actually
    redraws the point. ``x_coord``/``y_coord`` are the resolved spatial
    position (see :func:`resolve_pick_coords`), carried along for GIS export
    only; they play no part in redrawing the point on the seismic grid.
    """
    id: int
    trace_index: int
    time_ms: float
    x_coord: float
    y_coord: float
    description: str = ""


def resolve_pick_coords(obj: Any, trace_index: int) -> Tuple[float, float]:
    """Resolve the (x, y) spatial coordinate for one trace of the active
    profile/chain, reading directly from the SEG-Y-derived ``lons``/``lats``
    arrays — the SAME raw per-trace coordinates the existing FIX-mark export
    already reads (see ``geometry_export.compute_fix_positions``).

    Falls back to the trace number itself (as x, with y=0.0) ONLY when the
    coordinate is completely missing or exactly (0, 0)/non-finite — a real
    recorded navigation fix is never exactly (0, 0), so that pair is the
    reliable signature of "no navigation in this file," not a real position
    that happens to sit on the equator/prime-meridian.
    """
    lons = getattr(obj, "lons", None)
    lats = getattr(obj, "lats", None)
    if (lons is not None and lats is not None
            and 0 <= trace_index < len(lons) and 0 <= trace_index < len(lats)):
        x, y = float(lons[trace_index]), float(lats[trace_index])
        if math.isfinite(x) and math.isfinite(y) and (x != 0.0 or y != 0.0):
            return x, y
    return float(trace_index), 0.0


# ── Internal format (.tps, JSON) — the only round-trippable format ─────────────

def save_picks_tps(path: str, picks: List[PickPoint]) -> None:
    """Write every field of every pick to a JSON file — the internal format,
    the only one :func:`load_picks_tps` can read back."""
    out_path = path if path.lower().endswith(".tps") else path + ".tps"
    data = {"version": 1, "picks": [asdict(p) for p in picks]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_picks_tps(path: str) -> List[PickPoint]:
    """Read picks back from the internal .tps JSON format.

    Hardened against a malformed/hand-edited file: each entry is parsed
    independently and a bad one is SKIPPED (not fatal) — mirrors this
    codebase's existing "harden preset load against malformed JSON"
    convention (gui/components/pipeline_panel.py's preset loader). A
    completely unreadable file raises (there is nothing safe to return)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("picks", []) if isinstance(data, dict) else data
    picks: List[PickPoint] = []
    if not isinstance(raw, list):
        return picks
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            picks.append(PickPoint(
                id=int(item["id"]),
                trace_index=int(item["trace_index"]),
                time_ms=float(item["time_ms"]),
                x_coord=float(item["x_coord"]),
                y_coord=float(item["y_coord"]),
                description=str(item.get("description", "")),
            ))
        except (KeyError, TypeError, ValueError):
            continue   # skip this one malformed entry, keep the rest
    return picks


# ── GIS export dispatch (CSV / GeoJSON / SHP, by file extension) ───────────────

def export_picks(path: str, picks: List[PickPoint]) -> str:
    """Write ``picks`` in whichever GIS format ``path``'s extension implies
    (.csv / .geojson / .shp, default .shp) — mirrors the existing FIX-mark
    export's dispatch-by-extension convention. Returns the path actually
    written (writers append their own extension if missing)."""
    from .geometry_export import write_picks_csv, write_picks_geojson, write_picks_shp

    points = [(p.id, p.x_coord, p.y_coord, p.description) for p in picks]
    low = path.lower()
    if low.endswith((".geojson", ".json")):
        write_picks_geojson(path, points)
        return path if low.endswith(".geojson") else path + ".geojson"
    if low.endswith(".csv"):
        write_picks_csv(path, points)
        return path
    write_picks_shp(path, points)
    return path if low.endswith(".shp") else path + ".shp"
