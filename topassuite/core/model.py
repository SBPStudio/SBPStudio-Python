"""
model.py — Pure data-holder classes for SEG-Y profiles and profile chains.

SegyMetadata  : lightweight header-only record (no trace data).
SegyProfile   : full in-memory profile (metadata + trace matrix).
ProfileChain  : ordered concatenation of contiguous SegyProfile objects.

Array contract
--------------
- data      : (ns, n_traces) float32
- lons/lats : (n_traces,)   float64
- dist_km   : (n_traces,)   float64
- delays    : (n_traces,)   int/float (raw ms values from header)
- timestamps: list[str]    length n_traces, format "YYYY-DOYnnn HH:MM:SS"

Known limitations (carried forward from the monolith, NOT fixed here)
----------------------------------------------------------------------
- delay_ms uses trace[0] only; per-trace delay variability is exposed via
  the delays array but the scalar delay_ms is always traces[0].
- dist_km geographic-vs-projected heuristic uses coord_unit + range check;
  projected-CRS distance is assumed to be in metres (not validated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _scalar_fac(sc: int) -> float:
    """
    Convert a SEG-Y coordinate/elevation scalar value to a multiplicative factor.

    SEG-Y convention:
      sc < 0  → divide by abs(sc)   → fac = 1/abs(sc)
      sc > 0  → multiply by sc      → fac = sc
      sc == 0 → no scaling          → fac = 1.0
    """
    if sc < 0:
        return 1.0 / abs(sc)
    if sc > 0:
        return float(sc)
    return 1.0


# ── SegyMetadata ───────────────────────────────────────────────────────────────

@dataclass
class SegyMetadata:
    """
    Header-only metadata record for a single SEG-Y file.

    load_metadata() populates this WITHOUT reading the trace matrix,
    so it is very fast even for large files.

    Fields
    ------
    path         : absolute path to the SEG-Y file
    name         : filename only (Path.name)
    stem         : filename without extension (Path.stem)
    n_traces     : total trace count
    ns           : samples per trace
    dt_us        : sample interval in microseconds
    scalar_coord : raw SourceGroupScalar header value (first trace)
    scalar_elev  : raw ElevationScalar header value (first trace)
    coord_unit   : CoordinateUnits value (first trace)
    delay_ms     : DelayRecordingTime of trace[0] in ms  ← KNOWN LIMITATION
    min_delay    : min(delays) across all traces, ms
    max_delay    : max(delays) across all traces, ms
    dur_ms       : ns * dt_us / 1000  (record window length in ms)
    lons/lats    : (n_traces,) float64 in degrees (arc-sec already converted)
    dist_km      : (n_traces,) float64 cumulative along-track distance
    total_km     : dist_km[-1]
    water_depth  : (n_traces,) float64 in metres
    timestamps   : list[str] format "YYYY-DOYnnn HH:MM:SS"
    detected_crs : guessed EPSG string or None
    crs_notes    : list of detection notes
    clip_p99     : float, 99th percentile of abs(data) — None until traces loaded
    error        : None if loaded successfully, else the exception string
    """
    path:          str
    name:          str
    stem:          str
    n_traces:      int                  = 0
    ns:            int                  = 0
    dt_us:         int                  = 0
    scalar_coord:  int                  = 0
    scalar_elev:   int                  = 0
    coord_unit:    int                  = 0
    delay_ms:      int                  = 0
    min_delay:     float                = 0.0
    max_delay:     float                = 0.0
    delays:        Optional[np.ndarray] = field(default=None, repr=False)
    dur_ms:        float                = 0.0
    lons:          Optional[np.ndarray] = field(default=None, repr=False)
    lats:          Optional[np.ndarray] = field(default=None, repr=False)
    dist_km:       Optional[np.ndarray] = field(default=None, repr=False)
    total_km:      float                = 0.0
    water_depth:   Optional[np.ndarray] = field(default=None, repr=False)
    timestamps:    List[str]            = field(default_factory=list)
    detected_crs:  Optional[str]        = None
    crs_notes:     List[str]            = field(default_factory=list)
    clip_p99:      Optional[float]      = None
    error:         Optional[str]        = None

    def summary(self) -> str:
        """One-line human-readable summary."""
        if self.error:
            return f"ERROR: {self.error}"
        wd = float(np.nanmean(self.water_depth)) if self.water_depth is not None else float("nan")
        return (f"{self.n_traces} trazas · {self.dur_ms:.0f} ms · "
                f"{self.total_km:.1f} km · WD {wd:.0f} m")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (arrays converted to lists)."""
        return {
            "path":         self.path,
            "name":         self.name,
            "n_traces":     self.n_traces,
            "ns":           self.ns,
            "dt_us":        self.dt_us,
            "dur_ms":       self.dur_ms,
            "scalar_coord": self.scalar_coord,
            "scalar_elev":  self.scalar_elev,
            "coord_unit":   self.coord_unit,
            "delay_ms":     self.delay_ms,
            "min_delay":    self.min_delay,
            "max_delay":    self.max_delay,
            "total_km":     self.total_km,
            "detected_crs": self.detected_crs,
            "crs_notes":    self.crs_notes,
            "clip_p99":     self.clip_p99,
            "error":        self.error,
        }


# ── SegyProfile ────────────────────────────────────────────────────────────────

class SegyProfile:
    """
    Full in-memory SEG-Y profile: metadata + trace data matrix.

    Loaded by io_segy.load_profile(). The object mirrors the original
    GUI SegyProfile exactly so that existing processing code is unchanged.

    Array shapes
    ------------
    data       : (ns, n_traces) float32
    lons/lats  : (n_traces,) float64  — in degrees
    dist_km    : (n_traces,) float64
    amp_max    : (n_traces,) float32  — max(abs(data), axis=0)
    t_ms       : (ns,) float64        — two-way travel time axis in ms
    delays     : (n_traces,) int16/int32
    water_depth: (n_traces,) float64  — in metres
    """

    def __init__(self, path: str) -> None:
        self.path:  str            = path
        self.name:  str            = Path(path).name
        self.stem:  str            = Path(path).stem
        self.error: Optional[str]  = None
        # These are set by io_segy.load_profile after construction.
        self.data:        Optional[np.ndarray] = None
        self.lons:        Optional[np.ndarray] = None
        self.lats:        Optional[np.ndarray] = None
        self.dist_km:     Optional[np.ndarray] = None
        self.total_km:    float = 0.0
        self.amp_max:     Optional[np.ndarray] = None
        self.t_ms:        Optional[np.ndarray] = None
        self.delays:      Optional[np.ndarray] = None
        self.water_depth: Optional[np.ndarray] = None
        self.timestamps:  List[str] = []
        self.n_traces:    int   = 0
        self.ns:          int   = 0
        self.dt_us:       int   = 0
        self.dur_ms:      float = 0.0
        self.delay_ms:    int   = 0
        self.min_delay:   float = 0.0
        self.max_delay:   float = 0.0
        self.clip_p99:    float = 0.0
        self.scalar_coord:int   = 0
        self.scalar_elev: int   = 0
        self.coord_unit:  int   = 0
        self.detected_crs: Optional[str]  = None
        self.crs_notes:    List[str]       = []

    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        wd = float(np.nanmean(self.water_depth)) if self.water_depth is not None else float("nan")
        return (f"{self.n_traces} trazas · {self.dur_ms:.0f} ms · "
                f"{self.total_km:.1f} km · WD {wd:.0f} m")

    def to_metadata(self) -> SegyMetadata:
        """Return a SegyMetadata snapshot of this profile's header fields."""
        return SegyMetadata(
            path=self.path, name=self.name, stem=self.stem,
            n_traces=self.n_traces, ns=self.ns, dt_us=self.dt_us,
            scalar_coord=self.scalar_coord, scalar_elev=self.scalar_elev,
            coord_unit=self.coord_unit, delay_ms=self.delay_ms,
            min_delay=self.min_delay, max_delay=self.max_delay,
            delays=self.delays,
            dur_ms=self.dur_ms, lons=self.lons, lats=self.lats,
            dist_km=self.dist_km, total_km=self.total_km,
            water_depth=self.water_depth, timestamps=self.timestamps,
            detected_crs=self.detected_crs, crs_notes=self.crs_notes,
            clip_p99=self.clip_p99, error=self.error,
        )


# ── ProfileChain ───────────────────────────────────────────────────────────────

class ProfileChain:
    """
    Ordered concatenation of N contiguous SegyProfile objects.

    Detection criteria (greedy, chronological sort):
      1. Same dt_us (same acquisition configuration).
      2. Haversine distance between the last point of profile[i] and the
         first point of profile[i+1] <= gap_km threshold.

    dist_km is continuous across profiles including the geographic gap
    distance between consecutive segments — no artificial zero-reset.

    Known limitation: boundaries_km is computed from per-profile total_km
    (sum of intra-profile increments), NOT from dist_km directly, so there
    is a possible off-by-gap_offset difference versus the chain dist_km
    array at join points. Preserved from monolith.
    """

    GAP_KM_MAX: float = 2.0

    def __init__(self, profiles: List[SegyProfile]) -> None:
        self.profiles = profiles
        self.name  = " ⛓ ".join(p.stem for p in profiles)
        self.label = (f"[{len(profiles)} perfiles]  {profiles[0].name} … {profiles[-1].name}"
                      if len(profiles) > 1
                      else f"[1 perfil]  {profiles[0].name}")
        self._concat()

    # ── Concatenation ──────────────────────────────────────────────────────

    def _concat(self) -> None:
        # Seismic data: (ns, total_traces) float32
        self.data = np.concatenate([p.data for p in self.profiles], axis=1)

        # Delay recording time — preserved verbatim (NEVER altered)
        self.delays    = np.concatenate([p.delays    for p in self.profiles])
        self.min_delay = float(np.min(self.delays))
        self.max_delay = float(np.max(self.delays))

        # Coordinates (degrees)
        self.lons        = np.concatenate([p.lons        for p in self.profiles])
        self.lats        = np.concatenate([p.lats        for p in self.profiles])
        self.water_depth = np.concatenate([p.water_depth for p in self.profiles])

        # Timestamps
        self.timestamps: List[str] = []
        for p in self.profiles:
            self.timestamps.extend(p.timestamps)

        # Continuous cumulative distance including inter-profile gaps
        segments = []
        offset   = 0.0
        for idx_p, p in enumerate(self.profiles):
            segments.append(p.dist_km + offset)
            offset += p.total_km
            if idx_p + 1 < len(self.profiles):
                nxt = self.profiles[idx_p + 1]
                offset += ProfileChain._haversine_km(
                    float(p.lons[-1]),   float(p.lats[-1]),
                    float(nxt.lons[0]),  float(nxt.lats[0]))
        self.dist_km  = np.concatenate(segments)
        self.total_km = float(self.dist_km[-1])

        # Metadata from first profile (dt/ns must be equal across profiles)
        p0             = self.profiles[0]
        self.dt_us     = p0.dt_us
        self.ns        = p0.ns
        self.dur_ms    = p0.dur_ms
        self.delay_ms  = p0.delay_ms
        self.n_traces  = self.data.shape[1]

        # Boundary positions (km along chain) — start of each non-first profile.
        # NOTE: uses per-profile total_km cumulative sum, NOT dist_km directly.
        self.boundaries_km: List[float] = []
        d = 0.0
        for p in self.profiles[:-1]:
            d += p.total_km
            self.boundaries_km.append(d)

        self.clip_p99 = float(np.percentile(np.abs(self.data), 99))

    # ── Haversine distance helper ──────────────────────────────────────────

    @staticmethod
    def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Great-circle distance in km between two WGS84 points."""
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    # ── Chain detection ────────────────────────────────────────────────────

    @classmethod
    def detect(cls, profiles: List[SegyProfile],
               gap_km: Optional[float] = None) -> List["ProfileChain"]:
        """
        Group a list of valid SegyProfile objects into contiguous chains.

        Profiles are sorted chronologically by their first timestamp before
        grouping. Returns one ProfileChain per detected group; returns a
        single-element chain for isolated profiles. Returns [] if no valid
        profiles are provided.

        Parameters
        ----------
        profiles : list of SegyProfile (may include profiles with errors)
        gap_km   : maximum inter-profile gap to be considered contiguous;
                   defaults to GAP_KM_MAX (2.0 km)
        """
        if not profiles:
            return []

        gap   = gap_km if gap_km is not None else cls.GAP_KM_MAX
        valid = [p for p in profiles if not p.error]

        if len(valid) < 2:
            return [cls([p]) for p in valid] if valid else []

        def _ts_key(p: SegyProfile) -> str:
            return p.timestamps[0] if p.timestamps else ""

        valid_sorted = sorted(valid, key=_ts_key)

        groups: List[List[SegyProfile]] = []
        current = [valid_sorted[0]]

        for nxt in valid_sorted[1:]:
            prev    = current[-1]
            same_dt = prev.dt_us == nxt.dt_us
            gap_d   = cls._haversine_km(
                prev.lons[-1], prev.lats[-1],
                nxt.lons[0],  nxt.lats[0])
            if same_dt and gap_d <= gap:
                current.append(nxt)
            else:
                groups.append(current)
                current = [nxt]
        groups.append(current)

        return [cls(g) for g in groups]
