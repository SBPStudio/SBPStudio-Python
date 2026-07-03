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
    n_traces     : trace count AFTER duplicate-timestamp cleanup
    original_n_traces : raw file trace count before cleanup
    n_purged     : duplicate-timestamp traces removed (n_traces = original - purged)
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
    original_n_traces: int              = 0
    n_purged:      int                  = 0
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
            "original_n_traces": self.original_n_traces,
            "n_purged":     self.n_purged,
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
        self.lons:        Optional[np.ndarray] = None   # raw recorded (exports)
        self.lats:        Optional[np.ndarray] = None   # raw recorded (exports)
        self.track_lons:  Optional[np.ndarray] = None   # cleaned display track
        self.track_lats:  Optional[np.ndarray] = None   # cleaned display track
        self.dist_km:     Optional[np.ndarray] = None
        self.total_km:    float = 0.0
        self.amp_max:     Optional[np.ndarray] = None
        self.t_ms:        Optional[np.ndarray] = None
        self.delays:      Optional[np.ndarray] = None
        self.water_depth: Optional[np.ndarray] = None
        self.timestamps:  List[str] = []
        self.text_header:   str  = ""          # 3200-byte textual header (decoded)
        self.text_header_encoding: str = "ascii"  # "ascii" | "ebcdic" — auto-detected guess
        self.trace_headers: dict = {}          # {label: (n_traces,) raw header array}
        self.n_traces:    int   = 0
        # Duplicate-timestamp cleanup bookkeeping (io_segy). original_n_traces is
        # the raw file trace count; n_traces above is the cleaned count; n_purged
        # is how many consecutive duplicate-timestamp traces were removed (0 if
        # none / cleanup bypassed). Surfaced in the GUI header panel + status bar.
        self.original_n_traces: int = 0
        self.n_purged:    int   = 0
        self.ns:          int   = 0
        # Real signal band: [active_lo, active_ns) — the first/last+1 row
        # (sample) index across all traces with amplitude above a small
        # fraction of clip_p99, each padded by a safety margin (set by
        # io_segy._populate_profile_from_file once ``data`` is loaded; both
        # None for header-only stubs). Marine surveys commonly record every
        # shot with a FIXED window sized for the deepest expected water
        # depth, so a shallow segment's traces are mostly DEAD samples —
        # trailing zeros below the real signal (active_ns) — and, just as
        # often, LEADING zeros above it (active_lo): a high-res SBP system
        # starts recording at the ping, but sub-bottom reflectors of
        # interest don't arrive until after the water-column travel time,
        # which can be thousands of samples in deep water. See
        # active_band_for_traces.
        self.active_lo:   Optional[int] = None
        self.active_ns:   Optional[int] = None
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
        self.crs_is_unknown: bool  = False   # #21: True when detected_crs is None
        self.meters_per_unit: float = 1.0    # #13: 0.3048 when BinField.MeasurementSystem==2

    def summary(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        wd = float(np.nanmean(self.water_depth)) if self.water_depth is not None else float("nan")
        return (f"{self.n_traces} trazas · {self.dur_ms:.0f} ms · "
                f"{self.total_km:.1f} km · WD {wd:.0f} m")

    def active_ns_for_traces(self, c0: int, c1: int) -> Optional[int]:
        """Real listening-window depth for the trace range [c0, c1) — see
        ``active_ns``. A single profile has only one file, so the range is
        irrelevant; kept for a uniform duck-typed call from the live preview
        (``ProfileChain`` below resolves a per-file value for the range)."""
        return self.active_ns

    def active_band_for_traces(self, c0: int, c1: int) -> Optional[Tuple[int, int]]:
        """``(active_lo, active_ns)`` for the trace range [c0, c1) — the real
        signal band, top and bottom; see ``active_lo``. ``None`` if either
        bound hasn't been detected (header-only stub). A single profile has
        only one file, so the range is irrelevant; kept for a uniform duck-
        typed call from the live preview (``ProfileChain`` below resolves
        the per-file band for the range)."""
        if self.active_lo is None or self.active_ns is None:
            return None
        return self.active_lo, self.active_ns

    def read_columns(self, c0: int, c1: int) -> np.ndarray:
        """Column-bounded read — trivial for a single profile (one in-memory
        array already, no concatenation involved). Reloads from disk if
        evicted, mirroring ``ProfileChain.read_columns``'s fallback, so GUI
        code can call this uniformly on either a lone profile or a chain."""
        if self.data is None:
            from .io_segy import load_profile
            loaded = load_profile(self.path, load_traces=True)
            if loaded.error:
                raise ValueError(f"Cannot load {self.name}: {loaded.error}")
            self.data = loaded.data
            if not self.trace_headers:
                self.trace_headers = loaded.trace_headers
        c0 = max(0, min(int(c0), self.n_traces))
        c1 = max(c0, min(int(c1), self.n_traces))
        return self.data[:, c0:c1]

    def to_metadata(self) -> SegyMetadata:
        """Return a SegyMetadata snapshot of this profile's header fields."""
        return SegyMetadata(
            path=self.path, name=self.name, stem=self.stem,
            n_traces=self.n_traces, original_n_traces=self.original_n_traces,
            n_purged=self.n_purged, ns=self.ns, dt_us=self.dt_us,
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
        # LAZY by design: the heavy stitched trace matrix is NOT built here.
        # __init__ relies solely on the lightweight header stubs (always in RAM),
        # so chain detection stays fast and memory-flat and never collides with
        # LRU eviction (which may have dropped constituent profiles' .data). The
        # matrix is assembled on demand by load_chain_traces() when the chain is
        # actually viewed.
        self.data: Optional[np.ndarray] = None
        self.clip_p99: Optional[float] = None
        self._build_metadata()

    # ── Lightweight metadata (header-only; no trace data) ──────────────────

    def _build_metadata(self) -> None:
        """Assemble everything derivable from the header stubs alone — geometry,
        timestamps, distance axis, boundaries, inspector headers. Deliberately
        touches NO ``.data``, so it is safe even after LRU eviction has reverted
        constituent profiles to lazy stubs."""
        # Delay recording time — preserved verbatim (NEVER altered)
        self.delays    = np.concatenate([p.delays    for p in self.profiles])
        self.min_delay = float(np.min(self.delays))
        self.max_delay = float(np.max(self.delays))

        # Coordinates (degrees) — raw recorded (exports) + cleaned display track.
        self.lons        = np.concatenate([p.lons        for p in self.profiles])
        self.lats        = np.concatenate([p.lats        for p in self.profiles])
        self.track_lons  = np.concatenate([p.track_lons  for p in self.profiles])
        self.track_lats  = np.concatenate([p.track_lats  for p in self.profiles])
        self.water_depth = np.concatenate([p.water_depth for p in self.profiles])

        # Timestamps
        self.timestamps: List[str] = []
        for p in self.profiles:
            self.timestamps.extend(p.timestamps)

        # Header Inspector: keep the first profile's textual header; concatenate
        # the per-trace raw header fields across the whole chain.
        self.text_header = self.profiles[0].text_header if self.profiles else ""
        self.text_header_encoding = (
            self.profiles[0].text_header_encoding if self.profiles else "ascii")
        if self.profiles and all(p.trace_headers for p in self.profiles):
            labels = list(self.profiles[0].trace_headers.keys())
            self.trace_headers = {
                lab: np.concatenate([p.trace_headers[lab] for p in self.profiles])
                for lab in labels
            }
        else:
            self.trace_headers = {}
        # CRS inherited from the first profile so the map can reproject the chain
        # track even before the trace matrix is assembled. coord_unit travels
        # with it — safe_map_coords needs both to know whether the chain's
        # native coordinates are projected (1), arc-seconds (2), or already
        # decimal degrees (3). A chain only ever joins profiles with the SAME
        # acquisition config, so the first profile's value is authoritative.
        self.detected_crs = self.profiles[0].detected_crs if self.profiles else None
        self.coord_unit   = self.profiles[0].coord_unit   if self.profiles else 0

        # Continuous cumulative distance including inter-profile gaps
        segments = []
        offset   = 0.0
        for idx_p, p in enumerate(self.profiles):
            segments.append(p.dist_km + offset)
            offset += p.total_km
            if idx_p + 1 < len(self.profiles):
                nxt = self.profiles[idx_p + 1]
                # Cleaned-track endpoints: consistent with the smoothed per-profile
                # dist_km being concatenated here.
                offset += ProfileChain._haversine_km(
                    float(p.track_lons[-1]),   float(p.track_lats[-1]),
                    float(nxt.track_lons[0]),  float(nxt.track_lats[0]))
        self.dist_km  = np.concatenate(segments)
        self.total_km = float(self.dist_km[-1])

        # Metadata from first profile. dt_us must be equal across profiles —
        # the chain detector enforces it. ns may NOT be: heterogeneous record
        # lengths within one survey are real in the field (e.g. 6427 vs 32767
        # samples when the operator changed the recording window mid-line).
        # The chain's vertical extent is the DEEPEST constituent file; every
        # trace-matrix assembly bottom-pads shorter segments with zeros to
        # this height (see load_chain_traces / read_columns).
        p0             = self.profiles[0]
        self.dt_us     = p0.dt_us
        self.ns        = int(max(p.ns for p in self.profiles))
        self.dur_ms    = self.ns * self.dt_us / 1000.0
        self.delay_ms  = p0.delay_ms
        # Total trace count summed from the stubs (NOT data.shape — data is lazy).
        self.n_traces  = int(sum(p.n_traces for p in self.profiles))

        # Boundary positions (km along chain) — first dist_km value of each
        # non-first profile, read directly from the assembled dist_km so the
        # seam positions include inter-profile gaps and stay consistent with
        # every searchsorted call on dist_km.
        trace_offsets = np.cumsum([p.n_traces for p in self.profiles])
        self.boundaries_km: List[float] = [
            float(self.dist_km[trace_offsets[i]])
            for i in range(len(self.profiles) - 1)
        ]
        # Per-profile absolute column [start, end) — reused by
        # active_ns_for_traces to resolve which constituent file(s) a column
        # range overlaps without re-deriving it from boundaries_km.
        self._trace_ends   = trace_offsets
        self._trace_starts = np.concatenate(([0], trace_offsets[:-1]))

    # ── Lazy trace assembly (on demand, from disk if evicted) ──────────────

    @staticmethod
    def _ensure_profile_loaded(p: "SegyProfile") -> np.ndarray:
        """Return ``p.data``, reloading from disk via ``load_profile`` if it's
        an evicted/never-loaded stub — the SAME per-segment fallback used by
        both ``load_chain_traces`` (legacy, whole-chain) and ``read_columns``
        (segmented). Mutates ``p.data``/``p.trace_headers`` in place: a
        reload lands in the constituent ``SegyProfile``'s OWN ``data`` slot
        (the same object ``AppState``'s per-profile LRU already manages —
        see ``read_columns``'s docstring), never a separate chain-level copy."""
        data = getattr(p, "data", None)
        if data is None:
            from .io_segy import load_profile
            loaded = load_profile(p.path, load_traces=True)
            if loaded.error:
                raise ValueError(f"Cannot load chain segment "
                                 f"{p.name}: {loaded.error}")
            data = loaded.data
            p.data = data
            if not p.trace_headers:                # upgrade stub with full headers
                p.trace_headers = loaded.trace_headers
        return data

    def read_columns(self, c0: int, c1: int) -> np.ndarray:
        """Full-depth, column-bounded read across whichever constituent
        file(s) overlap absolute trace range [c0, c1) — the SEGMENTED
        alternative to ``load_chain_traces``. NEVER builds/touches the
        whole-chain ``self.data`` array: only the (typically one, rarely a
        handful at a chain seam) segment(s) the requested range actually
        spans are read, reloading any evicted one from disk via
        ``_ensure_profile_loaded``.

        Because a reload lands in the constituent ``SegyProfile.data`` slot
        (NOT a separate chain-level allocation), it is the SAME kind of
        object ``AppState``'s existing per-profile LRU (``MAX_HOT_PROFILES``)
        already governs — eliminating the leak where ``ProfileChain.data``
        escaped that eviction entirely by living outside it. The residual
        gap (a segment ONLY ever touched via chain reads, never standalone,
        won't be in the GUI's LRU ordering until something calls its
        ``_touch``) is a follow-up for the GUI layer, not this core method.

        This is the canonical access path for the live preview's per-window
        extraction and for chunked export (see ``export_filter.py``);
        ``load_chain_traces``/``self.data`` remain for legacy consumers that
        still need the whole chain as one materialised ndarray.
        """
        c0 = max(0, min(int(c0), self.n_traces))
        c1 = max(c0, min(int(c1), self.n_traces))
        mats = []
        for p, start, end in zip(self.profiles, self._trace_starts, self._trace_ends):
            if end <= c0 or start >= c1:
                continue                      # no overlap with the requested range
            data = self._ensure_profile_loaded(p)
            lo = max(c0, start) - start
            hi = min(c1, end) - start
            mats.append(data[:, lo:hi])
        if not mats:
            return np.zeros((self.ns, 0), dtype=np.float32)
        # Zero-copy fast path: one segment already at full chain height (the
        # overwhelmingly common case — most reads fall inside one file).
        if len(mats) == 1 and mats[0].shape[0] == self.ns:
            return mats[0]
        # Heterogeneous record lengths: chain ns is the DEEPEST file, so any
        # shorter segment is bottom-padded with zeros by writing it into the
        # top of a zero-filled destination — same strategy and same 0.0-vs-NaN
        # rationale as load_chain_traces.
        out = np.zeros((self.ns, sum(m.shape[1] for m in mats)), dtype=np.float32)
        col = 0
        for m in mats:
            out[:m.shape[0], col:col + m.shape[1]] = m
            col += m.shape[1]
        return out

    def columns_in_cache(self, c0: int, c1: int) -> bool:
        """True if every segment overlapping [c0, c1) has its data in memory.

        Returns False as soon as any overlapping segment has been LRU-evicted
        (its ``data`` attribute is None).  Used by the preview layer to decide
        whether to dispatch a ``BasePrepWorker`` before calling
        ``_prepared_base`` — avoiding a blocking disk read on the GUI thread."""
        c0 = max(0, min(int(c0), self.n_traces))
        c1 = max(c0, min(int(c1), self.n_traces))
        for p, start, end in zip(self.profiles, self._trace_starts, self._trace_ends):
            if end <= c0 or start >= c1:
                continue
            if getattr(p, "data", None) is None:
                return False
        return True

    def load_chain_traces(self, cancel=None) -> "ProfileChain":
        """Assemble the stitched ``(ns, total_traces)`` trace matrix on demand.

        Idempotent: a no-op once ``self.data`` is populated. Each constituent
        profile's traces are sourced from RAM when still resident, otherwise
        re-read from disk via ``_ensure_profile_loaded`` — so this works
        correctly even after LRU eviction reverted them to header-only stubs.

        Kept for LEGACY consumers that genuinely need the whole chain as one
        materialised ndarray (e.g. an un-migrated export path, or analysis
        code that runs a 2-D filter needing the full extent). The live
        preview and chunked export use ``read_columns`` instead and never
        trigger this — see that method's docstring for why ``self.data``
        living outside the per-profile LRU was the actual RAM leak.

        ``cancel`` (optional) is a CancelToken whose ``check()`` is polled per
        profile so a long assembly can be aborted cooperatively.
        """
        if self.data is not None:
            return self
        mats = []
        for p in self.profiles:
            if cancel is not None:
                cancel.check()
            mats.append(self._ensure_profile_loaded(p))
        # Write directly into one pre-allocated contiguous destination (a
        # np.concatenate would need every segment's row count to match — the
        # exact ValueError crash heterogeneous-ns chains used to hit). The
        # destination is self.ns tall (the DEEPEST file) and ZERO-filled, so
        # copying each segment into the TOP of its column slot bottom-pads
        # the shorter ones implicitly — one copy per segment, no per-segment
        # np.pad allocations. 0.0 (not NaN) is the padding by design: zeros
        # pass neutrally through every stats reduction (amp_max, clip_p99
        # percentile, the preview's global levels) and the DSP boundary,
        # whereas NaN would poison plain max/percentile math and force
        # nan-aware variants through the whole downstream pipeline.
        out = np.zeros((self.ns, sum(m.shape[1] for m in mats)),
                       dtype=np.float32)
        col = 0
        for m in mats:
            out[:m.shape[0], col:col + m.shape[1]] = m
            col += m.shape[1]
        self.data = out
        # Chain-wide clip_p99 used to be a true 99th percentile over every
        # sample of the stitched array — an O(ns × total_traces) full-array
        # `np.abs` + percentile pass on every load. It is metadata only (no
        # DSP/display-stability path reads it; the live preview's color
        # lock comes from PreviewController._compute_global_levels, not
        # this), so the MAX of the already-computed per-file clip_p99
        # values — never an underestimate, same "safe union" philosophy as
        # active_band_for_traces — is a deliberate, documented approximation
        # traded for skipping that full-array pass entirely.
        per_file = [p.clip_p99 for p in self.profiles if p.clip_p99]
        self.clip_p99 = max(per_file) if per_file else 0.0
        self.n_traces = self.data.shape[1]
        # Rebuild the chain-level header map now every constituent is fully loaded.
        # This populates the Header Inspector when the chain view re-emits
        # active_chain_changed after the worker completes.
        if not self.trace_headers and all(p.trace_headers for p in self.profiles):
            labels = list(self.profiles[0].trace_headers.keys())
            self.trace_headers = {
                lab: np.concatenate([p.trace_headers[lab] for p in self.profiles])
                for lab in labels
            }
        return self

    def active_ns_for_traces(self, c0: int, c1: int) -> Optional[int]:
        """Real listening-window depth across the constituent file(s)
        overlapping absolute trace range [c0, c1) — the MAX of their
        per-file ``SegyProfile.active_ns`` (the deepest of the overlapping
        files governs, since the live preview must not under-cut any of
        them). Returns None — caller falls back to the full ``ns`` — if any
        overlapping profile hasn't been trace-loaded yet (a header-only
        stub has no ``active_ns``), which only happens for a column range
        outside what ``load_chain_traces`` has actually populated."""
        vals: List[int] = []
        for p, start, end in zip(self.profiles, self._trace_starts, self._trace_ends):
            if end <= c0 or start >= c1:
                continue                      # no overlap with the requested range
            if p.active_ns is None:
                return None
            vals.append(p.active_ns)
        return max(vals) if vals else None

    def active_band_for_traces(self, c0: int, c1: int) -> Optional[Tuple[int, int]]:
        """``(lo, hi)`` real signal band across the constituent file(s)
        overlapping absolute trace range [c0, c1) — the SAFE UNION of their
        per-file bands: ``lo`` is the SHALLOWEST top-of-signal (min) and
        ``hi`` is the DEEPEST bottom-of-signal (max) among the overlapping
        files, so the combined extraction range never under-cuts any one of
        them. Returns None — caller falls back to the full ``(0, ns)`` band
        — if any overlapping profile hasn't been trace-loaded yet (mirrors
        ``active_ns_for_traces``)."""
        los: List[int] = []
        his: List[int] = []
        for p, start, end in zip(self.profiles, self._trace_starts, self._trace_ends):
            if end <= c0 or start >= c1:
                continue
            if p.active_lo is None or p.active_ns is None:
                return None
            los.append(p.active_lo)
            his.append(p.active_ns)
        if not los:
            return None
        return min(los), max(his)

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
