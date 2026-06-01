"""
make_synthetic_segy.py — Parametric synthetic SEG-Y generator for tests.

Creates SEG-Y files that exercise all code paths:
- Configurable coord_unit (1, 2, 3)
- Configurable scalar sign
- Variable delays (for delay-align tests)
- Known sinusoid injected for spectrum peak verification
- Contiguous pairs for chain detection tests
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import segyio


def make_synthetic_segy(
    path: str,
    n_traces: int = 50,
    ns: int = 256,
    dt_us: int = 250,
    scalar_coord: int = -100,
    coord_unit: int = 3,
    base_lon: float = -8.0,
    base_lat: float = 43.0,
    lon_step: float = 0.001,
    lat_step: float = 0.0,
    water_depth: float = 120.0,
    delay_ms: int = 0,
    delay_variable: bool = False,
    inject_freq_hz: Optional[float] = None,
    year: int = 2024,
    doy_start: int = 100,
    hour: int = 10,
    minute_start: int = 0,
) -> str:
    """
    Write a synthetic SEG-Y file and return its path.

    Parameters
    ----------
    path          : output file path
    n_traces      : number of traces
    ns            : samples per trace
    dt_us         : sample interval in microseconds
    scalar_coord  : SourceGroupScalar value (negative → divide, positive → multiply)
    coord_unit    : CoordinateUnits (1=m/ft, 2=arc-sec, 3=decimal degrees)
    base_lon/lat  : starting position in degrees
    lon/lat_step  : step between traces (degrees)
    water_depth   : constant water depth in metres
    delay_ms      : base delay recording time in ms
    delay_variable: if True, delay increases by 1 ms per trace (test align)
    inject_freq_hz: if given, add a sinusoid at this frequency to the data
    year, doy_start, hour, minute_start: timestamp fields
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Compute scalar factor for coord storage
    if scalar_coord < 0:
        fac = abs(scalar_coord)   # stored_value = degrees * fac
    elif scalar_coord > 0:
        fac = 1.0 / scalar_coord
    else:
        fac = 1.0

    spec = segyio.spec()
    spec.sorting = None
    spec.format  = 1          # IBM float
    spec.samples = np.arange(ns, dtype=np.float32)
    spec.tracecount = n_traces

    with segyio.create(path, spec) as f:
        # Binary header
        f.bin.update(tsort=segyio.TraceSortingFormat.UNKNOWN_SORTING,
                     hdt=dt_us, dto=dt_us)

        rng = np.random.default_rng(42)

        for i in range(n_traces):
            lon_deg = base_lon + i * lon_step
            lat_deg = base_lat + i * lat_step

            if coord_unit == 2:
                # Arc-seconds: stored as lon_deg * 3600 / fac
                sx = int(round(lon_deg * 3600 * fac))
                sy = int(round(lat_deg * 3600 * fac))
            else:
                sx = int(round(lon_deg * fac))
                sy = int(round(lat_deg * fac))

            delay = delay_ms + (i if delay_variable else 0)

            # Encode minute so timestamps differ per trace
            min_val = (minute_start + i) % 60
            hour_val = hour + (minute_start + i) // 60

            h = {
                segyio.TraceField.SourceX:           sx,
                segyio.TraceField.SourceY:           sy,
                segyio.TraceField.GroupX:            sx,
                segyio.TraceField.GroupY:            sy,
                segyio.TraceField.SourceGroupScalar: scalar_coord,
                segyio.TraceField.CoordinateUnits:   coord_unit,
                segyio.TraceField.DelayRecordingTime: delay,
                segyio.TraceField.SourceWaterDepth:  int(water_depth * 10),
                segyio.TraceField.ElevationScalar:   -10,
                segyio.TraceField.YearDataRecorded:  year,
                segyio.TraceField.DayOfYear:         doy_start,
                segyio.TraceField.HourOfDay:         hour_val % 24,
                segyio.TraceField.MinuteOfHour:      min_val,
                segyio.TraceField.SecondOfMinute:    0,
                segyio.TraceField.TraceNumber:       i + 1,
            }
            f.header[i] = h

            # Trace data: random noise + optional sinusoid
            t_s = np.arange(ns) * dt_us * 1e-6
            tr  = rng.standard_normal(ns).astype(np.float32)
            if inject_freq_hz is not None:
                tr += 2.0 * np.sin(2 * np.pi * inject_freq_hz * t_s).astype(np.float32)
            f.trace[i] = tr

    return path


def make_chain_pair(
    base_dir: str,
    prefix: str = "chain",
    n_traces: int = 40,
    ns: int = 128,
    dt_us: int = 500,
    gap_km: float = 0.05,
) -> tuple:
    """
    Create two contiguous SEG-Y files suitable for chain detection.

    The second file starts where the first ends, with a separation of
    approximately gap_km. Returns (path1, path2).
    """
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    path1 = str(Path(base_dir) / f"{prefix}_1.sgy")
    path2 = str(Path(base_dir) / f"{prefix}_2.sgy")

    # ~0.001 degree ≈ 0.11 km per step at lat 43°
    lon_step  = 0.001
    n_traces1 = n_traces
    end_lon   = -8.0 + n_traces1 * lon_step
    # gap: ~gap_km / 111.32 degrees
    gap_deg   = gap_km / 111.32

    make_synthetic_segy(path1, n_traces=n_traces1, ns=ns, dt_us=dt_us,
                        base_lon=-8.0, doy_start=100, minute_start=0)
    make_synthetic_segy(path2, n_traces=n_traces, ns=ns, dt_us=dt_us,
                        base_lon=end_lon + gap_deg, doy_start=100, minute_start=n_traces1)
    return path1, path2
