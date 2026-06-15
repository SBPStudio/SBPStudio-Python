"""
test_cli_figsize_clamp.py — the --x-scale width clamp is gone (and the default
CLI width path is unchanged).

The old `cmd_export_image._figsize` did `w = max(2.0, total_km / x_scale)`, which
froze short profiles at a 2.0-in width → tiny raster → pixelated PDF. The width
must now honour `total_km / x_scale` exactly, while the no-`--x-scale` default
(`max(8, n_traces·px/dpi)`) is preserved (Zero CLI Regressions).
"""
from __future__ import annotations

import re
import subprocess
import sys

from tests.make_synthetic_segy import make_synthetic_segy
from sbp_studio.core import load_profile

# Parse the ASCII "OK <file>  WxH px, <w>x<h> in @ <dpi> DPI" summary line
# (robust to console encoding of the '→'/'×' glyphs in the scale line).
_DIMS_RE = re.compile(r"([\d.]+)x([\d.]+)\s*in\s*@")


def _run_export(seg_path: str, out_path: str, *extra: str) -> tuple:
    """Run the CLI export-image and return (width_in, height_in) it reported."""
    cmd = [sys.executable, "-m", "sbp_studio.cli.main", "export-image", seg_path,
           "--velocity", "1500", "--format", "pdf", "--quality", "high",
           "--out", out_path, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    m = _DIMS_RE.search(r.stderr)
    assert m, f"no export dims found (rc={r.returncode}).\nSTDERR:\n{r.stderr}"
    return float(m.group(1)), float(m.group(2))


def _short_segy(tmp_path) -> tuple:
    p = make_synthetic_segy(str(tmp_path / "short.sgy"), n_traces=60, ns=1000,
                            base_lon=-3.0, base_lat=43.0, lon_step=0.0005)
    sd = load_profile(p, load_traces=False)
    return p, sd.total_km, sd.n_traces


def test_x_scale_width_is_not_clamped_at_2in(tmp_path):
    seg, total_km, _ = _short_segy(tmp_path)
    assert total_km / 2.0 < 2.0                       # this profile WOULD be clamped
    w, _h = _run_export(seg, str(tmp_path / "x2.pdf"), "--x-scale", "2")
    # Width follows total_km / x_scale, NOT the old 2.0 floor.
    assert abs(w - total_km / 2.0) < 0.05, f"width {w} not data-driven"
    assert w < 1.9, "width still clamped near 2.0"


def test_x_scale_responds_to_value(tmp_path):
    seg, total_km, _ = _short_segy(tmp_path)
    w_half, _ = _run_export(seg, str(tmp_path / "x05.pdf"), "--x-scale", "0.5")
    # A smaller km/in scale → a physically WIDER figure (was impossible under clamp).
    assert abs(w_half - total_km / 0.5) < 0.1
    assert w_half > 4.0


def test_default_width_is_trace_based(tmp_path):
    """Task 2: with no --x-scale the width is now (n_traces·px_per_trace)/dpi —
    the 8-in floor is removed, so every trace maps to exactly px_per_trace px."""
    seg, _total_km, n_traces = _short_segy(tmp_path)
    # Use a high px/trace so the figure is wide enough to render cleanly.
    w, _h = _run_export(seg, str(tmp_path / "def.pdf"),
                        "--px-per-trace", "60", "--dpi", "600")
    assert abs(w - n_traces * 60 / 600) < 0.05            # data-driven, no floor


def test_ve_is_length_independent_with_x_and_y_scale(tmp_path):
    """VE consistency: --x-scale + --y-scale gives VE = x·2e6/(v·y), independent
    of the profile length — the geometric proportions match longer profiles."""
    seg, total_km, _ = _short_segy(tmp_path)
    w, h = _run_export(seg, str(tmp_path / "ve.pdf"),
                       "--x-scale", "0.5", "--y-scale", "18")
    # horizontal km/in == x_scale (only true once the clamp is gone)
    assert abs(total_km / w - 0.5) < 0.02
    # closed-form VE depends only on x_scale, y_scale, velocity
    ve = (0.5 * 2_000_000) / (1500.0 * 18.0)
    assert abs(ve - 37.0) < 1.0


def test_ve_flag_gives_identical_height_across_lengths(tmp_path):
    """--x-scale + --ve fixes the vertical exaggeration: two lines of DIFFERENT
    length get the SAME figheight (= VE·depth_km/x_scale, length-independent),
    while the width still follows each line's km. This is the fix for the
    --ratio deformation (which couples height to width → length)."""
    # Short and long lines, SAME acquisition window (ns/dt) → same depth_km.
    short = make_synthetic_segy(str(tmp_path / "s.sgy"), n_traces=60, ns=1000,
                                base_lon=-3.0, base_lat=43.0, lon_step=0.0005)
    long_ = make_synthetic_segy(str(tmp_path / "l.sgy"), n_traces=400, ns=1000,
                                base_lon=-3.0, base_lat=43.0, lon_step=0.0010)
    w_s, h_s = _run_export(short, str(tmp_path / "s.pdf"), "--x-scale", "2", "--ve", "10")
    w_l, h_l = _run_export(long_, str(tmp_path / "l.pdf"), "--x-scale", "2", "--ve", "10")

    # Heights identical regardless of length / trace count.
    assert abs(h_s - h_l) < 0.02, f"VE height not length-independent: {h_s} vs {h_l}"
    # Matches the closed form h = VE · depth_km / x_scale (record 250 ms @ 1500 m/s).
    depth_km = (1000 * 250 / 1000.0) * 1500.0 / 2_000_000
    assert abs(h_s - 10 * depth_km / 2) < 0.05
    # Widths DO differ (longer line → wider paper at the same km/in).
    assert w_l > w_s


def test_max_aspect_clamps_only_extreme_lines(tmp_path):
    """--max-aspect is the 'infinite-noodle' guard on the --ve path: a normal-
    length line keeps the exact VE (aspect under the limit), while an extremely
    long line is locked at the max aspect (VE overridden only when necessary)."""
    short = make_synthetic_segy(str(tmp_path / "s.sgy"), n_traces=60, ns=1000,
                                base_lon=-3.0, base_lat=43.0, lon_step=0.0005)
    long_ = make_synthetic_segy(str(tmp_path / "l.sgy"), n_traces=400, ns=1000,
                                base_lon=-3.0, base_lat=43.0, lon_step=0.0100)
    w_s, h_s = _run_export(short, str(tmp_path / "s.pdf"),
                           "--x-scale", "2", "--ve", "67", "--max-aspect", "5")
    w_l, h_l = _run_export(long_, str(tmp_path / "l.pdf"),
                           "--x-scale", "2", "--ve", "67", "--max-aspect", "5")
    # Short line: under the limit → VE preserved (h = VE·depth_km/x_scale).
    depth_km = (1000 * 250 / 1000.0) * 1500.0 / 2_000_000
    assert abs(h_s - 67 * depth_km / 2) < 0.05
    assert w_s / h_s < 5.0                       # not clamped
    # Long line: would be a noodle → locked exactly at the 5:1 ceiling.
    assert abs(w_l / h_l - 5.0) < 0.02
