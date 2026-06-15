"""
test_export_mem_safety.py — the rasteriser RAM safeguard (--mem-budget-gb).

Very long / high-DPI lines used to raise numpy ArrayMemoryError because the
output RGBA array (figw·dpi × figh·dpi) exceeded RAM. `_figsize` now applies a
COUPLED down-scale: both figure dimensions are multiplied by a single factor
s = √(budget / requested) so the image fits the pixel budget while the aspect
ratio AND the vertical exaggeration (both functions of w/h only) stay identical.

These tests pin:
  * the byte→pixel budget maths,
  * a tiny --mem-budget-gb triggers the clamp and lands ON budget (max quality),
  * the aspect ratio is preserved through the clamp,
  * a generous budget is an exact pass-through (no scaling, Zero-regression).

NOTE: aspect is asserted from the PIXEL dimensions (full-precision int(w·dpi)),
NOT the "in" values — the CLI prints inches rounded to 1 decimal, which is far
too lossy at small figure sizes to verify proportions.
"""
from __future__ import annotations

import re
import subprocess
import sys

from tests.make_synthetic_segy import make_synthetic_segy

# "OK <file>  WxH px, <w>x<h> in @ <dpi> DPI"
_DIMS_RE = re.compile(r"(\d+)x(\d+)\s*px,\s*([\d.]+)x([\d.]+)\s*in\s*@\s*(\d+)\s*DPI")
_SAFE_RE = re.compile(r"RAM-SAFE")


def _run(seg_path: str, out_path: str, *extra: str):
    cmd = [sys.executable, "-m", "sbp_studio.cli.main", "export-image", seg_path,
           "--velocity", "1500", "--format", "png", "--quality", "high",
           "--out", out_path, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    m = _DIMS_RE.search(r.stderr)
    assert m, f"no export dims found (rc={r.returncode}).\nSTDERR:\n{r.stderr}"
    w_px, h_px = int(m.group(1)), int(m.group(2))
    return dict(w_px=w_px, h_px=h_px, aspect=w_px / h_px,
                px=w_px * h_px, dpi=int(m.group(5)),
                clamped=bool(_SAFE_RE.search(r.stderr)), stderr=r.stderr)


def _segy(tmp_path):
    # Sized so the 600-DPI --x-scale 2 --ratio 3 request is ≈32 Mpx: small enough
    # that a 64 GB budget passes through, large enough that 0.02/0.08 GB clamp.
    # 2000 traces × 0.0002° lon ≈ 32.6 km → w=16.3 in, h=5.4 in → ~9770×3258 px.
    return make_synthetic_segy(str(tmp_path / "m.sgy"), n_traces=2000, ns=1000,
                               base_lon=-3.0, base_lat=43.0, lon_step=0.0002)


def test_budget_pixel_maths():
    from sbp_studio.cli.commands import _safe_pixel_budget, _RASTER_BYTES_PER_PX
    max_px, budget_bytes = _safe_pixel_budget(1.0)        # exactly 1 GB
    assert abs(budget_bytes - 1024 ** 3) < 1
    assert abs(max_px - (1024 ** 3) / _RASTER_BYTES_PER_PX) < 1


def test_gui_dpi_cap_matches_cli_budget_and_bounds_pixels():
    """The GUI export DPI cap (dpi_for_budget) uses the SAME bytes/px as the CLI
    and bounds the raster to the budget — this is the fix for the GUI export hang
    (effective_export_dpi could otherwise push a long/deep line to gigapixels)."""
    from sbp_studio.gui.tabs._render import (
        dpi_for_budget, pixel_budget, RASTER_BYTES_PER_PX)
    from sbp_studio.cli.commands import _RASTER_BYTES_PER_PX
    assert RASTER_BYTES_PER_PX == _RASTER_BYTES_PER_PX     # GUI/CLI parity
    figsize = (28.3, 9.4)                                  # the hang scenario
    cap = dpi_for_budget(figsize, 6.0)
    px = (figsize[0] * cap) * (figsize[1] * cap)
    assert px <= pixel_budget(6.0) * 1.001                # bounded by the budget
    # No budget (0 / None) → no cap.
    assert dpi_for_budget(figsize, 0) is None
    assert dpi_for_budget(figsize, None) is None


def test_tiny_budget_clamps_to_budget_and_preserves_aspect(tmp_path):
    seg = _segy(tmp_path)
    r = _run(seg, str(tmp_path / "clamped.png"),
             "--x-scale", "2", "--ratio", "3", "--mem-budget-gb", "0.02")
    assert r["clamped"], f"expected the RAM-SAFE clamp to engage\n{r['stderr']}"
    # Lands at/under budget (0.02 GB / 18 B-per-px ≈ 1.19 Mpx) and uses most of it
    # (max-quality fit): between 85% and 100% of the ceiling.
    budget_px = 0.02 * 1024 ** 3 / 18.0
    assert r["px"] <= budget_px * 1.001
    assert r["px"] >= budget_px * 0.85
    # Aspect preserved at the requested 3:1 (asserted from full-precision pixels).
    assert abs(r["aspect"] - 3.0) < 0.05


def test_generous_budget_is_passthrough(tmp_path):
    seg = _segy(tmp_path)
    r = _run(seg, str(tmp_path / "big.png"),
             "--x-scale", "2", "--ratio", "3", "--mem-budget-gb", "64")
    assert not r["clamped"], f"64 GB budget must not clamp ~32 Mpx\n{r['stderr']}"
    assert abs(r["aspect"] - 3.0) < 0.05


def test_smaller_budget_yields_fewer_pixels_same_aspect(tmp_path):
    seg = _segy(tmp_path)
    a = _run(seg, str(tmp_path / "a.png"),
             "--x-scale", "2", "--ratio", "3", "--mem-budget-gb", "0.08")
    b = _run(seg, str(tmp_path / "b.png"),
             "--x-scale", "2", "--ratio", "3", "--mem-budget-gb", "0.02")
    assert a["clamped"] and b["clamped"]
    # A 4× smaller budget → ~4× fewer pixels (≈2× each axis), aspect unchanged.
    assert b["px"] < a["px"]
    assert abs(a["aspect"] - b["aspect"]) < 0.05
