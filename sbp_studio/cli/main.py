"""
cli/main.py — Entry point for the sbp_studio CLI.

Usage:
    python -m sbp_studio.cli.main <subcommand> [options]

Subcommands:
    info          Print SEG-Y metadata
    check         Inspect headers + scan a SEG-Y file for anomalies
    patch-header  Patch the dt binary header field in place (segyio r+)
    process       Run a headless DSP pipeline on a file → new SEG-Y
    reproject     Reproject one or more SEG-Y files to a new CRS
    join-chain    Reproject and join a chain of profiles into one SEG-Y
    export-image  Export seismic profile(s) or chain as an image
    batch-export  Render many SEG-Y files/dirs into one output folder
    spectrum      Compute and export the frequency spectrum figure
    navline       Export the navigation track as SHP/GeoJSON/CSV
    fix           Export FIX-point marks as SHP/GeoJSON/CSV

Progress is written to stderr. Errors produce a non-zero exit code.
No GUI imports; runs without a DISPLAY.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sbp_studio.cli.main",
        description="SBP Studio — headless SEG-Y CLI tool",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── info ─────────────────────────────────────────────────────────────────
    pi = sub.add_parser("info", help="Print SEG-Y file metadata")
    pi.add_argument("files", nargs="+", metavar="FILE")
    pi.add_argument("--json", action="store_true", help="Output as JSON array")

    # ── check ────────────────────────────────────────────────────────────────
    pc = sub.add_parser("check",
                        help="Inspect headers + scan a SEG-Y file for anomalies")
    pc.add_argument("files", nargs="+", metavar="FILE")
    pc.add_argument("--full-text", action="store_true", dest="full_text",
                    help="Print all 40 EBCDIC cards (default: first 6).")
    pc.add_argument("--no-stats", action="store_true", dest="no_stats",
                    help="Header-only: skip loading traces (no amplitude stats).")

    # ── patch-header ─────────────────────────────────────────────────────────
    pp = sub.add_parser("patch-header",
                        help="Patch the dt binary header field IN PLACE (segyio r+)")
    pp.add_argument("file", metavar="FILE")
    pp.add_argument("--dt", type=int, default=None, metavar="US",
                    help="New sample interval in microseconds. Also mass-propagated "
                         "to every trace header (TRACE_SAMPLE_INTERVAL). NOTE: ns "
                         "(samples/trace) is intentionally NOT patchable here — "
                         "changing it without resizing the trace data blocks would "
                         "corrupt the file (every trace boundary would misalign).")
    pp.add_argument("--text", metavar="TXTFILE", default=None,
                    help="Replace the 3200-byte EBCDIC textual header from a UTF-8 "
                         "text file (≤40 lines × 80 chars).")
    pp.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="Print what would change without modifying the file.")

    # ── reproject ─────────────────────────────────────────────────────────────
    pr = sub.add_parser("reproject", help="Reproject SEG-Y file(s) to a new CRS")
    pr.add_argument("files", nargs="+", metavar="FILE")
    pr.add_argument("--src", required=True, metavar="EPSG",
                    help="Source CRS (e.g. EPSG:4326 or bare integer 4326)")
    pr.add_argument("--dst", required=True, metavar="EPSG",
                    help="Destination CRS")
    pr.add_argument("--unit-hint", type=int, default=2, metavar="N",
                    dest="unit_hint",
                    help="Coord unit hint when header field is 0: "
                         "1=m/ft, 2=arc-sec (default), 3=decimal degrees")
    pr.add_argument("--out-dir", metavar="DIR", default=None,
                    dest="out_dir",
                    help="Output directory (default: same as input)")

    # ── join-chain ────────────────────────────────────────────────────────────
    pj = sub.add_parser("join-chain",
                        help="Reproject + join a chain of profiles into one SEG-Y")
    pj.add_argument("files", nargs="+", metavar="FILE")
    pj.add_argument("--dst", required=False, default=None, metavar="EPSG",
                    help="Destination CRS. Omit (or use with --no-reproject) "
                         "for a pure join with no coordinate change.")
    pj.add_argument("--src", default=None, metavar="EPSG",
                    help="Source CRS (auto-detected if omitted)")
    pj.add_argument("--no-reproject", action="store_true", dest="no_reproject",
                    help="Pure join: copy headers verbatim, no coordinate "
                         "transformation. ~3x faster than --src X --dst X. "
                         "Automatically selected when --src == --dst.")
    pj.add_argument("--gap-km", type=float, default=None, dest="gap_km",
                    metavar="K",
                    help="Maximum inter-profile gap to be considered contiguous (km)")
    pj.add_argument("--unit-hint", type=int, default=2, metavar="N",
                    dest="unit_hint")
    pj.add_argument("--out", metavar="PATH", default=None,
                    help="Output file path (default: auto-generated)")

    # ── export-image ──────────────────────────────────────────────────────────
    pe = sub.add_parser("export-image",
                        help="Export profile(s)/chain as image")
    pe.add_argument("files", nargs="+", metavar="FILE")
    pe.add_argument("--chain", action="store_true",
                    help="Detect chains and export each chain as one image")
    pe.add_argument("--preset", metavar="KEY", default=None,
                    help="Filter preset key (e.g. 'envelope', 'topas_narrow')")
    pe.add_argument("--bandpass", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None,
                    help="Bandpass filter: low and high cutoff frequencies in Hz")
    pe.add_argument("--agc", action="store_true",
                    help="Enable AGC (automatic gain control)")
    pe.add_argument("--tvg", type=float, default=None, metavar="ALPHA",
                    help="Enable TVG with exponential alpha")
    pe.add_argument("--align", action="store_true",
                    help="Compensate delay recording times")
    pe.add_argument("--clip", type=float, default=99.6, metavar="P",
                    help="Clip percentile for colour scaling (default: 99.6)")
    pe.add_argument("--cmap", metavar="NAME", default=None,
                    help="Colormap name: 'Viridis', 'Inferno', 'Jet', 'Greys', 'Terrain'")
    pe.add_argument("--invert", action="store_true",
                    help="Invert the colormap")
    pe.add_argument("--fix", type=int, default=None, metavar="MIN",
                    help="Show FIX marks at this interval (minutes)")
    # ── Quality / resolution ──────────────────────────────────────────────────
    pe.add_argument("--quality", choices=["screen", "print", "high", "ultra"],
                    default=None,
                    help="Quality preset (sets DPI; explicit --dpi overrides):\n"
                         "  screen=150 DPI (fast, for screen review)\n"
                         "  print=300 DPI  (good, standard plotter)\n"
                         "  high=600 DPI   (sharp, high-res plotter)\n"
                         "  ultra=900 DPI  (max quality, large files)")
    pe.add_argument("--dpi", type=int, default=None, metavar="N",
                    help="Output resolution in DPI. Overrides --quality. "
                         "Default: 150 (or quality preset value).")
    pe.add_argument("--px-per-trace", type=float, default=2.0,
                    dest="px_per_trace", metavar="N",
                    help="Pixels per trace — horizontal scale (default: 2). "
                         "Use 1 for exact 1:1 trace-to-pixel mapping.")
    pe.add_argument("--figheight", type=float, default=None, metavar="H",
                    help="Figure height in inches. Default: 7.0, or computed "
                         "automatically when --auto-height is set.")
    pe.add_argument("--auto-height", action="store_true", dest="auto_height",
                    help="Compute figure height automatically so that each sample "
                         "maps to approximately 1 vertical pixel "
                         "(figheight = ns / dpi). Combined with --quality print "
                         "--px-per-trace 1 gives full-resolution output.")
    pe.add_argument("--fill-zero", action="store_true", dest="fill_zero",
                    help="When --align is active, fill delay gaps with 0 "
                         "(white in Greys) instead of NaN.")
    # ── Physical scale (overrides --px-per-trace / --figheight / --auto-height) ─
    pe.add_argument("--x-scale", type=float, default=None, dest="x_scale",
                    metavar="KM_PER_IN",
                    help="Horizontal physical scale in km per inch.\n"
                         "figwidth = total_km / x_scale.\n"
                         "Typical SBP values: 1-4 km/in.")
    pe.add_argument("--y-scale", type=float, default=None, dest="y_scale",
                    metavar="MS_PER_IN",
                    help="Vertical scale in ms per inch.\n"
                         "figheight = record_ms / y_scale.\n"
                         "Typical SBP: 25-50. Can be used alone or with --x-scale.")
    pe.add_argument("--velocity", type=float, default=1500.0, dest="velocity",
                    metavar="M_S",
                    help="Sound velocity in m/s for depth conversion (default: 1500).\n"
                         "Used with --x-scale (and no --y-scale / --ratio) to compute\n"
                         "a physically consistent figheight:\n"
                         "  depth_km = record_ms × velocity / 2_000_000\n"
                         "  figheight = depth_km / x_scale  (VE=1, true scale)\n"
                         "Also prints the vertical exaggeration (VE) whenever set.\n"
                         "Display axis labels remain in milliseconds.")
    pe.add_argument("--ratio", type=float, default=None, dest="ratio",
                    metavar="W_H",
                    help="Target width:height aspect ratio of the figure.\n"
                         "figheight = figwidth / ratio.\n"
                         "Works with or without --x-scale.\n"
                         "Examples: --ratio 3  →  3:1 (typical seismic)\n"
                         "          --ratio 4  →  4:1 (wider, similar to TOPAS SW)\n"
                         "Overrides --y-scale, --auto-height and --figheight.\n"
                         "Print VE info when used with --velocity.")
    pe.add_argument("--ve", type=float, default=None, dest="ve", metavar="N",
                    help="FIXED vertical exaggeration — the length-independent way "
                         "to keep many lines visually comparable.\n"
                         "Requires --x-scale. Sets figheight = VE · depth_km / "
                         "x_scale, so the vertical scale (and VE) is IDENTICAL for "
                         "every line regardless of its length or trace count, while "
                         "--x-scale keeps the horizontal km/in constant.\n"
                         "Overrides --ratio and --y-scale.\n"
                         "Example: --x-scale 2 --ve 67  → 2 km/in horizontal, "
                         "VE=67× on every line.\n"
                         "(Equivalent --y-scale = x_scale·2e6 / (velocity·VE).)")
    pe.add_argument("--max-aspect", type=float, default=None, dest="max_aspect",
                    metavar="R",
                    help="'Infinite-noodle' safety limit for --ve. A constant VE "
                         "makes extremely long lines very wide vs tall; if the "
                         "width:height ratio would exceed R, the height is raised "
                         "to lock the aspect at R:1 (overriding the VE ONLY for "
                         "that extreme line, to keep the PDF usable). Lines below "
                         "the limit keep the exact --ve. Only affects the --ve path.\n"
                         "Example: --x-scale 2 --ve 67 --max-aspect 5")
    # ── RAM safety (prevents ArrayMemoryError on very long / high-DPI lines) ────
    pe.add_argument("--mem-budget-gb", type=float, default=6.0,
                    dest="mem_budget_gb", metavar="GB",
                    help="Peak RAM (GB) the rasteriser is allowed to use (default: "
                         "6). If the requested figure would exceed it, BOTH figure "
                         "dimensions are scaled down by a single factor so the image "
                         "fits — this preserves the aspect ratio AND the vertical "
                         "exaggeration (VE) exactly, only lowering pixel density. "
                         "Before rendering, the budget is checked against the "
                         "machine's actual free RAM and a prominent warning is "
                         "printed if it exceeds it. Raise it to squeeze more "
                         "sharpness out of a machine with free RAM; lower it to be "
                         "conservative. Pass 0 to fall back to ~55%% of free RAM.")
    # ── Bottom X-axis mode (distance vs trace) ─────────────────────────────────
    xg = pe.add_mutually_exclusive_group()
    xg.add_argument("--trace-axis", action="store_const", const="trace",
                    dest="x_axis",
                    help="Trace-based export: draw ONE column per trace "
                         "(× --px-per-trace), IGNORING the horizontal km scale and "
                         "--x-scale. The bottom axis reads in trace number; distance "
                         "in km varies along the profile (it depends on the ping "
                         "rate). No horizontal stretching/elongation of traces.")
    xg.add_argument("--km-no-stretch", action="store_const", const="km",
                    dest="x_axis",
                    help="Distance (km) on the bottom axis WITHOUT pixel "
                         "filling/stretching: the data is still drawn one column per "
                         "trace (no horizontal interpolation between traces), but km "
                         "labels are placed at the real trace positions. Prevents the "
                         "trace-elongation effect on variable ping-rate lines.")
    pe.set_defaults(x_axis="distance")
    # ── Visualisation options ──────────────────────────────────────────────────
    pe.add_argument("--x-tick", type=float, default=None, metavar="KM",
                    dest="x_tick",
                    help="Place distance grid-ticks every KM km on the x-axis.")
    pe.add_argument("--t-tick", type=float, default=None, metavar="MS",
                    dest="t_tick",
                    help="Place time grid-ticks every MS milliseconds on the y-axis.")
    pe.add_argument("--time-ticks", type=int, default=None, dest="time_ticks",
                    metavar="MIN",
                    help="Add secondary x-axis at TOP with UTC timestamps every MIN min.")
    pe.add_argument("--time-fmt",
                    choices=["hhmm", "fix", "position", "datetime", "full"],
                    default="hhmm", dest="time_fmt",
                    help="Content of top time-axis labels (default: hhmm):\n"
                         "  hhmm:     \"10:30\"\n"
                         "  fix:      \"#12  10:30\"\n"
                         "  position: \"10:30\\n43.1234N  8.5678W\"\n"
                         "  datetime: \"02/06/2026\\n10:30\"\n"
                         "  full:     \"#12  10:30\\n02/06/2026  43.1234N  8.5678W\"")
    pe.add_argument("--time-font-size", type=float, default=6.0,
                    dest="time_font_size", metavar="PT",
                    help="Font size for top time-axis labels (default: 6.0 pt).")
    pe.add_argument("--time-align",
                    choices=["left", "center", "right"],
                    default="left", dest="time_align",
                    help="Horizontal alignment of time labels (default: left).")
    pe.add_argument("--fix-font-size", type=float, default=5.0,
                    dest="fix_font_size", metavar="PT",
                    help="Font size for FIX-mark number labels inside the image "
                         "(default: 5.0 pt).")
    pe.add_argument("--fix-bbox-alpha", type=float, default=0.12,
                    dest="fix_bbox_alpha", metavar="A",
                    help="Opacity of the background box behind FIX labels "
                         "(0.0 = no box, default: 0.12).")
    pe.add_argument("--fix-color", default=None, dest="fix_color", metavar="HEX",
                    help="Colour for FIX-mark lines and number labels.\n"
                         "Defaults to the theme highlight colour.\n"
                         "Example: --fix-color '#cc4444'  (soft red)")
    # ── Margins ───────────────────────────────────────────────────────────────
    pe.add_argument("--margin-top", type=float, default=0.0, dest="margin_top",
                    metavar="MS",
                    help="Zero-filled margin to add ABOVE the record (ms).\n"
                         "Renders as white (Greys) / min-amplitude space.\n"
                         "Useful to show context above the first reflector.")
    pe.add_argument("--margin-bottom", type=float, default=0.0,
                    dest="margin_bottom", metavar="MS",
                    help="Zero-filled margin to add BELOW the record (ms).")
    # ── Grid ─────────────────────────────────────────────────────────────────
    pe.add_argument("--grid", action="store_true",
                    help="Overlay a semi-transparent grid on the image.")
    pe.add_argument("--no-axes", action="store_true", dest="no_axes",
                    help="Save pure data pixels — no axes, labels, colorbar or "
                         "matplotlib margins. Guarantees exact px-per-trace "
                         "mapping with zero wasted pixels.")
    pe.add_argument("--title", default=None, metavar="TEXT",
                    help="Override the auto-generated figure title.")
    # ── Colour / theme ────────────────────────────────────────────────────────
    pe.add_argument("--theme", choices=["dark", "light", "print"],
                    default="dark",
                    help="Colour theme (default: dark):\n"
                         "  dark:  dark navy bg, light text (matches SBP GUI)\n"
                         "  light: light grey bg, dark text\n"
                         "  print: pure white bg, black text (best for paper/PDF)")
    pe.add_argument("--bg-color", default=None, dest="bg_color", metavar="HEX",
                    help="Override figure/panel background colour (e.g. '#ffffff').")
    pe.add_argument("--text-color", default=None, dest="text_color", metavar="HEX",
                    help="Override all text and tick label colour.")
    pe.add_argument("--axes-bg-color", default=None, dest="axes_bg_color",
                    metavar="HEX",
                    help="Override seismic axes background colour.")
    pe.add_argument("--clip-lo", type=float, default=0.0, dest="clip_lo",
                    metavar="P",
                    help="Lower clip percentile for colour scaling (default: 0). "
                         "Raise to suppress low-amplitude noise floor.")
    # ── Output ────────────────────────────────────────────────────────────────
    pe.add_argument("--format", choices=["png", "pdf", "tif", "svg"],
                    default="png", dest="format",
                    help="Output image format (default: png)")
    pe.add_argument("--pdf-page",
                    choices=["auto", "A0", "A1", "A2", "A3", "A4"],
                    default="auto", dest="pdf_page", metavar="SIZE",
                    help="PDF page size (only used when --format pdf).\n"
                         "  auto  = natural size: px/dpi gives physical inches\n"
                         "  A0-A4 = scale to fit within paper (landscape),\n"
                         "          aspect ratio preserved.\n"
                         "With --no-axes: img2pdf (lossless, zero recompression).\n"
                         "With axes    : matplotlib PDF (vector text/lines + \n"
                         "               raster image), page scaled to fit paper.")
    pe.add_argument("--out", metavar="PATH", default=None,
                    help="Output file path (default: auto-generated)")
    pe.add_argument("--timeit", action="store_true",
                    help="Print wall-clock timing breakdown for each stage "
                         "(loading / processing / rendering / saving).")

    # ── process (headless DSP pipeline) ───────────────────────────────────────
    ppr = sub.add_parser(
        "process",
        help="Run a headless DSP pipeline on a file and write a new SEG-Y",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Apply an ordered DSP pipeline to one SEG-Y file and write a new\n"
            "SEG-Y with the processed samples (all geometry/headers preserved).\n\n"
            "Pipeline ops (comma-separated, applied left→right):\n"
            "  bandpass(flo_hz,fhi_hz)            Butterworth band-pass\n"
            "  whiten(flo_hz,fhi_hz,smooth_hz)    spectral whitening (zero-phase)\n"
            "  decon(op_ms,gap_ms,white_pct)      predictive deconvolution\n"
            "  swell(window_traces,max_shift_ms)  swell / heave correction\n"
            "  water_mute(threshold_pct,margin_ms) water-column mute\n"
            "  tvg(alpha)                         time-variant exponential gain\n"
            "  agc(window_ms)                     automatic gain control\n"
            "  preset(key)                        attribute/preset (e.g. envelope)\n"
            "  align()                            delay-recording-time alignment\n\n"
            "Example:\n"
            '  process in.sgy out.sgy --pipeline '
            '"bandpass(1000,8000),whiten(1000,8000,300),agc(200)"'))
    ppr.add_argument("input", metavar="INPUT")
    ppr.add_argument("output", metavar="OUTPUT")
    ppr.add_argument("--pipeline", required=True, metavar="SPEC",
                     help='Comma-separated op list, e.g. '
                          '"bandpass(1000,8000),agc(200)".')
    ppr.add_argument("--timeit", action="store_true",
                     help="Print a per-stage timing breakdown.")

    # ── batch-export ──────────────────────────────────────────────────────────
    pb = sub.add_parser(
        "batch-export",
        help="Render many SEG-Y files/dirs into one output folder")
    pb.add_argument("inputs", nargs="+", metavar="PATH",
                    help="SEG-Y files and/or directories (scanned for *.sgy/*.seg).")
    pb.add_argument("--out", required=True, metavar="DIR",
                    help="Output directory (one image per input file).")
    pb.add_argument("--format", choices=["png", "pdf", "tif", "svg"],
                    default="pdf", dest="format",
                    help="Output format (default: pdf).")
    pb.add_argument("--cmap", metavar="NAME", default=None,
                    help="Colormap (e.g. seismc, bwr, Viridis, Greys).")
    pb.add_argument("--ve", type=float, default=None, dest="ve", metavar="N",
                    help="Fixed, length-independent vertical exaggeration "
                         "(requires --x-scale).")
    pb.add_argument("--x-scale", type=float, default=None, dest="x_scale",
                    metavar="KM_PER_IN",
                    help="Horizontal physical scale in km/inch (anchors --ve).")
    pb.add_argument("--velocity", type=float, default=1500.0, dest="velocity",
                    metavar="M_S", help="Sound velocity for depth/VE (default 1500).")
    pb.add_argument("--preset", metavar="KEY", default=None,
                    help="Filter preset / attribute key (e.g. 'envelope').")
    pb.add_argument("--bandpass", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None, help="Band-pass low/high cutoff in Hz.")
    pb.add_argument("--agc", action="store_true", help="Enable AGC.")
    pb.add_argument("--align", action="store_true",
                    help="Compensate delay recording times.")
    pb.add_argument("--clip", type=float, default=99.6, metavar="P",
                    help="Clip percentile for colour scaling (default 99.6).")
    pb.add_argument("--invert", action="store_true", help="Invert the colormap.")
    pb.add_argument("--quality", choices=["screen", "print", "high", "ultra"],
                    default="print",
                    help="DPI preset (default: print = 300 DPI).")
    pb.add_argument("--dpi", type=int, default=None, metavar="N",
                    help="Explicit DPI (overrides --quality).")
    pb.add_argument("--theme", choices=["dark", "light", "print"], default="print",
                    help="Colour theme (default: print).")
    pb.add_argument("--no-axes", action="store_true", dest="no_axes",
                    help="Save pure data pixels (no axes/labels/colorbar).")
    pb.add_argument("--mem-budget-gb", type=float, default=6.0,
                    dest="mem_budget_gb", metavar="GB",
                    help="Peak rasteriser RAM budget (default 6).")
    pb.add_argument("--timeit", action="store_true",
                    help="Print per-stage timing for each file.")
    # Fill every attribute cmd_export_image reads directly so the shared engine
    # never hits an AttributeError on a flag batch-export does not expose.
    pb.set_defaults(tvg=None, fill_zero=False, fix=None, chain=False,
                    px_per_trace=2.0, figheight=None, auto_height=False,
                    y_scale=None, ratio=None, max_aspect=None, x_axis="distance",
                    pdf_page="auto", clip_lo=0.0,
                    x_tick=None, t_tick=None, time_ticks=None, grid=False,
                    title=None, margin_top=0.0, margin_bottom=0.0,
                    time_fmt="hhmm", time_font_size=6.0, time_align="left",
                    fix_font_size=5.0, fix_bbox_alpha=0.12, fix_color=None,
                    bg_color=None, text_color=None, axes_bg_color=None)

    # ── spectrum ──────────────────────────────────────────────────────────────
    ps = sub.add_parser("spectrum",
                        help="Compute and export the frequency spectrum figure")
    ps.add_argument("file", metavar="FILE",
                    help="SEG-Y file to analyse")
    ps.add_argument("--format", choices=["png", "pdf"], default="png",
                    dest="format")
    ps.add_argument("--out", metavar="PATH", default=None)

    # ── navline ───────────────────────────────────────────────────────────────
    pn = sub.add_parser("navline",
                        help="Export navigation track as SHP/GeoJSON/CSV")
    pn.add_argument("files", nargs="+", metavar="FILE")
    pn.add_argument("--chain", action="store_true",
                    help="Export each detected chain as one navline")
    pn.add_argument("--format", required=True, choices=["shp", "geojson", "csv"],
                    dest="format")
    pn.add_argument("--crs", metavar="EPSG", default=None,
                    help="Reproject navline to this output CRS")
    pn.add_argument("--attrs", action="store_true", default=True,
                    help="Include per-vertex attributes (dist_km, water_depth, timestamp)")
    pn.add_argument("--no-attrs", action="store_false", dest="attrs")
    pn.add_argument("--out", metavar="PATH", default=None)

    # ── fix ───────────────────────────────────────────────────────────────────
    pf = sub.add_parser("fix",
                        help="Export FIX-point marks as SHP/GeoJSON/CSV")
    pf.add_argument("files", nargs="+", metavar="FILE")
    pf.add_argument("--chain", action="store_true")
    pf.add_argument("--interval", required=True, type=int, metavar="MIN",
                    help="FIX interval in minutes")
    pf.add_argument("--format", required=True, choices=["shp", "geojson", "csv"],
                    dest="format")
    pf.add_argument("--out", metavar="PATH", default=None)

    # ── accel ──────────────────────────────────────────────────────────────────
    sub.add_parser("accel",
                   help="Show active hardware acceleration (GPU, pyfftw, CPU workers)")

    return p


def _force_utf8_console() -> None:
    """Make stdout/stderr tolerate non-ASCII glyphs (µ, ✔, →, ⚠) on a Windows
    console whose default code page is cp1252. Best-effort: reconfigure to UTF-8
    with replacement so a stray glyph can never crash a command mid-run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+ TextIO
        except Exception:
            pass


def main(argv=None) -> None:
    _force_utf8_console()
    from .commands import (
        cmd_info, cmd_check, cmd_patch_header, cmd_process,
        cmd_reproject, cmd_join_chain,
        cmd_export_image, cmd_batch_export, cmd_spectrum, cmd_navline, cmd_fix,
        cmd_accel,
    )

    parser = build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "info":         cmd_info,
        "check":        cmd_check,
        "patch-header": cmd_patch_header,
        "process":      cmd_process,
        "reproject":    cmd_reproject,
        "join-chain":   cmd_join_chain,
        "export-image": cmd_export_image,
        "batch-export": cmd_batch_export,
        "spectrum":     cmd_spectrum,
        "navline":      cmd_navline,
        "fix":          cmd_fix,
        "accel":        cmd_accel,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
