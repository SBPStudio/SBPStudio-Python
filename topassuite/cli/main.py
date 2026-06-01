"""
cli/main.py — Entry point for the topassuite CLI.

Usage:
    python -m topassuite.cli.main <subcommand> [options]

Subcommands:
    info          Print SEG-Y metadata
    reproject     Reproject one or more SEG-Y files to a new CRS
    join-chain    Reproject and join a chain of profiles into one SEG-Y
    export-image  Export seismic profile(s) or chain as an image
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
        prog="python -m topassuite.cli.main",
        description="TOPAS Suite — headless SEG-Y CLI tool",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── info ─────────────────────────────────────────────────────────────────
    pi = sub.add_parser("info", help="Print SEG-Y file metadata")
    pi.add_argument("files", nargs="+", metavar="FILE")
    pi.add_argument("--json", action="store_true", help="Output as JSON array")

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
    pj.add_argument("--dst", required=True, metavar="EPSG",
                    help="Destination CRS (use same as --src for pure join)")
    pj.add_argument("--src", default=None, metavar="EPSG",
                    help="Source CRS (auto-detected if omitted)")
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
    pe.add_argument("--clip", type=float, default=99.0, metavar="P",
                    help="Clip percentile for colour scaling (default: 99)")
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
    # ── Visualisation options ──────────────────────────────────────────────────
    pe.add_argument("--x-tick", type=float, default=None, metavar="KM",
                    dest="x_tick",
                    help="Place distance grid-ticks every KM km on the x-axis.")
    pe.add_argument("--t-tick", type=float, default=None, metavar="MS",
                    dest="t_tick",
                    help="Place time grid-ticks every MS milliseconds on the y-axis.")
    pe.add_argument("--grid", action="store_true",
                    help="Overlay a semi-transparent grid on the image.")
    pe.add_argument("--no-axes", action="store_true", dest="no_axes",
                    help="Save pure data pixels — no axes, labels, colorbar or "
                         "matplotlib margins. Guarantees exact px-per-trace "
                         "mapping with zero wasted pixels.")
    pe.add_argument("--title", default=None, metavar="TEXT",
                    help="Override the auto-generated figure title.")
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


def main(argv=None) -> None:
    from .commands import (
        cmd_info, cmd_reproject, cmd_join_chain,
        cmd_export_image, cmd_spectrum, cmd_navline, cmd_fix,
        cmd_accel,
    )

    parser = build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "info":         cmd_info,
        "reproject":    cmd_reproject,
        "join-chain":   cmd_join_chain,
        "export-image": cmd_export_image,
        "spectrum":     cmd_spectrum,
        "navline":      cmd_navline,
        "fix":          cmd_fix,
        "accel":        cmd_accel,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
