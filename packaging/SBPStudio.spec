# -*- mode: python ; coding: utf-8 -*-
"""
SBPStudio.spec — PyInstaller build spec for SBP Studio 0.5.0.

Produces a single onedir distribution containing BOTH executables, which share
one bundled set of libraries and data files:

    dist/SBPStudio/
        SBPStudioGUI.exe      ← windowed PyQt6 desktop app  (sbp_studio.gui.app:main)
        SBPStudioCLI.exe      ← console command-line tool    (sbp_studio.cli.main:main)
        _internal/...         ← shared Python runtime, libs, assets, translations

CPU vs CUDA variant
-------------------
The CUDA variant is selected via an environment variable at BUILD time:

    # CPU build (default):
    pyinstaller packaging/SBPStudio.spec

    # CUDA build (cupy-cuda12x must be installed in the build env):
    $env:SBP_BUILD_CUDA = "1"        # PowerShell
    pyinstaller packaging/SBPStudio.spec

When SBP_BUILD_CUDA=1, CuPy and its CUDA runtime libraries are collected into
the bundle. The application still falls back to the CPU path on machines without
a compatible NVIDIA GPU/driver (see sbp_studio/core/_backends.py).

Onedir (not onefile) is intentional: pikepdf/qpdf, pyproj's PROJ data, segyio
and the Qt/OpenGL stack are far more reliable unpacked, and field startup is
faster.
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Conda DLL discovery ───────────────────────────────────────────────────────
# Native extensions installed via conda (pyproj→proj_9.dll, segyio, pikepdf→qpdf,
# Pillow, …) link against DLLs that live in <env>\Library\bin, NOT in
# site-packages. PyInstaller's binary-dependency walker only finds them if that
# directory is on PATH during analysis. Prepending it here makes the build work
# whether or not the conda env was "activated" before invoking PyInstaller, and
# lets PyInstaller pull in ONLY the transitive DLLs each extension actually needs
# (rather than us hard-coding a fragile list).
_conda_lib_bin = Path(sys.prefix) / "Library" / "bin"
if _conda_lib_bin.is_dir():
    os.environ["PATH"] = str(_conda_lib_bin) + os.pathsep + os.environ.get("PATH", "")
    print(f"[SBPStudio.spec] Added conda DLL dir to PATH: {_conda_lib_bin}")

# ── Paths ─────────────────────────────────────────────────────────────────────
# SPECPATH is injected by PyInstaller and points at this file's directory.
REPO_ROOT = Path(SPECPATH).resolve().parent
WITH_CUDA = os.environ.get("SBP_BUILD_CUDA", "0") == "1"

APP_NAME = "SBPStudio"

# ── Data files bundled into _internal/ ──────────────────────────────────────
# Layout must match the runtime path resolution:
#   * map_view._asset_path()  → <_internal>/assets/coastlines_highres.geojson
#                               (Path(__file__).parents[3] / "assets")
#   * i18n._TRANSLATIONS_DIR  → <_internal>/sbp_studio/gui/translations/
datas = [
    (str(REPO_ROOT / "assets" / "coastlines_highres.geojson"), "assets"),
    (str(REPO_ROOT / "sbp_studio" / "gui" / "translations"), "sbp_studio/gui/translations"),
]
binaries = []
hiddenimports = []

# ── Native-dependency collection (data dirs + dylibs the auto-hooks may miss) ─
for pkg in ("pyproj", "pikepdf", "segyio", "pyqtgraph"):
    try:
        b, d, h = collect_all(pkg)
        binaries += b
        datas += d
        hiddenimports += h
    except Exception:
        # If a package isn't importable in this env, its built-in hook (if any)
        # still runs during Analysis; skip rather than fail the whole build.
        pass

# img2pdf pulls these in; make the PDF backend explicit.
hiddenimports += ["img2pdf", "PIL.Image", "PIL.ImageCms"]

# Matplotlib loads its output backends lazily by name, so PyInstaller's static
# analysis misses them. viz/render.py uses Agg (rasterise) + the PDF backend
# (vector with-axes export); SVG is included for completeness.
hiddenimports += [
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
]

# ── Optional CUDA (CuPy) ────────────────────────────────────────────────────
if WITH_CUDA:
    try:
        b, d, h = collect_all("cupy")
        binaries += b
        datas += d
        hiddenimports += h
        # CuPy's own helpers + the CUDA Toolkit runtime libraries. The CTK libs
        # ship as nvidia-*-cu12 wheels (installed via cupy-cuda12x[ctk]) under
        # site-packages/nvidia/**/bin. CuPy loads these DLLs dynamically by name,
        # so PyInstaller's static analysis misses them — collect_all on each
        # nvidia.* subpackage pulls them into the bundle so the frozen app can
        # actually run GPU kernels (not just detect the device).
        cuda_pkgs = [
            "cupy_backends", "cupyx", "fastrlock",
            "nvidia",                       # umbrella; pulls the bin/ DLLs
            "nvidia.cuda_runtime", "nvidia.cuda_nvrtc", "nvidia.cublas",
            "nvidia.cufft", "nvidia.curand", "nvidia.cusolver",
            "nvidia.cusparse", "nvidia.nvjitlink",
        ]
        for sub in cuda_pkgs:
            try:
                b, d, h = collect_all(sub)
                binaries += b
                datas += d
                hiddenimports += h
            except Exception:
                pass
        print("[SBPStudio.spec] CUDA variant: CuPy + CUDA Toolkit runtime bundled.")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "SBP_BUILD_CUDA=1 but CuPy could not be collected. "
            "Install it first:  pip install -r requirements-cuda.txt\n"
            f"  underlying error: {exc}"
        )
# Never drag the test suite into the bundle.
excludes = ["pytest", "tests"]

if not WITH_CUDA:
    # cupy may be installed in the build env (the project's conda env ships it),
    # but the CPU variant must NOT bundle it: the `import cupy` statements in
    # core/_backends.py and core/processing.py are guarded at runtime, so
    # excluding them here keeps the CPU build clean and small. The runtime
    # detection simply reports "GPU not available" and uses the NumPy path.
    excludes += ["cupy", "cupyx", "cupy_backends", "fastrlock"]
    print("[SBPStudio.spec] CPU variant: CuPy excluded from bundle.")

# ── Analysis: GUI ─────────────────────────────────────────────────────────────
a_gui = Analysis(
    [str(REPO_ROOT / "applications" / "SBPStudio_GUI.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(REPO_ROOT / "packaging" / "rthook_utf8.py")],
    excludes=excludes,
    noarchive=False,
)

# ── Analysis: CLI ─────────────────────────────────────────────────────────────
a_cli = Analysis(
    [str(REPO_ROOT / "applications" / "SBPStudio_CLI.py")],
    pathex=[str(REPO_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(REPO_ROOT / "packaging" / "rthook_utf8.py")],
    excludes=excludes,
    noarchive=False,
)

pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)

exe_gui = EXE(
    pyz_gui,
    a_gui.scripts,
    [],
    exclude_binaries=True,
    name="SBPStudioGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed app; crashes are captured in app.log
    disable_windowed_traceback=False,
)

exe_cli = EXE(
    pyz_cli,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="SBPStudioCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,           # CLI needs a console
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe_gui,
    a_gui.binaries,
    a_gui.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
