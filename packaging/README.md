# Packaging SBP Studio 0.5.0

This folder holds everything needed to build a distributable, end-user copy of
SBP Studio with [PyInstaller](https://pyinstaller.org/). The result is a
**onedir** bundle (a folder you ship/zip) containing two executables that share
one bundled runtime:

```
SBPStudio-0.5.0-win64-cpu/
├── SBPStudioGUI.exe     # windowed PyQt6 desktop application
├── SBPStudioCLI.exe     # console command-line tool
└── _internal/           # shared Python runtime, libraries, assets, translations
```

The final package is written **outside the repository**, to
`T:\workspace_py\SBPStudio-Python\releases\` (one level above the repo root), so
build output is never committed. PyInstaller's working dir stays in `build\`
(git-ignored).

## Status of the 0.5.0 release

| Variant  | State                | Notes                                              |
|----------|----------------------|----------------------------------------------------|
| **CPU**  | ✅ shipped & verified | Includes the `pyfftw` FFT accelerator (active).    |
| **CUDA** | ⏳ pending            | Authoring is ready; must be built on a CUDA machine — see below. |

## Why PyInstaller + onedir (not onefile)

The dependency stack has native components and data directories that are far
more reliable unpacked, and onedir starts faster in the field:

| Dependency      | Why onedir helps                                            |
|-----------------|-------------------------------------------------------------|
| `pyproj`        | ships the PROJ data directory (CRS/datum transforms)        |
| `segyio`        | compiled SEG-Y I/O extension                                |
| `img2pdf`/Pillow| lossless PDF export path                                    |
| `PyQt6`/OpenGL  | Qt 6 plugins + PyQtGraph's OpenGL views                     |
| `matplotlib`    | Agg + PDF/SVG backends (bundled explicitly; loaded lazily)  |

## Prerequisites

Use a clean Python 3.11+ environment (conda or venv). Install the build deps:

```powershell
pip install -r env\requirements.txt
pip install -r env\requirements-dev.txt
```

## Build (CPU)

```powershell
# from the repo root, in the activated env
.\packaging\build.ps1            # → releases\SBPStudio-0.5.0-win64-cpu\
.\packaging\build.ps1 -Zip       # also produces the release .zip
```

Or invoke PyInstaller directly (output lands in the repo's `dist\`):

```powershell
$env:SBP_BUILD_CUDA = "0"
pyinstaller --noconfirm --clean packaging\SBPStudio.spec
```

## Smoke-test the build

```powershell
$pkg = "..\releases\SBPStudio-0.5.0-win64-cpu"
& "$pkg\SBPStudioCLI.exe" accel          # acceleration status
& "$pkg\SBPStudioCLI.exe" info file.sgy  # SEG-Y metadata
& "$pkg\SBPStudioGUI.exe"                # launch the desktop app
```

`app.log` is written next to the executable (frozen-app path handling in
`core/logger.py`), so field crashes are captured even for the windowed GUI.

## CUDA variant (pending — build on a compatible system)

The CUDA build is **not shipped with 0.5.0** because it cannot be verified on
the current build machine: the available CuPy install lacks the CUDA Toolkit
runtime needed to JIT-compile GPU kernels, so GPU acceleration can't be
exercised here.

**What you lose by shipping CPU-only:** very little. The headline measured
acceleration (~2–5× faster FFT / Hilbert / bandpass, ~15–20% faster end-to-end
on SBP data) comes from **`pyfftw`, which IS included in the CPU build**. CuPy's
*additional* GPU speedup applies mainly to Hilbert, AGC and RGBA normalisation
on individual profiles, and only on cards with ≥4 GB VRAM for full chains (a
2 GB card like the GTX 960M accelerates single profiles only). The app always
falls back to the CPU path automatically, so nothing is broken — only an
incremental GPU boost is deferred.

**To build/use CUDA for now, do it manually on a CUDA-capable machine:**

```powershell
# 1. Fresh env (do NOT mix with a conda `cupy` install)
python -m venv .venv-cuda ; .\.venv-cuda\Scripts\Activate.ps1
pip install -r env\requirements.txt
pip install -r env\requirements-dev.txt

# 2. Self-contained CuPy + CUDA Toolkit runtime wheels (~1.5 GB).
#    [ctk] is REQUIRED — it bundles cudart/nvrtc/cuBLAS/… so the frozen exe
#    can find them. Match the wheel to your driver line (12x for CUDA 12/13).
pip install -r env\requirements-cuda.txt    # cupy-cuda12x[ctk]

# 3. Verify CuPy can actually run a kernel BEFORE building:
python -c "import cupy; print(float((cupy.arange(4)*2).sum()))"   # must print 12.0

# 4. Build:
.\packaging\build.ps1 -Cuda      # → releases\SBPStudio-0.5.0-win64-cuda\

# 5. Confirm the bundle detects the GPU:
..\releases\SBPStudio-0.5.0-win64-cuda\SBPStudioCLI.exe accel
#    → "GPU (CuPy) : <device> (… VRAM)"   not "not available"
```

The `.spec` already collects `cupy`, `cupyx`, `cupy_backends` and the
`nvidia-*-cu12` runtime packages when `SBP_BUILD_CUDA=1`.

## Notes / known points

- **Linux/macOS:** the `.spec` is OS-agnostic; run
  `pyinstaller packaging/SBPStudio.spec` on the target OS for native bundles.
  `build.ps1` is Windows-only (the PyInstaller call it wraps is the same
  everywhere).
- **UTF-8 console:** `rthook_utf8.py` (a runtime hook) switches the Windows
  console to UTF-8 and reconfigures stdio so the CLI's non-ASCII output
  (`→ × µ …`) renders instead of crashing on legacy code pages.
- **No application icon** yet (no `.ico` in `assets/`). Add one and wire `icon=`
  into the `EXE(...)` calls when available.
- **Translations:** only `sbp_studio_es.ts` ships (parsed at runtime by
  `TsTranslator`); no `.qm` compilation step is required.
