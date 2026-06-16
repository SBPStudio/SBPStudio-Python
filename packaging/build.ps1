<#
.SYNOPSIS
    Build the SBP Studio 0.5.0 packaged application with PyInstaller.

.DESCRIPTION
    Produces a onedir bundle containing SBPStudioGUI.exe and SBPStudioCLI.exe.
    The final package is written OUTSIDE the repository, to:

        T:\workspace_py\SBPStudio-Python\releases\SBPStudio-0.5.0-win64-<variant>\

    (one level above the repo root, so build output is never committed). The
    PyInstaller working directory stays in <repo>\build\ (git-ignored).

    Run from an activated environment that has the runtime + packaging deps:

        pip install -r env/requirements.txt
        pip install -r env/requirements-dev.txt
        # CUDA variant only (on a CUDA-capable machine — see env/requirements-cuda.txt):
        pip install -r env/requirements-cuda.txt

.PARAMETER Cuda
    Build the CUDA variant (bundles CuPy + CUDA Toolkit runtime). Requires
    cupy-cuda12x[ctk] installed. See env/requirements-cuda.txt.

.PARAMETER Zip
    Zip the resulting package folder into a release archive alongside it.

.EXAMPLE
    .\packaging\build.ps1                 # CPU build
    .\packaging\build.ps1 -Cuda           # CUDA build
    .\packaging\build.ps1 -Zip            # CPU build + release .zip
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$Version = "0.5.0"

# Repo root = parent of this script's folder; releases dir = parent of repo root.
$RepoRoot    = Split-Path -Parent $PSScriptRoot
$ReleasesDir = Join-Path (Split-Path -Parent $RepoRoot) "releases"
Set-Location $RepoRoot

$Variant = if ($Cuda) { "cuda" } else { "cpu" }
$PkgName = "SBPStudio-$Version-win64-$Variant"
Write-Host "==> SBP Studio build  (variant: $Variant)" -ForegroundColor Cyan
Write-Host "==> Output: $ReleasesDir\$PkgName" -ForegroundColor Cyan

# Clean the working dir for a reproducible build.
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
New-Item -ItemType Directory -Force $ReleasesDir | Out-Null
$OutDir = Join-Path $ReleasesDir $PkgName
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }

if ($Cuda) { $env:SBP_BUILD_CUDA = "1" } else { $env:SBP_BUILD_CUDA = "0" }

# --distpath sends the COLLECT folder straight into releases\; the spec names it
# "SBPStudio", so we rename it to the variant-tagged package name afterwards.
pyinstaller --noconfirm --clean `
    --distpath $ReleasesDir `
    --workpath (Join-Path $RepoRoot "build") `
    "packaging\SBPStudio.spec"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)." }

$Collected = Join-Path $ReleasesDir "SBPStudio"
if (-not (Test-Path $Collected)) { throw "Expected output not found: $Collected" }
Rename-Item -Path $Collected -NewName $PkgName

Write-Host "==> Build OK: $OutDir" -ForegroundColor Green

if ($Zip) {
    $ZipPath = Join-Path $ReleasesDir "$PkgName.zip"
    if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
    Compress-Archive -Path "$OutDir\*" -DestinationPath $ZipPath
    Write-Host "==> Archive: $ZipPath" -ForegroundColor Green
}
